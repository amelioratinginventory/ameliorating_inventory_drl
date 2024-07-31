import numpy as np
import random
import string
import scipy.signal
from typing import Dict, Optional

from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, attempt_count_timesteps
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID


@DeveloperAPI
class Postprocessing:
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"
    BOOTSTRAPPED_VALUES = "bootstrapped_values"
    ACTOR_ID = "actor_id"
    TRUNCATED_MEAN_REWARD = "truncated_mean_reward"
    TRUNCATED_MEAN_VF_PRED = "truncated_mean_vf_pred"
    DISCOUNTED_REWARDS = "discounted_rewards"


@DeveloperAPI
def adjust_nstep(n_step: int, gamma: float, batch: SampleBatch) -> None:
    """Rewrites `batch` to encode n-step rewards, terminateds, truncateds, and next-obs.

    Observations and actions remain unaffected. At the end of the trajectory,
    n is truncated to fit in the traj length.

    Args:
        n_step: The number of steps to look ahead and adjust.
        gamma: The discount factor.
        batch: The SampleBatch to adjust (in place).

    Examples:
        n-step=3
        Trajectory=o0 r0 d0, o1 r1 d1, o2 r2 d2, o3 r3 d3, o4 r4 d4=True o5
        gamma=0.9
        Returned trajectory:
        0: o0 [r0 + 0.9*r1 + 0.9^2*r2 + 0.9^3*r3] d3 o0'=o3
        1: o1 [r1 + 0.9*r2 + 0.9^2*r3 + 0.9^3*r4] d4 o1'=o4
        2: o2 [r2 + 0.9*r3 + 0.9^2*r4] d4 o1'=o5
        3: o3 [r3 + 0.9*r4] d4 o3'=o5
        4: o4 r4 d4 o4'=o5
    """

    assert (
        batch.is_single_trajectory()
    ), "Unexpected terminated|truncated in middle of trajectory!"

    len_ = len(batch)

    # Shift NEXT_OBS, TERMINATEDS, and TRUNCATEDS.
    batch[SampleBatch.NEXT_OBS] = np.concatenate(
        [
            batch[SampleBatch.OBS][n_step:],
            np.stack([batch[SampleBatch.NEXT_OBS][-1]] * min(n_step, len_)),
        ],
        axis=0,
    )
    batch[SampleBatch.TERMINATEDS] = np.concatenate(
        [
            batch[SampleBatch.TERMINATEDS][n_step - 1 :],
            np.tile(batch[SampleBatch.TERMINATEDS][-1], min(n_step - 1, len_)),
        ],
        axis=0,
    )
    # Only fix `truncateds`, if present in the batch.
    if SampleBatch.TRUNCATEDS in batch:
        batch[SampleBatch.TRUNCATEDS] = np.concatenate(
            [
                batch[SampleBatch.TRUNCATEDS][n_step - 1 :],
                np.tile(batch[SampleBatch.TRUNCATEDS][-1], min(n_step - 1, len_)),
            ],
            axis=0,
        )

    # Change rewards in place.
    for i in range(len_):
        for j in range(1, n_step):
            if i + j < len_:
                batch[SampleBatch.REWARDS][i] += (
                    gamma**j * batch[SampleBatch.REWARDS][i + j]
                )


@DeveloperAPI
def compute_advantages(
    rollout: SampleBatch,
    last_r: float,
    gamma: float = 0.9,
    lambda_: float = 1.0,
    use_gae: bool = True,
    use_critic: bool = True,
):
    """Given a rollout, compute its value targets and the advantages.

    Args:
        rollout: SampleBatch of a single trajectory.
        last_r: Value estimation for last observation.
        gamma: Discount factor.
        lambda_: Parameter for GAE.
        use_gae: Using Generalized Advantage Estimation.
        use_critic: Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.

    Returns:
        SampleBatch with experience from rollout and processed rewards.
    """

    assert (
        SampleBatch.VF_PREDS in rollout or not use_critic
    ), "use_critic=True but values not found"
    assert use_critic or not use_gae, "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate([rollout[SampleBatch.VF_PREDS], np.array([last_r])])
        delta_t = rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[Postprocessing.ADVANTAGES] = discount_cumsum(delta_t, gamma * lambda_)
        rollout[Postprocessing.VALUE_TARGETS] = (
            rollout[Postprocessing.ADVANTAGES] + rollout[SampleBatch.VF_PREDS]
        ).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS], np.array([last_r])]
        )
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(
            np.float32
        )

        if use_critic:
            rollout[Postprocessing.ADVANTAGES] = (
                discounted_returns - rollout[SampleBatch.VF_PREDS]
            )
            rollout[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            rollout[Postprocessing.ADVANTAGES] = discounted_returns
            rollout[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                rollout[Postprocessing.ADVANTAGES]
            )

    rollout[Postprocessing.ADVANTAGES] = rollout[Postprocessing.ADVANTAGES].astype(
        np.float32
    )

    return rollout


@DeveloperAPI
def compute_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy: The Policy used to generate the trajectory (`sample_batch`)
        sample_batch: The SampleBatch to postprocess.
        other_agent_batches: Optional dict of AgentIDs mapping to other
            agents' trajectory data (from the same episode).
            NOTE: The other agents use the same policy.
        episode: Optional multi-agent episode object in which the agents
            operated.

    Returns:
        The postprocessed, modified SampleBatch (or a new one).
    """

    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.TERMINATEDS][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last"
        )

        if policy.config["_enable_rl_module_api"]:
            # Note: During sampling you are using the parameters at the beginning of
            # the sampling process. If I'll be using this advantages during training
            # should it not be the latest parameters during training for this to be
            # correct? Does this mean that I need to preserve the trajectory
            # information during training and compute the advantages inside the loss
            # function?
            # TODO (Kourosh)
            # Another thing I need to figure out is which end point to call here?
            # forward_exploration? what if this method is getting called inside the
            # learner loop? or via another abstraction like
            # RLSampler.postprocess_trajectory() which is non-batched cpu/gpu task
            # running across different processes for different trajectories?
            # This implementation right now will compute even the action_dist which
            # will not be needed but takes time to compute.
            input_dict = policy._lazy_tensor_dict(input_dict)
            fwd_out = policy.model.forward_exploration(input_dict)
            last_r = fwd_out[SampleBatch.VF_PREDS][-1]
        else:
            last_r = policy._value(**input_dict)

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True),
    )

    if "truncation_length" in policy.config and policy.config["truncation_length"] > 0:
        batch = truncate_batch(batch, policy.config["truncation_length"])
        

    return batch


@DeveloperAPI
def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma: The discount factor gamma.

    Returns:
        The sequence containing the discounted cumulative sums
        for each individual reward in `x` till the end of the trajectory.

    Examples:
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> gamma = 0.9
        >>> discount_cumsum(x, gamma)
        ... array([0.0 + 0.9*1.0 + 0.9^2*2.0 + 0.9^3*3.0,
        ...        1.0 + 0.9*2.0 + 0.9^2*3.0,
        ...        2.0 + 0.9*3.0,
        ...        3.0])
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]

@DeveloperAPI
def compute_apo_advantage_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    lambda_,
    update_estimates = False,
    bias_regularization = 0,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Computes advantages according to APO concept https://arxiv.org/pdf/2106.03442.pdf.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy: The Policy used to generate the trajectory (`sample_batch`)
        sample_batch: The SampleBatch to postprocess.
        other_agent_batches: Optional dict of AgentIDs mapping to other
            agents' trajectory data (from the same episode).
            NOTE: The other agents use the same policy.
        episode: Optional multi-agent episode object in which the agents
            operated.

    Returns:
        The postprocessed, modified SampleBatch (or a new one).
    """

    
    if update_estimates:
        if "truncation_length" in policy.config and policy.config["truncation_length"] > 0:
            actor_range_truncated = {}
        alpha = policy.config["apo_step_size"]
        policy.average_reward_estimate = (1-alpha) * policy.average_reward_estimate + alpha * (np.mean(sample_batch[Postprocessing.TRUNCATED_MEAN_REWARD]) + bias_regularization)
        #print("VF_PREDS: ", sample_batch[SampleBatch.VF_PREDS])
        policy.bias_estimate = (1-alpha) * policy.bias_estimate + alpha * (np.mean(sample_batch[Postprocessing.TRUNCATED_MEAN_VF_PRED]) + bias_regularization)
        advantages = {}
        value_targets = {}
        horizon = len(sample_batch[SampleBatch.REWARDS]) // policy.config["num_workers"]
        for i in range(policy.config["num_workers"]):
            actor_range = range(i*horizon, (i+1)*horizon) 
            if "truncation_length" in policy.config and policy.config["truncation_length"] > 0:
                actor_range_truncated[i] = range(i*horizon, (i+1)*horizon-policy.config["truncation_length"])
            delta_t = sample_batch[SampleBatch.REWARDS][actor_range] - policy.average_reward_estimate + sample_batch[Postprocessing.BOOTSTRAPPED_VALUES][actor_range] - sample_batch[SampleBatch.VF_PREDS][actor_range]
            advantages[i] = discount_cumsum(delta_t, lambda_)
            value_targets[i] = (sample_batch[SampleBatch.REWARDS][actor_range] - policy.average_reward_estimate + sample_batch[Postprocessing.BOOTSTRAPPED_VALUES][actor_range]).astype(np.float32)
        sample_batch[Postprocessing.ADVANTAGES] = np.concatenate([advantages[i] for i in range(policy.config["num_workers"])])
        sample_batch[Postprocessing.VALUE_TARGETS] = np.concatenate([value_targets[i] for i in range(policy.config["num_workers"])])
        sample_batch[Postprocessing.ADVANTAGES] = sample_batch[Postprocessing.ADVANTAGES].astype(np.float32)
        if "truncation_length" in policy.config and policy.config["truncation_length"] > 0:
            sample_batch = truncate_multi_worker_batch(sample_batch, policy.config["num_workers"], actor_range_truncated)
        sample_batch.count = attempt_count_timesteps(sample_batch)
    else:
        #sample random string with 10 characters for actor id
        if "truncation_length" in policy.config and policy.config["truncation_length"] > 0:
            truncated_mean_reward = np.mean(sample_batch[SampleBatch.REWARDS][:-policy.config["truncation_length"]])
            sample_batch[Postprocessing.TRUNCATED_MEAN_REWARD] = np.array([truncated_mean_reward]*len(sample_batch[SampleBatch.REWARDS]))
            truncated_mean_vf_pred = np.mean(sample_batch[SampleBatch.VF_PREDS][:-policy.config["truncation_length"]])
            sample_batch[Postprocessing.TRUNCATED_MEAN_VF_PRED] = np.array([truncated_mean_vf_pred]*len(sample_batch[SampleBatch.REWARDS]))
        #get prediction of value of transition state for last state in batch
        input_dict = sample_batch.get_single_step_input_dict(policy.model.view_requirements, index="last")
        last_r = policy._value(**input_dict)
        sample_batch[Postprocessing.BOOTSTRAPPED_VALUES] = np.concatenate([sample_batch[SampleBatch.VF_PREDS][1:], np.array([last_r])])
        delta_t = sample_batch[SampleBatch.REWARDS] - policy.average_reward_estimate + sample_batch[Postprocessing.BOOTSTRAPPED_VALUES] - sample_batch[SampleBatch.VF_PREDS]
        sample_batch[Postprocessing.ADVANTAGES] = discount_cumsum(delta_t, lambda_)
        sample_batch[Postprocessing.VALUE_TARGETS] = (sample_batch[SampleBatch.REWARDS] - policy.average_reward_estimate + sample_batch[Postprocessing.BOOTSTRAPPED_VALUES]).astype(np.float32)
        sample_batch[Postprocessing.ADVANTAGES] = sample_batch[Postprocessing.ADVANTAGES].astype(np.float32)
        #sample_batch[Postprocessing.VF_PREDS] = sample_batch[SampleBatch.VF_PREDS]

    return sample_batch

@DeveloperAPI
def compute_dapo_advantage_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    lambda_,
    update_estimates = False,
    bias_regularization = 0,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Computes advantages according to DAPO concept https://arxiv.org/pdf/2106.03442.pdf.

    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.

    Args:
        policy: The Policy used to generate the trajectory (`sample_batch`)
        sample_batch: The SampleBatch to postprocess.
        other_agent_batches: Optional dict of AgentIDs mapping to other
            agents' trajectory data (from the same episode).
            NOTE: The other agents use the same policy.
        episode: Optional multi-agent episode object in which the agents
            operated.

    Returns:
        The postprocessed, modified SampleBatch (or a new one).
    """

    if update_estimates:
        if "truncation_length" in policy.config and policy.config["truncation_length"] > 0:
            actor_range_truncated = {}
        alpha = policy.config["dapo_step_size"]
        policy.discounted_average_reward_estimate = (1-alpha) * policy.discounted_average_reward_estimate + alpha * (np.mean(sample_batch[Postprocessing.DISCOUNTED_REWARDS]) + bias_regularization)
        #print("VF_PREDS: ", sample_batch[SampleBatch.VF_PREDS])
        policy.bias_estimate = (1-alpha) * policy.bias_estimate + alpha * (np.mean(sample_batch[SampleBatch.VF_PREDS]) + bias_regularization)
        advantages = {}
        value_targets = {}
        horizon = len(sample_batch[SampleBatch.REWARDS]) // policy.config["num_workers"]
        for i in range(policy.config["num_workers"]):
            actor_range = range(i*horizon, (i+1)*horizon) 
            delta_t = sample_batch[SampleBatch.REWARDS][actor_range] - policy.discounted_average_reward_estimate + policy.config["gamma"] * sample_batch[Postprocessing.BOOTSTRAPPED_VALUES][actor_range] - sample_batch[SampleBatch.VF_PREDS][actor_range]
            advantages[i] = discount_cumsum(delta_t, lambda_)
            value_targets[i] = (sample_batch[SampleBatch.REWARDS][actor_range] - policy.discounted_average_reward_estimate + policy.config["gamma"] * sample_batch[Postprocessing.BOOTSTRAPPED_VALUES][actor_range]).astype(np.float32)
        sample_batch[Postprocessing.ADVANTAGES] = np.concatenate([advantages[i] for i in range(policy.config["num_workers"])])
        sample_batch[Postprocessing.VALUE_TARGETS] = np.concatenate([value_targets[i] for i in range(policy.config["num_workers"])])
        sample_batch[Postprocessing.ADVANTAGES] = sample_batch[Postprocessing.ADVANTAGES].astype(np.float32)
    else:
        sample_batch[Postprocessing.DISCOUNTED_REWARDS] = discount_cumsum(sample_batch[SampleBatch.REWARDS], policy.config["gamma"])
        input_dict = sample_batch.get_single_step_input_dict(policy.model.view_requirements, index="last")
        last_r = policy._value(**input_dict)
        sample_batch[Postprocessing.BOOTSTRAPPED_VALUES] = np.concatenate([sample_batch[SampleBatch.VF_PREDS][1:], np.array([last_r])])
        #sample random string with 10 characters for actor id
        if "truncation_length" in policy.config and policy.config["truncation_length"] > 0:
            sample_batch = truncate_batch(sample_batch, policy.config["truncation_length"])
        #get prediction of value of transition state for last state in batch
        delta_t = sample_batch[SampleBatch.REWARDS] - policy.discounted_average_reward_estimate + policy.config["gamma"] * sample_batch[Postprocessing.BOOTSTRAPPED_VALUES] - sample_batch[SampleBatch.VF_PREDS]
        sample_batch[Postprocessing.ADVANTAGES] = discount_cumsum(delta_t, lambda_)
        sample_batch[Postprocessing.VALUE_TARGETS] = (sample_batch[SampleBatch.REWARDS] - policy.discounted_average_reward_estimate + policy.config["gamma"] * sample_batch[Postprocessing.BOOTSTRAPPED_VALUES]).astype(np.float32)
        sample_batch[Postprocessing.ADVANTAGES] = sample_batch[Postprocessing.ADVANTAGES].astype(np.float32)
       
    return sample_batch



@DeveloperAPI
def update_estimates(
    policy: Policy,
    sample_batch: SampleBatch,
) -> SampleBatch:
    
    alpha = policy.config["apo_step_size"]
    policy.average_reward_estimate = policy.average_reward_estimate = (1-alpha) * policy.average_reward_estimate + alpha * np.mean(sample_batch[SampleBatch.REWARDS])
    
    return SampleBatch

def truncate_batch(
        batch: SampleBatch, 
        truncation_length: int
) -> SampleBatch:
    """Truncates the batch to the given length.

    Args:
        batch: The SampleBatch to truncate.
        truncation_length: The length to truncate the batch by.

    Returns:
        The truncated batch.
    """
    if "advantages" in batch:
        batch[Postprocessing.ADVANTAGES] = batch[Postprocessing.ADVANTAGES][:-truncation_length]
    if "value_targets" in batch:
        batch[Postprocessing.VALUE_TARGETS] = batch[Postprocessing.VALUE_TARGETS][:-truncation_length]
    if "bootstrapped_values" in batch:
        batch[Postprocessing.BOOTSTRAPPED_VALUES] = batch[Postprocessing.BOOTSTRAPPED_VALUES][:-truncation_length]
    batch[SampleBatch.VF_PREDS] = batch[SampleBatch.VF_PREDS][:-truncation_length]
    batch[SampleBatch.ACTIONS] = batch[SampleBatch.ACTIONS][:-truncation_length]  
    batch[SampleBatch.REWARDS] = batch[SampleBatch.REWARDS][:-truncation_length]
    batch[SampleBatch.OBS] = batch[SampleBatch.OBS][:-truncation_length]
    batch[SampleBatch.NEXT_OBS] = batch[SampleBatch.NEXT_OBS][:-truncation_length]
    batch[SampleBatch.ACTION_LOGP] = batch[SampleBatch.ACTION_LOGP][:-truncation_length]
    batch[SampleBatch.ACTION_DIST_INPUTS] = batch[SampleBatch.ACTION_DIST_INPUTS][:-truncation_length]
    if "prev_actions" in batch:
        batch[SampleBatch.PREV_ACTIONS] = batch[SampleBatch.PREV_ACTIONS][:-truncation_length]
    if "prev_rewards" in batch:
        batch[SampleBatch.PREV_REWARDS] = batch[SampleBatch.PREV_REWARDS][:-truncation_length]
    if "discounted_rewards" in batch:
        batch[Postprocessing.DISCOUNTED_REWARDS] = batch[Postprocessing.DISCOUNTED_REWARDS][:-truncation_length]

    return batch

def truncate_multi_worker_batch(batch, num_workers, worker_ranges):
    """Truncates the batch to the given length.

    Args:
        batch: The SampleBatch to truncate.
        num_workers: The number of workers.
        worker_ranges: The ranges of the workers to keep in the truncated batch.

    Returns:
        The truncated batch.
    """

    batch[Postprocessing.ADVANTAGES] = np.concatenate([batch[Postprocessing.ADVANTAGES][worker_ranges[i]] for i in range(num_workers)])
    batch[Postprocessing.VALUE_TARGETS] = np.concatenate([batch[Postprocessing.VALUE_TARGETS][worker_ranges[i]] for i in range(num_workers)])
    batch[Postprocessing.BOOTSTRAPPED_VALUES] = np.concatenate([batch[Postprocessing.BOOTSTRAPPED_VALUES][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.VF_PREDS] = np.concatenate([batch[SampleBatch.VF_PREDS][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.ACTIONS] = np.concatenate([batch[SampleBatch.ACTIONS][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.REWARDS] = np.concatenate([batch[SampleBatch.REWARDS][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.INFOS] = np.concatenate([batch[SampleBatch.INFOS][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.EPS_ID] = np.concatenate([batch[SampleBatch.EPS_ID][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.OBS] = np.concatenate([batch[SampleBatch.OBS][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.NEXT_OBS] = np.concatenate([batch[SampleBatch.NEXT_OBS][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.ACTION_LOGP] = np.concatenate([batch[SampleBatch.ACTION_LOGP][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.ACTION_DIST_INPUTS] = np.concatenate([batch[SampleBatch.ACTION_DIST_INPUTS][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.TERMINATEDS] = np.concatenate([batch[SampleBatch.TERMINATEDS][worker_ranges[i]] for i in range(num_workers)])
    batch[SampleBatch.TRUNCATEDS] = np.concatenate([batch[SampleBatch.TRUNCATEDS][worker_ranges[i]] for i in range(num_workers)])
    if "truncated_mean_reward" in batch:
        batch[Postprocessing.TRUNCATED_MEAN_REWARD] = np.concatenate([batch[Postprocessing.TRUNCATED_MEAN_REWARD][worker_ranges[i]] for i in range(num_workers)])
    if "truncated_mean_vf_pred" in batch:
        batch[Postprocessing.TRUNCATED_MEAN_VF_PRED] = np.concatenate([batch[Postprocessing.TRUNCATED_MEAN_VF_PRED][worker_ranges[i]] for i in range(num_workers)])
    
    if "prev_actions" in batch:
        batch[SampleBatch.PREV_ACTIONS] = np.concatenate([batch[SampleBatch.PREV_ACTIONS][worker_ranges[i]] for i in range(num_workers)])
    if "prev_rewards" in batch:
        batch[SampleBatch.PREV_REWARDS] = np.concatenate([batch[SampleBatch.PREV_REWARDS][worker_ranges[i]] for i in range(num_workers)])

    return batch
