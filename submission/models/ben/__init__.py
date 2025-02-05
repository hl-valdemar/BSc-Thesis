import json
from datetime import datetime
from typing import Dict

import numpy as np
import torch

from environments.nchain import NChainEnv

from .QBayesNet import QBayesNetwork
from .training import train


def main(chain_length: int, gamma: float = 0.9):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    descriptor = f"chain-{chain_length}_{int(gamma * 100)}percent-gamma"

    bayes_model, q_net, env, training_metrics = train(
        timestamp=timestamp,
        descriptor=descriptor,
        chain_length=chain_length,
        gamma=gamma,
        N_episodes=50,
    )

    torch.save(
        bayes_model.state_dict(),
        f"ben_bayes-model_state_{descriptor}_{timestamp}.pth",
    )
    torch.save(
        q_net.state_dict(),
        f"ben_q-net_state_{descriptor}_{timestamp}.pth",
    )

    # Save JSON formatted training metrics to a file
    training_metrics_json = json.dumps(training_metrics, indent=2)
    with open(
        f"ben_training_metrics_{descriptor}_{timestamp}.json",
        "w",
    ) as f:
        f.write(training_metrics_json)

    # Evaluate the model
    eval_metrics = evaluate_learned_policy(
        q_net=q_net,
        env=env,
        num_trajectories=1000,
    )
    eval_metrics_json = json.dumps(eval_metrics, indent=2)
    with open(
        f"ben_eval_metrics_{descriptor}_{timestamp}.json",
        "w",
    ) as f:
        f.write(eval_metrics_json)


def evaluate_learned_policy(
    q_net: QBayesNetwork,
    env: NChainEnv,
    num_trajectories: int = 200,
) -> Dict[str, float]:
    """
    Evaluate the learned policy by sampling trajectories.

    Args:
        q_net: Conditioned QBayesNetwork to use
        env: Environment to use
        num_trajectories: Number of trajectories to sample

    Returns:
        Dictionary containing:
            - reward_counts: Number of times each reward was reached
            - reward_frequencies: Distribution of terminal rewards
            - branch_count: Number of times each branch was selected
            - branch_frequencies: Distribution of branch selections
            - average_trajectory_length: Mean steps to completion
    """
    rewards = []
    branches = []
    trajectory_lengths = []

    # Turn off gradients for evaluation
    with torch.no_grad():
        for _ in range(num_trajectories):
            state = env.reset()
            done = False
            steps = 0

            q_net.reset_net()
            hidden = q_net.init_hidden()

            action = env.sample_valid_actions(k=1)[0]
            reward = 0

            while not done:
                inputs = torch.cat(
                    (
                        state.to_tensor(dtype=torch.float64).view(1, -1),
                        torch.tensor(action, dtype=torch.float64).view(1, -1),
                        torch.tensor(reward, dtype=torch.float64).view(1, -1),
                    ),
                    dim=1,
                )
                q_values, hidden = q_net.forward(inputs, hidden)

                # Get valid actions
                valid_actions = env.get_valid_actions(state)

                # Get action from masked q-values
                mask = torch.zeros(
                    (q_values.shape[0], env.num_actions),
                    dtype=torch.bool,
                )
                mask[:, valid_actions] = True
                q_values = q_values.masked_fill(~mask, float("-inf"))
                action = torch.argmax(q_values, -1).item()

                # Update the environment
                next_state, reward, done = env.step(action)
                state = next_state
                steps += 1

            # Collect metrics
            rewards.append(reward)
            branches.append(state.branch)
            trajectory_lengths.append(steps)

    # Compute statistics
    reward_counts = {r: rewards.count(r) for r in set(rewards)}
    reward_freqs = {r: c / num_trajectories for r, c in reward_counts.items()}

    branch_counts = {b: branches.count(b) for b in set(branches)}
    branch_freqs = {b: c / num_trajectories for b, c in branch_counts.items()}

    reward_set = set(rewards)
    target_dist_err = 0
    for r in reward_set:
        if r >= 0:
            actual_freq = reward_freqs[r]
            expected_freq = r / np.sum(list(reward_set))
            target_dist_err += np.abs(actual_freq - expected_freq)

    return {
        "reward_counts": reward_counts,
        "reward_frequencies": reward_freqs,
        "branch_counts": branch_counts,
        "branch_frequencies": branch_freqs,
        "average_trajectory_length": np.mean(trajectory_lengths),
        "target_dist_error": target_dist_err,
    }
