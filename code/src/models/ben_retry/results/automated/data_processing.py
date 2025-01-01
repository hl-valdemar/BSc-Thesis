import glob
import json
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def calculate_mean_training_metrics(
    file_prefix: str, file_extension: str = "json"
) -> Dict[str, Any]:
    """
    Calculate mean metrics across multiple GFlowNet training files.

    This function specifically handles the training metrics structure which includes:
    - losses_q: Q-learning style losses
    - losses_epistemic: Epistemic uncertainty losses
    - rewards: Sequence of rewards received
    - branches: Branch choices made
    - states_visited: States traversed in the environment
    - cumulative_returns: Running sum of rewards
    """
    pattern = f"{file_prefix}*.{file_extension}"
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Initialize accumulators for each metric category
    all_metrics: List[Dict[str, Any]] = []

    # Read all training files
    for file_path in matching_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            all_metrics.append(data)

    # Calculate means for array-based metrics
    mean_metrics = {
        "losses_q": np.mean([m["losses_q"] for m in all_metrics], axis=0).tolist(),
        "losses_epistemic": np.mean(
            [m["losses_epistemic"] for m in all_metrics], axis=0
        ).tolist(),
        "rewards": np.mean([m["rewards"] for m in all_metrics], axis=0).tolist(),
        "branches": np.mean([m["branches"] for m in all_metrics], axis=0).tolist(),
        "cumulative_returns": np.mean(
            [m["cumulative_returns"] for m in all_metrics], axis=0
        ).tolist(),
    }

    # Handle states_visited specially - collect unique states
    all_states = set()
    for metrics in all_metrics:
        for state in metrics["states_visited"]:
            all_states.add(tuple(state))  # Convert list to tuple for set operations

    mean_metrics["states_visited"] = [list(state) for state in all_states]

    return mean_metrics


def calculate_mean_eval_metrics(
    file_prefix: str, file_extension: str = "json"
) -> Dict[str, Any]:
    """
    Calculate mean metrics across multiple GFlowNet evaluation files.

    This function specifically handles distributional statistics including:
    - Reward and branch counts/frequencies
    - Average trajectory metrics
    - Target distribution error
    """
    pattern = f"{file_prefix}*.{file_extension}"
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Initialize accumulators using defaultdict to handle varying keys
    total_reward_counts = defaultdict(float)
    total_reward_frequencies = defaultdict(float)
    total_branch_counts = defaultdict(float)
    total_branch_frequencies = defaultdict(float)

    # Metrics for scalar values
    trajectory_lengths: List[float] = []
    target_dist_errors: List[float] = []

    # Read and accumulate metrics
    n_files = len(matching_files)
    for file_path in matching_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            # Accumulate counts and frequencies
            for key, count in data["reward_counts"].items():
                total_reward_counts[key] += count
            for key, freq in data["reward_frequencies"].items():
                total_reward_frequencies[key] += freq
            for key, count in data["branch_counts"].items():
                total_branch_counts[key] += count
            for key, freq in data["branch_frequencies"].items():
                total_branch_frequencies[key] += freq

            # Accumulate scalar metrics
            trajectory_lengths.append(data["average_trajectory_length"])
            target_dist_errors.append(data["target_dist_error"])

    # Calculate means
    mean_metrics = {
        "reward_counts": {
            k: round(v / n_files, 0) for k, v in total_reward_counts.items()
        },
        "reward_frequencies": {
            k: round(v / n_files, 3) for k, v in total_reward_frequencies.items()
        },
        "branch_counts": {
            k: round(v / n_files, 0) for k, v in total_branch_counts.items()
        },
        "branch_frequencies": {
            k: round(v / n_files, 3) for k, v in total_branch_frequencies.items()
        },
        "average_trajectory_length": round(np.mean(trajectory_lengths), 1),
        "target_dist_error": round(np.mean(target_dist_errors), 3),
    }

    return mean_metrics


# Usage with more precise file handling
for chain_length in [3, 5, 7, 9, 11]:
    for metric_type in ["training", "eval"]:
        prefix = f"ben_{metric_type}_metrics_chain-{chain_length}"

        if metric_type == "training":
            mean_result = calculate_mean_training_metrics(prefix, "json")
        elif metric_type == "eval":
            mean_result = calculate_mean_eval_metrics(prefix, "json")

        output_file = f"ben_{metric_type}_metrics_mean_chain-{chain_length}.json"
        with open(output_file, "w") as f:
            json.dump(mean_result, f, indent=2)
