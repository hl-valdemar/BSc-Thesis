import glob
import json
from typing import Any, Dict, List, Union

import numpy as np


def calculate_mean_training_metrics(
    file_prefix: str,
    file_extension: str = "json",
) -> Dict[str, Any]:
    """
    Calculate mean metrics across multiple GFlowNet training files.

    Handles the specific structure of GFlowNet metrics JSON files, which contain:
    - metrics_history: List of per-episode metrics
    - summary_stats: Aggregated statistics
    - window_size: Configuration parameter
    """
    pattern = f"{file_prefix}*.{file_extension}"
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Initialize accumulators
    all_metrics_history: List[List[Dict[str, Union[float, bool, int]]]] = []
    all_summary_stats: List[Dict[str, Any]] = []
    window_sizes: List[int] = []

    # Read all files
    for file_path in matching_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            all_metrics_history.append(data["metrics_history"])
            all_summary_stats.append(data["summary_stats"])
            window_sizes.append(data["window_size"])

    # Verify consistent window sizes
    if len(set(window_sizes)) != 1:
        raise ValueError("Inconsistent window sizes across files")

    # Calculate mean metrics history
    # First, ensure all histories have the same length
    min_length = min(len(history) for history in all_metrics_history)

    # Initialize structure for mean metrics
    mean_metrics_history = []
    for episode_idx in range(min_length):
        episode_metrics = {}
        metrics = all_metrics_history[0][episode_idx].keys()

        for metric in metrics:
            values = [
                history[episode_idx][metric]
                for history in all_metrics_history
                if isinstance(history[episode_idx][metric], (int, float))
            ]
            if values:  # Only calculate mean for numeric values
                episode_metrics[metric] = float(np.mean(values))
            else:  # For non-numeric values (like terminal_state_reached)
                episode_metrics[metric] = all_metrics_history[0][episode_idx][metric]

        mean_metrics_history.append(episode_metrics)

    # Calculate mean summary stats
    mean_summary_stats = {}
    first_summary = all_summary_stats[0]

    for key in first_summary:
        if key == "branch_dist":
            # Special handling for branch distribution
            all_branch_dists = {}
            for stats in all_summary_stats:
                for branch, value in stats["branch_dist"].items():
                    if branch not in all_branch_dists:
                        all_branch_dists[branch] = []
                    all_branch_dists[branch].append(value)

            mean_summary_stats["branch_dist"] = {
                branch: np.mean(values) for branch, values in all_branch_dists.items()
            }
        elif isinstance(first_summary[key], (int, float)):
            values = [stats[key] for stats in all_summary_stats]
            mean_summary_stats[key] = float(np.mean(values))
        else:
            mean_summary_stats[key] = first_summary[key]

    # Construct final output
    mean_data = {
        "metrics_history": mean_metrics_history,
        "summary_stats": mean_summary_stats,
        "window_size": window_sizes[0],  # All window sizes are the same
    }

    return mean_data


def calculate_mean_eval_metrics(
    file_prefix: str,
    file_extension: str = "json",
) -> Dict[str, Any]:
    """
    Calculate mean metrics across multiple GFlowNet evaluation files.

    This function specifically handles the evaluation metrics structure which includes:
    - Discrete distributions (rewards and branches)
    - Count-based statistics
    - Frequency-based statistics
    - Trajectory and error metrics
    """
    pattern = f"{file_prefix}*.{file_extension}"
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Initialize accumulators for each metric category
    all_metrics: List[Dict[str, Any]] = []

    # Read all evaluation files
    for file_path in matching_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            all_metrics.append(data)

    # Initialize the structure for mean metrics
    mean_metrics = {
        "reward_counts": {},
        "reward_frequencies": {},
        "branch_counts": {},
        "branch_frequencies": {},
        "average_trajectory_length": 0.0,
        "target_dist_error": 0.0,
    }

    # Calculate means for scalar metrics
    mean_metrics["average_trajectory_length"] = np.mean(
        [m["average_trajectory_length"] for m in all_metrics]
    )
    mean_metrics["target_dist_error"] = np.mean(
        [m["target_dist_error"] for m in all_metrics]
    )

    # Handle distribution metrics
    for metric_type in [
        "reward_counts",
        "reward_frequencies",
        "branch_counts",
        "branch_frequencies",
    ]:
        # Collect all possible keys across all files
        all_keys = set()
        for metrics in all_metrics:
            all_keys.update(metrics[metric_type].keys())

        # Calculate mean for each key
        for key in all_keys:
            values = [
                metrics[metric_type].get(key, 0)  # Use 0 if key doesn't exist
                for metrics in all_metrics
            ]
            mean_metrics[metric_type][key] = float(np.mean(values))

    return mean_metrics


# Usage with more precise file handling
for chain_length in [3, 5, 7, 9, 11]:
    for metric_type in ["training", "eval"]:
        prefix = f"gflownet_{metric_type}_metrics_chain-{chain_length}"

        if metric_type == "training":
            mean_result = calculate_mean_training_metrics(prefix, "json")
        elif metric_type == "eval":
            mean_result = calculate_mean_eval_metrics(prefix, "json")

        output_file = f"gflownet_{metric_type}_metrics_mean_chain-{chain_length}.json"
        with open(output_file, "w") as f:
            json.dump(mean_result, f, indent=2)
