import glob
from functools import reduce

import pandas as pd


def calculate_mean_across_files(
    file_prefix: str,
    file_extension: str = "json",
) -> pd.DataFrame:
    """
    Calculate the mean across all files matching a given prefix.

    Parameters:
    -----------
    file_prefix : str
        The prefix to match files against
    file_extension : str
        The file extension to look for (default: 'csv')

    Returns:
    --------
    pd.DataFrame or dict
        The mean structure matching the input files
    """
    # Construct the pattern for glob
    pattern = f"{file_prefix}*.{file_extension}"

    # Get list of all matching files
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Read all files into a list
    data_list = [pd.read_json(f) for f in matching_files]

    # Calculate mean across all dataframes
    mean_data = reduce(lambda x, y: x.add(y, fill_value=0), data_list) / len(data_list)

    return mean_data


mean_result = calculate_mean_across_files("gflownet_training_metrics_chain-3", "json")
mean_result.to_json("gflownet_training_metrics_mean_chain-3.json", index=False)
mean_result = calculate_mean_across_files("gflownet_eval_metrics_chain-3", "json")
mean_result.to_json("gflownet_eval_metrics_mean_chain-3.json", index=False)

mean_result = calculate_mean_across_files("gflownet_training_metrics_chain-5", "json")
mean_result.to_json("gflownet_training_metrics_mean_chain-5.json", index=False)
mean_result = calculate_mean_across_files("gflownet_eval_metrics_chain-5", "json")
mean_result.to_json("gflownet_eval_metrics_mean_chain-5.json", index=False)

mean_result = calculate_mean_across_files("gflownet_training_metrics_chain-7", "json")
mean_result.to_json("gflownet_training_metrics_mean_chain-7.json", index=False)
mean_result = calculate_mean_across_files("gflownet_eval_metrics_chain-7", "json")
mean_result.to_json("gflownet_eval_metrics_mean_chain-7.json", index=False)

mean_result = calculate_mean_across_files("gflownet_training_metrics_chain-9", "json")
mean_result.to_json("gflownet_training_metrics_mean_chain-9.json", index=False)
mean_result = calculate_mean_across_files("gflownet_eval_metrics_chain-9", "json")
mean_result.to_json("gflownet_eval_metrics_mean_chain-9.json", index=False)

mean_result = calculate_mean_across_files("gflownet_training_metrics_chain-11", "json")
mean_result.to_json("gflownet_training_metrics_mean_chain-11.json", index=False)
mean_result = calculate_mean_across_files("gflownet_eval_metrics_chain-11", "json")
mean_result.to_json("gflownet_eval_metrics_mean_chain-11.json", index=False)
