import glob
import json
import re

import numpy as np
from scipy.stats import entropy

true_dist = np.array([0.1, 0.2, 0.7])
pattern = re.compile(r"gflownet_eval_metrics_mean_chain-(\d+)\.json")

files = sorted(glob.glob("gflownet_eval_metrics_mean_chain-*.json"))
files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

kl_values = []

# Glob for all matching files
for filepath in files:
    # Extract chain length from filename
    match = pattern.search(filepath)
    if not match:
        continue

    chain_length = int(match.group(1))

    with open(filepath, "r") as f:
        data = json.load(f)

    # Build observed reward distribution vector
    q = np.array(
        [
            data["reward_frequencies"]["10"],
            data["reward_frequencies"]["20"],
            data["reward_frequencies"]["70"],
        ]
    )

    # Compute the KL divergence
    kl_div = entropy(true_dist, q)
    kl_values.append(kl_div)
    print(f"Chain length {chain_length}, KL divergence: {kl_div:.5f}")

if kl_values:
    print(f"Mean KL divergence across all files: {np.mean(kl_values):.5f}")
else:
    print("No valid files found to compute KL divergence.")
