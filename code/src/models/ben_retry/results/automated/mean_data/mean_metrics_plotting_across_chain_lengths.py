import glob
import json

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(filepath):
    """Load metrics from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_metrics(data):
    """Extract relevant metrics from BEN model data."""
    return {
        "losses_q": data["losses_q"],
        "losses_epistemic": data["losses_epistemic"],
        "rewards": data["rewards"],
        "cumulative_returns": data["cumulative_returns"],
    }


def running_average(data, window_size=50):
    """Calculate running average with specified window size."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="valid")


def plot_training_curves(files_pattern):
    """Plot training curves for multiple chain lengths."""
    # Set up the plot style
    # plt.style.use("seaborn")

    # Create a figure with subplots - 2x2 with focused metrics
    fig, ((ax_q_log, ax_ep_log), (ax_rew, ax_ret)) = plt.subplots(
        2, 2, figsize=(15, 12)
    )

    # Get all matching files and sort by chain length
    files = glob.glob(files_pattern)
    files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

    # Color map for different chain lengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

    # Initialize lists for metrics
    all_losses_q = []
    all_losses_epistemic = []
    all_rewards = []
    all_returns = []

    # First pass: Collect metrics
    for file in files:
        data = load_metrics(file)
        metrics = extract_metrics(data)

        all_losses_q.append(metrics["losses_q"])
        all_losses_epistemic.append(metrics["losses_epistemic"])
        all_rewards.append(metrics["rewards"])
        all_returns.append(metrics["cumulative_returns"])

    # Process metrics with running averages
    window_size = 50

    def process_metrics(metrics_list):
        min_length = min(len(x) for x in metrics_list)
        truncated = [x[:min_length] for x in metrics_list]
        metrics_array = np.array(truncated)

        running_avgs = np.array(
            [running_average(seq, window_size) for seq in truncated]
        )
        mean_running = np.mean(running_avgs, axis=0)
        std_running = np.std(running_avgs, axis=0)

        # Calculate raw mean across all sequences
        raw_mean = np.mean(metrics_array, axis=0)
        raw_std = np.std(metrics_array, axis=0)

        return metrics_array, mean_running, std_running, raw_mean, raw_std, min_length

    # Process each metric type
    (
        losses_q_data,
        running_mean_losses_q,
        running_std_losses_q,
        _,
        _,
        q_length,
    ) = process_metrics(all_losses_q)
    (
        losses_ep_data,
        running_mean_losses_ep,
        running_std_losses_ep,
        _,
        _,
        ep_length,
    ) = process_metrics(all_losses_epistemic)
    (
        rewards_data,
        mean_rewards,
        std_rewards,
        raw_mean_rewards,
        raw_std_rewards,
        reward_length,
    ) = process_metrics(all_rewards)
    (
        returns_data,
        mean_returns,
        std_returns,
        raw_mean_returns,
        raw_std_returns,
        returns_length,
    ) = process_metrics(all_returns)

    # Create step arrays
    q_steps = np.arange(q_length)
    ep_steps = np.linspace(0, q_length - 1, ep_length)
    reward_steps = np.arange(reward_length)

    # Plot data for each chain length
    for idx, (file, color) in enumerate(zip(files, colors)):
        chain_length = int(file.split("-")[-1].split(".")[0])

        # Log-scale loss plots
        ax_q_log.plot(
            q_steps,
            all_losses_q[idx][:q_length],
            color=color,
            alpha=0.4,
            label=f"Chain-{chain_length}",
        )
        ax_ep_log.plot(
            ep_steps,
            all_losses_epistemic[idx][:ep_length],
            color=color,
            alpha=0.4,
            label=f"Chain-{chain_length}",
        )

        # Performance metrics
        ax_rew.plot(
            reward_steps,
            all_rewards[idx][:reward_length],
            color=color,
            alpha=0.4,
            label=f"Chain-{chain_length}",
        )
        ax_ret.plot(
            reward_steps,
            all_returns[idx][:returns_length],
            color=color,
            alpha=0.4,
            label=f"Chain-{chain_length}",
        )

    # Add mean curves with confidence intervals
    steps_q_avg = np.arange(window_size - 1, q_length)
    steps_ep_avg = np.linspace(
        window_size - 1, q_length - 1, len(running_mean_losses_ep)
    )
    steps_reward_avg = np.arange(window_size - 1, reward_length)

    # Mean curves for losses
    ax_q_log.plot(
        steps_q_avg,
        running_mean_losses_q,
        color=color,
        linewidth=2,
        label="Mean",
    )
    ax_q_log.fill_between(
        steps_q_avg,
        running_mean_losses_q - running_std_losses_q,
        running_mean_losses_q + running_std_losses_q,
        color="gray",
        alpha=0.5,
    )

    ax_ep_log.plot(
        steps_ep_avg,
        running_mean_losses_ep,
        color=color,
        linewidth=2,
        label="Mean",
    )
    ax_ep_log.fill_between(
        steps_ep_avg,
        running_mean_losses_ep - running_std_losses_ep,
        running_mean_losses_ep + running_std_losses_ep,
        color="gray",
        alpha=1,
    )

    # Raw means for rewards and returns (thicker red line)
    ax_rew.plot(
        reward_steps,
        raw_mean_rewards,
        color=color,
        linewidth=3,
        label="Mean Across Chains",
    )
    ax_rew.fill_between(
        reward_steps,
        raw_mean_rewards - raw_std_rewards,
        raw_mean_rewards + raw_std_rewards,
        color="gray",
        alpha=0.25,
    )

    ax_ret.plot(
        reward_steps,
        raw_mean_returns,
        color=color,
        linewidth=3,
        label="Mean Across Chains",
    )
    ax_ret.fill_between(
        reward_steps,
        raw_mean_returns - raw_std_returns,
        raw_mean_returns + raw_std_returns,
        color="gray",
        alpha=0.25,
    )

    # Set log scale for loss plots
    ax_q_log.set_yscale("log")
    ax_ep_log.set_yscale("log")

    # Customize plots
    ax_q_log.set_title("Q-Loss")
    ax_ep_log.set_title("Epistemic Loss")
    ax_rew.set_title("Rewards")
    ax_ret.set_title("Cumulative Returns")

    # Set labels
    for ax in [ax_q_log, ax_ep_log, ax_rew, ax_ret]:
        ax.set_xlabel("Training Steps")
        ax.legend(loc="upper right")

    ax_q_log.set_ylabel("Log Loss")
    ax_ep_log.set_ylabel("Log Loss")
    ax_rew.set_ylabel("Reward")
    ax_ret.set_ylabel("Cumulative Return")

    # Add common title
    plt.suptitle("BEN Model Training Metrics Across Chain Lengths", fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout()
    return fig


# Example usage
fig = plot_training_curves("ben_training_metrics_mean_chain-*.json")
plt.show()
