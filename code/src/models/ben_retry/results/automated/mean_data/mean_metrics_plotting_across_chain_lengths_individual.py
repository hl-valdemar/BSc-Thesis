import glob
import json

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(filepath):
    """Load metrics from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def running_average(data, window_size=50):
    """Calculate running average with specified window size."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="valid")


def process_metrics(metrics_list, window_size=50):
    """Process metrics to get statistics and running averages."""
    min_length = min(len(x) for x in metrics_list)
    truncated = [x[:min_length] for x in metrics_list]
    metrics_array = np.array(truncated)

    running_avgs = np.array([running_average(seq, window_size) for seq in truncated])
    mean_running = np.mean(running_avgs, axis=0)
    std_running = np.std(running_avgs, axis=0)

    raw_mean = np.mean(metrics_array, axis=0)
    raw_std = np.std(metrics_array, axis=0)

    return metrics_array, mean_running, std_running, raw_mean, raw_std, min_length


def plot_q_loss(files):
    """Plot Q-Loss metrics."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map for different chain lengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

    all_losses_q = []
    for file in files:
        data = load_metrics(file)
        all_losses_q.append(data["losses_q"])

    losses_q_data, running_mean, running_std, _, _, q_length = process_metrics(
        all_losses_q
    )
    steps = np.arange(q_length)
    steps_avg = np.arange(49, q_length)  # window_size - 1

    # Plot individual chains
    for idx, (file, color) in enumerate(zip(files, colors)):
        chain_length = int(file.split("-")[-1].split(".")[0])
        ax.plot(
            steps,
            all_losses_q[idx][:q_length],
            color=color,
            alpha=0.4,
            label=f"Chain-{chain_length}",
        )

    # Plot mean with confidence interval
    ax.plot(
        steps_avg,
        running_mean,
        color="orange",
        linewidth=2,
        label="Mean (Moving Avg, n=50)",
    )
    ax.fill_between(
        steps_avg,
        running_mean - running_std,
        running_mean + running_std,
        color="gray",
        alpha=0.5,
    )

    ax.set_yscale("log")
    ax.set_title("Q-Loss During Training")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Log Loss")
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig


def plot_epistemic_loss(files):
    """Plot Epistemic Loss metrics."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

    all_losses = []
    for file in files:
        data = load_metrics(file)
        all_losses.append(data["losses_epistemic"])

    _, running_mean, running_std, _, _, ep_length = process_metrics(all_losses)
    steps = np.linspace(0, ep_length - 1, ep_length)
    steps_avg = np.linspace(49, ep_length - 1, len(running_mean))

    # Plot individual chains
    for idx, (file, color) in enumerate(zip(files, colors)):
        chain_length = int(file.split("-")[-1].split(".")[0])
        ax.plot(
            steps,
            all_losses[idx][:ep_length],
            color=color,
            alpha=0.4,
            label=f"Chain-{chain_length}",
        )

    # Plot mean with confidence interval
    ax.plot(
        steps_avg,
        running_mean,
        color="orange",
        linewidth=2,
        label="Mean (Moving Avg, n=50)",
    )
    ax.fill_between(
        steps_avg,
        running_mean - running_std,
        running_mean + running_std,
        color="gray",
        alpha=0.5,
    )

    ax.set_yscale("log")
    ax.set_title("Epistemic Loss During Training")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Log Loss")
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig


def plot_rewards(files):
    """Plot Rewards metrics."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

    all_rewards = []
    for file in files:
        data = load_metrics(file)
        all_rewards.append(data["rewards"])

    _, _, _, raw_mean, raw_std, reward_length = process_metrics(all_rewards)
    steps = np.arange(reward_length)

    # Plot individual chains
    for idx, (file, color) in enumerate(zip(files, colors)):
        chain_length = int(file.split("-")[-1].split(".")[0])
        ax.plot(
            steps,
            all_rewards[idx][:reward_length],
            color=color,
            alpha=0.4,
            label=f"Chain-{chain_length}",
        )

    # Plot mean with confidence interval
    ax.plot(steps, raw_mean, color="orange", linewidth=3, label="Mean Across Chains")
    ax.fill_between(
        steps, raw_mean - raw_std, raw_mean + raw_std, color="gray", alpha=0.25
    )

    ax.set_title("Rewards During Training")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig


def plot_cumulative_returns(files):
    """Plot Cumulative Returns metrics."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

    all_returns = []
    for file in files:
        data = load_metrics(file)
        all_returns.append(data["cumulative_returns"])

    _, _, _, raw_mean, raw_std, returns_length = process_metrics(all_returns)
    steps = np.arange(returns_length)

    # Plot individual chains
    for idx, (file, color) in enumerate(zip(files, colors)):
        chain_length = int(file.split("-")[-1].split(".")[0])
        ax.plot(
            steps,
            all_returns[idx][:returns_length],
            color=color,
            alpha=0.4,
            label=f"Chain-{chain_length}",
        )

    # Plot mean with confidence interval
    ax.plot(steps, raw_mean, color="orange", linewidth=3, label="Mean Across Chains")
    ax.fill_between(
        steps, raw_mean - raw_std, raw_mean + raw_std, color="gray", alpha=0.25
    )

    ax.set_title("Cumulative Returns During Training")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig


def save_all_plots(files_pattern):
    """Generate and save individual plots for all metrics."""
    # Get and sort files
    files = glob.glob(files_pattern)
    files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

    # Generate and save individual plots
    plot_functions = {
        "q_loss": plot_q_loss,
        "epistemic_loss": plot_epistemic_loss,
        "rewards": plot_rewards,
        "cumulative_returns": plot_cumulative_returns,
    }

    for name, plot_func in plot_functions.items():
        fig = plot_func(files)
        fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    save_all_plots("ben_training_metrics_mean_chain-*.json")
