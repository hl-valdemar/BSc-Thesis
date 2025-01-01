import glob
import json

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(filepath):
    """Load metrics from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["metrics_history"]


def extract_metrics(metrics_history):
    """Extract relevant metrics from history."""
    return {
        "trajectory_balance_loss": [
            m["trajectory_balance_loss"] for m in metrics_history
        ],
        "terminal_reward": [m["terminal_reward"] for m in metrics_history],
        "exploration_ratio": [m["exploration_ratio"] for m in metrics_history],
        "forward_entropy": [m["forward_entropy"] for m in metrics_history],
        "backward_entropy": [m["backward_entropy"] for m in metrics_history],
    }


def running_average(data, window_size=50):
    """Calculate running average with specified window size."""
    kernel = np.ones(window_size) / window_size
    return np.array([np.convolve(chain, kernel, mode="valid") for chain in data])


def plot_trajectory_balance_loss(files, first_length, colors):
    """Plot trajectory balance loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    steps = range(first_length)

    for file, color in zip(files, colors):
        chain_length = int(file.split("-")[-1].split(".")[0])
        metrics = extract_metrics(load_metrics(file))

        # Linear scale plot
        ax1.plot(
            steps,
            metrics["trajectory_balance_loss"],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.8,
        )

        # Log scale plot
        ax2.plot(
            steps,
            metrics["trajectory_balance_loss"],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.8,
        )
        ax2.set_yscale("log")

    ax1.set_title("Trajectory Balance Loss (Linear Scale)")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.set_title("Trajectory Balance Loss (Log Scale)")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Log Loss")
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_terminal_reward(files, first_length, colors):
    """Plot terminal reward with running average."""
    fig, ax = plt.subplots(figsize=(10, 6))
    window_size = 50
    steps_running_avg = range(window_size - 1, first_length)

    # Collect rewards for mean calculation
    all_rewards = []
    for file in files:
        metrics = extract_metrics(load_metrics(file))
        all_rewards.append(metrics["terminal_reward"])

    rewards_array = np.array(all_rewards)
    running_avg_rewards = running_average(rewards_array)
    mean_running_rewards = np.mean(running_avg_rewards, axis=0)
    std_running_rewards = np.std(running_avg_rewards, axis=0)

    # Plot individual trajectories
    for file, color in zip(files, colors):
        chain_length = int(file.split("-")[-1].split(".")[0])
        metrics = extract_metrics(load_metrics(file))
        ax.plot(
            steps_running_avg,
            metrics["terminal_reward"][window_size - 1 :],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.3,
        )

    # Plot mean and std deviation
    ax.plot(
        steps_running_avg,
        mean_running_rewards,
        label="Mean (Moving Avg, n=50)",
        color=color,
        linewidth=2,
        alpha=1.0,
    )
    ax.fill_between(
        steps_running_avg,
        mean_running_rewards - std_running_rewards,
        mean_running_rewards + std_running_rewards,
        color="gray",
        alpha=1.0,
    )

    ax.set_title("Terminal Reward")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_exploration_ratio(files, first_length, colors):
    """Plot exploration ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for file, color in zip(files, colors):
        chain_length = int(file.split("-")[-1].split(".")[0])
        metrics = extract_metrics(load_metrics(file))
        ax.plot(
            range(15),
            metrics["exploration_ratio"][:15],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.8,
        )

    ax.set_title("Exploration Ratio")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Ratio")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_entropy(files, first_length, colors):
    """Plot forward and backward entropy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = range(first_length)

    for file, color in zip(files, colors):
        chain_length = int(file.split("-")[-1].split(".")[0])
        metrics = extract_metrics(load_metrics(file))

        ax.plot(
            steps,
            metrics["forward_entropy"],
            label=f"Chain-{chain_length} (Forward)",
            color=color,
            alpha=0.8,
        )
        ax.plot(
            steps,
            metrics["backward_entropy"],
            label=f"Chain-{chain_length} (Backward)",
            color=color,
            alpha=0.8,
            linestyle="--",
        )

    ax.set_title("Forward/Backward Entropy")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Entropy")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_all_metrics(files_pattern):
    """Generate and save individual plots for all metrics."""
    # Get and sort files
    files = glob.glob(files_pattern)
    files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

    # Set up colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

    # Get first trajectory length for consistency
    first_metrics = extract_metrics(load_metrics(files[0]))
    first_length = len(first_metrics["terminal_reward"])

    # Generate individual plots
    figs = {
        "trajectory_balance_loss": plot_trajectory_balance_loss(
            files, first_length, colors
        ),
        "terminal_reward": plot_terminal_reward(files, first_length, colors),
        "exploration_ratio": plot_exploration_ratio(files, first_length, colors),
        "entropy": plot_entropy(files, first_length, colors),
    }

    # Save individual plots
    for name, fig in figs.items():
        fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return figs


# Generate and save all plots
plot_all_metrics("gflownet_training_metrics_mean_chain-*.json")
