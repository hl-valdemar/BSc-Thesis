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


def plot_training_curves(files_pattern):
    """Plot training curves for multiple chain lengths."""
    # Set up the plot style
    # plt.style.use("seaborn")

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    ax1_log = ax1.twinx()  # instantiate a second axis that shares the same x-axis

    # Get all matching files
    files = glob.glob(files_pattern)

    # Extract and sort by the numeric value
    files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

    # Color map for different chain lengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

    # Initialize list to store rewards for mean calculation
    all_rewards = []
    first_length = None

    # First pass: Collect all rewards
    for file in files:
        metrics_history = load_metrics(file)
        metrics = extract_metrics(metrics_history)
        all_rewards.append(metrics["terminal_reward"])

        # Store the length of the first trajectory to verify others
        if first_length is None:
            first_length = len(metrics["terminal_reward"])
        elif len(metrics["terminal_reward"]) != first_length:
            raise ValueError(
                f"Inconsistent trajectory lengths found. Expected {first_length} but got {len(metrics['terminal_reward'])}"
            )

    def running_average(data, window_size=50):
        """Calculate running average with specified window size."""
        kernel = np.ones(window_size) / window_size
        return np.array([np.convolve(chain, kernel, mode="valid") for chain in data])

    # Convert to numpy array for calculations
    rewards_array = np.array(all_rewards)
    running_avg_rewards = running_average(rewards_array)

    # Calculate mean and std of running averages
    mean_running_rewards = np.mean(running_avg_rewards, axis=0)
    std_running_rewards = np.std(running_avg_rewards, axis=0)

    # Adjust steps to match the running average length
    window_size = 50  # Same as in running_average function
    steps_running_avg = range(window_size - 1, first_length)

    steps = range(first_length)

    # Second pass: Plot everything
    for file, color in zip(files, colors):
        # Extract chain length from filename
        chain_length = int(file.split("-")[-1].split(".")[0])

        # Load and extract metrics
        metrics_history = load_metrics(file)
        metrics = extract_metrics(metrics_history)

        # Plot curves
        ax1.plot(
            steps,
            metrics["trajectory_balance_loss"],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.4,
        )
        ax1_log.plot(
            steps,
            metrics["trajectory_balance_loss"],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.8,
        )
        # Plot individual reward trajectories with reduced opacity
        ax2.plot(
            steps_running_avg,
            metrics["terminal_reward"][window_size - 1 :],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.3,
        )
        ax3.plot(
            range(15),
            metrics["exploration_ratio"][:15],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.8,
        )
        ax4.plot(
            steps,
            metrics["forward_entropy"],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.8,
        )
        ax4.plot(
            steps,
            metrics["backward_entropy"],
            label=f"Chain-{chain_length}",
            color=color,
            alpha=0.8,
            linestyle="--",
        )

    # Plot mean reward and standard deviation
    ax2.plot(
        steps_running_avg,
        mean_running_rewards,
        label="Mean Reward (Moving Avg, n=50)",
        color=color,
        linewidth=2,
        alpha=1.0,
    )
    ax2.fill_between(
        steps_running_avg,
        mean_running_rewards - std_running_rewards,
        mean_running_rewards + std_running_rewards,
        color="gray",
        alpha=1.0,
    )

    # Customize plots
    ax1.set_title("Trajectory Balance Loss")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")

    ax1_log.set_yscale("log")
    ax1_log.set_title("Trajectory Balance Loss")
    ax1_log.set_xlabel("Training Steps")
    ax1_log.set_ylabel("Log Loss")
    ax1_log.legend()

    ax2.set_title("Terminal Reward")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Reward")
    ax2.legend()

    ax3.set_title("Exploration Ratio")
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("Ratio")
    ax3.legend()

    ax4.set_title("Forward/Backward Entropy")
    ax4.set_xlabel("Training Steps")
    ax4.set_ylabel("Entropy")
    ax4.legend()

    plt.tight_layout()
    return fig


plot_training_curves("gflownet_training_metrics_mean_chain-*.json")
plt.show()
