import glob
import json
from math import floor

import matplotlib.pyplot as plt
import numpy as np


def calculate_state_space_size(chain_length):
    """Calculate total number of possible states for a given chain length."""
    n = chain_length
    return floor(n / 2) + 3 * floor(n / 2)


def calculate_exploration_ratio(states_visited, total_states):
    """Calculate exploration ratio based on unique states visited."""
    # Convert states to tuples for hashable type
    unique_states = set(map(tuple, states_visited))
    return len(unique_states) / total_states


def load_metrics(filepath):
    """Load metrics from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_metrics(data, chain_length):
    """Extract relevant metrics from BEN model data."""
    # Calculate total possible states for this chain length
    total_states = calculate_state_space_size(chain_length)

    # Calculate cumulative exploration ratios
    states_sequence = data["states_visited"]
    exploration_ratios = []

    # Track unique states seen up to each point
    unique_states_seen = set()
    for state in states_sequence:
        unique_states_seen.add(tuple(state))
        exploration_ratios.append(len(unique_states_seen) / total_states)

    return {
        "losses_q": data["losses_q"],
        "losses_epistemic": data["losses_epistemic"],
        "rewards": data["rewards"],
        "cumulative_returns": data["cumulative_returns"],
        "exploration_ratios": exploration_ratios,
    }


def running_average(data, window_size=50):
    """Calculate running average with specified window size."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="valid")


def plot_training_curves(files_pattern):
    """Plot training curves for multiple chain lengths."""
    # Set up the plot style
    # plt.style.use("seaborn")

    # Create a figure with subplots - now 3x2 to include exploration ratio
    fig, ((ax_q_log, ax_ep_log), (ax_rew, ax_ret), (ax_exp, ax_dummy)) = plt.subplots(
        3, 2, figsize=(15, 18)
    )

    # Hide the dummy axis
    ax_dummy.set_visible(False)

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
    all_exploration = []

    # First pass: Collect metrics
    for file in files:
        chain_length = int(file.split("-")[-1].split(".")[0])
        data = load_metrics(file)
        metrics = extract_metrics(data, chain_length)

        all_losses_q.append(metrics["losses_q"])
        all_losses_epistemic.append(metrics["losses_epistemic"])
        all_rewards.append(metrics["rewards"])
        all_returns.append(metrics["cumulative_returns"])
        all_exploration.append(metrics["exploration_ratios"])

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
    losses_q_data, mean_losses_q, std_losses_q, _, _, q_length = process_metrics(
        all_losses_q
    )
    losses_ep_data, mean_losses_ep, std_losses_ep, _, _, ep_length = process_metrics(
        all_losses_epistemic
    )
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
    (
        exploration_data,
        mean_exploration,
        std_exploration,
        raw_mean_exploration,
        raw_std_exploration,
        exp_length,
    ) = process_metrics(all_exploration)

    # Create step arrays
    q_steps = np.arange(q_length)
    ep_steps = np.linspace(0, q_length - 1, ep_length)
    reward_steps = np.arange(reward_length)
    exp_steps = np.arange(exp_length)

    # Plot data for each chain length
    for idx, (file, color) in enumerate(zip(files, colors)):
        chain_length = int(file.split("-")[-1].split(".")[0])

        # Loss plots
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
            alpha=0.3,
            label=f"Chain-{chain_length}",
        )
        ax_ret.plot(
            reward_steps,
            all_returns[idx][:returns_length],
            color=color,
            alpha=0.3,
            label=f"Chain-{chain_length}",
        )

        # Exploration ratio
        ax_exp.plot(
            exp_steps,
            all_exploration[idx][:exp_length],
            color=color,
            alpha=0.3,
            label=f"Chain-{chain_length}",
        )

    # Add mean curves with confidence intervals
    steps_avg = np.arange(window_size - 1, q_length)
    steps_ep_avg = np.linspace(window_size - 1, q_length - 1, len(mean_losses_ep))
    steps_reward_avg = np.arange(window_size - 1, reward_length)
    steps_exp_avg = np.arange(window_size - 1, exp_length)

    # Mean curves for losses
    ax_q_log.plot(steps_avg, mean_losses_q, color="black", linewidth=2, label="Mean")
    ax_q_log.fill_between(
        steps_avg,
        mean_losses_q - std_losses_q,
        mean_losses_q + std_losses_q,
        color="gray",
        alpha=0.2,
    )

    ax_ep_log.plot(
        steps_ep_avg, mean_losses_ep, color="black", linewidth=2, label="Mean"
    )
    ax_ep_log.fill_between(
        steps_ep_avg,
        mean_losses_ep - std_losses_ep,
        mean_losses_ep + std_losses_ep,
        color="gray",
        alpha=0.2,
    )

    # Raw means for rewards and returns
    ax_rew.plot(
        reward_steps,
        raw_mean_rewards,
        color="red",
        linewidth=2.5,
        label="Mean Across Chains",
    )
    ax_rew.fill_between(
        reward_steps,
        raw_mean_rewards - raw_std_rewards,
        raw_mean_rewards + raw_std_rewards,
        color="red",
        alpha=0.15,
    )

    ax_ret.plot(
        reward_steps,
        raw_mean_returns,
        color="red",
        linewidth=2.5,
        label="Mean Across Chains",
    )
    ax_ret.fill_between(
        reward_steps,
        raw_mean_returns - raw_std_returns,
        raw_mean_returns + raw_std_returns,
        color="red",
        alpha=0.15,
    )

    # Raw mean for exploration ratio
    ax_exp.plot(
        exp_steps,
        raw_mean_exploration,
        color="red",
        linewidth=2.5,
        label="Mean Across Chains",
    )
    ax_exp.fill_between(
        exp_steps,
        raw_mean_exploration - raw_std_exploration,
        raw_mean_exploration + raw_std_exploration,
        color="red",
        alpha=0.15,
    )

    # Set log scale for loss plots
    ax_q_log.set_yscale("log")
    ax_ep_log.set_yscale("log")

    # Customize plots
    ax_q_log.set_title("Q-Loss")
    ax_ep_log.set_title("Epistemic Loss")
    ax_rew.set_title("Rewards")
    ax_ret.set_title("Cumulative Returns")
    ax_exp.set_title("Exploration Ratio")

    # Set labels
    for ax in [ax_q_log, ax_ep_log, ax_rew, ax_ret, ax_exp]:
        ax.set_xlabel("Training Steps")
        ax.legend(loc="upper right")

    ax_q_log.set_ylabel("Log Loss")
    ax_ep_log.set_ylabel("Log Loss")
    ax_rew.set_ylabel("Reward")
    ax_ret.set_ylabel("Cumulative Return")
    ax_exp.set_ylabel("Ratio")

    # Set y-axis limits for exploration ratio
    ax_exp.set_ylim(0, 1)

    # Add common title
    plt.suptitle("BEN Model Training Metrics Across Chain Lengths", fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout()
    return fig


# Example usage
fig = plot_training_curves("ben_training_metrics_mean_chain-*.json")
plt.show()
