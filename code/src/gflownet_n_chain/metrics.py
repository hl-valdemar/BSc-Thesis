from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TrainingMetrics:
    """Container for various training metrics.

    Attributes:
        loss: Training loss value
        episode_length: Length of trajectory
        episode_return: Total undiscounted return
        policy_entropy: Entropy of forward policy
        state_visits: Dictionary counting state visitations
        successful_episodes: Whether episode reached goal
        flow_values: Dictionary of flow values for each state
        pf_probs: Forward policy probabilities
        pb_probs: Backward policy probabilities
        value_estimation_error: Error in flow value estimation
        flow_normalization_error: Deviation from unit sum constraint
    """

    loss: float
    episode_length: int
    episode_return: float
    policy_entropy: float
    state_visits: Dict[int, int]
    successful_episodes: bool
    flow_values: Dict[int, float]
    pf_probs: Dict[int, np.ndarray]
    pb_probs: Dict[int, np.ndarray]
    flow_value_estimation_error: float
    flow_normalization_error: float


class MetricsTracker:
    """Tracks and aggregates training metrics."""

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Size of window for moving averages
        """
        self.window_size = window_size
        self.metrics_history: List[TrainingMetrics] = []

    def add_metrics(self, metrics: TrainingMetrics):
        """Add new metrics to history."""
        self.metrics_history.append(metrics)

    def compute_policy_entropy(
        self,
        logits: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> float:
        """Compute entropy of policy distribution.

        Args:
            logits: Raw logits from model [batch_size, num_actions]
            action_masks: Binary masks for valid actions [batch_size, num_actions]

        Returns:
            float: Average entropy across batch
        """
        # Apply masking
        masked_logits = logits + (action_masks + 1e-8).log()

        # Get probabilities
        probs = F.softmax(masked_logits, dim=-1)

        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * F.log_softmax(masked_logits, dim=-1)).sum(dim=-1)

        return entropy.mean().item()

    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary statistics over recent metrics.

        Returns:
            Dict containing averaged metrics over recent window
        """
        if not self.metrics_history:
            return {}

        recent = self.metrics_history[-self.window_size :]

        # Basic statistics
        summary = {
            "avg_loss": np.mean([m.loss for m in recent]),
            "avg_episode_length": np.mean([m.episode_length for m in recent]),
            "avg_episode_return": np.mean([m.episode_return for m in recent]),
            "avg_policy_entropy": np.mean([m.policy_entropy for m in recent]),
            "success_rate": np.mean([m.successful_episodes for m in recent]),
            "avg_value_error": np.mean([m.flow_value_estimation_error for m in recent]),
        }

        # State visitation statistics
        all_visits = defaultdict(list)
        for m in recent:
            # Calculate total visits for this trajectory
            total_visits = sum(m.state_visits.values())
            # Store normalized visit counts (as percentages)
            for state, count in m.state_visits.items():
                visit_rate = (count / total_visits) if total_visits > 0 else 0
                all_visits[state].append(visit_rate)

        max_state = max(all_visits.keys())
        summary["state_coverage"] = len(all_visits) / (
            max_state + 1
        )  # +1 because states are 0-based

        # Compute average visit rates across trajectories
        for state, rates in all_visits.items():
            summary[f"state_{state}_visit_rate"] = np.mean(rates)

        return summary

    def print_summary(self):
        """Print formatted summary of recent metrics."""
        summary = self.get_summary_metrics()
        if not summary:
            print("No metrics available yet")
            return

        print("\n=== Training Metrics Summary ===")
        print(f"Loss: {summary['avg_loss']:.4f}")
        print(f"Episode Length: {summary['avg_episode_length']:.1f}")
        print(f"Episode Return: {summary['avg_episode_return']:.2f}")
        print(f"Policy Entropy: {summary['avg_policy_entropy']:.4f}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"State Coverage: {summary['state_coverage']:.2%}")
        print(f"Value Estimation Error: {summary['avg_value_error']:.4f}")
        print("\nState Visitation Rates:")
        for k, v in summary.items():
            if k.startswith("state_") and k != "state_coverage":
                print(f"  {k}: {v:.2%}")
        print("=============================\n")

    def plot_metrics(self, include_raw: bool = False):
        """Plot training metrics over time.

        Args:
            include_raw: Whether to plot raw values alongside smoothed
        """
        if not self.metrics_history:
            print("No metrics to plot")
            return

        import matplotlib.pyplot as plt

        # Create subplots for different metrics
        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        fig.suptitle("GFlowNet Training Metrics")

        # Helper function for plotting with moving average
        def plot_metric(ax, values, title, ylabel):
            if include_raw:
                ax.plot(values, alpha=0.3, label="Raw")

            # Moving average
            if len(values) >= self.window_size:
                moving_avg = np.convolve(
                    values, np.ones(self.window_size) / self.window_size, mode="valid"
                )
                ax.plot(
                    range(self.window_size - 1, len(values)),
                    moving_avg,
                    label=f"Moving Average (w={self.window_size})",
                )

            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            if include_raw:
                ax.legend()

        # Extract metric histories
        losses = [m.loss for m in self.metrics_history]
        lengths = [m.episode_length for m in self.metrics_history]
        returns = [m.episode_return for m in self.metrics_history]
        entropies = [m.policy_entropy for m in self.metrics_history]
        successes = [m.successful_episodes for m in self.metrics_history]
        flow_value_errors = [
            m.flow_value_estimation_error for m in self.metrics_history
        ]

        # Get flow values per state
        all_states = sorted(self.metrics_history[-1].flow_values.keys())
        flow_values_per_state = {
            s: [m.flow_values.get(s, 0.0) for m in self.metrics_history]
            for s in all_states
        }

        # Plot standard metrics
        plot_metric(axes[0, 0], losses, "Training Loss", "Loss")
        plot_metric(axes[0, 1], lengths, "Episode Length", "Steps")
        plot_metric(axes[1, 0], returns, "Episode Return", "Return")
        plot_metric(axes[1, 1], entropies, "Policy Entropy", "Entropy")
        plot_metric(
            axes[2, 0],
            [m.flow_normalization_error for m in self.metrics_history],
            "Flow Normalization Error",
            "Error",
        )
        plot_metric(axes[2, 1], flow_value_errors, "Flow Value Error", "Error")

        # Plot flow values
        ax_flow = axes[3, 0]
        for state, values in flow_values_per_state.items():
            ax_flow.plot(values, label=f"State {state}", alpha=0.7)
        ax_flow.set_title("Flow Values per State")
        ax_flow.set_xlabel("Step")
        ax_flow.set_ylabel("Flow Value")
        ax_flow.grid(True)
        # ax_flow.legend()

        # Plot final flow distribution
        ax_final = axes[3, 1]
        final_flows = [self.metrics_history[-1].flow_values[s] for s in all_states]
        ax_final.bar(all_states, final_flows)
        ax_final.set_title("Final Flow Distribution")
        ax_final.set_xlabel("State")
        ax_final.set_ylabel("Flow Value")
        ax_final.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_state_visits(self):
        """Plot heatmap of state visitation frequencies."""
        if not self.metrics_history:
            print("No metrics to plot")
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get state visitation counts from recent window
        recent = self.metrics_history[-self.window_size :]
        state_visits = defaultdict(int)

        for m in recent:
            for state, count in m.state_visits.items():
                state_visits[state] += count

        # Convert to normalized frequencies
        total_visits = sum(state_visits.values())
        frequencies = {k: v / total_visits for k, v in state_visits.items()}

        # Create heatmap
        plt.figure(figsize=(10, 4))
        states = sorted(frequencies.keys())
        freqs = [frequencies[s] for s in states]

        sns.barplot(x=states, y=freqs)
        plt.title("State Visitation Distribution")
        plt.xlabel("State")
        plt.ylabel("Visitation Frequency")
        plt.show()

    def plot_policy_distribution(self):
        """Plot heatmap of forward and backward policy distributions over time."""
        if not self.metrics_history:
            print("No metrics to plot")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        # Get recent metrics
        recent = self.metrics_history[-self.window_size :]

        # Extract final forward and backward policies
        final_metrics = recent[-1]

        # Determine actual number of states from the policy probabilities
        num_states = (
            max(final_metrics.pf_probs.keys()) + 1
        )  # Since states are 0-indexed
        num_actions = len(
            next(iter(final_metrics.pf_probs.values()))
        )  # Get length of first prob array

        # Create matrices for forward and backward policies
        pf_matrix = np.zeros((num_states, num_actions))
        pb_matrix = np.zeros((num_states, num_actions))

        # Fill matrices only for states that exist in the policies
        for state in final_metrics.pf_probs.keys():
            pf_matrix[state] = final_metrics.pf_probs[state]
            pb_matrix[state] = final_metrics.pb_probs[state]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot forward policy
        sns.heatmap(
            pf_matrix,
            ax=ax1,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            xticklabels=["Stay", "Right"],
            yticklabels=[f"State {i}" for i in range(num_states)],
        )
        ax1.set_title("Forward Policy Distribution (PF)")
        ax1.set_xlabel("Actions")
        ax1.set_ylabel("States")

        # Plot backward policy
        sns.heatmap(
            pb_matrix,
            ax=ax2,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            xticklabels=["Stay", "Right"],
            yticklabels=[f"State {i}" for i in range(num_states)],
        )
        ax2.set_title("Backward Policy Distribution (PB)")
        ax2.set_xlabel("Actions")
        ax2.set_ylabel("States")

        plt.tight_layout()
        plt.show()

        # Plot policy changes over time
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Get policies over time for a few key states
        timestamps = np.linspace(
            0,
            len(self.metrics_history) - 1,
            min(10, len(self.metrics_history)),
            dtype=int,
        )

        # Only plot states that consistently appear in the data
        states_to_plot = []
        for state in range(num_states):
            if all(state in self.metrics_history[t].pf_probs for t in timestamps):
                states_to_plot.append(state)

        # If no consistent states found, use states from the last timestep
        if not states_to_plot:
            states_to_plot = sorted(list(self.metrics_history[-1].pf_probs.keys()))[:3]

        for state in states_to_plot:
            # Forward policy - probability of moving right
            # Only plot if state exists in policies
            if all(state in self.metrics_history[t].pf_probs for t in timestamps):
                right_probs = [
                    self.metrics_history[t].pf_probs[state][1] for t in timestamps
                ]
                ax1.plot(timestamps, right_probs, marker="o", label=f"State {state}")

            # Backward policy
            if all(state in self.metrics_history[t].pb_probs for t in timestamps):
                right_probs = [
                    self.metrics_history[t].pb_probs[state][1] for t in timestamps
                ]
                ax2.plot(timestamps, right_probs, marker="o", label=f"State {state}")

        ax1.set_title("Forward Policy Evolution (Probability of Moving Right)")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Probability")
        ax1.grid(True)
        # ax1.legend()

        ax2.set_title("Backward Policy Evolution (Probability of Moving Right)")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Probability")
        ax2.grid(True)
        # ax2.legend()

        plt.tight_layout()
        plt.show()
