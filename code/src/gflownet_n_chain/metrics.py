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
    value_estimation_error: float


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
        self, logits: torch.Tensor, action_masks: torch.Tensor, temperature: float
    ) -> float:
        """Compute entropy of policy distribution.

        Args:
            logits: Raw logits from model [batch_size, num_actions]
            action_masks: Binary masks for valid actions [batch_size, num_actions]
            temperature: Softmax temperature

        Returns:
            float: Average entropy across batch
        """
        # Apply temperature and masking
        scaled_logits = logits / temperature
        masked_logits = scaled_logits + (action_masks + 1e-8).log()

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
            "avg_value_error": np.mean([m.value_estimation_error for m in recent]),
        }

        # State visitation statistics
        all_visits = defaultdict(list)
        for m in recent:
            for state, count in m.state_visits.items():
                all_visits[state].append(count)

        summary["state_coverage"] = len(all_visits) / max(all_visits.keys())

        for state, visits in all_visits.items():
            summary[f"state_{state}_visit_rate"] = np.mean(visits)

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
        fig.suptitle("Training Metrics")

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
        value_errors = [m.value_estimation_error for m in self.metrics_history]

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
        plot_metric(axes[2, 0], successes, "Success Rate", "Rate")
        plot_metric(axes[2, 1], value_errors, "Value Error", "Error")

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
