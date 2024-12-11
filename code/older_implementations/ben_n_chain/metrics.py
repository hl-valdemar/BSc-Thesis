from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TrainingMetrics:
    """Container for BEN training metrics.

    Attributes:
        q_loss: Q-network loss value
        aleatoric_loss: Aleatoric network loss value
        epistemic_loss: Epistemic network loss value
        episode_length: Length of trajectory
        episode_return: Total undiscounted return
        state_visits: Dictionary counting state visitations
        aleatoric_uncertainty: Mean aleatoric uncertainty
        epistemic_uncertainty: Mean epistemic uncertainty
        exploration_bonus: Mean exploration bonus
        successful_episodes: Whether episode reached goal
    """

    q_loss: float
    aleatoric_loss: float
    epistemic_loss: float
    episode_length: int
    episode_return: float
    state_visits: Dict[int, int]
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    exploration_bonus: float
    successful_episodes: bool


class MetricsTracker:
    """Tracks and aggregates BEN training metrics."""

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

    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary statistics over recent metrics."""
        if not self.metrics_history:
            return {}

        recent = self.metrics_history[-self.window_size :]

        # Basic statistics
        summary = {
            "avg_q_loss": np.mean([m.q_loss for m in recent]),
            "avg_aleatoric_loss": np.mean([m.aleatoric_loss for m in recent]),
            "avg_epistemic_loss": np.mean([m.epistemic_loss for m in recent]),
            "avg_episode_length": np.mean([m.episode_length for m in recent]),
            "avg_episode_return": np.mean([m.episode_return for m in recent]),
            "success_rate": np.mean([m.successful_episodes for m in recent]),
            "avg_aleatoric_uncertainty": np.mean(
                [m.aleatoric_uncertainty for m in recent]
            ),
            "avg_epistemic_uncertainty": np.mean(
                [m.epistemic_uncertainty for m in recent]
            ),
            "avg_exploration_bonus": np.mean([m.exploration_bonus for m in recent]),
        }

        # State visitation statistics
        all_visits = defaultdict(list)
        for m in recent:
            total_visits = sum(m.state_visits.values())
            for state, count in m.state_visits.items():
                visit_rate = (count / total_visits) if total_visits > 0 else 0
                all_visits[state].append(visit_rate)

        # Compute state coverage and visit rates
        max_state = max(k for m in recent for k in m.state_visits.keys())
        summary["state_coverage"] = len(all_visits) / (max_state + 1)
        for state, rates in all_visits.items():
            summary[f"state_{state}_visit_rate"] = np.mean(rates)

        return summary

    def print_summary(self):
        """Print formatted summary of recent metrics."""
        summary = self.get_summary_metrics()
        if not summary:
            print("No metrics available yet")
            return

        print("\n=== BEN Training Metrics Summary ===")
        print("Losses:")
        print(f"  Q-Network: {summary['avg_q_loss']:.4f}")
        print(f"  Aleatoric: {summary['avg_aleatoric_loss']:.4f}")
        print(f"  Epistemic: {summary['avg_epistemic_loss']:.4f}")
        print("\nPerformance:")
        print(f"  Episode Length: {summary['avg_episode_length']:.1f}")
        print(f"  Episode Return: {summary['avg_episode_return']:.2f}")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print("\nUncertainty:")
        print(f"  Aleatoric: {summary['avg_aleatoric_uncertainty']:.4f}")
        print(f"  Epistemic: {summary['avg_epistemic_uncertainty']:.4f}")
        print(f"  Exploration Bonus: {summary['avg_exploration_bonus']:.4f}")
        print(f"  State Coverage: {summary['state_coverage']:.2%}")
        print("\nState Visitation Rates:")
        for k, v in summary.items():
            if k.startswith("state_") and k != "state_coverage":
                print(f"  {k}: {v:.2%}")
        print("===================================\n")

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
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("BEN Training Metrics")

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
        losses = {
            "Q-Network": [m.q_loss for m in self.metrics_history],
            "Aleatoric": [m.aleatoric_loss for m in self.metrics_history],
            "Epistemic": [m.epistemic_loss for m in self.metrics_history],
        }
        lengths = [m.episode_length for m in self.metrics_history]
        returns = [m.episode_return for m in self.metrics_history]
        successes = [m.successful_episodes for m in self.metrics_history]
        uncertainties = {
            "Aleatoric": [m.aleatoric_uncertainty for m in self.metrics_history],
            "Epistemic": [m.epistemic_uncertainty for m in self.metrics_history],
            "Exploration Bonus": [m.exploration_bonus for m in self.metrics_history],
        }

        # Plot losses
        ax = axes[0, 0]
        for name, vals in losses.items():
            ax.plot(vals, label=name)
        ax.set_title("Training Losses")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()

        # Plot episode statistics
        plot_metric(axes[0, 1], lengths, "Episode Length", "Steps")
        plot_metric(axes[1, 0], returns, "Episode Return", "Return")
        plot_metric(axes[1, 1], successes, "Success Rate", "Rate")

        # Plot uncertainties
        ax = axes[2, 0]
        for name, vals in uncertainties.items():
            ax.plot(vals, label=name)
        ax.set_title("Uncertainty Measures")
        ax.set_xlabel("Step")
        ax.set_ylabel("Uncertainty")
        ax.grid(True)
        ax.legend()

        # Plot state coverage over time
        state_coverage = []
        window = self.window_size
        for i in range(len(self.metrics_history)):
            recent = self.metrics_history[max(0, i - window) : i + 1]
            states_visited = set()
            for m in recent:
                states_visited.update(m.state_visits.keys())
            max_state = max(max(m.state_visits.keys()) for m in recent)
            coverage = len(states_visited) / (max_state + 1)
            state_coverage.append(coverage)

        plot_metric(axes[2, 1], state_coverage, "State Coverage", "Coverage Rate")

        plt.tight_layout()
        plt.show()
