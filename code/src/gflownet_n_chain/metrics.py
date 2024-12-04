from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingMetrics:
    """Enhanced container for GFlowNet training metrics."""

    # Basic metrics
    partition_estimate: float
    loss: float
    episode_length: int
    episode_return: float
    # state_visits: Dict[int, int]  # state_pos -> visit_count
    successful_episodes: bool

    # Flow metrics
    flow_values: Dict[int, float]  # state_idx -> flow_value
    flow_value_estimation_error: float
    # flow_normalization_error: float

    # Path analysis metrics
    path_length: int
    optimal_length: int
    path_efficiency: float
    flows_by_path_length: Dict[int, float]  # length -> avg_flow
    rewards_by_path_length: Dict[int, float]  # length -> avg_reward
    path_length_distribution: Dict[int, float]  # length -> probability


class MetricsTracker:
    """Enhanced metrics tracker with path length analysis."""

    def __init__(self, window_size: int = 100):
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
            "avg_z_estimate": np.mean([m.partition_estimate for m in recent]),
            "avg_loss": np.mean([m.loss for m in recent]),
            "avg_episode_length": np.mean([m.episode_length for m in recent]),
            "avg_episode_return": np.mean([m.episode_return for m in recent]),
            "success_rate": np.mean([m.successful_episodes for m in recent]),
            "avg_flow_error": np.mean([m.flow_value_estimation_error for m in recent]),
            # "avg_normalization_error": np.mean(
            #     [m.flow_normalization_error for m in recent]
            # ),
        }

        # Path efficiency metrics
        summary.update(
            {
                "avg_path_length": np.mean([m.path_length for m in recent]),
                "avg_optimal_length": np.mean([m.optimal_length for m in recent]),
                "avg_path_efficiency": np.mean([m.path_efficiency for m in recent]),
            }
        )

        # # State coverage statistics
        # all_visits = defaultdict(list)
        # for m in recent:
        #     total_visits = sum(m.state_visits.values())
        #     for state, count in m.state_visits.items():
        #         all_visits[state].append(
        #             count / total_visits if total_visits > 0 else 0
        #         )
        #
        # summary["state_coverage"] = len(all_visits) / (max(all_visits.keys()) + 1)

        # Path length distribution statistics
        all_path_lengths = defaultdict(list)
        for m in recent:
            for length, prob in m.path_length_distribution.items():
                all_path_lengths[length].append(prob)

        summary["path_length_entropy"] = self.compute_distribution_entropy(
            {k: np.mean(v) for k, v in all_path_lengths.items()}
        )

        return summary

    @staticmethod
    def compute_distribution_entropy(probs: Dict[int, float]) -> float:
        """Compute entropy of a discrete distribution."""
        probs_array = np.array(list(probs.values()))
        probs_array = probs_array / probs_array.sum()  # Normalize
        return -np.sum(probs_array * np.log(probs_array + 1e-8))

    def print_summary(self):
        """Print formatted summary of recent metrics."""
        summary = self.get_summary_metrics()
        if not summary:
            print("No metrics available yet")
            return

        print("\n=== GFlowNet Training Metrics Summary ===")
        print(f"  Z Estimate: {summary['avg_z_estimate']:.4f}")

        print("\nPerformance:")
        print(f"  Loss: {summary['avg_loss']:.4f}")
        print(f"  Episode Return: {summary['avg_episode_return']:.2f}")
        print(f"  Success Rate: {summary['success_rate']:.2%}")

        print("\nPath Analysis:")
        print(f"  Average Path Length: {summary['avg_path_length']:.1f}")
        print(f"  Average Optimal Length: {summary['avg_optimal_length']:.1f}")
        print(f"  Path Efficiency: {summary['avg_path_efficiency']:.2%}")
        print(f"  Path Length Entropy: {summary['path_length_entropy']:.2f}")

        print("\nFlow Analysis:")
        print(f"  Flow Estimation Error: {summary['avg_flow_error']:.4f}")
        # print(f"  Flow Normalization Error: {summary['avg_normalization_error']:.4f}")
        # print(f"  State Coverage: {summary['state_coverage']:.2%}")
        print("=====================================\n")

    def plot_metrics(self, include_raw: bool = False):
        """Enhanced plotting with path analysis."""
        if not self.metrics_history:
            print("No metrics to plot")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("GFlowNet Training Metrics")

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

        # Plot basic metrics
        plot_metric(
            axes[0, 0],
            [m.loss for m in self.metrics_history],
            "Training Loss",
            "Loss",
        )
        plot_metric(
            axes[0, 1],
            [m.episode_return for m in self.metrics_history],
            "Episode Return",
            "Return",
        )

        # Plot path efficiency metrics
        plot_metric(
            axes[1, 0],
            [m.path_efficiency for m in self.metrics_history],
            "Path Efficiency",
            "Optimal Length / Actual Length",
        )

        # Plot path length distribution over time
        ax = axes[1, 1]
        recent = self.metrics_history[-1]
        path_lengths = list(recent.path_length_distribution.keys())
        probabilities = list(recent.path_length_distribution.values())
        ax.bar(path_lengths, probabilities)
        ax.set_title("Path Length Distribution")
        ax.set_xlabel("Path Length")
        ax.set_ylabel("Probability")

        # Plot flow analysis
        plot_metric(
            axes[2, 0],
            [m.flow_value_estimation_error for m in self.metrics_history],
            "Flow Estimation Error",
            "Error",
        )

        # Plot partition function metrics
        plot_metric(
            axes[2, 1],
            [m.partition_estimate for m in self.metrics_history],
            "Z Estimate (Partition Function)",
            "Z",
        )

        plt.tight_layout()
        plt.show()
