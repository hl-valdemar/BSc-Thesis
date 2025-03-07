from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GFlowNetMetrics:
    """Training metrics for a single GFlowNet episode.

    Attributes:
        trajectory_balance_loss: Loss measuring flow consistency
        terminal_state_reached: Whether trajectory reached terminal
        terminal_reward: Reward at terminal state (0 if not reached)
        log_Z: Current estimate of partition function
        trajectory_length: Number of steps in episode
        branch_chosen: Which branch was taken (-1 if none)
        forward_entropy: Average entropy of forward policy
        backward_entropy: Average entropy of backward policy
        exploration_ratio: Ratio of states visited vs not visited
    """

    trajectory_balance_loss: float
    terminal_state_reached: bool
    terminal_reward: float
    log_Z: float
    trajectory_length: int
    branch_chosen: int
    forward_entropy: float
    backward_entropy: float
    exploration_ratio: float


class MetricsTracker:
    """Tracks and analyzes GFlowNet training metrics over time."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: List[GFlowNetMetrics] = []

    def to_json(self) -> str:
        """Convert metrics history to JSON string.

        Returns:
            str: JSON representation of metrics history, with each entry containing:
                - trajectory_balance_loss: float
                - terminal_state_reached: bool
                - terminal_reward: float
                - log_Z: float
                - trajectory_length: int
                - branch_chosen: int
                - forward_entropy: float
                - backward_entropy: float
        """
        import json

        # Convert metrics history to list of dicts
        history = []
        for metric in self.metrics_history:
            history.append(
                {
                    "trajectory_balance_loss": metric.trajectory_balance_loss,
                    "terminal_state_reached": metric.terminal_state_reached,
                    "terminal_reward": metric.terminal_reward,
                    "log_Z": metric.log_Z,
                    "trajectory_length": metric.trajectory_length,
                    "branch_chosen": metric.branch_chosen,
                    "forward_entropy": metric.forward_entropy,
                    "backward_entropy": metric.backward_entropy,
                    "exploration_ratio": metric.exploration_ratio,
                }
            )

        # Add summary statistics
        summary_stats = self.get_summary_stats()

        # Need to handle the branch_dist dictionary separately since
        # its keys are integers which need to be converted to strings for JSON
        if "branch_dist" in summary_stats:
            summary_stats["branch_dist"] = {
                str(k): v for k, v in summary_stats["branch_dist"].items()
            }

        output = {
            "metrics_history": history,
            "summary_stats": summary_stats,
            "window_size": self.window_size,
        }

        return json.dumps(output, indent=2)

    def add_metrics(self, metrics: GFlowNetMetrics):
        """Add new metrics from a training episode."""
        self.metrics_history.append(metrics)

    def get_summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics over recent window."""
        if not self.metrics_history:
            return {}

        recent = self.metrics_history[-self.window_size :]

        # Flow consistency metrics
        avg_loss = np.mean([m.trajectory_balance_loss for m in recent])
        avg_log_Z = np.mean([m.log_Z for m in recent])

        # Terminal state metrics
        success_rate = np.mean([m.terminal_state_reached for m in recent])
        avg_reward = np.mean([m.terminal_reward for m in recent])

        # Branch selection metrics
        branch_dist = {
            -1: 0,
            0: 0,
            1: 0,
            2: 0,
        }  # Count of no-branch, left, right
        for m in recent:
            branch_dist[m.branch_chosen] += 1
        branch_dist = {k: v / len(recent) for k, v in branch_dist.items()}

        # Policy behavior metrics
        avg_forward_entropy = np.mean([m.forward_entropy for m in recent])
        avg_backward_entropy = np.mean([m.backward_entropy for m in recent])
        avg_exploration_ratio = np.mean([m.exploration_ratio for m in recent])

        return {
            "avg_loss": avg_loss,
            "avg_log_Z": avg_log_Z,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "branch_dist": branch_dist,
            "avg_forward_entropy": avg_forward_entropy,
            "avg_backward_entropy": avg_backward_entropy,
            "avg_exploration_ratio": avg_exploration_ratio,
        }

    def plot_training_curves(
        self,
        save_plot: bool = False,
        transparent: bool = False,
        file_name: Optional[str] = None,
    ):
        """Visualize key training metrics over time."""
        if not self.metrics_history:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Loss curve
        losses = [m.trajectory_balance_loss for m in self.metrics_history]
        axes[0, 0].plot(losses)
        axes[0, 0].set_title("Trajectory Balance Loss")
        axes[0, 0].set_yscale("log")

        # Plot 2: Terminal rewards
        rewards = [m.terminal_reward for m in self.metrics_history]
        axes[0, 1].plot(rewards)
        axes[0, 1].set_title("Terminal Rewards")

        # Plot 3: Branch distribution
        steps = list(range(len(self.metrics_history)))
        branches = [m.branch_chosen for m in self.metrics_history]
        for b in [-1, 0, 1, 2]:
            mask = [br == b for br in branches]
            axes[1, 0].scatter(
                [s for s, m in zip(steps, mask) if m],
                [b for m in mask if m],
                label=f"Branch {b}",
                alpha=0.5,
            )
        axes[1, 0].set_title("Branch Selection")
        axes[1, 0].legend()

        # Plot 4: Policy entropies
        f_ent = [m.forward_entropy for m in self.metrics_history]
        b_ent = [m.backward_entropy for m in self.metrics_history]
        axes[1, 1].plot(f_ent, label="Forward")
        axes[1, 1].plot(b_ent, label="Backward")
        axes[1, 1].set_title("Policy Entropies")
        axes[1, 1].legend()

        plt.tight_layout()
        if save_plot:
            plt.savefig(file_name, transparent=transparent)
        else:
            plt.show()
