import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from .model import GFlowNetBase


@dataclass
class TrainingConfig:
    """Configuration for GFlowNet training."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 100
    max_grad_norm: float = 1.0
    temperature: float = 1.0  # For policy sampling
    validation_interval: int = 10
    flow_matching_weight: float = 1.0
    trajectory_balance_weight: float = 0.0
    entropy_weight: float = 0.01
    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ExperienceBuffer:
    """Buffer for storing and sampling trajectories."""

    def __init__(self, max_size: int = 10000):
        self.trajectories: List[List[Tuple[torch.Tensor, int]]] = []
        self.rewards: List[float] = []
        self.max_size = max_size

    def add(self, trajectory: List[Tuple[torch.Tensor, int]], reward: float):
        """Add trajectory and its reward to buffer."""
        if len(self.trajectories) >= self.max_size:
            self.trajectories.pop(0)
            self.rewards.pop(0)
        self.trajectories.append(trajectory)
        self.rewards.append(reward)

    def sample(
        self, batch_size: int
    ) -> Tuple[List[List[Tuple[torch.Tensor, int]]], List[float]]:
        """Sample batch of trajectories and rewards."""
        if batch_size >= len(self.trajectories):
            return self.trajectories.copy(), self.rewards.copy()

        indices = np.random.choice(len(self.trajectories), batch_size, replace=False)
        return (
            [self.trajectories[i] for i in indices],
            [self.rewards[i] for i in indices],
        )


class GFlowNetTrainer:
    """Trainer class for GFlowNet."""

    def __init__(
        self,
        model: GFlowNetBase,
        config: TrainingConfig,
        reward_function: callable,
        exp_buffer: Optional[ExperienceBuffer] = None,
    ):
        """
        Args:
            model: GFlowNet model
            config: Training configuration
            reward_function: Function that computes reward for terminal states
            exp_buffer: Optional experience buffer for replay
        """
        self.model = model.to(config.device)
        self.config = config
        self.reward_function = reward_function
        self.exp_buffer = exp_buffer or ExperienceBuffer()

        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("GFlowNetTrainer")

    def compute_losses(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all relevant losses.

        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size]
            rewards: Batch of rewards [batch_size]
            masks: Optional action masks [batch_size, n_actions]

        Returns:
            losses: Dictionary of loss components
        """
        policy_logits, state_flow, edge_flows = self.model(states)

        # Flow matching loss
        flow_matching_loss = self.model.compute_flow_matching_loss(
            states, policy_logits, state_flow, edge_flows, masks
        )

        # Policy entropy for exploration
        policy_probs = torch.softmax(policy_logits / self.config.temperature, dim=-1)
        entropy = (
            -(policy_probs * torch.log_softmax(policy_logits, dim=-1))
            .sum(dim=-1)
            .mean()
        )

        # Optional: trajectory balance loss if using both objectives
        trajectory_balance_loss = torch.tensor(0.0, device=states.device)
        if self.config.trajectory_balance_weight > 0:
            log_rewards = torch.log(rewards + 1e-8)
            log_probs = torch.log_softmax(policy_logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(len(actions)), actions]
            trajectory_balance_loss = (log_rewards - selected_log_probs).pow(2).mean()

        losses = {
            "flow_matching": flow_matching_loss,
            "entropy": -entropy,  # Negative because we want to maximize entropy
            "trajectory_balance": trajectory_balance_loss,
        }

        return losses

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []

        # Sample trajectories
        n_samples = self.config.batch_size * 10  # Generate multiple trajectories
        all_trajectories = []
        all_rewards = []

        for _ in range(n_samples):
            initial_state = torch.zeros(self.model.state_dim, device=self.config.device)
            initial_state[0] = 1  # Start position

            trajectory = self.model.sample_trajectory(initial_state)
            if trajectory:  # Only add if trajectory is valid
                final_state = trajectory[-1][0]
                reward = self.reward_function(final_state)

                all_trajectories.append(trajectory)
                all_rewards.append(reward)
                self.exp_buffer.add(trajectory, reward)

        # Training loop
        n_batches = len(all_trajectories) // self.config.batch_size
        for batch_idx in range(n_batches):
            self.optimizer.zero_grad()

            # Sample batch of trajectories
            batch_trajectories, batch_rewards = self.exp_buffer.sample(
                self.config.batch_size
            )

            # Prepare batch data
            states, actions, rewards = [], [], []
            for traj, rew in zip(batch_trajectories, batch_rewards):
                for state, action in traj:
                    states.append(state)
                    actions.append(action)
                    rewards.append(rew)  # Use final reward for all steps

            states = torch.stack(states).to(self.config.device)
            actions = torch.tensor(actions, device=self.config.device)
            rewards = torch.tensor(rewards, device=self.config.device)

            # Compute losses
            losses = self.compute_losses(states, actions, rewards)

            # Combine losses
            total_loss = (
                self.config.flow_matching_weight * losses["flow_matching"]
                + self.config.trajectory_balance_weight * losses["trajectory_balance"]
                + self.config.entropy_weight * losses["entropy"]
            )

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            self.optimizer.step()

            # Log batch metrics
            epoch_losses.append({k: v.item() for k, v in losses.items()})

        # Compute epoch metrics
        avg_losses = {
            k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()
        }
        avg_reward = np.mean(all_rewards)

        metrics = {
            **avg_losses,
            "avg_reward": avg_reward,
            "n_trajectories": len(all_trajectories),
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate model performance."""
        self.model.eval()
        validation_rewards = []
        validation_lengths = []

        with torch.no_grad():
            for _ in range(100):  # Validate with 100 trajectories
                initial_state = torch.zeros(
                    self.model.state_dim, device=self.config.device
                )
                initial_state[0] = 1  # Start position

                trajectory = self.model.sample_trajectory(initial_state)
                if trajectory:
                    final_state = trajectory[-1][0]
                    reward = self.reward_function(final_state)

                    validation_rewards.append(reward)
                    validation_lengths.append(len(trajectory))

        metrics = {
            "val_reward_mean": np.mean(validation_rewards),
            "val_reward_std": np.std(validation_rewards),
            "val_length_mean": np.mean(validation_lengths),
            "val_length_std": np.std(validation_lengths),
        }

        return metrics

    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        history = {
            "train_rewards": [],
            "val_rewards": [],
            "flow_matching_losses": [],
            "entropy_losses": [],
            "trajectory_balance_losses": [],
        }

        best_val_reward = float("-inf")

        for epoch in tqdm(range(self.config.n_epochs), desc="Training"):
            # Training
            train_metrics = self.train_epoch()

            # Logging
            self.logger.info(f"Epoch {epoch+1}/{self.config.n_epochs}")
            self.logger.info(f"Training metrics: {train_metrics}")

            # Validation
            if (epoch + 1) % self.config.validation_interval == 0:
                val_metrics = self.validate()
                self.logger.info(f"Validation metrics: {val_metrics}")

                # Learning rate scheduling
                self.scheduler.step(val_metrics["val_reward_mean"])

                # Save best model
                if val_metrics["val_reward_mean"] > best_val_reward:
                    best_val_reward = val_metrics["val_reward_mean"]
                    torch.save(
                        self.model.state_dict(),
                        f"{self.config.checkpoint_dir}/best_model.pt",
                    )

            # Update history
            history["train_rewards"].append(train_metrics["avg_reward"])
            history["flow_matching_losses"].append(train_metrics["flow_matching"])
            history["entropy_losses"].append(train_metrics["entropy"])
            history["trajectory_balance_losses"].append(
                train_metrics["trajectory_balance"]
            )

            if (epoch + 1) % self.config.validation_interval == 0:
                history["val_rewards"].append(val_metrics["val_reward_mean"])

        return history
