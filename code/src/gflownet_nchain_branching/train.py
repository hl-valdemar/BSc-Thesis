import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from nchain_branching import NChainEnv, NChainState, NChainTrajectory

from .metrics import GFlowNetMetrics, MetricsTracker
from .model import GFlowNet


class GFlowNetTrainer:
    def __init__(
        self,
        env: NChainEnv,
        model: GFlowNet,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        buffer_size: int = 10_000,
        metrics_window: int = 100,
    ) -> None:
        """
        Args:
            env: N-chain environment to train on
            model: GFlowNet model to train
        """
        self.env = env
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.replay_buffer: Deque[NChainTrajectory] = deque(maxlen=buffer_size)
        self.metrics = MetricsTracker(window_size=metrics_window)

    def get_tempered_policy(
        self,
        state_tensor: Tensor,
        action_mask: Tensor,
        epsilon: float = 0.1,
    ) -> Tensor:
        """Get exploration-enabled policy probabilities.

        Args:
            state_tensor: Current state [1, state_dim]
            action_mask: Valid actions mask [1, num_actions]
            temperature: Softmax temperature (higher = more uniform)
            epsilon: Probability of random action
        """
        policy = self.model.get_forward_policy(state_tensor)
        policy = policy * action_mask

        # Add exploration through epsilon-random
        num_valid = action_mask.sum().item()
        random_policy = action_mask / num_valid
        return (1 - epsilon) * policy + epsilon * random_policy

    def collect_trajectory(self) -> NChainTrajectory:
        """
        Collect a single trajectory using current policy.

        Returns:
            NChainTrajectory
        """
        state = self.env.reset()
        done = False

        states = [state]
        actions = []
        rewards = []
        action_masks = []

        while not done:
            valid_actions = self.env.get_valid_actions(state)
            state_tensor = state.to_tensor().unsqueeze(
                0
            )  # Shape: [batch_dim = 1, state_dim]

            # Create action mask
            action_mask = torch.zeros(
                1,
                self.env.num_actions,
                dtype=bool,
            )  # Shape: [1, self.env.num_actions]
            action_mask[0, valid_actions] = 1.0

            # Get masked action probabilities
            # action_probs = self.model.get_forward_policy(state_tensor)
            # action_probs = action_probs * action_mask

            action_probs = self.get_tempered_policy(
                state_tensor=state_tensor,
                action_mask=action_mask,
            )

            action = torch.multinomial(
                input=action_probs,
                num_samples=1,
            ).item()  # NOTE: make sure this samples from the correct dimension

            next_state, reward, done = self.env.step(action)

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            action_masks.append(action_mask)

            state = next_state

        return NChainTrajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            done=done,
            valid_actions_masks=action_masks,
        )

    def _create_action_mask(self, state: NChainState) -> Tensor:
        """
        Returns an action mask.

        Returns:
            Tensor: The action mask
        """
        valid_actions = self.env.get_valid_actions(state)
        action_mask = torch.zeros(
            1,
            self.env.num_actions,
            dtype=bool,
        )  # Shape: [1, self.env.num_actions]
        action_mask[0, valid_actions] = 1.0

        return action_mask

    def collect_trajectory_with_metrics(
        self,
    ) -> Tuple[NChainTrajectory, Dict[str, float]]:
        """Collect trajectory and compute relevant metrics."""
        state = self.env.reset()
        done = False

        states = [state]
        actions = []
        rewards = []
        action_masks = []

        # Track policy entropy during collection
        forward_entropies = []
        backward_entropies = []

        while not done:
            state_tensor = state.to_tensor().unsqueeze(0)
            action_mask = self._create_action_mask(state)

            # Get policies and compute entropies
            forward_policy = self.get_tempered_policy(
                state_tensor=state_tensor,
                action_mask=action_mask,
            )
            backward_policy = self.model.get_backward_policy(state_tensor)

            forward_entropies.append(
                -(forward_policy * torch.log(forward_policy + 1e-10)).sum().item()
            )
            backward_entropies.append(
                -(backward_policy * torch.log(backward_policy + 1e-10)).sum().item()
            )

            # Sample action and step environment
            action = torch.multinomial(forward_policy, 1).item()
            next_state, reward, done = self.env.step(action)

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            action_masks.append(action_mask)

            state = next_state

        trajectory = NChainTrajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            done=done,
            valid_actions_masks=action_masks,
        )

        # Compute episode metrics
        metrics = GFlowNetMetrics(
            trajectory_balance_loss=0.0,  # Will be updated during training
            terminal_state_reached=done,
            terminal_reward=rewards[-1] if done else 0.0,
            log_Z=self.model.log_Z.item(),
            trajectory_length=len(states),
            branch_chosen=states[-1].branch,
            forward_entropy=np.mean(forward_entropies),
            backward_entropy=np.mean(backward_entropies),
        )

        return trajectory, metrics

    def compute_batch_loss(self, batch: List[NChainTrajectory]) -> Tensor:
        """
        Compute weighted trajectory balance loss for a batch.

        Args:
            batch: Batch of trajectories

        Returns:
            Tensor: the total loss weighted by trajectory length
        """
        total_loss = torch.tensor(0.0)
        total_weight = torch.tensor(0.0)

        for trajectory in batch:
            state_tensors = [s.to_tensor() for s in trajectory.states]

            loss = self.model.compute_trajectory_balance_loss(
                states=state_tensors,
                actions=trajectory.actions,
                rewards=trajectory.rewards,
                terminated=trajectory.done,
                valid_actions_mask=trajectory.valid_actions_masks,
            )

            # Weight by trajectory length to balance short/long paths
            weight = len(trajectory.states)
            total_loss += weight * loss
            total_weight += weight

        return total_loss / total_weight if total_weight > 0 else total_loss

    def add_to_buffer(self, trajectory: NChainTrajectory) -> None:
        """
        Add trajectory to buffer with diversity consideration.

        Args:
            trajectory: Trajectory to add to buffer
        """
        if len(self.replay_buffer) == self.buffer_size:
            # Prefer keeping diverse trajectories
            if trajectory.rewards[-1] > 0:  # Terminal reward
                # Replace a zero-reward trajectory if possible
                for i, old_traj in enumerate(self.replay_buffer):
                    if old_traj.rewards[-1] == 0:
                        self.replay_buffer[i] = trajectory
                        return

        self.replay_buffer.append(trajectory)

    def train_step(self) -> Dict[str, float]:
        """Perform single training step and return metrics.

        Returns:
            Dict: Train step metrics
        """
        # Collect trajectory and initial metrics
        trajectory, metrics = self.collect_trajectory_with_metrics()
        self.replay_buffer.append(trajectory)

        if len(self.replay_buffer) < self.batch_size:
            self.metrics.add_metrics(metrics)
            return {"loss": 0.0}

        # Compute loss on batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        self.optimizer.zero_grad()

        loss = self.compute_batch_loss(batch)
        loss.backward()

        # Update metrics with training loss
        metrics.trajectory_balance_loss = loss.item()
        self.metrics.add_metrics(metrics)

        # Optimize
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "terminal_reward": metrics.terminal_reward,
            "log_Z": metrics.log_Z,
        }

    def train(self, num_steps: int, report_interval: int = 10) -> None:
        """Run training loop with periodic metric reporting."""
        for step in range(num_steps):
            metrics = self.train_step()

            if (step + 1) % report_interval == 0:
                summary = self.metrics.get_summary_stats()
                print(f"\nStep {step+1}/{num_steps}")
                print(f"Average Loss: {summary['avg_loss']:.4f}")
                print(f"Success Rate: {summary['success_rate']:.2%}")
                print(
                    "Branch Distribution:",
                    {k: f"{v:.2%}" for k, v in summary["branch_dist"].items()},
                )
