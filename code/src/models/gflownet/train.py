import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from environments.nchain import NChainAction, NChainEnv, NChainState, NChainTrajectory

from .metrics import GFlowNetMetrics, MetricsTracker
from .model import GFlowNet


class GFlowNetTrainer:
    def __init__(
        self,
        env: NChainEnv,
        model: GFlowNet,
        learning_rate: float = 1e-4,
        epsilon: float = 0.1,
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
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.replay_buffer: Deque[NChainTrajectory] = deque(maxlen=buffer_size)
        self.metrics = MetricsTracker(window_size=metrics_window)

    def get_tempered_forward_policy(
        self,
        state_tensor: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        """Get exploration-enabled policy probabilities.

        Args:
            state_tensor: Current state [batch_size, state_dim]
            action_mask: Valid actions mask [batch_size, num_actions]
        """
        policy = self.model.get_forward_policy(state_tensor, action_mask)

        # Add exploration through epsilon-random
        num_valid = action_mask.sum().item()
        random_policy = action_mask / num_valid
        return (1 - self.epsilon) * policy + self.epsilon * random_policy

    def create_forward_mask(self, state: NChainState) -> Tensor:
        """
        Returns a forward action mask.

        Returns:
            Tensor: The forward action mask
        """
        valid_actions = [int(a) for a in self.env.get_valid_actions(state)]
        mask = torch.zeros(1, self.env.num_actions, dtype=bool)
        mask[0, valid_actions] = True
        return mask

    def create_backward_mask(self, state: NChainState) -> Tensor:
        """
        Returns a backward action mask.

        Returns:
            Tensor: The backward action mask
        """
        mask = torch.zeros(1, self.env.num_actions, dtype=bool)

        if state.position == 0:  # Initial state has no parents
            return mask

        if state.branch == -1:  # Pre-split states
            if state.position > 0:  # Only forward action possible
                mask[0, int(NChainAction.FORWARD)] = True

        elif state.position == self.env.split_point + 1:  # Just after split
            # The mask should indicate which branch selection brought us here
            if state.branch == 0:  # Branch 0 was chosen
                mask[0, int(NChainAction.BRANCH_0)] = True
            elif state.branch == 1:  # Branch 1 was chosen
                mask[0, int(NChainAction.BRANCH_1)] = True
            elif state.branch == 2:
                mask[0, int(NChainAction.BRANCH_2)] = True

        elif self.env.is_terminal(state):  # Reached the terminal state
            mask[0, int(NChainAction.FORWARD)] = True
            mask[0, int(NChainAction.TERMINAL_STAY)] = True

        else:  # On branch after split point
            mask[0, int(NChainAction.FORWARD)] = True

        return mask

    def collect_trajectory_with_metrics(
        self,
    ) -> Tuple[NChainTrajectory, Dict[str, float]]:
        """Collect trajectory and compute relevant metrics."""
        state = self.env.reset()
        done = False

        states = [state]
        actions = []
        rewards = []
        forward_masks = []
        backward_masks = []

        # Track policy entropy during collection
        forward_entropies = []
        backward_entropies = []

        while not done:
            state_tensor = state.to_tensor().unsqueeze(
                0
            )  # Shape: [batch_dim = 1, state_dim]
            forward_mask = self.create_forward_mask(state)
            backward_mask = self.create_backward_mask(state)

            # Get policies and compute entropies
            # forward_policy = self.model.get_forward_policy(state_tensor, forward_mask)
            forward_policy = self.get_tempered_forward_policy(
                state_tensor, forward_mask
            )
            forward_entropies.append(
                -(forward_policy * torch.log(forward_policy + 1e-10)).sum().item()
            )

            if len(actions) > 0:  # Backward policy doesn't exist for the initial state
                backward_policy = self.model.get_backward_policy(
                    state_tensor, backward_mask
                )
                backward_entropies.append(
                    -(backward_policy * torch.log(backward_policy + 1e-10)).sum().item()
                )

            # Sample action and step environment
            action = torch.multinomial(forward_policy, 1).item()
            next_state, reward, done = self.env.step(NChainAction(action))

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            forward_masks.append(forward_mask)
            backward_masks.append(backward_mask)

            state = next_state

        # Add the backward mask for the terminal state
        terminal_backward_mask = self.create_backward_mask(state)
        backward_masks.append(terminal_backward_mask)

        trajectory = NChainTrajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            done=done,
            forward_masks=forward_masks,
            backward_masks=backward_masks,
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
                forward_masks=trajectory.forward_masks,
                backward_masks=trajectory.backward_masks,
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
        """
        Perform single training step and return metrics.

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

    def evaluate_learned_policy(
        self,
        num_trajectories: int = 200,
    ) -> Dict[str, float]:
        """
        Evaluate the learned policy by sampling trajectories.

        Args:
            num_trajectories: Number of trajectories to sample

        Returns:
            Dictionary containing:
                - reward_counts: Number of times each reward was reached
                - reward_frequencies: Distribution of terminal rewards
                - branch_count: Number of times each branch was selected
                - branch_frequencies: Distribution of branch selections
                - average_trajectory_length: Mean steps to completion
        """
        rewards = []
        branches = []
        trajectory_lengths = []

        # Turn off gradients for evaluation
        with torch.no_grad():
            for _ in range(num_trajectories):
                state = self.env.reset()
                done = False
                steps = 0

                while not done:
                    state_tensor = state.to_tensor().unsqueeze(0)
                    action_mask = self.create_forward_mask(state)

                    # Use learned policy directly (no exploration)
                    policy = self.model.get_forward_policy(state_tensor, action_mask)
                    action = torch.multinomial(policy, 1).item()

                    next_state, reward, done = self.env.step(NChainAction(action))
                    state = next_state
                    steps += 1

                rewards.append(reward)
                branches.append(state.branch)
                trajectory_lengths.append(steps)

        # Compute statistics
        reward_counts = {r: rewards.count(r) for r in set(rewards)}
        reward_freqs = {r: c / num_trajectories for r, c in reward_counts.items()}

        branch_counts = {b: branches.count(b) for b in set(branches)}
        branch_freqs = {b: c / num_trajectories for b, c in branch_counts.items()}

        return {
            "reward_counts": reward_counts,
            "reward_frequencies": reward_freqs,
            "branch_counts": branch_counts,
            "branch_frequencies": branch_freqs,
            "average_trajectory_length": np.mean(trajectory_lengths),
        }
