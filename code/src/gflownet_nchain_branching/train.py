import random
from typing import Deque, List

import torch
from torch.optim import Adam

from nchain_branching.nchain_branching import NChainEnv, NChainTrajectory

from .model import GFlowNet


class GFlowNetTrainer:
    def __init__(
        self,
        env: NChainEnv,
        model: GFlowNet,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        buffer_size: int = 10_000,
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
                1, self.env.num_actions
            )  # Shape: [1, self.env.num_actions]
            action_mask[0, valid_actions] = 1.0

            # Get masked action probabilities
            action_probs = self.model.get_forward_policy(state_tensor)
            action_probs = action_probs * action_mask
            action = torch.multinomial(
                action_probs, 1
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

    def train_step(self) -> float:
        """
        Perform a single training step.

        Returns:
            float: Loss value for this step
        """
        trajectory = self.collect_trajectory()
        self.replay_buffer.append(trajectory)

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)

        total_loss = torch.tensor(0.0)
        self.optimizer.zero_grad()

        for trajectory in batch:
            loss = self.model.compute_trajectory_balance_loss(
                states=trajectory.states,
                actions=trajectory.actions,
                rewards=trajectory.rewards,
                terminated=trajectory.done,
                valid_actions_mask=trajectory.valid_actions_masks,
            )
            total_loss += loss

        avg_loss = total_loss / self.batch_size
        avg_loss.backward()
        self.optimizer.step()

        loss_value = avg_loss.item()
        return loss_value

    def train(self, num_steps: int) -> List[float]:
        losses = []
        for step in range(num_steps):
            loss = self.train_step()
            losses.append(loss)

            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{num_steps}")
                # self.metrics.print_summary()

        return losses
