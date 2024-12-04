import random
from collections import defaultdict, deque
from typing import Deque, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from nchain_v2 import NChainEnv, Trajectory

from .metrics import MetricsTracker, TrainingMetrics
from .model import GFlowNet


class GFlowNetTrainer:
    def __init__(
        self,
        env: NChainEnv,
        model: GFlowNet,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 32,
        metrics_window: int = 100,
    ):
        self.env = env
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Initialize replay buffer
        self.replay_buffer: Deque[Trajectory] = deque(maxlen=buffer_size)

        self.Z_estimate = env.base_reward
        self.Z_momentum = 0.99

        # Training tracking
        self.steps_done = 0
        self.eps_threshold = 0.9
        self.metrics = MetricsTracker(window_size=metrics_window)

    def collect_trajectory(self) -> Trajectory:
        """Collect a single trajectory using current policy."""
        state = self.env.reset()
        done = False

        states = [state]
        actions = []
        rewards = []

        while not done:
            valid_actions = self.env.get_valid_actions(state)
            state_tensor = state.to_tensor().unsqueeze(0)

            # Create action mask for valid actions
            action_mask = torch.zeros(
                1, self.model.num_actions
            )  # Shape: [1, self.model.num_actions]
            action_mask[0, valid_actions] = 1.0

            # Get masked action probabilities
            action_probs = self.model.get_forward_policy(state_tensor)
            action_probs = action_probs * action_mask
            action_probs = action_probs / action_probs.sum()  # Renormalize
            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, done = self.env.step(action)

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        return Trajectory(states, actions, rewards, done)

    def create_action_masks(
        self, valid_actions_list: List[List[int]], num_actions: int
    ) -> Tensor:
        """Create binary mask tensor for valid actions.

        Args:
            valid_actions_list: List of lists containing valid actions for each state
            num_actions: Total number of possible actions

        Returns:
            Tensor: Binary mask of shape [batch_size, num_actions]
        """
        batch_size = len(valid_actions_list)
        masks = torch.zeros(batch_size, num_actions)
        for i, valid_actions in enumerate(valid_actions_list):
            masks[i, valid_actions] = 1.0
        return masks

    def compute_masked_log_probs(
        self,
        logits: Tensor,  # Shape [batch_size, num_actions]
        valid_actions: List[List[int]],
    ) -> Tensor:
        """Compute log probabilities with valid action masking.

        Args:
            logits: Raw logits from model
            valid_actions: List of valid actions for each state

        Returns:
            Tensor: Masked log probabilities
        """
        # Create action masks
        action_masks = self.create_action_masks(valid_actions, logits.shape[1])

        # Apply mask in log space
        masked_logits = logits + (action_masks + 1e-8).log()

        # Compute log_softmax
        log_probs = F.log_softmax(masked_logits, dim=-1)

        return log_probs

    def compute_trajectory_balance_loss(self, trajectory: Trajectory) -> Tensor:
        """Compute trajectory balance loss following GFlowNet principles."""
        states, actions, rewards = trajectory.to_tensors()
        valid_actions = [self.env.get_valid_actions(s) for s in trajectory.states]

        outputs = self.model.forward(states)

        # Get forward probabilities for taken actions
        pf_logprobs = self.compute_masked_log_probs(outputs.logits_pf, valid_actions)
        pf_logprobs = torch.gather(pf_logprobs, 1, actions.unsqueeze(1)).squeeze(1)

        # Get backward probabilities for taken actions
        pb_logprobs = self.compute_masked_log_probs(outputs.logits_pb, valid_actions)
        pb_logprobs = torch.gather(pb_logprobs, 1, actions.unsqueeze(1)).squeeze(1)

        # Forward term (in log space):
        # log(Z) + sum(log(PF(s_t+1|s_t)))
        log_forward = (
            torch.log(torch.tensor(self.Z_estimate) + 1e-8) + pf_logprobs.sum()
        )

        # Backward term (in log space):
        # log(R(x)) + sum(log(PB(s_t|s_t+1)))
        terminal_reward = rewards[-1]
        log_backward = torch.log(terminal_reward + 1e-8) + pb_logprobs.sum()

        # Trajectory balance loss
        balance_loss = (log_forward - log_backward).pow(2)

        return balance_loss

    def compute_trajectory_metrics(self, trajectory: Trajectory) -> TrainingMetrics:
        """Compute metrics including path length analysis."""
        states, actions, rewards = trajectory.to_tensors()
        valid_actions = [self.env.get_valid_actions(s) for s in trajectory.states]
        T = len(trajectory.actions)

        with torch.no_grad():
            outputs = self.model.forward(states)

            # Existing flow consistency metrics
            terminal_flow = outputs.log_state_flow[-1].exp()
            terminal_reward = rewards[-1]
            flow_value_error = (terminal_flow - terminal_reward) ** 2

            # Path length analysis
            path_length = T
            position = trajectory.states[-1].position
            optimal_length = position  # Minimum path length to reach position
            path_efficiency = optimal_length / path_length if path_length > 0 else 0.0

            # Flow distribution analysis by path length
            terminal_flows = []
            terminal_rewards = []

            # Analyze last N trajectories to get path length distribution
            recent_trajectories = list(self.replay_buffer)[-self.metrics.window_size :]
            for traj in recent_trajectories:
                if traj.done:  # Only consider completed trajectories
                    length = len(traj.actions)
                    pos = traj.states[-1].position
                    with torch.no_grad():
                        final_flow = self.model.forward(
                            traj.states[-1].to_tensor().unsqueeze(0)
                        )
                        terminal_flows.append(
                            (length, final_flow.log_state_flow[0].exp().item())
                        )
                        terminal_rewards.append((length, traj.rewards[-1]))

            # Group flows and rewards by path length
            length_flows = defaultdict(list)
            length_rewards = defaultdict(list)
            for length, flow in terminal_flows:
                length_flows[length].append(flow)
            for length, reward in terminal_rewards:
                length_rewards[length].append(reward)

            # Compute average flows and rewards per path length
            avg_flows_by_length = {
                length: sum(flows) / len(flows)
                for length, flows in length_flows.items()
            }
            avg_rewards_by_length = {
                length: sum(rewards) / len(rewards)
                for length, rewards in length_rewards.items()
            }

            return TrainingMetrics(
                partition_estimate=self.Z_estimate,
                loss=0.0,  # Updated during training step
                episode_length=path_length,
                episode_return=sum(trajectory.rewards),
                # state_visits=self.get_state_visits(trajectory),
                successful_episodes=trajectory.done,
                flow_values=dict(enumerate(outputs.log_state_flow.exp().tolist())),
                flow_value_estimation_error=flow_value_error.item(),
                # New path-specific metrics
                path_length=path_length,
                optimal_length=optimal_length,
                path_efficiency=path_efficiency,
                flows_by_path_length=avg_flows_by_length,
                rewards_by_path_length=avg_rewards_by_length,
                # Distribution metrics
                path_length_distribution={
                    length: len(flows) / len(terminal_flows)
                    for length, flows in length_flows.items()
                },
            )

    def update_Z_estimate(self, trajectory: Trajectory):
        """Update partition function estimate using terminal rewards."""
        terminal_reward = trajectory.rewards[-1]
        # Use rewards rather than flows for Z estimation
        current_Z = terminal_reward

        # Exponential moving average update
        self.Z_estimate = (
            self.Z_momentum * self.Z_estimate + (1 - self.Z_momentum) * current_Z
        )

    def train_step(self, total_steps: int) -> float:
        """Perform single training step."""
        trajectory = self.collect_trajectory()
        self.replay_buffer.append(trajectory)

        # Update Z estimate before loss computation
        self.update_Z_estimate(trajectory)

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)

        total_loss = 0
        self.optimizer.zero_grad()

        for traj in batch:
            # states, actions, rewards = traj.to_tensors()
            # valid_actions = [self.env.get_valid_actions(s) for s in traj.states]
            loss = self.compute_trajectory_balance_loss(traj)
            total_loss += loss

        avg_loss = total_loss / self.batch_size
        avg_loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        self.eps_threshold = max(0.4, 0.9 - 0.89 * self.steps_done / total_steps)

        loss_value = avg_loss.item()

        # Compute and store metrics
        metrics = self.compute_trajectory_metrics(trajectory)
        metrics.loss = loss_value
        self.metrics.add_metrics(metrics)

        return loss_value

    def train(self, num_steps: int) -> List[float]:
        """Train for specified number of steps."""
        losses = []
        for step in range(num_steps):
            loss = self.train_step(num_steps)
            losses.append(loss)

            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{num_steps}")
                self.metrics.print_summary()

        return losses
