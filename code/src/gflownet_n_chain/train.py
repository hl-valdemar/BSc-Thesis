import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from n_chain import NChainEnv, NChainState

from .metrics import MetricsTracker, TrainingMetrics
from .model import GFlowNet


@dataclass
class Trajectory:
    """
    Container for a single trajectory through the environment.

    Attributes:
        states: List of states visited
        actions: List of actions taken
        rewards: List of rewards received
        done: Whether trajectory ended at terminal state
    """

    states: List[NChainState]
    actions: List[int]
    rewards: List[float]
    done: bool

    def to_tensors(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert trajectory to tensor format.

        Returns:
            Tuple containing:
                - states_tensor: Shape [T, state_dim]
                - actions_tensor: Shape [T]
                - rewards_tensor: Shape [T]
        """
        states_tensor = torch.stack([s.to_tensor() for s in self.states])
        actions_tensor = torch.tensor(self.actions)
        rewards_tensor = torch.tensor(self.rewards)
        return states_tensor, actions_tensor, rewards_tensor


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
        self.replay_buffer = deque(maxlen=buffer_size)

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
            action_mask = torch.zeros(1, self.model.num_actions)
            action_mask[0, valid_actions] = 1.0

            # Get masked action probabilities
            action_probs = self.model.get_forward_policy(state_tensor)
            action_probs = action_probs * action_mask
            action_probs = action_probs / action_probs.sum()  # Renormalize

            # ε-greedy exploration
            if random.random() < self.eps_threshold:
                action = random.choice(valid_actions)
            else:
                action = action_probs.argmax().item()

            next_state, reward, done = self.env.step(action)

            states.append(next_state)
            actions.append(action)

            # Scale the reward according the number of actions taken
            # This should lead to policies that sample shorter paths with higher frequencies
            reward = self.env.n * reward / (2 * len(actions))
            # alpha = 3
            # beta = 1.1
            # reward = (
            #     (self.env.n / alpha)
            #     * reward
            #     * np.exp(len(actions)) ** (-beta / self.env.n)
            # )
            rewards.append(reward)

            state = next_state

        return Trajectory(states, actions, rewards, done)

    # def compute_returns(self, rewards: Tensor) -> Tensor:
    #     """
    #     Compute discounted returns.
    #
    #     Args:
    #         rewards: Rewards collected for a single trajectory
    #
    #     Returns:
    #         Tensor: The discounted returns
    #     """
    #     returns = torch.zeros_like(rewards)
    #     running_sum = 0
    #     for t in reversed(range(len(rewards))):
    #         running_sum = rewards[t] + self.gamma * running_sum
    #         returns[t] = running_sum
    #     return returns

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

    def compute_subtraj_balance_loss(
        self,
        states: Tensor,  # Shape [T, state_dim]
        actions: Tensor,  # Shape [T]
        rewards: Tensor,  # Shape [T]
        valid_actions: List[List[int]],
    ) -> Tensor:
        """Compute SubTrajectory Balance loss for a single trajectory."""
        T = states.shape[0]

        # Get model outputs
        outputs = self.model.forward(states)

        # Compute masked log probabilities
        pf_logprobs = self.compute_masked_log_probs(outputs.logits_pf, valid_actions)
        pb_logprobs = self.compute_masked_log_probs(outputs.logits_pb, valid_actions)

        # Gather chosen action probabilities
        pf_logprobs = torch.gather(pf_logprobs, 1, actions.unsqueeze(1)).squeeze(1)
        pb_logprobs = torch.gather(pb_logprobs, 1, actions.unsqueeze(1)).squeeze(1)

        # SubTrajectory Balance loss
        flow_matching_loss = 0
        for t in range(T - 1):
            left_term = outputs.log_state_flow[t] + pf_logprobs[t]
            right_term = outputs.log_state_flow[t + 1] + pb_logprobs[t]

            # Squared error in log space
            step_loss = (left_term - right_term) ** 2
            flow_matching_loss = flow_matching_loss + step_loss

        # Normalization loss for terminal states
        # terminal_flows = torch.exp(outputs.log_state_flow[-1])  # Flow at terminal state
        # normalization_loss = (terminal_flows - rewards[-1]) ** 2

        terminal_flows = torch.exp(outputs.log_state_flow)
        Z = terminal_flows.sum()
        normalized_flows = terminal_flows / Z

        # Compare normalized flows to normalized rewards
        terminal_rewards = rewards[-1]
        normalization_loss = (Z - 1.0) ** 2 + (
            normalized_flows[-1] - terminal_rewards
        ) ** 2

        # Combine losses with appropirate weighting
        lambda_norm = 0.9  # Hyperparameter to control normalization strength
        total_loss = flow_matching_loss + lambda_norm * normalization_loss

        return total_loss.mean()

    def compute_trajectory_metrics(self, trajectory: Trajectory) -> TrainingMetrics:
        """
        Shape information:
        - trajectory.states: [T+1] states (including initial and final)
        - trajectory.actions: [T] actions
        - trajectory.rewards: [T] rewards
        where T is the number of steps taken
        """
        """Compute metrics for a single trajectory.
        
        Args:
            trajectory: Completed trajectory
            
        Returns:
            TrainingMetrics: Computed metrics
        """
        # Convert trajectory to tensors
        states, actions, rewards = trajectory.to_tensors()
        valid_actions = [self.env.get_valid_actions(s) for s in trajectory.states]
        action_masks = self.create_action_masks(valid_actions, self.model.num_actions)

        # Verify tensor shapes
        T = len(trajectory.actions)  # Number of steps
        assert (
            len(trajectory.states) == T + 1
        ), f"Expected {T+1} states, got {len(trajectory.states)}"
        assert (
            len(trajectory.actions) == T
        ), f"Expected {T} actions, got {len(trajectory.actions)}"
        assert (
            len(trajectory.rewards) == T
        ), f"Expected {T} rewards, got {len(trajectory.rewards)}"
        assert (
            states.shape[0] == T + 1
        ), f"Expected states tensor of shape [{T+1}, state_dim], got {states.shape}"

        # Get model outputs
        with torch.no_grad():
            outputs = self.model.forward(states)

            # Get terminal flow and reward
            terminal_flow = outputs.log_state_flow[-1].exp()
            terminal_reward = rewards[-1]

            # Compute value error as deviation from desired normalized flow
            flow_value_error = (terminal_flow - terminal_reward) ** 2

            # Flow normalization error
            terminal_states_mask = torch.tensor(
                [self.env.is_terminal(s) for s in trajectory.states]
            )
            terminal_flows = outputs.log_state_flow[terminal_states_mask].exp()
            Z = terminal_flows.sum()
            flow_normalization_error = (Z - 1.0) ** 2

            # Compute policy entropy
            entropy = self.metrics.compute_policy_entropy(
                outputs.logits_pf, action_masks
            )

            # Get probabilities
            pf_probs = F.softmax(outputs.logits_pf, dim=-1)
            pb_probs = F.softmax(outputs.logits_pb, dim=-1)

            # Convert to numpy for storage
            pf_probs = {i: probs.numpy() for i, probs in enumerate(pf_probs)}
            pb_probs = {i: probs.numpy() for i, probs in enumerate(pb_probs)}

            # Get flow values
            flow_values = {
                i: flow.item()
                for i, flow in enumerate(torch.exp(outputs.log_state_flow))
            }

        # Count state visits
        state_visits = defaultdict(int)
        for state in trajectory.states:
            state_visits[state.position] += 1

        return TrainingMetrics(
            loss=0.0,  # Will be updated during training step
            episode_length=len(trajectory.states) - 1,
            episode_return=sum(trajectory.rewards),
            policy_entropy=entropy,
            state_visits=dict(state_visits),
            successful_episodes=trajectory.done,
            flow_values=flow_values,
            pf_probs=pf_probs,
            pb_probs=pb_probs,
            flow_value_estimation_error=flow_value_error.item(),
            flow_normalization_error=flow_normalization_error.item(),
        )

    def train_step(self) -> float:
        """Perform single training step."""
        trajectory = self.collect_trajectory()
        self.replay_buffer.append(trajectory)

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)

        total_loss = 0
        self.optimizer.zero_grad()

        for traj in batch:
            states, actions, rewards = traj.to_tensors()
            valid_actions = [self.env.get_valid_actions(s) for s in traj.states]

            loss = self.compute_subtraj_balance_loss(
                states, actions, rewards, valid_actions
            )
            total_loss += loss

        avg_loss = total_loss / self.batch_size
        avg_loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        self.eps_threshold = max(0.01, 0.9 - 0.89 * self.steps_done / 10000)

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
            loss = self.train_step()
            losses.append(loss)

            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{num_steps}")
                self.metrics.print_summary()

        return losses
