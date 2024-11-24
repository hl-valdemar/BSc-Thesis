from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from n_chain import NChain


class GFlowNet(nn.Module):
    """
    Basic GFlowNet implementation for N-Chain environment.
    """

    def __init__(self, n_states: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        # State embedding
        self.state_embedding = nn.Embedding(n_states, hidden_dim)

        # Forward policy network (PF)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Flow network for F(s)
        self.flow_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute policy and flows.

        Args:
            states: Tensor of state indices [batch_size]

        Returns:
            policy_logits: Unnormalized log probabilities of actions [batch_size, n_actions]
            log_flows: Log of state flows [batch_size, 1]
        """
        x: torch.Tensor = self.state_embedding(states)  # [batch_size, hidden_dim]

        # Compute policy logits
        policy_logits: torch.Tensor = self.policy_net(x)  # [batch_size, n_actions]

        # Compute log flows (ensure positivity through exponential)
        log_flows: torch.Tensor = self.flow_net(x)  # [batch_size, 1]

        return policy_logits, log_flows

    def compute_trajectory_balance_loss(
        self, trajectories: List[List[Tuple[int, int]]]
    ) -> torch.Tensor:
        """
        Compute trajectory balance loss for a batch of trajectories.

        Loss = (log F(s₀) + Σ log PF(aₜ|sₜ) - log R(x))²

        Args:
            trajectories: List of trajectories, each containing (state, action) pairs
            rewards: Terminal rewards for each trajectory [batch_size]

        Returns:
            loss: Trajectory balance loss
        """
        batch_size = len(trajectories)
        device = next(self.parameters()).device

        # Convert trajectories to tensors
        max_len = max(len(traj) for traj in trajectories)
        state_tensor = torch.full((batch_size, max_len), -1, device=device)
        action_tensor = torch.full((batch_size, max_len), -1, device=device)
        mask = torch.zeros((batch_size, max_len), device=device)

        for i, traj in enumerate(trajectories):
            states, actions = zip(*traj)
            traj_len = len(traj)
            state_tensor[i, :traj_len] = torch.tensor(states, device=device)
            action_tensor[i, :traj_len] = torch.tensor(actions, device=device)
            mask[i, :traj_len] = 1

        # Compute initial state flows
        _, initial_log_flows = self.forward(state_tensor[:, 0])

        # Compute action probabilities for each step
        total_log_probs = torch.zeros(batch_size, device=device)

        for t in range(max_len):
            # Skip if all trajectories are done
            if not mask[:, t].any():
                break

            # Get policy logits for current states
            policy_logits, _ = self.forward(state_tensor[:, t])

            # Compute log probabilities of chosen actions
            log_probs = F.log_softmax(policy_logits, dim=-1)
            step_log_probs = torch.gather(
                log_probs, 1, action_tensor[:, t].unsqueeze(-1)
            ).unsqueeze(-1)

            # Add to total, considering mask
            total_log_probs += step_log_probs * mask[:, t]

        # Compute loss: (log F(s₀) + Σ log PF(aₜ|sₜ) - log R(x))²
        log_rewards = torch.log(
            rewards + 1e-8
        )  # Add small epsilon for numerical stability
        loss = (
            (initial_log_flows.squeeze(-1) + total_log_probs - log_rewards)
            .pow(2)
            .mean()
        )

        return loss

    def sample_trajectory(self, env: NChain) -> Tuple[List[Tuple[int, int]], float]:
        """
        Sample a trajectory using the current policy.

        Args:
            env: N-Chain environment instance

        Returns:
            trajectory: List of (state, action) pairs
            reward: Final reward received
        """
        device = next(self.parameters()).device
        state = env.reset()
        trajectory = []
        done = False
        total_reward = 0

        while not done:
            # Get action probabilities
            state_tensor = torch.tensor([state], device=device)
            policy_logits, _ = self.forward(state_tensor)
            action_probs = F.softmax(policy_logits, dim=-1)

            # Sample action
            action = torch.multinomial(action_probs[0], 1).item()

            # Take step in environment
            next_state, reward, done = env.step(Action(action))

            # Store transition
            trajectory.append((state, action))
            total_reward += reward
            state = next_state

        return trajectory, total_reward
