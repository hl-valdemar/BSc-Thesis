from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFlowNetBase(nn.Module):
    """Base class for GFlowNet implementation."""

    def __init__(self, state_dim: int, hidden_dim: int, n_actions: int):
        """
        Args:
            state_dim: Dimension of state representation
            hidden_dim: Hidden layer dimension
            n_actions: Number of possible actions
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # Core network for state processing
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate network for combined state-action processing
        self.state_action_net = nn.Sequential(
            nn.Linear(hidden_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head (forward)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Flow heads
        self.state_flow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        self.edge_flow_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing policy logits and flows.

        Args:
            state: Current state tensor [batch_size, state_dim]

        Returns:
            policy_logits: Unnormalized action probabilities [batch_size, n_actions]
            state_flow: Flow through state [batch_size, 1]
            edge_flows: Flows between states [batch_size, n_actions]
        """
        # Get state embedding
        h = self.backbone(state)

        # Compute policy logits
        policy_logits = self.policy_head(h)

        # Compute state flow (non-negative through exp)
        state_flow = torch.exp(self.state_flow_head(h))

        # Compute edge flows for all possible actions
        # We'll need state embeddings for all possible next states
        # This is a simplification - in practice, you'd compute this more efficiently
        batch_size = state.shape[0]
        edge_flows = torch.zeros(batch_size, self.n_actions, device=state.device)

        for action in range(self.n_actions):
            next_state_h = self.get_next_state_embedding(h, action)
            edge_input = torch.cat([h, next_state_h], dim=-1)
            edge_flows[:, action] = torch.exp(self.edge_flow_head(edge_input)).squeeze(
                -1
            )

        return policy_logits, state_flow, edge_flows

    def get_next_state_embedding(
        self, state_h: torch.Tensor, action: int
    ) -> torch.Tensor:
        """
        Gets embedding for next state given current state embedding and action.
        """
        raise NotImplementedError

    def compute_flow_matching_loss(
        self,
        state: torch.Tensor,
        policy_logits: torch.Tensor,
        state_flow: torch.Tensor,
        edge_flows: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss based on paper.
        L = (log(∑F(s'→s)) - log(∑F(s→s')))²

        Args:
            state: Current state
            policy_logits: Unnormalized action probabilities
            state_flow: Flow through current state
            edge_flows: Flows between states
            mask: Optional mask for valid actions

        Returns:
            loss: Flow matching loss
        """
        # Apply mask if provided
        if mask is not None:
            edge_flows = edge_flows * mask

        # Compute incoming and outgoing flows
        outgoing_flow = torch.sum(edge_flows, dim=-1, keepdim=True)  # Sum over actions

        # The incoming flow should match the state flow
        loss = (
            (torch.log(state_flow + 1e-8) - torch.log(outgoing_flow + 1e-8))
            .pow(2)
            .mean()
        )

        return loss

    def sample_trajectory(
        self, initial_state: torch.Tensor, max_steps: int = 100
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Sample a trajectory using the current policy.

        Args:
            initial_state: Starting state
            max_steps: Maximum trajectory length

        Returns:
            trajectory: List of (state, action) pairs
        """
        trajectory = []
        current_state = initial_state

        for _ in range(max_steps):
            # Get policy logits
            with torch.no_grad():
                policy_logits, _, _ = self.forward(current_state.unsqueeze(0))
                policy_probs = F.softmax(policy_logits.squeeze(0), dim=-1)

                # Sample action
                action = int(torch.multinomial(policy_probs, 1).item())

                trajectory.append((current_state, action))

                # Get next state (environment-specific)
                current_state = self.get_next_state(current_state, action)

                # Check if terminal
                if self.is_terminal(current_state):
                    break

        return trajectory

    def get_next_state(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """Get next state given current state and action. Must be implemented."""
        raise NotImplementedError

    def is_terminal(self, state: torch.Tensor) -> bool:
        """Check if state is terminal. Must be implemented."""
        raise NotImplementedError
