from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GFlowNetOutput:
    """
    Container for GFlowNet forward pass outputs.

    Attributes:
        logits_pf: Forward policy logits, shape [batch_size, num_actions]
        logits_pb: Backward policy logits, shape [batch_size, num_actions]
        log_state_flow: Log flow through state, shape [batch_size]
    """

    logits_pf: Tensor
    logits_pb: Tensor
    log_state_flow: Tensor


class GFlowNetModel(nn.Module):
    """
    GFlowNet implementation for N-Chain problem.

    The model takes a state and outputs:
    1. Forward policy probabilities (PF)
    2. Backward policy probabilities (PB)
    3. State flow values

    Architecture:
    - Shared MLP backbone
    - Separate heads for PF, PB, and flow
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        """
        Args:
            state_dim: Dimension of state space (n for n-chain)
            num_actions: Number of possible actions (2 for n-chain)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy heads
        self.pf_head = nn.Linear(hidden_dim, num_actions)  # Forward policy
        self.pb_head = nn.Linear(hidden_dim, num_actions)  # Backward policy

        # State flow head (scalar)
        self.flow_head = nn.Linear(hidden_dim, 1)

        # Save dims for shape checking
        self.state_dim = state_dim
        self.num_actions = num_actions

    def _check_input_shapes(self, states: Tensor):
        """Validate input tensor shapes.

        Args:
            states: Shape [batch_size, state_dim]
        """
        assert (
            states.ndim == 2
        ), f"Expected states to have 2 dims, got shape {states.shape}"
        assert (
            states.shape[1] == self.state_dim
        ), f"Expected states to have shape [batch_size, {self.state_dim}], got {states.shape}"

    def forward(self, states: Tensor) -> GFlowNetOutput:
        """
        Forward pass computing policy and flow outputs.

        Args:
            states: Tensor of shape [batch_size, state_dim]

        Returns:
            GFlowNetOutput containing:
                - logits_pf: Forward policy logits [batch_size, num_actions]
                - logits_pb: Backward policy logits [batch_size, num_actions]
                - log_state_flow: Log flow values [batch_size]
        """
        self._check_input_shapes(states)

        # Get shared features
        features = self.backbone(states)  # Shape: [batch_size, hidden_dim]

        # Compute outputs
        logits_pf = self.pf_head(features)  # Shape: [batch_size, num_actions]
        logits_pb = self.pb_head(features)  # Shape: [batch_size, num_actions]
        log_state_flow = self.flow_head(features).squeeze(-1)  # Shape: [batch_size]

        return GFlowNetOutput(
            logits_pf=logits_pf,
            logits_pb=logits_pb,
            log_state_flow=log_state_flow,
        )

    def get_forward_policy(
        self,
        states: Tensor,
        valid_actions: Optional[List[List[int]]] = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Get forward policy probabilities PF(a|s).

        Args:
            states: States to get policy for, shape [batch_size, state_dim]
            valid_actions: List of lists containing valid actions for each state
            temperature: Softmax temperature (higher = more uniform)

        Returns:
            Tensor: Forward policy probs, shape [batch_size, num_actions]
        """
        outputs = self.forward(states)
        logits = outputs.logits_pf / temperature

        # Mask invalid actions if provided
        if valid_actions is not None:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for i, actions in enumerate(valid_actions):
                mask[i, actions] = True
            logits = logits.masked_fill(~mask, float("-inf"))

        return F.softmax(logits, dim=-1)

    def get_backward_policy(self, states: Tensor, temperature: float = 1.0) -> Tensor:
        """
        Get backward policy probabilities PB(s|s').

        Args:
            states: States to get policy for, shape [batch_size, state_dim]
            temperature: Softmax temperature

        Returns:
            Tensor: Backward policy probs, shape [batch_size, num_actions]
        """
        outputs = self.forward(states)
        return F.softmax(outputs.logits_pb / temperature, dim=-1)

    def get_state_flow(self, states: Tensor) -> Tensor:
        """
        Get flow values F(s) for states.

        Args:
            states: States to get flow for, shape [batch_size, state_dim]

        Returns:
            Tensor: Flow values, shape [batch_size]
        """
        outputs = self.forward(states)
        return torch.exp(outputs.log_state_flow)  # Convert from log space
