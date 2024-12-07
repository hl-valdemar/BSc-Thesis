from dataclasses import dataclass
from typing import List

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
        log_Z: Log of partition function (total flow)
    """

    logits_pf: Tensor
    logits_pb: Tensor
    log_Z: Tensor


class GFlowNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
    ) -> None:
        """
        Args:
            state_dim: Dimension of state space
            num_actions: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Encoder processes concatenated state features
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )

        # Forward and backward policy heads
        self.p_forward = nn.Linear(hidden_dim, num_actions)
        self.p_backward = nn.Linear(hidden_dim, num_actions)

        # Log partition function (total flow)
        self.log_Z = nn.Parameter(torch.tensor(0.0))

    def _check_input_shapes(self, states: Tensor):
        """
        Validate input tensor shapes.

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
        Process batch of states through network.

        Args:
            states: Tensor of shape [batch_size, state_dim]

        Returns:
            GFlowNetOutput
        """
        self._check_input_shapes(states)

        features: Tensor = self.encoder.forward(
            states
        )  # Shape: [batch_size, hidden_dim]

        logits_pf = self.p_forward.forward(features)  # Shape: [batch_size, num_actions]
        logits_pb = self.p_backward.forward(
            features
        )  # Shape: [batch_size, num_actions]

        # Expand log_Z to batch dimension
        log_Z = self.log_Z.expand(states.shape[0])

        return GFlowNetOutput(
            logits_pf=logits_pf,
            logits_pb=logits_pb,
            log_Z=log_Z,
        )

    def get_forward_policy(
        self,
        states: Tensor,
        forward_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            states: States to get forward policy for, shape [batch_size, state_dim]
            forward_mask: Mask of valid actions for state [batch_size, num_actions]

        Returns:
            Tensor: Forward policy probabilities, shape [batch_size, num_actions]
        """
        output = self.forward(states)
        logits = output.logits_pf  # Shape: [batch_size, num_actions]
        # print(f"Raw logits: {output.logits_pf}")  # Check pre-mask logits

        # Mask pre-softmax to preserve normalization
        logits = logits.masked_fill(~forward_mask, float("-inf"))
        # print(f"Masked logits: {logits}")  # Check post-mask logits

        # probs = F.softmax(logits, dim=-1)
        # print(f"Probabilities: {probs}")  # Check final probabilities

        # Return proper probabilities
        return F.softmax(logits, dim=-1)

    def get_backward_policy(
        self,
        states: Tensor,
        backward_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            states: States to get backward policy for, shape [batch_size, state_dim]
            backward_mask: Mask of valid actions from previous state to this state [batch_size, num_actions]

        Returns:
            Tensor: Backward policy probabilities, shape [batch_size, num_actions]
        """
        output = self.forward(states)
        logits = output.logits_pb  # Shape: [batch_size, num_actions]

        # Mask pre-softmax to preserve normalization
        logits = logits.masked_fill(~backward_mask, float("-inf"))

        # Return proper probabilities
        return F.softmax(logits, dim=-1)

    def compute_trajectory_balance_loss(
        self,
        states: List[Tensor],
        actions: List[int],
        rewards: List[float],
        terminated: bool,
        forward_masks: List[Tensor],
        backward_masks: List[Tensor],
        max_reward: float,
    ) -> Tensor:
        """
        Compute trajectory balance loss for a single trajectory.

        L = (log Z + sum log P_F - log R - sum log P_B)^2

        Args:
            states: List of states, each with shape [state_dim]
            actions: List of actions taken
            rewards: List of rewards received
            terminated: Whether trajectory reached terminal
            forward_masks: List of forward masks, each with shape [1, num_actions]
            backward_masks: List of backward masks, each with shape [1, num_actions]
            max_reward: maximum possible reward (for normalization)

        Returns:
            Tensor: The trajectory balance loss
        """
        if not terminated:
            return torch.tensor(0.0, requires_grad=True)

        # Trajectory Processing:
        # - Each trajectory might have a different length, making it awkward to batch.
        # - Instead, process each state in the trajectory individually.

        # Get network outputs for all states
        outputs = [
            self.forward(s.unsqueeze(0)) for s in states
        ]  # State shape: [batch_dim = 1, state_dim]

        # Forward log-probabilities
        log_pf = []
        for t, (output, action, mask) in enumerate(
            zip(outputs, actions, forward_masks)
        ):
            logits = output.logits_pf.squeeze(0)  # Remove batch dimension
            mask = mask.squeeze(0)  # Also squeeze the mask

            logits = logits.masked_fill(~mask, float("-inf"))
            log_pf.append(F.log_softmax(logits, dim=-1)[action])

        # Backward log-probabilities
        log_pb = []
        for t in range(len(states) - 1):
            output = outputs[t + 1]
            logits = output.logits_pb.squeeze(0)  # Remove batch dimension

            # Use the mask from the previous state
            mask = backward_masks[t + 1].squeeze(0)

            # print(f"\nBackward step {t}:")
            # print(f"  Raw logits: {logits}")
            # print(f"  Mask: {mask}")

            logits = logits.masked_fill(~mask, float("-inf"))

            # print(f"  Masked logits: {logits}")

            prev_action = actions[t]
            prob = F.log_softmax(logits, dim=-1)[prev_action]
            # print(f"  Action {prev_action}: log_prob = {prob}")

            log_pb.append(prob)

        # print(f"\nlog_pb: {log_pb}")

        # Compute loss components
        log_Z = outputs[0].log_Z[0]  # From initial state
        sum_log_pf = torch.stack(log_pf).sum()
        sum_log_pb = torch.stack(log_pb).sum()

        # Final reward is at terminal state
        terminal_reward = rewards[-1] + 1e-10
        # norm_reward = terminal_reward / max_reward
        log_R = torch.tensor(terminal_reward).log()

        # Trajectory balance loss
        loss = (log_Z + sum_log_pf - log_R - sum_log_pb).pow(
            2
        )  # NOTE: Maybe take mean here?

        # print("\nLoss components:")
        # print(f"  log_Z: {log_Z}")
        # print(f"  sum_log_pf: {sum_log_pf}")
        # print(f"  log_R: {log_R}")
        # print(f"  sum_log_pb: {sum_log_pb}")
        # print(f"  total_loss: {loss}")

        return loss
