from typing import Dict, List

import torch
import torch.nn as nn

from gridworld import Action

from .config import GFlowNetConfig, GFlowOutput
from .env import Trajectory


class FlowNetwork(nn.Module):
    """Main network that predicts flows F(s,a)"""

    def __init__(self, config: GFlowNetConfig):
        super().__init__()

        # Initialize with small positive values
        self.log_Z = nn.Parameter(torch.tensor(0.0))  # Log partition function

        # Neural network to predict flows
        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict log flows for all actions from a state
        Returns:
            logits: torch.Tensor of shape [batch_size, action_dim]
        """
        logits = self.network(state)
        return logits + self.log_Z  # Add in log space


class GFlowNet(nn.Module):
    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        self.config = config
        self.flow_network = FlowNetwork(config)

        # Create optimizer but allow configuration
        self.setup_optimizer(
            optimizer_type="adam",
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # self.optimizer = torch.optim.Adam(
        #     self.flow_network.parameters(),
        #     lr=config.learning_rate,
        #     weight_decay=config.weight_decay,   # L2 regularization
        # )

    def setup_optimizer(
        self,
        optimizer_type: str = "adam",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        """Configure the optimizer with flexibility"""
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Create a learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

    def forward(self, state: torch.Tensor) -> GFlowOutput:
        """
        Forward pass to get flows and policy
        Args:
            state: torch.Tensor of shape [batch_size, state_dim]
        """
        # Get log flows (logits)
        logits = self.flow_network(state)

        # Convert to actual flows (always positive)
        flows = torch.exp(logits)

        # Calculate total flow through state
        state_flow = flows.sum(dim=-1, keepdim=True)

        return GFlowOutput(
            flows=flows,
            state_flow=state_flow,
            logits=logits,
            log_Z=self.flow_network.log_Z,
        )

    def compute_trajectory_balance_loss(
        self, trajectory: Trajectory, epsilon: float = 1e-6
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=trajectory.states[0].device)
        cumulative_logprob = torch.tensor(0.0, device=trajectory.states[0].device)

        # Note: we have len(states) = len(action_indices) + 1 because the final state has no action
        for t in range(
            len(trajectory.states) - 1
        ):  # Only iterate up to the second-to-last state
            # Get current state and properly shape action index
            state = trajectory.states[t]
            action_idx = trajectory.action_indices[t].view(1, 1)  # Shape: [1, 1]

            # Forward pass with batch dimension
            output = self.forward(state.unsqueeze(0))  # Add batch dimension
            logits = output.logits  # Shape: [1, action_dim]

            # Get action log probability
            action_logprob = logits.gather(1, action_idx)  # Shape: [1, 1]
            log_Z = torch.logsumexp(logits, dim=-1, keepdim=True)  # Shape: [1, 1]
            step_logprob = action_logprob - log_Z  # Shape: [1, 1]

            cumulative_logprob = cumulative_logprob + step_logprob.squeeze()

        # After accumulating all probabilities, compute loss if this was a complete trajectory
        if trajectory.done:
            target_logprob = torch.log(trajectory.rewards[-1] + epsilon)
            loss = (cumulative_logprob - target_logprob) ** 2
            total_loss = total_loss + loss

        return total_loss.mean()

    def compute_flow_matching_loss(
        self, trajectory: Trajectory, epsilon: float = 1e-6
    ) -> torch.Tensor:
        losses = []

        for t in range(len(trajectory.states) - 1):
            # Add batch dimension and get flows
            curr_state = trajectory.states[t].unsqueeze(0)  # Shape: [1, state_dim]
            next_state = trajectory.states[t + 1].unsqueeze(0)  # Shape: [1, state_dim]

            curr_output = self.forward(curr_state)  # Get flows for current state
            next_output = self.forward(next_state)  # Get flows for next state

            # Properly shape action index for gathering
            # Shape: [1, 1] to match [batch_size, 1] for gathering from [batch_size, action_dim]
            action_idx = trajectory.action_indices[t].view(1, 1)

            # Gather current flow
            curr_flow = curr_output.flows.gather(1, action_idx)  # Shape: [1, 1]

            if t == len(trajectory.states) - 2 and trajectory.done:
                # For terminal states, match against reward
                target_flow = trajectory.rewards[-1].view_as(curr_flow)
            else:
                # For non-terminal states, match against next state flow
                target_flow = next_output.state_flow  # Already has shape [1, 1]

            # Compute loss using log space for numerical stability
            loss = (
                torch.log(curr_flow + epsilon) - torch.log(target_flow + epsilon)
            ) ** 2
            losses.append(loss)

        return torch.mean(torch.stack(losses))

    def compute_regularization_loss(self, output: GFlowOutput) -> torch.Tensor:
        # Encourage smooth flow distributions
        flow_entropy = -(output.flows * torch.log(output.flows + 1e-6)).sum(-1).mean()
        return -self.config.flow_entropy_coef * flow_entropy

    def update(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        flow_losses = []
        balance_losses = []
        reg_losses = []

        self.optimizer.zero_grad()

        for trajectory in trajectories:
            # Compute the individual losses
            flow_loss = self.compute_flow_matching_loss(trajectory)
            balance_loss = self.compute_trajectory_balance_loss(trajectory)
            reg_loss = self.compute_regularization_loss(
                self.forward(trajectory.states[0].unsqueeze(0))
            )

            # Store the losses
            flow_losses.append(flow_loss)
            balance_losses.append(balance_loss)
            reg_losses.append(reg_loss)

        # Combine all losses with their respective weights
        total_flow_loss = torch.stack(flow_losses).mean()
        total_balance_loss = torch.stack(balance_losses).mean()
        total_reg_loss = torch.stack(reg_losses).mean()

        # Weighted sum of losses
        total_loss = (
            1.0 * total_flow_loss
            + 0.1 * total_balance_loss  # Lower weight since this is supplementary
            + 0.01 * total_reg_loss  # Very small weight for regularization
        )

        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Update learning rate
        self.scheduler.step(total_loss)

        return {
            "total_loss": total_loss.item(),
            "flow_loss": total_flow_loss.item(),
            "balance_loss": total_balance_loss.item(),
            "reg_loss": total_reg_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
        temperature: float = 1.0,
    ) -> Action:
        if torch.rand(1) < epsilon:
            return Action(torch.randint(self.config.action_dim, (1,)).item())

        with torch.no_grad():
            output = self.forward(state.unsqueeze(0))
            # Temperature scaled probabilities
            flows = output.flows / temperature
            probs = flows / flows.sum(dim=-1, keepdim=True)
            action_idx = torch.multinomial(probs[0], 1).item()
            return Action(action_idx)
