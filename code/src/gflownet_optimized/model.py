from typing import Dict, List

import numpy as np

import torch
import torch.nn as nn

from gridworld import Action
from .config import GFlowNetConfig, GFlowOutput
from .env import Trajectory

class FlowNetwork(nn.Module):
    """Main network that predicts flows F(s,a)"""
    def __init__(self, config: GFlowNetConfig):
        super().__init__()

        # Wider but shallower network with batch optimization (compared to the unoptimized implementation)
        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim * 2),
            nn.BatchNorm1d(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.action_dim),
        )

        # Initialize with small positive values
        self.log_Z = nn.Parameter(torch.tensor(0.0)) # Log partition function

        # Layer normalization for better gradient flow
        self.layer_norm = nn.LayerNorm(config.action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict log flows for all actions from a state
        Args:
            state: torch.Tensor of shape [batch_size, state_dim]
        Returns:
            logits: torch.Tensor of shape [batch_size, action_dim]
        """
        # Ensure proper batching
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Forward pass with improved numerical stability
        logits = self.network(state)
        logits = self.layer_norm(logits)
        return logits + self.log_Z # Add in log space

class GFlowNet(nn.Module):
    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        self.config = config
        self.flow_network = FlowNetwork(config)

        # Add gradient clipping threshold
        self.grad_clip_val = 1.0

        # Create optimizer but allow configuration
        self.setup_optimizer(
            optimizer_type='adam',
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def setup_optimizer(
        self, 
        optimizer_type: str = 'adam',
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        """Configure the optimizer with flexibility"""
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Create a learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

    def forward(self, state: torch.Tensor) -> GFlowOutput:
        """
        Forward pass to get flows and policy
        Args:
            state: torch.Tensor of shape [batch_size, state_dim]
        """
        # Ensure proper batching
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Get log flows (logits)
        logits = self.flow_network(state)
        
        # More numerically stable flow computation (compared to unoptimized version)
        flows = torch.exp(logits - logits.max(dim=-1, keepdim=True)[0])
        
        # Calculate total flow through state
        state_flow = flows.sum(dim=-1, keepdim=True)

        return GFlowOutput(
            flows=flows,
            state_flow=state_flow,
            logits=logits,
            log_Z=self.flow_network.log_Z,
        )

    def compute_trajectory_balance_loss(
        self, 
        trajectory: Trajectory, 
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=trajectory.states[0].device)
        cumulative_logprob = torch.tensor(0.0, device=trajectory.states[0].device)
        
        # Note: we have len(states) = len(action_indices) + 1 because the final state has no action
        for t in range(len(trajectory.states) - 1):  # Only iterate up to the second-to-last state
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
        self, 
        trajectory: Trajectory, 
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        losses = []

        # Process trajectory in chunks for better memory efficiency
        chunk_size = 32
        for t in range(0, len(trajectory.states) - 1, chunk_size):
            chunk_end = min(t + chunk_size, len(trajectory.states) - 1)

            # Get current and next states for chunk
            curr_states = trajectory.states[t:chunk_end]
            next_states = trajectory.states[t + 1:chunk_end + 1]
            curr_actions = trajectory.action_indices[t:chunk_end]

            # Batch process states
            curr_output = self.forward(torch.stack(curr_states))
            next_output = self.forward(torch.stack(next_states))

            # Compute loss for chunk
            curr_flow = curr_output.flows.gather(1, torch.stack(curr_actions).unsqueeze(1))

            # Handle terminal states in chunk
            is_terminal = torch.tensor([
                t == len(trajectory.states) - 2 and trajectory.done
                for t in range(chunk_end - t)
            ], device=curr_flow.device)

            target_flow = torch.where(
                is_terminal.unsqueeze(1),
                torch.stack(trajectory.rewards[t:chunk_end]).unsqueeze(1),
                next_output.state_flow,
            )

            # Compute loss using log space for numerical stability
            loss = (torch.log(curr_flow + epsilon) - torch.log(target_flow + epsilon)) ** 2
            losses.append(loss)

        return torch.cat(losses).mean()


    def compute_regularization_loss(
        self, 
        output: GFlowOutput
    ) -> torch.Tensor:
        # Encourage smooth flow distributions
        flow_entropy = -(
            output.flows * 
            torch.log(output.flows + 1e-6)
        ).sum(-1).mean()
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
            1.0 * total_flow_loss +
            0.1 * total_balance_loss + # Lower weight since this is supplementary
            0.01 * total_reg_loss     # Very small weight for regularization
        )
        
        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Update learning rate
        self.scheduler.step(total_loss)
        
        return {
            'total_loss': total_loss.item(),
            'flow_loss': total_flow_loss.item(),
            'balance_loss': total_balance_loss.item(),
            'reg_loss': total_reg_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
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
