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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict log flows for all actions from a state
        Args:
            state: torch.Tensor of shape [batch_size, state_dim]
        Returns:
            logits: torch.Tensor of shape [batch_size, action_dim]
        """
        logits = self.network(state)
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
        trajectories: List[Trajectory],
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Compute trajectory balance loss for a batch of trajectories"""
        losses = []
        max_length = max(len(traj.states) - 1 for traj in trajectories)
        
        # Process trajectories in batches at each timestep
        for t in range(max_length):
            # Collect states and actions for this timestep from all trajectories
            batch_states = []
            batch_actions = []
            batch_is_valid = []  # Track which trajectories are still active
            
            for traj in trajectories:
                if t < len(traj.states) - 1:
                    batch_states.append(traj.states[t])
                    batch_actions.append(traj.action_indices[t])
                    batch_is_valid.append(True)
                else:
                    # Pad with zeros for shorter trajectories
                    batch_states.append(torch.zeros_like(trajectories[0].states[0]))
                    batch_actions.append(torch.zeros_like(trajectories[0].action_indices[0]))
                    batch_is_valid.append(False)
            
            # Convert to tensors
            states_tensor = torch.stack(batch_states)
            actions_tensor = torch.stack(batch_actions).view(-1, 1)
            valid_mask = torch.tensor(batch_is_valid, device=states_tensor.device)
            
            if valid_mask.sum() == 0:
                continue
                
            # Forward pass with batch
            output = self.forward(states_tensor)
            logits = output.logits
            
            # Compute probabilities
            action_logprob = logits.gather(1, actions_tensor)
            log_Z = torch.logsumexp(logits, dim=-1, keepdim=True)
            step_logprob = action_logprob - log_Z
            
            # Mask out padded trajectories
            step_logprob = step_logprob.squeeze() * valid_mask
            
            # Add to loss for valid trajectories
            for i, traj in enumerate(trajectories):
                if batch_is_valid[i] and traj.done:
                    target_logprob = torch.log(traj.rewards[-1] + epsilon)
                    loss = (step_logprob[i] - target_logprob) ** 2
                    losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    
    def compute_flow_matching_loss(
        self, 
        trajectories: List[Trajectory],
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Compute flow matching loss for a batch of trajectories"""
        losses = []
        
        # Process trajectories in chunks
        chunk_size = self.config.min_batch_size
        
        for start_idx in range(0, len(trajectories), chunk_size):
            chunk = trajectories[start_idx:start_idx + chunk_size]
            max_length = max(len(traj.states) - 1 for traj in chunk)
            
            # Process each timestep for this chunk
            for t in range(max_length):
                # Collect states for this timestep
                curr_states = []
                next_states = []
                curr_actions = []
                is_terminal = []
                rewards = []
                is_valid = []
                
                for traj in chunk:
                    if t < len(traj.states) - 1:
                        curr_states.append(traj.states[t])
                        next_states.append(traj.states[t + 1])
                        curr_actions.append(traj.action_indices[t])
                        is_terminal.append(t == len(traj.states) - 2 and traj.done)
                        rewards.append(traj.rewards[t])
                        is_valid.append(True)
                    else:
                        # Padding
                        curr_states.append(torch.zeros_like(chunk[0].states[0]))
                        next_states.append(torch.zeros_like(chunk[0].states[0]))
                        curr_actions.append(torch.zeros_like(chunk[0].action_indices[0]))
                        is_terminal.append(False)
                        rewards.append(torch.zeros_like(chunk[0].rewards[0]))
                        is_valid.append(False)
                
                # Convert to tensors
                curr_states = torch.stack(curr_states)
                next_states = torch.stack(next_states)
                curr_actions = torch.stack(curr_actions).view(-1, 1)
                is_terminal = torch.tensor(is_terminal, device=curr_states.device)
                rewards = torch.stack(rewards)
                valid_mask = torch.tensor(is_valid, device=curr_states.device)
                
                if valid_mask.sum() == 0:
                    continue
                    
                # Get flows
                curr_output = self.forward(curr_states)
                next_output = self.forward(next_states)
                
                # Get current flow
                curr_flow = curr_output.flows.gather(1, curr_actions)
                
                # Get target flow
                target_flow = torch.where(
                    is_terminal.unsqueeze(1),
                    rewards.unsqueeze(1),
                    next_output.state_flow
                )
                
                # Compute loss
                loss = (torch.log(curr_flow + epsilon) - torch.log(target_flow + epsilon)) ** 2
                loss = loss * valid_mask.unsqueeze(1)
                
                losses.append(loss.mean())
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)


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
        """Update the model using a batch of trajectories"""
        self.optimizer.zero_grad()
        
        # Compute losses using batched computations
        flow_loss = self.compute_flow_matching_loss(trajectories)
        balance_loss = self.compute_trajectory_balance_loss(trajectories)
        
        # Compute regularization loss on a sample batch
        sample_states = torch.stack([traj.states[0] for traj in trajectories])
        reg_loss = self.compute_regularization_loss(self.forward(sample_states))
        
        # Combine losses
        total_loss = (
            1.0 * flow_loss +
            0.1 * balance_loss +
            0.01 * reg_loss
        )
        
        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(total_loss)
        
        return {
            'total_loss': total_loss.item(),
            'flow_loss': flow_loss.item(),
            'balance_loss': balance_loss.item(),
            'reg_loss': reg_loss.item(),
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
