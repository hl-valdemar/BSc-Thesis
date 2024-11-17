from typing import Dict

import torch
import torch.nn as nn

from gridworld import Action
from .config import GFlowNetConfig, GFlowOutput

class FlowNetwork(nn.Module):
    """Main network that predicts flows F(s,a)"""
    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        self.config = config

        # Neural network to predict flows
        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict log flows for all actions from a state
        Returns:
            logits: torch.Tensor of shape [batch_size, action_dim]
        """
        return self.network(state)

class GFlowNet(nn.Module):
    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        self.config = config
        self.flow_network = FlowNetwork(config)
        self.optimizer = torch.optim.Adam(
            self.flow_network.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5   # L2 regularization
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
            logits=logits
        )

    def compute_flow_matching_loss(self, 
                             states: torch.Tensor,
                             next_states: torch.Tensor,
                             actions: torch.Tensor,
                             rewards: torch.Tensor,
                             dones: torch.Tensor,
                             epsilon: float = 1e-6) -> torch.Tensor:
        """
        Compute flow matching loss with proper terminal state handling
        Args:
            states: [batch_size, state_dim]
            next_states: [batch_size, state_dim]
            actions: [batch_size]
            rewards: [batch_size]
            dones: [batch_size] boolean tensor indicating terminal states
            epsilon: small constant for numerical stability
        """
        # Get flows
        output_current = self.forward(states)
        output_next = self.forward(next_states)

        # Get current action flows
        action_indices = actions.unsqueeze(-1)
        current_flows = torch.gather(output_current.flows, 1, action_indices)

        # Compute log flows
        log_current_flows = torch.log(current_flows + epsilon)

        # For non-terminal states: match incoming flow with outgoing flow
        # For terminal states: match incoming flow with reward
        non_terminal_mask = -dones

        # Initialize loss tensors
        terminal_loss = torch.zeros_like(rewards)
        non_terminal_loss = torch.zeros_like(rewards)

        # Terminal states: flow should match reward
        if dones.any():
            terminal_states_flows = log_current_flows[dones]
            terminal_rewards = torch.log(rewards[dones].unsqueeze(-1) + epsilon)
            terminal_loss[dones] = (terminal_states_flows - terminal_rewards) ** 2

        # Non-terminal states: flow should match sum of outgoing flows
        if non_terminal_mask.any():
            non_terminal_flows = log_current_flows[non_terminal_mask]
            next_state_outflow = torch.log(output_next.state_flow[non_terminal_mask] + epsilon)
            non_terminal_loss[non_terminal_mask] = (non_terminal_flows - next_state_outflow) ** 2

        # Combine losses
        loss = terminal_loss.mean() + non_terminal_loss.mean()

        return loss

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update network using flow matching loss"""
        # Compute loss with terminal state handling
        loss = self.compute_flow_matching_loss(
            states=batch["states"],
            next_states=batch["next_states"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            dones=batch["dones"],
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "flow_matching_loss": loss.item(),
            "terminal_ratio": batch["dones"].float().mean().item() # Log terminal state ratio
        }


    # def compute_flow_matching_loss(self, states, next_states, actions, rewards, epsilon=1e-6):
    #     """Compute flow matching loss from equation 12 in the paper"""
    #     # Get flows
    #     output_current = self.forward(states)
    #     output_next = self.forward(next_states)
    #
    #     # For each state s', sum all incoming flows F(s,a) where T(s,a)=s'
    #     incoming_flows = output_current.flows.gather(1, actions.unsqueeze(-1))
    #
    #     # Total outgoing flow is sum of F(s',a') plus reward R(s') if terminal
    #     outgoing_flows = output_next.state_flow + rewards.unsqueeze(-1)
    #
    #     # Compute loss in log space for numerical stability
    #     log_incoming = torch.log(epsilon + incoming_flows)
    #     log_outgoing = torch.log(epsilon + outgoing_flows)
    #
    #     # Flow matching objective from equation 12 in paper
    #     loss = torch.mean((log_incoming - log_outgoing)**2)
    #
    #     return loss

    # def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    #     """Update network using flow matching loss"""
    #     # Compute loss
    #     loss = self.compute_flow_matching_loss(
    #         states=batch['states'],
    #         next_states=batch['next_states'],
    #         actions=batch['actions'],
    #         rewards=batch['rewards']
    #     )
    #
    #     # Optimize
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     return {'flow_matching_loss': loss.item()}

    # def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> Action:
    #     """
    #     Get action using the flow-based policy
    #     Args:
    #         state: torch.Tensor of shape [state_dim]
    #         epsilon: probability of random action
    #     """
    #     if torch.rand(1) < epsilon:
    #         return Action(torch.randint(self.config.action_dim, (1,)).item())
    #
    #     with torch.no_grad():
    #         output = self.forward(state.unsqueeze(0))
    #         # Policy π(a|s) = F(s,a)/F(s)
    #         probs = output.flows / output.state_flow
    #         action_idx = torch.multinomial(probs[0], 1).item()
    #         return Action(action_idx)

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0, temperature: float = 1.0) -> Action:
        """
        Get action using the flow-based policy
        Args:
            state: torch.Tensor of shape [state_dim]
            epsilon: probability of random action
            temperature: value by which to scale the flows

        Returns: an Action
        """
        if torch.rand(1) < epsilon:
            return Action(torch.randint(self.config.action_dim, (1,)).item())
        
        with torch.no_grad():
            output = self.forward(state.unsqueeze(0))
            # Add temperature scaling
            flows = output.flows / temperature
            probs = flows / flows.sum(dim=-1, keepdim=True)
            action_idx = torch.multinomial(probs[0], 1).item()
            return Action(action_idx)
