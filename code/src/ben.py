from typing import List, Tuple, Optional, Dict, Any, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from dataclasses import dataclass

@dataclass
class BENConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    rnn_hidden_dim: int = 64
    num_flows: int = 2
    learning_rate: float = 1e-4
    gamma: float = 0.99

class RecurrentQNetwork(nn.Module):
    def __init__(self, config: BENConfig):
        super().__init__()
        self.config = config
        
        # Input layer
        self.input_layer = nn.Linear(
            config.state_dim + 1,  # state_dim + reward
            config.hidden_dim
        )
        
        # GRU for history encoding
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.rnn_hidden_dim,
            batch_first=True
        )
        
        # Output layer for Q-values
        self.q_layer = nn.Linear(
            config.rnn_hidden_dim,
            config.action_dim
        )
        
    def forward(self, 
                state: torch.Tensor,
                reward: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine state and reward
        x = torch.cat([state, reward.unsqueeze(-1)], dim=-1)
        x = F.relu(self.input_layer(x))
        
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Process through GRU
        out, hidden = self.gru(x, hidden)
        
        # Get Q-values
        q_values = self.q_layer(out)
        
        return q_values, hidden

class NormalizingFlow(nn.Module):
    """Base class for normalizing flows"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform input x and return transformed value z and log determinant
        """
        raise NotImplementedError
        
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform input z and return transformed value x and log determinant
        """
        raise NotImplementedError

class AleatoricNetwork(nn.Module):
    def __init__(self, config: BENConfig):
        super().__init__()
        self.config = config
        
        # Base distribution
        self.base_dist = Normal(0, 1)
        
        # Series of normalizing flows
        self.flows = nn.ModuleList([
            NormalizingFlow(dim=1) for _ in range(config.num_flows)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform from base distribution to target distribution
        Returns transformed sample and log probability
        """
        z = self.base_dist.sample(x.shape)
        log_prob = self.base_dist.log_prob(z)
        
        for flow in self.flows:
            z, ldj = flow(z)
            log_prob = log_prob - ldj
            
        return z, log_prob

class EpistemicNetwork(nn.Module):
    def __init__(self, config: BENConfig):
        super().__init__()
        self.config = config
        
        # Similar structure to AleatoricNetwork but for parameter uncertainty
        self.base_dist = Normal(0, 1)
        self.flows = nn.ModuleList([
            NormalizingFlow(dim=config.hidden_dim) 
            for _ in range(config.num_flows)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.base_dist.sample(x.shape)
        log_prob = self.base_dist.log_prob(z)
        
        for flow in self.flows:
            z, ldj = flow(z)
            log_prob = log_prob - ldj
            
        return z, log_prob

class BEN(nn.Module):
    def __init__(self, config: BENConfig):
        super().__init__()
        self.config = config
        
        # Main components
        self.q_network = RecurrentQNetwork(config)
        self.aleatoric_network = AleatoricNetwork(config)
        self.epistemic_network = EpistemicNetwork(config)
        
        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        self.aleatoric_optimizer = torch.optim.Adam(
            self.aleatoric_network.parameters(),
            lr=config.learning_rate
        )
        self.epistemic_optimizer = torch.optim.Adam(
            self.epistemic_network.parameters(),
            lr=config.learning_rate
        )
        
    def forward(self, 
                state: torch.Tensor,
                reward: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get Q-values
        q_values, hidden = self.q_network(state, reward, hidden)
        
        # Get aleatoric uncertainty
        aleatoric_sample, aleatoric_log_prob = self.aleatoric_network(q_values)
        
        # Get epistemic uncertainty
        epistemic_sample, epistemic_log_prob = self.epistemic_network(q_values)
        
        return q_values, aleatoric_sample, aleatoric_log_prob, epistemic_sample, epistemic_log_prob

    def update(self, 
               batch: Dict[str, torch.Tensor],
               ) -> Dict[str, float]:
        """
        Update networks using MSBBE and ELBO objectives
        Returns dict with loss values
        """
        # Unpack batch
        states = batch['states']
        rewards = batch['rewards']
        next_states = batch['next_states']
        actions = batch['actions']
        
        # Forward pass
        q_values, aleatoric, aleatoric_log_prob, epistemic, epistemic_log_prob = self(states, rewards)
        
        # Calculate MSBBE loss
        target_q = rewards + self.config.gamma * torch.max(
            self(next_states, rewards)[0], dim=-1
        )[0]
        msbbe_loss = F.mse_loss(q_values, target_q)
        
        # Calculate ELBO loss
        elbo_loss = -aleatoric_log_prob.mean() - epistemic_log_prob.mean()
        
        # Update networks
        self.q_optimizer.zero_grad()
        self.aleatoric_optimizer.zero_grad()
        self.epistemic_optimizer.zero_grad()
        
        msbbe_loss.backward()
        elbo_loss.backward()
        
        self.q_optimizer.step()
        self.aleatoric_optimizer.step() 
        self.epistemic_optimizer.step()
        
        return {
            'msbbe_loss': msbbe_loss.item(),
            'elbo_loss': elbo_loss.item()
        }
