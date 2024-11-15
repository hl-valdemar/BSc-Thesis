from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BENConfig, BENOutput
from .rqn import RecurrentQNetwork
from .flows import AleatoricNetwork, EpistemicNetwork

class BayesianExplorationNetwork(nn.Module):
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
               ) -> BENOutput:
        # print(f"\nForward pass tensor shapes:")
        # print(f"Input shapes - state: {state.shape}, reward: {reward.shape}")
        # if hidden is not None:
        #     print(f"Hidden: {hidden.shape}")

        # Get Q-values
        q_values, new_hidden = self.q_network.forward(state, reward, hidden)
        # print(f"Q-network - q_values: {q_values.shape}")
        # if new_hidden is not None:
        #     print(f"Q-network - new_hidden: {new_hidden.shape}")

        # Get aleatoric uncertainty
        # print(f"Aleatoric network input shape: {q_values.shape}")
        aleatoric_sample, aleatoric_log_prob = self.aleatoric_network.forward(q_values)
        # print(f"Aleatoric shapes - sample: {aleatoric_sample.shape}, log_prob: {aleatoric_log_prob.shape}")

        # Get epistemic uncertainty
        # print(f"Epistemic network input shape: {q_values.shape}")
        epistemic_sample, epistemic_log_prob = self.epistemic_network.forward(q_values)
        # print(f"Epistemic shapes - sample: {epistemic_sample.shape}, log_prob: {epistemic_log_prob.shape}")

        return BENOutput(
            q_values=q_values,
            aleatoric_sample=aleatoric_sample,
            aleatoric_log_prob=aleatoric_log_prob,
            epistemic_sample=epistemic_sample,
            epistemic_log_prob=epistemic_log_prob,
            hidden=new_hidden
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update networks using MSBBE and ELBO objectives.
        Returns dict with loss values.
        """
        # Unpack batch
        states = batch['states'] # Shape: [batch_size, state_dim]
        rewards = batch['rewards'] # Shape: [batch_size]
        next_states = batch['next_states'] # Shape: [batch_size, state_dim]
        actions = batch['actions'] # Shape: [batch_size]

        # Forward pass
        ben_now = self.forward(states, rewards)
        ben_next = self.forward(next_states, rewards)

        # Remove sequence dimension and get target Q-values
        current_q = ben_now.q_values.squeeze(1) # Shape: [batch_size, action_dim]
        next_q = ben_next.q_values.squeeze(1) # Shape: [batch_size, action_dim]

        # Calculate target Q-values
        next_value = next_q.max(dim=1)[0] # Shape: [batch_size]
        target_q = rewards + self.config.gamma * next_value # Shape: [batch_size]

        # Get Q-values for taken actions
        action_q_values = current_q[torch.arange(current_q.size(0)), actions] # Shape: [batch_size]

        # Calculate MSBBE loss
        msbbe_loss = F.mse_loss(action_q_values, target_q)

        # Calculate ELBO loss
        elbo_loss = -ben_now.aleatoric_log_prob.mean() - ben_now.epistemic_log_prob.mean()

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
