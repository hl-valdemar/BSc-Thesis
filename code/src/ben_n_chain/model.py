from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class BENOutput:
    """
    Container for BEN forward pass outputs.
    """

    q_value: Tensor
    aleatoric_params: Tensor
    epistemic_params: Tensor


class RecurrentQNetwork(nn.Module):
    """
    Recurrent Q-Network component of BEN.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Encode state
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # GRU for processing history
        self.gru = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )

        # Q-value head
        self.q_head = nn.Linear(hidden_dim, 2)  # 2 actions for N-Chain

    def forward(
        self, states: Tensor, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through Q-network."""
        batch_size = states.size(0)

        # Initialize hidden state if None
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=states.device)

        # Encode state
        x = self.encoder(states)
        x = x.unsqueeze(1)  # Add time dimension [batch_size, 1, hidden_dim]

        # Process with GRU
        x, new_hidden = self.gru(x, hidden)

        # Get Q-values
        q_values = self.q_head(x.squeeze(1))

        return q_values, new_hidden


class AleatoricNetwork(nn.Module):
    """
    Aleatoric Network component of BEN.
    """

    def __init__(self, hidden_dim: int = 64, flow_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Network to generate flow parameters
        self.param_net = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),  # hidden state + q_values
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * flow_layers),  # 2 params per flow layer
        )

        self.flow_layers = flow_layers

    def forward(self, q_values: Tensor, hidden: Tensor) -> Tensor:
        """Forward pass through aleatoric network."""
        # Reshape hidden state from [1, batch_size, hidden_dim] to [batch_size, hidden_dim]
        hidden = hidden.squeeze(0)

        # Concatenate q_values and hidden state
        x = torch.cat([q_values, hidden], dim=1)

        # Generate flow parameters
        flow_params = self.param_net(x)

        return flow_params


class EpistemicNetwork(nn.Module):
    """
    Epistemic Network component of BEN.
    """

    def __init__(self, hidden_dim: int = 64, latent_dim: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),  # μ and σ for each latent dim
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through epistemic network."""
        # Reshape hidden state from [1, batch_size, hidden_dim] to [batch_size, hidden_dim]
        hidden = hidden.squeeze(0)

        # Encode to get distribution parameters
        params = self.encoder(hidden)
        mean, logvar = torch.chunk(params, 2, dim=-1)

        # Sample using reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        # Decode
        decoded = self.decoder(z)

        return mean, logvar, decoded


class BEN(nn.Module):
    """
    Bayesian Exploration Network for N-Chain environment.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        flow_layers: int = 2,
        latent_dim: int = 8,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Main components
        self.q_net = RecurrentQNetwork(state_dim, hidden_dim)
        self.aleatoric_net = AleatoricNetwork(hidden_dim, flow_layers)
        self.epistemic_net = EpistemicNetwork(hidden_dim, latent_dim)

    def forward(
        self, states: Tensor, hidden: Optional[Tensor] = None
    ) -> Tuple[BENOutput, Tensor]:
        """Forward pass through BEN."""
        # Get Q-values and hidden state
        q_values, new_hidden = self.q_net(states, hidden)

        # Get uncertainty estimates
        aleatoric_params = self.aleatoric_net(q_values, new_hidden)
        epistemic_mean, epistemic_logvar, epistemic_decoded = self.epistemic_net(
            new_hidden
        )

        outputs = BENOutput(
            q_value=q_values,
            aleatoric_params=aleatoric_params,
            epistemic_params=torch.cat([epistemic_mean, epistemic_logvar], dim=-1),
        )

        return outputs, new_hidden

    def get_action(
        self, state: Tensor, hidden: Optional[Tensor] = None, epsilon: float = 0.0
    ) -> Tuple[int, Tensor]:
        """Get action using ε-greedy policy."""
        outputs, new_hidden = self.forward(state, hidden)

        if torch.rand(1).item() < epsilon:
            action = torch.randint(2, (1,)).item()
        else:
            action = outputs.q_value.argmax(-1).item()

        return action, new_hidden
