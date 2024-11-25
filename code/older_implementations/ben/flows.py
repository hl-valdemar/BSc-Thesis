from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from .config import BENConfig

class NormalizingFlow(nn.Module):
    """Base class for normalizing flows."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform input x and return transformed value z and log determinant.
        """
        raise NotImplementedError

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform input z and return transformed value x and log determinant.
        """
        raise NotImplementedError

class AffineCouplingLayer(NormalizingFlow):
    def __init__(self, dim: int):
        super().__init__(dim)

        # Ensure even dimension for splitting
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")

        self.split_dim = dim // 2
        hidden_dim = max(dim * 2, 4) # Ensure sufficient capacity

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_dim),
            nn.Tanh() # Keep scale factors bounded
        )

        self.translation_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_dim)
        )

    def _split_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split input tensor into two parts along last dimension.
        """
        return x[..., :self.split_dim], x[..., self.split_dim:]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: x -> z.
        Returns transformed variable and log determinant.
        """
        # Ensure input has the right shape: [batch_size, dim]
        if len(x.shape) == 3:
            x = x.squeeze(1) # Remove sequence dimension if present

        if x.shape[-1] != self.dim:
            x = torch.zeros_like(x).expand(-1, self.dim)

        x1, x2 = self._split_input(x)

        # Compute scale and translation based on x1
        s = self.scale_net.forward(x1)
        t = self.translation_net.forward(x1)

        # Identity transformation for the first half
        z1 = x1

        # Transform x2
        z2 = x2 * torch.exp(s) + t

        # Combine back
        z = torch.cat([z1, z2], dim=-1)

        # Log determinant of Jacobian
        log_det = torch.sum(s, dim=-1)

        return z, log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: z -> x.
        Returns transformed variable and log determinant.
        """
        # Ensure input has the right shape: [batch_size, dim]
        if len(z.shape) == 3:
            z = z.squeeze(1) # Remove sequence dimension if present

        if z.shape[-1] != self.dim:
            z = torch.zeros_like(z).expand(-1, self.dim)

        z1, z2 = self._split_input(z)

        # Compute scale and translation based on z1
        s = self.scale_net.forward(z1)
        t = self.translation_net.forward(z1)

        # Identity transformation for first half
        x1 = z1

        # Inverse transform z2
        x2 = (z2 - t) * torch.exp(-s)

        # Combine back
        x = torch.cat([x1, x2], dim=-1)

        # Log determinant of inverse Jacobian
        log_det = -torch.sum(s, dim=-1)

        return x, log_det

class AleatoricNetwork(nn.Module):
    def __init__(self, config: BENConfig):
        """
        The aleatoric network models uncertainty in the Bellman
        operator, which is essentially uncertainty in Q-values.
        Since Q-values are scalar, we extend it to 2D to make the
        coupling mechanism work for the affine coupling layer.
        """
        super().__init__()
        self.config = config

        # Use 4D space for aleatoric uncertainty
        self.flow_dim = 4
        # Base distribution
        self.base_dist = Normal(0, 1)

        # Project Q-values to flow dimension
        self.projection = nn.Sequential(
            nn.Linear(config.action_dim, self.flow_dim),
            nn.ReLU(),
        )

        # Series of normalizing flows
        # - Base distribution for Q-value uncertainty (1D -> 2D for coupling)
        self.flows = nn.ModuleList([
            AffineCouplingLayer(dim=self.flow_dim)
            for _ in range(config.num_flows)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform from base distribution to target distribution.
        Returns transformed sample and log probability.
        """
        # Remove sequence dimension if present and project
        if len(x.shape) == 3:
            x = x.squeeze(1)
        x = self.projection(x) # Shape: [batch_size, flow_dim]

        # Sample from base distribution with matching batch size
        z = self.base_dist.sample(x.shape[:-1]) # Shape: [batch_size]
        z = z.unsqueeze(-1).expand(-1, self.flow_dim) # Shape: [batch_size, flow_dim]
        log_prob = self.base_dist.log_prob(z).sum(-1) # Shape: [batch_size]

        # Transform through flows
        for flow in self.flows:
            z, ldj = flow.forward(z)
            log_prob = log_prob - ldj

        return z, log_prob

class EpistemicNetwork(nn.Module):
    def __init__(self, config: BENConfig):
        """
        This network models uncertainty in the model parameters.
        The parameters live in a higher-dimensional space, specifically
        the dimension of our hidden layers.
        """
        super().__init__()
        self.config = config

        # Ensure even dimension
        self.flow_dim = config.hidden_dim + (config.hidden_dim % 2)
        # Base distribution for parameter uncertainty (in hidden_dim space)
        self.base_dist = Normal(0, 1)

        # Project Q-values to flow dimension
        self.projection = nn.Sequential(
            nn.Linear(config.action_dim, self.flow_dim),
            nn.ReLU(),
        )

        self.flows = nn.ModuleList([
            # Model uncertainty in parameter space
            AffineCouplingLayer(dim=self.flow_dim)
            for _ in range(config.num_flows)
        ])

        # Project input to flow dimension
        self.projection = nn.Linear(config.action_dim, self.flow_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Remove sequence dimension if present and project
        if len(x.shape) == 3:
            x = x.squeeze(1)
        x = self.projection(x) # Shape: [batch_size, flow_dim]

        # Sample from base distribution with matching batch size
        z = self.base_dist.sample(x.shape[:-1]) # Shape: [batch_size]
        z = z.unsqueeze(-1).expand(-1, self.flow_dim) # Shape: [batch_size, flow_dim]
        log_prob = self.base_dist.log_prob(z).sum(-1) # Shape: [batch_size]

        # Transform through flows
        for flow in self.flows:
            z, ldj = flow.forward(z)
            log_prob = log_prob - ldj

        return z, log_prob
