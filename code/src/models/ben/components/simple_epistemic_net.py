from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch._prims_common import DeviceLikeType


class SimpleEpistemicNetwork(nn.Module):
    """
    A simplified version of the epistemic network using a basic normalizing flow.
    Uses a single transformation with careful conditioning.
    """

    def __init__(self, param_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.param_dim = param_dim

        # Simple conditioning network
        self.conditioner = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.Tanh(),  # More stable than ReLU
            nn.Linear(hidden_dim, 2 * param_dim),  # Outputs scale and shift
        )

        # Initialize close to identity transformation
        for layer in self.conditioner:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, z_ep: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Single-step flow transformation:
        phi = z_ep * exp(s) + t
        where s, t are learned scale and translation
        """
        params = self.conditioner(z_ep)
        scale, shift = params.chunk(2, dim=-1)

        # Constrain scale for stability
        scale = torch.tanh(scale) * 0.1

        # Perform transformation
        phi = z_ep * torch.exp(scale) + shift

        # Log determinant of Jacobian
        log_det = scale.sum(-1)

        return phi, log_det

    def sample_base(
        self,
        batch_size: int,
        n: int = 1,
        device: Optional[DeviceLikeType] = None,
    ) -> Tensor:
        """Sample z_ep ~ N(0,I_d) as described in paper"""
        shape = (
            (batch_size, n, self.param_dim) if n > 1 else (batch_size, self.param_dim)
        )

        return torch.randn(shape, device=device) * 0.1  # Scaled down samples
