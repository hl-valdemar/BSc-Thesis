from typing import Optional

import torch
from torch import Tensor, nn
from torch._prims_common import DeviceLikeType


class AleatoricConditioner(nn.Module):
    """Implements K_phi_i from the paper - conditions flow on history and Q-value."""

    def __init__(self, history_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(history_dim + 1, hidden_dim),  # +1 for q_t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output dimensionality determined by coupling function needs
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        inverse: bool = False,
    ) -> Tensor:
        """
        Args for forward pass (inverse=False):
            x1: History encoding [batch_size, seq_len]
            x2: Q-value [batch_size, 1]
        Args for inverse pass (inverse=True):
            x1: Q-values [batch_size, seq_len, 1]
            x2: Model parameters [batch_size, param_dim]
        """
        if inverse:
            # Handle inverse pass case
            x2 = x2.unsqueeze(1).expand(
                -1, x1.size(1), -1
            )  # Expand phi to match sequence dimension

        # Concatenate along feature dimension
        inputs = torch.cat([x1, x2], dim=-1)
        return self.net.forward(inputs)


class CouplingFunction(nn.Module):
    """Single coupling function f_i in the aleatoric flow."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Parameters for the inverse autoregressive transformation
        self.scale_net = nn.Linear(hidden_dim, 1)
        self.shift_net = nn.Linear(hidden_dim, 1)

    def forward(self, z: Tensor, conditioner_output: Tensor) -> Tensor:
        """
        Args:
            z: Input from previous flow layer [batch_size, 2]
            conditioner_output: Output from K_phi_i [batch_size, hidden_dim]
        """
        # Split input for autoregressive property
        z1, z2 = z.chunk(2, dim=-1)

        # Compute scale and shift based on first component
        scale = torch.exp(self.scale_net.forward(conditioner_output))
        shift = self.shift_net.forward(conditioner_output)

        # Transform second component
        z2_transformed = scale * z2 + shift

        return torch.cat([z1, z2_transformed], dim=-1)


class AleatoricNetwork(nn.Module):
    """Complete aleatoric network implementing B(z_al, q_t, phi)."""

    def __init__(
        self,
        history_dim: int,
        hidden_dim: int,
        num_coupling_layers: int = 6,  # L in the paper
    ):
        super().__init__()

        # K_phi_i
        self.conditioner = AleatoricConditioner(history_dim, hidden_dim)

        # Create L coupling functions
        self.coupling_layers = nn.ModuleList(
            [CouplingFunction(hidden_dim) for _ in range(num_coupling_layers)]
        )

        # Final dimensionality reduction (2D -> 1D)
        self.projection = nn.Linear(2, 1)

    def forward(
        self,
        z_al: Tensor,  # Base sample from N(0, I_2)
        h_hat: Tensor,  # History encoding
        q_t: Tensor,  # Q-value
    ) -> Tensor:
        """Implements B(zal, q_t, phi) as described in the paper.

        Args:
            -
            - h_hat: History encoding, shape [batch_size, seq_len]
            - q_t: Q-values so far, shape [batch_size, seq_len]
        """
        # Get conditioning parameters
        cond = self.conditioner.forward(h_hat, q_t)

        # Apply coupling functions in sequence
        x: Tensor = z_al
        for coupling in self.coupling_layers:
            x = coupling.forward(x, cond)

        # Reduce to 1D
        return self.projection.forward(x)

    def inverse(
        self, bootstrap_samples: Tensor, q_values: Tensor, phi: Tensor
    ) -> Tensor:
        """Inverse flow transformation that exactly undoes the forward transformation.
        Computes the inverse flow transformation B^(-1)(b_t, q_t, phi).

        Args:
            bootstrap_samples: Shape [batch_size, seq_len]  # The 1D Bellman samples
            q_values: Shape [batch_size, seq_len]  # Q-values for conditioning
            phi: Shape [batch_size, param_dim]  # Model parameters

        Returns:
            Tensor: Shape [batch_size, seq_len, 2] # Tensor in the base distribution space
        """
        batch_size, seq_len = bootstrap_samples.size()

        # 1. First undo the projection (1D -> 2D)
        W = self.projection.weight
        W_pinv = torch.pinverse(W)
        x = bootstrap_samples.unsqueeze(-1).matmul(W_pinv.T)

        # 2. Get conditioning parameters using phi instead of history
        cond = self.conditioner.forward(q_values, phi, inverse=True)

        # 3. Invert each coupling layer in reverse order
        for coupling in reversed(self.coupling_layers):
            # Split dimensions
            x1, x2 = x.chunk(2, dim=-1)

            # Get transformation parameters
            scale = torch.exp(coupling.scale_net.forward(cond))
            shift = coupling.shift_net.forward(cond)

            # Invert the transformation: z2 = (x2 - shift) / scale
            x2_orig = (x2 - shift) / scale

            # Recombine
            x = torch.cat([x1, x2_orig], dim=-1)

        return x  # Returns to base distribution space [batch_size, seq_len, 2]

    def log_det(
        self,
        bootstrap_samples: Tensor,
        q_values: Tensor,
        phi: Tensor,
    ) -> Tensor:
        """Computes log determinant of the Jacobian for the inverse transformation.
        In the ELBO computation, this accounts for the change in probability mass under
        the transformation.

        Args:
            bootstrap_samples: Shape [batch_size, seq_len]
            q_values: Shape [batch_size, seq_len]
            phi: Shape [batch_size, param_dim]

        Returns:
            Tensor: Shape [batch_size, seq_len] Log determinant for each timestep
        """
        batch_size, seq_len = bootstrap_samples.size()

        # Get conditioning parameters
        cond = self.conditioner.forward(q_values, phi, inverse=True)

        # Projection layer contribution
        # For a linear layer W, log|det(W)| = sum(log|singular_values|)
        W = self.projection.weight
        _, s, _ = torch.svd(W)
        log_det_proj = torch.sum(torch.log(torch.abs(s)))
        # Expand projection contribution to match sequence dimension
        log_det_sum = log_det_proj.expand(batch_size, seq_len)

        # Add contributions from each coupling layer, preserving dimensions
        for coupling in self.coupling_layers:
            scale = torch.exp(coupling.scale_net.forward(cond))
            log_det_sum = log_det_sum + torch.log(torch.abs(scale))

        return log_det_sum  # Shape: [batch_size, seq_len]

    def sample_base(
        self,
        batch_size: int,
        n: int = 1,
        device: Optional[DeviceLikeType] = None,
    ) -> Tensor:
        """Sample z_al ~ N(0,I_2) as described in Appendix C.1"""
        shape = (batch_size, n, 2) if n > 1 else (batch_size, 2)
        return (
            torch.randn(shape, device=device) * 0.1
        )  # 2D base variable as per paper (scaled down)
