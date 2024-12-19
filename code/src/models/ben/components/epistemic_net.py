from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch._prims_common import DeviceLikeType


class ActNorm(nn.Module):
    """
    Activation Normalization layer for normalizing flows.
    Performs an affine transformation with learned parameters.
    """

    def __init__(self, dim: int):
        super().__init__()

        # Initialize scale to ones and bias to zeros
        self.scale = nn.Parameter(torch.ones(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor [batch_size, dim]
        Returns:
            - Transformed tensor [batch_size, dim]
            - Log determinant of transformation [batch_size]
        """
        if not self.initialized:
            # More stable initialization using median
            with torch.no_grad():
                # Use median instead of mean for robustness
                median = x.median(dim=0, keepdim=True)[0]
                # Use robust scale estimation
                mad = (
                    torch.median(torch.abs(x - median), dim=0, keepdim=True)[0] * 1.4826
                )
                # Prevent division by zero
                mad = torch.clamp(mad, min=1e-5)

                self.bias.data = -median
                self.scale.data = 1.0 / mad
            self.initialized.fill_(True)

        # Apply normalization with numerical safeguards
        # scale = torch.exp(torch.clamp(self.scale, min=-10.0, max=10.0))
        scale = torch.clamp(self.scale, min=-5.0, max=5.0)
        normalized = scale * (x + self.bias)
        log_det = torch.sum(torch.log(torch.abs(scale) + 1e-6))

        return normalized, log_det


class MaskedLinear(nn.Linear):
    """Linear layer with a configurable mask on the weights."""

    def __init__(self, in_features: int, out_features: int, mask: Tensor):
        super().__init__(in_features, out_features)
        # Ensure mask shape matches weight matrix
        assert mask.shape == (
            out_features,
            in_features,
        ), f"Mask shape {mask.shape} must match weight shape {self.weight.shape}"
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation.
    Implements autoregressive property through masked linear layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_masks: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Create masks that enforce autoregressive property
        input_degrees = torch.arange(input_dim)
        hidden_degrees = torch.randint(1, input_dim - 1, (hidden_dim,))
        output_degrees = torch.arange(input_dim)

        # First layer mask: [hidden_dim, input_dim]
        mask1 = (hidden_degrees.unsqueeze(-1) >= input_degrees).float()

        # Output layer mask: [2*input_dim, hidden_dim] for mean and scale
        final_degrees = torch.cat([output_degrees, output_degrees])
        mask2 = (final_degrees.unsqueeze(-1) >= hidden_degrees).float()

        # Create network with proper masking
        self.layers = nn.ModuleList(
            [
                MaskedLinear(input_dim, hidden_dim, mask1),
                nn.ReLU(),
                MaskedLinear(hidden_dim, 2 * input_dim, mask2),  # 2* for mean and scale
            ]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Transform input through autoregressive network to produce
        mean and scale parameters for the flow transformation.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Tuple of:
            - means [batch_size, input_dim]
            - log_scales [batch_size, input_dim]
        """
        batch_size = x.shape[0]

        # Forward pass through masked layers
        h = x
        for layer in self.layers[:-1]:
            h = layer(h)

        # Final layer outputs concatenated means and log_scales
        outputs = self.layers[-1](h)

        # Split into means and log_scales
        means, log_scales = outputs.chunk(2, dim=-1)

        return means, log_scales


class ResMADE(nn.Module):
    """MADE with residual connections for improved gradient flow."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.made = MADE(input_dim, hidden_dim)
        self.skip = nn.Linear(input_dim, 2 * input_dim)

        # Crucial: Initialize with very small weights
        with torch.no_grad():
            self.skip.weight.data.mul_(0.01)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        means, log_scales = self.made(x)

        # Shape tracking for diagnosis
        batch_size = x.shape[0]

        # Maintain batch dimension throughout
        means = torch.clamp(means, min=-50.0, max=50.0)
        log_scales = torch.clamp(log_scales, min=-4.0, max=4.0)

        skip = self.skip(x)
        skip_mean, skip_scale = skip.chunk(2, dim=-1)

        means = means + 0.05 * torch.tanh(skip_mean)
        log_scales = log_scales + 0.05 * torch.tanh(skip_scale)

        z_new = means + torch.exp(log_scales) * x

        # Critical fix: Ensure log_det maintains batch dimension
        log_det = log_scales.sum(dim=-1)  # Shape: [batch_size]

        return z_new, log_det


# class MaskedAutoregressiveFlow(nn.Module):
#     """
#     Implements a single MAF layer with Masked Autoencoder for Distribution
#     Estimation (MADE) network architecture.
#     """
#
#     def __init__(self, dim: int, hidden_dim: int):
#         super().__init__()
#         self.made = MADE(dim, hidden_dim)
#
#     def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
#         """Transform input through coupling layer."""
#         means, log_scales = self.made.forward(z)
#
#         # Transform
#         # z_new = means + torch.exp(log_scales) * z
#         z_new = means + torch.exp(torch.clamp(log_scales, min=-5.0, max=5.0)) * z
#         log_det = torch.sum(torch.clamp(log_scales, min=-5.0, max=5.0), dim=-1)
#
#         return z_new, log_det


class LULinearTransform(nn.Module):
    """
    Implements an invertible linear transformation using LU decomposition.
    This provides a more numerically stable and efficient way to compute
    determinants compared to using a full weight matrix.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Initialize a random orthogonal matrix using QR decomposition
        w = torch.randn(dim, dim)
        q, _ = torch.linalg.qr(w, mode="reduced")
        w_init = q

        # Get LU decomposition with pivoting
        lu, pivots = torch.linalg.lu_factor(w_init)
        p, l, u = torch.lu_unpack(lu, pivots)

        # Create masks for lower and upper triangular matrices
        self.register_buffer("p", p)  # Fixed permutation matrix
        self.register_buffer("l_mask", torch.tril(torch.ones(dim, dim), -1))
        self.register_buffer("u_mask", torch.triu(torch.ones(dim, dim), 1))

        # Initialize trainable parameters
        # Note: we need to match dimensions carefully here
        self.l_diag = nn.Parameter(torch.ones(dim))
        self.u_diag = nn.Parameter(torch.ones(dim))
        self.l_params = nn.Parameter(l[self.l_mask.bool()])
        self.u_params = nn.Parameter(u[self.u_mask.bool()])

    def _assemble_matrices(self):
        """Assembles L and U matrices from parameters."""
        # Create base matrices of correct size
        l = torch.zeros(self.dim, self.dim, device=self.l_params.device)
        u = torch.zeros(self.dim, self.dim, device=self.u_params.device)

        # Fill in the diagonal elements
        l.diagonal().copy_(self.l_diag)
        u.diagonal().copy_(self.u_diag)

        # Fill in off-diagonal elements using masks
        l[self.l_mask.bool()] = self.l_params
        u[self.u_mask.bool()] = self.u_params

        return l, u

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply the LU transform and compute log determinant.

        Args:
            x: Input tensor [batch_size, dim]

        Returns:
            - Transformed tensor [batch_size, dim]
            - Log determinant of transformation [batch_size]
        """
        l, u = self._assemble_matrices()

        # NOTE: use @ for matrix multiplication.

        # Transform: x -> PLUx
        x = x @ self.p.t()  # Permutation
        x = x @ l  # Lower triangular
        x = x @ u  # Upper triangular

        # Log determinant is sum of log of diagonal elements
        # Note: P matrix has determinant ±1 so doesn't affect log det
        log_det = torch.sum(
            torch.log(torch.abs(self.l_diag)) + torch.log(torch.abs(self.u_diag))
        )

        return x, log_det


class EpistemicNetwork(nn.Module):
    """
    Implements variational inference over model parameters using normalizing flows.
    As described in Section 4.1 and Appendix C.2 of the paper, this network transforms
    a base distribution into an approximation of the posterior over parameters phi.
    """

    def __init__(
        self,
        param_dim: int,  # Dimension of parameter space phi
        hidden_dim: int = 64,
        num_flow_layers: int = 2,
    ):
        super().__init__()

        # Base distribution is standard normal in param_dim dimensions
        self.param_dim = param_dim

        # Following the sequence in the paper:
        # ActNorm -> MAF -> ActNorm -> MAF -> LU
        self.flow_sequence = nn.ModuleList(
            [
                # First block
                ActNorm(param_dim),
                # MaskedAutoregressiveFlow(dim=param_dim, hidden_dim=hidden_dim),
                ResMADE(param_dim, hidden_dim),  # Instead of MAF
                # Second block
                ActNorm(param_dim),
                # MaskedAutoregressiveFlow(dim=param_dim, hidden_dim=hidden_dim),
                ResMADE(param_dim, hidden_dim),  # Instead of MAF
                # Final LU transform
                LULinearTransform(param_dim),
            ]
        )

    def forward(self, z_ep: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Transform base samples into variational posterior samples.

        Args:
            z_ep: Samples from base distribution N(0, I^d) [batch_size, param_dim]

        Returns:
            - Transformed parameters phi [batch_size, param_dim]
            - Log determinant of Jacobian for ELBO computation [batch_size]
        """
        # print("\nForward pass - Epistemic Net:")
        # print(f"  z_ep shape: {z_ep.shape}")
        # print(f"  z_ep values: {z_ep}")
        #
        # log_det_sum = torch.zeros(z_ep.shape[0], device=z_ep.device)
        # phi = z_ep
        #
        # # Apply transformations sequentially
        # for i, transform in enumerate(self.flow_sequence):
        #     phi, ldj = transform.forward(phi)
        #     print(f"\nAfter transform {i}:")
        #     print(f"  phi shape: {phi.shape}")
        #     print(f"  phi values: {phi}")
        #     log_det_sum += ldj  # Adding tensors of shape [batch_size]
        #
        # return phi, log_det_sum

        ###########
        #
        # log_det_sum = torch.zeros(z_ep.shape[0], device=z_ep.device)
        # phi = z_ep
        #
        # # Track values for debugging
        # intermediate_values = []
        #
        # for i, transform in enumerate(self.flow_sequence):
        #     phi_prev = phi
        #     phi, ldj = transform.forward(phi)
        #
        #     # Monitor for extreme values
        #     with torch.no_grad():
        #         delta = (phi - phi_prev).abs().mean()
        #         max_val = phi.abs().max()
        #         if max_val > 100:
        #             print(f"Warning: Large values in layer {i}: max={max_val:.2f}")
        #         intermediate_values.append(
        #             {
        #                 "layer": i,
        #                 "mean": phi.mean().item(),
        #                 "std": phi.std().item(),
        #                 "delta": delta.item(),
        #             }
        #         )
        #
        #     # Clamp values if needed
        #     if i < len(self.flow_sequence) - 1:  # Don't clamp final LU output
        #         phi = torch.clamp(phi, min=-100, max=100)
        #
        #     log_det_sum += ldj
        #
        # return phi, log_det_sum

        ##############
        batch_size = z_ep.shape[0]
        phi = z_ep
        log_det_sum = torch.zeros(batch_size, device=z_ep.device)

        # Progressive clamping schedule
        clamp_schedule = [
            (100.0, 0.1),  # (clamp_val, damping_factor)
            (75.0, 0.08),
            (50.0, 0.06),
            (25.0, 0.04),
            (10.0, 0.02),
        ]

        for i, transform in enumerate(self.flow_sequence):
            if i < len(clamp_schedule):
                clamp_val, damp = clamp_schedule[i]
            else:
                clamp_val, damp = clamp_schedule[-1]

            phi_prev = phi
            phi, ldj = transform(phi)

            # Progressive stabilization with proper shape maintenance
            if i < len(self.flow_sequence) - 1:
                phi = torch.tanh(phi / clamp_val) * clamp_val * damp

            # Ensure ldj has shape [batch_size]
            if ldj.dim() == 0:
                ldj = ldj.expand(batch_size)

            log_det_sum = log_det_sum + ldj

        return phi, log_det_sum

    def sample_base(
        self,
        batch_size: int,
        n: int = 1,
        device: Optional[DeviceLikeType] = None,
    ) -> Tensor:
        """Sample z_ep ~ N(0,I_d)"""
        if n == 1:
            return torch.randn(batch_size, self.param_dim, device=device)

        return torch.randn(batch_size, n, self.param_dim, device=device)
