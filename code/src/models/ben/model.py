from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RecurrentQNetwork(nn.Module):
    """
    TODO: document this class.
    """

    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        hidden_dim: int = 64,
        gru_hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim

        # Encoder for the state-action-reward tuples
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + 1 + 1, hidden_dim),  # state + reward + action
            nn.ReLU(),
        )

        # Gated Recurrent Unit (GRU) for processing history
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)

        # Action embedding
        embedding_dim = min(
            hidden_dim // 2, num_actions * 4
        )  # Prevent action embedding from dominating the history encoding
        self.action_embedding = nn.Embedding(num_actions, embedding_dim)

        # Q-value head takes both history encoding and action
        self.q_head = nn.Linear(
            gru_hidden_dim + embedding_dim, 1
        )  # Input: history (processed by the GRU) + action -> corresponding q-value

    def forward(
        self,
        history_states: Tensor,
        history_actions: Tensor,
        history_rewards: Tensor,
        action: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Maps history-action pairs to Q-values using RNN encoding.

        As described in Section 4.1 of the BEN paper, this function approximator
        takes a history h_t (sequence of state-action-reward tuples) and an action a_t,
        and outputs the corresponding Q-value.

        Args:
            history_states: States [batch_size, seq_len, state_dim]
            history_actions: Actions [batch_size, seq_len]
            history_rewards: Rewards [batch_size, seq_len]
            action: Current actions [batch_size]
            hidden: Optional GRU hidden state [1, batch_size, gru_hidden_dim]

        Returns:
            Tuple of:
            - Q-value for given history-action pair [batch_size, 1]
            - History encoding for given history-action pair [batch_size, seq_len, hidden_dim]
            - Updated GRU hidden state [1, batch_size, gru_hidden_dim]
        """
        # Encode history
        batch_size = history_states.size(0)

        # Combine and encode history elements
        history = torch.cat(
            [
                history_states,  # Shape: [batch_size, seq_len, state_dim]
                history_actions.unsqueeze(-1),  # Shape: [batch_size, seq_len, 1]
                history_rewards.unsqueeze(-1),  # Shape: [batch_size, seq_len, 1]
            ],
            dim=-1,
        )

        # Encode each timestep
        encoded_history: Tensor = self.encoder.forward(
            history
        )  # Shape: [batch_size, seq_len, hidden_dim]

        # Get history encoding through GRU
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.gru_hidden_dim)

        history_encoding, hidden = self.gru.forward(encoded_history, hidden)
        history_encoding = history_encoding[:, -1]  # Take last hidden state

        # Embed current action
        action_embedding = self.action_embedding.forward(action)

        # Combine history encoding with action to get Q-value
        q_input = torch.cat([history_encoding, action_embedding], dim=-1)
        q_value = self.q_head.forward(q_input)

        return q_value, history_encoding, hidden


class AleatoricConditioner(nn.Module):
    """Implements κ_phi_i from the paper - conditions flow on history and Q-value."""

    def __init__(self, history_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(history_dim + 1, hidden_dim),  # +1 for q_t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output dimensionality determined by coupling function needs
        )

    def forward(self, h_hat: Tensor, q_t: Tensor) -> Tensor:
        """
        Args:
            h_hat: History encoding [batch_size, history_dim]
            q_t: Q-value [batch_size, 1]
        """
        inputs = torch.cat([h_hat, q_t], dim=-1)
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
            conditioner_output: Output from κ_phi_i [batch_size, hidden_dim]
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
        """Implements B(zal, q_t, phi) as described in the paper."""
        # Get conditioning parameters
        cond = self.conditioner.forward(h_hat, q_t)

        # Apply coupling functions in sequence
        x: Tensor = z_al
        for coupling in self.coupling_layers:
            x = coupling.forward(x, cond)

        # Reduce to 1D
        return self.projection.forward(x)


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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor [batch_size, dim]
        Returns:
            - Transformed tensor [batch_size, dim]
            - Log determinant of transformation [batch_size]
        """
        if not self.initialized:
            # Data-dependent initialization
            with torch.no_grad():
                # Initialize to make first batch have zero mean and unit variance
                mean = x.mean(0)
                std = x.std(0)

                self.bias.data = -mean
                self.scale.data = 1 / (std + 1e-6)
            self.initialized = True

        # Apply affine transformation
        y = self.scale * (x + self.bias)

        # Compute log determinant of the transformation
        # Note: sum over dimension because we're transforming each dimension independently
        log_det = torch.sum(torch.log(torch.abs(self.scale)))

        return y, log_det


class MaskedLinear(nn.Linear):
    """Linear layer with a configurable mask on the weights."""

    def __init__(self, in_features: int, out_features: int, mask: Tensor):
        super().__init__(in_features, out_features)
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
        self.num_masks = num_masks

        # Create masks for each layer
        masks = self.create_masks()

        # Masked linear layers
        self.net = nn.ModuleList(
            [
                MaskedLinear(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim
                    if i < num_masks - 1
                    else input_dim * 2,  # *2 for mean and scale
                    mask=masks[i],
                )
                for i in range(num_masks)
            ]
        )

        self.activation = nn.ReLU()

    def create_masks(self) -> List[Tensor]:
        """
        Create masks that enforce autoregressive property.
        Each output dimension i can only depend on input dimensions < i.
        """
        # Create degree vectors for each layer
        degrees = [
            torch.arange(self.input_dim),  # Input degrees
            torch.randint(1, self.input_dim - 1, (self.hidden_dim,)),  # Hidden degrees
            torch.arange(self.input_dim),  # Output degrees
        ]

        # Create masks based on degrees
        masks = []
        for d_in, d_out in zip(degrees[:-1], degrees[1:]):
            # Create mask where entry (i,j) is 1 if d_in[j] < d_out[i]
            mask = (d_in.unsqueeze(0) < d_out.unsqueeze(1)).float()
            masks.append(mask)

        return masks

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through MADE network.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Parameters for transformation [batch_size, input_dim * 2]
            (mean and log_scale for each dimension)
        """
        batch_size = x.shape[0]
        h = x

        # Pass through masked layers
        for i, layer in enumerate(self.net[:-1]):
            h = self.activation.forward(layer.forward(h))

        # Final layer outputs parameters
        params = self.net[-1](h)

        return params


class MaskedAutoregressiveFlow(nn.Module):
    """
    Implements a single MAF layer with Masked Autoencoder for Distribution
    Estimation (MADE) network architecture.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = MADE(dim, hidden_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        params = self.net.forward(x)
        mu, log_sigma = params.chunk(2, dim=-1)

        # Transform
        x_new = mu + torch.exp(log_sigma) * x
        log_det = torch.sum(log_sigma, dim=-1)

        return x_new, log_det


class LULinearTransform(nn.Module):
    """
    Implements an invertible linear transformation using LU decomposition.
    This provides a more numerically stable and efficient way to compute
    determinants compared to using a full weight matrix.
    """

    def __init__(self, dim: int):
        super().__init__()

        # Initialize a random rotation matrix using QR decomposition
        q, _ = torch.linalg.qr(torch.randn(dim, dim), mode="reduced")
        w_init = q

        # Get the LU decomposition
        lu, pivots = torch.linalg.lu_factor(w_init)

        # Unpack the LU factorization
        p, l, u = torch.lu_unpack(lu, pivots)

        # Register parameters
        # Note: we store l and u without diagonal elements
        self.register_buffer("p", p)  # Fixed permutation matrix
        self.l_mask = torch.tril(torch.ones(dim, dim), -1)
        self.u_mask = torch.triu(torch.ones(dim, dim), 1)

        self.l_diag = nn.Parameter(torch.ones(dim))
        self.u_diag = nn.Parameter(torch.ones(dim))
        self.l_params = nn.Parameter(l[self.l_mask.bool()])
        self.u_params = nn.Parameter(u[self.u_mask.bool()])

    def _assemble_matrices(self):
        """Assembles L and U matrices from parameters."""
        dim = self.p.shape[0]

        # Assemble L matrix
        l = self.l_mask * self.l_params.view(-1)
        l = l + torch.diag(self.l_diag)

        # Assemble U matrix
        u = self.u_mask * self.u_params.view(-1)
        u = u + torch.diag(self.u_diag)

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
                MaskedAutoregressiveFlow(dim=param_dim, hidden_dim=hidden_dim),
                # Second block
                ActNorm(param_dim),
                MaskedAutoregressiveFlow(dim=param_dim, hidden_dim=hidden_dim),
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
        batch_size = x.shape[0]

        x = z_ep
        log_det_sum = torch.zeros(batch_size, device=x.device)

        # Apply transformations sequentially
        for transform in self.flow_sequence:
            x, ldj = transform.forward(x)
            log_det_sum += ldj  # Adding tensors of shape [batch_size]

        return phi, log_det_sum


@dataclass
class BENOutput:
    q_value: Tensor
    aleatoric_params: Tensor
    epistemic_params: Tensor


class BayesianExplorationNetwork(nn.Module):
    """
    Implementation of Bayesian Exploration Network (BEN) combining:
    - RecurrentQNetwork for state-action-history encoding
    - AleatoricNetwork for modeling inherent environment randomness
    - TODO: EpistemicNetwork for capturing parameter uncertainty
    """

    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        hidden_dim: int = 64,
        gru_hidden_dim: int = 64,
        num_coupling_layers: int = 6,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Q-function approximator network
        self.q_net = RecurrentQNetwork(
            num_actions=num_actions,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
        )

        # Aleatoric network
        self.aleatoric_net = AleatoricNetwork(
            history_dim=gru_hidden_dim,
            hidden_dim=hidden_dim,
            num_coupling_layers=num_coupling_layers,
        )

        # Epistemic network
        self.epistemic_net = EpistemicNetwork(
            param_dim=64,
            hidden_dim=hidden_dim,
            num_flow_layers=num_coupling_layers,
        )

    def compute_elbo(self, history: Tensor, bootstrap_samples: Tensor) -> Tensor:
        """Compute ELBO objective for variational inference."""
        z_ep = torch.randn(batch_size, self.param_dim)  # Sample from base distribution
        phi, log_det = self.epistemic_net.forward(z_ep)

        # Compute log likelihood and prior terms
        log_likelihood = self.compute_log_likelihood(history, bootstrap_samples, phi)
        log_prior = self.compute_log_prior(phi)

        return log_likelihood + log_prior - log_det

    def compute_log_likelihood(
        self,
        history: Tensor,
        bootstrap_samples: Tensor,
        phi: Tensor,
    ) -> Tensor:
        pass

    def compute_log_prior(self, phi: Tensor) -> Tensor:
        pass
