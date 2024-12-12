from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


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


class AleatoricNetwork(nn.Module):
    """Complete aleatoric network implementing B(zal, q_t, phi)."""

    def __init__(
        self,
        history_dim: int,
        hidden_dim: int,
        num_coupling_layers: int = 6,  # L in the paper
    ):
        super().__init__()

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

        # ActNorm layers (as mentioned in paper)
        self.act_norms = nn.ModuleList(
            [ActNorm(param_dim) for _ in range(num_flow_layers)]
        )

        # Masked Autoregressive Flow layers
        self.flows = nn.ModuleList(
            [
                MaskedAutoregressiveFlow(dim=param_dim, hidden_dim=hidden_dim)
                for _ in range(num_flow_layers)
            ]
        )

        # LU decomposition for final transformation
        self.lu_transform = LUTransform(param_dim)

    def forward(self, z_ep: Tensor) -> Tensor:
        """
        Transform base samples into parameter space samples.

        Args:
            z_ep: Samples from base distribution N(0, I) [batch_size, param_dim]

        Returns:
            Transformed samples representing parameters phi [batch_size, param_dim]
        """
        x = z_ep

        # Apply flow transformations
        log_det = 0.0
        for act_norm, flow in zip(self.act_norms, self.flows):
            x, ldj_act = act_norm(x)
            x, ldj_flow = flow(x)
            log_det += ldj_act + ldj_flow

        # Final LU transformation
        x, ldj_final = self.lu_transform(x)
        log_det += ldj_final

        return x, log_det


class ActNorm(nn.Module):
    """Activation Normalization layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, dim))
        self.scale = nn.Parameter(torch.ones(1, dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y = self.scale * (x + self.loc)
        log_det = torch.sum(torch.log(torch.abs(self.scale)))
        return y, log_det


class MaskedAutoregressiveFlow(nn.Module):
    """
    Implements a single MAF layer with Masked Autoencoder for Distribution
    Estimation (MADE) network architecture.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = MADE(dim, hidden_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        params = self.net(x)
        mu, log_sigma = params.chunk(2, dim=-1)

        # Transform
        x_new = mu + torch.exp(log_sigma) * x
        log_det = torch.sum(log_sigma, dim=-1)

        return x_new, log_det
