from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.optim import Adam

from .components import AleatoricNetwork, EpistemicNetwork, RecurrentQNetwork


class MovingAverage:
    """Tracks exponential moving average for stable training."""

    def __init__(self, beta: float = 0.99):
        self.beta = beta
        self.value = None

    def update(self, new_value: float) -> None:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.beta * self.value + (1 - self.beta) * new_value


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
        param_dim: int = 64,
    ):
        super().__init__()
        self.num_actions = num_actions
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
        # self.epistemic_net = SimpleEpistemicNetwork(
        #     param_dim=param_dim,
        #     hidden_dim=hidden_dim,
        # )
        self.epistemic_net = EpistemicNetwork(
            param_dim=param_dim,
            hidden_dim=hidden_dim,
            num_flow_layers=num_coupling_layers,
        )

    def forward(
        self,
        history_states: Tensor,
        history_actions: Tensor,
        history_rewards: Tensor,
        current_action: Tensor,
        n_samples: int = 10,  # Number of samples for uncertainty estimation
    ) -> BENOutput:
        """
        Forward pass through the complete BEN architecture, implementing the full
        pipeline from Section 4.1 of the paper.

        Args:
            history_states: States sequence [batch_size, seq_len, state_dim]
            history_actions: Action sequence [batch_size, seq_len]
            history_rewards: Reward sequence [batch_size, seq_len]
            current_action: Current action to evaluate [batch_size]
            n_samples: Number of samples for uncertainty estimation

        Returns:
            BENOutput containing:
            - Q-value estimates
            - Aleatoric uncertainty parameters
            - Epistemic uncertainty parameters
        """
        batch_size = history_states.size(0)
        device = history_states.device

        # 1. Get Q-value and history encoding through RNN
        q_value, history_encoding, _ = self.q_net.forward(
            history_states,
            history_actions,
            history_rewards,
            current_action,
        )

        # 2. Sample from base distributions for uncertainty estimation
        aleatoric_samples = []
        epistemic_samples = []

        for _ in range(n_samples):
            # Sample from base distributions
            z_al = self.aleatoric_net.sample_base(batch_size, device=device)
            z_ep = self.epistemic_net.sample_base(batch_size, device=device)

            # Transform through epistemic network to get phi
            phi, _ = self.epistemic_net(z_ep)

            # Transform through aleatoric network
            bt = self.aleatoric_net.forward(z_al, history_encoding, q_value)

            aleatoric_samples.append(bt)
            epistemic_samples.append(phi)

        # 3. Aggregate samples into uncertainty estimates
        aleatoric_samples = torch.stack(
            aleatoric_samples, dim=1
        )  # [batch_size, n_samples, 1]
        epistemic_samples = torch.stack(
            epistemic_samples, dim=1
        )  # [batch_size, n_samples, param_dim]

        # Compute statistics for uncertainty estimation
        aleatoric_mean = aleatoric_samples.mean(dim=1)
        aleatoric_std = aleatoric_samples.std(dim=1)
        epistemic_mean = epistemic_samples.mean(dim=1)
        epistemic_std = epistemic_samples.std(dim=1)

        # Pack parameters into tensors
        aleatoric_params = torch.cat([aleatoric_mean, aleatoric_std], dim=-1)
        epistemic_params = torch.cat([epistemic_mean, epistemic_std], dim=-1)

        return BENOutput(
            q_value=q_value,
            aleatoric_params=aleatoric_params,
            epistemic_params=epistemic_params,
        )

    def compute_prior_loss(
        self,
        prior_data: Dict[str, Tensor],
        phi1: Tensor,
        phi2: Tensor,
    ) -> Tensor:
        """Computes loss term for known transitions from prior knowledge.

        This implements the prior knowledge incorporation described in
        Appendix C.3 of the paper.
        """
        states = prior_data["states"]
        actions = prior_data["actions"]
        rewards = prior_data["rewards"]
        next_states = prior_data["next_states"]

        # Compute Q-values for known transitions using both parameter sets
        q1 = self.q_net.forward(states, actions, rewards, actions)
        q2 = self.q_net.forward(states, actions, rewards, actions)

        # Compute Bellman targets using known transitions
        targets = (
            rewards
            + self.gamma
            * torch.max(self.q_net(next_states, actions, rewards, actions)[0], dim=1)[0]
        )

        return F.mse_loss(q1, targets) + F.mse_loss(q2, targets)

    def initialize_prior(
        self,
        num_steps: int,
        initial_state: Tensor,
        sampling_distribution: Optional[Distribution] = None,
        prior_data: Optional[Dict[str, Tensor]] = None,
        learning_rate: float = 1e-4,
    ):
        """Implements Algorithm 2 (Prior Initialization) from the BEN paper.

        Args:
            num_steps: Number of pretraining steps (N_Pretrain in paper)
            initial_state: Tensor containing initial state [batch_size, seq_len, state_dim]
            sampling_distribution: Optional distribution over actions.
                Defaults to uniform if not provided.
            prior_data: Optional dictionary containing known transitions
            learning_rate: Learning rate for optimization
        """
        optimizer = Adam(self.parameters(), lr=learning_rate)
        batch_size = initial_state.size(0)

        # Default to uniform if no distribution provided
        rho = sampling_distribution or torch.distributions.Categorical(
            torch.ones(self.num_actions) / self.num_actions
        )

        for i in range(num_steps):
            print(f"Step {i} of prior initialization...")

            # 1. Sample action from sampling distribution rho
            action = rho.sample((batch_size,))

            # 2. Sample two independent sets of parameters and base variables
            z_ep = self.epistemic_net.sample_base(batch_size=batch_size, n=2)
            z_ep1 = z_ep[:, 0]
            z_ep2 = z_ep[:, 1]

            z_al = self.aleatoric_net.sample_base(batch_size=batch_size, n=2)
            z_al1 = z_al[:, 0]
            z_al2 = z_al[:, 1]

            # 3. Get initial Q-value
            history_actions = torch.zeros_like(action).unsqueeze(
                -1
            )  # No prior actions, shape: [batch_size, seq_len]
            history_rewards = torch.zeros_like(action).unsqueeze(
                -1
            )  # No prior rewards, shape: [batch_size, seq_len]

            q0, history_encoding, _ = self.q_net.forward(
                history_states=initial_state,
                history_actions=history_actions,
                history_rewards=history_rewards,
                current_action=action,
            )

            # 4. Compute MSBBE using two independent samples
            b1 = self.aleatoric_net.forward(z_al1, history_encoding, q0)
            b2 = self.aleatoric_net.forward(z_al2, history_encoding, q0)

            msbbe = 0.5 * ((b1 - q0) ** 2 + (b2 - q0) ** 2)

            # 5. If prior data available, incorporate it
            if prior_data is not None:
                # Transform through networks
                phi1, _ = self.epistemic_net.forward(z_ep1)
                phi2, _ = self.epistemic_net.forward(z_ep2)

                prior_loss = self.compute_prior_loss(prior_data, phi1, phi2)
                msbbe = msbbe + prior_loss

            # Optimization step
            optimizer.zero_grad()
            msbbe.backward()
            optimizer.step()

    def update_posterior(
        self,
        history_states: Tensor,
        history_actions: Tensor,
        history_rewards: Tensor,
        bootstrap_samples: Tensor,
        n_update_steps: int = 10,
        n_posterior_steps: int = 5,
        alpha_psi: float = 1e-3,  # Faster learning rate
        alpha_omega: float = 1e-4,  # Slower learning rate
    ) -> Dict[str, float]:
        """Implements Algorithm 3 (Posterior Updating) from the BEN paper.

        This function implements the nested optimization procedure described
        in Section 4.2, using two-timescale stochastic approximation to
        ensure convergence to a fixed point.

        Args:
            history_states: Trajectory states [batch_size, seq_len, state_dim]
            history_actions: Actions taken [batch_size, seq_len]
            history_rewards: Rewards received [batch_size, seq_len]
            bootstrap_samples: Bellman samples b_t [batch_size, seq_len]
            n_update_steps: Outer loop iterations (N_Update in paper)
            n_posterior_steps: Inner loop iterations (N_Posterior in paper)
            alpha_psi: Learning rate for epistemic network (faster)
            alpha_omega: Learning rate for Q-network (slower)
        """
        bootstrap_mean = bootstrap_samples.mean()
        bootstrap_std = bootstrap_samples.std()
        normalized_bootstrap_samples = (bootstrap_samples - bootstrap_mean) / (
            bootstrap_std + 1e-6
        )

        # Initialize optimizers with appropriate learning rates
        epistemic_optimizer = torch.optim.Adam(
            self.epistemic_net.parameters(), lr=alpha_psi
        )
        q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=alpha_omega)

        # Maintain moving averages for stable training
        ema_elbo = MovingAverage(0.99)
        ema_msbbe = MovingAverage(0.99)

        elbo_loss = torch.tensor(0.0)
        msbbe_loss = torch.tensor(0.0)
        for update_step in range(n_update_steps):
            # 1. Update epistemic network (faster timescale)
            for _ in range(n_posterior_steps):
                epistemic_optimizer.zero_grad()

                # Compute ELBO loss with multiple samples
                elbo_samples = []
                for _ in range(5):  # Monte Carlo samples
                    elbo = self.compute_elbo(
                        history_states,
                        history_actions,
                        history_rewards,
                        # bootstrap_samples,
                        normalized_bootstrap_samples,
                    )
                    elbo_samples.append(elbo)

                # Average ELBO samples and update
                elbo_loss = -torch.stack(elbo_samples).mean()
                ema_elbo.update(elbo_loss.item())

                elbo_loss.backward()
                epistemic_optimizer.step()

            # 2. Update Q-network (slower timescale)
            q_optimizer.zero_grad()

            # Compute MSBBE with current parameter estimates
            msbbe_loss = self.compute_msbbe(
                history_states,
                history_actions,
                history_rewards,
                history_actions[:, -1],  # Current action
                # bootstrap_samples,
                normalized_bootstrap_samples,
            )
            ema_msbbe.update(msbbe_loss.item())

            msbbe_loss.backward()
            q_optimizer.step()

            # Optional: Log training progress
            if update_step % 1 == 0:
                print(f"\nStep {update_step}:")
                print(f"  ELBO={ema_elbo.value:.6f}")
                print(f"  MSBBE={ema_msbbe.value:.6f}")

        return {
            "elbo_loss": elbo_loss.float(),
            "msbbe_loss": msbbe_loss.float(),
            "ema_elbo_loss": ema_elbo.value,
            "ema_msbbe_loss": ema_msbbe.value,
        }

    def compute_msbbe(
        self,
        history_states: Tensor,
        history_actions: Tensor,
        history_rewards: Tensor,
        current_action: Tensor,
        next_state: Tensor,
        rho: Optional[Tensor] = None,  # Sampling distribution
    ) -> Tensor:
        """Computes Mean Squared Bayesian Bellman Error:
        MSBBE(omega; ht, psi) := |B+[Q_omega](ht, at) - Q_omega(ht, at)|^2_rho

        The rho parameter represents the sampling distribution over actions
        as described in Section 4.2 of the paper.

        Args:
            history_states: [batch, seq, state_dim]
            history_actions: [batch, seq]
            history_rewards: [batch, seq]
            current_action: [batch]
            next_state: [batch, state_dim]
            reward: [batch] - Current reward
            rho: Optional weighting distribution
        """
        # Compute current Q-values
        q_values, _, _ = self.q_net.forward(
            history_states,
            history_actions,
            history_rewards,
            current_action,
        )

        # Compute Bayesian Bellman values
        bellman_values = self.compute_bayesian_bellman_operator(
            history_states,
            history_actions,
            history_rewards,
            current_action,
            next_state,
        )

        # If sampling distribution provided, weight the loss
        if rho is not None:
            loss = ((q_values - bellman_values) ** 2) * rho
            return loss.mean()

        return F.mse_loss(q_values, bellman_values)

    def compute_bayesian_bellman_operator(
        self,
        history_states: Tensor,
        history_actions: Tensor,
        history_rewards: Tensor,
        current_action: Tensor,
        next_state: Tensor,
    ) -> Tensor:
        """
        Computes B+[Q_omega](ht, at) as described in Section 4.1 of the paper.

        This implements the predictive optimal Bellman operator by:
        1. Computing current Q-value qt
        2. Sampling from base distributions
        3. Transforming through aleatoric and epistemic networks
        4. Aggregating to estimate the expected value
        """
        # Get current Q-value and history encoding
        qt, history_encoding, hidden = self.q_net.forward(
            history_states,
            history_actions,
            history_rewards,
            current_action,
        )

        # Sample from base distributions
        batch_size = history_states.size(0)
        z_al = self.aleatoric_net.sample_base(
            batch_size=batch_size,
            device=qt.device,
        )  # Aleatoric samples
        z_ep = self.epistemic_net.sample_base(
            batch_size=batch_size,
            device=qt.device,
        )

        # Transform through epistemic network to get phi
        phi, _ = self.epistemic_net.forward(z_ep)

        # Transform through aleatoric network to get Bellman samples
        bt = self.aleatoric_net.forward(z_al, history_encoding, qt)

        # The paper suggests using multiple samples and averaging
        # for more stable estimation
        n_samples = 10  # Hyperparameter for number of samples
        bellman_samples = torch.zeros(batch_size, n_samples, device=qt.device)

        for i in range(n_samples):
            z_al = self.aleatoric_net.sample_base(
                batch_size=batch_size,
                device=qt.device,
            )
            bellman_samples[:, i] = self.aleatoric_net.forward(
                z_al,
                history_encoding,
                qt,
            ).squeeze(-1)

        # Average samples to estimate expectation
        return bellman_samples.mean(dim=1, keepdim=True)

    def compute_elbo(
        self,
        history_states: Tensor,
        history_actions: Tensor,
        history_rewards: Tensor,
        bootstrap_samples: Tensor,
    ) -> Tensor:
        """
        Computes the Evidence Lower Bound as defined in Appendix C.2:
        ELBO(psi; ht, omega) := E_{z_ep~P_ep}[sum_{i=0}^{t-1} (
            |B^(-1)(bi, qi, phi)|^2 - log|delta_b B^(-1)(bi, qi, phi)| - (1/t)log p_Phi(phi)
        )]_{phi=t_psi(z_ep)}

        Args:
            history_states: Tensor of states, shape [batch_size, seq_len, state_dim]
            history_actions: Tensor of actions, shape [batch_size, seq_len]
            history_rewards: Tensor of rewards, shape [batch_size, seq_len]
            bootstrap_samples: Tensor of bootstrap samples, shape [batch_size, seq_len]
        """
        batch_size = history_states.size(0)
        seq_length = history_states.size(1)

        # Sample from base distribution
        z_ep = self.epistemic_net.sample_base(
            batch_size=batch_size,
            device=history_states.device,
        )

        # Transform through epistemic network
        phi, log_det = self.epistemic_net.forward(z_ep)

        # Compute Q-values for history
        q_values = []
        hidden = None
        for t in range(seq_length):
            qt, _, hidden = self.q_net.forward(
                history_states[:, : t + 1],
                history_actions[:, : t + 1],
                history_rewards[:, : t + 1],
                history_actions[:, t],
                hidden,
            )
            q_values.append(qt)
        q_values = torch.stack(q_values, dim=1)  # [batch_size, seq_len, 1]

        # Compute inverse Bellman operator terms
        b_inverse = self.compute_inverse_bellman(
            bootstrap_samples,
            q_values,
            phi,
        )

        # Compute log determinant of inverse Bellman operator
        log_det_bellman = self.compute_bellman_log_det(
            bootstrap_samples,
            q_values,
            phi,
        )

        # Compute log prior
        log_prior = self.compute_log_prior(phi)

        # Reconstruction term with temperature
        beta = 1.0
        recon_loss = beta * b_inverse.pow(2).sum(dim=-1)

        # Scale log determinants to similar magnitude
        log_det_scale = 1.0 / phi.shape[-1]  # Normalize by dimensionality
        flow_term = log_det_bellman - log_det
        flow_term = log_det_scale * flow_term

        # Prior term with proper scaling
        prior_term = log_prior / phi.shape[-1]

        # Combined ELBO with monitored components
        elbo = -recon_loss + flow_term + prior_term

        # print("\nELBO loss components:")
        # print(f"  -recon_loss = {-recon_loss}")
        # print(f"  flow_term = {flow_term}")
        # print(f"  prior_term = {prior_term}")

        return -elbo.mean()  # Negative since we're minimizing

    def compute_inverse_bellman(
        self,
        bootstrap_samples: Tensor,
        q_values: Tensor,
        phi: Tensor,
    ) -> Tensor:
        """
        Computes B^(-1)(bi, qi, phi) using the aleatoric network.
        This represents inverting the flow transformation.
        """
        # Use inverse autoregressive flow
        return self.aleatoric_net.inverse(bootstrap_samples, q_values, phi)

    def compute_bellman_log_det(
        self,
        bootstrap_samples: Tensor,
        q_values: Tensor,
        phi: Tensor,
    ) -> Tensor:
        """
        Computes log|delta_b B^(-1)(bi, qi, phi)|, the log determinant
        of the Jacobian of the inverse Bellman operator.
        """
        # This is handled automatically by our normalizing flows
        return self.aleatoric_net.log_det(bootstrap_samples, q_values, phi)

    def compute_log_prior(self, phi: Tensor) -> Tensor:
        """
        Computes log p_Phi(phi), the log probability under the prior.
        In BEN, we typically use a standard normal prior.

        Args:
            phi: Model parameters [batch_size, param_dim]

        Returns:
            Log probability under standard normal prior
        """
        # Debug prints
        # print("\nCompute log prior method:")
        # print(f"  phi shape: {phi.shape}")
        # print(f"  phi values: {phi}")
        # print(f"  phi grad_fn: {phi.grad_fn}")

        # Add numerical stability checks
        if torch.isnan(phi).any():
            print("  NaN values detected in phi!")
            # Try to trace back where NaNs originated
            last_valid = None
            for name, param in self.named_parameters():
                if torch.isnan(param).any():
                    print(f"  NaN found in parameter {name}")
                else:
                    last_valid = (name, param)

        # Add numerical stability
        phi_stable = phi.clamp(-1e2, 1e2)  # Prevent extreme values

        # Compute log prob with better error handling
        try:
            return torch.distributions.Normal(0, 1).log_prob(phi_stable).sum(-1)
        except ValueError as e:
            print("\nError computing log prob. Phi stats:")
            print(f"  Min: {phi_stable.min()}, Max: {phi_stable.max()}")
            print(f"  Mean: {phi_stable.mean()}, Std: {phi_stable.std()}")
            raise e
