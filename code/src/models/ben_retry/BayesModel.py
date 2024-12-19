import torch
from torch import nn

from .BayesNormalizingFlow import AleatoricFlow, BayesNormalizingFlow
from .ConditionerMLP import ConditionerMLP
from .EmpiricalTarget import EmpiricalTarget


class BayesModel(nn.Module):
    def __init__(
        self,
        conditioner_MLP: ConditionerMLP,
        epistemic_flow: BayesNormalizingFlow,
        aleatoric_flows: AleatoricFlow,
        target_distribution: EmpiricalTarget,
    ):
        """Constructor for BayesModel

        :arg conditioner_MLP: MLP that takes in the input and outputs the parameters of the base distribution
        :arg epistemic_flow: Normalizing flow that takes in the parameters of the base distribution and outputs the parameters of the target distribution
        :arg aleatoric_flows: Normalizing flow that takes in the parameters of the base distribution and outputs the parameters of the aleatoric distribution
        """
        super().__init__()
        self.conditioner_MLP = conditioner_MLP
        self.epistemic_flow = epistemic_flow
        self.aleatoric_flows = aleatoric_flows
        self.target_distribution = target_distribution
        self.add_module("conditioner_MLP", self.conditioner_MLP)
        self.add_module("epistemic_flow", self.epistemic_flow)
        self.add_module("aleatoric_flows", self.aleatoric_flows)
        # helper values
        # self.add_module("target_distribution", self.target_distribution)

    def _enforce_single_conformable(self, x, phi):
        """Conform the input to the correct shape of the form (batch_samples_phi, batch_samples_x, -1)"""
        assert x.dim() == 4 and phi.dim() == 4
        batch_x = x.shape[1]
        batch_phi = phi.shape[0]
        if x.shape[0] != phi.shape[0]:
            x = x.repeat(phi.shape[0], 1, 1, 1)
        if x.shape[1] != phi.shape[1]:
            phi = phi.repeat(1, x.shape[1], 1, 1)
        return x, phi, batch_x, batch_phi

    def _enforce_conformable(self, x, phi):
        """Conform the input to the correct shape of the form (batch_samples_phi, batch_samples_x, -1)"""
        if isinstance(x, torch.Tensor):
            return self._enforce_single_conformable(x, phi)
        elif isinstance(x, list) and all([isinstance(x_i, torch.Tensor) for x_i in x]):
            vals = []
            for i in range(len(x)):
                vals.append(self._enforce_single_conformable(x[i], phi))
            return vals

    def sample_params(self, num_samples=1):
        """Samples parameters for the base distribution and the aleatoric flow"""
        phi, log_p = self.epistemic_flow.sample(num_samples=num_samples)
        return phi, log_p

    def sample_b(
        self,
        q_network,
        current_state,
        init_action,
        reward,
        hidden_state,
        current_action,
        phi,
        z_al,
    ):
        self.q_network = q_network
        inputs = torch.cat(
            [
                torch.tensor(current_state).view(1, -1),
                torch.tensor(init_action).view(1, -1),
                torch.tensor(reward).view(1, -1),
            ],
            dim=-1,
        )
        q0, next_hidden = q_network(inputs, hidden_state)
        q0.select(dim=-1, index=current_action).view(1, -1, 1, 1).retain_grad()
        q0, phi, _, _ = self._enforce_conformable(q0, phi)
        phi = torch.cat([q0, next_hidden, phi], dim=-1)
        b = self.aleatoric_flows.forward(z_al, phi, self.conditioner_MLP)
        return b, q0

    def sample(
        self,
        hidden_state,
        current_state,
        action,
        rewards,
        q_network,
        envs,
        num_samples=1,
        num_samples_per_param=1,
    ):
        """Samples from flow-based approximate distribution

        Args:
                                    hidden_state: hidden state
                                    current_state: current state
                                    action: action
                                    rewards: rewards
                                    q_network: q_network
                                    envs: environment

        Returns:
                    Samples

        """
        phi, log_p = self.sample_params(num_samples)
        q_val = self.target_distribution.forward(
            hidden_state, current_state, action, rewards, q_network, envs
        )
        q_val = q_val.unsqueeze(0).view(-1, 1, 1, 1).repeat(num_samples, 1, 1, 1)
        phi = torch.cat([phi, q_val], dim=-1)
        b = self.aleatoric_flows.sample(
            phi, self.conditioner_MLP, num_samples_per_param
        )
        # log_p -= log_q,
        return b  # ,log_p

    def forward_and_log_det(self, z_al, phi, conditioner_network):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
                                        z_al: Batch in the latent space

        Returns:
                                        Batch in the space of the target distribution,
                                        log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z_al), device=z_al.device)
        b, log_d = self.aleatoric_flows.forward_and_log_det(
            z_al, phi, conditioner_network
        )
        log_det += log_d
        return b, log_det

    def forward(self, z_al, phi, conditioner_network):
        """Transforms latent variable eps to the flow variable b

        Args:
                                        z_al: Batch in the latent space
                                        phi: Sample of the epistemic flow for weights

        Returns:
                                        Batch in the space of the target distribution
        """
        b, _ = self.aleatoric_flows.forward_and_log_det(z_al, phi, conditioner_network)
        return b

    def inverse(self, b):
        """Transforms flow variable b to the latent variable z

        Args:

                                        b: Batch in the space of the target distribution

        Returns:
                                        Batch in the latent space
        """
        phi = self.epistemic_flow.sample()
        z_al, _ = self.aleatoric_flows.inverse(b, phi, self.conditioner_MLP)
        return z_al

    def inverse_and_log_det(self, b, phi):
        """transforms flow variable b to the latent variable z and
        computes log determinant of the jacobian

        args:
                                        b: batch in the space of the target distribution

        returns:
                                        batch in the latent space, log determinant of the
                                        jacobian
        """
        log_det = torch.zeros(len(b), device=b.device)
        z_al, log_d = self.aleatoric_flows.inverse(b, phi, self.conditioner_MLP)
        log_det += log_d
        return z_al, log_det

    def reverse_kld(
        self,
        hidden_state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        q_network: torch.nn.Module,
        time_period: int,
        num_samples=1,
        num_samples_per_param=1,
        beta=1.0,
        score_fn=True,
    ):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
                                        num_samples: Number of samples to draw from base distribution for epistemic
                                        num_samples_per_param: Number of samples to draw from aleatoric flow for each base distribution sample
                                        beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
                                        score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
                                        Estimate of the reverse KL divergence averaged over latent samples
        """
        # Reshape as  appropriate
        phi, _ = self.epistemic_flow.sample(num_samples)

        phi = phi.view(-1, 1, 1, phi.shape[-1])

        prior_log_prob = self.epistemic_flow.prior.log_prob(phi).view(-1, 1, 1, 1)
        # log_q = prior_log_prob
        # log_q += log_q_

        # z_al, log_det = self.aleatoric_flows.set_params(params).sample(
        #    num_samples_per_param
        # )
        b_vals, q_values = self.target_distribution(
            hidden_state, next_state, action, rewards, q_network
        )
        b_vals = b_vals.view(1, -1, 1, 1)
        phi = torch.cat([phi, q_values.view(1, -1, 1, 1)], dim=-1)
        b_vals, phi, _, _ = self._enforce_conformable(b_vals, phi)
        b_det = self.aleatoric_flows.forward_kld(
            b_vals, phi, self.conditioner_MLP, time_period
        )

        # if not score_fn:
        #     z_al_ = z_al
        #     log_q = torch.zeros(len(z_al_), device=z_al_.device)
        #     nf.utils.set_requires_grad(self, False)
        #     z_al_, log_det = self.aleatoric_flows.set_params(params).inverse(z_al_)
        #     log_q += log_det
        #     nf.utils.set_requires_grad(self, True)
        log_p = self.aleatoric_flows.log_prob(b_vals, phi, self.conditioner_MLP)
        return torch.mean(
            b_det - 1 / (time_period + 1) * prior_log_prob
        ) - beta * torch.mean(log_p)

    def ELBO(self, b, q_vals, hidden_state, time_period: int):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
                                        b: Batch sampled from target distribution

        Returns:
                                        Estimate of forward KL divergence averaged over batch
        """

        log_q = torch.zeros(len(b), device=b.device).clone().view(-1, 1)
        z_al = b
        phi = self.epistemic_flow.sample()
        prior = self.epistemic_flow.prior.log_prob(phi).view(-1, 1)
        phi = phi.view(-1, 1, 1, phi.shape[-1])
        q_vals = q_vals.view(1, 1, 1, -1)
        hidden_state = hidden_state.view(1, 1, 1, -1)
        phi = torch.cat([phi, q_vals, hidden_state], dim=-1)
        phi.retain_grad()
        z_al, phi, _, _ = self._enforce_conformable(z_al, phi)
        z_al, log_det = self.aleatoric_flows.inverse_and_log_det(
            z_al, phi, self.conditioner_MLP
        )
        log_p = self.aleatoric_flows.base_dist.log_prob(z_al).view(1, 1, 1, -1)
        log_q += log_det.view(-1, 1)
        # self.base_dist.zero_grad()

        return -log_p - torch.mean(log_q) - 1 / (time_period + 1) * prior

    def log_prob(self, b):
        """Get log probability for batch

        Args:
                                        b: Batch

        Returns:
                                        log probability
        """
        phi = self.epistemic_flow.sample()
        log_q = (
            torch.zeros(len(b), dtype=torch.double, device=b.device).view(-1, 1).clone()
        )
        log_det_ = (
            torch.zeros(len(b), dtype=torch.double, device=b.device).view(-1, 1).clone()
        )
        z_al = b
        z_al, log_det = self.aleatoric_flows.inverse(z_al, phi, self.conditioner_MLP)
        log_det_ += log_det.view(-1, 1)
        # self.base_dist.zero_grad()
        log_q += log_det_.double() + self.base_dist_aleatoric.log_prob(
            z_al
        ).double().view(-1, 1)
        return log_q  # torch.max(log_q, -1e-4 * torch.ones_like(log_q))

    def log_posterior(self, b, time_period):
        prior = torch.distributions.MultivariateNormal(
            torch.ones((1, 2)) * 1.1, 3.0 * torch.eye(2)
        )
        log_q = super().log_prob(b)
        if prior is not None:
            log_q += 1 / (time_period + 1) * prior.log_prob(b)

        return log_q
