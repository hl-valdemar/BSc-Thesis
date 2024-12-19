import normflows as nf
import torch
import torch.nn as nn

from .ConditionerMLP import ConditionerMLP
from .flows import (
    MADE,
    AbsFlow,
    Autoregressive,
    ConditionalMaskedAffineAutoregressive,
    ConditionalReverse,
    MaskedFeedforwardBlock,
    MaskedLinear,
    MaskedResidualBlock,
)

ClassRequiresParams = (
    Autoregressive
    | MADE
    | ConditionalMaskedAffineAutoregressive
    | ConditionerMLP
    | ConditionalReverse
    | nn.Linear
    | MaskedLinear
    | MaskedResidualBlock
    | MaskedFeedforwardBlock
    | AbsFlow
)


def isScalar(det):
    if det.dim() == 0:
        return True
    else:
        return False


class BayesNormalizingFlow(nf.flows.Flow):
    def __init__(self, base_dist, flows, prior, target_dist=None):
        """Constructor

        Args:
          base_dist: Base distribution
          flows: List of flows
          target_dist: Target distribution
          prior:  Prior for epistemic
        """
        super().__init__()
        self.base_dist = base_dist
        self.flows = []
        self.flows = nn.ModuleList(flows)
        self.target_dist = target_dist
        self.prior = prior

    def sample(self, num_samples=1):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.base_dist(num_samples)
        log_q = torch.atleast_2d(log_q).view(-1, 1, 1, 1)
        x = z
        for i, flow in enumerate(self.flows):
            x, log_det = flow(x)
            try:
                log_q -= log_det
            except:
                k = 1
        return x  # log_q

    def forward_and_log_det(self, z):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det

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

    def forward(self, z):
        """Transforms latent variable z to the flow variable x

        Args:
                                                                        z: Batch in the latent space

        Returns:
                                                                        Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def inverse(self, x):
        """Transforms flow variable x to the latent variable z

        Args:
                                                                        x: Batch in the space of the target distribution

        Returns:
                                                                        Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        """transforms flow variable x to the latent variable z and
        computes log determinant of the jacobian

        args:
                                                                        x: batch in the space of the target distribution

        returns:
                                                                        batch in the latent space, log determinant of the
                                                                        jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def reverse_kld(
        self,
        num_samples=1,
        beta=1.0,
        score_fn=True,
    ):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
                                                                        num_samples: Number of samples to draw from base distribution
                                                                        beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
                                                                        score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
                                                                        Estimate of the reverse KL divergence averaged over latent samples
        """

        z, log_q_ = self.base_dist(num_samples)
        prior_log_prob = self.prior.log_prob(z)
        # prior_log_prob = -F.gaussian_nll_loss(
        #     input=torch.zeros((1,)),
        #     target=z,
        #     var=1 * torch.ones((1,)),
        # )
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            nf.utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.base_dist.log_prob(z_)
            nf.utils.set_requires_grad(self, True)
        log_p = self.target_dist.log_prob(z.reshape(-1, *z.shape()[2:])).reshape_as(
            log_q
        )
        assert torch.abs(log_p).max() < 50, "log_p diverged"
        assert torch.abs(log_q).max() < 50, "log_q diverged"
        assert torch.abs(prior_log_prob).max() < 50, "prior_log_prob diverged"
        return torch.mean(log_q + prior_log_prob) - beta * torch.mean(log_p)

    def forward_kld(self, x, time_period):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
                                                                        x: Batch sampled from target distribution

        Returns:
                                                                        Estimate of forward KL divergence averaged over batch
        """

        log_q = torch.zeros(len(x), device=x.device).clone().view(-1, 1)
        z = x

        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det.view(-1, 1)
        log_q += self.base_dist.log_prob(z).view(-1, 1)
        # self.base_dist.zero_grad()

        prior_log_prob = self.prior.log_prob(z).view(-1, 1)
        # prior_log_prob = -F.gaussian_nll_loss(
        #     input=torch.ones((1,)) * 1.1,
        #     target=z,
        #     var=1.0 * (xmax - xmin) * torch.ones((1,)),
        # )
        return (
            -torch.mean(log_q) + 1 / (1 + time_period) * prior_log_prob
        )  # prior_log_prob)

    def log_prob(self, x):
        """Get log probability for batch

        Args:
                                                                        x: Batch

        Returns:
                                                                        log probability
        """
        log_q = (
            torch.zeros(len(x), dtype=torch.double, device=x.device).view(-1, 1).clone()
        )
        log_det_ = (
            torch.zeros(len(x), dtype=torch.double, device=x.device).view(-1, 1).clone()
        )
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            # self.flows[i].zero_grad()
            z, log_det = self.flows[i].inverse(z)
            log_det_ += log_det.view(-1, 1)
        self.base_dist.zero_grad()
        log_q += log_det_.double() + self.base_dist.log_prob(z).double().view(-1, 1)
        return log_q  # torch.max(log_q, -1e-4 * torch.ones_like(log_q))

    # def log_posterior(self, x):
    #     prior = torch.distributions.MultivariateNormal(
    #         torch.ones((1, 2)) * 1.1, 0.6 * (xmax - xmin) * torch.eye(2)
    #     )
    #     log_q = super().log_prob(x)
    #     if prior is not None:
    #         log_q += prior.log_prob(x)
    #     return log_q
    def log_prob_posterior(self, z):
        log_prob = self.log_prob(z)
        log_posterior_prob = log_prob + self.prior.log_prob(z)
        return log_posterior_prob


class AleatoricFlow(BayesNormalizingFlow):
    def __init__(self, base_dist, flows, target_dist=None):
        """Constructor

        Args:
          base_dist: Base distribution
          flows: List of flows
          target_dist: Target distribution
          :param prior:
        """
        super().__init__(base_dist, flows, prior=None)
        self.base_dist = base_dist
        self.flows = nn.ModuleList(flows)
        self.target_dist = target_dist
        # self.feeder_aleatoric = ParameterFeeder(self)

    def sample(
        self,
        phi,
        conditioner_network,
        q_network,
        inputs_for_q,
        hidden_state,
        num_samples=1,
    ):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z_al, log_q = self.base_dist(num_samples)
        log_q = log_q.view(-1, 1, 1, 1)
        b = z_al
        for flow in self.flows:
            if isinstance(flow, ClassRequiresParams) or (
                isinstance(flow, nn.Module)
                and len([p for p in flow.parameters() if p.requires_grad]) > 0
                and not isinstance(flow, nf.flows.Reverse)
            ):
                b, log_det = flow(
                    b, phi, conditioner_network, q_network, inputs_for_q, hidden_state
                )
            else:
                b, log_det = flow(b)
            # log_q -= log_det
        return b  # , log_q

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
        b = z_al
        for flow in self.flows:
            if isinstance(flow, ClassRequiresParams) or (
                isinstance(flow, nn.Module)
                and len([p for p in flow.parameters() if p.requires_grad]) > 0
                and not isinstance(flow, nf.flows.Reverse)
            ):
                b, log_d = flow(b, phi, conditioner_network)
            else:
                b, log_d = flow(b)
            log_det += log_d
        return b, log_det

    def forward(
        self,
        z_al,
        phi,
        conditioner_network,
        q_network=None,
        inputs=None,
        hidden_state=None,
    ):
        """Transforms latent variable z_al to the flow variable x

        Args:
                                        z_al: Batch in the latent space

        Returns: [
                                                                                                                                                                        params.numel()
                                                                                                                                                                        for flow in flows_aleatoric
                                                                                                                                                                        for params in flow.parameters()
                                                                                                                                                                        if params.requires_grad
                                                                                                                                        ]
                                        Batch in the space of the target distribution
        """
        b = z_al
        for flow in self.flows:
            if isinstance(flow, ClassRequiresParams) or (
                isinstance(flow, nn.Module)
                and len([p for p in flow.parameters() if p.requires_grad]) > 0
                and not isinstance(flow, nf.flows.Reverse)
            ):
                b, _ = flow(
                    b, phi, conditioner_network, q_network, inputs, hidden_state
                )
            else:
                b, _ = flow(b)
            b.retain_grad()
        return b

    def inverse(self, b, phi, conditioner_network):
        """Transforms flow variable b to the latent variable z_al

        Args:
                                        b: Batch in the space of the target distribution
                                        phi: output of epistemic network
                                        conditioner_network: network that takes phi as input and outputs parameters of aleatoric flow

        Returns:
                                        Batch in the latent space
        """
        z_al = b
        for i in range(len(self.flows) - 1, -1, -1):
            if isinstance(self.flows[i], ClassRequiresParams) or (
                isinstance(self.flows[i], nn.Module)
                and len([p for p in self.flows[i].parameters() if p.requires_grad]) > 0
                and not isinstance(self.flows[i], nf.flows.Reverse)
            ):
                z_al, _ = self.flows[i].inverse(z_al, phi, conditioner_network)
            else:
                z_al, _ = self.flows[i].inverse(z_al)
        return z_al

    def inverse_and_log_det(self, b, phi, conditioner_network):
        """transforms flow variable b to the latent variable z_al and
        computes log determinant of the jacobian

        args:
                                        b: batch in the space of the target distribution

        returns:
                                        batch in the latent space, log determinant of the
                                        jacobian
        """
        log_det = torch.zeros_like(b, device=b.device)
        z_al = b
        for i in range(len(self.flows) - 1, -1, -1):
            if isinstance(self.flows[i], ClassRequiresParams) or (
                isinstance(self.flows[i], nn.Module)
                and len([p for p in self.flows[i].parameters() if p.requires_grad]) > 0
                and not isinstance(self.flows[i], nf.flows.Reverse)
            ):
                z_al, log_d = self.flows[i].inverse(z_al, phi, conditioner_network)
            else:
                z_al, log_d = self.flows[i].inverse(z_al)
            log_det += log_d
        return z_al, log_det

    def log_prob(self, b, phi, conditioner_network):
        """Get log probability for batch

        Args:
                                        b: Batch

        Returns:
                                        log probability
        """
        b, phi, _, _ = super()._enforce_conformable(b, phi)
        log_q = torch.zeros_like(b)
        log_det_ = torch.zeros_like(b)
        z_al = b
        for i in range(len(self.flows) - 1, -1, -1):
            # Could get rid of by allowing all to take and then just discard if not needed.
            if isinstance(self.flows[i], ClassRequiresParams) or (
                isinstance(self.flows[i], nn.Module)
                and len([p for p in self.flows[i].parameters() if p.requires_grad]) > 0
                and not isinstance(self.flows[i], nf.flows.Reverse)
            ):
                z_al, log_det = self.flows[i].inverse(z_al, phi, conditioner_network)
            else:
                z_al, log_det = self.flows[i].inverse(z_al)
            # self.flows[i].zero_grad()
            log_det_ += log_det
        # self.base_dist.zero_grad()
        assert torch.abs(log_det_).max() < 500, "log_det_ is too large"
        log_q += log_det_.double().view_as(log_q)
        log_q = log_q
        log_p = self.base_dist.log_prob(z_al.flatten(0, 1))
        assert torch.abs(log_p).max() < 1e6, "log_p is too large"
        log_q += log_p.reshape_as(log_q)
        return log_q

    def forward_kld(self, b, phi, conditioner_network, time_period):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
                                        b: Batch sampled from target distribution

        Returns:
                                        Estimate of forward KL divergence averaged over batch
        """

        self.phi = phi

        z_al = b
        z_al, b, _, _ = super()._enforce_conformable(z_al, phi)
        log_q = torch.zeros_like(z_al, dtype=torch.double, device=b.device)
        # split_params = self.feeder_aleatoric.feed_parameters(self.params)
        # param_ind = len(split_params) - 1
        for i in range(len(self.flows) - 1, -1, -1):
            if isinstance(self.flows[i], ClassRequiresParams) or (
                isinstance(self.flows[i], nn.Module)
                and len(
                    [par for par in self.flows[i].parameters() if par.requires_grad]
                )
                > 0
                and not isinstance(self.flows[i], nf.flows.Reverse)
            ):
                # param = split_params[param_ind]
                z_al, log_det = self.flows[i](z_al, phi, conditioner_network)
                # param_ind -= 1
            else:
                z_al, log_det = self.flows[i].inverse(z_al)
            if isScalar(log_det):
                log_q += log_det
            elif log_det.dim() == 2 and log_det.shape[0] * log_det.shape[1] == 1:
                log_q += log_det.view(-1, log_det.shape[0], log_det.shape[1]).expand_as(
                    log_q
                )
            else:
                log_q += log_det.view_as(log_q)
        log_p = self.base_dist.log_prob(z_al)
        log_q += log_p.view(-1, 1, 1, 1).expand_as(log_q)
        self.base_dist.zero_grad()
        prior_log_prob = self.prior.log_prob(z_al)
        # prior_log_prob = -F.gaussian_nll_loss(
        #     input=torch.ones((1,)) * 1.1,
        #     target=z_al,
        #     var=0.6 * (xmax - xmin) * torch.ones((1,)),
        # )
        return -torch.mean(log_q) + 1 / (1 + time_period) * prior_log_prob
