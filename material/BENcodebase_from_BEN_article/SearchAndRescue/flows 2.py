import itertools
import math

import normflows as nf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from normflows.flows import Flow
from normflows.utils import tile
from torch.distributions import Normal

from ConditionerMLP import ConditionerMLP


class SliceFlow(nf.flows.Flow):
    """Changes dims from 2d to 1d in the backward direction"""

    def __init__(self):
        super().__init__()

    def inverse(self, x):
        log_det = torch.zeros_like(x)

        # Inverse Surjective slice layer 1d->2d (Generate other output from Gaussian decoder):
        xu = x
        mu = torch.zeros_like(x)
        sig = 0.1 * torch.ones_like(x)
        decoder = Normal(loc=mu, scale=sig)
        xl = decoder.sample()
        z = torch.cat((xu, xl), dim=-1)

        return z, log_det

    def forward(self, z):
        # Surjective slice layer 2d->1d (take upper input from MAP as output):
        xu, zl = torch.split(z, 1, dim=-1)
        return xu, torch.zeros_like(xu)


class AbsFlow(nf.flows.Flow):
    """One dimensional abs surjective flow"""

    def __init__(self, K, conditioner_input_dim=2, hidden_dim=2, device="cuda:0"):
        super().__init__()
        self.conditioners = []
        self.K = K
        for i in range(K):
            self.conditioners.append(ConditionerMLP(conditioner_input_dim, hidden_dim, 2, device=device).to(device))
        self.conditioners = nn.ModuleList(self.conditioners)
        self.conditioner_input = torch.zeros(
            conditioner_input_dim,
        )

    def set_param(self, param):
        self.param = param

    def inverse(self, x, phi, conditioner_net, q_network=None, inputs_for_q=None, hidden_state=None):
        if q_network is not None:
            params = conditioner_net(phi, q_network, inputs_for_q, hidden_state)
        else:
            params = conditioner_net(phi)
        conditioner_input, _ = torch.split(params, [x.shape[-1] + 1, params[0, 0, 0, 1:].numel() - x.shape[-1]], dim=-1)
        xi = x
        log_det = torch.zeros_like(x)
        for k in range(self.K - 1):
            nu = self.conditioners[k](conditioner_input)
            a, b = torch.split(nu, 1, dim=-1)
            log_det = log_det + a - math.log(2)
            a = torch.exp(a)
            xi = torch.einsum("...bi,...bi->...bi", xi, a) + b
            xi = torch.abs(xi)
        nu = self.conditioners[self.K - 1](conditioner_input)
        a, b = torch.split(nu, 1, dim=-1)
        log_det = log_det + a - math.log(2)
        a = torch.exp(a)
        z = torch.einsum("...bi,...bi->...bi", xi, a) + b
        return z, log_det

    def forward(self, z, phi, conditioner_net, q_network=None, inputs_for_q=None, hidden_state=None):
        if q_network is not None:
            q_val = q_network(inputs_for_q, hidden_state)
            param = conditioner_net(phi, q_network, inputs_for_q, hidden_state)
        else:
            param = conditioner_net(phi)
        conditioner_input, _ = torch.split(param, [2 * (z.shape[-1]), param[0, 0, :].numel() - 2 * z.shape[-1]], dim=-1)
        "Doesn't return log dets as this direction will only be used for sampling"
        zi = z
        # if zi.dim() == 1 or zi.shape[0] == 1:

        nu = self.conditioners[self.K - 1](conditioner_input)
        a, b = torch.split(nu, 1, dim=-1)
        a = torch.exp(-a)
        zi = zi

        zi = torch.einsum("...bij,...bij->...bij", zi - b, a)

        probs = torch.ones_like(zi) * 0.5

        for k in reversed(range(self.K - 1)):
            nu = self.conditioners[k](conditioner_input)
            a, b = torch.split(nu, 1, dim=-1)
            a = torch.exp(-a)
            zi = torch.einsum("...bij,...bij->...bij", zi - b, a)
            s = 2 * (torch.bernoulli(probs) - 0.5)
            zi = torch.einsum("...bij,...bij->...bij", zi, s)
        x = zi
        _log_det = torch.zeros_like(x).sum(-1, keepdim=True)  # Filler
        return x, _log_det.mean()

    def get_parameters(self):
        parameters = []
        for conditioner in self.conditioners:
            parameters += conditioner.parameters()
        return parameters

    def zero_grad(self):
        for conditioner in self.conditioners:
            conditioner.zero_grad()


class _Linear(Flow):
    """Abstract base class for linear transforms that parameterize a weight matrix."""

    def __init__(self, features, using_cache=False):
        super().__init__()

        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))

        # Caching flag and values.
        self.using_cache = using_cache
        self.cache = nf.flows.mixing._LinearCache()

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            logabsdet = self.cache.logabsdet * torch.ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs)

    def _check_forward_cache(self):
        if self.cache.weight is None and self.cache.logabsdet is None:
            self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()

        elif self.cache.weight is None:
            self.cache.weight = self.weight()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            logabsdet = (-self.cache.logabsdet) * torch.ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs)

    def _check_inverse_cache(self):
        if self.cache.inverse is None and self.cache.logabsdet is None:
            (
                self.cache.inverse,
                self.cache.logabsdet,
            ) = self.weight_inverse_and_logabsdet()

        elif self.cache.inverse is None:
            self.cache.inverse = self.weight_inverse()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def train(self, mode=True):
        if mode:
            # If training again, invalidate cache.
            self.cache.invalidate()
        return super().train(mode)

    def use_cache(self, mode=True):
        self.using_cache = mode

    def weight_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # and its logabsdet together.
        return self.weight(), self.logabsdet()

    def weight_inverse_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # inverse and weight matrix logabsdet together.
        return self.weight_inverse(), self.logabsdet()

    def forward_no_cache(self, inputs):
        """Applies `forward` method without using the cache."""
        raise NotImplementedError()

    def inverse_no_cache(self, inputs):
        """Applies `inverse` method without using the cache."""
        raise NotImplementedError()

    def weight(self):
        """Returns the weight matrix."""
        raise NotImplementedError()

    def weight_inverse(self):
        """Returns the inverse weight matrix."""
        raise NotImplementedError()

    def logabsdet(self):
        """Returns the log absolute determinant of the weight matrix."""
        raise NotImplementedError()


class _LULinear(_Linear):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)

        self.eps = eps

        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        n_triangular_entries = ((features - 1) * features) // 2

        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)

        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        # The diagonal of L is taken to be all-ones without loss of generality.
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0

        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag

        return lower, upper

    def forward_no_cache(self, inputs):
        """
        Cost:

        ```
                                        output = O(D^2N)
                                        logabsdet = O(D)
        ```

        where:

        ```
                                        D = num of features
                                        N = num of inputs
        ```
        """
        lower, upper = self._create_lower_upper()
        outputs = F.linear(inputs, upper)
        outputs = F.linear(outputs, lower, self.bias)
        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """
        Cost:

        ```
                                        output = O(D^2N)
                                        logabsdet = O(D)
        ```

        where:

        ```
                                        D = num of features
                                        N = num of inputs
        ```
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        try:
            outputs = torch.linalg.solve_triangular(lower, outputs.t(), upper=False, unitriangular=True)
            outputs = torch.linalg.solve_triangular(upper, outputs, upper=True, unitriangular=False)
        except:
            outputs, _ = torch.triangular_solve(outputs.t(), lower, upper=False, unitriangular=True)
            outputs, _ = torch.triangular_solve(outputs, upper, upper=True, unitriangular=False)
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def weight(self):
        """
        Cost:

        ```
                                        weight = O(D^3)
        ```

        where:

        ```
                                        D = num of features
        ```
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """
        Cost:

        ```
                                        inverse = O(D^3)
        ```

        where:

        ```
                                        D = num of features
        ```
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features)
        lower_inverse, _ = torch.trtrs(identity, lower, upper=False, unitriangular=True)
        weight_inverse, _ = torch.trtrs(lower_inverse, upper, upper=True, unitriangular=False)
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """
        Cost:

        ```
                                        logabsdet = O(D)
        ```

        where:

        ```
                                        D = num of features
        ```
        """
        return torch.sum(torch.log(self.upper_diag))


class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        """Constructor

        Args:
          shape: Shape of the coupling layer
          scale: Flag whether to apply scaling
          shift: Flag whether to apply shift
          logscale_factor: Optional factor which can be used to control the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("s", torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("t", torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        # !!! Note: This is a change I made.
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        # !!! Note: This is a change I made.
        log_det = -prod_batch_dims * (-self.s.mean(dim=[0, 1], keepdim=True))
        return z_, log_det


class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_dims = []
        self.data_dep_init_done_cpu = torch.tensor(0.0)
        self.register_buffer("data_dep_init_done", self.data_dep_init_done_cpu)

    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.0:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().forward(z)

    def inverse(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.0)
        return super().inverse(z)


class LeakyReLU1d(nf.flows.Flow):
    def __init__(self, slope):
        self.slope = slope
        super().__init__()

    def inverse(self, x):
        log_det = torch.zeros_like(x)
        log_det = log_det.unsqueeze(-1)
        log_det[x < 0] = math.log(self.slope)
        leaky = nn.LeakyReLU(negative_slope=self.slope)
        z = leaky(x)
        return z, log_det

    def forward(self, z):
        "Doesn't return log dets as this direction will only be used for sampling"
        leaky = nn.LeakyReLU(negative_slope=1 / self.slope)
        x = leaky(z)
        return x


class ConditionalReverse(Flow):
    """
    Switches the forward transform of a flow layer with its inverse and vice versa
    """

    def __init__(self, flow):
        """Constructor

        Args:
          flow: Flow layer to be reversed
        """
        super().__init__()
        self.flow = flow

    def forward(self, z, params, conditioner_network, q_network=None, inputs_for_q=None, hidden_state=None):
        return self.flow.inverse(z, params, conditioner_network, q_network, inputs_for_q, hidden_state)

    def inverse(self, z, params, conditioner_network, q_network=None, inputs_for_q=None, hidden_state=None):
        return self.flow.forward(z, params, conditioner_network, q_network, inputs_for_q, hidden_state)


class Composite(Flow):
    """
    Composes several flows into one, in the order they are given.
    """

    def __init__(self, flows):
        """Constructor

        Args:
          flows: Iterable of flows to composite
        """
        super().__init__()
        self._flows = nn.ModuleList(flows)

    @staticmethod
    def _cascade(inputs, funcs):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = torch.zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs):
        funcs = self._flows
        return self._cascade(inputs, funcs)

    def inverse(self, inputs):
        funcs = (flow.inverse for flow in self._flows[::-1])
        return self._cascade(inputs, funcs)


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims, keepdim=True)


class Autoregressive(Flow):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    **NOTE** Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """

    def __init__(self, autoregressive_net):
        super(Autoregressive, self).__init__()
        self.autoregressive_net = autoregressive_net

    def forward(
        self, inputs, params, conditioner_net, q_network=None, inputs_for_q=None, hidden_state=None, context=None
    ):
        autoregressive_params = self.autoregressive_net(
            inputs, context, params, conditioner_net, q_network, inputs_for_q, hidden_state
        )
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(
        self, inputs, params, conditioner_net, q_network=None, inputs_for_q=None, hidden_state=None, context=None
    ):
        self.params = params
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(
                outputs, context, params, conditioner_net, q_network, inputs_for_q, hidden_state
            )
            outputs, logabsdet = self._elementwise_inverse(inputs, autoregressive_params)
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()


class MaskedAffineAutoregressive(nf.flows.affine.autoregressive.Autoregressive):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        self.features = features
        made = nf.nets.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super(MaskedAffineAutoregressive, self).__init__(made)

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 2.0) + 1e-3
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = sum_except_batch(log_scale, num_batch_dims=2)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 2.0) + 1e-3
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -sum_except_batch(log_scale, num_batch_dims=2)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            autoregressive_params.shape[0], -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0:1, 0:1]
        shift = autoregressive_params[..., 0:1, 1:2]
        return unconstrained_scale, shift


class ConditionalMaskedAffineAutoregressive(Autoregressive):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        self.features = features
        made = MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super(ConditionalMaskedAffineAutoregressive, self).__init__(made)

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 2.0) + 1e-3
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = sum_except_batch(log_scale, num_batch_dims=2)

        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 2.0) + 1e-3
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -sum_except_batch(log_scale, num_batch_dims=2)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            autoregressive_params.shape[0], -1, self.features, self._output_dim_multiplier()
        )
        unconstrained_scale = autoregressive_params[..., 0:1, 0:1]
        shift = autoregressive_params[..., 0:1, 1:2]
        return unconstrained_scale, shift


def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


class LULinearPermute(Flow):
    """
    Fixed permutation combined with a linear transformation parametrized
    using the LU decomposition, used in https://arxiv.org/abs/1906.04032
    """

    def __init__(self, num_channels, identity_init=True):
        """Constructor

        Args:
          num_channels: Number of dimensions of the data
          identity_init: Flag, whether to initialize linear transform as identity matrix
        """
        # Initialize
        super().__init__()

        # Define modules
        self.permutation = nf.flows.mixing._RandomPermutation(num_channels)
        self.linear = _LULinear(num_channels, identity_init=identity_init)

    def forward(self, z):
        batch_dims_phi = z.shape[0]
        batch_dims_x = z.shape[1]
        log_det = torch.zeros(
            (
                batch_dims_phi,
                batch_dims_x,
                z.shape[-2],
                z.shape[-1],
            ),
            dtype=torch.double,
        )
        # Ignore batch dimension:
        for i, j in itertools.product(range(batch_dims_phi), range(batch_dims_x)):
            (z[i, j, ...], log_det[i, j]) = self.linear.inverse(z[i, j, ...])
            (z[i, j, ...], _) = self.permutation.inverse(z[i, j, ...])
        return z, sum_except_batch(log_det, num_batch_dims=2)

    def inverse(self, z):
        batch_dims_phi = z.shape[0]
        batch_dims_x = z.shape[1]
        log_det = torch.zeros((batch_dims_phi, batch_dims_x, z.shape[-2], z.shape[-1]), dtype=torch.double)
        for i, j in itertools.product(range(batch_dims_x), range(batch_dims_phi)):
            (z[i, j, ...], _) = self.permutation(z[i, j, ...])
            (z[i, j, ...], log_det[i, j]) = self.linear(z[i, j, ...])
        return z, sum_except_batch(log_det, num_batch_dims=2)


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(
        self,
        in_degrees,
        out_features,
        autoregressive_features,
        random_mask,
        is_output,
        bias=True,
        out_degrees_=None,
    ):
        super().__init__(in_features=len(in_degrees), out_features=out_features, bias=bias)
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output,
            out_degrees_=out_degrees_,
        )
        self.register_buffer("mask", mask)
        self.register_buffer("degrees", degrees)

    @classmethod
    def _get_mask_and_degrees(
        cls,
        in_degrees,
        out_features,
        autoregressive_features,
        random_mask,
        is_output,
        out_degrees_=None,
    ):
        if is_output:
            if out_degrees_ is None:
                out_degrees_ = _get_input_degrees(autoregressive_features)
            out_degrees = tile(out_degrees_, out_features // autoregressive_features)
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long,
                )
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def forward(self, x, context, phi, conditioner_net, q_network=None, inputs=None, hidden_state=None):
        size_weight = self.weight.numel()
        size_bias = self.bias.numel()
        # if q_network is not None:
        if q_network is not None:
            params = conditioner_net(phi, q_network, inputs, hidden_state)
        else:
            params = conditioner_net(phi)
        weight, bias, _ = torch.split(
            params,
            [
                size_weight,
                size_bias,
                params[0, 0, :].numel() - size_weight - size_bias,
            ],
            dim=-1,
        )
        weight = weight.reshape(weight.shape[0], weight.shape[1], self.weight.shape[1], self.weight.shape[0])
        weight.retain_grad()
        assert x.dim() == 4
        # Up to two batch dims here....
        mask = self.mask.view(1, 1, self.mask.shape[1], self.mask.shape[0])
        mask = mask.expand_as(weight)
        masked_weights = weight * mask

        output = torch.einsum("...bij, ...bjk -> ...bik", x, masked_weights) + bias

        return output


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.

    **NOTE** In this implementation, the number of output features is taken to be equal to the number of input features.
    """

    def __init__(
        self,
        in_degrees,
        autoregressive_features,
        context_features=None,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        if context_features is not None:
            raise NotImplementedError()
        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(
        self, inputs, params, conditioner_net, q_network=None, inputs_for_q=None, hidden_state=None, context=None
    ):
        if context is not None:
            raise NotImplementedError()

        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.linear(outputs, context, params, conditioner_net, q_network, inputs_for_q, hidden_state)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(
        self,
        in_degrees,
        autoregressive_features,
        context_features=None,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        if random_mask:
            raise ValueError("Masked residual block can't be used with random masks.")
        super().__init__()
        features = len(in_degrees)

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)])

        # Masked linear.
        linear_0 = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        linear_1 = MaskedLinear(
            in_degrees=linear_0.degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=False,
            is_output=False,
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.degrees = linear_1.degrees
        if torch.all(self.degrees >= in_degrees).item() != 1:
            raise RuntimeError(
                "In a masked residual block, the output degrees can't be" " less than the corresponding input degrees."
            )

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-1e-3, b=1e-3)
            init.uniform_(self.linear_layers[-1].bias, a=-1e-3, b=1e-3)

    def forward(
        self, inputs, params, conditioner_net, q_network=None, inputs_for_q=None, hidden_state=None, context=None
    ):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps, params, conditioner_net, q_network, inputs_for_q, hidden_state)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps, params, conditioner_net, q_network, inputs_for_q, hidden_state)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class MADE(nn.Module):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        output_multiplier=1,
        use_residual_blocks=True,
        random_mask=False,
        permute_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        preprocessing=None,
    ):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()

        # Preprocessing
        if preprocessing is None:
            self.preprocessing = lambda inputs: inputs
        else:
            self.preprocessing = preprocessing

        # Initial layer.
        input_degrees_ = _get_input_degrees(features)
        if permute_mask:
            input_degrees_ = input_degrees_[torch.randperm(features)]
        self.initial_layer = MaskedLinear(
            in_degrees=input_degrees_,
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False,
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)

        # Residual blocks.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
            out_degrees_=input_degrees_,
        )

    def forward(self, inputs, context, params, conditioner_net, q_network=None, inputs_for_q=None, hidden_state=None):
        outputs = self.preprocessing(inputs)
        outputs = self.initial_layer(outputs, context, params, conditioner_net, q_network, inputs_for_q, hidden_state)
        if context is not None:
            outputs += self.context_layer(context)
        for block in self.blocks:
            outputs = block(outputs, params, conditioner_net, q_network, inputs_for_q, hidden_state)
        outputs = self.final_layer(outputs, context, params, conditioner_net, q_network, inputs_for_q, hidden_state)
        return outputs
