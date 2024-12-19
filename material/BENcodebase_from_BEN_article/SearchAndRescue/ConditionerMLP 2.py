import normflows as nf
import torch
import torch.nn as nn


class ConditionerMLP(nf.flows.Flow):
    """a conditioner MLP for the aleatoric flow parameters. Takes the epistemic flow as input and outputs the parameters for the reverse KL divergence, used for conditioning the aleatoric flow"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_features,
        num_layers=2,
        activation=torch.nn.modules.activation.LeakyReLU(0.3),
        epistemic_flow=None,
        q_net=None,
        device="cpu",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.epistemic_flow = epistemic_flow
        self.q_net = q_net
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            modules.append(activation)
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        modules.append(activation)
        modules.append(nn.Linear(hidden_dim, out_features))
        modules.append(activation)
        if device != "cuda:0":
            j = 1
        self.net = nn.Sequential(*modules).to(device)

    def fetch_epistemic_flow(self, split_epistemic_flow):
        self.register_module("epistemic_flow", split_epistemic_flow)
        self.epistemic_flow = split_epistemic_flow
        self.param_in = split_epistemic_flow.base_dist.mean
        self.input_dim = split_epistemic_flow.base_dist.mean.shape[0]

    def forward(self, x, q_network=None, inputs_for_q=None, hidden_state=None):
        if q_network is not None:
            self.q_net = q_network
            q0, next_hidden = self.q_net(inputs_for_q, hidden_state)
            q_max, _ = q0.max(dim=-1, keepdim=True)
            q_reshaped = q_max.view(1, 1, 1, -1)
            q_reshaped.retain_grad()
            next_hidden_reshaped = next_hidden.view(1, 1, 1, -1)
            next_hidden_reshaped.retain_grad()
            net_inputs = torch.cat([x, q_reshaped, next_hidden_reshaped], dim=-1)
        else:
            net_inputs = x
        return self.net(net_inputs)
