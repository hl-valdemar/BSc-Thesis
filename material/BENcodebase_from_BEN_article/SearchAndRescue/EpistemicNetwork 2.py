import normflows as nf
import torch

from BayesNormalizingFlow import BayesNormalizingFlow
from ConditionerMLP import ConditionerMLP
from flows import ActNorm, LULinearPermute, MaskedAffineAutoregressive


def construct_epsitemic_flow(
    num_params_aleatoric,
    num_layers,
    flow_aleatoric,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
):
    flows_epistemic = []
    prior_phi = nf.distributions.DiagGaussian([1, 1, num_params_aleatoric], trainable=False).to(device)
    prior_phi.log_scale = torch.nn.Parameter(torch.log(torch.tensor(0.1))).to(device)
    base_epistemic_dist = nf.distributions.DiagGaussian([1, 1, num_params_aleatoric], trainable=False).to(device)
    maf = MaskedAffineAutoregressive(
        num_params_aleatoric,
        num_params_aleatoric,
    ).to(device)

    ### Flow for epistemic uncertainty
    for _ in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable

        # Add flow layer
        flows_epistemic.append(ActNorm(num_params_aleatoric).to(device))
        flows_epistemic.append(maf)
        flows_epistemic.append(LULinearPermute(num_channels=num_params_aleatoric, identity_init=True).to(device))
    flow_epistemic = BayesNormalizingFlow(
        base_epistemic_dist, flows_epistemic, prior_phi, target_dist=flow_aleatoric
    ).to(device)
    # +1 for q_val size
    conditioner_MLP = ConditionerMLP(
        num_params_aleatoric + 1, num_params_aleatoric, num_params_aleatoric, device=device
    ).to(device)
    conditioner_MLP.requires_grad_(True)
    return flow_epistemic, prior_phi
