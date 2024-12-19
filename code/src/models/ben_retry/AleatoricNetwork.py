import normflows as nf
from torch import nn

from .BayesNormalizingFlow import AleatoricFlow
from .flows import (
    AbsFlow,
    ActNorm,
    ConditionalMaskedAffineAutoregressive,
    ConditionalReverse,
    LeakyReLU1d,
    LULinearPermute,
    SliceFlow,
)


def construct_aleatoric_flow(K=8, num_layers=2, device="cuda:0"):
    abs_flow = AbsFlow(K=K, device=device).to(device)  # K=8
    num_layers = num_layers
    flows_aleatoric = []

    maf = ConditionalMaskedAffineAutoregressive(
        features=2,
        hidden_features=2,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=False,
        random_mask=False,
        activation=nn.LeakyReLU(0.3),
        dropout_probability=0.0,
        use_batch_norm=False,
    ).to(device)

    leaky_relu = LeakyReLU1d(0.3).to(device)
    ### Flow for aleatoric uncertainty
    for _ in range(num_layers):
        flows_aleatoric.append(abs_flow)
        flows_aleatoric.append(nf.flows.Reverse(SliceFlow()).to(device))
        flows_aleatoric.append(nf.flows.Reverse(ActNorm(2)).to(device))
        nf.flows.Reverse(LULinearPermute(num_channels=2, identity_init=True).to(device))
        flows_aleatoric.append(ConditionalReverse(maf).to(device))
        flows_aleatoric.append(SliceFlow().to(device))

    # Epistemic flow specific

    # Calculate the probability density for each value
    # Range of possible vals
    # Priors for aleatoric and epistemic
    # Define the aleatoric network
    base_aleatoric_dist = nf.distributions.DiagGaussian([1, 1], trainable=False).to(
        device
    )
    flow_aleatoric = AleatoricFlow(base_aleatoric_dist, flows_aleatoric).to(device)
    # Define the feeder for params
    num_params_aleatoric = sum([p.numel() for p in flow_aleatoric.parameters()])
    return flow_aleatoric, num_params_aleatoric, base_aleatoric_dist
