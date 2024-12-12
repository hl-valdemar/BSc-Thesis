from environments.nchain import NChainEnv

from .model import BayesianExplorationNetwork


def main():
    env = NChainEnv(n=5)
    model = BayesianExplorationNetwork(
        num_actions=env.num_actions,
        state_dim=env.state_dim,
    )
