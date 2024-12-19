from environments.nchain import NChainEnv

from .model import BayesianExplorationNetwork
from .train import BENTrainer, BENTrainerConfig


def main():
    env = NChainEnv(n=5)

    model = BayesianExplorationNetwork(
        num_actions=env.num_actions,
        state_dim=env.state_dim,
    )

    trainer = BENTrainer(
        env=env,
        ben=model,
        config=BENTrainerConfig(
            n_pretrain_steps=50,
            n_episodes=100,
        ),
    )

    trainer.train()
