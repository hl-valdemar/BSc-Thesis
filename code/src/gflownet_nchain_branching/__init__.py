from nchain_branching import NChainEnv

from .model import GFlowNet
from .train import GFlowNetTrainer


def main():
    # Create environment
    env = NChainEnv(n=10)

    # Create model
    model = GFlowNet(
        state_dim=env.n,
        num_actions=env.num_actions,
    )

    # Create trainer
    trainer = GFlowNetTrainer(env=env, model=model)

    # Train the model
    losses = trainer.train(num_steps=10_000)
