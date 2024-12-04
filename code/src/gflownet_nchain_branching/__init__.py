from nchain_branching import NChainEnv

from .model import GFlowNet
from .train import GFlowNetTrainer


def main():
    # Create environment
    env = NChainEnv(n=10)

    # Create model
    model = GFlowNet(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
    )

    # Create trainer
    trainer = GFlowNetTrainer(env=env, model=model)

    # Train the model
    losses = trainer.train(num_steps=1000)

    # Plot the metrics
    trainer.metrics.plot_training_curves()
