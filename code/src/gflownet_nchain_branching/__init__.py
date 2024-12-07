from nchain_branching import NChainEnv

from .model import GFlowNet
from .train import GFlowNetTrainer


def main():
    # Create environment
    env = NChainEnv(n=5)

    # Create model
    model = GFlowNet(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
    )

    # Create trainer
    trainer = GFlowNetTrainer(env=env, model=model)

    # Train the model
    losses = trainer.train(num_steps=500)

    # Evaluate the model
    eval_metrics = trainer.evaluate_learned_policy(num_trajectories=500)

    print("\nEvaluation metrics:")
    print(f"  Reward count: {eval_metrics["reward_counts"]}")
    print(f"  Reward frequencies: {eval_metrics["reward_frequencies"]}\n")
    print(f"  Branch count: {eval_metrics["branch_counts"]}")
    print(f"  Branch frequencies: {eval_metrics["branch_frequencies"]}\n")
    print(f"  Avg traj length: {eval_metrics["average_trajectory_length"]}")

    # Plot the metrics
    trainer.metrics.plot_training_curves()
