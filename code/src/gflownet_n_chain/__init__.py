from n_chain import NChainEnv

from .model import GFlowNetModel
from .train import GFlowNetTrainer


def main():
    # Create environment and model
    env = NChainEnv(n=20, reward=10.0)
    model = GFlowNetModel(state_dim=env.n, num_actions=env.num_actions, hidden_dim=64)

    # Create trainer
    trainer = GFlowNetTrainer(
        env=env,
        model=model,
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=10000,
        batch_size=32,
        pf_temperature=1.0,
        pb_temperature=1.0,
    )

    # Train the model
    losses = trainer.train(num_steps=500)

    # Plot comprehensive metrics
    print("\nPlotting training metrics...")
    trainer.metrics.plot_metrics(include_raw=True)

    # Plot state visitation distribution
    print("\nPlotting state visitation distribution...")
    trainer.metrics.plot_state_visits()

    # Plot policy distribution
    print("\nPlotting policy distribution...")
    trainer.metrics.plot_policy_distribution()

    # Print final summary
    print("\nFinal training summary:")
    trainer.metrics.print_summary()

    # Save model if path provided
    # if save_model_path:
    #     print(f"\nSaving model to {save_model_path}")
    #     torch.save(model.state_dict(), save_model_path)
