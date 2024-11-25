from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from n_chain import NChainEnv

from .model import BEN
from .train import BENTrainer


def analyze_training_results(trainer: BENTrainer):
    """Analyze and visualize training results."""
    print("\nAnalyzing training results...")

    # Plot all training metrics
    print("\nPlotting training metrics...")
    trainer.metrics.plot_metrics(include_raw=True)

    # Perform uncertainty analysis
    print("\nUncertainty Analysis:")
    recent = trainer.metrics.metrics_history[-100:]
    aleatoric_mean = np.mean([m.aleatoric_uncertainty for m in recent])
    epistemic_mean = np.mean([m.epistemic_uncertainty for m in recent])
    bonus_mean = np.mean([m.exploration_bonus for m in recent])

    print("Final Mean Uncertainties:")
    print(f"  Aleatoric: {aleatoric_mean:.4f}")
    print(f"  Epistemic: {epistemic_mean:.4f}")
    print(f"  Exploration Bonus: {bonus_mean:.4f}")

    # Analyze state coverage
    print("\nState Coverage Analysis:")
    total_visits = defaultdict(int)
    for m in recent:
        for state, count in m.state_visits.items():
            total_visits[state] += count

    total_sum = sum(total_visits.values())
    visit_dist = {k: v / total_sum for k, v in total_visits.items()}

    print("State Visitation Distribution:")
    for state in sorted(visit_dist.keys()):
        print(f"  State {state}: {visit_dist[state]:.2%}")

    # Plot state visitation heatmap
    plt.figure(figsize=(10, 4))
    states = sorted(visit_dist.keys())
    visits = [visit_dist[s] for s in states]
    plt.bar(states, visits)
    plt.title("State Visitation Distribution")
    plt.xlabel("State")
    plt.ylabel("Visitation Frequency")
    plt.grid(True)
    plt.show()

    # Analyze learning performance
    print("\nLearning Performance Analysis:")
    episode_returns = [m.episode_return for m in trainer.metrics.metrics_history]
    success_rate = np.mean([m.successful_episodes for m in recent])
    mean_return = np.mean([m.episode_return for m in recent])
    std_return = np.std([m.episode_return for m in recent])

    print(f"Final Success Rate: {success_rate:.2%}")
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")

    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Returns
    ax1.plot(episode_returns)
    ax1.set_title("Episode Returns over Training")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.grid(True)

    # Success rate over time
    window = 100
    success_rate = [
        np.mean(
            [
                m.successful_episodes
                for m in trainer.metrics.metrics_history[max(0, i - window) : i + 1]
            ]
        )
        for i in range(len(trainer.metrics.metrics_history))
    ]
    ax2.plot(success_rate)
    ax2.set_title("Success Rate over Training")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.grid(True)
    plt.show()


def main():
    print("Initializing BEN for N-Chain environment...")

    # Initialize environment and model
    env = NChainEnv(n=100)
    model = BEN(state_dim=env.n)
    print(f"Created N-Chain environment with {env.n} states")

    # Create trainer
    trainer = BENTrainer(
        env=env,
        model=model,
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=10000,
        batch_size=32,
        exploration_bonus_scale=0.1,
        metrics_window=100,
    )
    print("Created BEN trainer with:")
    print(f"  Learning rate: {trainer.q_optimizer.param_groups[0]['lr']}")
    print(f"  Batch size: {trainer.batch_size}")
    print(f"  Buffer size: {trainer.buffer_size}")
    print(f"  Exploration bonus scale: {trainer.exploration_bonus_scale}")

    # Train the model
    print("\nStarting training...")
    losses = trainer.train(num_steps=1000)
    print("\nTraining completed!")

    # Analyze results
    analyze_training_results(trainer)

    # Optional: Save model
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': trainer.q_optimizer.state_dict(),
    #     'training_metrics': trainer.metrics.metrics_history,
    # }, 'ben_nchain.pt')
