import numpy as np
import torch

from .env import GridWorldGFlowNet
from .train import GFlowNetTrainer, TrainingConfig


def main():
    # Configuration
    grid_size = 5
    hidden_dim = 64

    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        n_epochs=100,
        flow_matching_weight=1.0,
        trajectory_balance_weight=0.1,
        entropy_weight=0.01,
        temperature=1.0,
    )

    # Initialize model
    model = GridWorldGFlowNet(grid_size, hidden_dim)

    # Define reward function (example: reward based on Manhattan distance to goal)
    def reward_function(state: torch.Tensor) -> float:
        pos = torch.where(state == 1)[0].item()
        x, y = pos % grid_size, pos // grid_size
        goal_x, goal_y = grid_size - 1, grid_size - 1
        manhattan_dist = abs(x - goal_x) + abs(y - goal_y)
        return float(np.exp(-manhattan_dist))  # Higher reward for closer to goal

    # Initialize trainer
    trainer = GFlowNetTrainer(model, config, reward_function)

    # Train
    history = trainer.train()

    # Plot training curves
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history["train_rewards"], label="Train")
    plt.plot(
        range(
            0,
            len(history["val_rewards"]) * config.validation_interval,
            config.validation_interval,
        ),
        history["val_rewards"],
        label="Validation",
    )
    plt.title("Average Reward")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history["flow_matching_losses"])
    plt.title("Flow Matching Loss")

    plt.subplot(2, 2, 3)
    plt.plot(history["entropy_losses"])
    plt.title("Entropy Loss")

    plt.subplot(2, 2, 4)
    plt.plot(history["trajectory_balance_losses"])
    plt.title("Trajectory Balance Loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
