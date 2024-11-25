from typing import List

import torch
from n_chain import NChain

from .model import GFlowNet


def train_gflownet(
    env: NChain,
    model: GFlowNet,
    n_episodes: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> List[float]:
    """
    Train GFlowNet using trajectory balance.

    Args:
        env: N-Chain environment
        model: GFlowNet model
        n_episodes: Number of episodes to train for
        batch_size: Batch size for training
        lr: Learning rate

    Returns:
        losses: List of losses during training
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for episode in range(n_episodes):
        # Collect batch of trajectories
        trajectories = []
        rewards = []

        for _ in range(batch_size):
            trajectory, reward = model.sample_trajectory(env)
            trajectories.append(trajectory)
            rewards.append(reward)

        # Convert rewards to tensor
        rewards = torch.tensor(rewards, device=device)

        # Compute loss and update
        optimizer.zero_grad()
        loss = model.compute_trajectory_balance_loss(trajectories, rewards)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss.item():.4f}")

    return losses


def evaluate_gflownet(env: NChain, model: GFlowNet, n_episodes: int = 100) -> float:
    """
    Evaluate GFlowNet performance.

    Args:
        env: N-Chain environment
        model: GFlowNet model
        n_episodes: Number of episodes to evaluate

    Returns:
        success_rate: Fraction of episodes reaching terminal state
    """
    successes = 0

    for _ in range(n_episodes):
        trajectory, reward = model.sample_trajectory(env)
        if reward > 0:  # Reached terminal state
            successes += 1

    return successes / n_episodes
