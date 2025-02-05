import json
from datetime import datetime

import torch

from environments.nchain import NChainEnv

from .model import GFlowNet
from .train import GFlowNetTrainer


def train(chain_length: int, epsilon: int = 0.1):
    # Create environment
    env = NChainEnv(n=chain_length, rewards=[10, 20, 70])

    # Create model
    model = GFlowNet(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
    )

    # Create trainer
    trainer = GFlowNetTrainer(
        env=env,
        model=model,
        epsilon=epsilon,
    )

    # Train the model
    trainer.train(num_steps=750)

    # Save the model state
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    descriptor = f"chain-{chain_length}_{int(epsilon * 100)}percent-eps"
    torch.save(
        model.state_dict(),
        f"gflownet_model_state_{descriptor}_{timestamp}.pth",
    )

    # Save JSON formatted training metrics to a file
    training_metrics_json = trainer.metrics.to_json()
    with open(
        f"gflownet_training_metrics_{descriptor}_{timestamp}.json",
        "w",
    ) as f:
        f.write(training_metrics_json)

    # Evaluate the model
    eval_metrics = trainer.evaluate_learned_policy(num_trajectories=1000)
    eval_metrics_json = json.dumps(eval_metrics, indent=2)
    with open(
        f"gflownet_eval_metrics_{descriptor}_{timestamp}.json",
        "w",
    ) as f:
        f.write(eval_metrics_json)

    print("\nEvaluation metrics:")
    print(f"  Reward count: {eval_metrics["reward_counts"]}")
    print(f"  Reward frequencies: {eval_metrics["reward_frequencies"]}\n")
    print(f"  Branch count: {eval_metrics["branch_counts"]}")
    print(f"  Branch frequencies: {eval_metrics["branch_frequencies"]}\n")
    print(f"  Avg traj length: {eval_metrics["average_trajectory_length"]}")
    print(f"  Target dist error: {eval_metrics["target_dist_error"]}")

    # Plot the metrics
    trainer.metrics.plot_training_curves(
        save_plot=True,
        transparent=False,
        file_name=f"gflownet_training_plot_{descriptor}_{timestamp}.png",
    )
