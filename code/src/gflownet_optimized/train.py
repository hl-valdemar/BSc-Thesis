from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from gridworld import Action

from .config import GFlowNetConfig, GFlowNetTrainingConfig
from .env import (Curriculum, GridWorldEnv, PrioritizedReplayBuffer,
                  Trajectory, create_curriculum_env_factory)
from .model import GFlowNet


def train_gflownet(
    curriculum: Curriculum,
    config: GFlowNetTrainingConfig,
) -> Tuple[Dict[str, Any], GFlowNet]:
    # Initialize buffer and other components
    buffer = PrioritizedReplayBuffer(
        capacity=config.replay_capacity,
        max_trajectory_length=config.trajectory_length,
        alpha=0.6,
        beta=0.4
    )
    
    rewards_history = []
    metrics_history = []
    epsilon_history = []
    temperature_history = []
    success_history = []
    success_rate_history = []
    curriculum_history = []  # Track curriculum changes

    epsilon = config.epsilon_start
    temperature = config.temperature_start

    # Create initial environment factory
    env_factory = create_curriculum_env_factory(curriculum)
    envs: List[GridWorldEnv] = []
    for _ in range(config.batch_size):
        env = env_factory()
        while env is None:
            print("Failed to generate necessary environment, retrying...")
            env = env_factory()

        envs.append(env)


    gflownet_config = GFlowNetConfig(
        state_dim=envs[0].state_dim,
        action_dim=envs[0].action_dim,
        num_episodes=config.num_episodes,
    )
    gflownet = GFlowNet(gflownet_config)
    
    # Training loop
    for episode in range(config.num_episodes):
        # Collect trajectories
        trajectories = collect_trajectories_batch(
            envs,
            gflownet,
            epsilon,
            temperature,
        )

        # Calculate success metrics
        successes = sum(1 for t in trajectories if t.done and t.total_reward > 0)
        success_rate = successes / len(trajectories)
        success_history.append(successes)  # Raw successes
        success_rate_history.append(success_rate)  # Rates
        
        # Update curriculum
        curriculum.record_success(success_rate)
        if curriculum.should_increase_difficulty():
            curriculum.increase_difficulty()
            # REPLACE environments with new ones
            envs = []  # Clear existing envs
            env_factory = create_curriculum_env_factory(curriculum)
            
            for _ in range(config.batch_size):
                env = env_factory()
                while env is None:
                    print("Failed to generate necessary environment, retrying...")
                    env = env_factory()
                envs.append(env)

            curriculum_history.append(curriculum.get_current_params())
            print(f"Episode {episode}: Increasing difficulty to {curriculum.get_current_params()}")

        # Add trajectories to buffer
        for trajectory in trajectories:
            buffer.add_trajectory(trajectory)
            rewards_history.append(trajectory.total_reward)

        # Training with gradient accumulation and prioritized replay
        if len(buffer) >= config.min_experiences:
            larger_batch_size = config.batch_size * gflownet.accumulation_steps
            result = buffer.sample(larger_batch_size)
            
            if result is not None:
                train_trajectories, indices, weights = result
                metrics = gflownet.update(
                    train_trajectories,
                    indices=indices,
                    weights=weights
                )
                buffer.update_priorities(indices, np.array([metrics['td_errors']]))
                metrics_history.append(metrics)
                
                # Logging
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}:")
                    print(f"  Grid size: {curriculum.current_grid_size}")
                    print(f"  Success rate: {success_rate:.3f}")
                    for k, v in metrics.items():
                        print(f"  {k}: {v:.8f}")
        
        # Update exploration parameters
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        temperature = max(config.temperature_end, temperature * config.temperature_decay)
        
        epsilon_history.append(epsilon)
        temperature_history.append(temperature)
    
    return {
        "rewards": rewards_history, 
        "metrics": metrics_history,
        "epsilons": epsilon_history,
        "temperatures": temperature_history,
        "successes": success_history,
        "success_rates": success_rate_history,
        "curriculum": curriculum_history,
    }, gflownet

# def train_gflownet(
#     env_factory: Callable[[], GridWorldEnv],
#     config: GFlowNetTrainingConfig,
# ) -> Tuple[Dict[str, Any], GFlowNet]:
#     # Initialize buffer and metrics tracking
#     # buffer = ReplayBuffer(config.replay_capacity, config.trajectory_length)
#     buffer = PrioritizedReplayBuffer(
#         capacity=config.replay_capacity,
#         max_trajectory_length=config.trajectory_length,
#         alpha=0.6, # Priority exponent
#         beta=0.4, # Initial importance sampling weight
#     )
#
#     rewards_history = []
#     metrics_history = []
#     epsilon_history = []
#     temperature_history = []
#     success_history = []
#
#     epsilon = config.epsilon_start
#     temperature = config.temperature_start
#
#     # Initialize environments and trajectories
#     envs = [env_factory() for _ in range(config.batch_size)]
#
#     gflownet_config = GFlowNetConfig(state_dim=envs[0].state_dim, action_dim=envs[0].action_dim)
#     gflownet = GFlowNet(gflownet_config)
#
#     # Training loop
#     for episode in range(config.num_episodes):
#         # Collect multiple trajectories
#         print("Collecting trajectories...")
#         trajectories = collect_trajectories_batch(
#             envs,
#             gflownet,
#             epsilon,
#             temperature,
#         )
#
#         # Add trajectories to buffer
#         for trajectory in trajectories:
#             buffer.add_trajectory(trajectory)
#             rewards_history.append(trajectory.total_reward)
#
#         # Compute success rate
#         successes = sum(1 for t in trajectories if t.done and t.total_reward > 0)
#         success_rate = successes / len(trajectories)
#         success_history.append(success_rate)
#
#         # Train on batch with prioritized replay
#         if len(buffer) >= config.min_experiences:
#             # Need larger batch for gradient accumulation
#             larger_batch_size = config.batch_size * gflownet.accumulation_steps
#
#             # Sample batch with importance sampling weights
#             result = buffer.sample(larger_batch_size)
#
#             if result is not None:
#                 train_trajectories, indices, weights = result
#
#                 # Update model
#                 print("Updating gflownet...")
#                 metrics = gflownet.update(
#                     train_trajectories,
#                     indices=indices,
#                     weights=weights,
#                 )
#
#                 # Update priorities based on new TD errors
#                 buffer.update_priorities(indices, np.array([metrics['td_errors']]))
#                 metrics_history.append(metrics)
#
#                 # Logging
#                 if (episode + 1) % 10 == 0:
#                     print(f"Episode {episode + 1}:")
#                     print(f"  Success rate: {success_rate:.3f}")
#                     for k, v in metrics.items():
#                         print(f"  {k}: {v:.4f}")
#
#         # Update exploration parameters
#         epsilon = max(
#             config.epsilon_end,
#             epsilon * config.epsilon_decay
#         )
#         temperature = max(
#             config.temperature_end,
#             temperature * config.temperature_decay
#         )
#
#         epsilon_history.append(epsilon)
#         temperature_history.append(temperature)
#
#     return {
#         "rewards": rewards_history, 
#         "metrics": metrics_history,
#         "epsilons": epsilon_history,
#         "temperatures": temperature_history,
#         "successes": success_history,
#     }, gflownet
#
def collect_trajectories_batch(
    envs: List[GridWorldEnv],
    gflownet: GFlowNet,
    epsilon: float,
    temperature: float,
) -> List[Trajectory]:
    """Collect multiple trajectories with batched processing."""
    trajectories = [Trajectory([], [], [], [], False, 0.0, 0) for _ in range(gflownet.config.batch_size)]
    
    # Track which environments are still active
    active_envs = list(range(gflownet.config.batch_size))
    
    # Get initial states
    current_steps = [env.reset() for env in envs]
    
    while active_envs:
        # Collect states from active environments
        states = []
        active_indices = []  # Keep track of which trajectories these states belong to
        
        for idx in active_envs:
            states.append(current_steps[idx].state)
            active_indices.append(idx)
            
            # Add state to trajectory
            if len(trajectories[idx].states) == 0:
                trajectories[idx].states.append(current_steps[idx].state)
        
        if not states:  # All environments are done
            break
            
        # Process batch of states
        state_batch = torch.stack(states)
        
        # Get actions for entire batch
        with torch.no_grad():
            output = gflownet.forward(state_batch)
            
            # Handle epsilon-greedy exploration
            random_mask = torch.rand(len(states)) < epsilon
            
            # Get actions from policy
            logits = output.logits / temperature
            probs = F.softmax(logits, dim=-1)
            policy_actions = torch.multinomial(probs, 1).squeeze(-1)
            
            # Random actions for exploration
            random_action_indices = torch.randint(
                gflownet.config.action_dim, 
                (len(states),)
            )
            
            # Combine random and policy actions
            action_indices = torch.where(
                random_mask,
                random_action_indices,
                policy_actions
            )
        
        # Take steps in environments
        for batch_idx, env_idx in enumerate(active_indices):
            action = Action(action_indices[batch_idx].item())
            
            # Take step in environment
            step_data = envs[env_idx].step(action)
            current_steps[env_idx] = step_data
            
            # Update trajectory
            trajectories[env_idx].action_indices.append(
                torch.tensor(action.value, dtype=torch.long)
            )
            trajectories[env_idx].actions.append(action)
            trajectories[env_idx].rewards.append(step_data.reward)
            trajectories[env_idx].total_reward += step_data.reward.item()
            trajectories[env_idx].length += 1
            trajectories[env_idx].states.append(step_data.state)
            
            # Check if environment is done
            if step_data.done:
                trajectories[env_idx].done = True
                active_envs.remove(env_idx)
    
    return trajectories

def visualize_training_metrics(
    rewards_history: List[float],
    metrics_history: List[Dict[str, float]],
    epsilon_history: List[float],
    temperature_history: List[float],
    success_history: List[int],  # Raw successes
    success_rate_history: List[float],  # Success rates
    gflownet: GFlowNet,
    env_factory: Callable[[], GridWorldEnv],
):
    """Create comprehensive visualization of training metrics"""
    # Set the seaborn style
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(20, 15))
    
    # 1. Rewards subplot
    ax1 = plt.subplot(331)
    ax1.plot(rewards_history, alpha=0.3, color='blue', label='Raw rewards')
    # Add moving average
    window_size = 50
    moving_avg = np.convolve(rewards_history, 
                            np.ones(window_size)/window_size, 
                            mode='valid')
    ax1.plot(range(window_size-1, len(rewards_history)), 
             moving_avg, 
             color='blue', 
             label='Moving average')
    ax1.set_title('Rewards over time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    
    # 2. Success Metrics
    ax2 = plt.subplot(332)
    # Plot raw successes
    ax2.plot(success_history, alpha=0.3, color='green', label='Raw successes')
    # Plot success rate
    ax2.plot(success_rate_history, color='blue', label='Success rate')
    # Add moving averages
    success_avg = np.convolve(success_history, 
                             np.ones(window_size)/window_size, 
                             mode='valid')
    rate_avg = np.convolve(success_rate_history, 
                          np.ones(window_size)/window_size, 
                          mode='valid')
    ax2.plot(range(window_size-1, len(success_history)), 
             success_avg, 
             color='darkgreen', 
             label='Success MA')
    ax2.plot(range(window_size-1, len(success_rate_history)), 
             rate_avg, 
             color='darkblue', 
             label='Rate MA')
    ax2.set_title('Success Metrics')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Count / Rate')
    ax2.legend()
    
    # 3. Loss components
    ax3 = plt.subplot(333)
    episodes = range(len(metrics_history))
    flow_losses = [m['flow_loss'] for m in metrics_history]
    balance_losses = [m['balance_loss'] for m in metrics_history]
    reg_losses = [m['reg_loss'] for m in metrics_history]
    entropy_losses = [m['entropy_loss'] for m in metrics_history]
    
    ax3.plot(episodes, flow_losses, label='Flow Loss')
    ax3.plot(episodes, balance_losses, label='Balance Loss')
    ax3.plot(episodes, reg_losses, label='Reg Loss')
    ax3.plot(episodes, entropy_losses, label='Entropy Loss')
    ax3.set_yscale('log')
    ax3.set_title('Training Losses')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss (log scale)')
    ax3.legend()
    
    # 4. Exploration parameters
    ax4 = plt.subplot(334)
    ax4.plot(epsilon_history, label='Epsilon')
    ax4.plot(temperature_history, label='Temperature')
    ax4.set_title('Exploration Parameters')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Value')
    ax4.legend()
    
    # 5. Flow network visualization for a sample environment
    ax5 = plt.subplot(335)
    env = env_factory()
    while env is None:
        print("Failed to generate valid environment, retrying...")
        env = env_factory()

    state = env.reset().state
    
    # Create grid for flow visualization
    flow_grid = np.zeros((env.grid_world.height, env.grid_world.width))
    value_grid = np.zeros((env.grid_world.height, env.grid_world.width))
    
    with torch.no_grad():
        for y in range(env.grid_world.height):
            for x in range(env.grid_world.width):
                if (x, y) not in env.grid_world.obstacles:
                    state_tensor = torch.tensor([x, y], dtype=torch.float32)
                    output = gflownet.forward(state_tensor.unsqueeze(0))
                    flow = output.flows.sum().item()
                    value = output.state_flow.item()
                    flow_grid[env.grid_world.height - 1 - y, x] = flow
                    value_grid[env.grid_world.height - 1 - y, x] = value
    
    im = ax5.imshow(flow_grid, cmap='viridis')
    plt.colorbar(im, ax=ax5)
    ax5.set_title('Flow Network Values')
    
    # 6. Policy distribution
    ax6 = plt.subplot(336)
    # Sample a state and show action probabilities
    with torch.no_grad():
        output = gflownet.forward(state.unsqueeze(0))
        probs = F.softmax(output.logits, dim=-1)[0]
        
    action_names = [a.name for a in Action]
    ax6.bar(action_names, probs.numpy())
    ax6.set_title('Action Distribution (Sample State)')
    ax6.set_ylabel('Probability')
    plt.xticks(rotation=45)

    # 7. Learning rate if available
    if metrics_history and 'learning_rate' in metrics_history[0]:
        ax7 = plt.subplot(337)
        learning_rates = [m['learning_rate'] for m in metrics_history]
        ax7.plot(episodes, learning_rates)
        ax7.set_title('Learning Rate')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Learning Rate')
        ax7.set_yscale('log')

    # 8. TD Errors if available
    if metrics_history and 'td_errors' in metrics_history[0]:
        ax8 = plt.subplot(338)
        td_errors = [m['td_errors'] for m in metrics_history]
        ax8.plot(episodes, td_errors)
        ax8.set_title('TD Errors')
        ax8.set_xlabel('Episode')
        ax8.set_ylabel('Error')
        ax8.set_yscale('log')

    plt.tight_layout()
    plt.show()
