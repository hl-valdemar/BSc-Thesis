from typing import Callable, List, Dict, Any

import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from gridworld import Action
from .model import GFlowNet
from .env import GridWorldEnv, ReplayBuffer, Trajectory
from .config import GFlowNetTrainingConfig

def train(
    env_factory: Callable[[], GridWorldEnv],
    gflownet: GFlowNet, 
    config: GFlowNetTrainingConfig,
) -> Dict[str, Any]:
    buffer = ReplayBuffer(config.replay_capacity, config.trajectory_length)
    rewards_history = []
    metrics_history = []
    epsilon_history = []
    temperature_history = []
    success_history = []

    epsilon = config.epsilon_start
    temperature = config.temperature_start
    
    # Training loop
    for episode in range(config.num_episodes):
        # Collect multiple trajectories
        print("Collecting trajectories...")
        trajectories = collect_trajectories_batch(
            env_factory,
            gflownet,
            epsilon,
            temperature,
            num_trajectories=config.batch_size,
            min_batch_size=config.min_batch_size
        )

        # Add trajectories to buffer
        for trajectory in trajectories:
            buffer.add_trajectory(trajectory)
            rewards_history.append(trajectory.total_reward)
        
        successes = sum(1 for t in trajectories if t.done and t.total_reward > 0)
        success_rate = successes / len(trajectories)
        success_history.append(success_rate)

        # Train on batch
        if len(buffer) >= config.min_experiences:
            train_trajectories = buffer.sample(config.batch_size)
            if train_trajectories is not None:
                print("Updating gflownet...")
                metrics = gflownet.update(train_trajectories)
                metrics_history.append(metrics)
                
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}:")
                    print(f"  Success rate: {success_rate:.3f}")
                    for k, v in metrics.items():
                        print(f"  {k}: {v:.4f}")
        
        # Update exploration parameters
        epsilon = max(
            config.epsilon_end,
            epsilon * config.epsilon_decay
        )
        temperature = max(
            config.temperature_end,
            temperature * config.temperature_decay
        )

        epsilon_history.append(epsilon)
        temperature_history.append(temperature)
        
    return {
        "rewards": rewards_history, 
        "metrics": metrics_history,
        "epsilons": epsilon_history,
        "temperatures": temperature_history,
        "successes": success_history,
    }

def collect_trajectories_batch(
    env_factory: Callable[[], GridWorldEnv],
    gflownet: GFlowNet,
    epsilon: float,
    temperature: float,
    num_trajectories: int,
    min_batch_size: int = 4,
) -> List[Trajectory]:
    """Collect multiple trajectories with batched processing."""
    # Initialize environments and trajectories
    envs = [env_factory() for _ in range(num_trajectories)]
    trajectories = [Trajectory([], [], [], [], False, 0.0, 0) for _ in range(num_trajectories)]
    
    # Track which environments are still active
    active_envs = list(range(num_trajectories))
    
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

def visualize_flows(
        # env: GridWorldEnv, 
        gflownet: GFlowNet, 
        state: torch.Tensor,
    ) -> None:
    """
    Visualize the flows and policy for a given state
    """
    with torch.no_grad():
        output = gflownet.forward(state)
        flows = output.flows[0]
        policy = flows / output.state_flow[0]
        
        print("\nFlows F(s,a):")
        for action in Action:
            print(f"{action.name}: {flows[action.value]:.3f}")
            
        print("\nPolicy π(a|s):")
        for action in Action:
            print(f"{action.name}: {policy[action.value]:.3f}")

def visualize_training_metrics(
    rewards_history: List[float],
    metrics_history: List[Dict[str, float]],
    epsilon_history: List[float],
    temperature_history: List[float],
    success_history: List[float],
    gflownet: GFlowNet,
    env_factory: Callable[[], GridWorldEnv],
):
    """Create comprehensive visualization of training metrics"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Rewards subplot
    ax1 = plt.subplot(231)
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
    
    # 2. Loss components
    ax2 = plt.subplot(232)
    episodes = range(len(metrics_history))
    flow_losses = [m['flow_loss'] for m in metrics_history]
    balance_losses = [m['balance_loss'] for m in metrics_history]
    reg_losses = [m['reg_loss'] for m in metrics_history]
    
    ax2.plot(episodes, flow_losses, label='Flow Loss')
    ax2.plot(episodes, balance_losses, label='Balance Loss')
    ax2.plot(episodes, reg_losses, label='Reg Loss')
    ax2.set_yscale('log')
    ax2.set_title('Training Losses')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss (log scale)')
    ax2.legend()
    
    # 3. Exploration parameters
    ax3 = plt.subplot(233)
    ax3.plot(epsilon_history, label='Epsilon')
    ax3.plot(temperature_history, label='Temperature')
    ax3.set_title('Exploration Parameters')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Value')
    ax3.legend()
    
    # 4. Success rate
    ax4 = plt.subplot(234)
    ax4.plot(success_history, color='green')
    ax4.set_title('Success Rate')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Success Rate')
    ax4.set_ylim((0, 1))
    
    # 5. Flow network visualization for a sample environment
    ax5 = plt.subplot(235)
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
    ax6 = plt.subplot(236)
    # Sample a state and show action probabilities
    with torch.no_grad():
        output = gflownet.forward(state.unsqueeze(0))
        probs = F.softmax(output.logits, dim=-1)[0]
        
    action_names = [a.name for a in Action]
    ax6.bar(action_names, probs.numpy())
    ax6.set_title('Action Distribution (Sample State)')
    ax6.set_ylabel('Probability')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
