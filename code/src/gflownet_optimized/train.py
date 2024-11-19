from typing import Callable, List

import torch
import torch.nn.functional as F

from gridworld import Action
from .model import GFlowNet
from .env import GridWorldEnv, ReplayBuffer, Trajectory
from .config import GFlowNetTrainingConfig

def train(
    env_factory: Callable[[], GridWorldEnv],
    gflownet: GFlowNet, 
    config: GFlowNetTrainingConfig,
):
    buffer = ReplayBuffer(config.replay_capacity, config.trajectory_length)
    rewards_history = []
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
        
        # Train on batch
        if len(buffer) >= config.min_experiences:
            train_trajectories = buffer.sample(config.batch_size)
            if train_trajectories is not None:
                print("Updating gflownet...")
                metrics = gflownet.update(train_trajectories)
                
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}:")
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
        
    return rewards_history

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
