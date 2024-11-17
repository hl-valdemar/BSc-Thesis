from typing import List
from dataclasses import dataclass

import torch

from gridworld import Action
from .model import GFlowNet
from .env import GridWorldEnv, ReplayBuffer

@dataclass
class GFlowNetTrainingConfig:
    num_episodes: int = 1000
    batch_size: int = 32
    replay_capacity: int = 10000
    min_experiences: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

def train(env: GridWorldEnv, 
                   gflownet: GFlowNet, 
                   config: GFlowNetTrainingConfig) -> List[float]:
    """
    Train GFlowNet on GridWorld environment
    Returns:
        rewards_history: List of episode rewards
    """
    # Initialize replay buffer
    buffer = ReplayBuffer.create(config.replay_capacity)
    rewards_history = []
    epsilon = config.epsilon_start
    
    for episode in range(config.num_episodes):
        # Reset environment
        step_data = env.reset()
        episode_reward = 0.0
        
        while not step_data.done:
            state = step_data.state
            
            # Get action using flow-based policy
            action = gflownet.get_action(state, epsilon)
            
            # Take step in environment
            next_step_data = env.step(action)
            episode_reward += next_step_data.reward.item()
            
            # Store transition in replay buffer
            buffer.add(
                state=step_data.state,
                action=action,
                reward=next_step_data.reward,
                next_state=next_step_data.state,
                done=next_step_data.done
            )
            
            # Train on batch if we have enough experiences
            if len(buffer) >= config.min_experiences:
                batch = buffer.sample(config.batch_size)
                if batch is not None:
                    losses = gflownet.update(batch)
                    
            step_data = next_step_data
        
        # Decay epsilon
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        
        # Record episode reward
        rewards_history.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode + 1}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {epsilon:.2f}")
    
    return rewards_history

def visualize_flows(env: GridWorldEnv, 
                   gflownet: GFlowNet, 
                   state: torch.Tensor) -> None:
    """
    Visualize the flows and policy for a given state
    """
    with torch.no_grad():
        output = gflownet.forward(state.unsqueeze(0))
        flows = output.flows[0]
        policy = flows / output.state_flow[0]
        
        print("\nFlows F(s,a):")
        for action in Action:
            print(f"{action.name}: {flows[action.value]:.3f}")
            
        print("\nPolicy π(a|s):")
        for action in Action:
            print(f"{action.name}: {policy[action.value]:.3f}")
