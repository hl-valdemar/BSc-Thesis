from typing import List
from dataclasses import dataclass

import torch

from gridworld import Action

from .model import BayesianExplorationNetwork
from .env import GridWorldEnv, ReplayBuffer

@dataclass
class TrainingConfig:
    num_episodes: int = 1000
    batch_size: int = 32
    replay_capacity: int = 10000
    min_experiences: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

def train_ben(env: GridWorldEnv, ben: BayesianExplorationNetwork, config: TrainingConfig) -> List[float]:
    """Train BEN on GridWorld environment"""
    # Initialize replay buffer
    buffer = ReplayBuffer.create(config.replay_capacity)
    rewards_history = []
    epsilon = config.epsilon_start
    
    for episode in range(config.num_episodes):
        # Reset environment
        step_data = env.reset()
        episode_reward = 0
        hidden = None  # Reset hidden state
        
        while not step_data.done:
            state = step_data.state
            
            # Epsilon-greedy action selection
            if torch.rand(1) < epsilon:
                action = Action(torch.randint(env.action_dim, (1,)).item())
            else:
                with torch.no_grad():
                    # Get Q-values and uncertainties
                    output = ben.forward(state.unsqueeze(0), 
                                      step_data.reward.unsqueeze(0), 
                                      hidden)
                    q_values = output.q_values.squeeze()
                    hidden = output.hidden
                    action = Action(torch.argmax(q_values).item())
            
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
                    losses = ben.update(batch)
            
            step_data = next_step_data
        
        # Decay epsilon
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        
        # Record episode reward
        rewards_history.append(episode_reward)
        
        # Print progress
        # if (episode + 1) % 10 == 0:
        if (episode + 1) % 1 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    return rewards_history

