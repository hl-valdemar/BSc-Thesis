# Wrapping the GridWorld structure

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import torch

from gridworld import GridWorld, Action

@dataclass
class StepData:
    state: torch.Tensor
    reward: torch.Tensor
    done: bool
    info: Dict

class GridWorldEnv:
    def __init__(self, grid_world: GridWorld):
        self.grid_world = grid_world
        self.state_dim = 2  # x, y coordinates
        self.action_dim = len(Action)  # UP, RIGHT, DOWN, LEFT

        # Keep track of episode stats
        self.steps = 0
        self.total_reward = 0.0

        # Reward scaling
        self.min_reward = -1.0
        self.max_reward = 100.0

    def _scale_reward(self, reward: float) -> float:
        """Scale reward to be positive"""
        return (reward - self.min_reward) / (self.max_reward - self.min_reward) + 1e-6

    def _state_to_tensor(self, state: Tuple[int, int]) -> torch.Tensor:
        """Convert state tuple to torch tensor"""
        return torch.tensor(state, dtype=torch.float32)

    def _reward_to_tensor(self, reward: float) -> torch.Tensor:
        """Convert reward to torch tensor"""
        return torch.tensor(reward, dtype=torch.float32)

    def reset(self) -> StepData:
        """Reset environment and return initial state"""
        state = self.grid_world.reset()
        self.steps = 0
        self.total_reward = 0.0

        return StepData(
            state=self._state_to_tensor(state),
            reward=torch.tensor(0.0),  # Initial reward is 0
            done=False,
            info={}
        )

    def step(self, action: Action) -> StepData:
        """Take a step in the environment"""
        state, reward, done = self.grid_world.step(action)

        # Scale the reward to be positive
        scaled_reward = self._scale_reward(reward)

        self.steps += 1
        self.total_reward += reward

        # Add additional info
        info = {
            "steps": self.steps,
            "total_reward": self.total_reward,
            "original_reward": reward,
        }

        # Check if episode should end due to max steps
        if self.steps >= self.grid_world.max_steps:
            done = True

        return StepData(
            state=self._state_to_tensor(state),
            reward=self._reward_to_tensor(scaled_reward),
            done=done,
            info=info
        )

@dataclass
class ReplayBuffer:
    capacity: int
    states: List[torch.Tensor]
    actions: List[Action]
    rewards: List[torch.Tensor]
    next_states: List[torch.Tensor]
    dones: List[bool]

    @classmethod
    def create(cls, capacity: int) -> 'ReplayBuffer':
        return cls(
            capacity=capacity,
            states=[],
            actions=[],
            rewards=[],
            next_states=[],
            dones=[]
        )

    def add(self, state: torch.Tensor, action: Action, reward: torch.Tensor, 
            next_state: torch.Tensor, done: bool) -> None:
        """Add transition to buffer"""
        if len(self.states) >= self.capacity:
            # Remove oldest transition
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch of transitions"""
        if len(self.states) < batch_size:
            return None

        indices = torch.randperm(len(self.states))[:batch_size]

        return {
            'states': torch.stack([self.states[i] for i in indices]),
            'actions': torch.tensor([self.actions[i].value for i in indices]),
            'rewards': torch.stack([self.rewards[i] for i in indices]),
            'next_states': torch.stack([self.next_states[i] for i in indices]),
            'dones': torch.tensor([self.dones[i] for i in indices])
        }

    def __len__(self) -> int:
        return len(self.states)

