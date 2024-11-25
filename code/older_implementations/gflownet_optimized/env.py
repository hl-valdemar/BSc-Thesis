# Wrapping the GridWorld structure

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import torch

from gridworld import GridWorld, Action

@dataclass
class Trajectory:
    states: List[torch.Tensor] # Each tensor has shape [state_dim]
    action_indices: List[torch.Tensor] # Each tensor should be a scalar tensor
    actions: List[Action] # Keep original actions for reference if needed
    rewards: List[torch.Tensor] # Each tensor should be a scalar tensor
    done: bool
    total_reward: float
    length: int

    def __post_init__(self):
        """Ensure tensors have correct shape"""
        # Convert action indices to tensors if they aren't already
        self.action_indices = [
            torch.as_tensor(idx, dtype=torch.long)
            if not isinstance(idx, torch.Tensor)
            else idx.clone() for idx in self.action_indices
        ]

        # Convert rewards to tensors if they aren't already
        self.rewards = [
            torch.as_tensor(r, dtype=torch.float32)
            if not isinstance(r, torch.Tensor)
            else r.clone() for r in self.rewards
        ]

class ReplayBuffer:
    def __init__(self, capacity: int, max_trajectory_length: int):
        self.capacity = capacity
        self.max_length = max_trajectory_length
        self.trajectories: List[Trajectory] = []
        
    def add_trajectory(self, trajectory: Trajectory) -> None:
        if len(self.trajectories) >= self.capacity:
            self.trajectories.pop(0)
        self.trajectories.append(trajectory)
    
    def sample(self, batch_size: int) -> Optional[List[Trajectory]]:
        if len(self.trajectories) < batch_size:
            return None
        indices = torch.randperm(len(self.trajectories))[:batch_size]
        return [self.trajectories[i] for i in indices]

    def __len__(self) -> int:
        """Return number of trajectories in buffer"""
        return len(self.trajectories)

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
        """Scale reward to be strictly positive and well-conditioned"""
        scaled = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        return scaled * 0.9 + 0.1  # Ensure rewards are between 0.1 and 1.0

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
