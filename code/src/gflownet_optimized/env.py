import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from gridworld import Action, GridWorld


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

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, max_trajectory_length: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            capacity: Maximum number of trajectories to store
            max_trajectory_length: Maximum length of each trajectory
            alpha: Priority exponent (0 = uniform sampling, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.max_length = max_trajectory_length
        self.alpha = alpha
        self.beta = beta
        self.trajectories: List[Trajectory] = []
        self.priorities: List[float] = []
        self.position = 0
        self.eps = 1e-6  # Small constant to ensure non-zero priorities
        
    def add_trajectory(self, trajectory: Trajectory, error: float | None = None) -> None:
        """Add trajectory with priority based on TD error or reward."""
        if error is None:
            # Use reward as priority if no error provided
            error = abs(trajectory.total_reward)
            
        priority = (abs(error) + self.eps) ** self.alpha
        
        if len(self.trajectories) < self.capacity:
            self.trajectories.append(trajectory)
            self.priorities.append(priority)
        else:
            self.trajectories[self.position] = trajectory
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[List[Trajectory], np.ndarray, np.ndarray] | None:
        """
        Sample batch of trajectories with importance sampling weights.
        
        Returns:
            trajectories: List of sampled trajectories
            indices: Indices of sampled trajectories
            weights: Importance sampling weights
        """
        if len(self.trajectories) < batch_size:
            return None
            
        # Convert priorities to probabilities
        probs = np.array(self.priorities)
        probs = probs / probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.trajectories),
            size=batch_size,
            p=probs,
            replace=False
        )
        
        # Calculate importance sampling weights
        weights = (len(self.trajectories) * probs[indices]) ** -self.beta
        weights = weights / weights.max()  # Normalize weights
        
        sampled_trajectories = [self.trajectories[i] for i in indices]
        
        return sampled_trajectories, indices, weights
        
    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        """Update priorities for trajectories at given indices."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + self.eps) ** self.alpha
            
    def __len__(self) -> int:
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

@dataclass
class CurriculumConfig:
    initial_grid_size: int = 3
    max_grid_size: int = 8
    initial_obstacle_density: float = 0.1
    max_obstacle_density: float = 0.3
    min_path_length: int = 2
    success_threshold: float = 0.7  # When to increase difficulty
    eval_window: int = 50  # Number of episodes to average success rate
    difficulty_increase_rate: float = 1.2  # How much to increase difficulty

class Curriculum:
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_grid_size = config.initial_grid_size
        self.current_obstacle_density = config.initial_obstacle_density
        self.current_min_path_length = config.min_path_length
        
        self.success_window = deque(maxlen=config.eval_window)
        self.min_evaluations = 100
        self.success_history = deque(maxlen=config.eval_window)
        self.true_grid_size = float(config.initial_grid_size)
        
        # Add counters for debugging
        self.total_evaluations = 0
        self.difficulty_increases = 0
        
    def should_increase_difficulty(self) -> bool:
        """Optimized difficulty check with detailed logging"""
        print("\nCurriculum Status:")
        print(f"Current grid size: {self.current_grid_size}")
        print(f"True grid size: {self.true_grid_size}")
        print(f"Success window size: {len(self.success_window)}/{self.config.eval_window}")
        print(f"Total evaluations: {self.total_evaluations}/{self.min_evaluations}")
        
        if len(self.success_window) < self.config.eval_window:
            print("❌ Window not full yet")
            return False

        if self.total_evaluations < self.min_evaluations:
            print("❌ Not enough total evaluations")
            return False
            
        success_rate = sum(self.success_window) / len(self.success_window)
        print(f"Current success rate: {success_rate:.3f} (threshold: {self.config.success_threshold})")
        
        should_increase = success_rate > self.config.success_threshold
        if should_increase:
            print("✅ All conditions met - should increase difficulty")
        else:
            print("❌ Success rate below threshold")
        return should_increase

    def increase_difficulty(self):
        """Enhanced difficulty increase with logging"""
        old_size = self.current_grid_size
        old_density = self.current_obstacle_density
        
        # Update the true grid size
        self.true_grid_size *= self.config.difficulty_increase_rate

        # Then round for actual grid size
        self.current_grid_size = min(
            self.config.max_grid_size,
            max(self.current_grid_size + 1, int(round(self.true_grid_size)))
        )
        
        self.current_obstacle_density *= self.config.difficulty_increase_rate
        if self.current_obstacle_density > self.config.max_obstacle_density:
            self.current_obstacle_density = self.config.max_obstacle_density
            
        self.current_min_path_length = min(
            self.current_grid_size - 1,
            int(self.current_min_path_length * self.config.difficulty_increase_rate)
        )
        
        self.difficulty_increases += 1
        
        print("\nDifficulty Increase:")
        print(f"Grid size: {old_size} -> {self.current_grid_size}")
        print(f"True grid size: {self.true_grid_size}")
        print(f"Obstacle density: {old_density:.3f} -> {self.current_obstacle_density:.3f}")
        print(f"Min path length: {self.current_min_path_length}")
        print(f"Total difficulty increases: {self.difficulty_increases}")
    
    def record_success(self, success_rate: float):
        """Enhanced success recording with validation"""
        assert 0 <= success_rate <= 1, f"Invalid success rate: {success_rate}"
        self.success_history.append(success_rate)
        self.success_window.append(success_rate)
        self.total_evaluations += 1  # Increment total evaluations counter
    
    def get_current_params(self) -> Dict[str, Any]:
        """Fast parameter access"""
        return {
            'grid_size': self.current_grid_size,
            'obstacle_density': self.current_obstacle_density,
            'min_path_length': self.current_min_path_length
        }


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def is_valid_position(pos: Tuple[int, int], width: int, height: int, obstacles: List[Tuple[int, int]]) -> bool:
    """Check if a position is valid (within bounds and not in obstacles)."""
    x, y = pos
    return (0 <= x < width and 
            0 <= y < height and 
            pos not in obstacles)

def has_valid_path(
    start: Tuple[int, int], 
    goal: Tuple[int, int], 
    width: int,
    height: int,
    obstacles: List[Tuple[int, int]]
) -> bool:
    """Optimized BFS path checking using numpy and sets"""
    obstacle_set = set(obstacles)
    visited = {start}
    queue = deque([start])
    
    # Pre-compute moves
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
            
        # Check all moves at once
        for dx, dy in moves:
            next_pos = (current[0] + dx, current[1] + dy)
            if (next_pos not in visited and 
                next_pos not in obstacle_set and
                0 <= next_pos[0] < width and 
                0 <= next_pos[1] < height):
                queue.append(next_pos)
                visited.add(next_pos)
    
    return False

def generate_random_layout(
    width: int,
    height: int,
    obstacle_density: float,
    min_path_length: int,
    max_attempts: int = 100
) -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
    """Optimized random layout generation"""
    num_obstacles = int(width * height * obstacle_density)
    
    # Pre-compute all positions once
    all_positions = [(x, y) for x in range(width) for y in range(height)]

    # Initialize with values
    start_pos = (0, 0)
    goal_pos = (0, 0)
    
    # Try to generate valid layouts
    for _ in range(max_attempts):
        # Quickly pick start and goal positions
        start_pos = (random.randint(0, width-1), random.randint(0, height-1))
        
        # Generate goal position with minimum distance
        while True:
            goal_pos = (random.randint(0, width-1), random.randint(0, height-1))
            if manhattan_distance(start_pos, goal_pos) >= min_path_length:
                break
        
        # Generate obstacles more efficiently
        available_positions = [
            pos for pos in all_positions 
            if pos != start_pos and pos != goal_pos
        ]
        obstacles = set(random.sample(available_positions, min(num_obstacles, len(available_positions))))
        
        # Quick validity check
        if has_valid_path(start_pos, goal_pos, width, height, list(obstacles)):
            return start_pos, goal_pos, list(obstacles)
    
    # Fallback to simple layout
    return start_pos, goal_pos, []

def create_curriculum_env_factory(curriculum: Curriculum) -> Callable[[], GridWorldEnv]:
    """Optimized curriculum-based factory"""
    def env_factory() -> GridWorldEnv:
        params = curriculum.get_current_params()
        width = params['grid_size']
        height = params['grid_size']
        
        start_pos, goal_pos, obstacles = generate_random_layout(
            width=width,
            height=height,
            obstacle_density=params['obstacle_density'],
            min_path_length=params['min_path_length']
        )
        
        grid_world = GridWorld(
            width=width,
            height=height,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacles
        )
        
        return GridWorldEnv(grid_world)
    
    return env_factory
