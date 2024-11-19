from typing import Callable, List, Tuple

import random

import matplotlib.pyplot as plt

from gridworld import GridWorld
from .env import GridWorldEnv
from .config import GFlowNetConfig
from .model import GFlowNet
from .train import GFlowNetTrainingConfig, train, visualize_flows, visualize_training_metrics

def create_random_env_factory(
    width: int = 8,
    height: int = 8,
    obstacle_density: float = 0.2,  # Percentage of grid to fill with obstacles
    min_path_length: int = 4,  # Minimum Manhattan distance between start and goal
    max_attempts: int = 100  # Max attempts to generate valid layout
) -> Callable[[], GridWorldEnv]:
    """Creates a factory that produces GridWorldEnv instances with random valid layouts.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        obstacle_density: Fraction of grid cells to fill with obstacles
        min_path_length: Minimum Manhattan distance between start and goal
        max_attempts: Maximum attempts to generate a valid layout
        
    Returns:
        Factory function that creates new random GridWorldEnv instances
    """
    print("Creating random env factory...")

    def is_valid_position(pos: Tuple[int, int], obstacles: List[Tuple[int, int]]) -> bool:
        """Check if a position is valid (within bounds and not in obstacles)."""
        x, y = pos
        return (0 <= x < width and 
                0 <= y < height and 
                pos not in obstacles)

    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def has_valid_path(start: Tuple[int, int], goal: Tuple[int, int], 
                      obstacles: List[Tuple[int, int]]) -> bool:
        """Check if there exists a valid path from start to goal using BFS."""
        if start == goal:
            return True
            
        queue = [(start, [])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            # Check all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (next_pos == goal):
                    return True
                    
                if (is_valid_position(next_pos, obstacles) and 
                    next_pos not in visited):
                    queue.append((next_pos, path + [next_pos]))
                    visited.add(next_pos)
        
        return False

    def generate_random_layout() -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
        """Generate random start, goal, and obstacle positions."""
        # Calculate number of obstacles
        num_obstacles = int(width * height * obstacle_density)
        
        for _ in range(max_attempts):
            # Generate random start and goal positions
            all_positions = [(x, y) for x in range(width) for y in range(height)]
            random.shuffle(all_positions)
            
            # Pick start and goal with minimum distance
            for i, start_pos in enumerate(all_positions):
                for goal_pos in all_positions[i+1:]:
                    if manhattan_distance(start_pos, goal_pos) >= min_path_length:
                        # Generate random obstacles
                        available_positions = [
                            pos for pos in all_positions 
                            if pos != start_pos and pos != goal_pos
                        ]
                        random.shuffle(available_positions)
                        obstacles = []
                        
                        # Add obstacles one by one, checking path validity
                        for pos in available_positions[:num_obstacles*2]:  # Try more positions than needed
                            obstacles.append(pos)
                            if not has_valid_path(start_pos, goal_pos, obstacles):
                                obstacles.pop()  # Remove invalid obstacle
                            if len(obstacles) == num_obstacles:
                                break
                                
                        if len(obstacles) == num_obstacles:
                            return start_pos, goal_pos, obstacles
        
        raise ValueError(f"Could not generate valid layout after {max_attempts} attempts")

    def env_factory() -> GridWorldEnv:
        """Create a new GridWorldEnv instance with random layout."""
        start_pos, goal_pos, obstacles = generate_random_layout()
        grid_world = GridWorld(
            width=width,
            height=height,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacles
        )
        return GridWorldEnv(grid_world)

    return env_factory

def test_random_env_factory():
    # Create factory
    factory = create_random_env_factory(
        width=6,
        height=6,
        obstacle_density=0.2,
        min_path_length=4
    )
    
    # Create a few environments and print their layouts
    for i in range(3):
        env = factory()
        print(f"\nRandom Environment {i+1}:")
        print(env.grid_world.get_layout_str())
        print()

def main():
    # Create a simple grid world
    layout = """
    #######
    #s   g#
    # o o #
    #     #
    #######
    """

    # Initialize environment
    grid_world = GridWorld(layout=layout)
    env = GridWorldEnv(grid_world)

    # Initialize GFlowNet
    config = GFlowNetConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        learning_rate=1e-4
    )

    gflownet = GFlowNet(config)

    # Training configuration
    training_config = GFlowNetTrainingConfig(
        num_episodes=1000,
        batch_size=32,
        replay_capacity=10000,
        min_experiences=100
    )

    # Train
    env_factory = create_random_env_factory(
        width=4,
        height=4,
        obstacle_density=0.10,  # 15% obstacles
        min_path_length=2
    )
    training_metrics = train(env_factory, gflownet, training_config)

    visualize_training_metrics(
        rewards_history=training_metrics["rewards"],
        metrics_history=training_metrics["metrics"],
        epsilon_history=training_metrics["epsilons"],
        temperature_history=training_metrics["temperatures"],
        success_history=training_metrics["successes"],
        gflownet=gflownet,
        env_factory=env_factory,
    )

    # Visualize flows for a few states
    print("\nFinal Flow Values:")
    test_state = env.reset().state
    visualize_flows(gflownet, test_state)
