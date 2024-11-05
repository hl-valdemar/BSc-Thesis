from typing import Dict, NamedTuple, Optional, Tuple, List
import numpy as np
import random
from enum import Enum, global_enum_repr
import matplotlib.pyplot as plt
from collections import deque

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class EpisodeMetrics(NamedTuple):
    """Stores metrics for a single episode."""
    total_reward: float
    steps: int
    success: bool
    final_position: Tuple[int, int]

class TrainingMetrics:
    """Tracks and computes various training metrics."""
    def __init__(self, window_size: int = 100) -> None:
        self.episode_metrics: List[EpisodeMetrics] = []
        self.window_size = window_size
        
    def add_episode(self, metrics: EpisodeMetrics) -> None:
        """Add metrics for a single episode."""
        self.episode_metrics.append(metrics)

    def get_success_rate(self, window: Optional[int] = None) -> float:
        """Calculate success rate over last n episodes."""
        n = window or len(self.episode_metrics)
        if not self.episode_metrics:
            return 0.0
        recent_episodes = self.episode_metrics[-n:]
        return sum(1 for m in recent_episodes if m.success) / len(recent_episodes)

    def get_average_reward(self, window: Optional[int] = None) -> float:
        """Calculate average reward over last n episodes."""
        n = window or len(self.episode_metrics)
        if not self.episode_metrics:
            return 0.0
        recent_episodes = self.episode_metrics[-n:]
        return sum(m.total_reward for m in recent_episodes) / len(recent_episodes)

    def get_average_steps(self, window: Optional[int] = None) -> float:
        """Calculate average steps over last n episodes."""
        n = window or len(self.episode_metrics)
        if not self.episode_metrics:
            return 0.0
        recent_episodes = self.episode_metrics[-n:]
        return sum(m.steps for m in recent_episodes) / len(recent_episodes)
        
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """Plot training metrics."""
        episodes = list(range(1, len(self.episode_metrics) + 1))
        rewards = [m.total_reward for m in self.episode_metrics]
        steps = [m.steps for m in self.episode_metrics]
        successes = [m.success for m in self.episode_metrics]

        # Calculate moving averages
        window = min(self.window_size, len(episodes))
        reward_ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
        steps_ma = np.convolve(steps, np.ones(window)/window, mode="valid")
        successes_ma = np.convolve(successes, np.ones(window)/window, mode="valid")

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # Plot rewards
        axes[0].plot(episodes, rewards, alpha=0.3, color="blue")
        axes[0].plot(range(window, len(episodes) + 1), reward_ma, color="blue", linewidth=2)
        axes[0].set_title("Episode Rewards")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].grid(True)

        # Plot steps
        axes[1].plot(episodes, steps, alpha=0.3, color="green")
        axes[1].plot(range(window, len(episodes) + 1), reward_ma, color="green", linewidth=2)
        axes[1].set_title("Steps per Episode")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].grid(True)

        # Plot success rate
        axes[2].plot(episodes, successes, alpha=0.3, color="orange")
        axes[2].plot(range(window, len(episodes) + 1), reward_ma, color="orange", linewidth=2)
        axes[2].set_title("Success Rate")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Success Rate")
        axes[2].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

class GridWorld:
    def __init__(self, width: int, height: int, start_pos: Tuple[int, int], 
                 goal_pos: Tuple[int, int], obstacles: List[Tuple[int, int]]) -> None:
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles
        self.current_pos = start_pos
        self.max_steps = width * height * 2 # Maximum steps before episode termination
        
    def reset(self) -> Tuple[int, int]:
        """Reset the environment to initial state."""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if the position is valid."""
        x, y = pos
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                pos not in self.obstacles)
    
    def get_next_state(self, state: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Get next state given current state and action."""
        x, y = state
        if action == Action.UP:
            next_pos = (x, y + 1)
        elif action == Action.RIGHT:
            next_pos = (x + 1, y)
        elif action == Action.DOWN:
            next_pos = (x, y - 1)
        else:  # LEFT
            next_pos = (x - 1, y)
            
        return next_pos if self.is_valid_position(next_pos) else state
    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool]:
        """Execute action and return new state, reward and done flag."""
        next_pos = self.get_next_state(self.current_pos, action)
        
        # Update current position
        self.current_pos = next_pos
        
        # Calculate reward
        if next_pos == self.goal_pos:
            reward = 100.0
            done = True
        elif next_pos in self.obstacles:
            reward = -100.0
            done = True
        else:
            reward = -1.0  # Small penalty for each move
            done = False
            
        return next_pos, reward, done

class QLearningAgent:
    def __init__(self, 
                 env: GridWorld, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, 
                 epsilon: float = 0.1,
                 min_epsilon: float = 0.01,
                 epsilon_decay: float = 0.995) -> None:
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.metrics = TrainingMetrics()
        self.q_table: Dict[Tuple[int, int], Dict[Action, float]] = {}
        
        # Initialize Q-table with zeros
        for x in range(env.width):
            for y in range(env.height):
                if (x, y) not in env.obstacles:
                    self.q_table[(x, y)] = {action: 0.0 for action in Action}
    
    def get_action(self, state: Tuple[int, int]) -> Action:
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(list(Action))
        else:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
    
    def update(self, 
              state: Tuple[int, int], 
              action: Action, 
              reward: float, 
              next_state: Tuple[int, int]) -> None:
        """Update Q-value using Q-learning update rule."""
        best_next_value = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        
        # Q-learning update formula
        new_q = current_q + self.lr * (reward + self.gamma * best_next_value - current_q)
        self.q_table[state][action] = new_q
    
    def train(self, episodes: int) -> TrainingMetrics:
        """Train the agent for given number of episodes."""
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            done = False
            
            while not done and steps < self.env.max_steps:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state)
                
                total_reward += reward
                steps += 1
                state = next_state

            # Record metrics
            metrics = EpisodeMetrics(
                total_reward=total_reward,
                steps=steps,
                success=(state == self.env.goal_pos),
                final_position=state
            )
            self.metrics.add_episode(metrics)
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Print progress
            if (episode + 1) % 100 == 0:
                success_rate = self.metrics.get_success_rate(100)
                avg_reward = self.metrics.get_average_reward(100)
                avg_steps = self.metrics.get_average_steps(100)
                print(f"Episode {episode + 1}")
                print(f"Success Rate: {success_rate:.2%}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average Steps: {avg_steps:.2f}")
                print(f"Current Epsilon: {self.epsilon:.3f}")
                print("-" * 40)
            
        return self.metrics

def main() -> None:
    # Create environment
    env = GridWorld(
        width=50,
        height=40,
        start_pos=(0, 0),
        goal_pos=(40, 40),
        obstacles=[(1, 1), (2, 2), (3, 3), (39, 39)]
    )
    
    # Create and train agent
    agent = QLearningAgent(env)
    metrics = agent.train(episodes=1000)
    
    # Plot training metrics
    metrics.plot_metrics()

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Final Success Rate: {metrics.get_success_rate(100):.2%}")
    print(f"Final Average Reward: {metrics.get_average_reward(100):.2f}")
    print(f"Final Average Steps: {metrics.get_average_steps(100):.2f}")

    # Print final Q-table for important states
    print("\nFinal Q-value for key states:")
    key_states = [env.start_pos, (0, 1), (1, 0), env.goal_pos]
    for state in key_states:
        if state in agent.q_table:
            print(f"\nState {state}:")
            for action, value in agent.q_table[state].items():
                print(f"  {action.name}: {value:.2f}")

if __name__ == "__main__":
    main()
