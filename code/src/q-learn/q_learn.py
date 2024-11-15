from typing import Dict, NamedTuple, Optional, Tuple, List
import numpy as np
import random
import matplotlib.pyplot as plt
from gridworld import GridWorld, Action

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
        success_ma = np.convolve(successes, np.ones(window)/window, mode="valid")

        # Create figure with shared x-axis
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # Add super title
        fig.suptitle("Q-Learning Training Metrics", fontsize=16, y=0.95)

        # Plot rewards
        l1, = axes[0].plot(episodes, rewards, alpha=0.3, color="lightblue", label="Raw Rewards")
        l2, = axes[0].plot(range(window, len(episodes) + 1), reward_ma, 
                          color="blue", linewidth=2, label=f"{window}-Episode Moving Average")
        axes[0].set_title("Episode Rewards", pad=10)
        axes[0].set_ylabel("Total Reward")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(handles=[l1, l2], loc="upper right", framealpha=0.9)
        
        # Plot steps
        l3, = axes[1].plot(episodes, steps, alpha=0.3, color="lightgreen", label="Raw Steps")
        l4, = axes[1].plot(range(window, len(episodes) + 1), steps_ma, 
                          color="green", linewidth=2, label=f"{window}-Episode Moving Average")
        axes[1].set_title("Steps per Episode", pad=10)
        axes[1].set_ylabel("Number of Steps")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(handles=[l3, l4], loc="upper right", framealpha=0.9)
        
        # Plot success rate
        l5, = axes[2].plot(episodes, successes, alpha=0.3, color="orange", label="Raw Success")
        l6, = axes[2].plot(range(window, len(episodes) + 1), success_ma, 
                          color="orange", linewidth=2, label=f"{window}-Episode Moving Average")
        axes[2].set_title("Success Rate", pad=10)
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Success Rate")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(handles=[l5, l6], loc="upper right", framealpha=0.9)
        
        # Set y-axis limits for success rate
        axes[2].set_ylim(-0.05, 1.05)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()

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
 
    def get_policy_grid(self) -> str:
        """Returns a string representation of the learned policy"""
        # Define arrow symbols for each action
        action_symbols = {
            Action.UP: "↑",
            Action.RIGHT: "→",
            Action.DOWN: "↓",
            Action.LEFT: "←"
        }
        
        # Create grid representation
        grid_rows = []
        for y in range(self.env.height - 1, -1, -1):  # Reverse y to match coordinate system
            row = []
            for x in range(self.env.width):
                pos = (x, y)
                if pos == self.env.start_pos:
                    cell = "S"
                elif pos == self.env.goal_pos:
                    cell = "G"
                elif pos in self.env.obstacles:
                    cell = "#"
                else:
                    # Get best action for this position
                    if pos in self.q_table:
                        best_action = max(self.q_table[pos].items(), key=lambda x: x[1])[0]
                        cell = action_symbols[best_action]
                    else:
                        cell = " "
                row.append(cell)
            grid_rows.append(" ".join(row))
        
        # Add border
        width = len(grid_rows[0])
        border = "+" + "-" * (width + 2) + "+"
        grid_with_border = [border]
        for row in grid_rows:
            grid_with_border.append(f"| {row} |")
        grid_with_border.append(border)
        
        return "\n".join(grid_with_border)

    def print_policy(self) -> None:
        """Prints the learned policy with a legend"""
        print("\nLearned Policy Grid:")
        print("S: Start")
        print("G: Goal")
        print("#: Obstacle")
        print("↑: Up")
        print("→: Right")
        print("↓: Down")
        print("←: Left")
        print()
        print(self.get_policy_grid())
    
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
    # Create environment from parameter settings
    # env = GridWorld(
    #     width=5,
    #     height=5,
    #     start_pos=(0, 0),
    #     goal_pos=(4, 4),
    #     obstacles=[(1, 1), (2, 2), (3, 3)]
    # )

    # Create environment from layout definition
    world_layout = """
    #######
    #s    #
    # o   #
    #  o  #
    #   o #
    #    g#
    #######
    """
    env = GridWorld(layout=world_layout)
    
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

    # Print the learned policy grid
    print("\nFinal Policy:")
    agent.print_policy()

if __name__ == "__main__":
    main()
