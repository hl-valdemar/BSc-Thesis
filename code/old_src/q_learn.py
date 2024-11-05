import time
import random
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from lib.gridworld import Action, Cell, GridWorld, State

def trainer(
    world: GridWorld,
    num_episodes: int,
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    epsilon: float = 0.1,
    max_steps: int = 1000,
    follow_agent: bool = False,
    show_policy: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    Q = np.zeros((world.height, world.width, world.action_size))

    # Performance metrics
    metrics = {
        "episode_lengths": [],
        "cumulative_rewards": [],
        "success_rate": [],
        "average_q_values": [],
        "q_value_variance": [],
        "q_value_std": [],
    }

    # Pre-compute valid actions and rewards for each state
    valid_actions = np.ones((world.height, world.width, world.action_size), dtype=bool)
    rewards = np.full((world.height, world.width, world.action_size), -1)

    for y in range(world.height):
        for x in range(world.width):
            cell = world.grid[y, x]

            if cell == Cell.WALL.value:
                valid_actions[y, x, :] = False
                rewards[y, x, :] = -5
            elif cell == Cell.EMPTY.value:
                rewards[y, x, :] = -1
            elif cell == Cell.GOAL.value:
                rewards[y, x, :] = 100
            elif cell == Cell.OBSTACLE.value:
                rewards[y, x, :] = -100

            if x == 0:
                valid_actions[y, x, Action.LEFT.value] = False
            if x == world.width - 1:
                valid_actions[y, x, Action.RIGHT.value] = False
            if y == 0:
                valid_actions[y, x, Action.UP.value] = False
            if y == world.height - 1:
                valid_actions[y, x, Action.DOWN.value] = False

    for episode in range(num_episodes):
        if (episode) % 1000 == 0:
            print(f"Episode {episode}/{num_episodes}")

        state = world.reset()
        x, y = state
        cumulative_reward = 0
        steps = 0
        cell: Cell | None = None

        while steps < max_steps:
            steps += 1

            if np.random.random() < epsilon:
                action = np.random.choice(np.where(valid_actions[y, x])[0])
            else:
                action = np.argmax(Q[y, x, valid_actions[y, x]])

            next_state, cell = world.step(Action(action))
            next_x, next_y = next_state

            reward = rewards[y, x, action]

            best_next_action = np.argmax(Q[next_y, next_x])
            td_target = reward + discount_factor * Q[next_y, next_x, best_next_action]
            td_error = td_target - Q[y, x, action]
            Q[y, x, action] += learning_rate * td_error

            cumulative_reward += reward

            if cell == Cell.GOAL.value:  # Goal reached
                break

            x, y = next_x, next_y

            if show_policy:
                # Extract current optimal policy from Q-table
                policy = np.argmax(Q, axis=2)
                world.set_policy(policy, render_policy=True)

            if follow_agent:
                # Allow time for the policy updates to be seen one at a time
                time.sleep(0.1)

        # Update metrics
        metrics["episode_lengths"].append(steps)
        metrics["cumulative_rewards"].append(cumulative_reward)
        metrics["success_rate"].append(1 if cell == Cell.GOAL.value else 0)
        metrics["average_q_values"].append(np.mean(Q))
        metrics["q_value_variance"].append(np.var(Q))
        metrics["q_value_std"].append(np.std(Q))

    # Extract optimal policy from Q-table
    policy = np.argmax(Q, axis=2)

    return Q, policy, metrics

def plot_metrics(metrics: dict):
    plt.figure(figsize=(15, 11))

    # Episode Length
    plt.subplot(3, 2, 1)
    plt.plot(metrics['episode_lengths'])
    plt.title('Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    # Cumulative Reward
    plt.subplot(3, 2, 2)
    plt.plot(metrics['cumulative_rewards'])
    plt.title('Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Success Rate
    plt.subplot(3, 2, 3)
    plt.plot(metrics['success_rate'])
    plt.title('Success Rate')
    plt.xlabel('Episode')
    plt.ylabel('Rate')

    # Q-value
    plt.subplot(3, 2, 4)
    plt.plot(metrics['average_q_values'])
    plt.title('Q-value')
    plt.xlabel('Episode')
    plt.ylabel('Value')

    # Q-value Variance
    plt.subplot(3, 2, 5)
    plt.plot(metrics['q_value_variance'])
    plt.title('Q-value Variance')
    plt.xlabel('Episode')
    plt.ylabel('Variance')

    # Average Q-value with Variance
    plt.subplot(3, 2, 6)
    episodes = range(len(metrics['average_q_values']))
    avg_q = np.array(metrics['average_q_values'])
    std_q = np.array(metrics['q_value_std'])

    plt.plot(episodes, avg_q, label='Average Q-value')
    plt.fill_between(episodes, avg_q - std_q, avg_q + std_q, alpha=0.3, label='Q-value Std Dev')
    plt.title('Average Q-value with Variance')
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    description = """
    ##########
    #s #     #
    #  #  #  #
    #  #  o  #
    #     # g#
    ##########
    """
    world = GridWorld(
        description,
        font_size=40,
        font_path="/usr/share/fonts/TTF/JetBrainsMonoNerdFontMono-Regular.ttf",
        render=True,
    )

    # Train with Q-learning
    Q, policy, metrics = world.run_training(lambda w: trainer(w, num_episodes=20000))
    print("Q-learning Policy:")
    for row in policy:
        print(' '.join([Action(action).name[0] for action in row]))

    plot_metrics(metrics)
