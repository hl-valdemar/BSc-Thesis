import time
import random
import numpy as np
from numpy.typing import NDArray
from lib.gridworld import Action, Cell, GridWorld

def q_learning(
    world: GridWorld,
    num_episodes: int,
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    epsilon: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    Q = np.zeros((world.height, world.width, world.action_size))

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
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode+1}/{num_episodes}")

        state = world.reset()
        x, y = state

        while True:
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

            if cell == Cell.GOAL.value:  # Goal reached
                break

            x, y = next_x, next_y

            # Extract policy from Q-table
            policy = np.argmax(Q, axis=2)
            world.set_policy(policy, render_policy=True)

    # Extract policy from Q-table
    policy = np.argmax(Q, axis=2)

    return Q, policy

# Setup GridWorld
# description = """
# ##########
# #s #     #
# #  #  #  #
# #  #  o  #
# #     # g#
# ##########
# """
description = """
###################
#s #     #g #     #
#  #  #  #  #  #  #
#  #  o  #  #  o  #
#     #  #     #  #
#oo####  #oo####  #
#g #     #  #     #
#  #  #  #  #  #  #
#  #  o     #  o  #
#     #  #     #  #
###################
"""
world = GridWorld(
    description,
    font_size=40,
    font_path="/usr/share/fonts/TTF/JetBrainsMonoNerdFontMono-Regular.ttf",
    render=True,
)

# Run Q-learning
num_episodes = 20000
Q, policy = world.run_training(lambda w: q_learning(w, num_episodes))
print("Q-learning Policy:")
for row in policy:
    print(' '.join([Action(action).name[0] for action in row]))
