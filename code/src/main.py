from lib.gridworld import GridWorld
from gflownet import GFlowNet
import time

import time
import random

def train(world: GridWorld):
    while world.running:
        actions = world.get_actions()
        action = random.choice(actions)
        new_state, done = world.step(action)
        time.sleep(0.1)
        if done:
            break

description = """
##########
#s #     #
#  #  #  #
#  #  o  #
#     # g#
##########
"""

world = GridWorld(description, render=True)
world.run_training(train)
