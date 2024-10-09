from lib.gridworld import GridWorld
from gflownet import GFlowNet
import time

import time
import random

def train(world: GridWorld):
    while not world.finished_training:
        print("running")
        actions = world.get_actions()
        action = random.choice(actions)
        new_state, goal_reached = world.step(action)
        time.sleep(0.1)
        if goal_reached:
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
