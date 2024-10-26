import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm

from lib.gridworld import Cell, GridWorld, State

# GRID_SIZE: tuple[int, int] = (5, 5)  # Grid dimensions (rows, columns)
START_POS: State = (0, 0)  # Starting position (row, column)
GOAL_POS: State = (4, 4)   # Goal position
# ACTIONS: list[str] = ['up', 'down', 'left', 'right']  # Possible actions
# ACTION_DICT = {
#     'up': (-1, 0),
#     'down': (1, 0),
#     'left': (0, -1),
#     'right': (0, 1)
# }

def state_to_tensor(world: GridWorld, state: State) -> torch.Tensor:
    """Converts the grid position to a one-hot tensor."""
    tensor = torch.zeros(world.width, world.height)
    tensor[state] = 1.0
    return tensor.flatten()

def get_reward(world: GridWorld, state: State) -> float:
    """Returns the reward for a given state."""
    x, y = state
    if world.grid[y, x] == Cell.GOAL.value:
        return 1.0  # Positive reward for reaching the goal
    else:
        return 0.0

class FlowModel(nn.Module):
    def __init__(self, world: GridWorld, num_hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(world.width * world.height, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, len(world.actions)),
        )

    def forward(self, state_tensor: torch.Tensor, available_actions_mask):
        logits = self.mlp(state_tensor)
        # Ensure positive flows and mask unavailable actions
        flows = (logits.exp() * available_actions_mask)
        return flows

def trainer(world: GridWorld):
    # Initialize the model and optimizer
    flow_model = FlowModel(world)
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-3)

    # Training parameters
    num_episodes = 10000
    update_freq = 10  # Update model every 'update_freq' episodes
    minibatch_loss = 0
    losses = []

    for episode in tqdm.tqdm(range(num_episodes)):
        state = START_POS
        trajectory: list[tuple[int, int]] = [state]
        state_tensor = state_to_tensor(world, state)
        done = False

        while not done:
            # Get available actions and mask
            available_actions, action_mask = world.get_valid_actions_for_state(state)
            action_mask = action_mask.float()

            # Predict flows F(s, a)
            flows = flow_model(state_tensor, action_mask)

            # Compute policy
            policy = flows / flows.sum()

            # Sample action
            action_dist = Categorical(probs=policy)
            action_idx = action_dist.sample().item()
            action = world.actions[action_idx]

            # Move to next state
            delta = world.action_dict[action]
            new_state = (state[0] + delta[0], state[1] + delta[1])
            # new_state, _ = world.step(action)
            new_state_tensor = state_to_tensor(world, new_state)
            trajectory.append(new_state)

            # Get parents of the new state
            parent_states, parent_actions = world.get_parents_for_state(new_state)

            # Compute parent flows
            parent_flows = []
            for parent_state, parent_action_idx in zip(parent_states, parent_actions):
                parent_state_tensor = state_to_tensor(world, parent_state)
                _, parent_action_mask = world.get_valid_actions_for_state(parent_state)
                parent_action_mask = parent_action_mask.float()
                parent_flow = flow_model(parent_state_tensor, parent_action_mask)[parent_action_idx]
                parent_flows.append(parent_flow)
            parent_flows = torch.stack(parent_flows)

            # Compute child flows
            _, child_action_mask = world.get_valid_actions_for_state(new_state)
            child_action_mask = child_action_mask.float()
            if new_state == GOAL_POS:
                child_flows = torch.zeros(len(world.actions))
                reward = get_reward(world, new_state)
                done = True
            else:
                child_flows = flow_model(new_state_tensor, child_action_mask)
                reward = 0.0

            # Flow matching loss
            total_in_flow = parent_flows.sum()
            total_out_flow = child_flows.sum() + reward
            flow_loss = (total_in_flow - total_out_flow).pow(2)
            minibatch_loss += flow_loss

            # Prepare for next iteration
            state = new_state
            state_tensor = new_state_tensor

        # Update model
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            minibatch_loss = 0

    plt.plot(losses)
    plt.xlabel('Updates')
    plt.ylabel('Flow Matching Loss')
    plt.title('Training Loss')
    plt.show()

def generate_trajectory(flow_model):
    state = START_POS
    trajectory = [state]
    state_tensor = state_to_tensor(world, state)
    done = False

    while not done:
        available_actions, action_mask = world.get_valid_actions_for_state(state)
        action_mask = action_mask.float()
        flows = flow_model(state_tensor, action_mask)
        policy = flows / flows.sum()
        action_idx = torch.argmax(policy).item()  # Choose action with highest probability
        action = world.actions[action_idx]
        delta = world.action_dict[action]
        new_state = (state[0] + delta[0], state[1] + delta[1])
        trajectory.append(new_state)
        if new_state == GOAL_POS:
            done = True
        state = new_state
        state_tensor = state_to_tensor(world, state)
    return trajectory

# def plot_trajectory(trajectory):
#     fig, ax = plt.subplots()
#     ax.set_xlim(-0.5, GRID_SIZE[1] - 0.5)
#     ax.set_ylim(-0.5, GRID_SIZE[0] - 0.5)
#     ax.set_xticks(range(GRID_SIZE[1]))
#     ax.set_yticks(range(GRID_SIZE[0]))
#     ax.invert_yaxis()
#     ax.grid(True)

#     # Plot start and goal
#     ax.add_patch(patches.Rectangle((START_POS[1] - 0.5, START_POS[0] - 0.5), 1, 1, facecolor='green', alpha=0.5))
#     ax.add_patch(patches.Rectangle((GOAL_POS[1] - 0.5, GOAL_POS[0] - 0.5), 1, 1, facecolor='red', alpha=0.5))

#     # Plot trajectory
#     for i in range(len(trajectory) - 1):
#         current = trajectory[i]
#         next_state = trajectory[i + 1]
#         plt.arrow(
#             current[1], current[0],
#             next_state[1] - current[1], next_state[0] - current[0],
#             head_width=0.1, length_includes_head=True, color='blue'
#         )

#     plt.show()

# # Generate and plot a trajectory
# trajectory = generate_trajectory()
# plot_trajectory(trajectory)

description = """
#####
#s  #
#   #
#  g#
#####
"""
world = GridWorld(description, render=True)

# world.run_training(lambda w: trainer(w))

trainer(world)
