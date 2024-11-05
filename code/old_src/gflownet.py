import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm

GRID_SIZE: tuple[int, int] = (5, 5)  # Grid dimensions (rows, columns)
START_POS: tuple[int, int] = (0, 0)  # Starting position (row, column)
GOAL_POS: tuple[int, int] = (4, 4)   # Goal position
ACTIONS: list[str] = ['up', 'down', 'left', 'right']  # Possible actions
ACTION_DICT = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

def state_to_tensor(state):
    """Converts the grid position to a one-hot tensor."""
    tensor = torch.zeros(GRID_SIZE[0], GRID_SIZE[1])
    tensor[state] = 1.0
    return tensor.flatten()

def get_reward(state):
    if state == GOAL_POS:
        return 1.0  # Positive reward for reaching the goal
    else:
        return 0.0

class FlowModel(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(GRID_SIZE[0] * GRID_SIZE[1], num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, len(ACTIONS)),
        )

    def forward(self, state_tensor, available_actions_mask):
        logits = self.mlp(state_tensor) # Shape: [batch_size, len(ACTIONS)]
        # Ensure positive flows and mask unavailable actions
        flows = (logits.exp() * available_actions_mask)
        return flows

def get_available_actions(state):
    if state == GOAL_POS:
           return [], torch.zeros(len(ACTIONS))

    available = []
    mask = torch.zeros(len(ACTIONS))
    for i, action in enumerate(ACTIONS):
        delta = ACTION_DICT[action]
        new_row = state[0] + delta[0]
        new_col = state[1] + delta[1]
        if 0 <= new_row < GRID_SIZE[0] and 0 <= new_col < GRID_SIZE[1]:
            available.append(action)
            mask[i] = 1.0
    return available, mask

def get_parents(state):
    parents = []
    parent_actions = []
    for i, action in enumerate(ACTIONS):
        delta = ACTION_DICT[action]
        prev_row = state[0] - delta[0]
        prev_col = state[1] - delta[1]
        if 0 <= prev_row < GRID_SIZE[0] and 0 <= prev_col < GRID_SIZE[1]:
            parents.append((prev_row, prev_col))
            parent_actions.append(i)
    return parents, parent_actions

# Initialize the model and optimizer
flow_model = FlowModel()
optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-3)

# Training parameters
num_episodes = 5000
update_freq = 10  # Update model every 'update_freq' episodes
losses = []

for episode in tqdm.tqdm(range(num_episodes)):
    state = START_POS
    trajectory: list[tuple[int, int]] = [state]
    state_tensor = state_to_tensor(state)
    done = False

    while not done:
        minibatch_loss = torch.tensor(0.0, requires_grad=True)

        # Get available actions and mask
        available_actions, action_mask = get_available_actions(state)
        action_mask = action_mask.float()

        # Predict flows F(s, a)
        flows = flow_model(state_tensor, action_mask)
        assert flows.shape == (len(ACTIONS),)

        # Compute policy (avoid division by zero)
        epsilon = 1e-8
        policy = flows / (flows.sum() + epsilon)

        # Sample action
        action_dist = Categorical(probs=policy)
        action_idx = action_dist.sample().item()
        action = ACTIONS[action_idx]

        # Move to next state
        delta = ACTION_DICT[action]
        new_state = (state[0] + delta[0], state[1] + delta[1])
        new_state_tensor = state_to_tensor(new_state)
        trajectory.append(new_state)

        # Get parents of the new state
        parent_states, parent_actions = get_parents(new_state)

        # Compute parent flows
        parent_flows = []
        for parent_state, parent_action_idx in zip(parent_states, parent_actions):
            parent_state_tensor = state_to_tensor(parent_state)
            _, parent_action_mask = get_available_actions(parent_state)
            parent_action_mask = parent_action_mask.float()
            parent_flow = flow_model(parent_state_tensor, parent_action_mask)[parent_action_idx]
            parent_flows.append(parent_flow)
        parent_flows = torch.stack(parent_flows)

        # Compute child flows
        _, child_action_mask = get_available_actions(new_state)
        child_action_mask = child_action_mask.float()
        if new_state == GOAL_POS:
            child_flows = torch.zeros(len(ACTIONS))
            reward = get_reward(new_state)
            done = True
        else:
            child_flows = flow_model(new_state_tensor, child_action_mask)
            reward = 0.0

        # Flow matching loss
        total_in_flow = parent_flows.sum()
        total_out_flow = child_flows.sum() + reward
        flow_loss = (total_in_flow - total_out_flow).pow(2)
        minibatch_loss = minibatch_loss + flow_loss

        # Prepare for next iteration
        state = new_state
        state_tensor = new_state_tensor

        # Update model
        minibatch_loss = minibatch_loss / len(trajectory) # Normalize the loss
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

plt.plot(np.log(losses))
plt.xlabel('Updates')
plt.ylabel('Log Flow Matching Loss')
plt.title('Log Training Loss')
plt.show()

def generate_trajectory(max_steps: int = 100, temperature: float = 1.0) -> list[tuple[int, int]]:
    state = START_POS
    trajectory = [state]
    state_tensor = state_to_tensor(state)
    done = False
    steps = 0
    visited_states = set()
    visited_states.add(state)

    while not done and steps < max_steps:
        available_actions, action_mask = get_available_actions(state)
        action_mask = action_mask.float()
        flows = flow_model(state_tensor, action_mask)
        policy = flows / flows.sum()

        # # Apply temperature to policy
        # policy = policy ** (1 / temperature)
        # policy = policy / policy.sum()

        # Sample action from policy
        action_dist = Categorical(probs=policy)
        action_idx = action_dist.sample().item()
        action = ACTIONS[action_idx]

        # Move to next state
        delta = ACTION_DICT[action]
        new_state = (state[0] + delta[0], state[1] + delta[1])
        trajectory.append(new_state)

        # Check if the new state is the goal
        if new_state == GOAL_POS:
            done = True
        elif new_state in visited_states:
            print("Detected a loop in the trajectory. Terminating.")
            break
        else:
            visited_states.add(new_state)

        # Prepare for next iteration
        state = new_state
        state_tensor = state_to_tensor(state)
        steps += 1

    if not done:
        print("Failed to reach the goal within the maximum number of steps.")
    return trajectory

def plot_trajectory(trajectory):
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, GRID_SIZE[1] - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE[0] - 0.5)
    ax.set_xticks(range(GRID_SIZE[1]))
    ax.set_yticks(range(GRID_SIZE[0]))
    ax.invert_yaxis()
    ax.grid(True)

    # Plot start and goal
    ax.add_patch(patches.Rectangle((START_POS[1] - 0.5, START_POS[0] - 0.5), 1, 1, facecolor='green', alpha=0.5))
    ax.add_patch(patches.Rectangle((GOAL_POS[1] - 0.5, GOAL_POS[0] - 0.5), 1, 1, facecolor='red', alpha=0.5))

    # Plot trajectory
    for i in range(len(trajectory) - 1):
        current = trajectory[i]
        next_state = trajectory[i + 1]
        plt.arrow(
            current[1], current[0],
            next_state[1] - current[1], next_state[0] - current[0],
            head_width=0.1, length_includes_head=True, color='blue'
        )

    plt.show()

# Generate and plot a trajectory
trajectory = generate_trajectory()
retries = 0
while not GOAL_POS in trajectory:
    retries += 1
    print(f"Trajectory did not lead to goal: retrying... ({retries})")
    trajectory = generate_trajectory()
plot_trajectory(trajectory)
