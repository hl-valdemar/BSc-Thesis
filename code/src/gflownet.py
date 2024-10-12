import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.gridworld import GridWorld, Cell, Action

# Constants for state and action sizes
STATE_SIZE = 2  # x and y coordinates
ACTION_SIZE = 4  # LEFT, RIGHT, UP, DOWN

class GFlowNet(nn.Module):
    def __init__(self):
        super(GFlowNet, self).__init__()

        # Shared state embedding
        self.state_embedding = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
        )

        # Forward policy network P_F(s -> a)
        self.forward_policy = nn.Sequential(
            nn.Linear(128, ACTION_SIZE),
            nn.LogSoftmax(dim=-1)  # Use LogSoftmax for numerical stability
        )

        # State log flow log F(s)
        self.state_log_flow = nn.Linear(128, 1)

    def forward(self, state_tensor):
        embedding = self.state_embedding(state_tensor)
        log_forward_policy = self.forward_policy(embedding)
        log_flow = self.state_log_flow(embedding)
        return log_forward_policy, log_flow.squeeze()

def reward(state, env: GridWorld):
    x, y = state
    if env.grid[y, x] == Cell.GOAL.value:
        return 1.0  # Positive reward for reaching the goal
    else:
        return 0.0  # Zero reward for non-goal states


def trainer(
    env: GridWorld,
    num_episodes=10000,
    max_steps_per_episode=50,
    show_policy: bool = False,
):
    flow_network = GFlowNet()
    optimizer = optim.Adam(flow_network.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Get log probabilities and flow
            log_probs, flow = flow_network(state_tensor)

            # Sample an action
            action_distribution = torch.exp(log_probs).squeeze().detach().numpy()
            epsilon = max(0.1, 1 - episode / (num_episodes / 2))  # Decrease epsilon over time
            # if np.random.rand() < epsilon:
            #     action = np.random.choice(ACTION_SIZE)
            # else:
            action = np.random.choice(ACTION_SIZE, p=action_distribution)

            # Take a step in the environment
            next_state, cell = env.step(Action(action))

            # Store the transition
            trajectory.append((state, action, next_state))

            # Check if we've reached the goal
            if cell == Cell.GOAL.value:
                done = True

            state = next_state
            steps += 1

            if show_policy:
                current_optimal_policy = optimal_policy(env, flow_network)
                env.set_policy(current_optimal_policy, render_policy=True)

        # Compute the loss over the trajectory
        loss = compute_flow_loss(trajectory, flow_network, env)
        if torch.isnan(loss) or torch.isinf(loss):
            print("Encountered NaN or Inf in loss!")
            # Take appropriate action (e.g., skip the update, reset the network)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Loss: {loss.item()}")
            current_optimal_policy = optimal_policy(env, flow_network)
            env.set_policy(current_optimal_policy, render_policy=True)
            print("Current optimal policy:")
            for row in current_optimal_policy:
                print(" ".join([Action(action).name[0] if action != -1 else "#" for action in row]))

    return flow_network

def compute_flow_loss(trajectory, flow_network, env):
    loss = torch.tensor(0.0)
    trajectory_length = len(trajectory)

    for t in range(trajectory_length):
        state, action, next_state = trajectory[t]

        # Convert states to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Get log probabilities and log flows
        log_probs_s, log_flow_s = flow_network(state_tensor)
        log_probs_s_prime, log_flow_s_prime = flow_network(next_state_tensor)

        # Get the log probability of the taken action
        log_p_forward = log_probs_s[0, action]

        # Compute log F_in(s') = log F(s) + log P_F(s -> s')
        log_F_in_s_prime = log_flow_s + log_p_forward

        # Reward at next_state
        R_s_prime = reward(next_state, env)
        if R_s_prime > 0:
            log_R_s_prime = torch.log(torch.tensor(R_s_prime))
            # Use log-sum-exp to compute log(F_in(s') + R(s'))
            log_sum = torch.logsumexp(torch.stack([log_F_in_s_prime, log_R_s_prime]), dim=0)
        else:
            log_sum = log_F_in_s_prime  # No reward to add

        # Flow consistency at next_state: log_sum = log F(s')
        flow_consistency = log_sum - log_flow_s_prime

        # Accumulate squared error
        loss += flow_consistency.pow(2)

    loss = loss / trajectory_length
    return loss

def optimal_policy(env: GridWorld, flow_network: GFlowNet):
    policy_grid = np.full((env.height, env.width), -1, dtype=int)

    for y in range(env.height):
        for x in range(env.width):
            cell = env.grid[y, x]
            if cell == Cell.WALL.value or cell == Cell.OBSTACLE.value:
                policy_grid[y, x] = -1
                pass
            else:
                state_tensor = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0)
                log_probs, _ = flow_network(state_tensor)
                action_probs = torch.exp(log_probs).detach().numpy().squeeze()
                best_action = np.argmax(action_probs)
                policy_grid[y, x] = int(best_action)

    return policy_grid

if __name__ == "__main__":
    # description = """
    # ##########
    # #s #     #
    # #  #  #  #
    # #  #  o  #
    # #     # g#
    # ##########
    # """
    description = """
    ##########
    #s       #
    #        #
    #        #
    #       g#
    ##########
    """
    world = GridWorld(
        description,
        font_size=40,
        font_path="/usr/share/fonts/TTF/JetBrainsMonoNerdFontMono-Regular.ttf",
        render=True,
    )

    # Train the GFlowNet
    world.run_training(lambda w: trainer(w, num_episodes=20000, show_policy=True))
