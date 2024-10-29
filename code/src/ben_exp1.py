import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

# GridWorld Environment
class GridWorld:
    def __init__(self, size=5, n_hazards=3, n_victims=2):
        self.size = size
        self.n_hazards = n_hazards
        self.n_victims = n_victims
        self.reset()

    def reset(self):
        # Initialize agent at center
        self.agent_pos = np.array([self.size//2, self.size//2])

        # Place hazards and victims randomly on edges
        edge_positions = self._get_edge_positions()
        positions = np.random.choice(len(edge_positions),
                                   size=self.n_hazards + self.n_victims,
                                   replace=False)

        self.hazards = [edge_positions[i] for i in positions[:self.n_hazards]]
        self.victims = [edge_positions[i] for i in positions[self.n_hazards:]]
        self.victims_saved = []

        # Initialize last_action
        self.last_action = None

        return self._get_state()

    def _get_edge_positions(self):
        edges = []
        for i in range(self.size):
            edges.extend([(i, 0), (i, self.size-1), (0, i), (self.size-1, i)])
        return edges

    def _get_state(self):
        state = np.zeros(2 + self.n_victims + self.n_hazards)
        state[0:2] = self.agent_pos

        # Add victim and hazard distances if listening
        if self.last_action == 'listen' or self.last_action is None:  # Also give information at start
            for i, victim in enumerate(self.victims):
                if victim not in self.victims_saved:
                    dist = np.linalg.norm(self.agent_pos - np.array(victim))
                    state[2+i] = np.exp(-dist/self.size + np.random.normal(0, 0.1))

            for i, hazard in enumerate(self.hazards):
                dist = np.linalg.norm(self.agent_pos - np.array(hazard))
                state[2+self.n_victims+i] = np.exp(-dist/self.size + np.random.normal(0, 0.1))

        return state

    def step(self, action):
        self.last_action = action
        old_pos = self.agent_pos.copy()
        reward = 0
        done = False

        if action == 'up':
            if self.agent_pos[1] < self.size-1:
                self.agent_pos[1] += 1
            else:
                # Try to rescue/hit hazard above
                pos = (self.agent_pos[0], self.size-1)
                reward = self._check_rescue_hazard(pos)

        elif action == 'down':
            if self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
            else:
                pos = (self.agent_pos[0], 0)
                reward = self._check_rescue_hazard(pos)

        elif action == 'left':
            if self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
            else:
                pos = (0, self.agent_pos[1])
                reward = self._check_rescue_hazard(pos)

        elif action == 'right':
            if self.agent_pos[0] < self.size-1:
                self.agent_pos[0] += 1
            else:
                pos = (self.size-1, self.agent_pos[1])
                reward = self._check_rescue_hazard(pos)

        elif action == 'listen':
            reward = -1

        # Check if all victims saved
        if len(self.victims_saved) == len(self.victims):
            done = True

        return self._get_state(), reward, done

    def _check_rescue_hazard(self, pos):
        if pos in self.hazards:
            return -100
        elif pos in self.victims and pos not in self.victims_saved:
            self.victims_saved.append(pos)
            return 10
        return 0

# Recurrent Q-Network
class RecurrentQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)  # Add sequence dimension
        x, hidden = self.gru(x, hidden)
        x = self.fc2(x.squeeze(1))
        return x, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)

class BENAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # Initialize networks
        self.q_net = RecurrentQNetwork(state_size, hidden_size, action_size)
        self.aleatoric_net = AleatoricNetwork(hidden_size)
        self.epistemic_net = EpistemicNetwork(hidden_size, hidden_size)

        # Use even smaller learning rates
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-4, eps=1e-8)
        self.aleatoric_optimizer = torch.optim.Adam(self.aleatoric_net.parameters(), lr=1e-4, eps=1e-8)
        self.epistemic_optimizer = torch.optim.Adam(self.epistemic_net.parameters(), lr=1e-4, eps=1e-8)

        self.memory = deque(maxlen=10000)
        self.hidden = self.q_net.init_hidden()

        # Initialize running statistics
        self.state_mean = np.zeros(state_size)
        self.state_std = np.ones(state_size)
        self.state_n = 0

        # Add reward normalization
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_n = 0

    def update_reward_stats(self, reward):
        self.reward_n += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_n
        delta2 = reward - self.reward_mean
        self.reward_std = np.sqrt(
            (self.reward_std ** 2 * (self.reward_n - 1) + delta * delta2) / self.reward_n
        )

    def normalize_reward(self, reward):
        return (reward - self.reward_mean) / (self.reward_std + 1e-8)

    def select_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(['up', 'down', 'left', 'right', 'listen'])

        try:
            # Normalize state
            state = self.normalize_state(state)
            state = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                q_values, self.hidden = self.q_net(state, self.hidden)

                # Simple action selection without aleatoric sampling during exploration
                action_idx = q_values.argmax(dim=-1).item()

            return ['up', 'down', 'left', 'right', 'listen'][action_idx]

        except Exception as e:
            print(f"Error in select_action: {e}")
            # Fallback to random action if there's an error
            return np.random.choice(['up', 'down', 'left', 'right', 'listen'])

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        try:
            # Sample and prepare batch
            batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
            batch = [self.memory[i] for i in batch_indices]

            # Convert to numpy arrays first
            states = np.vstack([self.normalize_state(item[0]) for item in batch])
            actions = [item[1] for item in batch]
            rewards = np.array([self.normalize_reward(item[2]) for item in batch])
            next_states = np.vstack([self.normalize_state(item[3]) for item in batch])
            dones = np.array([item[4] for item in batch])

            # Convert to tensors
            states = torch.FloatTensor(states)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Initialize hidden states
            hidden = self.q_net.init_hidden(batch_size)

            # Get current Q values
            q_values, hidden = self.q_net(states, hidden)
            action_indices = torch.tensor([['up', 'down', 'left', 'right', 'listen'].index(a) for a in actions])
            current_q = q_values.gather(1, action_indices.unsqueeze(1))

            # Get next Q values
            with torch.no_grad():
                next_hidden = self.q_net.init_hidden(batch_size)
                next_q_values, _ = self.q_net(next_states, next_hidden)
                next_q = next_q_values.max(dim=1)[0]

            # Compute target Q values
            target_q = rewards + (1 - dones) * 0.99 * next_q.detach()

            # Update Q network
            q_loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.q_optimizer.step()

        except Exception as e:
            print(f"Error in update: {e}")

    def store_transition(self, state, action, reward, next_state, done):
        self.update_state_stats(state)
        self.update_reward_stats(reward)
        self.memory.append((state, action, reward, next_state, done))

    def reset(self):
        self.hidden = self.q_net.init_hidden()

    def normalize_state(self, state):
        state = np.array(state)
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def update_state_stats(self, state):
        state = np.array(state)
        self.state_n += 1
        delta = state - self.state_mean
        self.state_mean += delta / self.state_n
        delta2 = state - self.state_mean
        self.state_std = np.sqrt(
            (self.state_std ** 2 * (self.state_n - 1) + delta * delta2) / self.state_n
        )

    def select_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(['up', 'down', 'left', 'right', 'listen'])

        # Normalize state
        state = self.normalize_state(state)
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values, self.hidden = self.q_net(state, self.hidden)
            aleatoric_dist = self.aleatoric_net(q_values, self.hidden)

            # Use mean of distribution for action selection
            action_idx = q_values.argmax(dim=-1).item()

        return ['up', 'down', 'left', 'right', 'listen'][action_idx]

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        # Sample and prepare batch
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]

        states = np.vstack([self.normalize_state(item[0]) for item in batch])
        actions = [item[1] for item in batch]
        rewards = np.array([item[2] for item in batch])
        next_states = np.vstack([self.normalize_state(item[3]) for item in batch])
        dones = np.array([item[4] for item in batch])

        # Convert to tensors
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Initialize hidden states
        hidden = self.q_net.init_hidden(batch_size)

        # Update epistemic network
        epistemic_dist = self.epistemic_net(hidden)
        kl_loss = -epistemic_dist.entropy().mean()

        self.epistemic_optimizer.zero_grad()
        kl_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.epistemic_net.parameters(), 1.0)
        self.epistemic_optimizer.step()

        # Get current Q values
        q_values, hidden = self.q_net(states, hidden)
        action_indices = torch.tensor([['up', 'down', 'left', 'right', 'listen'].index(a) for a in actions])
        current_q = q_values.gather(1, action_indices.unsqueeze(1))

        # Get next Q values
        with torch.no_grad():
            next_hidden = self.q_net.init_hidden(batch_size)
            next_q_values, _ = self.q_net(next_states, next_hidden)
            max_next_q = next_q_values.max(dim=1)[0]

        # Sample from aleatoric distribution
        aleatoric_dist = self.aleatoric_net(next_q_values, next_hidden)

        # Compute target Q values
        target_q = rewards + (1 - dones) * 0.99 * max_next_q.detach()

        # Compute losses
        q_loss = F.smooth_l1_loss(current_q.squeeze(), target_q)  # Use Huber loss for stability
        aleatoric_loss = -aleatoric_dist.log_prob(next_q_values).mean()

        # Update networks with gradient clipping
        self.q_optimizer.zero_grad()
        self.aleatoric_optimizer.zero_grad()

        total_loss = q_loss + 0.1 * aleatoric_loss  # Reduce weight of aleatoric loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.aleatoric_net.parameters(), 1.0)

        self.q_optimizer.step()
        self.aleatoric_optimizer.step()

    def store_transition(self, state, action, reward, next_state, done):
        self.update_state_stats(state)
        self.memory.append((state, action, reward, next_state, done))

    def reset(self):
        self.hidden = self.q_net.init_hidden()

    def select_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(['up', 'down', 'left', 'right', 'listen'])

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values, self.hidden = self.q_net(state, self.hidden)

        # Sample from aleatoric distribution
        aleatoric_dist = self.aleatoric_net(q_values, self.hidden)
        sampled_value = aleatoric_dist.rsample()

        action_idx = q_values.argmax(dim=-1).item()
        return ['up', 'down', 'left', 'right', 'listen'][action_idx]

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]

        # Convert batch to numpy arrays first
        states = np.array([item[0] for item in batch])
        actions = [item[1] for item in batch]
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])

        # Convert to tensors
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Initialize hidden states for batch
        hidden = self.q_net.init_hidden(batch_size)

        # Update epistemic network
        epistemic_dist = self.epistemic_net(hidden)
        kl_loss = -epistemic_dist.entropy().mean()

        self.epistemic_optimizer.zero_grad()
        kl_loss.backward(retain_graph=True)  # Add retain_graph=True
        self.epistemic_optimizer.step()

        # Get current Q values
        q_values, hidden = self.q_net(states, hidden)
        action_indices = torch.tensor([['up', 'down', 'left', 'right', 'listen'].index(a) for a in actions])
        current_q = q_values.gather(1, action_indices.unsqueeze(1))

        # Get next Q values
        next_hidden = self.q_net.init_hidden(batch_size)
        next_q_values, _ = self.q_net(next_states, next_hidden)

        # Get maximum Q value for next states
        max_next_q = next_q_values.max(dim=1)[0]

        # Sample from aleatoric distribution
        aleatoric_dist = self.aleatoric_net(next_q_values, next_hidden)
        next_q = aleatoric_dist.rsample()

        # Compute target Q values (use max_next_q instead of next_q)
        target_q = rewards + (1 - dones) * 0.99 * max_next_q.detach()

        # Compute losses
        q_loss = F.mse_loss(current_q.squeeze(), target_q)
        aleatoric_loss = -aleatoric_dist.log_prob(next_q).mean()

        # Update networks
        self.q_optimizer.zero_grad()
        self.aleatoric_optimizer.zero_grad()

        total_loss = q_loss + aleatoric_loss
        total_loss.backward()

        self.q_optimizer.step()
        self.aleatoric_optimizer.step()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def reset(self):
        self.hidden = self.q_net.init_hidden()

# Aleatoric Network (Normalizing Flow)
class AleatoricNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + 5, hidden_size),  # 5 is action_size
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # μ and σ for Normal distribution
        )

    def forward(self, q_values, hidden):
        # Ensure proper dimensions
        hidden = hidden.squeeze(0)  # Remove sequence dimension (1, batch, hidden) -> (batch, hidden)
        x = torch.cat([q_values, hidden], dim=1)  # Concatenate along feature dimension
        params = self.net(x)

        # Return distribution with proper shape
        return Normal(params[:, 0].unsqueeze(-1), F.softplus(params[:, 1]).unsqueeze(-1))

# Epistemic Network (Normalizing Flow)
class EpistemicNetwork(nn.Module):
    def __init__(self, param_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, param_size * 2)  # μ and σ for each parameter
        )

    def forward(self, hidden):
        params = self.net(hidden.squeeze(0))
        size = params.size(-1) // 2
        return Normal(params[:, :size], F.softplus(params[:, size:]))

# BEN Agent
class BENAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.q_net = RecurrentQNetwork(state_size, hidden_size, action_size)
        self.aleatoric_net = AleatoricNetwork(hidden_size)
        self.epistemic_net = EpistemicNetwork(hidden_size, hidden_size)

        self.q_optimizer = torch.optim.Adam(self.q_net.parameters())
        self.aleatoric_optimizer = torch.optim.Adam(self.aleatoric_net.parameters())
        self.epistemic_optimizer = torch.optim.Adam(self.epistemic_net.parameters())

        self.memory = deque(maxlen=10000)
        self.hidden = self.q_net.init_hidden()

    def select_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(['up', 'down', 'left', 'right', 'listen'])

        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        q_values, hidden = self.q_net(state, self.hidden)
        self.hidden = hidden  # Update hidden state

        # Sample from aleatoric distribution
        aleatoric_dist = self.aleatoric_net(q_values, hidden)
        sampled_value = aleatoric_dist.rsample()

        # Use original q_values for action selection
        action_idx = q_values.argmax(dim=-1).item()
        return ['up', 'down', 'left', 'right', 'listen'][action_idx]

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        # Sample batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch])
        actions = [self.memory[i][1] for i in batch]
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch])
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch])
        dones = torch.FloatTensor([self.memory[i][4] for i in batch])

        # Update epistemic network
        hidden = self.q_net.init_hidden(batch_size)
        epistemic_dist = self.epistemic_net(hidden)
        kl_loss = -epistemic_dist.entropy().mean()  # Negative entropy as KL with prior

        self.epistemic_optimizer.zero_grad()
        kl_loss.backward()
        self.epistemic_optimizer.step()

        # Get current Q values
        q_values, hidden = self.q_net(states, hidden)
        action_idx = torch.tensor([['up', 'down', 'left', 'right', 'listen'].index(a) for a in actions])
        current_q = q_values.gather(1, action_idx.unsqueeze(1))

        # Get next Q values with aleatoric and epistemic uncertainty
        next_q_values, next_hidden = self.q_net(next_states, hidden)
        aleatoric_dist = self.aleatoric_net(next_q_values, next_hidden)
        next_q = aleatoric_dist.rsample()

        # Compute target Q values
        target_q = rewards + (1 - dones) * 0.99 * next_q.max(1)[0].detach()

        # Compute losses
        q_loss = F.mse_loss(current_q, target_q.unsqueeze(1))
        aleatoric_loss = -aleatoric_dist.log_prob(next_q).mean()

        # Update networks
        self.q_optimizer.zero_grad()
        self.aleatoric_optimizer.zero_grad()

        total_loss = q_loss + aleatoric_loss
        total_loss.backward()

        self.q_optimizer.step()
        self.aleatoric_optimizer.step()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def reset(self):
        self.hidden = self.q_net.init_hidden()

# Training loop
def train(env, agent, episodes=1000):
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        agent.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Average Reward: {np.mean(rewards[-100:]):.2f}")

    return rewards

# Example usage
env = GridWorld()
agent = BENAgent(state_size=2+env.n_hazards+env.n_victims, action_size=5)
rewards = train(env, agent)
