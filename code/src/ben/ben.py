from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass

@dataclass
class BENConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    rnn_hidden_dim: int = 64
    num_flows: int = 2
    learning_rate: float = 1e-4
    gamma: float = 0.99

@dataclass
class BENOutput:
    q_values: torch.Tensor
    aleatoric_sample: torch.Tensor
    aleatoric_log_prob: torch.Tensor
    epistemic_sample: torch.Tensor
    epistemic_log_prob: torch.Tensor
    hidden: Optional[torch.Tensor]

class RecurrentQNetwork(nn.Module):
    def __init__(self, config: BENConfig):
        super().__init__()
        self.config = config

        # Input layer
        self.input_layer = nn.Linear(
            config.state_dim + 1,  # state_dim + reward
            config.hidden_dim
        )

        # GRU for history encoding
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.rnn_hidden_dim,
            batch_first=True
        )

        # Output layer for Q-values
        self.q_layer = nn.Linear(
            config.rnn_hidden_dim,
            config.action_dim
        )

    def forward(self, 
                state: torch.Tensor,
                reward: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine state and reward
        x = torch.cat([state, reward.unsqueeze(-1)], dim=-1)
        x = F.relu(self.input_layer.forward(x))

        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Process through GRU
        out, new_hidden = self.gru.forward(x, hidden)

        # Get Q-values
        q_values = self.q_layer.forward(out)

        return q_values, new_hidden

class NormalizingFlow(nn.Module):
    """Base class for normalizing flows."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform input x and return transformed value z and log determinant.
        """
        raise NotImplementedError

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform input z and return transformed value x and log determinant.
        """
        raise NotImplementedError

class AffineCouplingLayer(NormalizingFlow):
    def __init__(self, dim: int):
        super().__init__(dim)

        # Ensure even dimension for splitting
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")

        self.split_dim = dim // 2
        hidden_dim = max(dim * 2, 4) # Ensure sufficient capacity

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_dim),
            nn.Tanh() # Keep scale factors bounded
        )

        self.translation_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split_dim)
        )

    def _split_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split input tensor into two parts along last dimension.
        """
        return x[..., :self.split_dim], x[..., self.split_dim:]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: x -> z.
        Returns transformed variable and log determinant.
        """
        # Ensure input has the right shape: [batch_size, dim]
        if len(x.shape) == 3:
            x = x.squeeze(1) # Remove sequence dimension if present

        if x.shape[-1] != self.dim:
            x = torch.zeros_like(x).expand(-1, self.dim)

        x1, x2 = self._split_input(x)

        # Compute scale and translation based on x1
        s = self.scale_net.forward(x1)
        t = self.translation_net.forward(x1)

        # Identity transformation for the first half
        z1 = x1

        # Transform x2
        z2 = x2 * torch.exp(s) + t

        # Combine back
        z = torch.cat([z1, z2], dim=-1)

        # Log determinant of Jacobian
        log_det = torch.sum(s, dim=-1)

        return z, log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: z -> x.
        Returns transformed variable and log determinant.
        """
        # Ensure input has the right shape: [batch_size, dim]
        if len(z.shape) == 3:
            z = z.squeeze(1) # Remove sequence dimension if present

        if z.shape[-1] != self.dim:
            z = torch.zeros_like(z).expand(-1, self.dim)

        z1, z2 = self._split_input(z)

        # Compute scale and translation based on z1
        s = self.scale_net.forward(z1)
        t = self.translation_net.forward(z1)

        # Identity transformation for first half
        x1 = z1

        # Inverse transform z2
        x2 = (z2 - t) * torch.exp(-s)

        # Combine back
        x = torch.cat([x1, x2], dim=-1)

        # Log determinant of inverse Jacobian
        log_det = -torch.sum(s, dim=-1)

        return x, log_det

class AleatoricNetwork(nn.Module):
    def __init__(self, config: BENConfig):
        """
        The aleatoric network models uncertainty in the Bellman
        operator, which is essentially uncertainty in Q-values.
        Since Q-values are scalar, we extend it to 2D to make the
        coupling mechanism work for the affine coupling layer.
        """
        super().__init__()
        self.config = config

        # Use 4D space for aleatoric uncertainty
        self.flow_dim = 4
        # Base distribution
        self.base_dist = Normal(0, 1)

        # Project Q-values to flow dimension
        self.projection = nn.Sequential(
            nn.Linear(config.action_dim, self.flow_dim),
            nn.ReLU(),
        )

        # Series of normalizing flows
        # - Base distribution for Q-value uncertainty (1D -> 2D for coupling)
        self.flows = nn.ModuleList([
            AffineCouplingLayer(dim=self.flow_dim)
            for _ in range(config.num_flows)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform from base distribution to target distribution.
        Returns transformed sample and log probability.
        """
        # Remove sequence dimension if present and project
        if len(x.shape) == 3:
            x = x.squeeze(1)
        x = self.projection(x) # Shape: [batch_size, flow_dim]

        # Sample from base distribution with matching batch size
        z = self.base_dist.sample(x.shape[:-1]) # Shape: [batch_size]
        z = z.unsqueeze(-1).expand(-1, self.flow_dim) # Shape: [batch_size, flow_dim]
        log_prob = self.base_dist.log_prob(z).sum(-1) # Shape: [batch_size]

        # Transform through flows
        for flow in self.flows:
            z, ldj = flow.forward(z)
            log_prob = log_prob - ldj

        return z, log_prob

class EpistemicNetwork(nn.Module):
    def __init__(self, config: BENConfig):
        """
        This network models uncertainty in the model parameters.
        The parameters live in a higher-dimensional space, specifically
        the dimension of our hidden layers.
        """
        super().__init__()
        self.config = config

        # Ensure even dimension
        self.flow_dim = config.hidden_dim + (config.hidden_dim % 2)
        # Base distribution for parameter uncertainty (in hidden_dim space)
        self.base_dist = Normal(0, 1)

        # Project Q-values to flow dimension
        self.projection = nn.Sequential(
            nn.Linear(config.action_dim, self.flow_dim),
            nn.ReLU(),
        )

        self.flows = nn.ModuleList([
            # Model uncertainty in parameter space
            AffineCouplingLayer(dim=self.flow_dim)
            for _ in range(config.num_flows)
        ])

        # Project input to flow dimension
        self.projection = nn.Linear(config.action_dim, self.flow_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Remove sequence dimension if present and project
        if len(x.shape) == 3:
            x = x.squeeze(1)
        x = self.projection(x) # Shape: [batch_size, flow_dim]

        # Sample from base distribution with matching batch size
        z = self.base_dist.sample(x.shape[:-1]) # Shape: [batch_size]
        z = z.unsqueeze(-1).expand(-1, self.flow_dim) # Shape: [batch_size, flow_dim]
        log_prob = self.base_dist.log_prob(z).sum(-1) # Shape: [batch_size]

        # Transform through flows
        for flow in self.flows:
            z, ldj = flow.forward(z)
            log_prob = log_prob - ldj

        return z, log_prob

class BEN(nn.Module):
    def __init__(self, config: BENConfig):
        super().__init__()
        self.config = config

        # Main components
        self.q_network = RecurrentQNetwork(config)
        self.aleatoric_network = AleatoricNetwork(config)
        self.epistemic_network = EpistemicNetwork(config)

        # Optimizers
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        self.aleatoric_optimizer = torch.optim.Adam(
            self.aleatoric_network.parameters(),
            lr=config.learning_rate
        )
        self.epistemic_optimizer = torch.optim.Adam(
            self.epistemic_network.parameters(),
            lr=config.learning_rate
        )

    def forward(self, 
                state: torch.Tensor,
                reward: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
               ) -> BENOutput:
        # print(f"\nForward pass tensor shapes:")
        # print(f"Input shapes - state: {state.shape}, reward: {reward.shape}")
        # if hidden is not None:
        #     print(f"Hidden: {hidden.shape}")

        # Get Q-values
        q_values, new_hidden = self.q_network.forward(state, reward, hidden)
        # print(f"Q-network - q_values: {q_values.shape}")
        # if new_hidden is not None:
        #     print(f"Q-network - new_hidden: {new_hidden.shape}")

        # Get aleatoric uncertainty
        # print(f"Aleatoric network input shape: {q_values.shape}")
        aleatoric_sample, aleatoric_log_prob = self.aleatoric_network.forward(q_values)
        # print(f"Aleatoric shapes - sample: {aleatoric_sample.shape}, log_prob: {aleatoric_log_prob.shape}")

        # Get epistemic uncertainty
        # print(f"Epistemic network input shape: {q_values.shape}")
        epistemic_sample, epistemic_log_prob = self.epistemic_network.forward(q_values)
        # print(f"Epistemic shapes - sample: {epistemic_sample.shape}, log_prob: {epistemic_log_prob.shape}")

        return BENOutput(
            q_values=q_values,
            aleatoric_sample=aleatoric_sample,
            aleatoric_log_prob=aleatoric_log_prob,
            epistemic_sample=epistemic_sample,
            epistemic_log_prob=epistemic_log_prob,
            hidden=new_hidden
        )

    def update(self, 
            batch: Dict[str, torch.Tensor],
        ) -> Dict[str, float]:
        """
        Update networks using MSBBE and ELBO objectives.
        Returns dict with loss values.
        """
        # Unpack batch
        states = batch['states'] # Shape: [batch_size, state_dim]
        rewards = batch['rewards'] # Shape: [batch_size]
        next_states = batch['next_states'] # Shape: [batch_size, state_dim]
        actions = batch['actions'] # Shape: [batch_size]

        # Forward pass
        ben_now = self.forward(states, rewards)
        ben_next = self.forward(next_states, rewards)

        # Remove sequence dimension and get target Q-values
        current_q = ben_now.q_values.squeeze(1) # Shape: [batch_size, action_dim]
        next_q = ben_next.q_values.squeeze(1) # Shape: [batch_size, action_dim]

        # Calculate target Q-values
        next_value = next_q.max(dim=1)[0] # Shape: [batch_size]
        target_q = rewards + self.config.gamma * next_value # Shape: [batch_size]

        # Get Q-values for taken actions
        action_q_values = current_q[torch.arange(current_q.size(0)), actions] # Shape: [batch_size]

        # Calculate MSBBE loss
        msbbe_loss = F.mse_loss(action_q_values, target_q)

        # Calculate ELBO loss
        elbo_loss = -ben_now.aleatoric_log_prob.mean() - ben_now.epistemic_log_prob.mean()

        # Update networks
        self.q_optimizer.zero_grad()
        self.aleatoric_optimizer.zero_grad()
        self.epistemic_optimizer.zero_grad()

        msbbe_loss.backward()
        elbo_loss.backward()

        self.q_optimizer.step()
        self.aleatoric_optimizer.step() 
        self.epistemic_optimizer.step()

        return {
            'msbbe_loss': msbbe_loss.item(),
            'elbo_loss': elbo_loss.item()
        }

# Wrapping the GridWorld structure
from dataclasses import dataclass
import torch
from typing import Tuple, Dict, List, Optional
from ..gridworld import GridWorld, Action

@dataclass
class StepData:
    state: torch.Tensor
    reward: torch.Tensor
    done: bool
    info: Dict

class GridWorldEnv:
    def __init__(self, grid_world: GridWorld):
        self.grid_world = grid_world
        self.state_dim = 2  # x, y coordinates
        self.action_dim = len(Action)  # UP, RIGHT, DOWN, LEFT

        # Keep track of episode stats
        self.steps = 0
        self.total_reward = 0.0

    def _state_to_tensor(self, state: Tuple[int, int]) -> torch.Tensor:
        """Convert state tuple to torch tensor"""
        return torch.tensor(state, dtype=torch.float32)

    def _reward_to_tensor(self, reward: float) -> torch.Tensor:
        """Convert reward to torch tensor"""
        return torch.tensor(reward, dtype=torch.float32)

    def reset(self) -> StepData:
        """Reset environment and return initial state"""
        state = self.grid_world.reset()
        self.steps = 0
        self.total_reward = 0.0

        return StepData(
            state=self._state_to_tensor(state),
            reward=torch.tensor(0.0),  # Initial reward is 0
            done=False,
            info={}
        )

    def step(self, action: Action) -> StepData:
        """Take a step in the environment"""
        state, reward, done = self.grid_world.step(action)

        self.steps += 1
        self.total_reward += reward

        # Add additional info
        info = {
            'steps': self.steps,
            'total_reward': self.total_reward
        }

        # Check if episode should end due to max steps
        if self.steps >= self.grid_world.max_steps:
            done = True

        return StepData(
            state=self._state_to_tensor(state),
            reward=self._reward_to_tensor(reward),
            done=done,
            info=info
        )

@dataclass
class ReplayBuffer:
    capacity: int
    states: List[torch.Tensor]
    actions: List[Action]
    rewards: List[torch.Tensor]
    next_states: List[torch.Tensor]
    dones: List[bool]

    @classmethod
    def create(cls, capacity: int) -> 'ReplayBuffer':
        return cls(
            capacity=capacity,
            states=[],
            actions=[],
            rewards=[],
            next_states=[],
            dones=[]
        )

    def add(self, state: torch.Tensor, action: Action, reward: torch.Tensor, 
            next_state: torch.Tensor, done: bool) -> None:
        """Add transition to buffer"""
        if len(self.states) >= self.capacity:
            # Remove oldest transition
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch of transitions"""
        if len(self.states) < batch_size:
            return None

        indices = torch.randperm(len(self.states))[:batch_size]

        return {
            'states': torch.stack([self.states[i] for i in indices]),
            'actions': torch.tensor([self.actions[i].value for i in indices]),
            'rewards': torch.stack([self.rewards[i] for i in indices]),
            'next_states': torch.stack([self.next_states[i] for i in indices]),
            'dones': torch.tensor([self.dones[i] for i in indices])
        }

    def __len__(self) -> int:
        return len(self.states)

@dataclass
class TrainingConfig:
    num_episodes: int = 1000
    batch_size: int = 32
    replay_capacity: int = 10000
    min_experiences: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

def train_ben(env: GridWorldEnv, ben: BEN, config: TrainingConfig) -> List[float]:
    """Train BEN on GridWorld environment"""
    # Initialize replay buffer
    buffer = ReplayBuffer.create(config.replay_capacity)
    rewards_history = []
    epsilon = config.epsilon_start
    
    for episode in range(config.num_episodes):
        # Reset environment
        step_data = env.reset()
        episode_reward = 0
        hidden = None  # Reset hidden state
        
        while not step_data.done:
            state = step_data.state
            
            # Epsilon-greedy action selection
            if torch.rand(1) < epsilon:
                action = Action(torch.randint(env.action_dim, (1,)).item())
            else:
                with torch.no_grad():
                    # Get Q-values and uncertainties
                    output = ben.forward(state.unsqueeze(0), 
                                      step_data.reward.unsqueeze(0), 
                                      hidden)
                    q_values = output.q_values.squeeze()
                    hidden = output.hidden
                    action = Action(torch.argmax(q_values).item())
            
            # Take step in environment
            next_step_data = env.step(action)
            episode_reward += next_step_data.reward.item()
            
            # Store transition in replay buffer
            buffer.add(
                state=step_data.state,
                action=action,
                reward=next_step_data.reward,
                next_state=next_step_data.state,
                done=next_step_data.done
            )
            
            # Train on batch if we have enough experiences
            if len(buffer) >= config.min_experiences:
                batch = buffer.sample(config.batch_size)
                if batch is not None:
                    losses = ben.update(batch)
            
            step_data = next_step_data
        
        # Decay epsilon
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        
        # Record episode reward
        rewards_history.append(episode_reward)
        
        # Print progress
        # if (episode + 1) % 10 == 0:
        if (episode + 1) % 1 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    return rewards_history

def main():
    # Create a simple grid world
    layout = """
    #######
    #s   g#
    # o o #
    #     #
    #######
    """

    # Initialize environment
    grid_world = GridWorld(layout=layout)
    env = GridWorldEnv(grid_world)

    # Initialize BEN
    config = BENConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        rnn_hidden_dim=64,
        num_flows=2,
        learning_rate=1e-4,
        gamma=0.99
    )

    ben = BEN(config)

    # Train
    training_config = TrainingConfig(
        num_episodes=1000,
        batch_size=32,
        replay_capacity=10000,
        min_experiences=100
    )

    rewards = train_ben(env, ben, training_config)

    # Plot learning curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == "__main__":
    main()
