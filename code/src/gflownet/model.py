from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from gridworld import Action, GridWorld

class GFlowNetwork:
    def __init__(
        self,
        env: GridWorld,
        hidden_size: int = 64,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Initialize GFlowNet.
        
        Args:
            env: GridWorld environment
            hidden_size: Number of hidden units in neural network
            learning_rate: Learning rate for optimizer
            device: Device to run computations on
        """
        self.env = env
        self.device = device
        self.hidden_size = hidden_size
        
        # Initialize flow predictor network
        self.flow_predictor = nn.Sequential(
            nn.Linear(2, hidden_size),  # 2 = x, y coordinates of current position
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(Action))  # Predict log flow for each action
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.flow_predictor.parameters(), lr=learning_rate)
        
    def state_to_tensor(self, state: Tuple[int, int]) -> torch.Tensor:
        """Convert state to tensor representation.
        
        Args:
            state: Current (x, y) position
            
        Returns:
            Tensor representation of state
        """
        return torch.tensor([*state], dtype=torch.float32, device=self.device)
    
    def predict_flows(self, state: Tuple[int, int]) -> Dict[Action, float]:
        """Predict flows for all actions from given state.
        
        Args:
            state: Current (x, y) position
            
        Returns:
            Dictionary mapping actions to their predicted flows
        """
        state_tensor = self.state_to_tensor(state)
        log_flows = self.flow_predictor(state_tensor)
        flows = torch.exp(log_flows)  # Convert from log space
        
        # Create dictionary of valid actions and their flows
        action_flows: Dict[Action, float] = {}
        for action in Action:
            next_state = self.env.get_next_state(state, action)
            if self.env.is_valid_position(next_state):
                action_flows[action] = flows[action.value].item()
                
        return action_flows
    
    def sample_action(self, state: Tuple[int, int], epsilon: float = 0.1) -> Action:
        """Sample action according to flow probabilities with epsilon exploration.
        
        Args:
            state: Current (x, y) position
            epsilon: Probability of random action
            
        Returns:
            Sampled action
        """
        if np.random.random() < epsilon:
            # Random valid action for exploration
            valid_actions = [a for a in Action 
                            if self.env.is_valid_position(self.env.get_next_state(state, a))]
            return valid_actions[np.random.randint(len(valid_actions))]
        
        action_flows = self.predict_flows(state)
        total_flow = sum(action_flows.values())
        
        if total_flow == 0:
            # If all flows are 0, use random valid action
            valid_actions = [a for a in Action 
                            if self.env.is_valid_position(self.env.get_next_state(state, a))]
            return valid_actions[np.random.randint(len(valid_actions))]
            
        # Sample according to flow probabilities
        action_probs = {a: f/total_flow for a, f in action_flows.items()}
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        # Convert to numpy arrays for proper sampling
        probs = np.array(probs)
        # Handle numerical instability
        probs = probs / probs.sum()  
        return actions[np.random.choice(len(actions), p=probs)]
    
    def compute_loss(self, trajectory: List[Tuple[Tuple[int, int], Action]], 
                    final_reward: float) -> torch.Tensor:
        """Compute flow matching loss for a trajectory.
        
        Args:
            trajectory: List of (state, action) pairs
            final_reward: Reward received at terminal state
            
        Returns:
            Loss value
        """
        loss = 0.0
        
        # For each state in trajectory
        for i, (state, _) in enumerate(trajectory):
            # Get predicted flows for all actions
            state_tensor = self.state_to_tensor(state, self.env.goal_pos)
            log_flows = self.flow_predictor(state_tensor)
            flows = torch.exp(log_flows)
            
            # Calculate outgoing flow
            valid_flows = torch.zeros_like(flows)
            for action in Action:
                next_state = self.env.get_next_state(state, action)
                if self.env.is_valid_position(next_state):
                    valid_flows[action.value] = flows[action.value]
            outgoing_flow = valid_flows.sum()
            
            # Target flow is final reward for last state, outgoing flow for others
            target_flow = final_reward if i == len(trajectory)-1 else outgoing_flow.detach()
            
            # Flow matching loss (in log space)
            loss += (torch.log(outgoing_flow + 1e-8) - torch.log(target_flow + 1e-8))**2
            
        return loss
    
    def train_step(self, trajectory: List[Tuple[Tuple[int, int], Action]], 
                  final_reward: float) -> float:
        """Perform a single training step.
        
        Args:
            trajectory: List of (state, action) pairs
            final_reward: Reward received at terminal state
            
        Returns:
            Loss value
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(trajectory, final_reward)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def generate_episode(self, epsilon: float = 0.1) -> Tuple[List[Tuple[Tuple[int, int], Action]], float]:
        """Generate a complete episode using current policy.
        
        Args:
            epsilon: Exploration probability
            
        Returns:
            Trajectory and final reward
        """
        state = self.env.reset()
        trajectory: List[Tuple[Tuple[int, int], Action]] = []
        
        while True:
            action = self.sample_action(state, epsilon)
            trajectory.append((state, action))
            
            next_state, reward, done = self.env.step(action)
            state = next_state
            
            if done:
                return trajectory, reward
