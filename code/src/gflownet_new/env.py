import torch

from .model import GFlowNetBase


class GridWorldGFlowNet(GFlowNetBase):
    """GFlowNet implementation for GridWorld environment."""

    def __init__(self, grid_size: int, hidden_dim: int):
        """
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            hidden_dim: Hidden layer dimension
        """
        # State dimension is grid_size^2 (one-hot encoding of position)
        state_dim = grid_size * grid_size
        # 4 actions: UP, RIGHT, DOWN, LEFT
        n_actions = 4

        super().__init__(state_dim, hidden_dim, n_actions)

        self.grid_size = grid_size

    def get_next_state(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """
        Get next state for GridWorld.

        Args:
            state: Current state (one-hot encoded position)
            action: Action (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)

        Returns:
            next_state: Next state tensor
        """
        # Convert one-hot to position
        pos = torch.where(state == 1)[0].item()
        x, y = int(pos % self.grid_size), int(pos // self.grid_size)

        # Get new position based on action
        if action == 0:  # UP
            y = max(0, y - 1)
        elif action == 1:  # RIGHT
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # DOWN
            y = min(self.grid_size - 1, y + 1)
        else:  # LEFT
            x = max(0, x - 1)

        # Convert back to one-hot
        next_state = torch.zeros_like(state)
        next_state[y * self.grid_size + x] = 1

        return next_state

    def is_terminal(self, state: torch.Tensor) -> bool:
        """
        Check if state is terminal (e.g., goal position).

        Args:
            state: Current state

        Returns:
            is_terminal: Whether state is terminal
        """
        # Example: goal is bottom-right corner
        goal_pos = self.grid_size * self.grid_size - 1
        return state[goal_pos].item() == 1

    def get_next_state_embedding(
        self, state_h: torch.Tensor, action: int
    ) -> torch.Tensor:
        """
        Get embedding for next state in GridWorld.

        Args:
            state_h: Current state embedding [batch_size, hidden_dim]
            action: Action to take

        Returns:
            next_state_h: Next state embedding
        """
        batch_size = state_h.shape[0]

        # Create action one-hot encoding
        action_onehot = torch.zeros(batch_size, self.n_actions, device=state_h.device)
        action_onehot[:, action] = 1

        # Combine state embedding with action
        combined = torch.cat([state_h, action_onehot], dim=-1)

        # Process through state-action network
        next_state_h = self.state_action_net(combined)

        return next_state_h
