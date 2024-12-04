from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class NChainState:
    """
    Represents a state in the Branching N-Chain environment.

    Attributes:
        position: Integer position in the chain [0, n-1]
        n: Length of each branch
        branch: Which branch we're on (-1=pre-split, 0=left, 1=right)
    """

    position: int
    n: int  # Length of each branch (total length = n + split_point)
    branch: int  # -1 = pre-split, 0 = left branch, 1 = right branch

    def __post_init__(self):
        assert (
            0 <= self.position < self.n * 2
        ), f"Position {self.position} must be in [0, {self.n * 2 - 1}]"
        assert self.branch in [
            -1,
            0,
            1,
        ], "Branch must be -1 (pre-split), 0 (left), or 1 (right)"

    def to_tensor(self) -> Tensor:
        """Convert state to tensor representation.

        Returns:
            Tensor: Shape [n*2 + 3], concatenation of:
                - one-hot position (n*2)
                - one-hot branch encoding (3) for pre-split, left, right
        """
        state_tensor = torch.zeros(self.n * 2 + 3)
        state_tensor[self.position] = 1.0
        state_tensor[self.n * 2 + (self.branch + 1)] = 1.0
        return state_tensor

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "NChainState":
        """Convert tensor back to BranchingState."""
        n = (len(tensor) - 3) // 2
        position = torch.argmax(tensor[: n * 2]).item()
        branch = torch.argmax(tensor[n * 2 :]).item() - 1
        return cls(position=position, n=n, branch=branch)


@dataclass
class NChainTrajectory:
    """
    Container for a single trajectory through the environment.

    Attributes:
        states: List of states visited
        actions: List of actions taken
        rewards: List of rewards received
        done: Whether trajectory ended at terminal state
        valid_actions_masks: List of masks over valid actions
    """

    states: List[NChainState]
    actions: List[int]
    rewards: List[float]
    done: bool
    valid_actions_masks: List[Tensor]

    def to_tensors(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert trajectory to tensor format.

        Returns:
            Tuple containing:
                - states_tensor: Shape [T, state_dim]
                - actions_tensor: Shape [T]
                - rewards_tensor: Shape [T]
        """
        states_tensor = torch.stack([s.to_tensor() for s in self.states])
        actions_tensor = torch.tensor(self.actions)
        rewards_tensor = torch.tensor(self.rewards)
        return states_tensor, actions_tensor, rewards_tensor


class NChainEnv:
    """
    Branching N-Chain environment with two distinct terminal states.

    The agent starts at position 0 and moves right until reaching the split point.
    At the split, it must choose either the left or right branch, each leading to
    different terminal states with distinct rewards.
    """

    def __init__(
        self,
        n: int = 5,  # Length of each branch
        left_reward: float = 5.0,
        right_reward: float = 10.0,
    ):
        self.n = n
        self.left_reward = left_reward
        self.right_reward = right_reward
        self.split_point = n // 2  # Split occurs at n // 2 (the middle)
        self.current_state: Optional[NChainState] = None

        # Actions: 0 = stay, 1 = right (pre-split)
        # At split: 0 = stay, 1 = choose left, 2 = choose right
        # Post-split: 0 = stay, 1 = forward on chosen branch
        self.num_actions = 3  # Maximum number of actions available

    def reset(self) -> NChainState:
        """Reset environment to initial state (position 0)."""
        self.current_state = NChainState(position=0, n=self.n, branch=-1)
        return self.current_state

    def step(self, action: int) -> Tuple[NChainState, float, bool]:
        """Take a step in the environment.

        Args:
            action: Integer action
                Pre-split: 0=stay, 1=right
                At split: 0=stay, 1=left branch, 2=right branch
                Post-split: 0=stay, 1=forward

        Returns:
            Tuple of (next_state, reward, done)
        """
        assert self.current_state is not None
        assert action in self.get_valid_actions(self.current_state)

        position = self.current_state.position
        branch = self.current_state.branch

        # Handle pre-split movement
        if branch == -1:
            if action == 0:  # Stay
                new_position = position
                new_branch = -1
            elif action == 1:  # Move right
                new_position = position + 1
                # Check if we've reached the split
                if new_position == self.split_point:
                    new_branch = -1
                else:
                    new_branch = -1

        # Handle at-split decisions
        elif position == self.split_point:
            if action == 0:  # Stay at split
                new_position = position
                new_branch = branch
            elif action == 1:  # Choose left branch
                new_position = position + 1
                new_branch = 0
            else:  # Choose right branch
                new_position = position + 1
                new_branch = 1

        # Handle post-split movement
        else:
            if action == 0:  # Stay
                new_position = position
                new_branch = branch
            else:  # Move forward on chosen branch
                new_position = position + 1
                new_branch = branch

        # Update state
        self.current_state = NChainState(
            position=new_position, n=self.n, branch=new_branch
        )

        # Check if we've reached a terminal state
        done = self.is_terminal(self.current_state)

        # Determine reward
        if done:
            reward = self.left_reward if new_branch == 0 else self.right_reward
        else:
            reward = 0.0

        return self.current_state, reward, done

    def get_valid_actions(self, state: Optional[NChainState] = None) -> List[int]:
        """Get list of valid actions for given state."""
        if state is None:
            assert self.current_state is not None
            state = self.current_state

        # Terminal states
        if self.is_terminal(state):
            return [0]  # Can only stay

        # At split point
        if state.position == self.split_point and state.branch == -1:
            return [0, 1, 2]  # Stay, left branch, right branch

        # Pre-split or post-split
        return [0, 1]  # Stay or move forward

    def is_terminal(self, state: NChainState) -> bool:
        """Check if state is terminal (end of either branch)."""
        return state.branch in [0, 1] and state.position == self.split_point + (
            self.n - self.split_point
        )
