from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class NChainState:
    """
    Represents a state in the N-Chain environment.

    Attributes:
        position: Integer position in the chain [0, n-1]
        n: Length of the chain
    """

    position: int
    n: int

    def __post_init__(self):
        assert (
            0 <= self.position < self.n
        ), f"Position {self.position} must be in [0, {self.n-1}]"

    def to_tensor(self) -> Tensor:
        """Convert state to one-hot tensor representation.

        Returns:
            Tensor: Shape [n], one-hot encoding of position
        """
        state_tensor = torch.zeros(self.n)
        state_tensor[self.position] = 1.0
        return state_tensor

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "NChainState":
        """Convert one-hot tensor back to NChainState.

        Args:
            tensor: Shape [n], one-hot encoding of position

        Returns:
            NChainState: Corresponding state object
        """
        position = torch.argmax(tensor).item()
        return cls(position=position, n=len(tensor))


class NChainEnv:
    """
    N-Chain environment with sparse rewards.

    The agent starts at position 0 and can move right or stay.
    The only reward is at the rightmost state (n-1).

    Attributes:
        n: Length of the chain
        sparse_reward: Reward given at the final state
    """

    def __init__(self, n: int = 5, sparse_reward: float = 10.0):
        """
        Args:
            n: Length of the chain
            sparse_reward: Reward given at the final state
        """
        self.n = n
        self.sparse_reward = sparse_reward
        self.current_state: Optional[NChainState] = None

        # Define action space: 0 = stay, 1 = right
        self.num_actions = 2

    def reset(self) -> NChainState:
        """Reset environment to initial state (position 0).

        Returns:
            NChainState: Initial state
        """
        self.current_state = NChainState(position=0, n=self.n)
        return self.current_state

    def step(self, action: int) -> Tuple[NChainState, float, bool]:
        """Take a step in the environment.

        Args:
            action: Integer in {0, 1} representing stay/right

        Returns:
            Tuple containing:
                - NChainState: New state
                - float: Reward (0 except at final state)
                - bool: Whether episode is done
        """
        assert self.current_state is not None, "Must call reset() before step()"
        assert action in {0, 1}, f"Action must be 0 or 1, got {action}"

        # Update position based on action
        new_position = min(self.current_state.position + action, self.n - 1)

        self.current_state = NChainState(position=new_position, n=self.n)

        # Check if we reached the end
        done = new_position == self.n - 1
        reward = self.sparse_reward if done else 0.0

        return self.current_state, reward, done

    def get_valid_actions(self, state: Optional[NChainState] = None) -> List[int]:
        """Get list of valid actions for given state.

        Args:
            state: State to get actions for, uses current state if None

        Returns:
            List[int]: Valid actions (0=stay always valid, 1=right if not at end)
        """
        if state is None:
            assert self.current_state is not None
            state = self.current_state

        if state.position == self.n - 1:
            return [0]  # Can only stay at the end
        return [0, 1]  # Can stay or move right

    def is_terminal(self, state: NChainState) -> bool:
        """Check if state is terminal (rightmost position).

        Args:
            state: State to check

        Returns:
            bool: Whether state is terminal
        """
        return state.position == self.n - 1
