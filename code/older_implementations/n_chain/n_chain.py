from enum import Enum
from typing import Tuple


class Action(Enum):
    FORWARD = 0
    STAY = 1


class NChain:
    """
    N-Chain environment:
    - Linear chain of n states
    - Two actions: move forward or stay
    - Reward only at the end of chain
    - Episode terminates when reaching end or max steps
    """

    def __init__(self, n: int = 5, reward: float = 1.0):
        """
        Args:
            n: Length of the chain
            reward: Reward at the end of chain
        """
        self.n = n
        self.reward = reward
        self.max_steps = n * 2  # Allow some redundancy
        self.reset()

    def reset(self) -> int:
        """Reset environment to initial state."""
        self.current_state = 0
        self.steps = 0
        return self.current_state

    def step(self, action: Action) -> Tuple[int, float, bool]:
        """
        Take action in environment.

        Args:
            action: FORWARD or STAY

        Returns:
            (next_state, reward, done)
        """
        self.steps += 1

        # Move according to action
        if action == Action.FORWARD:
            self.current_state = min(self.current_state + 1, self.n - 1)

        # Check if done
        done = self.current_state == self.n - 1 or self.steps >= self.max_steps

        # Get reward
        reward = self.reward if self.current_state == self.n - 1 else 0.0

        return self.current_state, reward, done

    def get_state_space_size(self) -> int:
        """Return number of possible states."""
        return self.n

    def get_action_space_size(self) -> int:
        """Return number of possible actions."""
        return len(Action)

    def is_terminal(self, state: int) -> bool:
        """Check if state is terminal."""
        return state == self.n - 1
