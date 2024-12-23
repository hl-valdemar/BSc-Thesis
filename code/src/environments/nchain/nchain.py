import random
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, dtype

MaskNDArray = npt.NDArray[np.int8]


@dataclass
class NChainState:
    """
    Represents a state in the Branching N-Chain environment.

    Attributes:
        position: Integer position in the chain [0, n-1]
        n: Length of each branch
        branch: Which branch we're on (-1=pre-split, 0=branch_0, 1=branch_1, ...)
    """

    position: int
    n: int  # Length of each branch (total length = n + split_point)
    branch: int  # -1 = pre-split, 0 = branch 0, 1 = branch 1, ...

    def __post_init__(self):
        assert (
            0 <= self.position < self.n * 3
        ), f"Position {self.position} must be in [0, {self.n * 3 - 1}]"
        assert self.branch in [
            -1,
            0,
            1,
            2,
        ], "Branch must be -1 (pre-split), 0 (first branch), 1 (second branch), ..."

        # Add the state dim
        self.state_dim = self.n * 3 + 4

    def to_tensor(self, dtype: Optional[dtype] = None) -> Tensor:
        """Convert state to tensor representation.

        Returns:
            Tensor: Shape [n*3 + 4], concatenation of:
                - one-hot position (n*3)
                - one-hot branch encoding (4) for pre-split, first branch, second branch, ...
        """
        state_tensor = torch.zeros(self.state_dim, dtype=dtype)
        state_tensor[self.position] = 1.0
        state_tensor[self.state_dim - 4 + (self.branch + 1)] = 1.0
        return state_tensor

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "NChainState":
        """Convert tensor back to BranchingState."""
        n = (len(tensor) - 4) // 2
        position = torch.argmax(tensor[: n * 3]).item()
        branch = torch.argmax(tensor[n * 3 :]).item() - 1
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
        forward_masks: List of masks over forward actions
    """

    states: List[NChainState]
    actions: List[int]
    rewards: List[float]
    done: bool
    forward_masks: List[Tensor]
    backward_masks: List[Tensor]

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


class NChainAction(IntEnum):
    TERMINAL_STAY = 0
    FORWARD = 1
    BRANCH_0 = 2
    BRANCH_1 = 3
    BRANCH_2 = 4


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
        rewards: List[float] = [50.0, 250.0, 1250.0],
    ):
        assert len(rewards) == 3, "Reward count should match branch count (3)"

        self.n = n
        self.rewards = rewards
        self.split_point = n // 2  # Split occurs at n // 2 (the middle)
        self.current_state = NChainState(position=0, n=self.n, branch=-1)
        self.state_dim = self.current_state.state_dim

        # Actions: forward (pre-split)
        # At split: choose one of 3 branches [0..2]
        # Post-split: forward on chosen branch
        # At terminal: stay
        self.num_actions = 5

    def reset(self) -> NChainState:
        """Reset environment to initial state (position 0)."""
        self.current_state = NChainState(position=0, n=self.n, branch=-1)
        return self.current_state

    def step(self, action: int) -> Tuple[NChainState, float, bool]:
        """Take a step in the environment.

        Args:
            action: NChainAction to take in environment

        Returns:
            Tuple containing:
            - Resulting state
            - Reward
            - Done - whether terminal was reached
        """
        assert action in self.get_valid_actions(self.current_state)

        position = self.current_state.position
        branch = self.current_state.branch

        # Initialize new state variables
        new_position = position
        new_branch = branch

        # Handle pre-split movement
        if branch == -1 and position < self.split_point:
            if action == NChainAction.FORWARD:  # Move right
                new_position = position + 1

        # Handle at-split decisions
        elif branch == -1 and position == self.split_point:
            if action == NChainAction.BRANCH_0:  # Choose first branch
                new_position = position + 1
                new_branch = 0
            elif action == NChainAction.BRANCH_1:  # Choose second branch
                new_position = position + 1
                new_branch = 1
            elif action == NChainAction.BRANCH_2:  # ...
                new_position = position + 1
                new_branch = 2

        # Handle post-split movement
        elif branch in [0, 1, 2]:  # On either branch
            if self.is_terminal(self.current_state):
                if action == NChainAction.TERMINAL_STAY:
                    new_position = position
            elif not self.is_terminal(self.current_state):
                if action == NChainAction.FORWARD:  # Move forward
                    new_position = position + 1

        # Update state
        self.current_state = NChainState(
            position=new_position, n=self.n, branch=new_branch
        )

        # Terminal state is reached when we're on a branch and have moved
        # n-split_point steps after the split point
        done = branch in [0, 1, 2] and position >= self.split_point + (
            self.n - self.split_point
        )

        reward = 0.0
        if done:
            reward = self.rewards[branch]

        return self.current_state, reward, done

    def get_valid_actions(self, state: Optional[NChainState] = None) -> List[int]:
        """
        Get list of valid actions for given state.

        Args:
            state: Optional state - if not specified, current state is used

        Returns:
            List[int]: List of valid actions for given state
        """
        if state is None:
            state = self.current_state

        # Terminal states
        if self.is_terminal(state):
            return [NChainAction.TERMINAL_STAY]

        # At split point
        if state.position == self.split_point and state.branch == -1:
            return [
                NChainAction.BRANCH_0,
                NChainAction.BRANCH_1,
                NChainAction.BRANCH_2,
            ]

        # Pre-split or post-split
        return [NChainAction.FORWARD]  # Move forward

    def is_terminal(self, state: NChainState) -> bool:
        """Check if state is terminal (end of either branch)."""
        return state.branch in [0, 1, 2] and state.position >= self.split_point + (
            self.n - self.split_point
        )

    def max_reward(self) -> float:
        """
        Returns the maximum possible reward in the environment.

        Returns:
            float: Maximum reward possible
        """
        return reduce(lambda x, y: max(x, y), self.rewards)

    def sample(self, mask: MaskNDArray | None = None) -> np.int64:
        """Generates a single random sample from this space.

        A sample will be chosen uniformly at random with the mask if provided

        Args:
            mask: An optional mask for if an action can be selected.
                Expected `np.ndarray` of shape ``(n,)`` and dtype ``np.int8`` where ``1`` represents valid actions and ``0`` invalid / infeasible actions.
                If there are no possible actions (i.e. ``np.all(mask == 0)``) then ``space.start`` will be returned.

        Returns:
            A sampled integer from the space
        """
        if mask is not None:
            assert isinstance(
                mask, np.ndarray
            ), f"The expected type of the mask is np.ndarray, actual type: {type(mask)}"
            assert (
                mask.dtype == np.int8
            ), f"The expected dtype of the mask is np.int8, actual dtype: {mask.dtype}"
            assert (
                mask.shape == (self.n,)
            ), f"The expected shape of the mask is {(self.n,)}, actual shape: {mask.shape}"
            valid_action_mask = mask == 1
            assert np.all(
                np.logical_or(mask == 0, valid_action_mask)
            ), f"All values of a mask should be 0 or 1, actual values: {mask}"
            if np.any(valid_action_mask):
                return np.random.choice(np.where(valid_action_mask)[0])
            else:
                return 0

        return random.randint(0, self.num_actions)

    def sample_valid_actions(
        self,
        k: int,
        state: Optional[NChainState] = None,
    ) -> List[NChainAction]:
        if state is None:
            state = self.current_state

        valid_actions = self.get_valid_actions(state)
        return random.sample(valid_actions, k=k)

    def translate_action(self, action: NChainAction) -> str:
        """Translates an action (int) to it's descriptor/name."""
        name = "UNKNOWN_ACTION"
        if action == NChainAction.TERMINAL_STAY:
            name = "TERMINAL_STAY"
        elif action == NChainAction.FORWARD:
            name = "FORWARD"
        elif action == NChainAction.BRANCH_0:
            name = "BRANCH_0"
        elif action == NChainAction.BRANCH_1:
            name = "BRANCH_1"
        elif action == NChainAction.BRANCH_2:
            name = "BRANCH_2"
        return name
