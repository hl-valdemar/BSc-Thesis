from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from environments.nchain import NChainEnv, NChainState

from .model import BayesianExplorationNetwork, MovingAverage


@dataclass
class Trajectory:
    """TODO..."""

    states: Tensor  # [batch_size, seq_len, state_dim]
    actions: Tensor  # [batch_size, seq_len]
    rewards: Tensor  # [batch_size, seq_len]


@dataclass
class BENTrainerConfig:
    """Configuration for BEN training hyperparameters."""

    # Prior initialization params (Algorithm 2)
    n_pretrain_steps: int = 1000
    prior_learning_rate: float = 1e-4

    # Posterior updating params (Algorithm 3)
    n_update_steps: int = 10
    n_posterior_steps: int = 5
    alpha_psi: float = 1e-3  # Epistemic network learning rate
    alpha_omega: float = 1e-4  # Q-network learning rate

    # Training params
    batch_size: int = 32
    n_episodes: int = 1000
    max_steps_per_episode: int = 500
    gamma: float = 0.99
    exploration_factor: float = 1.0

    # Evaluation params
    eval_frequency: int = 100
    n_eval_episodes: int = 10


class BENTrainer:
    """
    Implements the complete training procedure for Bayesian Exploration Networks
    as described in Section 4 and Appendix C of the paper.
    """

    def __init__(
        self,
        env: NChainEnv,
        ben: BayesianExplorationNetwork,
        config: BENTrainerConfig,
        prior_data: Optional[Dict[str, Tensor]] = None,
    ):
        self.env = env
        self.ben = ben
        self.config = config
        self.prior_data = prior_data

        # Initialize training trackers
        self.episode_rewards: List[float] = []
        self.elbo_values: List[float] = []
        self.msbbe_values: List[float] = []

        # Moving averages for stable training
        self.ema_elbo = MovingAverage(beta=0.99)
        self.ema_msbbe = MovingAverage(beta=0.99)

    def train(self) -> None:
        """Main training loop implementing the full BEN training procedure."""

        # 1. Prior Initialization (Algorithm 2)
        print("\nInitializing prior...")
        self._initialize_prior()

        # 2. Main training loop
        for episode in range(self.config.n_episodes):
            # Collect episode experience
            print("\nRunning episode...")
            trajectory = self._run_episode()

            # Update posterior using collected experience
            print("\nUpdating posterior...")
            self._update_posterior(trajectory)

            # # Evaluate periodically
            # if episode % self.config.eval_frequency == 0:
            #     eval_reward = self._evaluate()
            #     self._log_metrics(episode, eval_reward)

    def _initialize_prior(self) -> None:
        """Implements Algorithm 2: Prior Initialization."""
        initial_state = (
            self.env.reset().to_tensor().unsqueeze(0).unsqueeze(0)
        )  # Shape: [batch_size, seq_len, state_dim]

        self.ben.initialize_prior(
            num_steps=self.config.n_pretrain_steps,
            initial_state=initial_state,
            prior_data=self.prior_data,
            learning_rate=self.config.prior_learning_rate,
        )

    def _run_episode(self) -> Trajectory:
        """Collects a single episode of experience using current policy."""
        states: List[NChainState] = []
        actions: List[int] = []
        rewards: List[float] = []

        state = self.env.reset()
        done = False
        total_reward = 0

        for step in range(self.config.max_steps_per_episode):
            if done:
                break

            valid_actions = self.env.get_valid_actions(state)

            # Select action using BEN's uncertainty estimates
            action = self._select_action(
                state=state.to_tensor(),
                actions=torch.tensor(actions) if actions else None,
                rewards=torch.tensor(rewards) if rewards else None,
                valid_actions=valid_actions,
            )

            # Take step in environment
            next_state, reward, done = self.env.step(action)

            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            total_reward += reward

        self.episode_rewards.append(total_reward)

        # Process state tensors:
        # - First convert all states to tensors
        # - Then stack them along a new dimension (creating the sequence dimension)
        # - Add batch dimension for BEN's expected format
        state_tensors = [state.to_tensor() for state in states]
        history_states = torch.stack(state_tensors, dim=0)  # [seq_len, state_dim]
        history_states = history_states.unsqueeze(0)  # [1, seq_len, state_dim]

        # Process actions tensors
        history_actions = torch.tensor(actions).unsqueeze(0)  # [1, seq_len]

        # Process rewards tensors
        history_rewards = torch.tensor(rewards).unsqueeze(0)  # [1, seq_len]

        return Trajectory(
            states=history_states,
            actions=history_actions,
            rewards=history_rewards,
        )

    def _update_posterior(self, trajectory: Trajectory) -> None:
        """Implements Algorithm 3: Posterior Updating."""
        # Compute bootstrap samples
        with torch.no_grad():
            bootstrap_samples = self._compute_bootstrap_samples(trajectory)

        # Update networks using two-timescale optimization
        self.ben.update_posterior(
            history_states=trajectory.states,
            history_actions=trajectory.actions,
            history_rewards=trajectory.rewards,
            bootstrap_samples=bootstrap_samples,
            n_update_steps=self.config.n_update_steps,
            n_posterior_steps=self.config.n_posterior_steps,
            alpha_psi=self.config.alpha_psi,
            alpha_omega=self.config.alpha_omega,
        )

    def _compute_bootstrap_samples(self, trajectory: Trajectory) -> Tensor:
        """Computes Bellman bootstrap samples b_t for posterior updating.

        Args: A trajectory with:
            - States visisted, shape [batch_size, seq_len, state_dim]
            - Actions taken, shape [batch_size, seq_len]
            - Rewards received, shape [batch_size, seq_len]

        Returns: Bootstrap samples, shape [batch_size, seq_len]
        """
        states = trajectory.states  # [batch_size, seq_len, state_dim]
        actions = trajectory.actions  # [batch_size, seq_len]
        rewards = trajectory.rewards  # [batch_size, seq_len]

        batch_size = states.size(0)
        seq_len = states.size(1)

        bootstrap_samples = []
        for t in range(seq_len):
            if t == seq_len - 1:
                # Terminal state case
                bootstrap = rewards[:, t]  # [batch_size]
            else:
                # Get next state and action, preserving batch dim
                next_state = states[:, t + 1]  # [batch_size, state_dim]
                next_action = actions[:, t + 1]  # [batch_size]

                # Forward pass through Q-network
                with torch.no_grad():
                    next_value, _, _ = self.ben.q_net.forward(
                        states[:, : t + 2],  # [batch_size, t+2, state_dim]
                        actions[:, : t + 2],  # [batch_size, t+2]
                        rewards[:, : t + 2],  # [batch_size, t+2]
                        next_action,  # [batch_size]
                    )

                # Compute bootstrap sample while preserving batch dimension
                bootstrap = rewards[:, t] + self.config.gamma * next_value.squeeze(-1)

            bootstrap_samples.append(bootstrap)

        # Stack along sequence dimension
        return torch.stack(bootstrap_samples, dim=1)  # [batch_size, seq_len]

    def _select_action(
        self,
        state: Tensor,
        actions: Optional[Tensor],
        rewards: Optional[Tensor],
        valid_actions: List[int],
    ) -> int:
        """Selects action using BEN's uncertainty estimates for exploration."""
        batch_size = 1  # We're processing one state at a time

        # Handle initial state with no history, ensuring correct dimensionality
        if actions is None:
            # Create empty sequence tensors with correct dimensions
            actions = torch.zeros(batch_size, 0, dtype=torch.long, device=state.device)
            rewards = torch.zeros(batch_size, 0, device=state.device)
        else:
            # Ensure actions and rewards maintain batch dimension
            actions = actions.view(batch_size, -1)
            rewards = rewards.view(batch_size, -1)

        # Properly shape the state tensor
        if state.dim() == 1:
            state = state.view(batch_size, 1, -1)  # [batch, seq=1, dim]
        elif state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension

        # Initialize action values tensor
        action_values = torch.full(
            (self.ben.q_net.num_actions,), float("-inf"), device=state.device
        )

        for action in valid_actions:
            # Ensure action tensor has correct shape
            current_action = torch.tensor(
                [action], dtype=torch.long, device=state.device
            )

            output = self.ben.forward(
                history_states=state,
                history_actions=actions,
                history_rewards=rewards,
                current_action=current_action,
            )

            exploration_bonus = (
                output.aleatoric_params[:, 1] + output.epistemic_params[:, 1].mean()
            )

            value = output.q_value + self.config.exploration_factor * exploration_bonus
            action_values[action] = value.squeeze()

        return action_values.argmax().item()

    def _evaluate(self) -> float:
        """Evaluates current policy without exploration."""
        eval_rewards = []

        for _ in range(self.config.n_eval_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Select action greedily (no exploration bonus)
                state_tensor = state.to_tensor().unsqueeze(0)
                output = self.ben.forward(
                    history_states=state_tensor.unsqueeze(0),
                    history_actions=torch.zeros(1, 0),
                    history_rewards=torch.zeros(1, 0),
                    current_action=torch.zeros(1, dtype=int),
                )
                action = output.q_value.argmax().item()

                # Take step
                state, reward, done = self.env.step(action)
                episode_reward += reward

            eval_rewards.append(episode_reward)

        return sum(eval_rewards) / len(eval_rewards)

    def _log_metrics(self, episode: int, eval_reward: float) -> None:
        """Logs training metrics."""
        print(f"\nEpisode {episode}")
        print(f"  Evaluation Reward: {eval_reward:.2f}")
        print(f"  Average ELBO: {self.ema_elbo.value:.4f}")
        print(f"  Average MSBBE: {self.ema_msbbe.value:.4f}")
        print("----------------------------------------")
