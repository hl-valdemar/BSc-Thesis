import random
from collections import defaultdict, deque
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from n_chain import NChainEnv

from .metrics import MetricsTracker, TrainingMetrics
from .model import BEN, BENOutput


def normal_log_likelihood(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Compute log likelihood under Normal distribution."""
    return -0.5 * (logvar + (x - mu).pow(2) / torch.exp(logvar)).mean()


class BENTrainer:
    """Trainer for Bayesian Exploration Network."""

    def __init__(
        self,
        env: NChainEnv,
        model: BEN,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        metrics_window: int = 100,
        exploration_bonus_scale: float = 0.1,
    ):
        self.env = env
        self.model = model
        self.target_model = BEN(env.n, model.hidden_dim)
        self.target_model.load_state_dict(model.state_dict())

        # Optimizers
        self.q_optimizer = Adam(model.q_net.parameters(), lr=learning_rate)
        self.aleatoric_optimizer = Adam(
            model.aleatoric_net.parameters(), lr=learning_rate
        )
        self.epistemic_optimizer = Adam(
            model.epistemic_net.parameters(), lr=learning_rate
        )

        # Training parameters
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.exploration_bonus_scale = exploration_bonus_scale

        # Initialize replay buffer and training tracking
        self.replay_buffer = deque(maxlen=buffer_size)
        self.steps_done = 0
        self.eps_threshold = 0.9
        self.metrics = MetricsTracker(window_size=metrics_window)

    def get_uncertainties(self, outputs: BENOutput) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute aleatoric and epistemic uncertainties from network outputs.

        Returns:
            Tuple containing:
                - aleatoric_uncertainty
                - epistemic_uncertainty
                - exploration_bonus
        """
        # Aleatoric uncertainty from logvar
        aleatoric_uncertainty = torch.exp(outputs.aleatoric_params[:, 1])

        # Epistemic uncertainty from latent space
        epistemic_mean = outputs.epistemic_params[
            :, : self.model.epistemic_net.latent_dim
        ]
        epistemic_uncertainty = torch.norm(epistemic_mean, dim=1)

        # Combined exploration bonus
        exploration_bonus = (
            aleatoric_uncertainty + epistemic_uncertainty
        ) * self.exploration_bonus_scale

        return aleatoric_uncertainty, epistemic_uncertainty, exploration_bonus

    def _prepare_batch(self, batch: List[Tuple]) -> Tuple[Tensor, ...]:
        """Convert batch of experiences to tensors."""
        states = torch.stack([s.to_tensor() for s, *_ in batch])
        actions = torch.tensor([a for _, a, *_ in batch], dtype=torch.long)
        rewards = torch.tensor([r for _, _, r, *_ in batch], dtype=torch.float)
        next_states = torch.stack([s.to_tensor() for _, _, _, s, _ in batch])
        dones = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float)

        return states, actions, rewards, next_states, dones

    def collect_trajectory(self) -> List[Tuple]:
        """Collect a single trajectory using current policy."""
        state = self.env.reset()
        hidden = None
        done = False
        trajectory = []

        while not done:
            # Get action using ε-greedy with uncertainty bonus
            state_tensor = state.to_tensor().unsqueeze(0)
            outputs, hidden = self.model(state_tensor, hidden)

            # Add exploration bonus to Q-values
            _, _, bonus = self.get_uncertainties(outputs)
            q_values = outputs.q_value + bonus.unsqueeze(-1)

            # ε-greedy action selection
            if torch.rand(1).item() < self.eps_threshold:
                action = torch.randint(2, (1,)).item()
            else:
                action = q_values.argmax(-1).item()

            # Take step in environment
            next_state, reward, done = self.env.step(action)

            # Store transition
            trajectory.append((state, action, reward, next_state, done))

            state = next_state

        return trajectory

    def compute_q_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute Q-learning loss with uncertainty-based exploration bonus.

        Returns:
            Tuple containing:
                - q_loss: Q-network loss
                - exploration_bonus: Mean exploration bonus used
        """
        # Get current Q-values and uncertainties
        outputs, hidden = self.model(states, hidden)
        q_values = outputs.q_value
        current_q = q_values.gather(1, actions.unsqueeze(1))

        _, _, exploration_bonus = self.get_uncertainties(outputs)

        # Get next state values from target network
        with torch.no_grad():
            target_outputs, _ = self.target_model(next_states)
            next_q = target_outputs.q_value.max(1)[0].unsqueeze(1)

            # Add exploration bonus to targets
            targets = rewards.unsqueeze(1) + self.gamma * next_q * (
                1 - dones.unsqueeze(1)
            )
            targets = targets + exploration_bonus.unsqueeze(1)

        q_loss = F.mse_loss(current_q, targets)

        return q_loss, exploration_bonus.mean()

    def compute_aleatoric_loss(
        self, states: Tensor, rewards: Tensor, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute loss for aleatoric network using negative log likelihood.

        Returns:
            Tuple containing:
                - aleatoric_loss: Loss for aleatoric network
                - aleatoric_uncertainty: Mean aleatoric uncertainty
        """
        outputs, hidden = self.model(states, hidden)

        # Get distribution parameters
        mu = outputs.aleatoric_params[:, 0]
        logvar = outputs.aleatoric_params[:, 1]

        # Compute loss and uncertainty
        aleatoric_loss = -normal_log_likelihood(rewards, mu, logvar)
        aleatoric_uncertainty = torch.exp(logvar).mean()

        return aleatoric_loss, aleatoric_uncertainty

    def compute_epistemic_loss(
        self, states: Tensor, hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute KL divergence loss for epistemic network.

        Returns:
            Tuple containing:
                - epistemic_loss: KL divergence loss
                - epistemic_uncertainty: Mean epistemic uncertainty
        """
        outputs, hidden = self.model.forward(states, hidden)

        # Split parameters into mean and logvar
        mean = outputs.epistemic_params[:, : self.model.epistemic_net.latent_dim]
        logvar = outputs.epistemic_params[:, self.model.epistemic_net.latent_dim :]

        # KL divergence with standard normal prior
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)

        # Add reconstruction term to prevent collapse
        reconstruction = self.model.epistemic_net.decoder(mean)
        rec_loss = F.mse_loss(reconstruction, states)

        # Combined loss with weighting
        beta = 0.1  # Hyperparameter
        epistemic_loss = rec_loss + beta * kl_div.mean()

        # Compute uncertainty
        epistemic_uncertainty = torch.norm(mean, dim=1).mean()

        return epistemic_loss, epistemic_uncertainty

    def optimize_model(self) -> Tuple[float, float, float, float, float, float]:
        """
        Perform one step of optimization on all networks.

        Returns:
            Tuple containing:
                - q_loss
                - aleatoric_loss
                - epistemic_loss
                - aleatoric_uncertainty
                - epistemic_uncertainty
                - exploration_bonus
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = self._prepare_batch(batch)

        # Optimize Q-network
        self.q_optimizer.zero_grad()
        q_loss, exploration_bonus = self.compute_q_loss(
            states, actions, rewards, next_states, dones
        )
        q_loss.backward()
        self.q_optimizer.step()

        # Optimize aleatoric network
        self.aleatoric_optimizer.zero_grad()
        aleatoric_loss, aleatoric_uncertainty = self.compute_aleatoric_loss(
            states, rewards
        )
        aleatoric_loss.backward()
        self.aleatoric_optimizer.step()

        # Optimize epistemic network
        self.epistemic_optimizer.zero_grad()
        epistemic_loss, epistemic_uncertainty = self.compute_epistemic_loss(states)
        epistemic_loss.backward()
        self.epistemic_optimizer.step()

        return (
            q_loss.item(),
            aleatoric_loss.item(),
            epistemic_loss.item(),
            aleatoric_uncertainty.item(),
            epistemic_uncertainty.item(),
            exploration_bonus.item(),
        )

    def train_step(self) -> Tuple[float, float, float]:
        """Perform single training step."""
        # Collect trajectory and store in buffer
        trajectory = self.collect_trajectory()
        self.replay_buffer.extend(trajectory)

        # Optimize model
        q_loss, al_loss, ep_loss, al_unc, ep_unc, bonus = self.optimize_model()

        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Update exploration rate
        self.steps_done += 1
        self.eps_threshold = max(0.01, 0.9 - 0.89 * self.steps_done / 10000)

        # Compute metrics
        states = torch.stack([s.to_tensor() for s, *_ in trajectory])
        actions = torch.tensor([a for _, a, *_ in trajectory])
        rewards = torch.tensor([r for _, _, r, *_ in trajectory])

        # Count state visits
        state_visits = defaultdict(int)
        for state, *_ in trajectory:
            state_visits[state.position] += 1

        metrics = TrainingMetrics(
            q_loss=q_loss,
            aleatoric_loss=al_loss,
            epistemic_loss=ep_loss,
            episode_length=len(trajectory),
            episode_return=rewards.sum().item(),
            state_visits=dict(state_visits),
            aleatoric_uncertainty=al_unc,
            epistemic_uncertainty=ep_unc,
            exploration_bonus=bonus,
            successful_episodes=trajectory[-1][-1],
        )

        self.metrics.add_metrics(metrics)

        return q_loss, al_loss, ep_loss

    def train(self, num_steps: int) -> List[Tuple[float, float, float]]:
        """Train for specified number of steps."""
        losses = []
        for step in range(num_steps):
            step_losses = self.train_step()
            losses.append(step_losses)

            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{num_steps}")
                self.metrics.print_summary()

        return losses
