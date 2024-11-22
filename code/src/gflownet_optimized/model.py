from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from gridworld import Action

from .config import GFlowNetConfig, GFlowOutput
from .env import GridWorldEnv, Trajectory


class FlowNetwork(nn.Module):
    """Main network that predicts flows F(s,a)"""

    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        self.device = torch.device(config.device)

        # Wider but shallower network with batch optimization
        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.action_dim),
        ).to(self.device)

        # Initialize with small positive values
        self.log_Z = nn.Parameter(torch.tensor(0.0, device=self.device))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict log flows for all actions from a state
        Args:
            state: torch.Tensor of shape [batch_size, state_dim]
        Returns:
            logits: torch.Tensor of shape [batch_size, action_dim]
        """
        state = state.to(self.device)
        logits = self.network(state)
        return logits + self.log_Z  # Add in log space


class GFlowNet(nn.Module):
    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        print(f"Initializing GFlowNet on device: {self.device}")

        # Initialize components
        self.flow_network = FlowNetwork(config).to(self.device)
        self.total_steps = 0
        self.training_steps = 0
        self.visit_counts = defaultdict(int)
        self.action_values = defaultdict(int)
        self.grad_clip_val = config.grad_clip
        self.accumulation_steps = 4

        # Move model to GPU
        self.to(self.device)

        # Setup optimizer and mixed precision
        self.setup_training_tools(config)

    def setup_training_tools(self, config: GFlowNetConfig):
        """Setup optimizer, scheduler and mixed precision scaler"""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.scheduler = AdaptiveLRScheduler(self.optimizer)

        # Initialize mixed precision scaler if using GPU
        self.scaler = (
            torch.amp.GradScaler()
            if config.use_amp and config.device == "cuda"
            else None
        )

    def move_batch_to_device(self, batch_data):
        """Helper to move batch data to correct device"""
        if isinstance(batch_data, torch.Tensor):
            return batch_data.to(self.device, non_blocking=True)
        elif isinstance(batch_data, (list, tuple)):
            return [self.move_batch_to_device(item) for item in batch_data]
        elif isinstance(batch_data, dict):
            return {k: self.move_batch_to_device(v) for k, v in batch_data.items()}
        return batch_data

    # def to_device(self, x):
    #     """Helper to move tensors or lists of tensors to correct device"""
    #     if isinstance(x, (list, tuple)):
    #         return [self.to_device(item) for item in x]
    #     if isinstance(x, torch.Tensor):
    #         return x.to(self.device)
    #     return x

    # def setup_optimizer(
    #     self,
    #     optimizer_type: str = "adam",
    #     lr: float = 1e-4,
    #     weight_decay: float = 1e-5,
    #     **kwargs,
    # ):
    #     """Configure the optimizer with flexibility"""
    #     if optimizer_type.lower() == "adam":
    #         self.optimizer = torch.optim.Adam(
    #             self.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
    #         )
    #     elif optimizer_type.lower() == "sgd":
    #         self.optimizer = torch.optim.SGD(
    #             self.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
    #         )
    #     else:
    #         raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    #     # Create a learning rate scheduler
    #     # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     #     self.optimizer,
    #     #     mode='min',
    #     #     factor=0.5,
    #     #     patience=5,
    #     #     min_lr=1e-6,
    #     # )

    #     # Optimized scheduler
    #     self.scheduler = AdaptiveLRScheduler(self.optimizer)

    def forward(self, state: torch.Tensor) -> GFlowOutput:
        """
        Forward pass to get flows and policy
        Args:
            state: torch.Tensor of shape [batch_size, state_dim]
        """
        # Get log flows (logits)
        logits = self.flow_network(state)

        # More numerically stable flow computation (compared to unoptimized version)
        flows = torch.exp(logits - logits.max(dim=-1, keepdim=True)[0])

        # Calculate total flow through state
        state_flow = flows.sum(dim=-1, keepdim=True)

        return GFlowOutput(
            flows=flows,
            state_flow=state_flow,
            logits=logits,
            log_Z=self.flow_network.log_Z,
        )

    def _prepare_batch(self, trajectories, timestep):
        """Helper to prepare batch data for given timestep"""
        batch_states = []
        batch_actions = []
        batch_is_valid = []

        # Collect batch data
        for traj in trajectories:
            if timestep < len(traj.states) - 1:
                batch_states.append(traj.states[timestep])
                batch_actions.append(traj.action_indices[timestep])
                batch_is_valid.append(True)
            else:
                batch_states.append(torch.zeros_like(trajectories[0].states[0]))
                batch_actions.append(
                    torch.zeros_like(trajectories[0].action_indices[0])
                )
                batch_is_valid.append(False)

        # Skip if no valid states
        if not any(batch_is_valid):
            return None

        # Move to GPU and return
        return (
            torch.stack(batch_states).to(self.device),
            torch.stack(batch_actions).view(-1, 1).to(self.device),
            torch.tensor(batch_is_valid, device=self.device),
        )

    def compute_trajectory_balance_loss(
        self, trajectories: List[Trajectory], epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Compute trajectory balance loss for a batch of trajectories"""
        losses = []
        max_length = max(len(traj.states) - 1 for traj in trajectories)

        # Process trajectories in batches at each timestep
        for t in range(max_length):
            # Prepare batch data
            batch_data = self._prepare_batch(trajectories, t)
            if batch_data is None:
                continue

            # Convert to tensors
            states_tensor, actions_tensor, valid_mask = batch_data

            # Compute outputs with mixed precision where appropriate
            with torch.amp.autocast(device_type="cuda", enabled=self.config.use_amp):
                # Forward pass with batch
                output = self.forward(states_tensor)
                logits = output.logits

                # Compute probabilities
                action_logprob = logits.gather(1, actions_tensor)
                log_Z = torch.logsumexp(logits, dim=-1, keepdim=True)
                step_logprob = action_logprob - log_Z

                # Mask out padded trajectories
                step_logprob = step_logprob.squeeze() * valid_mask

                # Compute losses for valid trajectories
                for i, traj in enumerate(trajectories):
                    if valid_mask[i] and traj.done:
                        target_logprob = torch.log(
                            traj.rewards[-1].to(self.device) + epsilon
                        )
                        loss = (step_logprob[i] - target_logprob) ** 2
                        losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(losses).mean()

    # def compute_flow_matching_loss(
    #     self, trajectories: List[Trajectory], epsilon: float = 1e-6
    # ) -> torch.Tensor:
    #     """Compute flow matching loss for a batch of trajectories"""
    #     losses = []

    #     # Move trajectories to correct device
    #     trajectories = [traj.to(self.device) for traj in trajectories]

    #     # Process trajectories in chunks
    #     chunk_size = self.config.min_batch_size

    #     for start_idx in range(0, len(trajectories), chunk_size):
    #         chunk = trajectories[start_idx : start_idx + chunk_size]
    #         max_length = max(len(traj.states) - 1 for traj in chunk)

    #         # Process each timestep for this chunk
    #         for t in range(max_length):
    #             # Collect states for this timestep
    #             curr_states = []
    #             next_states = []
    #             curr_actions = []
    #             is_terminal = []
    #             rewards = []
    #             is_valid = []

    #             for traj in chunk:
    #                 if t < len(traj.states) - 1:
    #                     curr_states.append(traj.states[t])
    #                     next_states.append(traj.states[t + 1])
    #                     curr_actions.append(traj.action_indices[t])
    #                     is_terminal.append(t == len(traj.states) - 2 and traj.done)
    #                     rewards.append(traj.rewards[t])
    #                     is_valid.append(True)
    #                 else:
    #                     # Padding
    #                     curr_states.append(
    #                         torch.zeros_like(chunk[0].states[0], device=self.device)
    #                     )
    #                     next_states.append(
    #                         torch.zeros_like(chunk[0].states[0], device=self.device)
    #                     )
    #                     curr_actions.append(
    #                         torch.zeros_like(
    #                             chunk[0].action_indices[0], device=self.device
    #                         )
    #                     )
    #                     is_terminal.append(False)
    #                     rewards.append(
    #                         torch.zeros_like(chunk[0].rewards[0], device=self.device)
    #                     )
    #                     is_valid.append(False)

    #             # Convert to tensors
    #             curr_states = torch.stack(curr_states)
    #             next_states = torch.stack(next_states)
    #             curr_actions = torch.stack(curr_actions).view(-1, 1)
    #             is_terminal = torch.tensor(is_terminal, device=self.device)
    #             rewards = torch.stack(rewards)
    #             valid_mask = torch.tensor(is_valid, device=self.device)

    #             if valid_mask.sum() == 0:
    #                 continue

    #             # Get flows
    #             curr_output = self.forward(curr_states)
    #             next_output = self.forward(next_states)

    #             # Get current flow
    #             curr_flow = curr_output.flows.gather(1, curr_actions)

    #             # Get target flow
    #             target_flow = torch.where(
    #                 is_terminal.unsqueeze(1),
    #                 rewards.unsqueeze(1),
    #                 next_output.state_flow,
    #             )

    #             # Compute loss
    #             loss = (
    #                 torch.log(curr_flow + epsilon) - torch.log(target_flow + epsilon)
    #             ) ** 2
    #             loss = loss * valid_mask.unsqueeze(1)

    #             losses.append(loss.mean())

    #     torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
    #     loss = torch.stack(losses).mean() if losses else torch.tensor(0.0)

    #     return loss

    def compute_flow_matching_loss(
        self, trajectories: List[Trajectory], epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Compute flow matching loss with proper trajectory length handling"""
        losses = []
        device = self.device

        # Process trajectories in chunks
        chunk_size = self.config.min_batch_size

        for start_idx in range(0, len(trajectories), chunk_size):
            chunk = trajectories[start_idx : start_idx + chunk_size]
            max_length = max(len(traj.states) - 1 for traj in chunk)
            chunk_size = len(chunk)

            # Pre-allocate tensors for this chunk
            curr_states = torch.zeros(
                (chunk_size, max_length, self.config.state_dim), device=device
            )
            next_states = torch.zeros(
                (chunk_size, max_length, self.config.state_dim), device=device
            )
            curr_actions = torch.zeros(
                (chunk_size, max_length), dtype=torch.long, device=device
            )
            is_terminal = torch.zeros(
                (chunk_size, max_length), dtype=torch.bool, device=device
            )
            rewards = torch.zeros((chunk_size, max_length), device=device)
            valid_mask = torch.zeros(
                (chunk_size, max_length), dtype=torch.bool, device=device
            )

            # Fill tensors with proper length handling
            for b, traj in enumerate(chunk):
                # Get actual length for this trajectory (excluding last state)
                length = len(traj.states) - 1

                # Safely fill tensors up to the actual length
                curr_states[b, :length] = torch.stack(
                    [s.to(device) for s in traj.states[:-1]]
                )
                next_states[b, :length] = torch.stack(
                    [s.to(device) for s in traj.states[1:]]
                )
                curr_actions[b, :length] = torch.stack(
                    [a.to(device) for a in traj.action_indices[:length]]
                )
                rewards[b, :length] = torch.stack(
                    [r.to(device) for r in traj.rewards[:length]]
                )
                valid_mask[b, :length] = True
                if traj.done:
                    is_terminal[b, length - 1] = True

            # Process each timestep for this chunk
            for t in range(max_length):
                t_mask = valid_mask[:, t]
                if not t_mask.any():
                    continue

                with torch.amp.autocast(
                    device_type="cuda", enabled=self.config.use_amp
                ):
                    # Get current states and actions for this timestep
                    t_curr_states = curr_states[:, t][t_mask]
                    t_curr_actions = curr_actions[:, t][t_mask].unsqueeze(1)

                    # Get flows for current states
                    curr_output = self.forward(t_curr_states)
                    curr_flow = curr_output.flows.gather(1, t_curr_actions)

                    # Get flows for next states
                    t_next_states = next_states[:, t][t_mask]
                    next_output = self.forward(t_next_states)

                    # Compute target flow
                    t_is_terminal = is_terminal[:, t][t_mask].unsqueeze(1)
                    t_rewards = rewards[:, t][t_mask].unsqueeze(1)

                    target_flow = torch.where(
                        t_is_terminal, t_rewards, next_output.state_flow
                    )

                    # Compute loss for this timestep
                    loss = (
                        torch.log(curr_flow + epsilon)
                        - torch.log(target_flow + epsilon)
                    ) ** 2
                    losses.append(loss.mean())

        if not losses:
            return torch.tensor(0.0, device=device)

        return torch.stack(losses).mean()

    def compute_regularization_loss(self, output: GFlowOutput) -> torch.Tensor:
        # Encourage smooth flow distributions
        flow_entropy = -(output.flows * torch.log(output.flows + 1e-6)).sum(-1).mean()
        return -self.config.flow_entropy_coef * flow_entropy

    def compute_action_entropy_loss(
        self, output: GFlowOutput, epsilon: float = 1e-6
    ) -> torch.Tensor:
        """Compute entropy regularization loss to encourage exploration"""
        # Convert flows to probabilities
        probs = output.flows / (output.state_flow + epsilon)

        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + epsilon)).sum(-1).mean()

        # Return negative because we want to maximize entropy
        return -self.config.action_entropy_coef * entropy

    def compute_td_errors(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """Compute TD errors for prioritized replay."""
        errors = []

        for trajectory in trajectories:
            if not trajectory.states:
                continue

            # Get predicted flows for all states in trajectory
            states = torch.stack(trajectory.states)
            with torch.no_grad():
                outputs = self.forward(states)
                flows = outputs.flows

            # Compute TD errors along trajectory
            for t in range(len(trajectory.states) - 1):
                curr_flow = flows[t].gather(
                    0, trajectory.action_indices[t].unsqueeze(0)
                )
                next_flow = outputs.state_flow[t + 1]

                if t == len(trajectory.states) - 2 and trajectory.done:
                    target = trajectory.rewards[-1]
                else:
                    target = next_flow

                error = abs((curr_flow - target).item())
                errors.append(error)

        return torch.tensor(errors).mean() if errors else torch.tensor(0.0)

    def update(
        self,
        trajectories: List[Trajectory],
        indices: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        # print(f"Update called with: {len(trajectories)} trajectories")
        # print(f"Indices type: {type(indices)}, Weights type: {type(weights)}")

        # Convert numpy arrays to tensors on GPU if provided
        if weights is None:
            # Default weights if none provided
            weights = torch.ones(len(trajectories), device=self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Track accumulated losses
        accumulated_losses = {
            "flow_loss": 0.0,
            "balance_loss": 0.0,
            "reg_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
        }

        for i in range(self.accumulation_steps):
            start_idx = i * self.config.batch_size
            end_idx = min((i + 1) * self.config.batch_size, len(trajectories))

            if start_idx >= len(trajectories):
                break

            batch_trajectories = trajectories[start_idx:end_idx]
            batch_weights = weights[start_idx:end_idx] if weights is not None else None

            # Compute losses with automatic mixed precision if enabled
            with torch.amp.autocast(
                device_type="cuda",
                enabled=hasattr(self, "scaler") and self.scaler is not None,
            ):
                flow_loss = self.compute_flow_matching_loss(batch_trajectories)
                if batch_weights is not None:
                    flow_loss = flow_loss * batch_weights.mean()

                balance_loss = self.compute_trajectory_balance_loss(batch_trajectories)
                if batch_weights is not None:
                    balance_loss = balance_loss * batch_weights.mean()

                # Get sample states for regularization
                sample_states = torch.stack(
                    [traj.states[0] for traj in batch_trajectories]
                )
                sample_output = self.forward(sample_states)

                # debug_trajectories(batch_trajectories, "Before loss computation: ")

                reg_loss = self.compute_regularization_loss(sample_output)
                entropy_loss = self.compute_action_entropy_loss(sample_output)

                batch_total_loss = (
                    1.0 * flow_loss
                    + 0.1 * torch.exp(-10 * balance_loss) * balance_loss
                    + 0.02 * reg_loss
                    + entropy_loss
                ) / self.accumulation_steps

            # Backward pass with gradient scaling if enabled
            if hasattr(self, "scaler") and self.scaler is not None:
                self.scaler.scale(batch_total_loss).backward()
            else:
                batch_total_loss.backward()

            # Accumulate losses
            accumulated_losses["flow_loss"] += (
                flow_loss.item() / self.accumulation_steps
            )
            accumulated_losses["balance_loss"] += (
                balance_loss.item() / self.accumulation_steps
            )
            accumulated_losses["reg_loss"] += reg_loss.item() / self.accumulation_steps
            accumulated_losses["entropy_loss"] += (
                entropy_loss.item() / self.accumulation_steps
            )
            accumulated_losses["total_loss"] += batch_total_loss.item()

        # Clip gradients
        if hasattr(self, "scaler") and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)

        # Optimizer step with gradient scaling if enabled
        if hasattr(self, "scaler") and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Update learning rate
        if hasattr(self, "scheduler"):
            self.scheduler.step(accumulated_losses["total_loss"])

        # Increment training steps
        self.training_steps += 1

        # Compute TD errors for priority updates
        td_errors = self.compute_td_errors(trajectories)
        accumulated_losses["td_errors"] = td_errors.item()

        # Add learning rate to metrics
        accumulated_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return accumulated_losses

    # def update(
    #     self,
    #     trajectories: List[Trajectory],
    #     indices: np.ndarray | None = None,
    #     weights: np.ndarray | None = None,
    # ) -> Dict[str, float]:
    #     """Wrapper for gradient accumulation update"""
    #     # Move indices and weights to GPU if provided
    #     if indices is not None:
    #         indices = torch.from_numpy(indices).to(self.device)
    #     if weights is not None:
    #         weights = torch.from_numpy(weights).to(self.device)

    #     return self.update_with_grad_accumulation(
    #         trajectories=trajectories,
    #         indices=indices,
    #         weights=weights,
    #         accumulation_steps=self.accumulation_steps,
    #     )

    # def update_with_grad_accumulation(
    #     self,
    #     trajectories: List[Trajectory],
    #     indices: np.ndarray | None = None,
    #     weights: np.ndarray | None = None,
    #     accumulation_steps: int = 4,
    # ) -> Dict[str, float]:
    #     """Update with gradient accumulation and importance sampling weights."""
    #     # Track accumulated losses
    #     accumulated_losses = {
    #         "flow_loss": 0.0,
    #         "balance_loss": 0.0,
    #         "reg_loss": 0.0,
    #         "entropy_loss": 0.0,
    #         "total_loss": 0.0,
    #     }

    #     # Only zero gradients once at the start
    #     self.optimizer.zero_grad()

    #     # Split trajectories into smaller batches
    #     batch_size = len(trajectories) // accumulation_steps

    #     for i in range(accumulation_steps):
    #         # Sample subset of trajectories for this accumulation step
    #         start_idx = i * batch_size
    #         end_idx = (i + 1) * batch_size

    #         batch_trajectories = trajectories[start_idx:end_idx]
    #         batch_weights = (
    #             weights[start_idx:end_idx]
    #             if weights is not None
    #             else torch.ones(len(batch_trajectories)).to(self.device)
    #         )
    #         batch_indices = indices[start_idx:end_idx] if indices is not None else None

    #         # Add gradient noise for better stability
    #         noise_scale = max(0.01, 1.0 - self.training_steps / 10000)

    #         # Compute losses with importance sampling weights for this batch
    #         flow_loss = (
    #             self.compute_flow_matching_loss(batch_trajectories)
    #             * batch_weights.mean()
    #         )
    #         balance_loss = (
    #             self.compute_trajectory_balance_loss(batch_trajectories)
    #             * batch_weights.mean()
    #         )

    #         sample_states = torch.stack([traj.states[0] for traj in batch_trajectories])
    #         sample_output = self.forward(sample_states)
    #         reg_loss = self.compute_regularization_loss(sample_output)
    #         entropy_loss = self.compute_action_entropy_loss(sample_output)

    #         # Add gradient noise if in training mode
    #         if self.training and noise_scale > 0:
    #             flow_loss += torch.randn_like(flow_loss) * noise_scale
    #             balance_loss += torch.randn_like(balance_loss) * noise_scale
    #             reg_loss += torch.randn_like(reg_loss) * noise_scale
    #             entropy_loss += torch.randn_like(entropy_loss) * noise_scale

    #         # Scale losses by accumulation steps
    #         batch_total_loss = (
    #             1.0 * flow_loss
    #             + 0.1 * torch.exp(-10 * balance_loss) * balance_loss
    #             + 0.02 * reg_loss
    #             + entropy_loss
    #         ) / accumulation_steps

    #         # Backward pass
    #         batch_total_loss.backward()

    #         # Accumulate losses
    #         accumulated_losses["flow_loss"] += flow_loss.item() / accumulation_steps
    #         accumulated_losses["balance_loss"] += (
    #             balance_loss.item() / accumulation_steps
    #         )
    #         accumulated_losses["reg_loss"] += reg_loss.item() / accumulation_steps
    #         accumulated_losses["entropy_loss"] += (
    #             entropy_loss.item() / accumulation_steps
    #         )
    #         accumulated_losses["total_loss"] += batch_total_loss.item()

    #     # Clip gradients
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)

    #     # Step optimizer
    #     self.optimizer.step()

    #     # Update learning rate
    #     self.scheduler.step(accumulated_losses["total_loss"])

    #     # Increment training steps
    #     self.training_steps += 1

    #     # Compute TD errors for priority updates
    #     td_errors = self.compute_td_errors(trajectories)
    #     accumulated_losses["td_errors"] = td_errors.item()

    #     # Add learning rate to metrics
    #     accumulated_losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]

    #     return accumulated_losses

    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float,
        temperature: float,
    ) -> Action:
        """Enhanced exploration strategy"""
        # Convert tensor state to hashable format for counting
        state_key = tuple(state.cpu().numpy().flatten())

        if torch.rand(1) < epsilon:
            # Mix of random and count-based exploration
            counts = self.visit_counts.get(state_key, 0)
            bonus = 1.0 / (1.0 + counts) ** 0.5

            if torch.rand(1) < bonus:
                return Action(torch.randint(self.config.action_dim, (1,)).item())

        with torch.no_grad():
            output = self.forward(state.unsqueeze(0))
            flows = output.flows / temperature

            # Add UCB-style exploration bonus
            if hasattr(self, "action_values"):
                bonus = torch.sqrt(
                    2
                    * torch.log(torch.tensor(self.total_steps + 1))
                    / (torch.tensor(self.action_values.get(state_key, 0) + 1))
                )
                flows += bonus

            probs = flows / flows.sum(dim=-1, keepdim=True)
            action_idx = torch.multinomial(probs[0], 1).item()

            # Update statistics
            self.total_steps += 1
            self.visit_counts[state_key] = self.visit_counts.get(state_key, 0) + 1
            self.action_values[state_key] = self.action_values.get(state_key, 0) + 1

            return Action(action_idx)

    def clear_statistics(self):
        """Clear visit counts and action values"""
        self.visit_counts.clear()
        self.action_values.clear()
        self.total_steps = 0


class AdaptiveLRScheduler:
    def __init__(self, optimizer, initial_lr=1e-4):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.best_loss = float("inf")
        self.patience = 500  # Much longer patience
        self.cooldown = 0
        self.factor = 0.5
        self.min_lr = 1e-6  # Add minimum LR
        self.window_size = 50  # Average loss over window
        self.loss_history = []

    def step(self, loss):
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

        # Use average loss over window
        avg_loss = sum(self.loss_history) / len(self.loss_history)

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.cooldown = 0
        else:
            self.cooldown += 1

        if self.cooldown >= self.patience:
            self.cooldown = 0
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group["lr"] * self.factor, self.min_lr)
                param_group["lr"] = new_lr


def debug_trajectories(trajectories: List[Trajectory], prefix: str = ""):
    """Helper to debug trajectory structure"""
    print(f"\n{prefix}Trajectory Debug:")
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i}:")
        print(f"  States: {len(traj.states)}")
        print(f"  Actions: {len(traj.action_indices)}")
        print(f"  Rewards: {len(traj.rewards)}")
        print(f"  Done: {traj.done}")
        if len(traj.states) != len(traj.action_indices) + 1:
            print("  WARNING: Inconsistent lengths!")
            print(f"  States device: {traj.states[0].device}")
            print(f"  Actions device: {traj.action_indices[0].device}")
            print(f"  Rewards device: {traj.rewards[0].device}")
