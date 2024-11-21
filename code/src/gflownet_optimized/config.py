from dataclasses import dataclass

import torch


@dataclass
class GFlowNetConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    num_episodes: int = 1000
    batch_size: int = 16
    min_batch_size: int = 4
    learning_rate: float = 5e-5
    lr_warmup_steps: int = 1000
    min_lr: float = 1e-6
    weight_decay: float = 1e-5
    flow_entropy_coef: float = 0.01  # For flow regularization
    grad_clip: float = 1.0
    action_entropy_coef: float = 0.01
    
@dataclass 
class GFlowOutput:
    flows: torch.Tensor  # F(s,a) for each action
    state_flow: torch.Tensor  # F(s) total flow through state
    logits: torch.Tensor  # log F(s,a) for numerical stability
    log_Z: torch.Tensor  # log partition function

@dataclass
class GFlowNetTrainingConfig:
    num_episodes: int = 1000
    batch_size: int = 16
    min_batch_size: int = 4
    replay_capacity: int = 10000
    min_experiences: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    temperature_start: float = 5.0
    temperature_end: float = 0.1
    temperature_decay: float = 0.995
    trajectory_length: int = 50  # Max length of trajectories to store
    grad_clip: float = 1.0
    # learning_rate: float = 1e-4
    # weight_decay: float = 1e-5
    chunk_size: int = 32
    num_workers: int = 4
