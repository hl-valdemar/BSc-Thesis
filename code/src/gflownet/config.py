import torch

from dataclasses import dataclass

@dataclass
class GFlowNetConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    learning_rate: float = 1e-4
    epsilon: float = 0.05  # For exploratory actions

@dataclass
class GFlowOutput:
    flows: torch.Tensor  # F(s,a) for each action
    state_flow: torch.Tensor  # F(s) total flow through state
    logits: torch.Tensor  # log F(s,a) for numerical stability
