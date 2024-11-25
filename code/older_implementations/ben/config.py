from typing import Optional
from dataclasses import dataclass

import torch

@dataclass
class BENConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    rnn_hidden_dim: int = 64
    num_flows: int = 2
    learning_rate: float = 1e-4
    gamma: float = 0.99

@dataclass
class BENOutput:
    q_values: torch.Tensor
    aleatoric_sample: torch.Tensor
    aleatoric_log_prob: torch.Tensor
    epistemic_sample: torch.Tensor
    epistemic_log_prob: torch.Tensor
    hidden: Optional[torch.Tensor]
