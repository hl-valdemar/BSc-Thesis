from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BENConfig

class RecurrentQNetwork(nn.Module):
    def __init__(self, config: BENConfig):
        super().__init__()
        self.config = config

        # Input layer
        self.input_layer = nn.Linear(
            config.state_dim + 1,  # state_dim + reward
            config.hidden_dim
        )

        # GRU for history encoding
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.rnn_hidden_dim,
            batch_first=True
        )

        # Output layer for Q-values
        self.q_layer = nn.Linear(
            config.rnn_hidden_dim,
            config.action_dim
        )

    def forward(self, 
                state: torch.Tensor,
                reward: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine state and reward
        x = torch.cat([state, reward.unsqueeze(-1)], dim=-1)
        x = F.relu(self.input_layer.forward(x))

        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Process through GRU
        out, new_hidden = self.gru.forward(x, hidden)

        # Get Q-values
        q_values = self.q_layer.forward(out)

        return q_values, new_hidden
