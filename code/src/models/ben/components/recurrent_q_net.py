from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class RecurrentQNetwork(nn.Module):
    """
    TODO: document this class.
    """

    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        hidden_dim: int = 64,
        gru_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim

        # Encoder for the state-action-reward tuples
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + 1 + 1, hidden_dim),  # state + reward + action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Gated Recurrent Unit (GRU) for processing history
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)

        # History projection
        self.history_projection = nn.Linear(gru_hidden_dim, hidden_dim)

        # Action embedding
        embedding_dim = min(
            hidden_dim // 2, num_actions * 4
        )  # Prevent action embedding from dominating the history encoding
        self.action_embedding = nn.Embedding(num_actions, embedding_dim)

        # Q-value head takes both history encoding and action
        self.q_head = nn.Linear(
            hidden_dim + embedding_dim, 1
        )  # Input: history (processed by the GRU) + action -> corresponding q-value

    def forward(
        self,
        history_states: Tensor,
        history_actions: Tensor,
        history_rewards: Tensor,
        current_action: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Maps history-action pairs to Q-values using RNN encoding.

        As described in Section 4.1 of the BEN paper, this function approximator
        takes a history h_t (sequence of state-action-reward tuples) and an action a_t,
        and outputs the corresponding Q-value.

        Args:
            history_states: States [batch_size, seq_len, state_dim]
            history_actions: Actions [batch_size, seq_len]
            history_rewards: Rewards [batch_size, seq_len]
            action: Current actions [batch_size]
            hidden: Optional GRU hidden state [1, batch_size, gru_hidden_dim]

        Returns:
            Tuple of:
            - Q-value for given history-action pair [batch_size, 1]
            - History encoding for given history-action pair [batch_size, seq_len]
            - Updated GRU hidden state [1, batch_size, gru_hidden_dim]
        """
        assert history_states.dim() == 3, "history_states must be [batch, seq, dim]"
        assert history_actions.dim() == 2, "history_actions must be [batch, seq]"
        assert history_rewards.dim() == 2, "history_rewards must be [batch, seq]"
        assert current_action.dim() == 1, "history_rewards must be [batch]"
        assert history_states.size(0) == history_actions.size(
            0
        ), "Batch sizes must match"

        # Encode history
        batch_size = history_states.size(0)
        seq_len = history_states.size(1)

        # Ensure consistent sequence dimensions
        if history_actions.size(1) != seq_len:
            history_actions = history_actions[:, :seq_len]
        if history_rewards.size(1) != seq_len:
            history_rewards = history_rewards[:, :seq_len]

        # Combine and encode history elements
        if history_actions.size(1) == 0:  # No history case
            # Create a single-step "dummy" history with zeros
            history = torch.zeros(
                batch_size,
                1,
                self.state_dim + 1 + 1,  # state_dim + action + reward
                device=history_states.device,
            )
            # Copy current state into the dummy history
            history[:, 0, : self.state_dim] = history_states[:, 0, :]
        else:
            # Normal case - concatenate actual history
            history = torch.cat(
                [
                    history_states,  # Shape: [batch_size, seq_len, state_dim]
                    history_actions.unsqueeze(-1),  # Shape: [batch_size, seq_len, 1]
                    history_rewards.unsqueeze(-1),  # Shape: [batch_size, seq_len, 1]
                ],
                dim=-1,
            )

        # Encode each timestep
        encoded_history: Tensor = self.encoder.forward(
            history
        )  # Shape: [batch_size, seq_len, hidden_dim]

        # Get history encoding through GRU
        if hidden is None:
            hidden = torch.zeros(
                1, batch_size, self.gru_hidden_dim, device=history_states.device
            )

        history_encoding, hidden = self.gru.forward(encoded_history, hidden)
        history_encoding = self.history_projection.forward(history_encoding)
        history_encoding = history_encoding[:, -1]  # Take last hidden state

        # Embed current action
        action_embedding = self.action_embedding.forward(current_action)
        if action_embedding.dim() == 1:  # Make sure the batch dimension exists
            action_embedding = action_embedding.unsqueeze(0)

        # Combine history encoding with action to get Q-value
        q_input = torch.cat([history_encoding, action_embedding], dim=-1)
        q_value = self.q_head.forward(q_input)

        return q_value, history_encoding, hidden
