from typing import Union

import numpy as np
import torch
from torch import nn

from environments.nchain import NChainEnv

from .QBayesNet import QBayesNetwork


class EmpiricalTarget(nn.Module):
    def __init__(
        self,
        q_network: Union[QBayesNetwork, torch.distributions.Distribution],
        envs,
    ):
        super().__init__()
        self.q_network = q_network
        self.hidden_state = None
        self.rewards = None
        self.next_state = None
        self.action = None
        self.observation = None
        if isinstance(q_network, QBayesNetwork):
            self.q_vals = None
            self.add_module("q_network", q_network)
        elif isinstance(q_network, torch.distributions.Distribution):
            self.q_vals = self.q_network.sample(
                torch.Size(
                    [
                        1,
                    ]
                )
            )
        self.observation = envs.reset()
        self.gamma = 0.99

    def estimate_q_opt_action(
        self,
        hidden_state,
        current_state,
        action,
        rewards,
        q_network,
    ):
        batch_size = hidden_state.shape[0]
        self.observation = torch.cat([current_state, action, rewards], dim=-1).reshape(
            batch_size, -1
        )
        self.hidden_state = hidden_state.reshape(batch_size, -1)
        self.rewards = rewards
        self.next_state = current_state
        self.action = action
        self.q_network = q_network
        if isinstance(self.q_network, QBayesNetwork):
            q_vals, self.hidden_state = q_network(self.observation, self.hidden_state)
            self.q_vals, self.new_action = q_vals.max(dim=-1)
            self.hidden_state = self.hidden_state.reshape(
                self.hidden_state.shape[0], -1
            )
        elif isinstance(self.q_network, torch.distributions.Distribution):
            self.q_vals = self.q_network.sample(torch.Size([10, 1]))
        return self.q_vals, self.new_action

    def forward(
        self,
        hidden_state: torch.Tensor,
        current_state: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        q_network: Union[QBayesNetwork, torch.distributions.Distribution],
        # envs: gym.vector.sync_vector_env.SyncVectorEnv,
        envs: NChainEnv,
    ):
        """
        :param hidden_state: hidden state of the agent
        :param current_state: current state of the agent
        :param action: action taken by the agent
        :param rewards: reward received by the agent
        :param q_network: q_network to be used for the target

        :return: estimated bellman target, estimated q_vals off path

        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        batch_size = hidden_state.shape[0]
        self.observation = torch.cat(
            [
                current_state.reshape(1, -1),
                action.reshape(1, -1),
                rewards.reshape(1, -1),
            ],
            dim=-1,
        )
        hidden_state = hidden_state.reshape(batch_size, -1)
        if isinstance(q_network, QBayesNetwork):
            # Fixed set of params for q_networ, train epistemic, train model based, train both, kl divergence
            states = q_network.state_dict()
            q_vals, next_hidden_state = q_network(self.observation, hidden_state)
            states = q_network.state_dict()
            action_for_env = int(action.item())
            next_state, rewards, _, _ = envs.step(
                [
                    action_for_env,
                ]
            )
            # random noise
            next_state = next_state + np.random.standard_normal(next_state.shape) * 0.2
            inputs = torch.cat(
                [
                    torch.tensor(next_state).view(1, -1),
                    torch.tensor(action).view(1, -1),
                    torch.tensor(rewards).view(1, -1),
                ],
                dim=-1,
            )
            next_q, next_next_hidden = q_network(inputs, next_hidden_state)

            envs.set_attr("state", current_state)
            q_network.load_state_dict(states)

            with torch.no_grad():
                b_est = torch.tensor(rewards) + self.gamma * next_q.max()
        return b_est, q_vals

    def sample_b_vals(
        self,
        current_state,
        hidden_state,
        q_network,
        action,
        reward,
        # envs: gym.vector.sync_vector_env.SyncVectorEnv,
        envs: NChainEnv,
        num_samples=100,
    ):
        b_vals = []
        inputs = torch.cat(
            [current_state.view(1, -1), action.view(1, -1), reward.view(1, -1)], dim=-1
        )
        q_vals, next_hidden_state = q_network(inputs, hidden_state)
        states = q_network.state_dict()
        action_for_env = int(action.item())
        for i in range(num_samples):
            next_state, reward, _, _ = envs.step(
                [
                    action_for_env,
                ]
            )
            next_state = next_state + np.random.standard_normal(next_state.shape) * 0.2
            inputs = torch.cat(
                [
                    torch.tensor(next_state).view(1, -1),
                    torch.tensor(action).view(1, -1),
                    torch.tensor(reward).view(1, -1),
                ],
                dim=-1,
            )
            next_q, next_next_hidden = q_network(inputs, next_hidden_state)
            with torch.no_grad():
                b = torch.tensor(reward) + self.gamma * next_q.max()
            envs.set_attr("state", current_state)
            q_network.load_state_dict(states)
            b_vals.append(b)
        return torch.cat(b_vals, dim=0).mean()
