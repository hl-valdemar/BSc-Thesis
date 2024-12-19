import warnings
from typing import Any, NamedTuple, Optional, Union

import gym
import numpy as np
import psutil
import torch
from numpy import ndarray
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class HiddenReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    hidden_states: torch.Tensor
    next_hidden_states: torch.Tensor


class HiddenStateReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        single_observation_space: gym.spaces.Space,
        single_action_space: gym.spaces.Space,
        hidden_state_size: int,
        device: Union[torch.device, str],
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(HiddenStateReplayBuffer, self).__init__(
            buffer_size,
            single_observation_space,
            single_action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.hidden_states = np.zeros((self.buffer_size, self.n_envs, hidden_state_size), dtype=np.float32)
        self.next_hidden_states = np.zeros((self.buffer_size, self.n_envs, hidden_state_size), dtype=np.float32)

        if psutil is not None:
            mem_available = psutil.virtual_memory().available
            total_memory_usage = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes
            if self.hidden_states is not None:
                total_memory_usage += self.hidden_states.nbytes
            if self.next_hidden_states is not None:
                total_memory_usage += self.next_hidden_states.nbytes
            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: ndarray,
        next_obs: ndarray,
        action: ndarray,
        reward: ndarray,
        done: ndarray,
        hidden_state: ndarray,
        next_hidden_states: torch.Tensor,
        infos: list[dict[str, Any]],
    ) -> None:
        self.hidden_states[self.pos] = np.array(hidden_state).copy()
        if self.optimize_memory_usage:
            self.hidden_states[(self.pos + 1) % self.buffer_size] = np.array(next_hidden_states).copy()
        else:
            self.next_hidden_states[self.pos] = next_hidden_states.detach().numpy()
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> HiddenReplayBufferSamples:
        return super().sample(batch_size, env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> HiddenReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self._normalize_reward(self.rewards[batch_inds], env),
            self.hidden_states[batch_inds],
            self.next_hidden_states[batch_inds],
        )
        return HiddenReplayBufferSamples(*tuple(map(self.to_torch, data)))
