"""Residual policy learning as an rsl_rl VecEnv wrapper.

The wrapped env sees `base(obs) + residual_scale * residual`, where `base` is a frozen,
exported hand-rl policy and `residual` is what rsl_rl trains. Because it is just a VecEnv, the
same wrapper trains the residual in Genesis (many envs) or fine-tunes it on the robot (one env).
"""

from typing import override

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch import Tensor

BASE_ACTION_KEY = "base_action"
"""Observation group holding the base policy's action; add it to the residual's obs_groups."""


class ResidualEnv(VecEnv):
    """Adds a frozen base policy's action to every action rsl_rl sends."""

    def __init__(
        self,
        env: VecEnv,
        base_policy: torch.nn.Module,
        base_obs_groups: list[str],
        residual_scale: float = 0.1,
        action_limit: float | None = None,
    ):
        """
        Args:
            env: The environment the combined action is applied to.
            base_policy: A TorchScript policy exported with `OnPolicyRunner.export_policy_to_jit`,
                which takes the concatenated `base_obs_groups`.
            base_obs_groups: The base policy's `obs_groups["actor"]`, in training order.
            residual_scale: Multiplies the residual, so an untrained residual (std ~1) stays a
                small correction around the base behavior.
            action_limit: Optional symmetric clamp on the combined normalized action.
        """
        self._env = env
        self._base = base_policy.eval()
        self._base_obs_groups = base_obs_groups
        self.residual_scale = residual_scale
        self._action_limit = action_limit

        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.cfg = env.cfg
        self._obs = self._with_base_action(env.get_observations())

    @property
    def episode_length_buf(self) -> Tensor:
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: Tensor):
        self._env.episode_length_buf = value

    @property
    def unwrapped(self) -> VecEnv:
        return self._env

    @torch.no_grad()
    def _with_base_action(self, obs: TensorDict) -> TensorDict:
        # Typed observations (e.g. PianoObservation) reject undeclared keys, so copy to a plain one.
        obs = TensorDict(dict(obs.items()), batch_size=obs.batch_size, device=obs.device)
        flat = torch.cat([obs[k].float() for k in self._base_obs_groups], dim=-1)
        base_device = next(self._base.parameters(), torch.empty(0)).device
        obs[BASE_ACTION_KEY] = self._base(flat.to(base_device)).to(flat.device)
        return obs

    @override
    def get_observations(self) -> TensorDict:
        return self._obs

    @override
    def step(self, actions: Tensor):
        action = self._obs[BASE_ACTION_KEY] + self.residual_scale * actions
        if self._action_limit is not None:
            action = action.clamp(-self._action_limit, self._action_limit)
        obs, rewards, dones, extras = self._env.step(action)
        # Base actions for reset envs are recomputed from their post-reset observations here.
        self._obs = self._with_base_action(obs)
        return self._obs, rewards, dones, extras


class ResidualPolicy(torch.nn.Module):
    """Combined base + residual policy for deployment from two TorchScript exports."""

    def __init__(
        self,
        base_policy: torch.nn.Module,
        residual_policy: torch.nn.Module | None,
        base_obs_groups: list[str],
        residual_obs_groups: list[str],
        residual_scale: float,
        action_limit: float | None = None,
    ):
        super().__init__()
        self.base = base_policy.eval()
        self.residual = None if residual_policy is None else residual_policy.eval()
        self.base_obs_groups = base_obs_groups
        self.residual_obs_groups = residual_obs_groups
        self.residual_scale = residual_scale
        self.action_limit = action_limit

    @torch.no_grad()
    def forward(self, obs: TensorDict) -> Tensor:
        obs = TensorDict(dict(obs.items()), batch_size=obs.batch_size, device=obs.device)
        flat = torch.cat([obs[k].float() for k in self.base_obs_groups], dim=-1)
        obs[BASE_ACTION_KEY] = self.base(flat)
        action = obs[BASE_ACTION_KEY]
        if self.residual is not None:
            flat = torch.cat([obs[k].float() for k in self.residual_obs_groups], dim=-1)
            action = action + self.residual_scale * self.residual(flat)
        if self.action_limit is not None:
            action = action.clamp(-self.action_limit, self.action_limit)
        return action
