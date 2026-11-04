"""A minimal joint-space env over Ros2ControlState, for wiring tests and as a task template."""

from typing import Literal, override

import torch
from pydantic import Field
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from torch import Tensor

from src.base_config import Config
from src.reward import RewardFunction
from src.sim import VecSystem

from .ros2_control_system import Ros2ControlAction, Ros2ControlState


def velocity_norm(state: Ros2ControlState, _: Ros2ControlAction) -> Tensor:
    """Penalizes joint speed; a placeholder until a real task reward exists."""
    return -state.joint_velocities.norm(dim=-1)


class JointStateEnvConfig(Config):
    class_name: Literal["rozart_residual.joint_state_env:JointStateEnv"]

    max_episode_length: int = Field(default=600, gt=0)
    """Control steps before the robot is sent home and the episode restarts."""

    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {"velocity_norm": 1.0}
    )
    """The weights for each reward term, keyed by the term's name."""


class JointStateEnv(VecEnv):
    """Observes every Ros2ControlState field as its own group; actions pass straight through."""

    def __init__(
        self,
        system: VecSystem[Ros2ControlState, Ros2ControlAction, Config],
        cfg: JointStateEnvConfig,
    ):
        self._system = system
        self.cfg = cfg
        self.num_envs = system.num_envs
        self.num_actions = system.num_actions
        self.device = system.state.joint_positions.device
        self.max_episode_length = cfg.max_episode_length
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reward = RewardFunction[Ros2ControlState, Ros2ControlAction](
            [velocity_norm],
            weights=cfg.reward_weights,
            batch_size=self.num_envs,
            device=self.device,
        )

    @override
    def get_observations(self) -> TensorDict:
        # Clone so later steps cannot mutate an observation held in rollout storage.
        return TensorDict(
            {k: v.clone().float() for k, v in self._system.state.items()},
            batch_size=self.num_envs,
            device=self.device,
        )

    @override
    def step(self, actions: Tensor):
        action = Ros2ControlAction.from_tensor(actions.to(self.device))
        self._system.step(action)
        rewards = self._reward(self._system.state, action)
        log = self._reward.log()

        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        if time_outs.any():
            self._system.reset()
            self.episode_length_buf[:] = 0

        return (
            self.get_observations(),
            rewards,
            time_outs,
            {"time_outs": time_outs, "log": log},
        )
