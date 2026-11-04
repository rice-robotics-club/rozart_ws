"""A hand-rl VecSystem backed by a real robot running ros2_control.

Commands go to a forward_command_controller (e.g. position_controllers/JointGroupPositionController)
and state comes from joint_state_broadcaster, so any hand-rl VecEnv written against this system's
state runs unchanged on hardware with `num_envs == 1`.
"""

import threading
import time
from typing import Literal, override

import torch
from jaxtyping import Float
from pydantic import Field, model_validator
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tensordict import TypedTensorDict

from src.base_config import Config
from src.sim.vec_system import VecAction, VecSystem


class Ros2ControlState(TypedTensorDict):
    """Latest measured joint state, in `joint_names` order; matches what a Genesis twin can report."""

    joint_positions: Float[torch.Tensor, "*batch dofs"]
    joint_velocities: Float[torch.Tensor, "*batch dofs"]
    commanded_positions: Float[torch.Tensor, "*batch dofs"]
    """Position targets last sent to the controller, after clamping."""
    time: Float[torch.Tensor, "*batch 1"]
    """Seconds since the last reset, counted in control steps to match simulation."""
    control_dt: Float[torch.Tensor, "*batch 1"]


class Ros2ControlAction(VecAction):
    """Normalized offsets from `default_positions`, used as position targets."""

    joint_positions: Float[torch.Tensor, "*batch dofs"]

    @classmethod
    @override
    def from_tensor(cls, tensor: torch.Tensor) -> "Ros2ControlAction":
        return cls(joint_positions=tensor, batch_size=tensor.shape[:-1])


class Ros2ControlSystemConfig(Config):
    class_name: Literal["rozart_residual.ros2_control_system:Ros2ControlSystem"] = (
        "rozart_residual.ros2_control_system:Ros2ControlSystem"
    )

    joint_names: list[str] = Field(min_length=1)
    """Joints commanded by the controller, in the controller's `joints` order."""
    default_positions: list[float]
    """Home pose in radians; actions are offsets from it, as in hand-rl's simulations."""
    lower_limits: list[float]
    """Hard lower position limits in radians; commands are clamped to them."""
    upper_limits: list[float]
    """Hard upper position limits in radians; commands are clamped to them."""

    command_topic: str = "/position_controller/commands"
    """Float64MultiArray topic of the forward_command_controller."""
    joint_state_topic: str = "/joint_states"
    """JointState topic of joint_state_broadcaster."""

    control_dt: float = Field(default=1.0 / 60.0, gt=0, allow_inf_nan=False)
    """Seconds per VecSystem.step; must equal the control_dt the policy was trained with."""
    action_scale: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    """Scale applied to actions before adding them to the default pose; match training."""
    max_step_delta: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    """Largest change in any commanded position per step, in radians."""
    stale_timeout: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    """Seconds without a joint state before stepping raises instead of commanding blind."""
    reset_duration: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    """Seconds spent interpolating back to the home pose on reset."""

    @model_validator(mode="after")
    def validate_lengths(self):
        n = len(self.joint_names)
        for name in ("default_positions", "lower_limits", "upper_limits"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have one entry per joint in joint_names")
        if any(lo >= hi for lo, hi in zip(self.lower_limits, self.upper_limits)):
            raise ValueError("Each lower limit must be below its upper limit")
        if any(
            not lo <= d <= hi
            for d, lo, hi in zip(
                self.default_positions, self.lower_limits, self.upper_limits
            )
        ):
            raise ValueError("default_positions must lie within the limits")
        return self


class StaleJointStateError(RuntimeError):
    """Raised when the robot stops reporting its state."""


class Ros2ControlSystem(VecSystem[Ros2ControlState, Ros2ControlAction, Ros2ControlSystemConfig]):
    """One real robot, stepped at a fixed rate.

    The node must be spun on another thread (see residual_node); `step` blocks for one control
    period and is meant to be called from the learning or inference loop.
    """

    def __init__(self, cfg: Ros2ControlSystemConfig, node: Node):
        super().__init__(cfg)
        self._node = node
        self._index = {name: i for i, name in enumerate(cfg.joint_names)}
        self._default = torch.tensor(cfg.default_positions)
        self._lower = torch.tensor(cfg.lower_limits)
        self._upper = torch.tensor(cfg.upper_limits)

        n = len(cfg.joint_names)
        self._lock = threading.Lock()
        self._measured_pos = torch.zeros(n)
        self._measured_vel = torch.zeros(n)
        self._seen = torch.zeros(n, dtype=torch.bool)
        self._stamp: float | None = None
        self._steps = 0
        self._deadline = time.monotonic()

        self._state = Ros2ControlState(
            joint_positions=torch.zeros((1, n)),
            joint_velocities=torch.zeros((1, n)),
            commanded_positions=self._default.clone().unsqueeze(0),
            time=torch.zeros((1, 1), dtype=torch.float64),
            control_dt=torch.full((1, 1), cfg.control_dt, dtype=torch.float64),
            batch_size=1,
        )

        self._publisher = node.create_publisher(Float64MultiArray, cfg.command_topic, 1)
        self._subscription = node.create_subscription(
            JointState, cfg.joint_state_topic, self._on_joint_state, 10
        )

    @property
    @override
    def num_envs(self) -> int:
        return 1

    @property
    @override
    def num_actions(self) -> int:
        return len(self.cfg.joint_names)

    @property
    @override
    def state(self) -> Ros2ControlState:
        return self._state

    def wait_for_state(self, timeout: float = 5.0):
        """Blocks until every configured joint has been reported once."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._seen.all():
                    break
            time.sleep(0.01)
        else:
            missing = [n for n, s in zip(self.cfg.joint_names, self._seen) if not s]
            raise StaleJointStateError(f"No joint state received for {missing}")
        self._refresh_state()
        self._state.commanded_positions.copy_(self._state.joint_positions)

    @override
    def step(self, action: Ros2ControlAction | torch.Tensor):
        action = Ros2ControlAction.from_(action)
        offsets = action.joint_positions.detach().to("cpu", torch.float32).reshape(1, -1)
        targets = self._default + self.cfg.action_scale * offsets
        self._command(targets)
        self._sleep_until_next_period()
        self._steps += 1
        self._refresh_state()

    @override
    def reset(self, idx: int | torch.Tensor | None = None):
        """Interpolates from the current command back to the home pose, then restarts the clock."""
        start = self._state.commanded_positions.clone()
        n = max(1, round(self.cfg.reset_duration / self.cfg.control_dt))
        for k in range(1, n + 1):
            self._command(torch.lerp(start, self._default.unsqueeze(0), k / n), limit_rate=False)
            self._sleep_until_next_period()
        self._steps = 0
        self._refresh_state()

    def hold(self):
        """Commands the last measured position, e.g. on shutdown or error."""
        with self._lock:
            measured = self._measured_pos.clone()
        if self._seen.all():
            self._command(measured.unsqueeze(0), limit_rate=False)

    def _command(self, targets: torch.Tensor, limit_rate: bool = True):
        if limit_rate:
            previous = self._state.commanded_positions
            delta = self.cfg.max_step_delta
            targets = targets.clamp(previous - delta, previous + delta)
        targets = targets.clamp(self._lower, self._upper)
        self._state.commanded_positions.copy_(targets)
        self._publisher.publish(Float64MultiArray(data=targets[0].tolist()))

    def _sleep_until_next_period(self):
        # Absolute deadlines keep the average rate exact; after an overrun, resynchronize
        # instead of bursting commands to catch up.
        self._deadline += self.cfg.control_dt
        now = time.monotonic()
        if self._deadline > now:
            time.sleep(self._deadline - now)
        else:
            self._deadline = now

    def _on_joint_state(self, msg: JointState):
        with self._lock:
            for i, name in enumerate(msg.name):
                j = self._index.get(name)
                if j is None:
                    continue
                if i < len(msg.position):
                    self._measured_pos[j] = msg.position[i]
                    self._seen[j] = True
                if i < len(msg.velocity):
                    self._measured_vel[j] = msg.velocity[i]
            self._stamp = time.monotonic()

    def _refresh_state(self):
        with self._lock:
            if self._stamp is None or time.monotonic() - self._stamp > self.cfg.stale_timeout:
                raise StaleJointStateError(
                    f"No {self.cfg.joint_state_topic} message for {self.cfg.stale_timeout}s"
                )
            self._state.joint_positions[0].copy_(self._measured_pos)
            self._state.joint_velocities[0].copy_(self._measured_vel)
        self._state.time.fill_(self._steps * self.cfg.control_dt)
