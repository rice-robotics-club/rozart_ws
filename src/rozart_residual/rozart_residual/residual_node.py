"""Runs a hand-rl base policy plus a learned residual on Rozart through ros2_control.

Modes:
    deploy:   run base + residual (or base alone) from TorchScript exports.
    finetune: train the residual on the robot with rsl_rl PPO, starting from a sim checkpoint.

The env and runner sections of the config are the same ones hand-rl uses for training; only the
system section changes, from a Genesis simulation to Ros2ControlSystem.
"""

import signal
import threading
from pathlib import Path
from typing import Annotated, Any, Literal

import rclpy
import torch
import yaml
from pydantic import Field, TypeAdapter
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_class
from src.base_config import Config
from src.config.config import EnvConfig
from src.config.runner import RunnerConfig

from .joint_state_env import JointStateEnvConfig
from .residual import ResidualEnv, ResidualPolicy
from .ros2_control_system import Ros2ControlSystem, Ros2ControlSystemConfig

AnyEnvConfig = Annotated[
    EnvConfig | JointStateEnvConfig, Field(discriminator="class_name")
]
"""hand-rl's envs plus this package's; an env must only read fields Ros2ControlState has."""


class ResidualConfig(Config):
    base_policy: str
    """TorchScript base policy from `OnPolicyRunner.export_policy_to_jit`."""
    base_obs_groups: list[str] = Field(min_length=1)
    """The base policy's obs_groups["actor"], in training order."""
    residual_policy: str | None = None
    """TorchScript residual for deploy mode; None runs the base policy alone."""
    residual_obs_groups: list[str] = Field(default_factory=list)
    """The residual's obs_groups["actor"]; usually base_obs_groups plus "base_action"."""
    residual_scale: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    """Must match the scale the residual was trained with."""
    action_limit: float | None = Field(default=None, gt=0)
    """Optional clamp on the combined normalized action."""


class DeploymentConfig(Config):
    """Root configuration; env and runner are validated with hand-rl's own config types."""

    system: Ros2ControlSystemConfig
    residual: ResidualConfig
    env: AnyEnvConfig
    runner: dict[str, Any] | None = None
    """A hand-rl/rsl_rl runner config for the residual; required for finetune."""


class ResidualNode(Node):
    def __init__(self):
        super().__init__("residual_node")
        self.declare_parameter("config", "")
        self.declare_parameter("mode", "deploy")
        # rsl_rl checkpoint of a sim-trained residual to start fine-tuning from.
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("log_dir", "logs/rozart_finetune")
        self.declare_parameter("iterations", 100)
        self.declare_parameter("device", "cpu")

        path = self.get_parameter("config").value
        if not path:
            raise ValueError("Set the 'config' parameter to a deployment YAML file")
        self.cfg = DeploymentConfig.model_validate(
            yaml.safe_load(Path(path).read_text())
        )
        self.mode: Literal["deploy", "finetune"] = self.get_parameter("mode").value  # type: ignore
        if self.mode not in ("deploy", "finetune"):
            raise ValueError("mode must be 'deploy' or 'finetune'")
        self.device = self.get_parameter("device").value

        self.system = Ros2ControlSystem(self.cfg.system, self)

    def run(self):
        """Blocking control loop; call with the node spinning on another thread."""
        self.get_logger().info("Waiting for joint states...")
        self.system.wait_for_state()
        self.get_logger().info("Moving to the home pose")
        self.system.reset()

        env_class, _ = resolve_class(self.cfg.env.model_dump())
        env = env_class(self.system, self.cfg.env)
        res = self.cfg.residual
        base = torch.jit.load(res.base_policy, map_location=self.device)

        if self.mode == "deploy":
            self._deploy(env, base)
        else:
            self._finetune(env, base)

    def _deploy(self, env, base):
        res = self.cfg.residual
        residual = (
            None
            if res.residual_policy is None
            else torch.jit.load(res.residual_policy, map_location=self.device)
        )
        policy = ResidualPolicy(
            base,
            residual,
            res.base_obs_groups,
            res.residual_obs_groups,
            res.residual_scale,
            res.action_limit,
        )
        self.get_logger().info(
            "Running base policy" + ("" if residual is None else " with residual")
        )
        obs = env.get_observations()
        while rclpy.ok():
            obs, _, dones, extras = env.step(policy(obs.to(self.device)))
            if dones.any():
                terms = ", ".join(f"{k}: {v:.4f}" for k, v in extras["log"].items())
                self.get_logger().info(f"Episode done ({terms})")

    def _finetune(self, env, base):
        if self.cfg.runner is None:
            raise ValueError("finetune mode needs a runner section")
        res = self.cfg.residual
        env = ResidualEnv(
            env, base, res.base_obs_groups, res.residual_scale, res.action_limit
        )
        runner_cfg = TypeAdapter(RunnerConfig).validate_python(self.cfg.runner)
        runner_class, _ = resolve_class(runner_cfg.model_dump())
        log_dir = self.get_parameter("log_dir").value
        runner: OnPolicyRunner = runner_class(
            env, train_cfg=runner_cfg.model_dump(), log_dir=log_dir, device=self.device
        )
        checkpoint = self.get_parameter("checkpoint").value
        if checkpoint:
            self.get_logger().info(f"Loading residual checkpoint {checkpoint}")
            runner.load(checkpoint)
        # PPO updates block the loop; the forward command controller holds the last target meanwhile.
        runner.learn(num_learning_iterations=self.get_parameter("iterations").value)
        runner.export_policy_to_jit(log_dir, "residual.pt")
        self.get_logger().info(f"Exported fine-tuned residual to {log_dir}/residual.pt")


def _interrupt(*_):
    raise KeyboardInterrupt


def main(args=None):
    # rclpy's own handlers invalidate the context on SIGINT, before hold() can publish.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        node = ResidualNode()
    except Exception:
        rclpy.try_shutdown()
        raise
    # One spinning thread is enough for one subscription; a thread pool contends for the GIL.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.system.hold()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
