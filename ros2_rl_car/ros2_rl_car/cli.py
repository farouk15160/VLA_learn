"""Command-line lifecycle for headless simulation, training, and scoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .core.constants import LEFT, RIGHT, STRAIGHT
from .core.track import ParametricTrack
from .evaluation.runner import (
    NullBaselineDiscriminationError,
    evaluate_suite,
    write_reports,
)
from .learning.ppo import PPOConfig
from .learning.trainer import PPOTrainer, TrainingConfig, train_with_dashboard
from .sim.bridge import RosBridge
from .sim.environment import GazeboDrivingEnv, GazeboEnvConfig
from .sim.processes import refuse_if_running, stop_gzservers
from .sim.world import DEFAULT_SAMPLES, generate_assets


def _share_directory() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("ros2_rl_car"))
    except (ImportError, LookupError):
        return Path(__file__).resolve().parents[1]


def _default_world() -> Path:
    return _share_directory() / "assets/worlds/alternating_track.world"


def _default_track() -> Path:
    return _share_directory() / "assets/worlds/centerline.csv"


class SimulatorProcess:
    def __init__(self, world: Path, gui: bool) -> None:
        refuse_if_running()
        if not world.is_file():
            raise FileNotFoundError(f"Gazebo world not found: {world}")
        environment = dict(os.environ)
        environment.setdefault("ROS_LOG_DIR", "/tmp/ros2-rl-car-logs")
        environment.setdefault("GAZEBO_LOG_PATH", "/tmp/gazebo-rl-car")
        Path(environment["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(environment["GAZEBO_LOG_PATH"]).mkdir(parents=True, exist_ok=True)
        command = [
            "gzserver", "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so", str(world),
        ]
        self.server = subprocess.Popen(command, env=environment)
        try:
            self.client = subprocess.Popen(["gzclient"], env=environment) if gui else None
        except Exception:
            self.server.terminate()
            self.server.wait(timeout=5.0)
            raise

    def close(self) -> None:
        for process in (self.client, self.server):
            if process is not None and process.poll() is None:
                process.terminate()
        for process in (self.client, self.server):
            if process is None:
                continue
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    def __enter__(self) -> "SimulatorProcess":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--world", type=Path, default=_default_world())
    parser.add_argument("--track", type=Path, default=_default_track())
    parser.add_argument(
        "--no-launch", action="store_true",
        help="attach to an already-running simulator instead of launching one",
    )
    parser.add_argument(
        "--gazebo-gui", action="store_true", help="also open gzclient (training remains accelerated)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-car")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="launch Gazebo and train discrete PPO")
    _add_runtime_arguments(train)
    train.add_argument("--config", type=Path, help="JSON defaults; explicit flags win")
    train.add_argument("--steps", type=int)
    train.add_argument("--rollout-steps", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--checkpoint-dir")
    train.add_argument("--lr", type=float)
    train.add_argument("--gamma", type=float)
    train.add_argument("--gae-lambda", type=float)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--clip", type=float)
    train.add_argument("--entropy-coef", type=float)
    train.add_argument("--value-coef", type=float)
    train.add_argument("--max-grad-norm", type=float)
    train.add_argument("--ppo-epochs", type=int)
    train.add_argument("--hidden-size", type=int)
    train.add_argument("--torch-threads", type=int)
    train.add_argument(
        "--gui", action="store_true",
        help="open the live training dashboard (headless is the default)",
    )
    evaluate = commands.add_parser("evaluate", help="score null, reference, and PPO policies")
    _add_runtime_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--seeds", nargs="+", type=int, default=[7, 19])
    evaluate.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    start = commands.add_parser("start", help="run Gazebo Classic in the foreground")
    start.add_argument("--world", type=Path, default=_default_world())
    start.add_argument("--gui", action="store_true", help="open gzclient")
    commands.add_parser("stop", help="stop orphaned Gazebo Classic servers")
    commands.add_parser("doctor", help="report exact ROS/Gazebo/Python/Torch capabilities")
    generate = commands.add_parser("generate", help="regenerate the deterministic track assets")
    generate.add_argument("--output", type=Path, default=_share_directory() / "assets/worlds")
    generate.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    smoke = commands.add_parser("smoke", help="test real reset, sensors, motion, and turn signs")
    _add_runtime_arguments(smoke)
    return parser


def _version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else f"exit {result.returncode}"


def _doctor() -> None:
    details: dict[str, object] = {
        "python": sys.version.split()[0],
        "ros_distro": os.environ.get("ROS_DISTRO", "not sourced"),
        "ros2": _version(["ros2", "--help"]),
        "gazebo": _version(["gazebo", "--version"]),
        "gzserver": _version(["gzserver", "--version"]),
    }
    try:
        import rclpy

        details["rclpy"] = str(Path(rclpy.__file__).resolve())
    except ImportError as exc:
        details["rclpy"] = f"unavailable: {exc}"
    try:
        import torch

        details.update(
            torch=str(torch.__version__),
            cuda_available=bool(torch.cuda.is_available()),
            selected_device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except ImportError as exc:
        details["torch"] = f"unavailable: {exc}"
    print(json.dumps(details, indent=2))


def _smoke(env: GazeboDrivingEnv) -> dict[str, object]:
    from .sim.environment import wrap_angle

    observations: dict[str, object] = {}
    env.bridge.wait_for_camera(timeout=10.0)
    for name, action in (("straight", STRAIGHT), ("left", LEFT), ("right", RIGHT)):
        _observation, initial = env.reset()
        start_pose = tuple(float(value) for value in initial["pose"])
        start_frames = int(initial["sensor_frames"])
        start_camera = env.bridge.snapshot().camera_frames
        result = None
        for _ in range(4):
            result = env.step(action)
            if result.terminated or result.truncated:
                break
        assert result is not None
        final_pose = tuple(float(value) for value in result.info["pose"])
        observations[name] = {
            "distance_m": float(np.hypot(final_pose[0] - start_pose[0], final_pose[1] - start_pose[1])),
            "yaw_delta_rad": wrap_angle(final_pose[2] - start_pose[2]),
            "fresh_sensor_frames": int(result.info["sensor_frames"]) - start_frames,
            "fresh_camera_frames": env.bridge.snapshot().camera_frames - start_camera,
        }
    straight = observations["straight"]
    left = observations["left"]
    right = observations["right"]
    failures = []
    if straight["distance_m"] <= 0.05:
        failures.append("straight command did not move at least 0.05 m")
    if left["yaw_delta_rad"] <= 0.0:
        failures.append("LEFT did not produce positive ROS yaw")
    if right["yaw_delta_rad"] >= 0.0:
        failures.append("RIGHT did not produce negative ROS yaw")
    for name, values in observations.items():
        if values["fresh_sensor_frames"] <= 0:
            failures.append(f"{name} received no fresh lidar/odom")
        if values["fresh_camera_frames"] <= 0:
            failures.append(f"{name} received no fresh camera frame")
    report = {"checks": observations, "passed": not failures, "failures": failures}
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("smoke test failed: " + "; ".join(failures))
    return report


def _run_train(args: argparse.Namespace, env: GazeboDrivingEnv) -> int:
    configured: dict[str, object] = {}
    if args.config is not None:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("training config must be a JSON object")
        configured = loaded

    def setting(argument: str, key: str, fallback: object) -> object:
        explicit = getattr(args, argument)
        return explicit if explicit is not None else configured.get(key, fallback)

    ppo_config = PPOConfig(
        learning_rate=float(setting("lr", "learning_rate", 3e-4)),
        gamma=float(setting("gamma", "gamma", 0.99)),
        gae_lambda=float(setting("gae_lambda", "gae_lambda", 0.95)),
        clip_coefficient=float(setting("clip", "clip_coefficient", 0.2)),
        entropy_coefficient=float(setting("entropy_coef", "entropy_coefficient", 0.01)),
        value_coefficient=float(setting("value_coef", "value_coefficient", 0.5)),
        max_gradient_norm=float(setting("max_grad_norm", "max_gradient_norm", 0.5)),
        update_epochs=int(setting("ppo_epochs", "update_epochs", 4)),
        minibatch_size=int(setting("batch_size", "minibatch_size", 64)),
        hidden_size=int(setting("hidden_size", "hidden_size", 128)),
    )
    trainer = PPOTrainer(
        env,
        ppo_config=ppo_config,
        training_config=TrainingConfig(
            total_steps=int(setting("steps", "total_steps", 200_000)),
            rollout_steps=int(setting("rollout_steps", "rollout_steps", 1024)),
            seed=int(setting("seed", "seed", 7)),
            checkpoint_dir=str(setting("checkpoint_dir", "checkpoint_dir", "checkpoints")),
            checkpoint_interval=int(configured.get("checkpoint_interval", 10)),
            torch_threads=int(setting("torch_threads", "torch_threads", 1)),
        ),
    )
    if args.gui:
        train_with_dashboard(trainer, env.track.points)
    else:
        destination = trainer.train()
        print(f"checkpoint: {destination}")
    return 0


def _run_evaluate(args: argparse.Namespace, env: GazeboDrivingEnv) -> int:
    try:
        reports = evaluate_suite(
            env,
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            seeds=tuple(args.seeds),
        )
        status = 0
    except NullBaselineDiscriminationError as exc:
        reports = exc.reports
        print(f"ACCEPTANCE FAILURE: {exc}", file=sys.stderr)
        status = 2
    destination = write_reports(reports, args.output)
    columns = ("policy", "seed", "success_rate", "mean_cross_track_error", "max_cross_track_error", "laps_completed", "collision_rate")
    print("\t".join(columns))
    for report in reports:
        values = report.as_dict()
        print("\t".join(str(values[column]) for column in columns))
    print(f"results: {destination}")
    return status


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        _doctor()
        return
    if args.command == "generate":
        world, centerline = generate_assets(args.output, args.samples)
        print(f"generated {world}\ngenerated {centerline}")
        return
    if args.command == "stop":
        stopped = stop_gzservers()
        print(f"stopped gzserver PID(s): {stopped}" if stopped else "no gzserver running")
        return
    if args.command == "start":
        simulator = SimulatorProcess(args.world, args.gui)
        try:
            return_code = simulator.server.wait()
        except KeyboardInterrupt:
            return_code = 0
        finally:
            simulator.close()
        if return_code:
            raise SystemExit(return_code)
        return
    track = ParametricTrack.load_csv(args.track)
    simulator = None if args.no_launch else SimulatorProcess(args.world, args.gazebo_gui)
    try:
        with RosBridge() as bridge:
            env = GazeboDrivingEnv(
                bridge,
                track,
                config=GazeboEnvConfig(max_steps=1200),
            )
            if args.command == "train":
                status = _run_train(args, env)
            elif args.command == "evaluate":
                status = _run_evaluate(args, env)
            else:
                _smoke(env)
                status = 0
    finally:
        if simulator is not None:
            simulator.close()
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
