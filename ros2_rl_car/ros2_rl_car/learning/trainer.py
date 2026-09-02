"""CPU-friendly PPO training, telemetry, and checkpoint lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock, Thread
import time

import numpy as np

from ..core.constants import NUMBER_OF_ACTIONS
from ..dashboard.telemetry import EpisodeTelemetry, TelemetryStore, UpdateTelemetry
from ..sim.environment import GazeboDrivingEnv
from .ppo import ActorCritic, PPOConfig, compute_gae, parameter_count, ppo_update


@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int = 200_000
    rollout_steps: int = 1024
    seed: int = 7
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 10
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.rollout_steps < 1:
            raise ValueError("total_steps and rollout_steps must be positive")
        if self.checkpoint_interval < 1 or self.torch_threads < 1:
            raise ValueError("checkpoint_interval and torch_threads must be positive")
        if not self.checkpoint_dir:
            raise ValueError("checkpoint_dir must not be empty")


class PPOTrainer:
    def __init__(
        self,
        env: GazeboDrivingEnv,
        *,
        ppo_config: PPOConfig = PPOConfig(),
        training_config: TrainingConfig = TrainingConfig(),
        telemetry: TelemetryStore | None = None,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("activate the Python 3.10 torch virtualenv") from exc
        self.torch = torch
        if training_config.torch_threads < 1:
            raise ValueError("torch_threads must be at least one")
        # Tiny MLP minibatches are dramatically faster without intra-op thread
        # coordination (measured 0.027 s vs 1.26 s/update on this 16-thread CPU).
        torch.set_num_threads(training_config.torch_threads)
        self.env = env
        self.ppo_config = ppo_config
        self.config = training_config
        # Explicit capability check. This installation is expected to report False.
        self.cuda_available = bool(torch.cuda.is_available())
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.model = ActorCritic(
            env.observation_size, NUMBER_OF_ACTIONS, ppo_config.hidden_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=ppo_config.learning_rate
        )
        self.telemetry = telemetry or TelemetryStore()
        self._paused = Event()
        self._greedy = Event()
        self._save_requested = Event()
        self._reset_requested = Event()
        self._stop = Event()
        self._checkpoint_lock = Lock()
        self._global_step = 0
        self._update = 0
        self._best_running_reward = -np.inf
        self.telemetry.set_hyperparameters(
            {
                "lr": ppo_config.learning_rate,
                "gamma": ppo_config.gamma,
                "lambda": ppo_config.gae_lambda,
                "batch": ppo_config.minibatch_size,
                "rollout": training_config.rollout_steps,
                "clip": ppo_config.clip_coefficient,
                "entropy coef": ppo_config.entropy_coefficient,
                "seed": training_config.seed,
                "device": str(self.device),
                "cuda available": self.cuda_available,
                "parameters": parameter_count(self.model),
                "torch threads": training_config.torch_threads,
            }
        )

    def set_paused(self, paused: bool) -> None:
        self.env.bridge.command(0.0, 0.0)
        self._paused.set() if paused else self._paused.clear()
        if not paused:
            self._reset_requested.set()
        self.telemetry.set_flags(paused=paused)
        self.telemetry.event("training paused" if paused else "training resumed")

    def set_greedy(self, greedy: bool) -> None:
        self.env.bridge.command(0.0, 0.0)
        self._greedy.set() if greedy else self._greedy.clear()
        self._reset_requested.set()
        self.telemetry.set_flags(greedy=greedy)
        self.telemetry.event(f"greedy watch mode {'enabled' if greedy else 'disabled'}")

    def request_save(self) -> None:
        self._save_requested.set()

    def stop(self) -> None:
        self._stop.set()
        self.env.bridge.command(0.0, 0.0)

    def train(self) -> Path:
        torch = self.torch
        observation, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_errors: list[float] = []
        episode_direction_correct = 0
        update_count = 0
        self.telemetry.event("received lidar and odometry; PPO training started")
        while self._global_step < self.config.total_steps and not self._stop.is_set():
            batch: dict[str, list[object]] = {
                key: [] for key in (
                    "observations", "actions", "log_probabilities", "rewards",
                    "values", "next_values", "terminated", "truncated",
                )
            }
            collected = 0
            while collected < self.config.rollout_steps:
                if self._global_step >= self.config.total_steps or self._stop.is_set():
                    break
                while self._paused.is_set() and not self._stop.is_set():
                    time.sleep(0.05)
                if self._stop.is_set():
                    break
                if self._reset_requested.is_set():
                    if batch["truncated"] and not batch["terminated"][-1]:
                        # Cut GAE at the user-induced reset, retaining V(s_last).
                        batch["truncated"][-1] = True
                    observation, info = self.env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                    episode_errors = []
                    episode_direction_correct = 0
                    self._reset_requested.clear()
                observation_tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                greedy_action = self._greedy.is_set()
                with torch.no_grad():
                    logits, current_value = self.model(observation_tensor)
                    distribution = torch.distributions.Categorical(logits=logits)
                    action_tensor = (
                        torch.argmax(logits, dim=-1)
                        if greedy_action
                        else distribution.sample()
                    )
                    log_probability = distribution.log_prob(action_tensor)
                    probabilities = distribution.probs.squeeze(0).cpu().numpy()
                action = int(action_tensor.item())
                result = self.env.step(action)
                with torch.no_grad():
                    next_value = self.model(
                        torch.as_tensor(result.observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    )[1].item()
                training_sample = bool(
                    not greedy_action
                    and not self._paused.is_set()
                    and not self._reset_requested.is_set()
                )
                if training_sample:
                    batch["observations"].append(observation.copy())
                    batch["actions"].append(action)
                    batch["log_probabilities"].append(float(log_probability.item()))
                    batch["rewards"].append(result.reward)
                    batch["values"].append(float(current_value.item()))
                    batch["next_values"].append(float(next_value))
                    batch["terminated"].append(result.terminated)
                    batch["truncated"].append(result.truncated)
                    self._global_step += 1
                    collected += 1
                observation = result.observation
                info = result.info
                episode_reward += result.reward
                episode_length += 1
                episode_errors.append(abs(float(info["cross_track_error"])))
                episode_direction_correct += self._direction_correct(observation, action)
                self.telemetry.set_live(
                    trajectory=self.env.trajectory,
                    ray_endpoints=np.asarray(info["ray_endpoints"]),
                    car_pose=tuple(info["pose"]),
                    action_probabilities=probabilities,
                    value_estimate=float(current_value.item()),
                )
                if result.terminated or result.truncated:
                    success = bool(info["lap_completed"])
                    reason = str(info.get("reason") or "unknown")
                    self.telemetry.add_episode(
                        EpisodeTelemetry(
                            episode_reward,
                            episode_length,
                            success,
                            reason,
                            float(np.mean(episode_errors)),
                            float(np.max(episode_errors)),
                        )
                    )
                    direction_accuracy = episode_direction_correct / max(1, episode_length)
                    self.telemetry.event(
                        f"episode {len(self.telemetry.snapshot().episodes)}: {reason}; "
                        f"reward={episode_reward:.2f}, steps={episode_length}, "
                        f"direction_accuracy={direction_accuracy:.1%}"
                    )
                    observation, info = self.env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                    episode_errors = []
                    episode_direction_correct = 0

            if not batch["rewards"]:
                break
            metrics = self._update_policy(batch)
            update_count += 1
            self._update += 1
            self.telemetry.add_update(
                UpdateTelemetry(
                    metrics.policy_loss, metrics.value_loss, metrics.entropy,
                    metrics.gradient_norm, metrics.approximate_kl, metrics.clip_fraction,
                )
            )
            rewards = [episode.reward for episode in self.telemetry.snapshot().episodes[-20:]]
            running_reward = float(np.mean(rewards)) if rewards else -np.inf
            if running_reward > self._best_running_reward:
                self._best_running_reward = running_reward
                self.telemetry.event(f"new best running reward: {running_reward:.2f}")
                self.save_checkpoint("best.pt")
            if self._save_requested.is_set() or update_count % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
                self._save_requested.clear()
        path = self.save_checkpoint("final.pt")
        self.telemetry.set_flags(finished=True)
        self.telemetry.event(f"training finished at {self._global_step} environment steps")
        return path

    @staticmethod
    def _direction_correct(observation: np.ndarray, action: int) -> int:
        # Layout tail is speed, sin(heading error), cos(error), CTE.
        sine_error = float(observation[-3])
        desired = 2 if sine_error > 0.08 else 0 if sine_error < -0.08 else 1
        return int(action == desired)

    def _update_policy(self, batch: dict[str, list[object]]):
        torch = self.torch
        advantages, returns = compute_gae(
            np.asarray(batch["rewards"], dtype=np.float32),
            np.asarray(batch["values"], dtype=np.float32),
            np.asarray(batch["next_values"], dtype=np.float32),
            np.asarray(batch["terminated"], dtype=bool),
            np.asarray(batch["truncated"], dtype=bool),
            self.ppo_config.gamma,
            self.ppo_config.gae_lambda,
        )
        return ppo_update(
            self.model,
            self.optimizer,
            torch.as_tensor(np.asarray(batch["observations"]), dtype=torch.float32, device=self.device),
            torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device),
            torch.as_tensor(batch["log_probabilities"], dtype=torch.float32, device=self.device),
            torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            self.ppo_config,
        )

    def save_checkpoint(self, filename: str | None = None) -> Path:
        directory = Path(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        name = filename or f"ppo_step_{self._global_step:08d}.pt"
        target = directory / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        with self._checkpoint_lock:
            self.torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "global_step": self._global_step,
                    "update": self._update,
                    "ppo_config": asdict(self.ppo_config),
                    "training_config": asdict(self.config),
                    "observation_size": self.env.observation_size,
                    "action_count": NUMBER_OF_ACTIONS,
                },
                temporary,
            )
            temporary.replace(target)
        self.telemetry.event(f"checkpoint saved: {target}")
        return target


def train_with_dashboard(trainer: PPOTrainer, track_points: np.ndarray) -> None:
    from ..dashboard.ui import TrainingDashboard

    failures: list[Exception] = []

    def run_training() -> None:
        try:
            trainer.train()
        except Exception as exc:
            failures.append(exc)
            trainer.env.bridge.command(0.0, 0.0)
            trainer.telemetry.event(f"training failed: {type(exc).__name__}: {exc}")
            trainer.telemetry.set_flags(finished=True)

    worker = Thread(target=run_training, name="ppo-training", daemon=False)
    worker.start()
    dashboard = TrainingDashboard(
        trainer.telemetry,
        track_points,
        on_pause=trainer.set_paused,
        on_save=trainer.request_save,
        on_greedy=trainer.set_greedy,
    )
    dashboard.run()
    trainer.stop()
    worker.join()
    if failures:
        raise failures[0]


def main() -> None:
    from ..cli import main as cli_main
    import sys

    cli_main(["train", *sys.argv[1:]])
