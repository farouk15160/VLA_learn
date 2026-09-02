"""Run learned, null-baseline, and pure-pursuit evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from ..core.constants import ACTION_NAMES, NUMBER_OF_ACTIONS
from ..learning.ppo import ActorCritic
from ..sim.environment import GazeboDrivingEnv
from .controllers import ConstantController, PurePursuitController
from .metrics import EvaluationAccumulator, EvaluationSummary


class EvaluationPolicy(Protocol):
    name: str

    def action(self, observation: np.ndarray, info: dict[str, object]) -> int: ...


@dataclass(frozen=True)
class EvaluationReport:
    policy: str
    seed: int
    summary: EvaluationSummary
    direction_accuracy: float

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            **asdict(self.summary),
            "direction_accuracy": self.direction_accuracy,
        }


@dataclass(frozen=True)
class ConstantPolicy:
    action_index: int

    @property
    def name(self) -> str:
        return f"null_constant_{ACTION_NAMES[self.action_index]}"

    def action(self, _observation: np.ndarray, _info: dict[str, object]) -> int:
        return ConstantController(self.action_index).select_action()


@dataclass(frozen=True)
class PurePursuitPolicy:
    controller: PurePursuitController
    name: str = "pure_pursuit_ground_truth"

    def action(self, _observation: np.ndarray, info: dict[str, object]) -> int:
        x, y, yaw = info["pose"]
        return self.controller.select_action(float(x), float(y), float(yaw))


class LearnedPolicy:
    name = "ppo_greedy"

    def __init__(self, checkpoint: str | Path, observation_size: int) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("activate the Python 3.10 torch virtualenv") from exc
        self._torch = torch
        # ``weights_only=True`` prevents pickle globals from executing when a
        # user points the evaluator at an untrusted checkpoint.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("checkpoint must be a PPO checkpoint dictionary with model weights")
        if not isinstance(payload, Mapping) or not {"model", "observation_size"}.issubset(payload):
            raise ValueError("checkpoint must be a weights-only PPO mapping")
        if not isinstance(payload["model"], Mapping):
            raise ValueError("checkpoint model entry must be a state-dict mapping")
        if int(payload.get("observation_size", observation_size)) != observation_size:
            raise ValueError("checkpoint observation size does not match environment")
        ppo_config = payload.get("ppo_config", {})
        hidden_size = int(ppo_config.get("hidden_size", 128))
        self._model = ActorCritic(observation_size, NUMBER_OF_ACTIONS, hidden_size)
        self._model.load_state_dict(payload["model"])
        self._model.eval()

    def action(self, observation: np.ndarray, _info: dict[str, object]) -> int:
        with self._torch.no_grad():
            logits, _ = self._model(
                self._torch.as_tensor(observation, dtype=self._torch.float32).unsqueeze(0)
            )
        return int(logits.argmax(dim=-1).item())


class NullBaselineDiscriminationError(RuntimeError):
    def __init__(self, message: str, reports: list[EvaluationReport]) -> None:
        super().__init__(message)
        self.reports = reports


def evaluate_policy(
    env: GazeboDrivingEnv,
    policy: EvaluationPolicy,
    *,
    episodes: int = 10,
    seed: int = 7,
) -> EvaluationReport:
    if episodes < 10:
        raise ValueError("acceptance evaluation requires at least 10 episodes")
    np.random.seed(seed)
    accumulator = EvaluationAccumulator()
    direction_correct = 0
    actions_taken = 0
    reference = PurePursuitController(env.track)
    for _ in range(episodes):
        observation, info = env.reset()
        first_frames = int(info["sensor_frames"])
        errors: list[float] = []
        collision = False
        success = False
        while True:
            action = policy.action(observation, info)
            pose = tuple(float(value) for value in info["pose"])
            ideal = reference.select_action(*pose)
            direction_correct += int(action == ideal)
            actions_taken += 1
            result = env.step(action)
            observation, info = result.observation, result.info
            errors.append(abs(float(info["cross_track_error"])))
            collision = collision or bool(info["collision"])
            success = success or bool(info["lap_completed"])
            if result.terminated or result.truncated:
                break
        received = int(info["sensor_frames"]) - first_frames
        if received <= 0 or not errors:
            raise RuntimeError(
                "evaluation received zero sensor frames after reset; refusing false result"
            )
        accumulator.record_episode(
            errors,
            laps_completed=int(success),
            collision=collision,
            success=success,
            sensor_frames=received,
        )
    return EvaluationReport(
        policy.name,
        seed,
        accumulator.summary(),
        direction_correct / max(1, actions_taken),
    )


def evaluate_suite(
    env: GazeboDrivingEnv,
    *,
    checkpoint: str | Path | None,
    episodes: int = 10,
    seeds: tuple[int, ...] = (7, 19),
) -> list[EvaluationReport]:
    if len(seeds) < 2:
        raise ValueError("reproducibility evaluation requires at least two seeds")
    policies: list[EvaluationPolicy] = [
        *(ConstantPolicy(index) for index in range(NUMBER_OF_ACTIONS)),
        PurePursuitPolicy(PurePursuitController(env.track)),
    ]
    if checkpoint is not None:
        policies.append(LearnedPolicy(checkpoint, env.observation_size))
    reports = [
        evaluate_policy(env, policy, episodes=episodes, seed=seed)
        for seed in seeds
        for policy in policies
    ]
    bad_nulls = [
        report for report in reports
        if report.policy.startswith("null_constant_")
        and (report.summary.success_rate > 0.0 or report.summary.laps_completed > 0)
    ]
    if bad_nulls:
        names = ", ".join(f"{item.policy}@{item.seed}" for item in bad_nulls)
        raise NullBaselineDiscriminationError(
            f"track is not discriminative: constant policy completed a lap ({names})",
            reports,
        )
    return reports


def write_reports(reports: list[EvaluationReport], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([report.as_dict() for report in reports], indent=2) + "\n")
    return path


def main() -> None:
    from ..cli import main as cli_main
    import sys

    cli_main(["evaluate", *sys.argv[1:]])
