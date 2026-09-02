"""Thread-safe snapshots shared by the trainer and dashboard."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from threading import Lock
from typing import Deque, Mapping

import numpy as np


@dataclass(frozen=True)
class EpisodeTelemetry:
    reward: float
    length: int
    success: bool
    reason: str
    cross_track_mean: float = 0.0
    cross_track_max: float = 0.0


@dataclass(frozen=True)
class UpdateTelemetry:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    grad_norm: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0


@dataclass(frozen=True)
class LiveTelemetry:
    episodes: tuple[EpisodeTelemetry, ...] = ()
    updates: tuple[UpdateTelemetry, ...] = ()
    trajectory: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    ray_endpoints: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    car_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    action_probabilities: tuple[float, ...] = ()
    value_estimate: float = 0.0
    hyperparameters: Mapping[str, object] = field(default_factory=dict)
    events: tuple[str, ...] = ()
    paused: bool = False
    greedy: bool = False
    finished: bool = False


class TelemetryStore:
    """A small immutable-snapshot store safe for Tk and a training thread."""

    def __init__(self, history: int = 1000) -> None:
        self._lock = Lock()
        self._episodes: Deque[EpisodeTelemetry] = deque(maxlen=history)
        self._updates: Deque[UpdateTelemetry] = deque(maxlen=history)
        self._events: Deque[str] = deque(maxlen=history)
        self._state = LiveTelemetry()

    def set_hyperparameters(self, values: Mapping[str, object]) -> None:
        self._replace(hyperparameters=dict(values))

    def add_episode(self, episode: EpisodeTelemetry) -> None:
        with self._lock:
            self._episodes.append(episode)
            self._state = replace(self._state, episodes=tuple(self._episodes))

    def add_update(self, update: UpdateTelemetry) -> None:
        with self._lock:
            self._updates.append(update)
            self._state = replace(self._state, updates=tuple(self._updates))

    def event(self, message: str) -> None:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        with self._lock:
            self._events.append(f"[{timestamp}] {message}")
            self._state = replace(self._state, events=tuple(self._events))

    def set_live(
        self,
        *,
        trajectory: np.ndarray,
        ray_endpoints: np.ndarray,
        car_pose: tuple[float, float, float],
        action_probabilities: np.ndarray | tuple[float, ...],
        value_estimate: float,
    ) -> None:
        self._replace(
            trajectory=np.asarray(trajectory, dtype=float).copy(),
            ray_endpoints=np.asarray(ray_endpoints, dtype=float).copy(),
            car_pose=tuple(float(v) for v in car_pose),
            action_probabilities=tuple(float(v) for v in action_probabilities),
            value_estimate=float(value_estimate),
        )

    def set_flags(self, **flags: bool) -> None:
        allowed = {"paused", "greedy", "finished"}
        unknown = set(flags) - allowed
        if unknown:
            raise ValueError(f"unknown telemetry flags: {sorted(unknown)}")
        self._replace(**flags)

    def snapshot(self) -> LiveTelemetry:
        with self._lock:
            return replace(
                self._state,
                trajectory=self._state.trajectory.copy(),
                ray_endpoints=self._state.ray_endpoints.copy(),
                hyperparameters=dict(self._state.hyperparameters),
            )

    def _replace(self, **changes: object) -> None:
        with self._lock:
            self._state = replace(self._state, **changes)
