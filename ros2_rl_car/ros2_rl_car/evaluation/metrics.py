"""Evaluation metrics protected against dead simulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    sensor_frames: int
    cross_track_errors: tuple[float, ...]
    laps_completed: int = 0
    collision: bool = False
    success: bool = False

    def __post_init__(self) -> None:
        if self.sensor_frames < 0 or self.laps_completed < 0:
            raise ValueError("frame and lap counts cannot be negative")
        if any(not np.isfinite(value) for value in self.cross_track_errors):
            raise ValueError("cross-track errors must be finite")


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    episodes: int
    success_rate: float
    mean_cross_track_error: float
    max_cross_track_error: float
    laps_completed: int
    collision_rate: float
    sensor_frames: int


@dataclass(slots=True)
class EvaluationAccumulator:
    _episodes: list[EpisodeEvaluation] = field(default_factory=list)
    _sensor_frames: int = 0
    _live_cross_track_errors: list[float] = field(default_factory=list)

    def record_sensor_frame(self, cross_track_error: float | None = None) -> None:
        self._sensor_frames += 1
        if cross_track_error is not None:
            value = abs(float(cross_track_error))
            if not np.isfinite(value):
                raise ValueError("cross-track error must be finite")
            self._live_cross_track_errors.append(value)

    def add_episode(self, episode: EpisodeEvaluation) -> None:
        self._episodes.append(episode)

    def record_episode(
        self,
        cross_track_errors: Iterable[float],
        *,
        laps_completed: int = 0,
        collision: bool = False,
        success: bool | None = None,
        sensor_frames: int | None = None,
    ) -> None:
        errors = tuple(abs(float(value)) for value in cross_track_errors)
        frames = len(errors) if sensor_frames is None else int(sensor_frames)
        self.add_episode(EpisodeEvaluation(frames, errors, laps_completed, collision, bool(laps_completed) if success is None else success))

    def summary(self) -> EvaluationSummary:
        total_frames = self._sensor_frames + sum(episode.sensor_frames for episode in self._episodes)
        if total_frames == 0:
            raise RuntimeError("evaluation received zero sensor frames; simulator or QoS is not ready")
        errors = list(self._live_cross_track_errors)
        for episode in self._episodes:
            errors.extend(abs(value) for value in episode.cross_track_errors)
        episodes = len(self._episodes)
        return EvaluationSummary(
            episodes=episodes,
            success_rate=(sum(episode.success for episode in self._episodes) / episodes) if episodes else 0.0,
            mean_cross_track_error=float(np.mean(errors)) if errors else 0.0,
            max_cross_track_error=float(np.max(errors)) if errors else 0.0,
            laps_completed=sum(episode.laps_completed for episode in self._episodes),
            collision_rate=(sum(episode.collision for episode in self._episodes) / episodes) if episodes else 0.0,
            sensor_frames=total_frames,
        )
