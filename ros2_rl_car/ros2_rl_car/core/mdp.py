"""Core observation, reward, and boundary definitions for the driving MDP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    lidar_beams: int = 31
    lidar_max_range: float = 10.0
    max_speed: float = 2.0

    def __post_init__(self) -> None:
        if self.lidar_beams < 1:
            raise ValueError("lidar_beams must be positive")
        if self.lidar_max_range <= 0.0 or self.max_speed <= 0.0:
            raise ValueError("range and speed scales must be positive")

    @property
    def size(self) -> int:
        return self.lidar_beams + 4


def build_observation(
    ranges: Sequence[float] | np.ndarray,
    speed: float,
    heading_error: float,
    cross_track_error: float,
    road_half_width: float,
    config: ObservationConfig,
) -> np.ndarray:
    """Build lidar + proprioception features without absolute world position.

    Layout: fixed-width normalized lidar, normalized speed, sine and cosine of
    heading error, and signed normalized cross-track error.
    """

    scan = np.asarray(ranges, dtype=np.float64).reshape(-1)
    if scan.size == 0:
        raise ValueError("ranges must contain at least one measurement")
    if road_half_width <= 0.0:
        raise ValueError("road_half_width must be positive")
    clean = np.nan_to_num(scan, nan=config.lidar_max_range, posinf=config.lidar_max_range, neginf=0.0)
    clean = np.clip(clean, 0.0, config.lidar_max_range)
    if clean.size != config.lidar_beams:
        source = np.linspace(0.0, 1.0, clean.size)
        target = np.linspace(0.0, 1.0, config.lidar_beams)
        clean = np.interp(target, source, clean)
    lidar = clean / config.lidar_max_range
    extras = np.array(
        [
            np.clip(float(speed) / config.max_speed, -1.0, 1.0),
            math.sin(float(heading_error)),
            math.cos(float(heading_error)),
            np.clip(float(cross_track_error) / road_half_width, -1.0, 1.0),
        ],
        dtype=np.float64,
    )
    observation = np.concatenate((lidar, extras)).astype(np.float32)
    # Invalid scalar sensor values fail safe instead of entering the network.
    return np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    progress_weight: float = 10.0
    cross_track_potential_weight: float = 1.5
    time_cost: float = 0.01
    crash_penalty: float = -25.0
    lap_bonus: float = 10.0
    road_half_width: float = 1.5
    track_length: float = 50.0
    collision_range: float = 0.12

    def __post_init__(self) -> None:
        if self.road_half_width <= 0.0 or self.track_length <= 0.0:
            raise ValueError("road_half_width and track_length must be positive")
        if self.progress_weight < 0.0 or self.cross_track_potential_weight < 0.0:
            raise ValueError("reward weights must be non-negative")
        if self.time_cost < 0.0 or self.crash_penalty >= 0.0:
            raise ValueError("time cost must be non-negative and crash penalty negative")
        if abs(self.crash_penalty) <= self.max_shortcut_shaping:
            raise ValueError("crash penalty must exceed bankable shortcut shaping")

    @property
    def max_shortcut_shaping(self) -> float:
        """Largest potential reward bankable before crossing the road edge."""

        return self.progress_weight + (
            self.cross_track_potential_weight * self.road_half_width
        )


@dataclass(frozen=True, slots=True)
class RewardInput:
    progress_delta: float
    previous_cross_track: float
    cross_track: float
    collision: bool = False
    lap_completed: bool = False
    step_count: int = 0
    max_steps: int | None = None


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    terminated: bool
    truncated: bool
    reason: str | None
    progress_reward: float
    shaping_reward: float


def reward_step(inputs: RewardInput, config: RewardConfig) -> RewardResult:
    """Calculate one transition reward with separate terminal semantics.

    Cross-track shaping is the potential difference ``Phi(s')-Phi(s)`` where
    ``Phi=-w*|cte|``.  Its sum telescopes, so steering in and out cannot farm it.
    """

    values = (inputs.progress_delta, inputs.previous_cross_track, inputs.cross_track)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("reward inputs must be finite")
    off_road = abs(inputs.cross_track) > config.road_half_width
    terminated = bool(inputs.collision or off_road or inputs.lap_completed)
    truncated = bool(
        not terminated and inputs.max_steps is not None and inputs.step_count >= inputs.max_steps
    )
    reason = (
        "collision"
        if inputs.collision
        else "off_road"
        if off_road
        else "lap_complete"
        if inputs.lap_completed
        else "time_limit"
        if truncated
        else None
    )
    progress_reward = config.progress_weight * inputs.progress_delta / config.track_length
    shaping_reward = config.cross_track_potential_weight * (
        abs(inputs.previous_cross_track) - abs(inputs.cross_track)
    )
    reward = progress_reward + shaping_reward - config.time_cost
    if inputs.lap_completed and not (inputs.collision or off_road):
        reward += config.lap_bonus
    if inputs.collision or off_road:
        reward += config.crash_penalty
    return RewardResult(
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        progress_reward=float(progress_reward),
        shaping_reward=float(shaping_reward),
    )
