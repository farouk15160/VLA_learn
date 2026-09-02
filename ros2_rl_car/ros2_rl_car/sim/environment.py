"""Synchronous MDP environment around the single ROS bridge."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from ..core.control import command_from_action
from ..core.mdp import (
    ObservationConfig,
    RewardConfig,
    RewardInput,
    build_observation,
    reward_step,
)
from ..core.track import ParametricTrack, TrackProjection, wrapped_progress_delta
from .bridge import RosBridge, SensorSnapshot


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


@dataclass(frozen=True)
class GazeboEnvConfig:
    control_period: float = 0.10
    max_steps: int = 1200
    road_half_width: float = 1.5
    sensor_timeout: float = 5.0
    max_progress_per_step: float = 0.5


class GazeboDrivingEnv:
    """Discrete steering environment; forward speed is held constant.

    Fixed throttle makes PPO's action space small enough to train reliably on
    CPU. The only action is LEFT/STRAIGHT/RIGHT, mapped centrally by control.py.
    Absolute pose is used only by the environment for reward/scoring and never
    appears in the policy observation.
    """

    def __init__(
        self,
        bridge: RosBridge,
        track: ParametricTrack,
        *,
        config: GazeboEnvConfig = GazeboEnvConfig(),
        observation_config: ObservationConfig = ObservationConfig(),
        reward_config: RewardConfig = RewardConfig(),
    ) -> None:
        self.bridge = bridge
        self.track = track
        self.config = config
        self.observation_config = observation_config
        self.reward_config = replace(
            reward_config,
            road_half_width=config.road_half_width,
            track_length=track.length,
        )
        self._snapshot: SensorSnapshot | None = None
        self._projection: TrackProjection | None = None
        self._step_count = 0
        self._travelled = 0.0
        self._trajectory: list[tuple[float, float]] = []

    @property
    def observation_size(self) -> int:
        return self.observation_config.size

    @property
    def trajectory(self) -> np.ndarray:
        return np.asarray(self._trajectory, dtype=float).reshape(-1, 2)

    @property
    def travelled_progress(self) -> float:
        return self._travelled

    def reset(self) -> tuple[np.ndarray, dict[str, object]]:
        point = self.track.points[0]
        tangent = self.track.segment_vectors[0]
        heading = math.atan2(float(tangent[1]), float(tangent[0]))
        before = self.bridge.snapshot()
        self.bridge.reset_pose((float(point[0]), float(point[1]), heading))
        snapshot = self.bridge.wait_for_fresh(before, self.config.sensor_timeout)
        projection = self.track.project(snapshot.pose[:2])
        self._snapshot = snapshot
        self._projection = projection
        self._step_count = 0
        self._travelled = 0.0
        self._trajectory = [snapshot.pose[:2]]
        return self._observation(snapshot, projection), self._info(snapshot, projection, False)

    def step(self, action: int) -> StepResult:
        if self._snapshot is None or self._projection is None:
            raise RuntimeError("reset() must be called before step()")
        previous_snapshot = self._snapshot
        previous_projection = self._projection
        command = command_from_action(action)
        self.bridge.command(command.linear_x, command.angular_z)
        snapshot = self._wait_control_period(previous_snapshot)
        projection = self.track.project(snapshot.pose[:2])
        progress_delta = wrapped_progress_delta(
            projection.progress, previous_projection.progress, self.track.length
        )
        # Teleport-like jumps cannot become reward. Normal driving is far below this bound.
        if abs(progress_delta) > self.config.max_progress_per_step:
            progress_delta = 0.0
        self._travelled += progress_delta
        self._step_count += 1
        scan = np.nan_to_num(
            snapshot.ranges,
            nan=self.observation_config.lidar_max_range,
            posinf=self.observation_config.lidar_max_range,
            neginf=0.0,
        )
        collision = bool(
            snapshot.collision
            or (scan.size and np.min(scan) <= self.reward_config.collision_range)
        )
        lap_completed = self._travelled >= self.track.length
        outcome = reward_step(
            RewardInput(
                progress_delta=progress_delta,
                previous_cross_track=previous_projection.cross_track_error,
                cross_track=projection.cross_track_error,
                collision=collision,
                lap_completed=lap_completed,
                step_count=self._step_count,
                max_steps=self.config.max_steps,
            ),
            self.reward_config,
        )
        terminated = bool(outcome.terminated or lap_completed)
        lap_success = bool(lap_completed and outcome.reason == "lap_complete")
        reason = outcome.reason
        if terminated or outcome.truncated:
            self.bridge.command(0.0, 0.0)
        self._snapshot = snapshot
        self._projection = projection
        self._trajectory = [*self._trajectory, snapshot.pose[:2]]
        return StepResult(
            observation=self._observation(snapshot, projection),
            reward=outcome.reward,
            terminated=terminated,
            truncated=outcome.truncated,
            info={
                **self._info(snapshot, projection, collision),
                "reason": reason,
                "lap_completed": lap_success,
            },
        )

    def _wait_control_period(self, previous: SensorSnapshot) -> SensorSnapshot:
        target = previous.sim_time + self.config.control_period
        snapshot = previous
        while snapshot.sim_time < target:
            snapshot = self.bridge.wait_for_fresh(snapshot, self.config.sensor_timeout)
            if snapshot.sim_time <= previous.sim_time and snapshot.scan_frames > previous.scan_frames + 5:
                raise RuntimeError("simulation clock is not advancing")
        return snapshot

    def _observation(self, snapshot: SensorSnapshot, projection: TrackProjection) -> np.ndarray:
        return build_observation(
            snapshot.ranges,
            snapshot.linear_speed,
            wrap_angle(snapshot.pose[2] - projection.heading),
            projection.cross_track_error,
            self.config.road_half_width,
            self.observation_config,
        )

    def _info(
        self, snapshot: SensorSnapshot, projection: TrackProjection, collision: bool
    ) -> dict[str, object]:
        return {
            "pose": snapshot.pose,
            "cross_track_error": projection.cross_track_error,
            "progress": projection.progress,
            "travelled_progress": self._travelled,
            "collision": collision,
            "sensor_frames": min(snapshot.scan_frames, snapshot.odom_frames),
            "ray_endpoints": self.ray_endpoints(snapshot),
        }

    @staticmethod
    def ray_endpoints(snapshot: SensorSnapshot) -> np.ndarray:
        if snapshot.ranges.size == 0:
            return np.empty((0, 2))
        angles = snapshot.pose[2] + snapshot.angle_min + np.arange(snapshot.ranges.size) * snapshot.angle_increment
        ranges = np.nan_to_num(snapshot.ranges, nan=0.0, posinf=10.0, neginf=0.0)
        return np.column_stack(
            (snapshot.pose[0] + ranges * np.cos(angles), snapshot.pose[1] + ranges * np.sin(angles))
        )

    def close(self) -> None:
        self.bridge.close()
