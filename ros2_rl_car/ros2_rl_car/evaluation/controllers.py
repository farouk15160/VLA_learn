"""Reference and null controllers establishing evaluation bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.constants import STRAIGHT
from ..core.control import action_from_angular_z
from ..core.track import ParametricTrack


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True, slots=True)
class ConstantController:
    """Sensor-ignoring null baseline; straight is intentionally the default."""

    action: int = STRAIGHT

    def select_action(self, _observation: np.ndarray | None = None) -> int:
        return self.action


@dataclass(frozen=True, slots=True)
class PurePursuitController:
    """Ground-truth-pose upper reference, not an input to the learned policy."""

    track: ParametricTrack
    lookahead_distance: float = 1.2
    wheelbase: float = 0.32
    max_angular_z: float = 1.2

    def __post_init__(self) -> None:
        if self.lookahead_distance <= 0.0 or self.wheelbase <= 0.0 or self.max_angular_z <= 0.0:
            raise ValueError("controller distances and rate must be positive")

    def angular_z(self, x: float, y: float, yaw: float, speed: float = 0.8) -> float:
        projection = self.track.project((x, y))
        target = self.track.point_at(projection.progress + self.lookahead_distance)
        target_heading = math.atan2(target[1] - y, target[0] - x)
        alpha = wrap_angle(target_heading - yaw)
        curvature = 2.0 * math.sin(alpha) / self.lookahead_distance
        # For cmd_vel control, yaw rate is v*kappa. wheelbase remains documented
        # for converting curvature to the equivalent Ackermann steering angle.
        return float(np.clip(max(speed, 0.0) * curvature, -self.max_angular_z, self.max_angular_z))

    def steering_angle(self, x: float, y: float, yaw: float) -> float:
        projection = self.track.project((x, y))
        target = self.track.point_at(projection.progress + self.lookahead_distance)
        alpha = wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw)
        return math.atan2(2.0 * self.wheelbase * math.sin(alpha), self.lookahead_distance)

    def select_action(self, x: float, y: float, yaw: float, speed: float = 0.8) -> int:
        return action_from_angular_z(self.angular_z(x, y, yaw, speed))
