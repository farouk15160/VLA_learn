"""Core conversions between policy actions and ROS velocity commands."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

from .constants import (
    ACTION_ANGULAR_Z,
    ACTION_LINEAR_X,
    ACTION_YAW_DEADBAND,
    LEFT,
    NUMBER_OF_ACTIONS,
    RIGHT,
    STRAIGHT,
)


@dataclass(frozen=True, slots=True)
class VelocityCommand:
    """A transport-independent equivalent of the relevant ``Twist`` fields."""

    linear_x: float
    angular_z: float


def angular_z_from_action(action: int) -> float:
    """Return angular velocity for a discrete action.

    Positive output always means a left turn, following ROS REP-103.
    """

    if not isinstance(action, Integral) or isinstance(action, bool):
        raise TypeError("action must be an integer")
    action_index = int(action)
    if not 0 <= action_index < NUMBER_OF_ACTIONS:
        raise ValueError(f"action must be in [0, {NUMBER_OF_ACTIONS - 1}]")
    return ACTION_ANGULAR_Z[action_index]


def action_from_angular_z(angular_z: float) -> int:
    """Quantize angular velocity to the closest configured policy action."""

    value = float(angular_z)
    if not math.isfinite(value):
        raise ValueError("angular_z must be finite")
    if value > ACTION_YAW_DEADBAND:
        return LEFT
    if value < -ACTION_YAW_DEADBAND:
        return RIGHT
    return STRAIGHT


def command_from_action(action: int, linear_x: float = ACTION_LINEAR_X) -> VelocityCommand:
    speed = float(linear_x)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("linear_x must be finite and non-negative")
    return VelocityCommand(speed, angular_z_from_action(action))
