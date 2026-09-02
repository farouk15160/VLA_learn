"""Core shared control conventions.

ROS uses a right-handed frame: +x is forward and positive angular velocity
about +z turns left.  This module is the single source of truth for that sign
convention.
"""

from __future__ import annotations

LEFT: int = 0
STRAIGHT: int = 1
RIGHT: int = 2

ACTION_NAMES: tuple[str, ...] = ("left", "straight", "right")
ACTION_ANGULAR_Z: tuple[float, ...] = (0.6, 0.0, -0.6)
ACTION_LINEAR_X: float = 0.8
ACTION_YAW_DEADBAND: float = 0.05
NUMBER_OF_ACTIONS: int = len(ACTION_ANGULAR_Z)

# Explicit name for code which needs to transform a model steering value to ROS.
ROS_LEFT_TURN_SIGN: float = 1.0
