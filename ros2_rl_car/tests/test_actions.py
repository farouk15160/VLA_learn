from __future__ import annotations

import pytest

from ros2_rl_car.core.constants import ACTION_ANGULAR_Z, LEFT, RIGHT, STRAIGHT
from ros2_rl_car.core.control import action_from_angular_z, angular_z_from_action


@pytest.mark.parametrize("action", [LEFT, STRAIGHT, RIGHT])
def test_action_sign_round_trip(action: int) -> None:
    assert action_from_angular_z(angular_z_from_action(action)) == action


def test_ros_positive_angular_z_is_left_everywhere() -> None:
    assert ACTION_ANGULAR_Z[LEFT] > 0.0
    assert ACTION_ANGULAR_Z[STRAIGHT] == 0.0
    assert ACTION_ANGULAR_Z[RIGHT] < 0.0


def test_reference_controller_small_yaw_request_keeps_its_direction() -> None:
    assert action_from_angular_z(0.1) == LEFT
    assert action_from_angular_z(-0.1) == RIGHT
