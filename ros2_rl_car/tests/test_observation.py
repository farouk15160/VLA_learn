from __future__ import annotations

import math

import numpy as np
import pytest

from ros2_rl_car.core.mdp import ObservationConfig, build_observation


def test_observation_is_normalized_finite_and_has_no_absolute_pose() -> None:
    config = ObservationConfig(lidar_beams=5, lidar_max_range=10.0, max_speed=2.0)
    observation = build_observation(
        ranges=[float("inf"), -1.0, 2.5, float("nan"), 12.0],
        speed=1.0,
        heading_error=math.pi / 2,
        cross_track_error=-0.75,
        road_half_width=1.5,
        config=config,
    )
    assert observation.shape == (9,)
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)
    assert np.allclose(observation[:5], [1.0, 0.0, 0.25, 1.0, 1.0])
    assert np.allclose(observation[5:], [0.5, 1.0, 0.0, -0.5], atol=1e-7)


def test_observation_resamples_scan_to_fixed_width() -> None:
    observation = build_observation(
        ranges=np.linspace(0.0, 10.0, 101),
        speed=0.0,
        heading_error=0.0,
        cross_track_error=0.0,
        road_half_width=1.5,
        config=ObservationConfig(lidar_beams=11, lidar_max_range=10.0),
    )
    assert observation.shape == (15,)
    assert np.allclose(observation[:11], np.linspace(0.0, 1.0, 11), atol=0.011)


def test_empty_scan_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_observation([], 0.0, 0.0, 0.0, 1.5, ObservationConfig())
