from __future__ import annotations

import numpy as np

from ros2_rl_car.core.track import ParametricTrack, signed_curvature


def test_generated_track_has_left_and_right_turns() -> None:
    track = ParametricTrack.generate(samples=800)
    curvature = signed_curvature(track.points)
    assert np.min(curvature) < -0.01
    assert np.max(curvature) > 0.01


def test_projection_returns_cross_track_and_heading_without_pose_in_policy() -> None:
    track = ParametricTrack.generate(samples=800)
    projected = track.project(tuple(track.points[10] + np.array([0.1, 0.1])))
    assert 0.0 <= projected.progress < track.length
    assert abs(projected.cross_track_error) < 0.5
