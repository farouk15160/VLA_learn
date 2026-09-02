from __future__ import annotations

import csv
import math
from xml.etree import ElementTree

import pytest

from ros2_rl_car.sim.world import (
    DEFAULT_SAMPLES,
    build_world,
    generate_assets,
    generate_centerline,
    track_length,
    validate_centerline,
)
from ros2_rl_car.core.track import ParametricTrack


def test_circuit_is_closed_simple_and_turns_both_ways() -> None:
    points = generate_centerline()

    validate_centerline(points)

    assert len(points) == DEFAULT_SAMPLES
    assert min(point.curvature for point in points) < -0.1
    assert max(point.curvature for point in points) > 0.1
    assert 65.0 < track_length(points) < 75.0
    seam = math.hypot(points[0].x - points[-1].x, points[0].y - points[-1].y)
    assert seam < 0.5


def test_generation_is_byte_for_byte_deterministic(tmp_path) -> None:
    first_world, first_csv = generate_assets(tmp_path / "first")
    second_world, second_csv = generate_assets(tmp_path / "second")

    assert first_world.read_bytes() == second_world.read_bytes()
    assert first_csv.read_bytes() == second_csv.read_bytes()


def test_csv_has_monotonic_arc_length_and_expected_schema(tmp_path) -> None:
    _, csv_path = generate_assets(tmp_path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert tuple(rows[0]) == (
        "index",
        "x_m",
        "y_m",
        "s_m",
        "heading_rad",
        "curvature_1pm",
    )
    distances = [float(row["s_m"]) for row in rows]
    assert distances == sorted(distances)
    assert len(set(distances)) == len(distances)
    loaded = ParametricTrack.load_csv(csv_path)
    assert len(loaded.points) == DEFAULT_SAMPLES


def test_world_contains_required_ros_interfaces_and_child_frame_axes() -> None:
    root = ElementTree.fromstring(build_world(generate_centerline()))
    plugins = {plugin.attrib["filename"]: plugin for plugin in root.iter("plugin")}

    assert "libgazebo_ros_diff_drive.so" in plugins
    assert "libgazebo_ros_ray_sensor.so" in plugins
    assert "libgazebo_ros_camera.so" in plugins
    assert "libgazebo_ros_bumper.so" in plugins
    assert "libgazebo_ros_state.so" in plugins
    assert plugins["libgazebo_ros_ray_sensor.so"].findtext("output_type") == (
        "sensor_msgs/LaserScan"
    )
    physics = root.find(".//physics")
    assert physics is not None
    assert physics.findtext("real_time_update_rate") == "1000"
    assert plugins["libgazebo_ros_bumper.so"].findtext("ros/remapping") == (
        "bumper_states:=contacts"
    )
    assert plugins["libgazebo_ros_state.so"].findtext("ros/namespace") == "/gazebo"

    joints = {
        joint.attrib["name"]: joint for joint in root.findall(".//joint")
    }
    for name in ("left_wheel_joint", "right_wheel_joint"):
        axis = joints[name].find("axis/xyz")
        assert axis is not None
        assert axis.attrib == {}  # default is the rotated child-link frame
        assert axis.text == "0 0 1"


def test_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 36"):
        generate_centerline(20)
