"""Start the RL circuit in Gazebo Classic 11 (headless by default)."""

from __future__ import annotations

import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _refuse_orphaned_gazebo(_context):
    """Fail early instead of mixing topics from two plausible simulators."""

    result = subprocess.run(
        ("pgrep", "-x", "gzserver"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        pids = ", ".join(result.stdout.split())
        raise RuntimeError(
            f"gzserver is already running (PID(s): {pids}). "
            "Stop the owned simulator with: ./scripts/rl_car stop"
        )
    return []


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("ros2_rl_car")
    gazebo_share = get_package_share_directory("gazebo_ros")
    world = f"{package_share}/assets/worlds/alternating_track.world"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="Set true to open gzclient and watch training.",
            ),
            DeclareLaunchArgument(
                "verbose",
                default_value="false",
                description="Enable verbose Gazebo server output.",
            ),
            DeclareLaunchArgument(
                "seed",
                default_value="1",
                description="Gazebo physics random seed.",
            ),
            OpaqueFunction(function=_refuse_orphaned_gazebo),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    f"{gazebo_share}/launch/gazebo.launch.py"
                ),
                launch_arguments={
                    "world": world,
                    "gui": LaunchConfiguration("gui"),
                    "verbose": LaunchConfiguration("verbose"),
                    "seed": LaunchConfiguration("seed"),
                    "pause": "false",
                }.items(),
            ),
        ]
    )
