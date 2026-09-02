"""Generate the deterministic Gazebo Classic training circuit.

The centreline is the polar curve ``r(theta) = R + A cos(3 theta)``.  Since
``R > A > 0``, every polar angle has exactly one positive radius and the curve
cannot self-intersect.  The chosen amplitude is large enough that the signed
curvature changes sign, yielding alternating left and right turns.

The generated CSV is the source of truth shared by reward, evaluation, and UI
code.  Re-running this module is deterministic and overwrites only its explicit
output files.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


TAU = 2.0 * math.pi
DEFAULT_SAMPLES = 180
MEAN_RADIUS_M = 10.0
RADIAL_AMPLITUDE_M = 2.3
LOBE_COUNT = 3
ROAD_HALF_WIDTH_M = 1.35
WALL_THICKNESS_M = 0.12
WALL_HEIGHT_M = 0.35


@dataclass(frozen=True)
class TrackPoint:
    """One arc-length-indexed sample of the circuit centreline."""

    index: int
    x: float
    y: float
    s: float
    heading: float
    curvature: float


def _polar_geometry(theta: float) -> tuple[float, float, float, float, float]:
    radius = MEAN_RADIUS_M + RADIAL_AMPLITUDE_M * math.cos(LOBE_COUNT * theta)
    radius_d = -RADIAL_AMPLITUDE_M * LOBE_COUNT * math.sin(LOBE_COUNT * theta)
    radius_dd = -RADIAL_AMPLITUDE_M * LOBE_COUNT**2 * math.cos(LOBE_COUNT * theta)
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    dx = radius_d * math.cos(theta) - radius * math.sin(theta)
    dy = radius_d * math.sin(theta) + radius * math.cos(theta)
    curvature = (
        radius**2 + 2.0 * radius_d**2 - radius * radius_dd
    ) / (radius**2 + radius_d**2) ** 1.5
    return x, y, dx, dy, curvature


def generate_centerline(samples: int = DEFAULT_SAMPLES) -> tuple[TrackPoint, ...]:
    """Return an immutable, uniformly parameterised closed centreline."""

    if samples < 36:
        raise ValueError("samples must be at least 36 for safe wall geometry")

    raw = [_polar_geometry(TAU * index / samples) for index in range(samples)]
    cumulative = 0.0
    points: list[TrackPoint] = []
    for index, (x, y, dx, dy, curvature) in enumerate(raw):
        if index:
            cumulative += math.hypot(x - raw[index - 1][0], y - raw[index - 1][1])
        points.append(
            TrackPoint(
                index=index,
                x=x,
                y=y,
                s=cumulative,
                heading=math.atan2(dy, dx),
                curvature=curvature,
            )
        )
    return tuple(points)


def track_length(points: tuple[TrackPoint, ...]) -> float:
    """Return closed-loop polyline length in metres."""

    return points[-1].s + math.hypot(
        points[0].x - points[-1].x, points[0].y - points[-1].y
    )


def _orientation(a: TrackPoint, b: TrackPoint, c: TrackPoint) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _segments_cross(
    a: TrackPoint, b: TrackPoint, c: TrackPoint, d: TrackPoint
) -> bool:
    return (
        _orientation(a, b, c) * _orientation(a, b, d) < 0.0
        and _orientation(c, d, a) * _orientation(c, d, b) < 0.0
    )


def validate_centerline(points: tuple[TrackPoint, ...]) -> None:
    """Raise ``ValueError`` unless the loop is simple and turns both ways."""

    if len(points) < 4:
        raise ValueError("centreline needs at least four points")
    signs = {math.copysign(1.0, point.curvature) for point in points}
    if signs != {-1.0, 1.0}:
        raise ValueError("centreline must contain positive and negative curvature")

    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count}:
                continue
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            if _segments_cross(a, b, c, d):
                raise ValueError(f"centreline segments {first} and {second} cross")


def write_centerline_csv(points: tuple[TrackPoint, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("index", "x_m", "y_m", "s_m", "heading_rad", "curvature_1pm"))
        for point in points:
            writer.writerow(
                (
                    point.index,
                    f"{point.x:.9f}",
                    f"{point.y:.9f}",
                    f"{point.s:.9f}",
                    f"{point.heading:.9f}",
                    f"{point.curvature:.9f}",
                )
            )


def _box_link(
    name: str,
    x: float,
    y: float,
    z: float,
    yaw: float,
    length: float,
    width: float,
    height: float,
    material: str,
    collision: bool,
) -> str:
    collision_xml = ""
    if collision:
        collision_xml = f"""
        <collision name='collision'>
          <geometry><box><size>{length:.6f} {width:.6f} {height:.6f}</size></box></geometry>
          <surface><contact><ode><kp>1000000</kp><kd>10</kd></ode></contact></surface>
        </collision>"""
    return f"""
      <link name='{name}'>
        <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 {yaw:.9f}</pose>{collision_xml}
        <visual name='visual'>
          <geometry><box><size>{length:.6f} {width:.6f} {height:.6f}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{material}</name></script></material>
        </visual>
      </link>"""


def _track_model(points: tuple[TrackPoint, ...]) -> str:
    links: list[str] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        dx, dy = end.x - start.x, end.y - start.y
        length = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        midpoint_x, midpoint_y = (start.x + end.x) / 2.0, (start.y + end.y) / 2.0
        normal_x, normal_y = -dy / length, dx / length
        links.append(
            _box_link(
                f"road_{index:03d}", midpoint_x, midpoint_y, 0.003, yaw,
                length + 0.04, 2.0 * ROAD_HALF_WIDTH_M, 0.006,
                "Gazebo/Black", False,
            )
        )
        for side_name, side_sign, material in (
            ("inner", -1.0, "Gazebo/White"),
            ("outer", 1.0, "Gazebo/Red"),
        ):
            offset = side_sign * (ROAD_HALF_WIDTH_M + WALL_THICKNESS_M / 2.0)
            links.append(
                _box_link(
                    f"{side_name}_wall_{index:03d}",
                    midpoint_x + normal_x * offset,
                    midpoint_y + normal_y * offset,
                    WALL_HEIGHT_M / 2.0,
                    yaw,
                    length + 0.05,
                    WALL_THICKNESS_M,
                    WALL_HEIGHT_M,
                    material,
                    True,
                )
            )
    return "\n".join(("    <model name='track'>", "      <static>true</static>", *links, "    </model>"))


def _car_model(start: TrackPoint) -> str:
    # Wheel links are rolled -pi/2. Gazebo expresses each joint axis in the
    # child-link frame, hence local +Z (not model +Y) is the rolling axle.
    return f"""    <model name='rl_car'>
      <pose>{start.x:.6f} {start.y:.6f} 0.140000 0 0 {start.heading:.9f}</pose>
      <link name='base_link'>
        <inertial><mass>4.0</mass><inertia><ixx>0.050</ixx><iyy>0.110</iyy><izz>0.140</izz></inertia></inertial>
        <collision name='chassis_collision'><pose>0.015 0 0 0 0 0</pose><geometry><box><size>0.55 0.34 0.12</size></box></geometry></collision>
        <visual name='chassis_visual'><pose>0.015 0 0 0 0 0</pose><geometry><box><size>0.55 0.34 0.12</size></box></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Blue</name></script></material></visual>
        <sensor name='bumper' type='contact'>
          <always_on>true</always_on><update_rate>50</update_rate>
          <contact><collision>chassis_collision</collision></contact>
          <plugin name='bumper_ros' filename='libgazebo_ros_bumper.so'><ros><remapping>bumper_states:=contacts</remapping></ros><frame_name>base_link</frame_name></plugin>
        </sensor>
      </link>
      <link name='left_wheel'>
        <pose relative_to='base_link'>0 0.210 -0.050 -1.570796327 0 0</pose>
        <inertial><mass>0.25</mass><inertia><ixx>0.000590</ixx><iyy>0.000590</iyy><izz>0.001013</izz></inertia></inertial>
        <collision name='collision'><geometry><cylinder><radius>0.09</radius><length>0.04</length></cylinder></geometry><surface><friction><ode><mu>2.0</mu><mu2>2.0</mu2></ode></friction></surface></collision>
        <visual name='visual'><geometry><cylinder><radius>0.09</radius><length>0.04</length></cylinder></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Black</name></script></material></visual>
      </link>
      <link name='right_wheel'>
        <pose relative_to='base_link'>0 -0.210 -0.050 -1.570796327 0 0</pose>
        <inertial><mass>0.25</mass><inertia><ixx>0.000590</ixx><iyy>0.000590</iyy><izz>0.001013</izz></inertia></inertial>
        <collision name='collision'><geometry><cylinder><radius>0.09</radius><length>0.04</length></cylinder></geometry><surface><friction><ode><mu>2.0</mu><mu2>2.0</mu2></ode></friction></surface></collision>
        <visual name='visual'><geometry><cylinder><radius>0.09</radius><length>0.04</length></cylinder></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Black</name></script></material></visual>
      </link>
      <joint name='left_wheel_joint' type='revolute'><parent>base_link</parent><child>left_wheel</child><axis><xyz>0 0 1</xyz><limit><lower>-1.79769e+308</lower><upper>1.79769e+308</upper></limit></axis></joint>
      <joint name='right_wheel_joint' type='revolute'><parent>base_link</parent><child>right_wheel</child><axis><xyz>0 0 1</xyz><limit><lower>-1.79769e+308</lower><upper>1.79769e+308</upper></limit></axis></joint>
      <link name='caster'><pose relative_to='base_link'>-0.22 0 -0.100 0 0 0</pose><inertial><mass>0.08</mass><inertia><ixx>0.000051</ixx><iyy>0.000051</iyy><izz>0.000051</izz></inertia></inertial><collision name='collision'><geometry><sphere><radius>0.04</radius></sphere></geometry><surface><friction><ode><mu>0.05</mu><mu2>0.05</mu2></ode></friction></surface></collision><visual name='visual'><geometry><sphere><radius>0.04</radius></sphere></geometry></visual></link>
      <joint name='caster_joint' type='ball'><parent>base_link</parent><child>caster</child></joint>
      <link name='lidar_link'>
        <pose relative_to='base_link'>0.22 0 0.13 0 0 0</pose>
        <inertial><mass>0.02</mass><inertia><ixx>0.00001</ixx><iyy>0.00001</iyy><izz>0.00001</izz></inertia></inertial>
        <visual name='visual'><geometry><cylinder><radius>0.035</radius><length>0.04</length></cylinder></geometry></visual>
        <sensor name='lidar' type='ray'>
          <pose>0 0 0.03 0 0 0</pose><always_on>true</always_on><visualize>true</visualize><update_rate>20</update_rate>
          <ray><scan><horizontal><samples>61</samples><resolution>1</resolution><min_angle>-2.094395102</min_angle><max_angle>2.094395102</max_angle></horizontal></scan><range><min>0.08</min><max>8.0</max><resolution>0.01</resolution></range><noise><type>gaussian</type><mean>0</mean><stddev>0.005</stddev></noise></ray>
          <plugin name='lidar_ros' filename='libgazebo_ros_ray_sensor.so'><ros><remapping>~/out:=scan</remapping></ros><output_type>sensor_msgs/LaserScan</output_type><frame_name>lidar_link</frame_name></plugin>
        </sensor>
      </link>
      <joint name='lidar_joint' type='fixed'><parent>base_link</parent><child>lidar_link</child></joint>
      <link name='camera_link'>
        <pose relative_to='base_link'>0.23 0 0.17 0 0 0</pose>
        <inertial><mass>0.02</mass><inertia><ixx>0.00001</ixx><iyy>0.00001</iyy><izz>0.00001</izz></inertia></inertial>
        <sensor name='front_camera' type='camera'>
          <always_on>true</always_on><update_rate>10</update_rate>
          <camera><horizontal_fov>1.3962634</horizontal_fov><image><width>160</width><height>120</height><format>R8G8B8</format></image><clip><near>0.05</near><far>30</far></clip></camera>
          <plugin name='camera_ros' filename='libgazebo_ros_camera.so'><camera_name>camera</camera_name><frame_name>camera_link</frame_name></plugin>
        </sensor>
      </link>
      <joint name='camera_joint' type='fixed'><parent>base_link</parent><child>camera_link</child></joint>
      <plugin name='diff_drive' filename='libgazebo_ros_diff_drive.so'>
        <ros><remapping>cmd_vel:=cmd_vel</remapping><remapping>odom:=odom</remapping></ros>
        <update_rate>50</update_rate><left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.42</wheel_separation><wheel_diameter>0.18</wheel_diameter><max_wheel_torque>8.0</max_wheel_torque><max_wheel_acceleration>12.0</max_wheel_acceleration>
        <publish_odom>true</publish_odom><publish_odom_tf>true</publish_odom_tf><publish_wheel_tf>true</publish_wheel_tf><odometry_frame>odom</odometry_frame><robot_base_frame>base_link</robot_base_frame>
      </plugin>
    </model>"""


def build_world(points: tuple[TrackPoint, ...]) -> str:
    """Build a complete SDF 1.7 Gazebo Classic world."""

    validate_centerline(points)
    return f"""<?xml version='1.0'?>
<sdf version='1.7'>
  <world name='rl_track'>
    <physics name='fast_ode' type='ode'><max_step_size>0.005</max_step_size><real_time_update_rate>1000</real_time_update_rate><real_time_factor>5</real_time_factor><ode><solver><type>quick</type><iters>50</iters></solver></ode></physics>
    <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.75 0.82 0.90 1</background><shadows>false</shadows></scene>
    <spherical_coordinates><surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation><latitude_deg>0</latitude_deg><longitude_deg>0</longitude_deg><elevation>0</elevation><heading_deg>0</heading_deg></spherical_coordinates>
    <include><uri>model://sun</uri></include>
    <model name='ground'><static>true</static><link name='link'><collision name='collision'><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision><visual name='visual'><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Grass</name></script></material></visual></link></model>
    <plugin name='gazebo_ros_state' filename='libgazebo_ros_state.so'><ros><namespace>/gazebo</namespace></ros><update_rate>50</update_rate></plugin>
{_track_model(points)}
{_car_model(points[0])}
  </world>
</sdf>
"""


def write_world(points: tuple[TrackPoint, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    world = build_world(points)
    ElementTree.fromstring(world)  # Refuse to emit malformed XML.
    path.write_text(world, encoding="utf-8")


def generate_assets(output_directory: Path, samples: int = DEFAULT_SAMPLES) -> tuple[Path, Path]:
    """Generate and return ``(world_path, csv_path)``."""

    points = generate_centerline(samples)
    validate_centerline(points)
    world_path = output_directory / "alternating_track.world"
    csv_path = output_directory / "centerline.csv"
    write_world(points, world_path)
    write_centerline_csv(points, csv_path)
    return world_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="asset output directory")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    arguments = parser.parse_args()
    world, csv_path = generate_assets(arguments.output, arguments.samples)
    print(f"generated {world}")
    print(f"generated {csv_path}")


if __name__ == "__main__":
    main()
