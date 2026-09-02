"""Single-node ROS transport shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Condition, Lock, Thread
import time
from typing import Any

import numpy as np


class RosUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SensorSnapshot:
    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    pose: tuple[float, float, float]
    linear_speed: float
    sim_time: float
    scan_frames: int
    odom_frames: int
    camera_frames: int
    collision: bool


def _yaw(quaternion: Any) -> float:
    siny = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy = 1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2)
    return math.atan2(siny, cosy)


class RosBridge:
    """One rclpy Node in one executor, spun by one background thread.

    Laser and camera subscriptions deliberately use BEST_EFFORT QoS because
    Gazebo sensor publishers commonly use that policy.
    """

    def __init__(
        self,
        *,
        model_name: str = "rl_car",
        scan_topic: str = "/scan",
        camera_topic: str = "/camera/image_raw",
        odom_topic: str = "/odom",
        cmd_topic: str = "/cmd_vel",
    ) -> None:
        try:
            import rclpy
            from builtin_interfaces.msg import Time
            from gazebo_msgs.msg import ContactsState, EntityState
            from gazebo_msgs.srv import SetEntityState
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from rosgraph_msgs.msg import Clock
            from sensor_msgs.msg import Image, LaserScan
        except ImportError as exc:
            raise RosUnavailable(
                "ROS Python packages unavailable; source /opt/ros/humble/setup.bash "
                "before running the Python 3.10 virtualenv"
            ) from exc

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._Time = Time
        self._EntityState = EntityState
        self._SetEntityState = SetEntityState
        self._Twist = Twist
        self._node = rclpy.create_node("rl_car_runtime")
        self._node.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time", value=True)]
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._model_name = model_name
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._ranges = np.empty(0, dtype=np.float32)
        self._angle_min = 0.0
        self._angle_increment = 0.0
        self._pose = (0.0, 0.0, 0.0)
        self._speed = 0.0
        self._sim_time = 0.0
        self._scan_frames = self._odom_frames = self._camera_frames = 0
        self._collision = False
        self._closed = False

        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._cmd = self._node.create_publisher(Twist, cmd_topic, 10)
        self._node.create_subscription(LaserScan, scan_topic, self._on_scan, sensor_qos)
        self._node.create_subscription(Image, camera_topic, self._on_camera, sensor_qos)
        self._node.create_subscription(Odometry, odom_topic, self._on_odom, sensor_qos)
        self._node.create_subscription(ContactsState, "/contacts", self._on_collision, sensor_qos)
        self._node.create_subscription(Clock, "/clock", self._on_clock, sensor_qos)
        self._reset = self._node.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )
        self._thread = Thread(target=self._spin, name="rclpy-executor", daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        while not self._closed and self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_scan(self, msg: Any) -> None:
        with self._condition:
            self._ranges = np.asarray(msg.ranges, dtype=np.float32)
            self._angle_min = float(msg.angle_min)
            self._angle_increment = float(msg.angle_increment)
            self._scan_frames += 1
            self._condition.notify_all()

    def _on_odom(self, msg: Any) -> None:
        pose = msg.pose.pose
        twist = msg.twist.twist
        with self._condition:
            self._pose = (float(pose.position.x), float(pose.position.y), _yaw(pose.orientation))
            self._speed = float(twist.linear.x)
            self._odom_frames += 1
            self._condition.notify_all()

    def _on_camera(self, _msg: Any) -> None:
        # Pixels are published for watching, but intentionally not decoded or learned from.
        with self._condition:
            self._camera_frames += 1
            self._condition.notify_all()

    def _on_collision(self, msg: Any) -> None:
        # The chassis bumper can briefly touch the ground as the three-point
        # differential-drive body pitches. Only generated track walls are task
        # collisions; ground settling must not end an episode.
        wall_contact = any(
            "wall_" in state.collision1_name or "wall_" in state.collision2_name
            for state in msg.states
        )
        with self._condition:
            self._collision = self._collision or wall_contact
            self._condition.notify_all()

    def _on_clock(self, msg: Any) -> None:
        with self._condition:
            self._sim_time = float(msg.clock.sec) + float(msg.clock.nanosec) * 1e-9
            self._condition.notify_all()

    def snapshot(self) -> SensorSnapshot:
        with self._lock:
            return SensorSnapshot(
                ranges=self._ranges.copy(),
                angle_min=self._angle_min,
                angle_increment=self._angle_increment,
                pose=self._pose,
                linear_speed=self._speed,
                sim_time=self._sim_time,
                scan_frames=self._scan_frames,
                odom_frames=self._odom_frames,
                camera_frames=self._camera_frames,
                collision=self._collision,
            )

    def wait_for_fresh(self, previous: SensorSnapshot | None, timeout: float = 5.0) -> SensorSnapshot:
        old_scan = -1 if previous is None else previous.scan_frames
        old_odom = -1 if previous is None else previous.odom_frames
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._scan_frames <= old_scan or self._odom_frames <= old_odom:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "no fresh lidar+odometry frame; check Gazebo and BEST_EFFORT QoS"
                    )
                self._condition.wait(remaining)
        return self.snapshot()

    def wait_for_camera(self, timeout: float = 5.0) -> SensorSnapshot:
        """Wait for an actual BEST_EFFORT camera frame, not just topic discovery."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while self._camera_frames < 1:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "no camera frame; check /camera/image_raw and BEST_EFFORT QoS"
                    )
                self._condition.wait(remaining)
        return self.snapshot()

    def command(self, linear_x: float, angular_z: float) -> None:
        message = self._Twist()
        message.linear.x = float(linear_x)
        message.angular.z = float(angular_z)
        self._cmd.publish(message)

    def reset_pose(
        self, pose: tuple[float, float, float], *, timeout: float = 5.0
    ) -> None:
        if not self._reset.wait_for_service(timeout_sec=timeout):
            raise TimeoutError("/gazebo/set_entity_state service unavailable")
        state = self._EntityState()
        state.name = self._model_name
        state.reference_frame = "world"
        state.pose.position.x, state.pose.position.y = map(float, pose[:2])
        state.pose.orientation.z = math.sin(float(pose[2]) / 2.0)
        state.pose.orientation.w = math.cos(float(pose[2]) / 2.0)
        request = self._SetEntityState.Request()
        request.state = state
        future = self._reset.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise TimeoutError("SetEntityState call timed out")
        response = future.result()
        if response is None or not response.success:
            status = "no response" if response is None else response.status_message
            raise RuntimeError(f"SetEntityState failed: {status}")
        self.command(0.0, 0.0)
        with self._lock:
            self._collision = False

    def close(self) -> None:
        if self._closed:
            return
        self.command(0.0, 0.0)
        self._closed = True
        self._thread.join(timeout=2.0)
        self._executor.remove_node(self._node)
        self._node.destroy_node()

    def __enter__(self) -> "RosBridge":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
