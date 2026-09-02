# Vehicle physics and Gazebo integration

Gazebo Classic 11.10.2 is used because ROS 2 Humble's native diff-drive, ray,
camera, bumper, state, and factory plugins are installed locally. The default is
`gzserver` only. Gazebo's GUI is opt-in and independent of the training
dashboard.

For wheel radius `r`, track width `b`, left wheel rate `omega_l`, and right wheel
rate `omega_r`, ideal differential-drive kinematics are

```text
v = r(omega_r + omega_l)/2
yaw_rate = r(omega_r - omega_l)/b
```

Equivalently, desired body velocity maps to wheel rates

```text
omega_l = (v - yaw_rate*b/2)/r
omega_r = (v + yaw_rate*b/2)/r
```

Positive yaw therefore requires the right wheel to move faster and turns left.
The wheel cylinder is rotated so its axle lies across the vehicle. SDF joint
axes are expressed in the child link frame; the model follows the working
Gazebo ROS diff-drive demo convention rather than copying a model-frame axis.

The environment owns one ROS node and one executor. It subscribes to odometry,
LaserScan, contact state, and `/clock`, and publishes `/cmd_vel`. Sensor-data QoS
is BEST_EFFORT for lidar/camera compatibility. Reset publishes zero velocity,
calls `/gazebo/set_entity_state` with zero twist, clears episode bookkeeping,
and waits for fresh post-reset odometry and scan sequence numbers.

All control timing follows `/clock`, not wall time. Evaluators reject zero or
stale sensor frames, so a dead simulator cannot be scored as a perfect parked
car. The process wrapper refuses to start if any `gzserver` already exists and
offers an explicit stop command for the process it owns.

The camera publishes RGB images for inspection. The learner never imports
`cv_bridge`; this avoids the common NumPy 2 / NumPy 1 ABI failure. If preview is
added later, image rows must be decoded using the message's byte `step`, which
is not guaranteed to equal `width*3`.
