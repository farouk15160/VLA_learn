# Architecture

```text
Gazebo Classic
  /scan, /odom, /contacts, /camera/image_raw, /clock
       |
       v
RosBridge (one rclpy node + one executor)
       |
       v
GazeboCarEnv ---- Track projection ---- Reward/termination
       |
       v
PPO trainer ---- immutable telemetry snapshots ---- Tk/Matplotlib dashboard
       |
       +---- atomic checkpoints
       +---- JSON/CSV evaluation artifacts
```

The package is feature-oriented: `core/` owns track geometry and the MDP,
`learning/` owns PPO and training, `sim/` owns ROS/Gazebo integration,
`evaluation/` owns controllers and scoring, and `dashboard/` owns telemetry and
the live UI. Pure math remains importable without ROS or Gazebo, so it is
covered by fast pytest tests. The UI consumes immutable snapshots through a
bounded queue and never touches ROS callbacks from Tk's main thread.

The scorer provides three policy families through one interface: a sweep of
constant sensor-blind actions (required negative baseline), pure pursuit using
privileged pose (upper reference), and a loaded PPO checkpoint. It records
success rate, mean/max absolute cross-track error, laps, collision/off-road
rate, frame counts, and direction accuracy.

`cli.py` is the composition root for all commands. Feature packages do not
depend on it except for their tiny console-script adapters.
