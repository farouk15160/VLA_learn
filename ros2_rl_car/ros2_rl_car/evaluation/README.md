# Evaluation package

`controllers.py` defines sensor-blind constant policies and the privileged
pure-pursuit upper reference. `metrics.py` accumulates frame counts, cross-track
error, laps, termination causes, and direction accuracy. `runner.py` evaluates
those references and an optional greedy PPO checkpoint with fixed seeds.

An episode with zero odometry, lidar, or control frames raises an error. This
prevents a dead simulator from being reported as a car that stayed on the road.
The evaluator emits a JSON evidence packet; retained packets and checkpoints
are described in [`../../outputs/README.md`](../../outputs/README.md).
