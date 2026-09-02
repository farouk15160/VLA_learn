# Live dashboard

`telemetry.py` defines immutable snapshots passed from the trainer to the UI.
`ui.py` renders a scrollable Tk/Matplotlib window with the track, trajectory,
lidar rays, reward and length histories, success rate, PPO losses, entropy,
gradient norm, KL/clip fraction, action probabilities, value estimate,
hyperparameters, and timestamped events.

Pause/resume, immediate checkpoint save, and greedy-watch controls communicate
through thread-safe events. The UI never calls ROS from Tk's main thread.
Training remains fully headless unless `train --gui` is requested.
