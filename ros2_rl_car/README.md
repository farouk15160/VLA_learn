# ROS 2 reinforcement-learning car

This folder is a self-contained ROS 2 Humble project in which a Gazebo car
learns by PPO trial and error. It does not use demonstrations or behavioral
cloning. The road has alternating signed curvature, so a constant steering
policy cannot solve it.

## Verified local stack and design choice

The target machine has ROS 2 Humble, Python 3.10.12, Gazebo Classic 11.10.2,
and the Humble `gazebo_ros` plugins. The existing Python 3.10 virtualenv reports
PyTorch 2.13.0 and `torch.cuda.is_available() == False`. Gazebo Classic was
chosen over gz/ros_gz because all required native ROS plugins and entity reset
services are installed and testable without a bridge.

The v1 observation is normalized lidar + speed + heading-error sine/cosine +
signed cross-track error. It contains no absolute position and no camera
pixels. The camera still publishes. This is a deliberate CPU tradeoff: the
small MLP can improve in a practical run, while pixel CNN RL on CPU can take
overnight and is much less reliable.

Documentation lives beside the code it explains: the [MDP and reward
math](ros2_rl_car/core/README.md), [PPO learner](ros2_rl_car/learning/README.md),
[vehicle physics and simulator](ros2_rl_car/sim/README.md), [evaluation
contract](ros2_rl_car/evaluation/README.md), [live dashboard](ros2_rl_car/dashboard/README.md),
and [package architecture](ros2_rl_car/README.md). Every operator-facing folder
also contains its own short README.

## Existing simulator/world survey

Existing projects were checked before generating a track:

- [F1TENTH Gym ROS](https://github.com/f1tenth/f1tenth_gym_ros) is a useful ROS
  2 bridge, but its documented native target is Ubuntu 20.04/ROS 2 Foxy and the
  vehicle is a 2-D Gym simulation rather than the installed Gazebo Classic
  physics/sensor stack.
- [AWS DeepRacer Simapp](https://github.com/aws-deepracer-community/deepracer-simapp)
  contains tracks and Gazebo assets, but its current container stack is ROS 2
  Jazzy, Gazebo Harmonic, Python 3.12, and TensorFlow, a large incompatible
  dependency surface for this Humble/Python 3.10/PyTorch task.
- [TurtleBot3 simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations)
  has an official Humble branch and Apache-2.0 license, but the package was not
  installed locally and its navigation worlds do not supply the required
  alternating-curvature racetrack plus shared centre-line CSV.
- The installed `gazebo_ros` demo worlds validate individual diff-drive, ray,
  and camera plugins, but provide no racetrack.

The deterministic three-lobed polar circuit was therefore generated locally.
It is non-self-intersecting, has positive and negative signed curvature, has no
downloaded model dependencies, and emits the exact CSV used by reward, UI, and
evaluation.

## Setup

Do not install Torch into `/usr/bin/python3` and do not pip-install `rclpy`.

```bash
cd /home/farouk/code/VLA_learn/ros2_rl_car
./scripts/setup_venv.sh
```

The wrapper sources ROS and adds Humble's Python 3.10 packages to `PYTHONPATH`:

```bash
./scripts/rl_car doctor
./scripts/rl_car generate
./scripts/rl_car start                 # headless gzserver
./scripts/rl_car start --gui           # opt-in Gazebo GUI
./scripts/rl_car stop
```

It intentionally does not use `set -u`, because ROS's setup scripts read unset
shell variables. It also refuses to launch if `pgrep -x gzserver` finds an
existing simulator; two servers publishing plausible data are worse than an
obvious failure. `stop` terminates exact-name `gzserver` processes owned by the
current user, so do not use it while another Gazebo project under the same user
is meant to remain running.

## Validate the car before learning

The simplest smoke command launches and cleans up its own headless server:

```bash
./scripts/rl_car smoke
```

To keep a simulator open, use terminal 1:

```bash
./scripts/rl_car start
```

In terminal 2:

```bash
./scripts/rl_car smoke --no-launch
```

The smoke test requires fresh lidar and odometry frames, drives straight, then
turns left and right, verifies displacement/yaw signs, checks the camera topic,
and exercises teleport reset. Do not train until it passes.

## Train and watch live

Headless training, suitable for automation:

```bash
./scripts/rl_car train --steps 20000 --seed 7 \
  --checkpoint-dir outputs/ppo-seed-7-v2
# Or load every PPO/training default from config/default.json.
./scripts/rl_car train --config config/default.json
```

First-class training dashboard:

```bash
./scripts/rl_car train --gui --steps 20000 --seed 7 \
  --checkpoint-dir outputs/ppo-seed-7-v2
# Add --gazebo-gui if you also want Gazebo's 3-D client.
```

The scrollable window continuously displays the track, trajectory, and lidar
rays; episode reward/running mean, length, and success rate; PPO policy/value
losses, entropy, gradient norm, approximate KL, and clip fraction; current
action probabilities and value estimate; every hyperparameter, seed, device,
and parameter count; and a timestamped event log. Controls pause/resume, save a
checkpoint immediately, and toggle greedy-watch mode. Closing the window does
not silently relabel an incomplete run as successful.

## Evaluate (measure, never infer)

Each evaluated policy must receive frames in every episode. The evaluator
raises if it sees zero lidar/odom/control frames.

```bash
# Null action sweep + privileged pure pursuit, two fixed evaluation seeds.
./scripts/rl_car evaluate --episodes 10 --seeds 200 201 \
  --output outputs/evaluation-baselines.json

# The same packet plus a learned greedy policy.
./scripts/rl_car evaluate --checkpoint outputs/ppo-seed-7-v2/best.pt \
  --episodes 10 --seeds 200 201 --output outputs/evaluation-seed-7-v2.json
```

Repeat PPO training with at least two training seeds. Evaluation writes a JSON
summary packet. A constant action completing any lap is a
track-test failure, not a success to hide.

## Measured results

The table below is populated only from saved evaluator artifacts produced on
this machine; placeholder numbers would violate the zero-frame guard and the
project's measurement rule.

| Policy | Train seed | Eval seed / N | Success | Mean abs CTE (m) | Max abs CTE (m) | Laps | Collision/off-road rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best constant (right) | n/a | 200 / 10 | 0% | 0.3549 | 1.0393 | 0 | 100% |
| Best constant (straight) | n/a | 201 / 10 | 0% | 0.3555 | 1.0581 | 0 | 100% |
| Pure pursuit | n/a | 200 / 10 | 100% | 0.0738 | 0.1429 | 10 | 0% |
| Pure pursuit | n/a | 201 / 10 | 100% | 0.0740 | 0.1413 | 10 | 0% |
| PPO attempt 1 (collapsed) | 7 | 100 / 10 | 0% | 0.3583 | 1.0433 | 0 | 100% |
| PPO attempt 1 (collapsed) | 7 | 101 / 10 | 0% | 0.3720 | 1.0459 | 0 | 100% |
| PPO v2 | 7 | 200 / 10 | 100% | 0.3005 | 0.8989 | 10 | 0% |
| PPO v2 | 7 | 201 / 10 | 100% | 0.3120 | 0.8970 | 10 | 0% |
| PPO v2 | 17 | 200 / 10 | 0% | 0.5044 | 1.1040 | 0 | 100% |
| PPO v2 | 17 | 201 / 10 | 0% | 0.4930 | 1.1658 | 0 | 100% |

These are real Gazebo runs saved in the timestamped JSON files under `outputs/`.
Every constant-action episode failed, so the track passes the discriminative
null-baseline criterion. With the final action set, the upper reference passed
all 20 episodes. PPO is **not reproducible at this 20,000-step budget**: seed 7
passed all 20 evaluation episodes, while independently trained seed 17 passed
none. Direction agreement with pure pursuit was about 9% for seed 7 and 13% for
seed 17; this pulse-timing metric is low even for the successful policy and is
reported to make action collapse visible rather than hiding it behind reward.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  -p pytest_cov --cov=ros2_rl_car.core --cov=ros2_rl_car.learning.ppo \
  --cov=ros2_rl_car.sim.world --cov=ros2_rl_car.evaluation.metrics \
  --cov-report=term-missing
```

The project wrapper also accepts the workspace-level `../.venv` used for the
measured run, but a fresh setup creates the self-contained `.venv` shown above.

The 54-test fast suite covers reward telescoping and shortcut bounds, separate terminal
and truncation masks, observation normalization/no-position design, track
curvature, sign round-trips, and evaluator frame guards. The ROS-free core has
94% statement coverage. Gazebo/UI code is integration-tested separately rather
than counted as covered by mocks; whole-package unit coverage is therefore 37%.

## Bugs found while building

- ROS pytest entry points auto-loaded `launch_testing`, which imported an
  unavailable optional YAML dependency before tests collected. The pure suite
  disables only those ROS plugins; simulator integration remains an explicit
  command.
- System Python has no Torch. The wrapper uses a Python 3.10 virtualenv and
  exposes ROS's existing site-packages through `PYTHONPATH`.
- Gazebo can fail while trying to write under a restricted home directory. The
  wrapper sets `ROS_LOG_DIR` and Matplotlib's config directory inside the local
  output tree.
- Wheel joint axes are child-frame values. The SDF follows the installed,
  working Gazebo diff-drive model convention and is covered by a model test.
- An unbounded Gazebo physics rate made a nominal 0.4 simulated-second smoke
  command travel 3.59 m because the server outran DDS/Python callbacks. Physics
  is now capped at 1,000 updates/s with a 5 ms step (at most 5x real time), so
  action duration follows `/clock` without starving the one ROS executor.
- PyTorch defaulted to 16 intra-op threads, making a tiny PPO update take 1.26 s;
  one thread took 0.027 s on the same batch. Training now defaults to one Torch
  thread and records that setting in checkpoints and the live hyperparameter panel.
- The bumper initially treated brief chassis/ground settling as a crash; pure
  pursuit then "collided" only 4.6 cm from the centre line. Contact handling now
  terminates only when either collision name belongs to a generated track wall.
- PPO seed 7's first 20,000-step run collapsed to a sensor-ignoring action:
  0% success and 0-1% direction accuracy. At 0.8 m/s, expected per-step progress
  shaping was only about +0.0046 while time cost was -0.01, so good centred
  driving was net negative. Progress weight is now 10 (the tested crash bound is
  still strict) and discrete turns are gentler at +/-0.6 rad/s instead of +/-1.2.
- A 13-hour-old delegated-test `gzserver` ignored SIGTERM. Gazebo Classic's
  orderly Ctrl-C path is SIGINT, so `stop` now sends SIGINT first, re-runs
  exact-name/current-user discovery, and uses SIGTERM only as fallback.
- The first standalone smoke run received lidar/odom immediately but zero camera
  frames because DDS discovery had not completed. Smoke now waits for a real
  BEST_EFFORT camera frame before starting its short motion measurements.
