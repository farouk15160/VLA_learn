# VLA Learn

A collection of runnable learning systems, from small teaching examples to a
tested ROS 2/Gazebo autonomous-driving project.

## Repository map

| Section | Learning method | Purpose |
|---|---|---|
| [`ros2_rl_car/`](ros2_rl_car/README.md) | Reinforcement learning (PPO) | A Gazebo car learns an alternating-curvature track through trial and error. This is the primary self-driving project. |
| [`point_reach/`](point_reach/README.md) | Behavioral cloning and DAgger | A compact 2-D demonstration of imitation learning and covariate shift. It is not the Gazebo car. |
| [`learning_examples/`](learning_examples/README.md) | Supervised learning and reinforcement learning | Three focused educational programs with documentation beside their code. |
| `vla-arm-sim/` | Independent nested repository | The arm simulator remains at the repository root and keeps its own Git history. |

## Which car learns to drive?

The accepted Gazebo task is [`ros2_rl_car`](ros2_rl_car/README.md). It uses PPO,
lidar, speed, heading error, and cross-track error. It does **not** copy a
demonstrator, use behavioral cloning, or train from camera pixels in v1. The RGB
camera is published for inspection, while the CPU-friendly policy learns from
ray features.

The historical camera behavioral-cloning car was a different project and is
not part of the current branch. `point_reach/` remains as the repository's
small behavioral-cloning/DAgger example.

## Setup

The root virtualenv supports the educational examples:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The ROS car has additional ROS/Gazebo setup and its own CPU-Torch wrapper:

```bash
cd ros2_rl_car
./scripts/setup_venv.sh
./scripts/rl_car doctor
./scripts/rl_car smoke
./scripts/rl_car train --gui --steps 20000 --seed 7
```

See each section README for its full commands, algorithms, mathematics, and
measured results.

## Live verification

The automated test sources were removed as requested after the reorganized
suite passed all 72 checks. Verify the driving integration with the real
headless simulator:

```bash
cd ros2_rl_car
./scripts/rl_car smoke
```
