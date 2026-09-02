# Learning examples

These self-contained sections explain the progression from labelled-data
learning to trial-and-error learning:

- [`supervised_learning/`](supervised_learning/README.md): an SVHN digit CNN.
- [`reinforcement_learning/`](reinforcement_learning/README.md): a small
  REINFORCE car environment with a learned value baseline.
- [`grid_delivery/`](grid_delivery/README.md): a goal-conditioned stochastic
  navigation policy with GAE and curriculum learning.

Run them from the repository root so generated data and output paths remain
consistent:

```bash
.venv/bin/python learning_examples/supervised_learning/supervised_learning.py
.venv/bin/python learning_examples/reinforcement_learning/reinforcement_learning.py
.venv/bin/python learning_examples/grid_delivery/grid_delivery_robot.py
```

These are educational simulations. The real ROS 2/Gazebo driving system lives
in [`../ros2_rl_car/`](../ros2_rl_car/README.md).
