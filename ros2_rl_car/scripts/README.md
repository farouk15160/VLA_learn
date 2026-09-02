# Operator scripts

`setup_venv.sh` creates a Python 3.10 virtualenv, installs CPU-only PyTorch and
project dependencies, and installs this package in editable mode. `rl_car`
sources ROS 2 Humble, exposes ROS's Python packages through `PYTHONPATH`, places
runtime logs inside `outputs/`, and delegates to the typed Python CLI.

The shell scripts intentionally use `set -eo pipefail`, not `set -u`, because
ROS setup scripts read unset variables.
