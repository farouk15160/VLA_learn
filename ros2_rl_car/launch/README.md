# ROS 2 launch

`simulation.launch.py` launches Gazebo Classic headlessly by default and accepts
`gui`, `verbose`, and physics `seed` arguments. It refuses to launch when an
existing `gzserver` could contaminate topics. The project wrapper is preferred:
`./scripts/rl_car start [--gui]` and `./scripts/rl_car stop`.
