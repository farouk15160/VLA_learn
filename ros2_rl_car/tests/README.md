# Test suite

The fast pytest suite covers action sign round-trips, normalized observations
without absolute position, potential-based reward bounds, distinct terminated
and truncated handling, alternating track curvature, world generation, PPO
math, telemetry snapshots, and zero-frame evaluator guards.

Run it from this directory with the CPU virtualenv command shown in the root
README. `./scripts/rl_car smoke` is the integration test for Gazebo motion,
sensors, camera QoS, and teleport reset.
