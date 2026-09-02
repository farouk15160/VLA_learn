# Alternating-curvature circuit

`alternating_track.world` contains the car, road boundaries, diff-drive plugin,
31-ray lidar fan, RGB camera, bumper, odometry, and `/clock` configuration.
`centerline.csv` is the single source of truth for track projection and scoring.
The curve turns both left and right; `./scripts/rl_car generate` reproduces both
files from the parametric definition in `ros2_rl_car/sim/world.py`.
