#!/usr/bin/env bash
# End-to-end Gazebo demo: build the world, collect, train, drive.
#
#   ./behavioral_cloning/run_demo.sh collect     record expert demonstrations
#   ./behavioral_cloning/run_demo.sh train       clone them
#   ./behavioral_cloning/run_demo.sh drive       drive with the clone and score it
#   ./behavioral_cloning/run_demo.sh all         all three, in order
#
# Everything runs headless (gzserver, no GUI). Add GUI=1 to watch in gzclient.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PORT=${GAZEBO_PORT:-11370}
export GAZEBO_MASTER_URI="http://localhost:${PORT}"

# rclpy is a python3.10 package and this venv is python3.10, so the venv
# interpreter can import it once ROS is on the path — which is how one process
# gets both rclpy and torch. See the header of ros2_bc_driver.py.
if [ -f /opt/ros/humble/setup.bash ]; then
  # ROS's setup.bash reads unbound variables (AMENT_TRACE_SETUP_FILES), so it
  # dies instantly under `set -u`. Relax it just for the source.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
  export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
fi

[ -f behavioral_cloning/track.world ] || $PY -m behavioral_cloning.make_track

start_sim() {
  # A simulator left over from an earlier run publishes to the SAME topics as
  # this one. Two of them fighting over /camera/image_raw and /cmd_vel produced
  # a spectacular fake result once (a car that "left the road" while every other
  # run was fine) and, later, a run that received no frames at all. So refuse to
  # start rather than produce a number nobody can trust.
  if pgrep -x gzserver >/dev/null 2>&1; then
    echo "ERROR: a gzserver is already running. Two simulators share topics and"
    echo "       will corrupt the measurement. Stop it first:  pkill -x gzserver"
    exit 1
  fi
  if [ "${GUI:-0}" = "1" ]; then
    gazebo --verbose behavioral_cloning/track.world &
  else
    gzserver behavioral_cloning/track.world &
  fi
  SIM=$!
  echo "gazebo pid $SIM — waiting for /camera/image_raw"
  # `ros2 topic list` costs several seconds per call (it waits on discovery), so
  # polling it once a second is not a 1 s poll -- it is closer to 5 s, and sixty
  # "iterations" can outlast the caller's timeout while the simulator has in
  # fact been ready for minutes. Give Gazebo a fixed head start, then poll a
  # handful of times.
  sleep 12
  for _ in $(seq 1 8); do
    if ros2 topic list 2>/dev/null | grep -q "^/camera/image_raw$"; then
      sleep 2                      # let the first frames flow
      return 0
    fi
  done
  echo "ERROR: /camera/image_raw never appeared — the simulator did not start."
  kill "$SIM" 2>/dev/null || true
  exit 1
}
stop_sim() { kill "$SIM" 2>/dev/null || true; wait "$SIM" 2>/dev/null || true; }

case "${1:-all}" in
  collect) start_sim; $PY -m behavioral_cloning.collect "${@:2}"; stop_sim ;;
  train)   $PY -m behavioral_cloning.train --headless --data data/gazebo_track \
             --epochs "${EPOCHS:-25}" --out bc_gazebo.pt ;;
  drive)   start_sim; $PY -m behavioral_cloning.evaluate --model bc_gazebo.pt "${@:2}"; stop_sim ;;
  all)
    start_sim; $PY -m behavioral_cloning.collect --laps 6 --timeout 600; stop_sim
    $PY -m behavioral_cloning.train --headless --data data/gazebo_track \
      --epochs "${EPOCHS:-25}" --out bc_gazebo.pt
    start_sim; $PY -m behavioral_cloning.evaluate --model bc_gazebo.pt --seconds 120; stop_sim ;;
  *) echo "usage: $0 {collect|train|drive|all}"; exit 1 ;;
esac
