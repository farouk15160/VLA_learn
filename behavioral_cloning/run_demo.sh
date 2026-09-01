#!/usr/bin/env bash
# End-to-end Gazebo demo: build the world, collect, train, drive.
#
#   ./behavioral_cloning/run_demo.sh collect     record expert demonstrations
#   ./behavioral_cloning/run_demo.sh train       clone them
#   ./behavioral_cloning/run_demo.sh drive       drive with the clone and score it
#   ./behavioral_cloning/run_demo.sh all         all three, in order
#   ./behavioral_cloning/run_demo.sh stop        kill a leftover simulator
#
# NOTE: gzserver does not always die with the script (it is a wrapper that
# execs the real server, and the survivor gets reparented to init). A leftover
# simulator shares topics with the next one and corrupts the measurement, so
# start_sim REFUSES to run when one is already alive. If it does, run `stop`.
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
  # Just wait. Polling `ros2 topic list` for readiness looks tidier but is not:
  # each call costs seconds waiting on discovery, and from inside this script it
  # kept reporting the topic missing while the camera plugin was demonstrably
  # publishing. A dead simulator is caught properly downstream -- evaluate.py
  # refuses to report a result from zero frames -- which is the check that
  # actually matters.
  echo "gazebo pid $SIM — giving it ${SIM_WARMUP:-20}s to come up"
  sleep "${SIM_WARMUP:-20}"
}
stop_sim() {
  # `gzserver` on Ubuntu is a wrapper that execs the real server, so killing the
  # PID that $! captured does not always reap the process that holds the topics.
  # A survivor poisons the NEXT run, so verify and escalate. `pkill -x` matches
  # the exact process name, which cannot match this script.
  kill "$SIM" 2>/dev/null || true
  wait "$SIM" 2>/dev/null || true
  for _ in 1 2 3; do
    pgrep -x gzserver >/dev/null 2>&1 || return 0
    pkill -x gzserver 2>/dev/null || true
    sleep 1
  done
  pgrep -x gzserver >/dev/null 2>&1 && echo "warning: a gzserver is still running"
  return 0
}

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
  stop)
    pkill -x gzserver 2>/dev/null || true
    sleep 1
    if pgrep -x gzserver >/dev/null 2>&1; then
      pkill -9 -x gzserver 2>/dev/null || true
      sleep 1
    fi
    echo "simulators still running: $(pgrep -x gzserver | wc -l)" ;;
  *) echo "usage: $0 {collect|train|drive|all|stop}"; exit 1 ;;
esac
