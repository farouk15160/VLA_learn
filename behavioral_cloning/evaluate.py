#!/usr/bin/env python3
"""Drive the Gazebo car with a trained policy and measure how well it holds the lane.

    python -m behavioral_cloning.evaluate --model bc_gazebo.pt --seconds 120

Reports the only numbers that matter for a driving policy, none of which are
the validation loss:

  * mean and max CROSS-TRACK ERROR — metres from the centre line. The lane is
    3 m wide, so anything past 1.5 m is off the road.
  * DISTANCE and LAPS completed before leaving the road.
  * time to first departure, if it leaves.

A model can post an excellent validation MSE and still fail all of these,
because validation frames are drawn from the EXPERT's trajectory while these
are drawn from the policy's own. That gap is covariate shift, and this script
is what makes it visible.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import rclpy                                     # noqa: E402
from geometry_msgs.msg import Twist              # noqa: E402
from nav_msgs.msg import Odometry                # noqa: E402
from rclpy.node import Node                      # noqa: E402

from behavioral_cloning.track import Progress, lap_length, load_centreline  # noqa: E402
from behavioral_cloning.drive_node import BCDriver  # noqa: E402


class ConstantDriver(Node):
    """Publishes a fixed steering angle forever, ignoring the camera.

    The control that tells you whether the driving test is discriminative. A
    figure-eight should be undrivable this way; if it is not, any policy would
    'pass' and the numbers mean nothing.
    """

    def __init__(self, steer, speed):
        super().__init__("constant_driver")
        from behavioral_cloning.train import KAPPA_MAX
        self.n = 0
        self.speed = speed
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = speed * (-steer * KAPPA_MAX)
        self.cmd = cmd
        self.create_timer(0.05, self.tick)

    def tick(self):
        self.pub.publish(self.cmd)
        self.n += 1

    def stop(self):
        self.pub.publish(Twist())


class Monitor(Node):
    def __init__(self, centre, lane_half):
        super().__init__("lane_monitor")
        self.centre, self.lane_half = centre, lane_half
        self.prog = Progress(centre)
        self.err, self.dist, self.prev = [], 0.0, None
        self.left_at = None
        self.t0 = time.time()
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)

    def on_odom(self, m):
        p = m.pose.pose.position
        if self.prev is not None:
            self.dist += math.hypot(p.x - self.prev[0], p.y - self.prev[1])
        self.prev = (p.x, p.y)
        _, e = self.prog.update(p.x, p.y)
        self.err.append(e)
        if e > self.lane_half and self.left_at is None:
            self.left_at = (time.time() - self.t0, self.dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bc_gazebo.pt")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--speed", type=float, default=0.9)
    ap.add_argument("--lane-width", type=float, default=3.0)
    ap.add_argument("--constant", type=float, default=None,
                    help="ignore the model and steer by this constant instead; "
                         "the null baseline that validates the test")
    a = ap.parse_args()

    centre = load_centreline()
    lap = lap_length(centre)

    rclpy.init()
    if a.constant is None:
        driver = BCDriver(model=a.model, speed=a.speed)
        what = f"{a.model}"
    else:
        # A null policy that ignores the camera entirely. If this scores as well
        # as the network, the track is too easy and the whole evaluation is
        # measuring nothing -- which is worth knowing BEFORE reporting a result.
        driver = ConstantDriver(a.constant, a.speed)
        what = f"constant steering {a.constant:+.3f} (null baseline)"
    mon = Monitor(centre, a.lane_width / 2)

    print(f"driving with {what} for {a.seconds:.0f}s "
          f"(lap = {lap:.1f} m, lane half-width {a.lane_width/2:.2f} m)")
    end = time.time() + a.seconds
    try:
        while time.time() < end:
            rclpy.spin_once(driver, timeout_sec=0.01)
            rclpy.spin_once(mon, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()

    # A run that received no camera frames is not a result, it is a broken
    # simulator -- and printing "stayed on the road" for a car that never moved
    # is the worst possible failure mode for a measurement tool.
    if driver.n == 0 or not mon.err:
        driver.destroy_node(); mon.destroy_node(); rclpy.try_shutdown()
        raise SystemExit(
            f"NO DATA: {driver.n} camera frames and {len(mon.err)} odometry "
            f"samples received. The simulator is not publishing — check that "
            f"gzserver is running and that no second one is competing for the "
            f"same topics.")

    e = np.array(mon.err)
    print(f"\nframes processed      {driver.n}")
    print(f"distance driven       {mon.dist:.1f} m  ({mon.dist/lap:.2f} laps)")
    print(f"cross-track error     mean {e.mean():.3f} m   max {e.max():.3f} m")
    print(f"time inside the lane  {100*np.mean(e <= a.lane_width/2):.1f}% of samples")
    if mon.left_at:
        print(f"LEFT THE ROAD after {mon.left_at[0]:.1f} s / {mon.left_at[1]:.1f} m")
    else:
        print("stayed on the road for the whole run")
    driver.destroy_node(); mon.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__":
    main()
