#!/usr/bin/env python3
"""Record demonstrations in Gazebo: an expert drives, the camera watches.

    python -m behavioral_cloning.collect --laps 6 --out data/gazebo_track

WHY THIS EXISTS
    A network trained on the Udacity recordings will NOT drive the Gazebo car.
    Not because the idea is wrong, but because it is a different world: the
    Udacity track is a photoreal canyon road, this one is grey slabs on green.
    Behavioral cloning copies a demonstrator inside the distribution it was
    demonstrated in, and nothing in the method crosses that gap. That is
    DOMAIN SHIFT, and pretending otherwise is how demos lie.

    So we make demonstrations HERE. The expert is a pure-pursuit controller
    with access to ground-truth odometry and the true centre line -- privileged
    information the network never sees. The network gets only the camera. This
    is exactly the asymmetry in the UR5e answer sheet (docs/grid_delivery_robot
    section 7): the teacher may cheat, the student may not.

WHAT IS RECORDED
    driving_log.csv in the same 7-column layout Udacity uses, so the same
    loader and the same trainer read it with no special cases. The side-camera
    columns are left empty: this robot has one camera.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import rclpy                                             # noqa: E402
from geometry_msgs.msg import Twist                      # noqa: E402
from nav_msgs.msg import Odometry                        # noqa: E402
from rclpy.node import Node                              # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Image                        # noqa: E402

from behavioral_cloning.train import KAPPA_MAX           # noqa: E402
from behavioral_cloning.track import Progress            # noqa: E402
from behavioral_cloning.drive_node import image_to_rgb   # noqa: E402

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=1)


def yaw_of(q):
    """Yaw from a quaternion (the only Euler angle a planar robot needs)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y ** 2 + q.z ** 2))


class Expert(Node):
    """Pure pursuit around the known centre line, publishing /cmd_vel."""

    def __init__(self, centre, speed=0.9, lookahead=1.6):
        super().__init__("expert")
        self.centre = centre
        self.speed, self.lookahead = speed, lookahead
        self.prog = Progress(centre)
        self.step = float(np.hypot(*(centre[1] - centre[0])))
        self.pose = None
        self.frame = None
        self.steer = 0.0
        self.dist = 0.0
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Image, "/camera/image_raw", self.on_img, SENSOR_QOS)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def on_img(self, msg):
        self.frame = image_to_rgb(msg)

    def on_odom(self, msg):
        p = msg.pose.pose
        prev = self.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        if prev is not None:
            self.dist += math.hypot(self.pose[0] - prev[0], self.pose[1] - prev[1])
        self.control()

    def nearest(self, x, y):
        return self.prog.update(x, y)

    def control(self):
        x, y, th = self.pose
        self.nearest(x, y)
        # Aim at a point `lookahead` metres further along the centre line. Short
        # lookahead oscillates, long lookahead cuts corners -- the classic
        # pure-pursuit trade-off, and 1.6 m is tuned for this 3 m-wide lane.
        tx, ty = self.prog.ahead_point(self.lookahead, self.step)
        alpha = math.atan2(ty - y, tx - x) - th
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))    # wrap to [-pi,pi]
        # Pure pursuit: the arc through the look-ahead point has curvature
        #     kappa = 2 sin(alpha) / L
        # My first version instead set steer = -alpha*gain and then yaw rate =
        # -steer*gain, applying the gain TWICE. The car oscillated so hard it
        # averaged 4.3 m of cross-track error on a 3 m road and covered 1.9 laps
        # in the time 5 should have taken. Deriving the yaw rate from the
        # curvature, and the label from the same curvature, removes the second
        # gain and the ambiguity along with it.
        kappa = 2.0 * math.sin(alpha) / max(self.lookahead, 1e-6)
        self.steer = float(np.clip(-kappa / KAPPA_MAX, -1.0, 1.0))
        cmd = Twist()
        cmd.linear.x = self.speed
        cmd.angular.z = self.speed * kappa          # +ve = left, ROS convention
        self.pub.publish(cmd)

    def stop(self):
        self.pub.publish(Twist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "data" / "gazebo_track"))
    ap.add_argument("--laps", type=float, default=6.0)
    ap.add_argument("--hz", type=float, default=10.0, help="recording rate")
    ap.add_argument("--speed", type=float, default=0.9)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()

    from behavioral_cloning.track import lap_length, load_centreline
    centre = load_centreline()
    lap_len = lap_length(centre)
    out = Path(a.out)
    (out / "IMG").mkdir(parents=True, exist_ok=True)
    # Start from a clean slate: frame names restart at 000000 every run, so a
    # shorter run would otherwise leave orphaned images from a longer previous
    # one lying around, and the directory would stop describing the CSV.
    stale = sorted((out / "IMG").glob("center_*.jpg"))
    for f in stale:
        f.unlink()
    if stale:
        print(f"cleared {len(stale)} frames from a previous run")
    rows = []

    rclpy.init()
    node = Expert(centre, speed=a.speed)
    print(f"lap length {lap_len:.1f} m; recording {a.laps} laps at {a.hz} Hz")
    t_end = time.time() + a.timeout
    next_shot = 0.0
    off = []
    try:
        while time.time() < t_end and node.dist < a.laps * lap_len:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.frame is None or node.pose is None:
                continue
            now = time.time()
            if now < next_shot:
                continue
            next_shot = now + 1.0 / a.hz
            name = f"center_{len(rows):06d}.jpg"
            from PIL import Image as PILImage
            PILImage.fromarray(node.frame).save(out / "IMG" / name, quality=92)
            # Same 7 columns as Udacity; side cameras blank (one camera here).
            rows.append([name, "", "", f"{node.steer:.6f}", "1", "0",
                         f"{a.speed:.3f}"])
            off.append(node.nearest(node.pose[0], node.pose[1])[1])
            if len(rows) % 100 == 0:
                print(f"  {len(rows)} frames, {node.dist:.1f} m "
                      f"({node.dist/lap_len:.2f} laps), "
                      f"cross-track {np.mean(off[-100:]):.3f} m")
    finally:
        node.stop()
        with open(out / "driving_log.csv", "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"\nwrote {len(rows)} frames to {out}")
        if off:
            print(f"expert mean cross-track error {np.mean(off):.3f} m "
                  f"(max {np.max(off):.3f} m) — this is the bar the clone "
                  f"is trying to match")
        st = np.array([float(r[3]) for r in rows]) if rows else np.zeros(1)
        print(f"steering: mean {st.mean():+.3f}  std {st.std():.3f}  "
              f"|s|>0.05 on {100*np.mean(np.abs(st)>0.05):.0f}% of frames")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
