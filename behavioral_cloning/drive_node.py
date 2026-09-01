#!/usr/bin/env python3
"""
ROS 2 node: drive a robot with the behavioral-cloning policy.
================================================================================
    sensor_msgs/Image  ->  [ crop, YUV, blur, resize ]  ->  NvidiaNet  ->  steer
                       ->  geometry_msgs/Twist

Run (see run_demo.sh beside this file, which does all of this for you):
    source /opt/ros/humble/setup.bash
    PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages \
      .venv/bin/python -m behavioral_cloning.drive_node --ros-args \
      -p model:=bc_model.pt -p image_topic:=/camera/image_raw

WHY THE VENV PYTHON AND NOT ROS's
    rclpy is a normal Python package with compiled extensions built for
    python3.10, and this venv is python3.10, so putting ROS's site-packages on
    PYTHONPATH lets the venv interpreter import rclpy while keeping torch. The
    alternative -- installing torch into the system interpreter -- duplicates
    600 MB to work around a path.

WHY NO cv_bridge
    cv_bridge on this machine is compiled against NumPy 1.x and raises
    "_ARRAY_API not found" under the NumPy 2 in this venv, and cv2 is not in
    the venv at all. A sensor_msgs/Image is a height, a width, a row stride and
    a byte buffer, so decoding it by hand is about ten lines (`image_to_rgb`
    below) and removes both dependencies. It is also a good look at what
    cv_bridge is actually doing for you.

THE PREPROCESSING IS IMPORTED, NOT REIMPLEMENTED
    `from behavioral_cloning.train import preprocess`. Training and serving must
    apply identical preprocessing; the standard way this breaks is that someone
    tweaks the training pipeline and the serving copy silently diverges, the
    robot drives worse, and no metric anywhere moves. One function, one place.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from behavioral_cloning.train import KAPPA_MAX, load_policy, preprocess  # noqa: E402

import torch  # noqa: E402

# The ROS imports are optional so that the pure functions below (image_to_rgb,
# fit_to_training_size) can be unit-tested by a plain `pytest` run, which has no
# ROS on its path. Only the node itself actually needs rclpy.
try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32
    HAVE_ROS = True
except ImportError:                     # no ROS on PYTHONPATH
    HAVE_ROS = False
    Node = object

CAM_H, CAM_W = 160, 320          # what the network was trained on


def image_to_rgb(msg):
    """sensor_msgs/Image -> HxWx3 uint8 RGB, without cv_bridge.

    `step` is the row stride in bytes and is NOT always width*channels — rows
    can be padded — so the buffer is reshaped by step and then sliced.
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    h, w, step = msg.height, msg.width, msg.step
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return arr[:, :, ::-1] if enc == "bgr8" else arr
    if enc in ("mono8", "8uc1"):
        return np.repeat(buf.reshape(h, step)[:, :w, None], 3, axis=2)
    if enc in ("rgba8", "bgra8"):
        arr = buf.reshape(h, step)[:, : w * 4].reshape(h, w, 4)[:, :, :3]
        return arr[:, :, ::-1] if enc == "bgra8" else arr
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


def fit_to_training_size(rgb):
    """The network expects the 320x160 frame geometry it was trained on.

    A camera of a different size is not an error -- but it must be resampled
    BEFORE preprocess(), because preprocess crops fixed pixel rows (60:135).
    Crop rows chosen for one image height mean something different in another,
    and this is a silent failure: the network still returns a number.
    """
    if rgb.shape[:2] == (CAM_H, CAM_W):
        return rgb
    from PIL import Image as PILImage
    return np.asarray(PILImage.fromarray(rgb).resize((CAM_W, CAM_H),
                                                     PILImage.BILINEAR))


class BCDriver(Node):
    def __init__(self, **overrides):
        # `overrides` lets a caller in the same process (gazebo/evaluate.py) set
        # parameters BEFORE __init__ reads them. Declaring a parameter and then
        # calling set_parameters() afterwards is too late: the checkpoint has
        # already been loaded from the default, which silently evaluated the
        # wrong model for every run until it was noticed.
        params = [rclpy.parameter.Parameter(k, value=v)
                  for k, v in overrides.items()] if overrides else None
        super().__init__("bc_driver", parameter_overrides=params)
        self.declare_parameter("model", "bc_model.pt")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("speed", 0.6)        # m/s, constant like the original
        self.declare_parameter("steer_gain", 1.0)   # scales KAPPA_MAX
        self.declare_parameter("max_rate", 20.0)    # Hz cap on inference

        g = lambda k: self.get_parameter(k).value
        # Resolve a relative checkpoint against the working directory first and
        # the repo root second. Resolving it against THIS file's directory (the
        # obvious choice) broke the moment the code moved into a package: the
        # checkpoints live at the repo root, not beside the node.
        path = Path(g("model"))
        if not path.is_absolute():
            here = Path(__file__).resolve().parent
            for base in (Path.cwd(), here.parent, here):
                if (base / path).exists():
                    path = base / path
                    break
            else:
                path = Path.cwd() / path
        if not path.exists():
            raise SystemExit(f"no checkpoint at {path} — train one first:\n"
                             f"  .venv/bin/python -m behavioral_cloning.train "
                             f"--headless")
        self.model = load_policy(str(path))
        self.speed = float(g("speed"))
        self.gain = float(g("steer_gain"))
        self.min_dt = 1.0 / max(1e-6, float(g("max_rate")))
        self.last = self.get_clock().now()
        self.n = 0

        # Cameras publish with BEST_EFFORT; a RELIABLE subscriber silently
        # never matches the publisher and you get a node that runs happily and
        # receives nothing at all.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(Image, g("image_topic"),
                                            self.on_image, qos)
        self.pub = self.create_publisher(Twist, g("cmd_topic"), 10)
        self.dbg = self.create_publisher(Float32, "~/steering", 10)
        self.get_logger().info(
            f"bc_driver: {path.name} | {g('image_topic')} -> {g('cmd_topic')} | "
            f"speed {self.speed} m/s, gain {self.gain}")

    def on_image(self, msg):
        now = self.get_clock().now()
        if (now - self.last).nanoseconds * 1e-9 < self.min_dt:
            return                                   # cap inference rate
        self.last = now
        try:
            rgb = fit_to_training_size(image_to_rgb(msg))
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        x = torch.from_numpy(
            np.ascontiguousarray(preprocess(rgb).transpose(2, 0, 1))[None]).float()
        with torch.no_grad():
            steer = float(self.model(x)[0, 0])
        steer = float(np.clip(steer, -1.0, 1.0))

        cmd = Twist()
        cmd.linear.x = self.speed
        # The inverse of the expert's mapping, using the SAME shared constant:
        # kappa = -steer * KAPPA_MAX, then omega = v * kappa. ROS's +angular.z
        # is counter-clockwise (left) while the dataset's +steer is right, which
        # is where the minus sign comes from. Getting this backwards gives a car
        # that steers confidently into every wall -- it looks like a broken
        # model and is a broken unit convention.
        cmd.angular.z = self.speed * (-steer * KAPPA_MAX * self.gain)
        self.pub.publish(cmd)
        self.dbg.publish(Float32(data=steer))
        self.n += 1
        if self.n % 50 == 0:
            self.get_logger().info(f"{self.n} frames, last steer {steer:+.3f}")

    def stop(self):
        self.pub.publish(Twist())


def main():
    if not HAVE_ROS:
        raise SystemExit(
            "rclpy not importable. Source ROS and put it on PYTHONPATH:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages "
            ".venv/bin/python -m behavioral_cloning.drive_node")
    rclpy.init()
    node = BCDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
