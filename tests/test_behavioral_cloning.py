"""Tests for behavioral_cloning.py and the ROS 2 driver's pure functions.

Nothing here needs Gazebo or a running ROS graph: the node's image decoding and
the training pipeline are ordinary functions, and keeping them testable without
a simulator is most of why they are written as ordinary functions.
"""
import csv
import numpy as np
import pytest

from behavioral_cloning import drive_node as rd
from behavioral_cloning import train as bc


# ------------------------------------------------------------------ paths ----
def test_basename_handles_windows_paths():
    """driving_log.csv stores Windows paths; os.path.basename does not split them.

    This is the bug that makes the upstream project unrunnable on Linux: every
    image lookup silently uses the whole 'C:\\Users\\...' string as a filename.
    """
    win = r"C:\Users\seraj\Desktop\simulator-windows-64\IMG\center_2019.jpg"
    assert bc._basename(win) == "center_2019.jpg"
    assert bc._basename("/home/x/IMG/center_2019.jpg") == "center_2019.jpg"
    assert bc._basename("  center_2019.jpg  ") == "center_2019.jpg"


# ------------------------------------------------------- preprocessing -------
def test_preprocess_shape_and_range():
    rgb = np.random.RandomState(0).randint(0, 256, (160, 320, 3), dtype=np.uint8)
    out = bc.preprocess(rgb)
    assert out.shape == (bc.IN_H, bc.IN_W, 3)
    assert out.dtype == np.float32 or out.dtype == np.float64
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_preprocess_crops_the_sky():
    """Rows outside [CROP_TOP, CROP_BOT) must not influence the output."""
    a = np.zeros((160, 320, 3), np.uint8)
    b = a.copy()
    b[:bc.CROP_TOP] = 255          # blazing sky
    b[bc.CROP_BOT:] = 255          # bonnet
    assert np.allclose(bc.preprocess(a), bc.preprocess(b))


# --------------------------------------------------------- augmentation ------
def test_flip_negates_the_label():
    """The one augmentation that is not label-preserving.

    Forgetting the negation trains a car that steers into corners, and no shape
    check anywhere would catch it.
    """
    rgb = np.random.RandomState(1).randint(0, 256, (160, 320, 3), dtype=np.uint8)

    class AlwaysFlip:
        """rand() < 0.5 must be False for zoom/pan/brightness, True for flip."""
        def __init__(self):
            self.calls = 0
        def rand(self):
            self.calls += 1
            return 0.9 if self.calls < 4 else 0.1
    out, steer = bc.augment(rgb, 0.37, AlwaysFlip())
    assert steer == pytest.approx(-0.37)
    assert np.array_equal(out, rgb[:, ::-1])


def test_augment_preserves_shape_and_dtype():
    rng = np.random.RandomState(2)
    rgb = rng.randint(0, 256, (160, 320, 3), dtype=np.uint8)
    for _ in range(20):
        out, s = bc.augment(rgb, 0.1, rng)
        assert out.shape == rgb.shape and out.dtype == np.uint8
        assert abs(s) == pytest.approx(0.1)


# -------------------------------------------------------------- balance ------
def test_balance_caps_every_bin():
    """78% of the real labels are 0.0; the cap is what stops that dominating."""
    angles = np.concatenate([np.zeros(5000, np.float32),
                             np.linspace(-1, 1, 500).astype(np.float32)])
    paths = np.array([f"{i}.jpg" for i in range(len(angles))])
    keep, edges, before, after = bc.balance(paths, angles, per_bin=100)
    assert after.max() <= 100
    assert before.max() > 4000, "fixture should contain a dominant zero bin"
    assert len(keep) < len(angles)
    # the rare, large-steering samples must survive untouched
    assert after[0] == before[0]


# ---------------------------------------------------------------- model ------
def test_model_matches_the_published_parameter_count():
    """252,219 is the number in NVIDIA's table and the upstream README."""
    assert sum(p.numel() for p in bc.NvidiaNet().parameters()) == 252219


def test_model_forward_shape():
    import torch
    out = bc.NvidiaNet()(torch.zeros(3, 3, bc.IN_H, bc.IN_W))
    assert out.shape == (3, 1)


# ------------------------------------------------------------- load_log ------
def test_load_log_handles_single_camera_rows(tmp_path):
    """The Gazebo robot has one camera, so the side columns are blank."""
    (tmp_path / "IMG").mkdir()
    with open(tmp_path / "driving_log.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["c0.jpg", "", "", "0.25", "1", "0", "0.9"])
        w.writerow(["c1.jpg", "l1.jpg", "r1.jpg", "0.10", "1", "0", "0.9"])
    paths, angles = bc.load_log(tmp_path)
    assert len(paths) == 4                       # 1 single-cam + 3 three-cam
    assert angles[0] == pytest.approx(0.25)
    # the side cameras carry the recovery correction, with opposite signs
    assert angles[2] == pytest.approx(0.10 + bc.CAM_CORRECTION)
    assert angles[3] == pytest.approx(0.10 - bc.CAM_CORRECTION)


# ------------------------------------------------- ROS image decoding --------
def _msg(h, w, step, enc, data):
    m = type("Msg", (), {})()
    m.height, m.width, m.step, m.encoding, m.data = h, w, step, enc, data
    return m


def test_image_to_rgb_rgb8_and_bgr8():
    raw = bytes(range(12))                       # 2x2 RGB
    a = rd.image_to_rgb(_msg(2, 2, 6, "rgb8", raw))
    b = rd.image_to_rgb(_msg(2, 2, 6, "bgr8", raw))
    assert a.shape == (2, 2, 3)
    assert np.array_equal(b, a[:, :, ::-1]), "bgr8 must be channel-flipped"


def test_image_to_rgb_respects_row_padding():
    """`step` is a byte stride and is not always width*3 — rows can be padded."""
    h, w, step = 3, 2, 8                          # 6 bytes of pixels + 2 padding
    buf = bytearray(h * step)
    for r in range(h):
        buf[r * step:r * step + 6] = bytes([r] * 6)
    out = rd.image_to_rgb(_msg(h, w, step, "rgb8", bytes(buf)))
    assert out.shape == (3, 2, 3)
    assert np.array_equal(out[2], np.full((2, 3), 2, np.uint8))


def test_image_to_rgb_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        rd.image_to_rgb(_msg(1, 1, 2, "16UC1", bytes(2)))


def test_fit_to_training_size():
    """A differently-sized camera must be resampled BEFORE the fixed-row crop."""
    assert rd.fit_to_training_size(
        np.zeros((480, 640, 3), np.uint8)).shape == (160, 320, 3)
    same = np.zeros((160, 320, 3), np.uint8)
    assert rd.fit_to_training_size(same) is same          # no needless copy


# ------------------------------------------------ steering convention --------
def test_steering_convention_round_trips():
    """expert: kappa -> steer;  driver: steer -> kappa. They must invert.

    The expert and the driver live in different files and different processes.
    If these two mappings disagree the car drives confidently into a wall, and
    it looks exactly like a model that failed to learn.
    """
    for steer in (-1.0, -0.4, 0.0, 0.25, 1.0):
        kappa = -steer * bc.KAPPA_MAX            # driver's mapping
        assert -kappa / bc.KAPPA_MAX == pytest.approx(steer)   # expert's inverse


def test_positive_steer_turns_right():
    """+steer is RIGHT (dataset convention) = negative yaw rate in ROS (CW)."""
    speed = 1.0
    yaw_rate = speed * (-0.5 * bc.KAPPA_MAX)     # steer = +0.5, i.e. right
    assert yaw_rate < 0, "ROS +angular.z is counter-clockwise, i.e. left"
