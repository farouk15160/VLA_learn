"""
BEHAVIORAL CLONING — a car learns to steer by copying a human, then drives in ROS 2.
================================================================================
Run:
    .venv/bin/python -m behavioral_cloning.train --fetch    # dataset (498 MB)
    .venv/bin/python -m behavioral_cloning.train            # live dashboard
    .venv/bin/python -m behavioral_cloning.train --headless --epochs 10

    # then drive a simulated car with the trained policy:
    ./behavioral_cloning/run_demo.sh

Full write-up: behavioral_cloning/README.md
--------------------------------------------------------------------------------

WHY THIS FILE EXISTS
    README.md claims "behavior cloning IS supervised learning" and that today's
    VLAs are trained by it. This file is that claim, executable. It is the same
    shape as supervised_learning.py -- images in, a number out, cross-entropy
    swapped for MSE -- and the same shape as a VLA action head, which regresses
    (or discretizes) an action from a camera frame.

    It also demonstrates the failure that makes BC hard, and that no amount of
    validation loss will warn you about: COVARIATE SHIFT. See section 4 below,
    and docs/behavioral_cloning.md section 5.

CREDIT
    The dataset and the original project are by seraj94ai:
        https://github.com/seraj94ai/A-self-driving-car-using-behavioral-cloning
    That repo carries no licence file, so NONE of its code is copied here --
    this is an independent PyTorch implementation, and --fetch clones their
    repository onto your machine to obtain the recordings. The architecture is
    NVIDIA's, from "End to End Learning for Self-Driving Cars" (Bojarski et al.
    2016), and is reproduced from the paper's published layer table.

    Their code additionally cannot run on a current stack: it is Keras 1.x
    (`Convolution2D(24, 5, 5, subsample=(2,2))`, `Adam(lr=...)`,
    `fit_generator`), and driving_log.csv stores Windows paths that no POSIX
    basename call will split. Both are fixed here.

THE PROBLEM
    DATA ..... 5,186 timesteps recorded from a human driving a simulated track,
        each with THREE camera frames (left / centre / right, 320x160 RGB) and
        the steering angle the human was holding, in [-1, 1]. 15,558 images.

        The three cameras are a free tripling of the data, and more importantly
        a free source of RECOVERY behaviour: the left camera sees what the car
        would see if it had drifted left, so it is labelled with the human's
        steering PLUS a correction back toward the centre. Without that trick
        the car has no idea what to do once it is off-centre, because a good
        human demonstrator is never off-centre.

    THE LABEL IMBALANCE, which is the thing that actually decides whether this
        works: 78.1% of the recorded steering angles are EXACTLY ZERO. The
        track is mostly straight. Train on that as-is and MSE is minimised by
        a model that predicts ~0 for everything -- it scores well, and it
        drives straight into the first corner. `balance()` caps how many
        samples any one steering bin may contribute.

    INPUT .... the NVIDIA preprocessing, 66x200x3:
        crop rows 60:135 (drop sky and bonnet), RGB -> YUV, 3x3 Gaussian blur,
        resize to 200x66, scale to [0, 1].

    MODEL .... NVIDIA's CNN, 252,219 parameters:
        conv 24@5x5 s2 -> 36@5x5 s2 -> 48@5x5 s2 -> 64@3x3 -> 64@3x3
        -> flatten(1152) -> 100 -> 50 -> 10 -> 1,  ELU throughout.

    LOSS ..... MSE on the steering angle. This is a REGRESSION head, unlike
        supervised_learning.py's classification head. OpenVLA and RT-2 instead
        discretize each action dimension into 256 bins and use cross-entropy --
        the same choice you would face here, and section 7 of the doc measures
        both.

    OUTPUT ... one float, the steering angle. Throttle is a fixed controller,
        not learned, exactly as in the original.

DEPLOYMENT (drive_node.py + the Gazebo scripts beside it)
    The trained network is a function from a camera frame to a steering angle,
    so it drops straight into a ROS 2 node: subscribe to sensor_msgs/Image,
    publish geometry_msgs/Twist, and a Gazebo car drives itself. That is the
    same interface a real robot exposes, which is the point of doing it this
    way rather than through the original project's Unity simulator.
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "udacity_sim"
UPSTREAM = "https://github.com/seraj94ai/A-self-driving-car-using-behavioral-cloning.git"

# --- NVIDIA preprocessing geometry -------------------------------------------
CROP_TOP, CROP_BOT = 60, 135      # rows kept from the 160-row frame
IN_H, IN_W = 66, 200              # network input
CAM_CORRECTION = 0.20             # steering offset applied to the side cameras

# The one place the steering convention is defined, shared by the Gazebo expert
# (gazebo/collect.py) and the ROS 2 driver (ros2_bc_driver.py):
#   steer in [-1, 1], POSITIVE MEANS RIGHT (the Udacity dataset's convention)
#   path curvature kappa = -steer * KAPPA_MAX          [1/m]
#   yaw rate       omega = v * kappa                    [rad/s, +ve = left/CCW]
# Defining it twice is how you get a car that steers confidently into walls.
KAPPA_MAX = 1.5                   # 1/m; full lock is a 0.67 m turning radius

# RGB -> YUV (BT.601), the transform cv2.COLOR_RGB2YUV applies.
_YUV = np.array([[0.299, -0.14713, 0.615],
                 [0.587, -0.28886, -0.51499],
                 [0.114, 0.436, -0.10001]], np.float32)
_YUV_OFF = np.array([0.0, 128.0, 128.0], np.float32)


# ================================================================== data ======
def fetch(dest=DATA, url=UPSTREAM):
    """Clone the upstream repository to get the recordings.

    Their data is not vendored into this repo: it is 253 MB of JPEG, and the
    repository carries no licence. Cloning it onto your own machine is a
    different thing from redistributing it in mine.
    """
    dest = Path(dest)
    if (dest / "driving_log.csv").exists():
        print(f"dataset already present at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {url}\n  -> {dest}   (498 MB, this takes a few minutes)")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    return dest


def _basename(p):
    """Last component of a path recorded on ANY platform.

    driving_log.csv holds Windows paths ('C:\\Users\\seraj\\...\\center.jpg').
    os.path.basename splits on '/' only, so on Linux it returns the ENTIRE
    string and every image lookup fails. PureWindowsPath understands both
    separators, which is the whole fix.
    """
    return PureWindowsPath(str(p).strip()).name


def load_log(data_dir=DATA):
    """Read driving_log.csv into (image_path, steering) pairs, one per camera.

    Returns paths and angles as parallel arrays. The side cameras get the
    correction described in the module docstring: the left camera is a view
    from a car that has drifted left, so the correct action for it is the
    human's steering plus a nudge to the right.
    """
    data_dir = Path(data_dir)
    log = data_dir / "driving_log.csv"
    if not log.exists():
        raise SystemExit(f"no dataset at {data_dir}. Run with --fetch first.")
    img_dir = data_dir / "IMG"
    paths, angles = [], []
    with open(log) as fh:
        for row in csv.reader(fh):
            if len(row) < 7:
                continue
            steer = float(row[3])
            for col, corr in ((0, 0.0), (1, +CAM_CORRECTION), (2, -CAM_CORRECTION)):
                # Recordings from a single-camera robot (the Gazebo track) leave
                # the side-camera columns empty. Only the centre view exists, so
                # only it is used -- and the free recovery data the three-camera
                # trick provides is simply not available there. Section 6 of the
                # doc measures what that costs.
                cell = row[col].strip() if col < len(row) else ""
                if not cell:
                    continue
                paths.append(str(img_dir / _basename(cell)))
                angles.append(steer + corr)
    return np.array(paths), np.array(angles, np.float32)


def balance(paths, angles, n_bins=25, per_bin=400, seed=0):
    """Cap how many samples any one steering bin may contribute.

    78% of the raw angles are exactly 0. Left alone, the minimum-MSE model is
    "always predict roughly zero" -- which posts a fine validation loss and
    drives straight off the first bend. This is the single most important line
    in the file, and it is data curation rather than modelling.

    Returns the kept indices, plus the before/after histograms so the GUI can
    show you what was thrown away.
    """
    rng = np.random.RandomState(seed)
    edges = np.linspace(-1.0 - CAM_CORRECTION, 1.0 + CAM_CORRECTION, n_bins + 1)
    which = np.clip(np.digitize(angles, edges) - 1, 0, n_bins - 1)
    keep = []
    for b in range(n_bins):
        idx = np.flatnonzero(which == b)
        if len(idx) > per_bin:
            idx = rng.choice(idx, per_bin, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep)) if keep else np.array([], int)
    before = np.array([(which == b).sum() for b in range(n_bins)])
    after = np.array([(which[keep] == b).sum() for b in range(n_bins)])
    return keep, edges, before, after


# ========================================================== preprocessing =====
def preprocess(rgb):
    """The NVIDIA preprocessing: crop, YUV, blur, resize, scale. Returns 66x200x3.

    THIS FUNCTION IS IMPORTED BY drive_node.py ON PURPOSE. Training and
    serving must apply byte-identical preprocessing; when they drift the model
    sees a different distribution at deployment than it trained on, the car
    drives badly, and nothing in the training logs hints at why. Sharing one
    function is the cheapest possible defence against that.

    Implemented with numpy/PIL rather than cv2: the original used
    cv2.cvtColor/GaussianBlur/resize, but cv2 is not needed for any of it, and
    dropping the dependency is what lets the ROS 2 node run inside this venv.
    """
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    img = np.asarray(rgb)[CROP_TOP:CROP_BOT, :, :].astype(np.float32)
    yuv = img @ _YUV + _YUV_OFF                    # RGB -> YUV, BT.601
    yuv = gaussian_filter(yuv, sigma=(0.8, 0.8, 0))  # cv2's 3x3 kernel, sigma~0.8
    small = Image.fromarray(np.clip(yuv, 0, 255).astype(np.uint8)).resize(
        (IN_W, IN_H), Image.BILINEAR)
    return np.asarray(small, np.float32) / 255.0


def load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


# ========================================================== augmentation ======
def augment(rgb, steer, rng):
    """Zoom / pan / brightness / flip — the four the original project shows.

    Augmentation is doing something specific here, not just "more data": the
    human drove one racing line, so the model would otherwise only ever see the
    track from that line. Panning and zooming manufacture the off-line views
    that the car will actually encounter once its own small errors move it,
    which is a partial answer to the covariate-shift problem in section 4.

    Only flipping changes the label — it is the one transform that is not
    label-preserving, and forgetting to negate the steering there is the
    classic silent bug: the model learns to steer into corners.
    """
    img = rgb
    if rng.rand() < 0.5:                       # zoom in up to 30%
        s = rng.uniform(1.0, 1.3)
        h, w = img.shape[:2]
        ch, cw = int(h / s), int(w / s)
        r0 = rng.randint(0, h - ch + 1)
        c0 = rng.randint(0, w - cw + 1)
        img = np.asarray(_pil(img[r0:r0 + ch, c0:c0 + cw]).resize((w, h)))
    if rng.rand() < 0.5:                       # pan +/-10%
        h, w = img.shape[:2]
        dx, dy = int(rng.uniform(-.1, .1) * w), int(rng.uniform(-.1, .1) * h)
        img = np.roll(np.roll(img, dx, axis=1), dy, axis=0)
    if rng.rand() < 0.5:                       # brightness
        img = np.clip(img.astype(np.float32) * rng.uniform(0.2, 1.2), 0, 255
                      ).astype(np.uint8)
    if rng.rand() < 0.5:                       # mirror -- and negate the label
        img = img[:, ::-1]
        steer = -steer
    return img, steer


def _pil(a):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(a))


# ================================================================ dataset =====
class DrivingData(torch.utils.data.Dataset):
    """(preprocessed frame, steering angle). Augments only the training split."""

    def __init__(self, paths, angles, train=True, seed=0):
        self.paths, self.angles, self.train = paths, angles, train
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # Per-worker, per-item RNG: a single shared RandomState in a DataLoader
        # worker is forked identically into every worker, so all of them would
        # draw the SAME "random" augmentations.
        rng = np.random.RandomState((self.seed * 1000003 + i) % (2 ** 31))
        rgb = load_rgb(self.paths[i])
        steer = float(self.angles[i])
        if self.train:
            rgb, steer = augment(rgb, steer, rng)
        x = preprocess(rgb).transpose(2, 0, 1)        # HWC -> CHW
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor([steer])


# ================================================================== model =====
class NvidiaNet(nn.Module):
    """NVIDIA's end-to-end steering CNN (Bojarski et al. 2016). 252,219 params.

    Reproduced from the published layer table. Note there is no pooling: the
    downsampling is done by strided convolutions, and the receptive field is
    engineered so the 1152-unit flatten is a strip of the road ahead rather
    than a full-image summary.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.head(self.conv(x))


# ================================================================ trainer =====
def split(paths, angles, val_frac=0.2, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(paths))
    cut = int(len(idx) * (1 - val_frac))
    return idx[:cut], idx[cut:]


class Trainer:
    """One epoch at a time, so the GUI and the headless loop share the code."""

    def __init__(self, data_dir=DATA, per_bin=400, batch=64, lr=1e-3, seed=0,
                 workers=4, device=None, balanced=True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        paths, angles = load_log(data_dir)
        self.raw_n = len(paths)
        if balanced:
            keep, edges, before, after = balance(paths, angles, per_bin=per_bin,
                                                 seed=seed)
        else:
            # The ablation: train on the raw, 78%-zero distribution.
            keep = np.arange(len(paths))
            edges = np.linspace(-1.2, 1.2, 26)
            before = after = np.histogram(angles, bins=edges)[0]
        self.paths, self.angles = paths[keep], angles[keep]
        self.hist = {"edges": edges, "before": before, "after": after}
        tr, va = split(self.paths, self.angles, seed=seed)
        self.tr_idx, self.va_idx = tr, va
        self.device = torch.device(device or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = NvidiaNet().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        mk = lambda idx, train: torch.utils.data.DataLoader(
            DrivingData(self.paths[idx], self.angles[idx], train=train, seed=seed),
            batch_size=batch, shuffle=train, num_workers=workers,
            persistent_workers=workers > 0)
        self.dl_tr, self.dl_va = mk(tr, True), mk(va, False)
        # The number every result must be compared against: the MSE of simply
        # predicting the training mean. With 78% zeros that constant predictor
        # is strong, and a model that does not clearly beat it has learned
        # nothing about the road.
        self.baseline = float(np.mean((self.angles[va] -
                                       self.angles[tr].mean()) ** 2))
        self.epochs = 0
        self.steps = 0            # optimiser steps taken, i.e. weight updates
        self.seen = 0             # training images consumed
        self.t_start = time.time()
        self.log = {"train": [], "val": [], "mae": [], "corr": [], "sign": []}

    def _evaluate(self):
        self.model.eval()
        preds, trues = [], []
        tot = 0.0
        with torch.no_grad():
            for x, y in self.dl_va:
                x, y = x.to(self.device), y.to(self.device)
                p = self.model(x)
                tot += float(nn.functional.mse_loss(p, y, reduction="sum"))
                preds.append(p.cpu().numpy().ravel())
                trues.append(y.cpu().numpy().ravel())
        p = np.concatenate(preds)
        t = np.concatenate(trues)
        big = np.abs(t) > 0.05           # sign only means something off-centre
        return {
            "val": tot / len(t),
            "mae": float(np.mean(np.abs(p - t))),
            # Correlation is the honest headline: MSE can be beaten by
            # predicting a constant, correlation cannot.
            "corr": float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else 0.0,
            "sign": float(np.mean(np.sign(p[big]) == np.sign(t[big]))) if big.any() else float("nan"),
            "preds": p, "trues": t,
        }

    def train_epoch(self, on_batch=None):
        """One pass over the training split.

        `on_batch(info)` fires after every optimiser step, which is what lets the
        dashboard show the weights changing rather than a bar that moves once an
        epoch. It carries the batch loss and the gradient norm -- the gradient
        norm because it is the one number that tells you whether learning has
        stalled (norm -> 0) or is diverging (norm spiking), and neither is
        visible in the loss alone.
        """
        self.model.train()
        tot = n = 0
        t0 = time.time()
        for x, y in self.dl_tr:
            x, y = x.to(self.device), y.to(self.device)
            loss = nn.functional.mse_loss(self.model(x), y)
            self.opt.zero_grad()
            loss.backward()
            gnorm = float(nn.utils.clip_grad_norm_(self.model.parameters(), 1e9))
            self.opt.step()
            tot += loss.item() * len(y)
            n += len(y)
            self.steps += 1
            self.seen += len(y)
            if on_batch is not None:
                on_batch({"step": self.steps, "loss": loss.item(), "gnorm": gnorm,
                          "seen": self.seen, "epoch": self.epochs + 1,
                          "imgs_per_s": n / max(1e-9, time.time() - t0)})
        ev = self._evaluate()
        self.epochs += 1
        self.log["train"].append(tot / max(1, n))
        for k in ("val", "mae", "corr", "sign"):
            self.log[k].append(ev[k])
        return {"epoch": self.epochs, "train": tot / max(1, n), **ev,
                "baseline": self.baseline, "log": self.log}

    def save(self, path):
        torch.save({"model": self.model.state_dict(),
                    "arch": "NvidiaNet", "in_hw": [IN_H, IN_W]}, path)
        return path


def load_policy(path, device="cpu"):
    """Load a trained checkpoint. Used by the ROS 2 node."""
    ck = torch.load(path, map_location=device, weights_only=True)
    m = NvidiaNet().to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


# =============================================================== headless =====
def run_headless(args):
    t = Trainer(data_dir=args.data, per_bin=args.per_bin, batch=args.batch,
                lr=args.lr, seed=args.seed, workers=args.workers,
                balanced=not args.unbalanced)
    print(f"dataset {args.data}")
    print(f"  {t.raw_n} camera frames -> {len(t.paths)} after balancing "
          f"(cap {args.per_bin}/bin)" if not args.unbalanced else
          f"  {t.raw_n} camera frames, UNBALANCED (ablation)")
    print(f"  train {len(t.tr_idx)}   val {len(t.va_idx)}   device {t.device}")
    print(f"  params {sum(p.numel() for p in t.model.parameters()):,}")
    print(f"  baseline MSE (predict the mean) = {t.baseline:.4f}\n")
    print(f"{'ep':>3} {'train':>8} {'val':>8} {'MAE':>7} {'corr':>7} {'sign':>7}")
    print("-" * 46)
    t0 = time.time()
    for _ in range(args.epochs):
        s = t.train_epoch()
        print(f"{s['epoch']:>3} {s['train']:>8.4f} {s['val']:>8.4f} "
              f"{s['mae']:>7.4f} {s['corr']:>7.3f} {s['sign']:>7.3f}")
    print(f"\ntrained in {time.time() - t0:.0f}s")
    best = min(t.log["val"])
    print(f"best val MSE {best:.4f}  vs baseline {t.baseline:.4f}  "
          f"({t.baseline / max(best, 1e-9):.2f}x better)")
    print(f"final steering correlation {t.log['corr'][-1]:.3f}, "
          f"turn-direction agreement {t.log['sign'][-1]*100:.1f}%")
    out = t.save(args.out)
    print(f"saved {out}")


# ==================================================================== GUI =====
DASHBOARD_DOC = """\
WHAT YOU ARE LOOKING AT

  STATE      the camera frame the network is fed, and underneath it the 66x200
             YUV tensor it actually sees after cropping and resizing. The yellow
             lines are the crop rows: everything outside them is discarded
             before the network ever sees it.

  ACTION     two steering angles for the same frame -- the human's (yellow) and
             the network's (blue), as arrows on the image and as bars. This is a
             REGRESSION policy: one continuous action in [-1, 1], + is right.

  SCORE      behavioral cloning has NO REWARD. It is supervised learning, so the
             score is prediction error against a recorded human action, not
             return from an environment. The numbers that matter:
               MSE   what is optimised. Beatable by a constant, so never read
                     it without the dashed 'predict the mean' baseline.
               corr  correlation with the human. A constant scores 0 here, so
                     this is the honest headline.
               sign  how often the model turns the right WAY on frames where
                     the human turned at all.
             The only score that measures driving is cross-track error, and it
             needs the car in the loop -- see run_demo.sh drive.

  LEARNING   per-batch training loss (one point per weight update) with the
             per-epoch validation loss over it. Gradient norm is plotted
             alongside: -> 0 means learning has stalled, spikes mean it is
             diverging, and neither is visible in the loss curve alone.
"""


def run_gui(args):
    """A live training dashboard: state, action, score, and the log, updating
    on every weight update rather than once an epoch."""
    import queue
    import threading
    import tkinter as tk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    BG, PANEL = "#f4f4f6", "#ffffff"
    root = tk.Tk()
    root.title("Behavioral cloning — live training dashboard")
    root.configure(bg=BG)
    SW, SH = root.winfo_screenwidth(), root.winfo_screenheight()

    # --- scrolling shell (the layout is taller than most laptop screens) ----
    shell = tk.Frame(root, bg=BG); shell.pack(fill="both", expand=True)
    scroller = tk.Canvas(shell, bg=BG, highlightthickness=0)
    vbar = tk.Scrollbar(shell, orient="vertical", command=scroller.yview)
    hbar = tk.Scrollbar(shell, orient="horizontal", command=scroller.xview)
    scroller.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
    scroller.grid(row=0, column=0, sticky="nsew")
    vbar.grid(row=0, column=1, sticky="ns"); hbar.grid(row=1, column=0, sticky="ew")
    shell.rowconfigure(0, weight=1); shell.columnconfigure(0, weight=1)
    content = tk.Frame(scroller, bg=BG)
    scroller.create_window((0, 0), window=content, anchor="nw")

    def _resize(_=None):
        box = scroller.bbox("all")
        if box is None:
            return
        scroller.configure(scrollregion=box)
        (vbar.grid if box[3] - box[1] > scroller.winfo_height() + 2 else vbar.grid_remove)()
        (hbar.grid if box[2] - box[0] > scroller.winfo_width() + 2 else hbar.grid_remove)()
    content.bind("<Configure>", _resize); scroller.bind("<Configure>", _resize)

    def _wheel(ev, horiz=False):
        d = -3 if (ev.num == 4 or getattr(ev, "delta", 0) > 0) else 3
        (scroller.xview_scroll if horiz else scroller.yview_scroll)(d, "units")
    for sq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
        root.bind_all(sq, _wheel)
    for sq in ("<Shift-Button-4>", "<Shift-Button-5>", "<Shift-MouseWheel>"):
        root.bind_all(sq, lambda e: _wheel(e, True))

    banner = tk.Label(content, text="", bg=BG, anchor="w", justify="left",
                      font=("DejaVu Sans", 12, "bold"))
    banner.pack(side="top", fill="x", padx=12, pady=(8, 2))

    # --- figure sizes scaled to the screen ---------------------------------
    u = max(3.4, min(5.2, (SW - 160) / 300.0))
    row1 = tk.Frame(content, bg=BG); row1.pack(side="top", fill="x", padx=8)
    fig_state = Figure(figsize=(u, u * 0.78), dpi=100); ax_cam = fig_state.add_subplot(211)
    ax_in = fig_state.add_subplot(212)
    c_state = FigureCanvasTkAgg(fig_state, master=row1); c_state.get_tk_widget().pack(side="left", padx=4)

    fig_learn = Figure(figsize=(u * 1.25, u * 0.78), dpi=100)
    ax_loss = fig_learn.add_subplot(111); ax_gn = ax_loss.twinx()
    c_learn = FigureCanvasTkAgg(fig_learn, master=row1); c_learn.get_tk_widget().pack(side="left", padx=4)

    fig_sc = Figure(figsize=(u * 0.95, u * 0.78), dpi=100); ax_sc = fig_sc.add_subplot(111)
    c_sc = FigureCanvasTkAgg(fig_sc, master=row1); c_sc.get_tk_widget().pack(side="left", padx=4)

    row2 = tk.Frame(content, bg=BG); row2.pack(side="top", fill="x", padx=8, pady=(4, 0))
    metrics = tk.Label(row2, text="", font=("DejaVu Sans Mono", 10), justify="left",
                       bg=PANEL, anchor="nw", relief="solid", bd=1, padx=10, pady=8)
    metrics.pack(side="left", fill="y")

    fig_diag = Figure(figsize=(u * 1.05, u * 0.62), dpi=100); ax_diag = fig_diag.add_subplot(111)
    c_diag = FigureCanvasTkAgg(fig_diag, master=row2); c_diag.get_tk_widget().pack(side="left", padx=4)
    fig_hist = Figure(figsize=(u * 1.05, u * 0.62), dpi=100); ax_hist = fig_hist.add_subplot(111)
    c_hist = FigureCanvasTkAgg(fig_hist, master=row2); c_hist.get_tk_widget().pack(side="left", padx=4)

    row3 = tk.Frame(content, bg=BG); row3.pack(side="top", fill="both", expand=True, padx=8, pady=4)
    tk.Label(row3, text="event log", bg=BG, anchor="w",
             font=("DejaVu Sans", 9, "bold")).pack(side="top", fill="x")
    logbox = tk.Text(row3, height=9, font=("DejaVu Sans Mono", 9), bg="#1e1e24",
                     fg="#d8d8e0", wrap="none", relief="flat")
    logsb = tk.Scrollbar(row3, command=logbox.yview); logbox.configure(yscrollcommand=logsb.set)
    logsb.pack(side="right", fill="y"); logbox.pack(side="left", fill="both", expand=True)
    ctl = tk.Frame(content, bg=BG); ctl.pack(side="top", fill="x", pady=6)

    # ------------------------------------------------------------- state ---
    state = {"quit": False, "run": True, "phase": "training", "i": 0,
             "frames": None, "target": args.epochs, "snap": None,
             "bstep": [], "bloss": [], "bgn": [], "vx": [], "vy": [],
             "best": float("inf"), "last": None}
    q = queue.Queue()
    lock = threading.Lock()
    trainer = {"t": Trainer(data_dir=args.data, per_bin=args.per_bin,
                            batch=args.batch, lr=args.lr, seed=args.seed,
                            workers=args.workers, balanced=not args.unbalanced)}

    def log(msg, tag=""):
        stamp = time.strftime("%H:%M:%S")
        logbox.insert("end", f"[{stamp}] {msg}\n")
        logbox.see("end")

    def worker():
        while not state["quit"]:
            if not state["run"] or state["phase"] != "training":
                time.sleep(0.05); continue
            with lock:
                snap = trainer["t"].train_epoch(
                    on_batch=lambda info: q.put(("batch", info)))
            q.put(("epoch", snap))
            if trainer["t"].epochs >= state["target"]:
                state["run"] = False
                state["phase"] = "review"
    wt = threading.Thread(target=worker, daemon=True); wt.start()

    # ------------------------------------------------------------- draws ---
    def draw_state_action():
        if state["frames"] is None:
            return
        path, truth = state["frames"][state["i"] % len(state["frames"])]
        rgb = load_rgb(path)
        pre = preprocess(rgb)
        with lock:
            x = torch.from_numpy(pre.transpose(2, 0, 1)[None]).float()
            with torch.no_grad():
                pred = float(trainer["t"].model(x.to(trainer["t"].device))[0, 0])
        state["last"] = (truth, pred)

        ax_cam.clear()
        ax_cam.imshow(rgb)
        ax_cam.axhline(CROP_TOP, color="#ffcc00", lw=1)
        ax_cam.axhline(CROP_BOT, color="#ffcc00", lw=1)
        h, w = rgb.shape[:2]
        for val, col in ((truth, "#ffcc00"), (pred, "#3fa9ff")):
            ax_cam.arrow(w / 2, h - 6, val * w * 0.42, -32, width=2.2, color=col,
                         length_includes_head=True, head_width=8)
        ax_cam.set_title(f"STATE + ACTION   human {truth:+.3f}  model {pred:+.3f}"
                         f"   err {abs(pred - truth):.3f}", fontsize=9)
        ax_cam.set_xticks([]); ax_cam.set_yticks([])
        ax_in.clear()
        ax_in.imshow(np.clip(pre, 0, 1))
        ax_in.set_title("what the network is fed: 66x200 YUV, scaled to [0,1]",
                        fontsize=8)
        ax_in.set_xticks([]); ax_in.set_yticks([])
        fig_state.tight_layout(); c_state.draw_idle()

    def draw_learning():
        ax_loss.clear(); ax_gn.clear()
        if state["bstep"]:
            ax_loss.plot(state["bstep"], state["bloss"], color="#8fa8e8", lw=.8,
                         label="train, per weight update")
            k = max(1, len(state["bloss"]) // 60)
            sm = np.convolve(state["bloss"], np.ones(k) / k, "valid")
            ax_loss.plot(state["bstep"][k - 1:], sm, color="#3355bb", lw=1.6,
                         label="train, smoothed")
        if state["vx"]:
            ax_loss.plot(state["vx"], state["vy"], "o-", color="#2e9e2e", ms=3,
                         label="validation, per epoch")
        if state["snap"]:
            ax_loss.axhline(state["snap"]["baseline"], color="#d43d3d", ls="--",
                            lw=1, label="predict the mean (the bar)")
            ax_gn.plot(state["bstep"], state["bgn"], color="#c46a1e", lw=.7,
                       alpha=.5)
            ax_gn.set_ylabel("gradient norm", fontsize=8, color="#c46a1e")
        ax_loss.set_yscale("log")
        ax_loss.set_xlabel("weight updates", fontsize=8)
        ax_loss.set_ylabel("MSE (log)", fontsize=8)
        ax_loss.set_title("LEARNING", fontsize=9)
        ax_loss.grid(alpha=.3)
        if ax_loss.get_legend_handles_labels()[0]:      # nothing plotted yet
            ax_loss.legend(fontsize=6, loc="upper right")
        fig_learn.tight_layout(); c_learn.draw_idle()

    def draw_scatter(snap):
        ax_sc.clear()
        p, t = snap["preds"], snap["trues"]
        ax_sc.scatter(t, p, s=6, alpha=.35, color="#3355bb")
        lim = max(0.4, float(np.abs(t).max()) * 1.1)
        ax_sc.plot([-lim, lim], [-lim, lim], color="#d43d3d", lw=1)
        ax_sc.set_xlabel("human action", fontsize=8)
        ax_sc.set_ylabel("model action", fontsize=8)
        ax_sc.set_title(f"SCORE: corr {snap['corr']:.3f} (a constant scores 0)",
                        fontsize=9)
        ax_sc.set_xlim(-lim, lim); ax_sc.set_ylim(-lim, lim); ax_sc.grid(alpha=.3)
        fig_sc.tight_layout(); c_sc.draw_idle()

        lg = snap["log"]
        ax_diag.clear(); ax_diag.grid(alpha=.3)
        ax_diag.plot(lg["corr"], "o-", ms=3, color="#7a4fbf", label="correlation")
        ax_diag.plot(lg["sign"], "o-", ms=3, color="#c46a1e", label="turn-direction")
        ax_diag.plot(lg["mae"], "o-", ms=3, color="#2e9e2e", label="MAE")
        ax_diag.set_ylim(0, 1.05); ax_diag.set_xlabel("epoch", fontsize=8)
        ax_diag.set_title("DIAGNOSTICS", fontsize=9); ax_diag.legend(fontsize=6)
        fig_diag.tight_layout(); c_diag.draw_idle()

        ax_hist.clear()
        e = trainer["t"].hist["edges"]; ctr = (e[:-1] + e[1:]) / 2
        wd = (e[1] - e[0]) * 0.9
        ax_hist.bar(ctr, trainer["t"].hist["before"], width=wd, color="#d9d9e3",
                    label="recorded")
        ax_hist.bar(ctr, trainer["t"].hist["after"], width=wd, color="#2e9e2e",
                    label="used for training")
        ax_hist.hist(p, bins=e, histtype="step", color="#3355bb", lw=1.4,
                     label="model predictions")
        ax_hist.set_title("DATA: what was recorded, kept, and predicted", fontsize=9)
        ax_hist.set_xlabel("steering", fontsize=8); ax_hist.legend(fontsize=6)
        fig_hist.tight_layout(); c_hist.draw_idle()

    def draw_metrics():
        t = trainer["t"]
        s_ = state["snap"]
        el = time.time() - t.t_start
        truth, pred = state["last"] if state["last"] else (float("nan"),) * 2
        rows = [
            ("device", str(t.device)),
            ("parameters", f"{sum(p.numel() for p in t.model.parameters()):,}"),
            ("train / val images", f"{len(t.tr_idx)} / {len(t.va_idx)}"),
            ("", ""),
            ("epoch", f"{t.epochs} / {state['target']}"),
            ("weight updates", f"{t.steps:,}"),
            ("images seen", f"{t.seen:,}"),
            ("elapsed", f"{el:.0f} s"),
            ("", ""),
            ("train MSE", f"{s_['train']:.5f}" if s_ else "—"),
            ("val MSE", f"{s_['val']:.5f}" if s_ else "—"),
            ("baseline MSE", f"{s_['baseline']:.5f}" if s_ else "—"),
            ("best val so far", f"{state['best']:.5f}" if s_ else "—"),
            ("vs baseline", f"{s_['baseline'] / max(state['best'], 1e-12):.2f}x"
                            if s_ else "—"),
            ("", ""),
            ("MAE", f"{s_['mae']:.4f}" if s_ else "—"),
            ("correlation", f"{s_['corr']:.3f}" if s_ else "—"),
            ("turn-direction", f"{s_['sign']*100:.1f}%" if s_ else "—"),
            ("", ""),
            ("this frame: human", f"{truth:+.3f}"),
            ("this frame: model", f"{pred:+.3f}"),
        ]
        metrics.config(text="\n".join(
            f"{k:<20}{v:>14}" if k else "" for k, v in rows))

    # -------------------------------------------------------------- loops --
    def poll():
        got_epoch = None
        n_batch = 0
        while True:
            try:
                kind, payload = q.get_nowait()
            except queue.Empty:
                break
            if kind == "batch":
                state["bstep"].append(payload["step"])
                state["bloss"].append(payload["loss"])
                state["bgn"].append(payload["gnorm"])
                n_batch += 1
                state["ips"] = payload["imgs_per_s"]
            else:
                got_epoch = payload
        if got_epoch is not None:
            state["snap"] = got_epoch
            state["vx"].append(trainer["t"].steps)
            state["vy"].append(got_epoch["val"])
            better = got_epoch["val"] < state["best"]
            state["best"] = min(state["best"], got_epoch["val"])
            draw_scatter(got_epoch)
            log(f"epoch {got_epoch['epoch']:>3}  train {got_epoch['train']:.5f}"
                f"  val {got_epoch['val']:.5f}  corr {got_epoch['corr']:.3f}"
                f"  sign {got_epoch['sign']*100:.0f}%"
                + ("   <- best so far" if better else ""))
            if state["frames"] is None:
                state["frames"] = val_frames()
                log(f"holding out {len(state['frames'])} validation frames for review")
            if trainer["t"].epochs >= state["target"]:
                log("training finished — stepping through validation frames. "
                    "'Train 5 more' to continue, 'Save checkpoint' to keep it.")
        if n_batch:
            draw_learning()
            draw_metrics()
        refresh_banner()
        if not state["quit"]:
            root.after(150, poll)

    def tick():
        if state["frames"] is not None:
            draw_state_action()
            state["i"] += 1
        if not state["quit"]:
            root.after(int(1000 / max(1, args.fps)), tick)

    def refresh_banner():
        t = trainer["t"]
        if state["phase"] == "training":
            ips = state.get("ips", 0.0)
            banner.config(fg="#c46a1e",
                          text=f"① TRAINING — epoch {t.epochs + 1}/{state['target']}"
                               f"   {t.steps:,} weight updates   {ips:.0f} img/s"
                               + ("" if state["run"] else "   [paused]"))
            btn.config(text="⏸  Pause" if state["run"] else "▶  Resume")
        else:
            banner.config(fg="#2e7d32",
                          text="② REVIEW — validation frames the model never "
                               "trained on.  ← → to step.")
            btn.config(text="＋ Train 5 more")

    def val_frames(n=80):
        t = trainer["t"]
        return [(t.paths[i], float(t.angles[i])) for i in t.va_idx[:n]]

    # ----------------------------------------------------------- controls --
    def toggle():
        if state["phase"] == "review":
            state["target"] = trainer["t"].epochs + 5
            state["phase"] = "training"; state["run"] = True
            log(f"training resumed, target {state['target']} epochs")
        else:
            state["run"] = not state["run"]
            log("paused" if not state["run"] else "resumed")
        refresh_banner()

    def step(d):
        state["i"] += d
        draw_state_action(); draw_metrics()

    def save():
        with lock:
            out = trainer["t"].save(args.out)
        log(f"checkpoint saved to {out}")

    btn = tk.Button(ctl, text="⏸  Pause", command=toggle, width=13,
                    font=("DejaVu Sans", 11, "bold")); btn.pack(side="left", padx=8)
    tk.Button(ctl, text="◀ prev frame", command=lambda: step(-1),
              font=("DejaVu Sans", 10)).pack(side="left", padx=2)
    tk.Button(ctl, text="next frame ▶", command=lambda: step(1),
              font=("DejaVu Sans", 10)).pack(side="left", padx=2)
    tk.Button(ctl, text="Save checkpoint", command=save,
              font=("DejaVu Sans", 10)).pack(side="left", padx=8)
    tk.Button(ctl, text="What am I looking at?",
              command=lambda: log("\n" + DASHBOARD_DOC),
              font=("DejaVu Sans", 10)).pack(side="left", padx=8)
    root.bind_all("<Left>", lambda e: step(-1))
    root.bind_all("<Right>", lambda e: step(1))

    def on_close():
        state["quit"] = True; state["run"] = False
        root.after(50, lambda: (wt.join(timeout=3.0), root.destroy()))
    root.protocol("WM_DELETE_WINDOW", on_close)

    t0 = trainer["t"]
    log(f"dataset {args.data}: {t0.raw_n} camera frames -> {len(t0.paths)} after "
        f"balancing (cap {args.per_bin}/bin)")
    log(f"train {len(t0.tr_idx)} / val {len(t0.va_idx)} | device {t0.device} | "
        f"{sum(p.numel() for p in t0.model.parameters()):,} parameters")
    log(f"baseline: predicting the training mean gives val MSE "
        f"{t0.baseline:.5f} — the number to beat")
    log("press 'What am I looking at?' for a description of every panel")
    root.update_idletasks(); _resize()
    root.geometry("%dx%d+20+20" % (min(content.winfo_reqwidth() + 20, SW - 40),
                                   min(content.winfo_reqheight() + 20, SH - 90)))
    refresh_banner(); draw_learning(); draw_metrics(); poll(); tick()
    if args.screenshot_after:
        root.after(int(args.screenshot_after * 1000), on_close)
    root.mainloop()


# =================================================================== main =====
def build_argparser():
    import argparse
    ap = argparse.ArgumentParser(
        description="Behavioral cloning: learn to steer from human demonstrations.")
    ap.add_argument("--fetch", action="store_true",
                    help="clone the upstream dataset (498 MB) into data/udacity_sim")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--data", type=str, default=str(DATA))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--per-bin", type=int, default=400,
                    help="max samples per steering bin; the imbalance fix")
    ap.add_argument("--unbalanced", action="store_true",
                    help="ablation: keep all 78%% of zero-steering frames and "
                         "watch a good-looking val loss produce a car that "
                         "cannot turn")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(ROOT / "bc_model.pt"))
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--autostart", action="store_true")
    ap.add_argument("--screenshot-after", type=float, default=0)
    ap.add_argument("--map-px", type=int, default=0)
    return ap


if __name__ == "__main__":
    a = build_argparser().parse_args()
    if a.fetch:
        fetch(Path(a.data))
        sys.exit(0)
    run_headless(a) if a.headless else run_gui(a)
