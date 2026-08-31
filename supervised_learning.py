"""
SUPERVISED LEARNING — recognising digits in real RGB photographs.
================================================================================
Run:
    .venv/bin/python supervised_learning.py              # GUI
    .venv/bin/python supervised_learning.py --headless   # terminal only

Full write-up: docs/supervised_learning.md
--------------------------------------------------------------------------------

THE DATA
    SVHN (Street View House Numbers): 26,032 crops of house numbers
    photographed from Google Street View.  http://ufldl.stanford.edu/housenumbers

    RESOLUTION ............ 32 x 32 pixels, 3 colour channels = 3,072 numbers
        per image. Small on purpose: big enough that a digit is legible and a
        convolution has something to pool over, small enough that an epoch
        takes ~2.5 s so you can watch it learn. The digit is centred in the
        crop, but neighbouring house-number digits often intrude at the edges
        -- that is real, and it causes some of the model's errors.

    THE RGB CHANNELS ...... the tensor is (N, 3, 32, 32) in PyTorch's
        channels-first layout: [batch, channel, height, width]. The raw .mat
        file is (32, 32, 3, N), so we transpose. Values are scaled 0..255 -> 0..1
        and then standardized PER CHANNEL:

            mu = X_train.mean(axis=(0, 2, 3))   # one number per R, G, B
            sd = X_train.std(axis=(0, 2, 3))

        Per channel, NOT per pixel. A digit can sit anywhere in the crop, so
        per-pixel statistics would bake in position, which is not what we want
        to normalize away. The three channels genuinely differ (measured on
        this split: R 0.452, G 0.452, B 0.468 mean; sd 0.219 / 0.226 / 0.228)
        because of daylight colour temperature and sensor response.

        Where RGB actually enters the network: the FIRST convolution is
        Conv2d(3, 32, kernel_size=3). Each of its 32 filters is a 3x3x3 cube --
        3 wide, 3 tall, and 3 deep across the colour channels -- so a filter
        CAN respond to a red-on-white edge differently from a blue-on-white
        one. After that first layer the notion of colour is gone: layer 2 sees
        32 abstract feature maps, not R/G/B.

        BUT -- and this is worth more than the mechanism -- I measured whether
        the colour actually helps, and it does not:

            RGB (3 real channels)      0.9262 test accuracy
            greyscale, copied to 3ch   0.9299        (`--sweep`, 2 seeds)

        Identical architecture, identical parameter count; only the information
        differs. That is a tie within noise (one standard error on 4,032 test
        images is ~0.4pp). The honest conclusion is that on SVHN, digit
        identity is carried by SHAPE and local contrast, not by hue -- house
        numbers come in every colour, so colour is mostly nuisance variation
        the network has to learn to ignore. Run `--sweep` and see for yourself.
        I assumed colour would help before I measured it. It did not.

    LABELS ................ 0-9. Careful: SVHN stores the digit '0' as class
        10, not 0. Forget to remap and you get a model that is confidently
        wrong about every zero. We do `y % 10`.

    CLASS BALANCE ......... genuinely skewed, and printed at startup: this
        training split holds 3,569 examples of '1' but only 1,103 of '9'.
        House numbers really are like that.

    SPLIT ................. 18,000 train / 4,000 validation / 4,032 test.
        Validation is what YOU tune against; test is opened once.

THE MODEL — a small CNN, ~667k parameters
        Conv(3->32) BN ReLU  Conv(32->32) BN ReLU  MaxPool   32x32 -> 16x16
        Conv(32->64) BN ReLU Conv(64->64) BN ReLU  MaxPool   16x16 -> 8x8
        Conv(64->128) BN ReLU                      MaxPool   8x8   -> 4x4
        Flatten(2048) Dropout Linear(256) ReLU Dropout Linear(10)

    Why convolutions rather than the flat MLP you would use on tabular data?
    Flattening a 32x32x3 photo into 3,072 independent inputs throws away the
    fact that neighbouring pixels are related. A convolution slides the SAME
    small filter across every position, so it needs far fewer parameters and it
    recognises a '7' wherever it appears, instead of learning
    '7-in-the-top-left' and '7-in-the-middle' as unrelated patterns.

    Loss: CrossEntropyLoss on raw logits (it applies log-softmax internally --
    do not add your own softmax). Optimiser: Adam, lr 1e-3, weight decay 1e-4.

    Result on this machine: ~94% test accuracy in 8-13 epochs. Compare with
    97.5% on clean 8x8 digits. That gap is the point -- the toy number was
    never the real number.

THE VLA CONNECTION
    This is the shape of a Vision-Language-Action policy head. OpenVLA
    discretizes "each dimension of the robot actions separately into one of 256
    bins" and trains with "a standard next-token prediction objective,
    evaluating the cross-entropy loss on the predicted action tokens only".
    Image in, discrete-class logits out, cross-entropy -- exactly this file, at
    1/1000 scale. Behavior cloning IS supervised learning.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat

DATA = Path(__file__).parent / "data" / "svhn_test_32x32.mat"
SVHN_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"


def ensure_data():
    """Fetch SVHN on first run. The 62 MB .mat is not in git, so a fresh clone
    downloads it once, here, rather than failing with a confusing traceback."""
    if DATA.exists():
        return
    import urllib.request
    DATA.parent.mkdir(parents=True, exist_ok=True)
    print(f"SVHN not found locally. Downloading ~62 MB from\n  {SVHN_URL}")

    # Only redraw when the whole-percent changes. urlretrieve calls back on
    # every 8 KB block, so printing each time produces ~8,000 lines when stdout
    # is a pipe rather than a terminal.
    last = [-1]

    def progress(blocks, bs, total):
        done = blocks * bs
        pct = int(min(100, 100 * done / total)) if total > 0 else 0
        if pct != last[0]:
            last[0] = pct
            print(f"\r  {done/1e6:6.1f} MB / {total/1e6:.1f} MB  ({pct:3d}%)",
                  end="", flush=True)

    tmp = DATA.with_suffix(".part")
    urllib.request.urlretrieve(SVHN_URL, tmp, reporthook=progress)
    tmp.rename(DATA)          # rename only on success, so a half download
                              # never gets mistaken for a complete one
    print(f"\n  saved to {DATA}")


def load_svhn(seed=0, n_train=18000, n_val=4000, verbose=True):
    ensure_data()
    m = loadmat(DATA)
    X = m["X"]                                   # (32, 32, 3, N) uint8
    y = m["y"].ravel().astype(np.int64) % 10     # 10 -> 0  (see docstring)
    X = np.transpose(X, (3, 2, 0, 1)).astype(np.float32) / 255.0   # (N,3,32,32)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, len(y))

    # Per-CHANNEL statistics from the training split only. One number per RGB
    # channel rather than per pixel: a digit can appear anywhere in the crop, so
    # per-pixel statistics would encode position, which is not what we want to
    # normalize away. Channels differ because of lighting and sensor response.
    mu = X[tr].mean(axis=(0, 2, 3))
    sd = X[tr].std(axis=(0, 2, 3))
    norm = lambda a: (a - mu[None, :, None, None]) / sd[None, :, None, None]

    if verbose:
        print(f"  SVHN: {len(y)} RGB photos, 32x32x3")
        print(f"  split: train {n_train} / val {n_val} / test {len(y)-n_train-n_val}")
        print(f"  class counts (train): {np.bincount(y[tr], minlength=10).tolist()}")
        print(f"  channel means R{mu[0]:.3f} G{mu[1]:.3f} B{mu[2]:.3f}")
        print(f"  channel stds  R{sd[0]:.3f} G{sd[1]:.3f} B{sd[2]:.3f}")

    t = torch.from_numpy
    return {"Xtr": t(norm(X[tr])), "ytr": t(y[tr]),
            "Xva": t(norm(X[va])), "yva": t(y[va]),
            "Xte": t(norm(X[te])), "yte": t(y[te]),
            "raw_va": X[va], "raw_te": X[te], "mu": mu, "sd": sd}


class CNN(nn.Module):
    """A small convolutional network — the right tool for images.

    Why not the flat MLP from 00_basics? Because flattening a 32x32x3 image
    into 3072 numbers throws away the fact that neighbouring pixels are
    related. A convolution slides the SAME small filter across every position,
    so it (a) needs far fewer parameters and (b) recognizes a '7' wherever it
    appears in the crop, instead of learning '7-in-the-top-left' and
    '7-in-the-middle' as unrelated patterns.

    The very first layer takes 3 input channels — that is where RGB enters. Its
    filters are 3x3x3 cubes that can respond to colour, not just brightness.
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        def blk(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.features = nn.Sequential(
            *blk(3, 32), *blk(32, 32), nn.MaxPool2d(2),      # 32x32 -> 16x16
            *blk(32, 64), *blk(64, 64), nn.MaxPool2d(2),     # 16x16 -> 8x8
            *blk(64, 128), nn.MaxPool2d(2),                  # 8x8   -> 4x4
            # Adaptive pool so the head's input size is fixed no matter what
            # resolution comes in. That is what lets --sweep compare 32x32 vs
            # 16x16 vs 8x8 with an otherwise identical network.
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


class Trainer:
    """One epoch at a time, so a GUI can draw between epochs."""

    def __init__(self, data, lr=1e-3, batch=128, device=None, dropout=0.3, seed=0):
        torch.manual_seed(seed)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.d = data
        self.model = CNN(dropout).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.lossf = nn.CrossEntropyLoss()
        self.batch = batch
        self.epoch = 0
        self.hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
        self.best = {"acc": -1, "epoch": -1, "state": None}
        self.Xtr = data["Xtr"].to(self.dev); self.ytr = data["ytr"].to(self.dev)
        self.Xva = data["Xva"].to(self.dev); self.yva = data["yva"].to(self.dev)

    @torch.no_grad()
    def evaluate(self, X, y, bs=1024):
        self.model.eval()
        L = C = 0
        for i in range(0, len(X), bs):
            o = self.model(X[i:i + bs])
            L += self.lossf(o, y[i:i + bs]).item() * len(o)
            C += (o.argmax(1) == y[i:i + bs]).sum().item()
        return L / len(X), C / len(X)

    @torch.no_grad()
    def predict(self, X, bs=1024):
        self.model.eval()
        out = []
        for i in range(0, len(X), bs):
            out.append(torch.softmax(self.model(X[i:i + bs].to(self.dev)), 1).cpu())
        return torch.cat(out)

    def train_epoch(self):
        self.model.train()
        n = len(self.Xtr)
        perm = torch.randperm(n, device=self.dev)
        L = C = 0
        for i in range(0, n, self.batch):
            idx = perm[i:i + self.batch]
            xb, yb = self.Xtr[idx], self.ytr[idx]
            out = self.model(xb)
            loss = self.lossf(out, yb)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            L += loss.item() * len(idx)
            C += (out.argmax(1) == yb).sum().item()
        tr_loss, tr_acc = L / n, C / n
        va_loss, va_acc = self.evaluate(self.Xva, self.yva)
        self.epoch += 1
        h = self.hist
        h["tr_loss"].append(tr_loss); h["va_loss"].append(va_loss)
        h["tr_acc"].append(tr_acc); h["va_acc"].append(va_acc)
        if va_acc > self.best["acc"]:
            self.best = {"acc": va_acc, "epoch": self.epoch,
                         "state": {k: v.detach().clone()
                                   for k, v in self.model.state_dict().items()}}
        return {"epoch": self.epoch, "tr_loss": tr_loss, "va_loss": va_loss,
                "tr_acc": tr_acc, "va_acc": va_acc, "hist": h,
                "best": self.best["acc"], "best_epoch": self.best["epoch"]}


def preprocess_upload(path, mu, sd):
    """Turn any user image file into a normalized (1,3,32,32) tensor."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0            # (32,32,3)
    x = np.transpose(a, (2, 0, 1))[None]                    # (1,3,32,32)
    x = (x - mu[None, :, None, None]) / sd[None, :, None, None]
    return torch.from_numpy(x.astype(np.float32)), a


import argparse
import queue
import threading
import time
from pathlib import Path


OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
N_TILES = 40


def _sweep_run(data, epochs, seed, dropout=0.3, lr=1e-3):
    t = Trainer(data, lr=lr, batch=128, dropout=dropout, seed=seed)
    for _ in range(epochs):
        t.train_epoch()
    t.model.load_state_dict(t.best["state"])
    p = t.predict(data["Xte"])
    return float((p.argmax(1) == data["yte"]).float().mean()), t.best["epoch"]


def _variant(d, mode):
    """Return a copy of the dataset transformed for one ablation arm."""
    import torch.nn.functional as F
    out = dict(d)
    for k in ("Xtr", "Xva", "Xte"):
        x = d[k]
        if mode == "gray":
            # ITU-R BT.601 luma, then repeated back into 3 channels so the
            # network architecture and parameter count stay identical --
            # only the INFORMATION changes.
            g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])
            x = g.unsqueeze(1).repeat(1, 3, 1, 1)
        elif mode.startswith("res"):
            r = int(mode[3:])
            if r != x.shape[-1]:
                x = F.interpolate(x, size=(r, r), mode="area")
        out[k] = x
    return out


def run_sweep(a):
    """Ablations that answer 'why RGB?' and 'why 32x32?' with measurements."""
    base = load_svhn(seed=a.seed, verbose=False)
    E, seeds = a.epochs, (0, 1)

    print(f"\nEach cell: mean TEST accuracy over seeds {seeds}, {E} epochs, "
          f"best-val checkpoint.\n")

    print("=== 1. DOES COLOUR ACTUALLY HELP? (identical net, identical params) ===")
    print(f"  {'input':<28}{'test acc':>10}{'best ep':>9}")
    for mode, label in (("rgb", "RGB (3 real channels)"),
                        ("gray", "greyscale, copied to 3ch")):
        d = base if mode == "rgb" else _variant(base, "gray")
        r = [_sweep_run(d, E, s) for s in seeds]
        print(f"  {label:<28}{np.mean([x[0] for x in r]):>10.4f}"
              f"{np.mean([x[1] for x in r]):>9.1f}")

    print("\n=== 2. HOW MUCH RESOLUTION IS NEEDED? ===")
    print(f"  {'input':<28}{'test acc':>10}{'best ep':>9}")
    for r_ in (32, 16, 8):
        d = _variant(base, f"res{r_}")
        r = [_sweep_run(d, E, s) for s in seeds]
        print(f"  {f'{r_}x{r_}x3':<28}{np.mean([x[0] for x in r]):>10.4f}"
              f"{np.mean([x[1] for x in r]):>9.1f}")

    print("\n=== 3. DROPOUT ===")
    print(f"  {'dropout':<28}{'test acc':>10}{'best ep':>9}")
    for dr in (0.0, 0.15, 0.3, 0.5):
        r = [_sweep_run(base, E, s, dropout=dr) for s in seeds]
        print(f"  {dr:<28}{np.mean([x[0] for x in r]):>10.4f}"
              f"{np.mean([x[1] for x in r]):>9.1f}")


def run_headless(a):
    d = load_svhn(seed=a.seed)
    t = Trainer(d, lr=a.lr, batch=a.batch, seed=a.seed)
    print(f"  device {t.dev}   params {sum(p.numel() for p in t.model.parameters()):,}")
    for _ in range(a.epochs):
        s = t.train_epoch()
        print(f"  ep {s['epoch']:>3}  tr_loss {s['tr_loss']:.4f}  va_loss {s['va_loss']:.4f}"
              f"  tr_acc {s['tr_acc']:.4f}  va_acc {s['va_acc']:.4f}")
    t.model.load_state_dict(t.best["state"])
    p = t.predict(d["Xte"])
    print(f"\n  best val {t.best['acc']:.4f} (epoch {t.best['epoch']})")
    print(f"  TEST accuracy {float((p.argmax(1) == d['yte']).float().mean()):.4f}")


def run_gui(a):
    import tkinter as tk
    from tkinter import filedialog
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("SL — recognising digits in real RGB photos (SVHN + CNN)")
    root.configure(bg="#f4f4f6")
    root.geometry("1580x820")

    print("loading SVHN ...")
    data = load_svhn(seed=a.seed)
    trainer = {"t": Trainer(data, lr=a.lr, batch=a.batch, seed=a.seed)}
    lock = threading.Lock()
    state = {"run": False, "quit": False}
    q = queue.Queue()

    rng = np.random.RandomState(0)
    tile_idx = rng.choice(len(data["yva"]), N_TILES, replace=False)
    tile_true = data["yva"][tile_idx].numpy()

    # ------------------------------------------------------------- layout ---
    main = tk.Frame(root, bg="#f4f4f6"); main.pack(fill="both", expand=True)

    # --- left: curves
    left = tk.Frame(main, bg="#f4f4f6"); left.pack(side="left", fill="y", padx=8, pady=8)
    fig = Figure(figsize=(4.6, 5.6), dpi=100)
    axL = fig.add_subplot(211); axA = fig.add_subplot(212)
    cfig = FigureCanvasTkAgg(fig, master=left); cfig.get_tk_widget().pack()
    info = tk.Label(left, text="press ▶ Train", justify="left", anchor="w",
                    font=("DejaVu Sans Mono", 10), bg="#f4f4f6")
    info.pack(fill="x", pady=4)

    # --- middle: the live tile panel
    mid = tk.Frame(main, bg="#f4f4f6"); mid.pack(side="left", fill="y",
                                                 padx=8, pady=8)
    tk.Label(mid, text="40 validation photos the network never trains on",
             font=("DejaVu Sans", 11, "bold"), bg="#f4f4f6").pack(anchor="w")
    tile_hdr = tk.Label(mid, text="green = correct, red = wrong",
                        font=("DejaVu Sans", 10), fg="#555", bg="#f4f4f6")
    tile_hdr.pack(anchor="w")
    COLS, Z = 8, 62
    tc = tk.Canvas(mid, width=COLS * (Z + 10) + 10, height=(N_TILES // COLS) * (Z + 26) + 12,
                   bg="#ffffff", highlightthickness=1, highlightbackground="#c9c9d0")
    tc.pack(pady=6)

    photos, rects, texts = [], [], []
    for k, ix in enumerate(tile_idx):
        r, c = divmod(k, COLS)
        x, y = 10 + c * (Z + 10), 8 + r * (Z + 26)
        img = (data["raw_va"][ix].transpose(1, 2, 0) * 255).astype(np.uint8)
        ph = ImageTk.PhotoImage(Image.fromarray(img).resize((Z, Z), Image.NEAREST))
        photos.append(ph)                       # keep a ref or Tk garbage-collects it
        tc.create_image(x, y, image=ph, anchor="nw")
        rects.append(tc.create_rectangle(x - 2, y - 2, x + Z + 2, y + Z + 2,
                                         outline="#cccccc", width=2))
        texts.append(tc.create_text(x + Z / 2, y + Z + 11, text=f"?  (true {tile_true[k]})",
                                    font=("DejaVu Sans Mono", 8), fill="#666"))

    # --- right: upload + prediction
    right = tk.Frame(main, bg="#f4f4f6"); right.pack(side="left", fill="y", padx=8, pady=8)
    tk.Label(right, text="Try your own image", font=("DejaVu Sans", 12, "bold"),
             bg="#f4f4f6").pack(anchor="w")
    tk.Label(right, justify="left", anchor="w", fg="#555", bg="#f4f4f6",
             font=("DejaVu Sans", 9), wraplength=300,
             text="Crop tightly to ONE digit — the model was trained on tight "
                  "32x32 crops of single house numbers, so a wide photo of a "
                  "whole door will not work. Anything the file dialog accepts "
                  "is resized to 32x32."
             ).pack(anchor="w", pady=(0, 6))

    up = tk.Canvas(right, width=300, height=132, bg="#ffffff",
                   highlightthickness=1, highlightbackground="#c9c9d0")
    up.pack()
    figp = Figure(figsize=(3.2, 2.3), dpi=100)
    axp = figp.add_subplot(111)
    cp = FigureCanvasTkAgg(figp, master=right); cp.get_tk_widget().pack(pady=6)
    verdict = tk.Label(right, text="", font=("DejaVu Sans", 22, "bold"), bg="#f4f4f6")
    verdict.pack()
    upstate = {"photos": []}

    def show_prediction(arr01, title):
        """arr01: (32,32,3) float 0..1 — exactly what the net will be fed."""
        up.delete("all"); upstate["photos"].clear()
        big = (arr01 * 255).astype(np.uint8)
        ph = ImageTk.PhotoImage(Image.fromarray(big).resize((96, 96), Image.NEAREST))
        upstate["photos"].append(ph)
        up.create_image(8, 24, image=ph, anchor="nw")
        up.create_text(8, 12, text="what the net sees", anchor="w",
                       font=("DejaVu Sans", 8), fill="#555")
        # RGB channels, split out, so "3 channels" stops being an abstraction
        up.create_text(116, 12, text="its 3 colour channels", anchor="w",
                       font=("DejaVu Sans", 8), fill="#555")
        for j, nm in enumerate("RGB"):
            ch = np.zeros_like(big); ch[:, :, j] = big[:, :, j]
            p2 = ImageTk.PhotoImage(Image.fromarray(ch).resize((54, 54), Image.NEAREST))
            upstate["photos"].append(p2)
            up.create_image(116 + j * 60, 38, image=p2, anchor="nw")
            up.create_text(116 + j * 60 + 27, 30, text=nm,
                           font=("DejaVu Sans", 9, "bold"), fill="#333")

        x = torch.from_numpy(((np.transpose(arr01, (2, 0, 1))[None] -
                               data["mu"][None, :, None, None]) /
                              data["sd"][None, :, None, None]).astype(np.float32))
        with lock:
            p = trainer["t"].predict(x)[0].numpy()
        axp.clear()
        axp.bar(range(10), p, color=["#3355bb"] * 10)
        axp.patches[int(p.argmax())].set_color("#2e9e2e")
        axp.set_xticks(range(10)); axp.set_ylim(0, 1)
        axp.set_title(title, fontsize=9); axp.set_xlabel("digit", fontsize=8)
        figp.tight_layout(); cp.draw_idle()
        verdict.config(text=f"{int(p.argmax())}   ({p.max()*100:.1f}%)",
                       fg="#2e9e2e" if p.max() > .6 else "#c46a1e")

    def do_upload():
        f = filedialog.askopenfilename(
            title="Pick an image of a single digit",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All", "*.*")])
        if not f:
            return
        _, arr = preprocess_upload(f, data["mu"], data["sd"])
        show_prediction(arr, f"prediction for {Path(f).name}")

    def do_random_test():
        i = np.random.randint(len(data["yte"]))
        show_prediction(data["raw_te"][i].transpose(1, 2, 0),
                        f"random test photo (true = {int(data['yte'][i])})")

    tk.Button(right, text="📁  Upload an image…", command=do_upload,
              font=("DejaVu Sans", 11, "bold")).pack(fill="x", pady=3)
    tk.Button(right, text="🎲  Use a random test photo", command=do_random_test,
              font=("DejaVu Sans", 10)).pack(fill="x")
    tk.Label(right, justify="left", anchor="w", fg="#555", bg="#f4f4f6",
             font=("DejaVu Sans", 9), wraplength=300,
             text="\nThe bars are the softmax output — the model's full "
                  "probability distribution, not just its guess. A tall single "
                  "bar means confident; several similar bars mean it is torn. "
                  "Watch how confident it is when it is WRONG: that is the "
                  "calibration problem."
             ).pack(anchor="w")

    # ------------------------------------------------------------- worker ---
    def worker():
        while not state["quit"]:
            if not state["run"]:
                time.sleep(0.05); continue
            with lock:
                snap = trainer["t"].train_epoch()
                probs = trainer["t"].predict(data["Xva"][tile_idx])
            snap["pred"] = probs.argmax(1).numpy()
            snap["conf"] = probs.max(1).values.numpy()
            q.put(snap)

    threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------------- draw ----
    def draw(s):
        h = s["hist"]
        e = range(1, len(h["tr_loss"]) + 1)
        axL.clear(); axA.clear()
        axL.plot(e, h["tr_loss"], label="train"); axL.plot(e, h["va_loss"], label="val")
        axL.set_title("loss", fontsize=10); axL.legend(fontsize=8); axL.grid(alpha=.3)
        axA.plot(e, h["tr_acc"], label="train"); axA.plot(e, h["va_acc"], label="val")
        axA.set_title("accuracy", fontsize=10); axA.legend(fontsize=8); axA.grid(alpha=.3)
        axA.set_xlabel("epoch", fontsize=9)
        fig.tight_layout(); cfig.draw_idle()

        ok = 0
        for k in range(N_TILES):
            good = s["pred"][k] == tile_true[k]
            ok += good
            tc.itemconfig(rects[k], outline="#2e9e2e" if good else "#d43d3d")
            tc.itemconfig(texts[k], fill="#2e9e2e" if good else "#d43d3d",
                          text=f"{s['pred'][k]} ({s['conf'][k]*100:.0f}%) t{tile_true[k]}")
        tile_hdr.config(text=f"epoch {s['epoch']}:  {ok}/{N_TILES} correct here   "
                             f"|  green = correct, red = wrong")
        gap = s["tr_acc"] - s["va_acc"]
        info.config(text=(
            f"epoch      {s['epoch']:>4}\n"
            f"train acc  {s['tr_acc']:.4f}\n"
            f"val   acc  {s['va_acc']:.4f}   (best {s['best']:.4f} @ ep {s['best_epoch']})\n"
            f"gap        {gap:+.4f}   <- grows = overfitting\n"
            f"train loss {s['tr_loss']:.4f}\n"
            f"val   loss {s['va_loss']:.4f}"))

    def poll():
        got = None
        while True:
            try:
                got = q.get_nowait()
            except queue.Empty:
                break
        if got is not None:
            draw(got)
        if not state["quit"]:
            root.after(80, poll)

    # ----------------------------------------------------------- controls ---
    ctl = tk.Frame(root, bg="#f4f4f6"); ctl.pack(fill="x", pady=6)

    def toggle():
        state["run"] = not state["run"]
        btn.config(text="⏸  Pause" if state["run"] else "▶  Train")

    def reset():
        state["run"] = False; btn.config(text="▶  Train")
        with lock:
            trainer["t"] = Trainer(data, lr=a.lr, batch=a.batch,
                                     seed=np.random.randint(9999))
        axL.clear(); axA.clear(); cfig.draw_idle()
        for k in range(N_TILES):
            tc.itemconfig(rects[k], outline="#cccccc")
            tc.itemconfig(texts[k], fill="#666", text=f"?  (true {tile_true[k]})")
        info.config(text="reset — press ▶ Train")

    def test_now():
        with lock:
            p = trainer["t"].predict(data["Xte"])
        acc = float((p.argmax(1) == data["yte"]).float().mean())
        info.config(text=info.cget("text") + f"\nTEST accuracy {acc:.4f}  "
                                             f"({len(data['yte'])} photos)")

    btn = tk.Button(ctl, text="▶  Train", command=toggle, width=12,
                    font=("DejaVu Sans", 11, "bold"))
    btn.pack(side="left", padx=8)
    tk.Button(ctl, text="Reset (new init)", command=reset,
              font=("DejaVu Sans", 10)).pack(side="left", padx=4)
    tk.Button(ctl, text="Score the TEST set (once!)", command=test_now,
              font=("DejaVu Sans", 10)).pack(side="left", padx=4)
    tk.Label(ctl, bg="#f4f4f6", fg="#555", font=("DejaVu Sans", 9),
             text="   ~2.5 s per epoch on your GPU. 8–15 epochs is plenty; "
                  "watch the tiles go green."
             ).pack(side="left", padx=10)

    def on_close():
        state["quit"] = True; state["run"] = False
        root.after(120, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)
    poll()
    if a.autostart:
        toggle()
    if a.screenshot_after:
        root.after(int(a.screenshot_after * 1000), on_close)
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="run the RGB / resolution / dropout ablations from the docs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--autostart", action="store_true")
    ap.add_argument("--screenshot-after", type=float, default=0)
    x = ap.parse_args()
    if x.sweep:
        run_sweep(x)
    elif x.headless:
        run_headless(x)
    else:
        run_gui(x)
