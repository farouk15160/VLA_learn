"""DAgger on 2D PointReach, with a Tk GUI: live training, stats, and interaction.

Behavioral cloning learns the expert's *actions*; DAgger learns the expert's
*corrections*. The loop is:

    D <- expert demonstrations                       (iteration 0 = plain BC)
    repeat:
        pi <- argmin_pi  E_{(o,a) in D} ||pi(o) - a||^2       # retrain
        roll pi out in the environment                        # its own states
        D <- D  U  {(o, expert(o)) : o visited by pi}         # expert relabels
                                                              # -- the action pi
                                                              # took is thrown away

Every state the learner stumbles into gets a label saying what the expert would
have done there, so the policy learns to recover from its own mistakes instead of
only replaying perfect trajectories.

The environment, expert, policy and trainer are imported from bc_pointreach.py.

    .venv/bin/python point_reach/dagger_pointreach.py              # live GUI
"""

from __future__ import annotations

import argparse
import copy
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bc_pointreach import (  # noqa: E402
    Episode, MLPPolicy, Normalizer, PointReachConfig, PointReachEnv,
    action_mse, draw_obstacles, expert_action, expert_policy, flatten,
    make_policy_fn, rollout, split_by_episode, train_bc,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class DaggerConfig:
    """Knobs for the DAgger run. Every one of these is editable in the GUI."""

    n_demos: int = 500            # expert demonstrations for iteration 0 (pure BC)
    iterations: int = 8           # DAgger iterations
    epochs_per_iter: int = 60     # gradient epochs per iteration
    rollouts_per_iter: int = 60   # learner episodes collected and relabelled
    eval_episodes: int = 100      # episodes used to score each iteration
    batch_size: int = 256
    lr: float = 1e-3
    hidden: int = 128
    seed: int = 0
    beta_decay: float = 0.0       # P(execute expert) = beta_decay^i during collection
    obstacles: bool = True        # obstacle course: the regime where BC actually fails
    demo_band: bool = True        # demos start only in the left corridor
    noise_std: float = 0.02       # execution noise, applied every step
    start_perturb: float = 0.0    # extra jitter added to the initial position

    def env_cfg(self) -> PointReachConfig:
        """Deployment config: the full state space the policy is graded on."""
        base = (PointReachConfig.obstacle_course(noise_std=self.noise_std)
                if self.obstacles else PointReachConfig(noise_std=self.noise_std))
        return base

    def demo_cfg(self) -> PointReachConfig:
        """Demonstration config: possibly only a slice of the state space.

        This is the covariate-shift lever. With demo_band on, the expert only ever
        starts in the left corridor, so BC never sees what to do elsewhere -- and
        the deployment distribution covers the whole arena.
        """
        cfg = self.env_cfg()
        return cfg.replace(start_band=(-1.0, -0.6)) if self.demo_band else cfg


# --------------------------------------------------------------------------
# Helpers for matched comparisons
# --------------------------------------------------------------------------


def sample_pairs(cfg: PointReachConfig, n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Draw n (start, goal) pairs, so several policies can be run on the same task."""
    env = PointReachEnv(cfg, seed=seed)
    return [env.sample_start_goal() for _ in range(n)]


def rollout_pairs(cfg: PointReachConfig, policy, pairs, seed: int,
                  record_expert: bool = False, perturb: float = 0.0,
                  on_episode=None) -> list[Episode]:
    """Roll `policy` out once from every (start, goal) pair."""
    env = PointReachEnv(cfg, seed=seed)
    jitter = np.random.default_rng(seed + 1)
    out = []
    for p, g in pairs:
        start = p + jitter.normal(0, perturb, 2) if perturb > 0 else p
        start = np.clip(start, -cfg.bounds, cfg.bounds)
        if cfg.blocked(start):
            start = p
        env.reset(pos=start, goal=g)
        ep = rollout(env, policy, record_expert=record_expert)
        out.append(ep)
        if on_episode is not None:
            on_episode(ep)
    return out


def mixed_policy(learner, cfg: PointReachConfig, beta: float, rng):
    """DAgger's beta-mixture: execute the expert with probability beta.

    beta = 0 (the default here) is pure DAgger -- the learner drives, the expert
    only comments. A positive beta keeps early iterations closer to the expert's
    distribution, which can stabilise the first couple of rounds.
    """
    if beta <= 0:
        return learner
    return lambda obs: (expert_action(obs, cfg) if rng.random() < beta else learner(obs))


def stats_of(episodes: list[Episode]) -> tuple[float, float, float]:
    """(success rate, mean final distance, mean steps) for a batch of episodes."""
    if not episodes:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean([e.success for e in episodes])),
            float(np.mean([e.final_distance for e in episodes])),
            float(np.mean([len(e) for e in episodes])))


# --------------------------------------------------------------------------
# The DAgger engine is independent of the Tk rendering layer.
# --------------------------------------------------------------------------


@dataclass
class IterStats:
    iteration: int
    samples: int              # cumulative dataset size after this iteration
    success: float
    distance: float
    steps: float
    test_mse: float           # held-out *expert* transitions: the supervised metric


@dataclass
class DaggerState:
    """Everything the run owns. Survives Stop, so training can be resumed."""

    cfg: DaggerConfig
    model: MLPPolicy | None = None
    norm: Normalizer | None = None
    optimizer: torch.optim.Optimizer | None = None
    train_x: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    train_y: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    val: tuple = field(default_factory=lambda: (np.zeros((0, 4), np.float32),
                                                np.zeros((0, 2), np.float32)))
    test: tuple = field(default_factory=lambda: (np.zeros((0, 4), np.float32),
                                                 np.zeros((0, 2), np.float32)))
    demo_states: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    dagger_states: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    demo_starts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    dagger_starts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    iteration: int = 0
    epoch_curve: list[tuple[int, float, float]] = field(default_factory=list)
    iter_marks: list[int] = field(default_factory=list)
    history: list[IterStats] = field(default_factory=list)
    bc_state_dict: dict | None = None       # frozen copy of the iteration-0 policy
    bc_episodes: list[Episode] = field(default_factory=list)
    last_episodes: list[Episode] = field(default_factory=list)
    expert_stats: tuple | None = None

    def policy(self):
        return make_policy_fn(self.model, self.norm, self.cfg.env_cfg())

    def bc_policy(self):
        """The frozen iteration-0 policy: plain behavioral cloning, for comparison."""
        if self.bc_state_dict is None:
            return None
        model = MLPPolicy(self.cfg.hidden)
        model.load_state_dict(self.bc_state_dict)
        model.eval()
        return make_policy_fn(model, self.norm, self.cfg.env_cfg())


class DaggerEngine:
    """Runs the DAgger loop, reporting progress through `emit(kind, **payload)`."""

    def __init__(self, cfg: DaggerConfig, emit, should_stop=None, wait_if_paused=None,
                 model_lock: threading.Lock | None = None):
        self.state = DaggerState(cfg=cfg)
        self.emit = emit
        self.should_stop = should_stop or (lambda: False)
        self.wait_if_paused = wait_if_paused or (lambda: None)
        self.model_lock = model_lock or threading.Lock()

    # -- phase 1: expert demonstrations ------------------------------------

    def collect_demonstrations(self) -> None:
        s, cfg = self.state, self.state.cfg
        demo_cfg = cfg.demo_cfg()
        self.emit("status", text=f"Collecting {cfg.n_demos} expert demonstrations ...")
        pairs = sample_pairs(demo_cfg, cfg.n_demos, cfg.seed)
        episodes = rollout_pairs(demo_cfg, expert_policy(demo_cfg), pairs, seed=cfg.seed)

        # Episode-level 70/15/15 split; DAgger only ever grows the train split.
        tr, va, te = split_by_episode(episodes, seed=cfg.seed)
        s.train_x, s.train_y = flatten(tr)
        s.val, s.test = flatten(va), flatten(te)
        s.demo_states = s.train_x.copy()
        s.demo_starts = np.array([e.positions[0] for e in tr], np.float32)
        s.norm = Normalizer(s.train_x, s.train_y)      # training statistics only
        s.model = MLPPolicy(cfg.hidden)
        s.optimizer = torch.optim.Adam(s.model.parameters(), lr=cfg.lr)

        succ, dist, steps = stats_of(episodes)
        s.expert_stats = (succ, dist, steps)
        self.emit("demos", n_episodes=len(episodes), n_train=len(s.train_x),
                  n_val=len(s.val[0]), n_test=len(s.test[0]),
                  expert_success=succ, expert_distance=dist,
                  demo_states=s.demo_states.copy(), episodes=episodes[:40])

    # -- phase 2: one DAgger iteration -------------------------------------

    def train_once(self) -> None:
        """Retrain on the whole aggregated dataset (DAgger trains from scratch data,
        but warm-starting the weights is standard practice and much faster)."""
        s, cfg = self.state, self.state.cfg
        base_epoch = len(s.epoch_curve)
        s.iter_marks.append(base_epoch)

        def on_epoch(epoch, tr_loss, va_loss):
            self.wait_if_paused()
            s.epoch_curve.append((base_epoch + epoch, tr_loss, va_loss))
            self.emit("epoch", iteration=s.iteration, epoch=epoch,
                      train_loss=tr_loss, val_loss=va_loss,
                      global_epoch=base_epoch + epoch)

        self.emit("status", text=f"Iteration {s.iteration}: training on "
                                 f"{len(s.train_x):,} samples ...")
        with self.model_lock:
            train_bc(s.model, s.norm, (s.train_x, s.train_y), s.val,
                     epochs=cfg.epochs_per_iter, batch_size=cfg.batch_size,
                     lr=cfg.lr, seed=cfg.seed + s.iteration,
                     optimizer=s.optimizer, on_epoch=on_epoch,
                     should_stop=self.should_stop)

    def evaluate(self) -> IterStats:
        s, cfg = self.state, self.state.cfg
        env_cfg = cfg.env_cfg()
        self.emit("status", text=f"Iteration {s.iteration}: evaluating "
                                 f"{cfg.eval_episodes} rollouts ...")
        pairs = sample_pairs(env_cfg, cfg.eval_episodes, seed=10_000 + s.iteration)
        with self.model_lock:
            policy = s.policy()
            eps = rollout_pairs(env_cfg, policy, pairs, seed=20_000 + s.iteration,
                                perturb=cfg.start_perturb)
            mse = action_mse(s.model, s.norm, s.test)[1]
        succ, dist, steps = stats_of(eps)
        st = IterStats(s.iteration, len(s.train_x), succ, dist, steps, mse)
        s.history.append(st)
        s.last_episodes = eps[:40]
        if s.iteration == 0:                    # freeze the pure-BC policy
            s.bc_state_dict = copy.deepcopy(s.model.state_dict())
            s.bc_episodes = eps[:40]
        self.emit("iteration", stats=st, episodes=s.last_episodes,
                  is_bc=(s.iteration == 0))
        return st

    def aggregate(self) -> None:
        """Roll the learner out, ask the expert to label every state it visited."""
        s, cfg = self.state, self.state.cfg
        env_cfg = cfg.env_cfg()
        beta = cfg.beta_decay ** s.iteration if cfg.beta_decay > 0 else 0.0
        self.emit("status", text=f"Iteration {s.iteration}: collecting "
                                 f"{cfg.rollouts_per_iter} learner rollouts and "
                                 f"relabelling them with the expert (beta={beta:.2f}) ...")
        pairs = sample_pairs(env_cfg, cfg.rollouts_per_iter, seed=30_000 + s.iteration)
        rng = np.random.default_rng(40_000 + s.iteration)
        with self.model_lock:
            behave = mixed_policy(s.policy(), env_cfg, beta, rng)
            new = rollout_pairs(env_cfg, behave, pairs, seed=50_000 + s.iteration,
                                record_expert=True, perturb=cfg.start_perturb)
        nx, ny = flatten(new)
        s.train_x = np.concatenate([s.train_x, nx])
        s.train_y = np.concatenate([s.train_y, ny])
        s.dagger_states = np.concatenate([s.dagger_states, nx])
        s.dagger_starts = np.concatenate(
            [s.dagger_starts, np.array([e.positions[0] for e in new], np.float32)])
        self.emit("aggregate", added=len(nx), total=len(s.train_x),
                  dagger_states=s.dagger_states.copy(), episodes=new[:20])

    # -- the loop ----------------------------------------------------------

    def run(self) -> None:
        s, cfg = self.state, self.state.cfg
        if s.model is None:
            self.collect_demonstrations()
        while s.iteration < cfg.iterations and not self.should_stop():
            self.wait_if_paused()
            self.train_once()
            if self.should_stop():
                break
            self.evaluate()
            if s.iteration + 1 < cfg.iterations:
                self.aggregate()
            s.iteration += 1
        self.emit("finished", stopped=self.should_stop(), iteration=s.iteration)


# --------------------------------------------------------------------------
# Worker thread: pause / resume / stop
# --------------------------------------------------------------------------


class DaggerWorker(threading.Thread):
    def __init__(self, engine: DaggerEngine, stop_event, pause_event):
        super().__init__(daemon=True)
        self.engine = engine
        self.stop_event = stop_event
        self.pause_event = pause_event

    def run(self) -> None:
        try:
            self.engine.run()
        except Exception as exc:                       # surface it in the GUI log
            self.engine.emit("error", text=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Plotting -- shared by the GUI tabs and the headless figures
# --------------------------------------------------------------------------

ARENA_KW = dict(aspect="equal")


def setup_arena(ax, cfg: PointReachConfig, title: str = "") -> None:
    ax.clear()
    draw_obstacles(ax, cfg)
    ax.set_xlim(-cfg.bounds * 1.05, cfg.bounds * 1.05)
    ax.set_ylim(-cfg.bounds * 1.05, cfg.bounds * 1.05)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)


def draw_episodes(ax, eps: list[Episode], cfg: PointReachConfig, limit: int = 30,
                  color_by_success: bool = True, color: str = "tab:blue") -> None:
    for e in eps[:limit]:
        c = ("tab:green" if e.success else "tab:red") if color_by_success else color
        ax.plot(e.positions[:, 0], e.positions[:, 1], color=c, lw=1.0, alpha=0.6)
        ax.plot(*e.positions[0], "o", color="black", ms=3)
        ax.plot(*e.goal, "*", color="tab:orange", ms=8)


def _integer_xaxis(ax) -> None:
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_progress(axes, state: DaggerState) -> None:
    """Success rate, final distance, dataset size, and test MSE versus iteration."""
    hist = state.history
    it = [h.iteration for h in hist]
    ax = axes[0]
    ax.clear()
    if hist:
        ax.plot(it, [h.success for h in hist], "o-", color="tab:blue", label="DAgger")
        ax.axhline(hist[0].success, ls="--", color="tab:red",
                   label=f"BC (iter 0) = {hist[0].success:.0%}")
    if state.expert_stats:
        ax.axhline(state.expert_stats[0], ls=":", color="tab:green", label="expert")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("success rate")
    ax.set_title("Success rate vs iteration", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.clear()
    if hist:
        ax.plot(it, [h.distance for h in hist], "o-", color="tab:purple", label="DAgger")
        ax.axhline(hist[0].distance, ls="--", color="tab:red", label="BC (iter 0)")
    if state.expert_stats:
        ax.axhline(state.expert_stats[1], ls=":", color="tab:green", label="expert")
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("mean final distance")
    ax.set_title("Final distance vs iteration", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.clear()
    if hist:
        ax.bar(it, [h.samples for h in hist], color="tab:cyan", edgecolor="0.3")
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("transitions in D")
    ax.set_title("Collected samples", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[3]
    ax.clear()
    if hist:
        ax.plot(it, [h.test_mse for h in hist], "o-", color="tab:brown")
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("action MSE (raw)")
    ax.set_title("Held-out expert action MSE\n(barely moves -- that is the point)",
                 fontsize=9)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    for ax in axes[:4]:
        _integer_xaxis(ax)


def plot_coverage(axes, state: DaggerState, cfg: PointReachConfig) -> None:
    """Which states each dataset actually contains -- the covariate-shift picture."""
    for ax, states, starts, title, color in (
        (axes[0], state.demo_states, state.demo_starts,
         "BC data: expert demonstrations", "tab:green"),
        (axes[1], state.dagger_states, state.dagger_starts,
         "DAgger data: states the learner visited", "tab:blue"),
    ):
        setup_arena(ax, cfg, title)
        if len(states):
            alpha = float(np.clip(4000.0 / len(states), 0.05, 0.45))
            ax.scatter(states[:, 0], states[:, 1], s=4, alpha=alpha, color=color,
                       edgecolors="none", zorder=2, label="visited states")
        if len(starts):
            ax.scatter(starts[:, 0], starts[:, 1], s=9, color="black", alpha=0.6,
                       edgecolors="none", zorder=4, label="episode starts")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
        ax.set_title(f"{title}\n{len(states):,} states", fontsize=9)


def plot_trajectories(axes, expert_eps, bc_eps, dagger_eps, cfg: PointReachConfig) -> None:
    for ax, eps, name in ((axes[0], expert_eps, "Expert"),
                          (axes[1], bc_eps, "BC (iteration 0)"),
                          (axes[2], dagger_eps, "DAgger (final)")):
        setup_arena(ax, cfg)
        draw_episodes(ax, eps or [], cfg, limit=25)
        rate = np.mean([e.success for e in eps]) if eps else float("nan")
        ax.set_title(f"{name}\nsuccess {rate:.0%} (green ok / red fail)", fontsize=9)


def plot_losses(ax, state: DaggerState) -> None:
    ax.clear()
    if state.epoch_curve:
        curve = np.asarray(state.epoch_curve, dtype=float)
        ax.plot(curve[:, 0], curve[:, 1], lw=1.0, label="train")
        ax.plot(curve[:, 0], curve[:, 2], lw=1.0, label="validation")
        for m in state.iter_marks:
            ax.axvline(m, color="0.8", lw=0.8, zorder=0)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax.set_xlabel("epoch (vertical lines = new DAgger iteration)", fontsize=8)
    ax.set_ylabel("MSE (normalized)", fontsize=8)
    ax.set_title("Live training loss", fontsize=10)
    ax.grid(alpha=0.3)


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------


def launch_gui(cfg: DaggerConfig) -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    class App(tk.Tk):
        def __init__(self, cfg: DaggerConfig):
            super().__init__()
            self.title("DAgger on 2D PointReach")
            # Never open larger than the screen -- the figures resize with the window.
            w = min(1480, self.winfo_screenwidth() - 60)
            h = min(900, self.winfo_screenheight() - 90)
            self.geometry(f"{w}x{h}+20+20")
            self.minsize(900, 620)

            self.cfg = cfg
            self.events: queue.Queue = queue.Queue()
            self.engine: DaggerEngine | None = None
            self.worker: DaggerWorker | None = None
            self.stop_event = threading.Event()
            self.pause_event = threading.Event()       # set == running
            self.pause_event.set()
            self.model_lock = threading.Lock()
            self.start_time = None
            self.last_draw = 0.0
            self.test_job = None
            self.test_runs: list[dict] = []
            self.test_start = np.array([-0.85, -0.85])
            self.test_goal = np.array([0.85, 0.85])

            self._build_controls()
            self._build_tabs()
            self._set_buttons(running=False)
            self.after(60, self._drain)
            self.protocol("WM_DELETE_WINDOW", self._on_close)
            self._log("Ready. Press Start to collect demonstrations and train.")
            self._log("Iteration 0 is plain behavioral cloning; 1+ are DAgger.")

        # -- layout --------------------------------------------------------

        def _build_controls(self):
            panel = ttk.Frame(self, padding=8)
            panel.pack(side="left", fill="y")

            ttk.Label(panel, text="DAgger — PointReach",
                      font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

            box = ttk.LabelFrame(panel, text="Run", padding=6)
            box.pack(fill="x", pady=6)
            self.vars = {}
            spec = [
                ("n_demos", "Expert demos", 50, 3000, 50),
                ("iterations", "DAgger iterations", 1, 40, 1),
                ("epochs_per_iter", "Epochs / iteration", 5, 500, 5),
                ("rollouts_per_iter", "Rollouts / iteration", 5, 500, 5),
                ("eval_episodes", "Eval episodes", 20, 500, 10),
                ("hidden", "Hidden units", 16, 512, 16),
                ("seed", "Seed", 0, 9999, 1),
            ]
            for i, (key, label, lo, hi, step) in enumerate(spec):
                ttk.Label(box, text=label).grid(row=i, column=0, sticky="w", pady=1)
                var = tk.IntVar(value=getattr(cfg, key))
                ttk.Spinbox(box, from_=lo, to=hi, increment=step, textvariable=var,
                            width=8).grid(row=i, column=1, sticky="e")
                self.vars[key] = var
            row = len(spec)
            for key, label, lo, hi, step in [("noise_std", "Execution noise", 0.0, 0.2, 0.005),
                                             ("start_perturb", "Start perturbation", 0.0, 0.5, 0.01),
                                             ("beta_decay", "Expert mix beta", 0.0, 0.99, 0.05)]:
                ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", pady=1)
                var = tk.DoubleVar(value=getattr(cfg, key))
                ttk.Spinbox(box, from_=lo, to=hi, increment=step, textvariable=var,
                            width=8, format="%.3f").grid(row=row, column=1, sticky="e")
                self.vars[key] = var
                row += 1
            self.vars["obstacles"] = tk.BooleanVar(value=cfg.obstacles)
            ttk.Checkbutton(box, text="Obstacle course", variable=self.vars["obstacles"]
                            ).grid(row=row, column=0, columnspan=2, sticky="w")
            self.vars["demo_band"] = tk.BooleanVar(value=cfg.demo_band)
            ttk.Checkbutton(box, text="Demos only from left corridor",
                            variable=self.vars["demo_band"]
                            ).grid(row=row + 1, column=0, columnspan=2, sticky="w")

            btns = ttk.Frame(panel)
            btns.pack(fill="x", pady=4)
            self.btn_start = ttk.Button(btns, text="Start", command=self.on_start)
            self.btn_pause = ttk.Button(btns, text="Pause", command=self.on_pause)
            self.btn_stop = ttk.Button(btns, text="Stop", command=self.on_stop)
            self.btn_reset = ttk.Button(btns, text="Reset", command=self.on_reset)
            for i, b in enumerate((self.btn_start, self.btn_pause, self.btn_stop, self.btn_reset)):
                b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
            btns.columnconfigure(0, weight=1)
            btns.columnconfigure(1, weight=1)

            io = ttk.Frame(panel)
            io.pack(fill="x")
            ttk.Button(io, text="Save checkpoint", command=self.on_save).grid(
                row=0, column=0, sticky="ew", padx=2)
            ttk.Button(io, text="Load", command=self.on_load).grid(
                row=0, column=1, sticky="ew", padx=2)
            io.columnconfigure(0, weight=1)
            io.columnconfigure(1, weight=1)

            stats = ttk.LabelFrame(panel, text="Live stats", padding=6)
            stats.pack(fill="x", pady=8)
            self.stat_vars = {}
            for key, label in [("phase", "Phase"), ("iter", "Iteration"), ("epoch", "Epoch"),
                               ("train", "Train loss"), ("val", "Val loss"),
                               ("samples", "Dataset"), ("succ", "Success"),
                               ("dist", "Final dist"), ("mse", "Test MSE"),
                               ("bc", "BC baseline"), ("exp", "Expert"),
                               ("time", "Elapsed")]:
                r = ttk.Frame(stats)
                r.pack(fill="x")
                ttk.Label(r, text=label, width=12).pack(side="left")
                v = tk.StringVar(value="–")
                ttk.Label(r, textvariable=v, width=22, anchor="w",
                          font=("TkFixedFont", 9)).pack(side="left", fill="x", expand=True)
                self.stat_vars[key] = v

            ttk.Label(panel, text="Log").pack(anchor="w")
            self.log = tk.Text(panel, height=12, width=42, wrap="word",
                               font=("TkFixedFont", 8))
            self.log.pack(fill="both", expand=True)

        def _build_tabs(self):
            nb = ttk.Notebook(self)
            nb.pack(side="right", fill="both", expand=True, padx=6, pady=6)
            self.nb = nb

            def add_tab(name, nrows, ncols, figsize):
                frame = ttk.Frame(nb)
                nb.add(frame, text=name)
                fig = Figure(figsize=figsize, dpi=100, layout="constrained")
                axes = fig.subplots(nrows, ncols)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                return frame, fig, np.atleast_1d(np.asarray(axes)).ravel(), canvas

            _, self.fig_live, self.ax_live, self.canvas_live = add_tab("Live", 1, 2, (9.4, 4.2))
            _, self.fig_prog, self.ax_prog, self.canvas_prog = add_tab("Progress", 2, 2, (9.0, 6.0))
            _, self.fig_cov, self.ax_cov, self.canvas_cov = add_tab("State distribution", 1, 2, (9.4, 4.4))
            _, self.fig_traj, self.ax_traj, self.canvas_traj = add_tab("Trajectories", 1, 3, (10.4, 3.9))

            # -- interactive tab
            frame = ttk.Frame(nb)
            nb.add(frame, text="Interactive test")
            bar = ttk.Frame(frame, padding=4)
            bar.pack(side="top", fill="x")
            ttk.Label(bar, text="Left-click = goal   |   Right-click = start").pack(side="left")
            self.test_which = tk.StringVar(value="both")
            for label, val in (("Expert", "expert"), ("BC", "bc"),
                               ("DAgger", "dagger"), ("All three", "both")):
                ttk.Radiobutton(bar, text=label, value=val,
                                variable=self.test_which).pack(side="left", padx=3)
            ttk.Button(bar, text="Run", command=self.on_test_run).pack(side="left", padx=6)
            ttk.Button(bar, text="Clear", command=self.on_test_clear).pack(side="left")
            self.test_status = tk.StringVar(value="Pick a goal, then press Run.")
            ttk.Label(bar, textvariable=self.test_status).pack(side="left", padx=10)

            self.fig_test = Figure(figsize=(6.0, 5.6), dpi=100, layout="constrained")
            self.ax_test = self.fig_test.add_subplot(111)
            self.canvas_test = FigureCanvasTkAgg(self.fig_test, master=frame)
            self.canvas_test.get_tk_widget().pack(fill="both", expand=True)
            self.canvas_test.mpl_connect("button_press_event", self.on_click)
            self._draw_test_arena()

            for ax in list(self.ax_live) + list(self.ax_prog) + list(self.ax_cov) + list(self.ax_traj):
                ax.grid(alpha=0.3)
            setup_arena(self.ax_live[0], self.cfg.env_cfg(), "Latest evaluation rollouts")
            plot_losses(self.ax_live[1], DaggerState(cfg=self.cfg))
            self.canvas_live.draw_idle()

        # -- control handlers ----------------------------------------------

        def _read_cfg(self) -> DaggerConfig:
            kw = {k: (v.get() if not isinstance(v, tk.BooleanVar) else bool(v.get()))
                  for k, v in self.vars.items()}
            return DaggerConfig(lr=self.cfg.lr, batch_size=self.cfg.batch_size, **kw)

        def on_start(self):
            if self.worker is not None and self.worker.is_alive():
                return
            resuming = self.engine is not None and self.engine.state.model is not None
            if not resuming:
                self.cfg = self._read_cfg()
                self.engine = DaggerEngine(
                    self.cfg, self._emit,
                    should_stop=self.stop_event.is_set,
                    wait_if_paused=self.pause_event.wait,
                    model_lock=self.model_lock)
                self._log("Starting a fresh run.")
            else:
                # Resume: keep the model and dataset, honour a raised iteration count.
                self.engine.state.cfg.iterations = int(self.vars["iterations"].get())
                self.engine.state.cfg.epochs_per_iter = int(self.vars["epochs_per_iter"].get())
                self.engine.state.cfg.rollouts_per_iter = int(self.vars["rollouts_per_iter"].get())
                self._log(f"Resuming at iteration {self.engine.state.iteration} "
                          f"with {len(self.engine.state.train_x):,} samples.")
            self.stop_event.clear()
            self.pause_event.set()
            self.start_time = self.start_time or time.time()
            self.worker = DaggerWorker(self.engine, self.stop_event, self.pause_event)
            self.worker.start()
            self._set_buttons(running=True)

        def on_pause(self):
            if self.pause_event.is_set():
                self.pause_event.clear()
                self.btn_pause.config(text="Resume")
                self._set_stat("phase", "paused")
                self._log("Paused between epochs. The model and data are untouched.")
            else:
                self.pause_event.set()
                self.btn_pause.config(text="Pause")
                self._log("Resumed.")

        def on_stop(self):
            self.stop_event.set()
            self.pause_event.set()
            self._log("Stop requested — finishing the current epoch. "
                      "Press Start to resume from here.")

        def on_reset(self):
            self.on_stop()
            self.engine = None
            self.worker = None
            self.start_time = None
            self.test_runs = []
            for v in self.stat_vars.values():
                v.set("–")
            for ax in list(self.ax_prog) + list(self.ax_cov) + list(self.ax_traj):
                ax.clear()
            setup_arena(self.ax_live[0], self._read_cfg().env_cfg(), "Latest evaluation rollouts")
            plot_losses(self.ax_live[1], DaggerState(cfg=self._read_cfg()))
            for c in (self.canvas_live, self.canvas_prog, self.canvas_cov, self.canvas_traj):
                c.draw_idle()
            self._draw_test_arena()
            self._log("Reset. All learned state discarded.")
            self._set_buttons(running=False)

        def on_save(self):
            if self.engine is None or self.engine.state.model is None:
                self._log("Nothing to save yet.")
                return
            path = filedialog.asksaveasfilename(defaultextension=".pt",
                                                initialfile="dagger_pointreach.pt")
            if not path:
                return
            s = self.engine.state
            with self.model_lock:
                torch.save({"model": s.model.state_dict(), "norm": s.norm.state_dict(),
                            "bc": s.bc_state_dict, "train_x": s.train_x,
                            "train_y": s.train_y, "val": s.val, "test": s.test,
                            "demo_states": s.demo_states, "dagger_states": s.dagger_states,
                            "iteration": s.iteration, "history": s.history,
                            "epoch_curve": s.epoch_curve, "iter_marks": s.iter_marks,
                            "expert_stats": s.expert_stats, "cfg": s.cfg}, path)
            self._log(f"Saved checkpoint to {path}")

        def on_load(self):
            path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt")])
            if not path:
                return
            ck = torch.load(path, weights_only=False)
            self.cfg = ck["cfg"]
            self.engine = DaggerEngine(self.cfg, self._emit,
                                       should_stop=self.stop_event.is_set,
                                       wait_if_paused=self.pause_event.wait,
                                       model_lock=self.model_lock)
            s = self.engine.state
            s.model = MLPPolicy(self.cfg.hidden)
            s.model.load_state_dict(ck["model"])
            s.norm = Normalizer(np.zeros((1, 4), np.float32), np.zeros((1, 2), np.float32))
            s.norm.load_state_dict(ck["norm"])
            s.optimizer = torch.optim.Adam(s.model.parameters(), lr=self.cfg.lr)
            for key in ("bc_state_dict", "train_x", "train_y", "val", "test",
                        "demo_states", "dagger_states", "iteration", "history",
                        "epoch_curve", "iter_marks", "expert_stats"):
                setattr(s, key, ck["bc" if key == "bc_state_dict" else key])
            self._log(f"Loaded {path}: iteration {s.iteration}, "
                      f"{len(s.train_x):,} samples. Press Start to continue.")
            self._refresh_all()
            self._set_buttons(running=False)

        def _set_buttons(self, running: bool):
            self.btn_start.config(state="disabled" if running else "normal")
            self.btn_pause.config(state="normal" if running else "disabled", text="Pause")
            self.btn_stop.config(state="normal" if running else "disabled")

        # -- event pump ----------------------------------------------------

        def _emit(self, kind, **payload):
            self.events.put((kind, payload))

        def _drain(self):
            redraw = set()
            try:
                while True:
                    kind, p = self.events.get_nowait()
                    redraw |= self._handle(kind, p)
            except queue.Empty:
                pass
            now = time.time()
            if redraw and now - self.last_draw > 0.25:
                self.last_draw = now
                for c in redraw:
                    c.draw_idle()
            if self.start_time and self.worker and self.worker.is_alive():
                self._set_stat("time", f"{now - self.start_time:6.1f} s")
            self.after(60, self._drain)

        def _handle(self, kind, p) -> set:
            s = self.engine.state if self.engine else None
            if kind == "status":
                self._set_stat("phase", p["text"].split(":")[-1].strip()[:28])
                self._log(p["text"])
            elif kind == "error":
                self._log("ERROR: " + p["text"])
                self._set_buttons(running=False)
            elif kind == "demos":
                self._set_stat("samples", f"{p['n_train']:,}")
                self._set_stat("exp", f"{p['expert_success']:.0%} / {p['expert_distance']:.3f}")
                self._log(f"Demos: {p['n_episodes']} episodes -> "
                          f"{p['n_train']:,} train / {p['n_val']:,} val / "
                          f"{p['n_test']:,} test transitions (split by episode).")
                plot_coverage(self.ax_cov, s, self.cfg.env_cfg())
                return {self.canvas_cov}
            elif kind == "epoch":
                self._set_stat("iter", str(p["iteration"]))
                self._set_stat("epoch", str(p["epoch"]))
                self._set_stat("train", f"{p['train_loss']:.5f}")
                self._set_stat("val", f"{p['val_loss']:.5f}")
                if p["epoch"] % 5 == 0:
                    plot_losses(self.ax_live[1], s)
                    return {self.canvas_live}
            elif kind == "iteration":
                st = p["stats"]
                self._set_stat("succ", f"{st.success:.1%}")
                self._set_stat("dist", f"{st.distance:.3f}")
                self._set_stat("mse", f"{st.test_mse:.2e}")
                if p["is_bc"]:
                    self._set_stat("bc", f"{st.success:.0%} / {st.distance:.3f}")
                self._log(f"iter {st.iteration}: success {st.success:.1%}, "
                          f"dist {st.distance:.3f}, MSE {st.test_mse:.2e}, "
                          f"|D| {st.samples:,}")
                setup_arena(self.ax_live[0], self.cfg.env_cfg(),
                            f"Iteration {st.iteration} rollouts — "
                            f"success {st.success:.0%}")
                draw_episodes(self.ax_live[0], p["episodes"], self.cfg.env_cfg())
                plot_progress(self.ax_prog, s)
                plot_losses(self.ax_live[1], s)
                self._refresh_trajectories()
                return {self.canvas_live, self.canvas_prog, self.canvas_traj}
            elif kind == "aggregate":
                self._set_stat("samples", f"{p['total']:,}")
                self._log(f"  aggregated {p['added']:,} expert-labelled states "
                          f"from the learner's own rollouts (|D| = {p['total']:,})")
                plot_coverage(self.ax_cov, s, self.cfg.env_cfg())
                return {self.canvas_cov}
            elif kind == "finished":
                self._set_buttons(running=False)
                self._set_stat("phase", "stopped" if p["stopped"] else "finished")
                self._log("Stopped. Press Start to resume." if p["stopped"]
                          else "Run finished. Try the Interactive test tab.")
                self._refresh_all()
                return {self.canvas_live, self.canvas_prog, self.canvas_cov, self.canvas_traj}
            return set()

        # -- redraw helpers --------------------------------------------------

        def _refresh_trajectories(self):
            s = self.engine.state
            if not s.history:
                return
            env_cfg = self.cfg.env_cfg()
            pairs = sample_pairs(env_cfg, 20, seed=777)
            expert_eps = rollout_pairs(env_cfg, expert_policy(env_cfg), pairs, seed=778)
            bc = s.bc_policy()
            with self.model_lock:
                bc_eps = rollout_pairs(env_cfg, bc, pairs, seed=778) if bc else []
                dagger_eps = rollout_pairs(env_cfg, s.policy(), pairs, seed=778)
            plot_trajectories(self.ax_traj, expert_eps, bc_eps, dagger_eps, env_cfg)

        def _refresh_all(self):
            if self.engine is None:
                return
            s = self.engine.state
            plot_progress(self.ax_prog, s)
            plot_coverage(self.ax_cov, s, self.cfg.env_cfg())
            plot_losses(self.ax_live[1], s)
            self._refresh_trajectories()
            for c in (self.canvas_live, self.canvas_prog, self.canvas_cov, self.canvas_traj):
                c.draw_idle()

        def _set_stat(self, key, value):
            self.stat_vars[key].set(value)

        def _log(self, text):
            self.log.insert("end", text + "\n")
            self.log.see("end")

        # -- interactive tab ---------------------------------------------------

        def _draw_test_arena(self):
            cfg = self.cfg.env_cfg() if self.engine else self._read_cfg().env_cfg()
            setup_arena(self.ax_test, cfg, "Click to place the goal, then Run")
            self.ax_test.plot(*self.test_start, "o", color="black", ms=8, label="start")
            self.ax_test.plot(*self.test_goal, "*", color="tab:orange", ms=16, label="goal")
            self.ax_test.add_artist(
                __import__("matplotlib").patches.Circle(
                    self.test_goal, cfg.goal_radius, fill=False, ec="tab:orange", ls="--"))
            self.ax_test.legend(loc="upper left", fontsize=8)
            self.canvas_test.draw_idle()

        def on_click(self, event):
            if event.inaxes is not self.ax_test or event.xdata is None:
                return
            cfg = self.cfg.env_cfg()
            p = np.array([event.xdata, event.ydata])
            if cfg.blocked(p):
                self.test_status.set("That point is inside an obstacle.")
                return
            if event.button == 3:
                self.test_start = p
            else:
                self.test_goal = p
            self.test_runs = []
            self._draw_test_arena()
            self.test_status.set(f"start {self.test_start.round(2)} "
                                 f"-> goal {self.test_goal.round(2)}. Press Run.")

        def on_test_clear(self):
            self.test_runs = []
            self._draw_test_arena()

        def on_test_run(self):
            if self.test_job is not None:
                self.after_cancel(self.test_job)
                self.test_job = None
            cfg = self.cfg.env_cfg()
            which = self.test_which.get()
            wanted = ["expert", "bc", "dagger"] if which == "both" else [which]
            runs = []
            with self.model_lock:
                bc = self.engine.state.bc_policy() if self.engine else None
                dagger = (self.engine.state.policy()
                          if self.engine and self.engine.state.model else None)
                # Snapshot the weights so the animation is unaffected by training.
                if dagger is not None:
                    m = copy.deepcopy(self.engine.state.model)
                    m.eval()
                    dagger = make_policy_fn(m, self.engine.state.norm, cfg)
            for name, pol, color in (("expert", expert_policy(cfg), "tab:green"),
                                     ("bc", bc, "tab:red"),
                                     ("dagger", dagger, "tab:blue")):
                if name in wanted and pol is not None:
                    env = PointReachEnv(cfg, seed=int(time.time()) % 10000)
                    env.reset(pos=self.test_start, goal=self.test_goal)
                    runs.append({"name": name, "env": env, "policy": pol, "color": color,
                                 "path": [env.pos.copy()], "done": False,
                                 "info": {"distance": env.distance(), "success": False}})
            if not runs:
                self.test_status.set("That policy does not exist yet — train first.")
                return
            self.test_runs = runs
            self._draw_test_arena()
            self._animate_test()

        def _animate_test(self):
            active = False
            for r in self.test_runs:
                if r["done"]:
                    continue
                obs = r["env"].observation()
                _, done, info = r["env"].step(r["policy"](obs))
                r["path"].append(r["env"].pos.copy())
                r["info"] = info
                r["done"] = done
                active = True
            self._draw_test_arena()
            for r in self.test_runs:
                path = np.asarray(r["path"])
                self.ax_test.plot(path[:, 0], path[:, 1], color=r["color"], lw=1.8,
                                  label=r["name"])
                self.ax_test.plot(*path[-1], "o", color=r["color"], ms=6)
            self.ax_test.legend(loc="upper left", fontsize=8)
            self.canvas_test.draw_idle()
            self.test_status.set("   ".join(
                f"{r['name']}: d={r['info']['distance']:.3f} "
                f"({len(r['path']) - 1} steps)"
                f"{' OK' if r['info']['success'] else ''}" for r in self.test_runs))
            if active:
                self.test_job = self.after(40, self._animate_test)
            else:
                self.test_job = None

        def _on_close(self):
            self.stop_event.set()
            self.pause_event.set()
            self.destroy()

    App(cfg).mainloop()


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="DAgger on 2D PointReach (GUI by default)")
    p.add_argument("--demos", type=int, default=DaggerConfig.n_demos)
    p.add_argument("--iterations", type=int, default=DaggerConfig.iterations)
    p.add_argument("--epochs", type=int, default=DaggerConfig.epochs_per_iter)
    p.add_argument("--rollouts", type=int, default=DaggerConfig.rollouts_per_iter)
    p.add_argument("--eval-episodes", type=int, default=DaggerConfig.eval_episodes)
    p.add_argument("--noise-std", type=float, default=DaggerConfig.noise_std)
    p.add_argument("--start-perturb", type=float, default=DaggerConfig.start_perturb)
    p.add_argument("--beta-decay", type=float, default=DaggerConfig.beta_decay)
    p.add_argument("--no-obstacles", action="store_true")
    p.add_argument("--no-demo-band", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = DaggerConfig(
        n_demos=args.demos, iterations=args.iterations, epochs_per_iter=args.epochs,
        rollouts_per_iter=args.rollouts, eval_episodes=args.eval_episodes,
        noise_std=args.noise_std, start_perturb=args.start_perturb,
        beta_decay=args.beta_decay, obstacles=not args.no_obstacles,
        demo_band=not args.no_demo_band, seed=args.seed)

    launch_gui(cfg)


if __name__ == "__main__":
    main()
