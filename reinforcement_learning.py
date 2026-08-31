"""
REINFORCEMENT LEARNING — a car learns to drive from A to B.
================================================================================
Run:
    .venv/bin/python reinforcement_learning.py              # GUI
    .venv/bin/python reinforcement_learning.py --headless   # terminal only

Full write-up: docs/reinforcement_learning.md
--------------------------------------------------------------------------------

THE PROBLEM, STATED AS AN MDP
    Reinforcement learning problems are written as a Markov Decision Process,
    the 5-tuple (S, A, P, R, gamma). Here is this file's, exactly:

    S  STATE / observation .... 10 floats, all scaled to roughly [-1, 1]:
         [0:7]  seven lidar ray distances / RAY_MAX  -> how far the wall is in
                seven directions spanning 135 degrees ahead of the car
         [7]    cos(bearing to goal - heading)
         [8]    sin(bearing to goal - heading)
                ^ the goal direction RELATIVE to where the car points. Split
                  into cos/sin instead of a raw angle so there is no
                  discontinuity when the angle wraps past +/-pi.
         [9]    distance to goal / diagonal of the map

         Note what is NOT in here: the car's absolute (x, y). The policy is
         deliberately given only what a real robot's sensors would provide, so
         it must learn to drive by what it sees, not by memorizing coordinates.

    A  ACTIONS ............... 3 discrete choices, every step:
         0 = steer left  (heading -= STEER)
         1 = go straight
         2 = steer right (heading += STEER)
         Speed is CONSTANT. The car cannot brake or reverse; it can only turn.

    P  TRANSITION ............ heading += (a-1)*STEER, then the car moves
         SPEED units along its new heading. Deterministic. The agent never
         sees this function and cannot differentiate through it -- which is
         precisely why we need a policy gradient instead of backprop.

    R  REWARD ................ per step:
         + 0.10 * (previous distance to goal - current distance)   <- progress
         - 0.002                                                   <- time cost
         - 3.00  and end the episode, if it hits a wall            <- crash
         + 5.00  and end the episode, if it gets within GOAL_R of B

         Why shaped and not just "+1 for reaching B"? Because a sparse reward
         is unlearnable here: random steering essentially never reaches B, so
         the gradient would be zero forever. The progress term is a POTENTIAL-
         BASED shaping (a difference of distances), which is the safe kind --
         it cannot be farmed by driving in circles, because moving away costs
         exactly what coming back pays.

         Why is the crash penalty as large as 3.0? My first version used 1.0,
         and a car that crashed 20 units closer to B still netted +0.9 overall.
         "Drive straight into the wall" was a viable local optimum. Reward-design
         bugs masquerade as agent stupidity.

    gamma  DISCOUNT .......... 0.99

    Episode ends on crash, on arrival, or after MAX_STEPS (truncation).

THE ALGORITHM
    REINFORCE with a learned value baseline -- the policy gradient

        grad J = E[ sum_t  grad log pi(a_t|s_t) * A_t ],   A_t = G_t - V(s_t)

    Policy network : 10 -> 128 -> 128 -> 3   (tanh, categorical output)
    Value network  : 10 -> 128 -> 128 -> 1   (tanh, scalar; trained by MSE
                     regression onto the observed returns -- ordinary
                     supervised learning, sitting inside an RL loop)
    Optimiser      : Adam, lr 3e-3 (policy) / 1e-2 (value)
    Batch          : 8 episodes per gradient step, advantages normalized
    Runs on CPU on purpose: the bottleneck is the Python physics loop, not
    matrix maths, so shipping 10 floats to a GPU each step costs more than it
    saves.

WHAT YOU SEE IN THE GUI
    left   the map, replaying one full episode, with lidar beams and a live bar
           chart of the policy's action probabilities
    right  V(s) as a heatmap, plus white arrows showing the direction the policy
           would actually steer from each point -- the learned behaviour, drawn
    bottom success rate, mean return, policy entropy, per update
"""
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------- world ------
W, H = 100.0, 60.0                     # world size in arbitrary units
START = np.array([10.0, 30.0])
GOAL = np.array([90.0, 30.0])
GOAL_R = 5.0                           # how close counts as "arrived"
CAR_R = 1.6                            # car radius, for collision

# Axis-aligned obstacles as (x0, y0, x1, y1). Two staggered walls force an
# S-curve: the car must go OVER the first and UNDER the second. A straight line
# from A to B is blocked, so the policy cannot succeed by driving straight.
OBSTACLES = np.array([
    [30.0,  0.0, 38.0, 40.0],
    [60.0, 20.0, 68.0, 60.0],
], dtype=np.float32)

N_RAYS = 7
RAY_SPREAD = np.pi * 0.75              # total fan angle (135 deg)
RAY_MAX = 35.0
SPEED = 1.6
STEER = 0.30                           # radians per step
MAX_STEPS = 400
OBS_DIM = N_RAYS + 3
N_ACT = 3                              # 0 = left, 1 = straight, 2 = right


def _walls_inflated(r):
    """Obstacles grown by the car radius, so the car can be treated as a point."""
    o = OBSTACLES.copy()
    o[:, 0] -= r; o[:, 1] -= r; o[:, 2] += r; o[:, 3] += r
    return o


def raycast(origins, angles, boxes):
    """Vectorized ray vs axis-aligned-box distance, plus the world boundary.

    origins (P,2), angles (P,R) -> distances (P,R). Uses the slab method:
    for each axis compute the interval of t where the ray is inside the box;
    the ray hits iff those intervals overlap at a positive t.
    """
    P, R = angles.shape
    d = np.stack([np.cos(angles), np.sin(angles)], axis=-1)      # (P,R,2)
    o = origins[:, None, :]                                      # (P,1,2)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d                                            # (P,R,2)
        lo = (boxes[None, None, :, 0:2] - o[..., None, :]) * inv[:, :, None, :]
        hi = (boxes[None, None, :, 2:4] - o[..., None, :]) * inv[:, :, None, :]
        t1 = np.minimum(lo, hi)                                  # (P,R,B,2)
        t2 = np.maximum(lo, hi)
        t_enter = np.nanmax(t1, axis=-1)
        t_exit = np.nanmin(t2, axis=-1)
        hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 0.0)
        t_box = np.where(hit, np.where(t_enter > 0, t_enter, 0.0), np.inf)
        t_box = t_box.min(axis=-1)                               # (P,R)

        # world boundary: distance to the [0,W]x[0,H] rectangle from inside
        tx = np.where(d[..., 0] > 0, (W - o[..., 0]) / d[..., 0],
                      np.where(d[..., 0] < 0, (0 - o[..., 0]) / d[..., 0], np.inf))
        ty = np.where(d[..., 1] > 0, (H - o[..., 1]) / d[..., 1],
                      np.where(d[..., 1] < 0, (0 - o[..., 1]) / d[..., 1], np.inf))
        t_wall = np.minimum(tx, ty)

    return np.minimum(np.minimum(t_box, t_wall), RAY_MAX)


class CarEnv:
    """A tiny driving MDP. Same interface shape as a Gymnasium env."""

    def __init__(self, seed=0, randomize_start=True):
        self.rng = np.random.RandomState(seed)
        self.boxes = _walls_inflated(CAR_R)
        self.randomize_start = randomize_start
        self.diag = float(np.hypot(W, H))

    # -- observation ---------------------------------------------------------
    def _lidar(self, pos, th):
        offs = np.linspace(-RAY_SPREAD / 2, RAY_SPREAD / 2, N_RAYS)
        return raycast(pos[None, :], (th + offs)[None, :], self.boxes)[0]

    def _obs(self):
        to_goal = GOAL - self.pos
        dist = float(np.linalg.norm(to_goal))
        bearing = np.arctan2(to_goal[1], to_goal[0]) - self.th
        return np.concatenate([
            self.lidar / RAY_MAX,          # 7 normalized wall distances
            [np.cos(bearing), np.sin(bearing)],   # goal direction, relative to
                                                  # where the car is pointing.
                                                  # cos/sin instead of the raw
                                                  # angle so there is no wrap
                                                  # discontinuity at +/-pi.
            [dist / self.diag],            # normalized range to goal
        ]).astype(np.float32)

    # -- dynamics ------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.pos = START.copy()
        self.th = 0.0
        if self.randomize_start:
            # Jitter the start so the policy cannot memorize one trajectory.
            # This is the cheapest form of domain randomization, and the same
            # idea used when training robot policies in simulation.
            self.pos = self.pos + self.rng.uniform(-4, 4, size=2)
            self.th = self.rng.uniform(-0.5, 0.5)
        self.t = 0
        self.lidar = self._lidar(self.pos, self.th)
        self.prev_d = float(np.linalg.norm(GOAL - self.pos))
        self.done_reason = None
        return self._obs()

    def _crashed(self):
        x, y = self.pos
        if x < CAR_R or x > W - CAR_R or y < CAR_R or y > H - CAR_R:
            return True
        b = self.boxes
        return bool(np.any((x > b[:, 0]) & (x < b[:, 2]) &
                           (y > b[:, 1]) & (y < b[:, 3])))

    def step(self, a):
        self.th += (a - 1) * STEER          # 0 -> -STEER, 1 -> 0, 2 -> +STEER
        self.pos = self.pos + SPEED * np.array([np.cos(self.th), np.sin(self.th)])
        self.t += 1
        self.lidar = self._lidar(self.pos, self.th)

        d = float(np.linalg.norm(GOAL - self.pos))

        # --- REWARD DESIGN, and why it looks like this -----------------------
        # A pure "+1 for reaching B" reward is SPARSE: random steering will
        # essentially never reach B, so the gradient is zero forever and nothing
        # is learned (problem 3.5 in the RL doc). So we SHAPE the reward with a
        # dense progress term: every step, the reward is how much closer to the
        # goal we got. Shaping as a difference of distances (a "potential") is
        # the safe kind -- it cannot be farmed by driving in circles, because
        # going away costs exactly what coming back pays.
        r = 0.10 * (self.prev_d - d)        # progress
        r -= 0.002                          # small time penalty: prefer short routes
        self.prev_d = d

        terminated = False
        if self._crashed():
            # Crash penalty must OUTWEIGH the progress already banked, or
            # "drive straight into the wall" becomes a viable local optimum:
            # at -1.0 a car that crashes 20 units closer to B still nets +0.9.
            r -= 3.0
            terminated = True
            self.done_reason = "crash"
        elif d < GOAL_R:
            r += 5.0
            terminated = True
            self.done_reason = "goal"

        truncated = self.t >= MAX_STEPS
        if truncated and not terminated:
            self.done_reason = "timeout"
        return self._obs(), r, terminated, truncated


# ---------------------------------------------------------------- nets -------
class Policy(nn.Module):
    def __init__(self, obs=OBS_DIM, hid=128, act=N_ACT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(),
                                 nn.Linear(hid, act))

    def forward(self, x):
        return self.net(x)

    def dist(self, x):
        return torch.distributions.Categorical(logits=self.net(x))


class ValueNet(nn.Module):
    def __init__(self, obs=OBS_DIM, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(),
                                 nn.Linear(hid, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def returns_to_go(rew, gamma):
    out = np.empty(len(rew), dtype=np.float32)
    run = 0.0
    for t in reversed(range(len(rew))):
        run = rew[t] + gamma * run
        out[t] = run
    return out


def grid_cache(nx, ny, _c={}):
    """Precomputed observations on a grid, heading fixed toward the goal.

    The map never changes and the heading is pinned, so each cell's observation
    is a constant. Build it once; then every refresh of the heatmap or the
    arrow field is a single batched forward pass instead of thousands of
    raycasts.
    """
    key = (nx, ny)
    if key not in _c:
        xs = np.linspace(2, W - 2, nx)
        ys = np.linspace(2, H - 2, ny)
        gx, gy = np.meshgrid(xs, ys)
        pos = np.stack([gx.ravel(), gy.ravel()], axis=1)
        to_goal = GOAL[None, :] - pos
        th = np.arctan2(to_goal[:, 1], to_goal[:, 0])
        offs = np.linspace(-RAY_SPREAD / 2, RAY_SPREAD / 2, N_RAYS)
        lid = raycast(pos, th[:, None] + offs[None, :], _walls_inflated(CAR_R))
        dist = np.linalg.norm(to_goal, axis=1)
        obs = np.concatenate([lid / RAY_MAX,
                              np.ones((len(pos), 1)),      # cos(bearing) = 1
                              np.zeros((len(pos), 1)),     # sin(bearing) = 0
                              (dist / np.hypot(W, H))[:, None]], axis=1)
        inside = np.zeros(len(pos), bool)
        for (x0, y0, x1, y1) in _walls_inflated(0.0):
            inside |= ((pos[:, 0] > x0) & (pos[:, 0] < x1) &
                       (pos[:, 1] > y0) & (pos[:, 1] < y1))
        _c[key] = {"obs": torch.from_numpy(obs.astype(np.float32)),
                   "shape": (ny, nx), "pos": pos, "th": th, "inside": inside,
                   "gx": gx, "gy": gy}
    return _c[key]


def value_grid(value, nx=60, ny=36):
    """V(s) over the map, with wall interiors masked out (they are unreachable)."""
    c = grid_cache(nx, ny)
    with torch.no_grad():
        v = value(c["obs"]).numpy().astype(float)
    v[c["inside"]] = np.nan
    return v.reshape(c["shape"])


def policy_field(policy, nx=22, ny=13):
    """Where would the car actually steer, from each point on the map?

    At each cell we point the car at the goal, ask the policy for its preferred
    action, apply that steering, and draw the resulting direction. This is the
    learned DRIVING BEHAVIOUR made visible: early on the arrows all point
    straight at B (straight into the walls); once trained they bend around the
    obstacles and trace the route the car actually takes.
    """
    c = grid_cache(nx, ny)
    with torch.no_grad():
        a = policy(c["obs"]).argmax(dim=1).numpy()
    new_th = c["th"] + (a - 1) * STEER
    u = np.cos(new_th); v = np.sin(new_th)
    keep = ~c["inside"]
    return (c["pos"][keep, 0], c["pos"][keep, 1], u[keep], v[keep], a[keep])


# ---------------------------------------------------------------- trainer ----
class Trainer:
    """REINFORCE + value baseline, exposed one update at a time.

    `update()` runs a batch of episodes, does ONE policy-gradient step, and
    returns a snapshot. The GUI calls this from a worker thread and draws the
    snapshot; the headless mode just loops over it. Same code either way.
    """

    def __init__(self, seed=0, gamma=0.99, lr=3e-3, batch_eps=8, hid=128,
                 use_baseline=True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.policy = Policy(hid=hid)
        self.value = ValueNet(hid=hid)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.vopt = torch.optim.Adam(self.value.parameters(), lr=1e-2)
        self.env = CarEnv(seed)
        self.gamma, self.batch_eps = gamma, batch_eps
        self.use_baseline = use_baseline
        self.updates = 0
        self.episodes = 0
        self.hist = {"ret": [], "success": [], "steps": [], "entropy": []}

    def rollout(self, greedy=False, seed=None):
        o = self.env.reset(seed=seed)
        O, A, R = [], [], []
        traj = {"x": [], "y": [], "th": [], "lidar": [], "probs": []}
        while True:
            ot = torch.from_numpy(o)
            with torch.no_grad():
                d = self.policy.dist(ot)
                a = int(d.probs.argmax()) if greedy else int(d.sample())
            traj["x"].append(self.env.pos[0]); traj["y"].append(self.env.pos[1])
            traj["th"].append(self.env.th); traj["lidar"].append(self.env.lidar.copy())
            traj["probs"].append(d.probs.numpy().copy())   # for the live policy bars
            O.append(o); A.append(a)
            o, r, term, trunc = self.env.step(a)
            R.append(r)
            if term or trunc:
                break
        traj["reason"] = self.env.done_reason
        traj["greedy"] = greedy
        traj["ret"] = float(sum(R))
        return np.array(O, np.float32), np.array(A), np.array(R, np.float32), traj

    def update(self):
        logps, coefs, ents, Os, Gs = [], [], [], [], []
        rets, succ, steps = [], [], []
        last = None
        for _ in range(self.batch_eps):
            O, A, R, traj = self.rollout()
            G = returns_to_go(R, self.gamma)
            Ot = torch.from_numpy(O)
            d = self.policy.dist(Ot)
            with torch.no_grad():
                v = self.value(Ot) if self.use_baseline else torch.zeros(len(G))
            logps.append(d.log_prob(torch.from_numpy(A)))
            coefs.append(torch.from_numpy(G) - v)   # advantage (= raw return if
                                                    # the baseline is disabled)
            ents.append(d.entropy())
            Os.append(Ot); Gs.append(torch.from_numpy(G))
            rets.append(traj["ret"]); succ.append(traj["reason"] == "goal")
            steps.append(len(R))
            self.episodes += 1
            last = traj

        logp = torch.cat(logps)
        adv_raw = torch.cat(coefs).detach()
        adv = adv_raw
        # Normalize the advantage. This task's returns vary a lot in scale
        # depending on how far the car got, and unnormalized advantages make the
        # effective step size depend on that. Standard practice in PPO/A2C too.
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = -(logp * adv).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 2.0)
        self.opt.step()

        S, Gt = torch.cat(Os), torch.cat(Gs)
        for _ in range(5 if self.use_baseline else 0):
            vl = nn.functional.mse_loss(self.value(S), Gt)
            self.vopt.zero_grad(); vl.backward(); self.vopt.step()

        self.updates += 1
        self.hist["ret"].append(float(np.mean(rets)))
        self.hist["success"].append(float(np.mean(succ)))
        self.hist["steps"].append(float(np.mean(steps)))
        self.hist["entropy"].append(float(torch.cat(ents).mean().detach()))
        self.hist.setdefault("coef_std", []).append(float(adv_raw.std()))
        return {"update": self.updates, "episodes": self.episodes,
                "ret": self.hist["ret"][-1], "success": self.hist["success"][-1],
                "steps": self.hist["steps"][-1],
                "entropy": self.hist["entropy"][-1],
                "traj": last, "hist": self.hist}


import argparse
import queue
import threading
import time
from pathlib import Path


OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


# ============================================================== headless ======
def run_headless(args):
    t = Trainer(seed=args.seed, lr=args.lr, batch_eps=args.batch_eps,
                use_baseline=not args.no_baseline)
    print(f"{'upd':>5} {'eps':>6} {'return':>8} {'success':>8} {'steps':>7} {'entropy':>8}")
    print("-" * 48)
    t0 = time.time()
    for i in range(1, args.updates + 1):
        s = t.update()
        if i % 10 == 0 or i == 1:
            print(f"{s['update']:>5} {s['episodes']:>6} {s['ret']:>8.2f} "
                  f"{s['success']:>8.2f} {s['steps']:>7.1f} {s['entropy']:>8.3f}")
    print(f"\ntrained in {time.time() - t0:.0f}s")
    wins = sum(t.rollout(greedy=True, seed=9000 + i)[3]["reason"] == "goal"
               for i in range(30))
    print(f"greedy eval: reached the goal in {wins}/30 episodes")


# ============================================================== the GUI =======
def run_gui(args):
    import tkinter as tk
    from tkinter import ttk
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.patches
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    S = 8.0                                   # world units -> pixels
    CW, CH = int(W * S), int(H * S)

    root = tk.Tk()
    root.title("RL — a car learns to drive from A to B (REINFORCE + baseline)")
    root.configure(bg="#f4f4f6")

    state = {"run": False, "snap": None, "frame": 0, "traj": None,
             "pending": None, "quit": False, "fps": args.fps}
    show_lidar = tk.BooleanVar(value=True)
    show_arrows = tk.BooleanVar(value=True)
    watch_greedy = tk.BooleanVar(value=False)
    state["throttle"] = args.throttle
    q = queue.Queue()

    # ---------------------------------------------------------------- layout
    top = tk.Frame(root, bg="#f4f4f6"); top.pack(side="top", fill="both", expand=True)
    canvas = tk.Canvas(top, width=CW, height=CH, bg="#ffffff",
                       highlightthickness=1, highlightbackground="#c9c9d0")
    canvas.pack(side="left", padx=10, pady=10)

    right = tk.Frame(top, bg="#f4f4f6"); right.pack(side="left", fill="both", expand=True)
    figv = Figure(figsize=(5.2, 3.4), dpi=100)
    axv = figv.add_subplot(111)
    cv = FigureCanvasTkAgg(figv, master=right); cv.get_tk_widget().pack(fill="both", expand=True)

    stat = tk.Label(right, text="", font=("DejaVu Sans Mono", 10), justify="left",
                    bg="#f4f4f6", anchor="w")
    stat.pack(fill="x", padx=6)

    figc = Figure(figsize=(13.5, 2.9), dpi=100)
    ax1 = figc.add_subplot(131); ax2 = figc.add_subplot(132); ax3 = figc.add_subplot(133)
    cc = FigureCanvasTkAgg(figc, master=root)
    cc.get_tk_widget().pack(side="top", fill="both", expand=True, padx=10)

    ctl = tk.Frame(root, bg="#f4f4f6"); ctl.pack(side="top", fill="x", pady=6)

    # ------------------------------------------------------------ map drawing
    def wx(x): return x * S
    def wy(y): return (H - y) * S          # flip: screen y grows downward

    def draw_static():
        canvas.delete("static")
        for (x0, y0, x1, y1) in OBSTACLES:
            canvas.create_rectangle(wx(x0), wy(y1), wx(x1), wy(y0),
                                    fill="#4a4a55", outline="#2e2e36",
                                    tags="static")
        gx, gy = GOAL
        canvas.create_oval(wx(gx - GOAL_R), wy(gy + GOAL_R),
                           wx(gx + GOAL_R), wy(gy - GOAL_R),
                           fill="#ffd9d9", outline="#d43d3d", width=2, tags="static")
        canvas.create_text(wx(gx), wy(gy), text="B", font=("DejaVu Sans", 15, "bold"),
                           fill="#d43d3d", tags="static")
        sx, sy = START
        canvas.create_oval(wx(sx - 3), wy(sy + 3), wx(sx + 3), wy(sy - 3),
                           fill="#d9f2d9", outline="#2e9e2e", width=2, tags="static")
        canvas.create_text(wx(sx), wy(sy), text="A", font=("DejaVu Sans", 15, "bold"),
                           fill="#2e9e2e", tags="static")

    def draw_frame():
        tr = state["traj"]
        canvas.delete("dyn")
        if tr is None:
            return
        n = len(tr["x"])
        i = min(state["frame"], n - 1)

        pts = []
        for k in range(i + 1):
            pts += [wx(tr["x"][k]), wy(tr["y"][k])]
        if len(pts) >= 4:
            canvas.create_line(*pts, fill="#8f9fd8", width=2, tags="dyn")

        x, y, th = tr["x"][i], tr["y"][i], tr["th"][i]
        if show_lidar.get():
            offs = np.linspace(-RAY_SPREAD / 2, RAY_SPREAD / 2, N_RAYS)
            for off, d in zip(offs, tr["lidar"][i]):
                a = th + off
                canvas.create_line(wx(x), wy(y), wx(x + d * np.cos(a)),
                                   wy(y + d * np.sin(a)),
                                   fill="#e0b0b0", width=1, tags="dyn")

        L, Wd = 3.4, 2.2                       # car triangle
        pts = [(L, 0), (-L * 0.6, Wd), (-L * 0.6, -Wd)]
        poly = []
        for px, py in pts:
            poly += [wx(x + px * np.cos(th) - py * np.sin(th)),
                     wy(y + px * np.sin(th) + py * np.cos(th))]
        col = {"goal": "#2e9e2e", "crash": "#d43d3d"}.get(
            tr["reason"] if i == n - 1 else None, "#3355bb")
        canvas.create_polygon(*poly, fill=col, outline="#1a1a22", tags="dyn")

        # live policy bars: what the network wanted to do at this instant
        p = tr["probs"][i]
        bx, by, bw, bh = 14, 14, 90, 13
        canvas.create_text(bx, by - 8, text="policy  ←  ↑  →", anchor="w",
                           font=("DejaVu Sans", 9), fill="#333", tags="dyn")
        for k, (lbl, pv) in enumerate(zip(("left", "straight", "right"), p)):
            yy = by + 6 + k * (bh + 3)
            canvas.create_rectangle(bx, yy, bx + bw, yy + bh, outline="#bbb", tags="dyn")
            canvas.create_rectangle(bx, yy, bx + bw * float(pv), yy + bh,
                                    fill="#5b7fd4", outline="", tags="dyn")
            canvas.create_text(bx + bw + 6, yy + bh / 2, anchor="w",
                               text=f"{lbl} {float(pv):.2f}",
                               font=("DejaVu Sans Mono", 8), fill="#333", tags="dyn")

        done = i >= n - 1
        outcome = tr["reason"] if done else "driving…"
        src = ("GREEDY policy" if tr.get("greedy") else
               f"training episode from update {tr.get('update', '?')}")
        canvas.create_text(CW - 10, 14, anchor="e", tags="dyn",
                           font=("DejaVu Sans Mono", 10),
                           fill={"goal": "#2e9e2e", "crash": "#d43d3d"}.get(
                               outcome, "#444"),
                           text=f"step {i + 1}/{n}   {outcome}")
        canvas.create_text(CW - 10, 30, anchor="e", tags="dyn",
                           font=("DejaVu Sans Mono", 9), fill="#777", text=src)

    # ------------------------------------------------------------- worker ----
    trainer = {"t": Trainer(seed=args.seed, lr=args.lr, batch_eps=args.batch_eps,
                            use_baseline=not args.no_baseline)}
    lock = threading.Lock()          # the GUI also runs the policy (greedy replays),
                                     # so guard the weights while they are updated

    def worker():
        while not state["quit"]:
            if not state["run"]:
                time.sleep(0.05); continue
            with lock:
                snap = trainer["t"].update()
                snap["vgrid"] = value_grid(trainer["t"].value)
                snap["field"] = policy_field(trainer["t"].policy)
            snap["traj"]["update"] = snap["update"]
            q.put(snap)
            time.sleep(state["throttle"] / 1000.0)

    th = threading.Thread(target=worker, daemon=True); th.start()

    # -------------------------------------------------------------- plots ----
    def redraw_plots(snap):
        h = snap["hist"]
        axv.clear()
        v = snap["vgrid"]
        # Robust colour limits. Without them the few very negative cells next to
        # the walls stretch the scale and the whole map renders as one flat
        # colour -- the gradient you actually want to see disappears.
        lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        cmap = matplotlib.colormaps["viridis"].copy()   # cm.get_cmap removed in mpl 3.9+
        cmap.set_bad("#2e2e36")                       # wall interiors -> grey
        axv.imshow(v, origin="lower", extent=[0, W, 0, H], aspect="auto",
                   cmap=cmap, vmin=lo, vmax=hi)
        if show_arrows.get() and snap.get("field") is not None:
            fx, fy, fu, fv, _ = snap["field"]
            axv.quiver(fx, fy, fu, fv, color="#ffffff", alpha=.85,
                       scale=32, width=.004, headwidth=4)
        axv.plot(*START, "o", color="#7CFC00", ms=9, mec="k")
        axv.plot(*GOAL, "*", color="#ff4d4d", ms=17, mec="k")
        axv.set_title("V(s) heat + the direction the policy would steer",
                      fontsize=9)
        axv.set_xticks([]); axv.set_yticks([])
        axv.set_xlim(0, W); axv.set_ylim(0, H)
        figv.tight_layout(); cv.draw_idle()

        for a in (ax1, ax2, ax3):
            a.clear(); a.grid(alpha=.3)
        ax1.plot(h["success"], color="#2e9e2e")
        ax1.set_title("success rate (reached B)", fontsize=9); ax1.set_ylim(-.05, 1.05)
        ax2.plot(h["ret"], color="#3355bb")
        ax2.set_title("mean return per episode", fontsize=9)
        ax3.plot(h["entropy"], color="#c46a1e")
        ax3.set_title("policy entropy (exploration)", fontsize=9)
        for a in (ax1, ax2, ax3):
            a.set_xlabel("update", fontsize=8)
        figc.tight_layout(); cc.draw_idle()

    # -------------------------------------------------------------- loops ----
    def poll():
        got = None
        while True:
            try:
                got = q.get_nowait()
            except queue.Empty:
                break
        if got is not None:
            state["snap"] = got
            # Queue it as PENDING rather than swapping it in now. Updates land
            # every ~0.2 s but a replay takes ~1.5 s, so assigning here restarted
            # the animation after ~9 frames and you never saw a full drive.
            state["pending"] = got["traj"]
            redraw_plots(got)
            stat.config(text=(
                f"update {got['update']:>5}   episodes {got['episodes']:>6}\n"
                f"success  {got['success']*100:>5.1f}%   mean return {got['ret']:>7.2f}\n"
                f"avg steps {got['steps']:>6.1f}   entropy {got['entropy']:>6.3f}"))
        if not state["quit"]:
            root.after(60, poll)

    def next_trajectory():
        """Pick what to replay next, once the current replay has finished."""
        if watch_greedy.get():
            with lock:
                return trainer["t"].rollout(greedy=True)[3]
        if state["pending"] is not None:
            tr, state["pending"] = state["pending"], None
            return tr
        return state["traj"]            # nothing new yet: loop the same episode

    HOLD = 30                            # frames to linger on the final frame

    def animate():
        if state["traj"] is None:
            if state["pending"] is not None or watch_greedy.get():
                state["traj"] = next_trajectory(); state["frame"] = 0
        else:
            draw_frame()
            state["frame"] += 1
            if state["frame"] >= len(state["traj"]["x"]) + HOLD:
                state["traj"] = next_trajectory()
                state["frame"] = 0
        if not state["quit"]:
            root.after(int(1000 / max(1, state["fps"])), animate)

    # ------------------------------------------------------------ controls ---
    def toggle():
        state["run"] = not state["run"]
        btn.config(text="⏸  Pause" if state["run"] else "▶  Train")

    def reset():
        state["run"] = False; btn.config(text="▶  Train")
        trainer["t"] = Trainer(seed=np.random.randint(10000),
                                 lr=args.lr, batch_eps=args.batch_eps)
        state["traj"] = None; state["snap"] = None; state["frame"] = 0
        state["pending"] = None
        for a in (ax1, ax2, ax3):
            a.clear()
        axv.clear(); cv.draw_idle(); cc.draw_idle()
        stat.config(text="reset — press Train")

    btn = tk.Button(ctl, text="▶  Train", command=toggle, width=12,
                    font=("DejaVu Sans", 11, "bold"))
    btn.pack(side="left", padx=8)
    tk.Checkbutton(ctl, text="watch greedy policy", variable=watch_greedy,
                   bg="#f4f4f6", font=("DejaVu Sans", 10, "bold")
                   ).pack(side="left", padx=6)
    tk.Button(ctl, text="Reset (new seed)", command=reset,
              font=("DejaVu Sans", 10)).pack(side="left", padx=4)

    tk.Checkbutton(ctl, text="show lidar", variable=show_lidar, bg="#f4f4f6",
                   font=("DejaVu Sans", 10)).pack(side="left", padx=12)
    tk.Checkbutton(ctl, text="policy arrows", variable=show_arrows, bg="#f4f4f6",
                   font=("DejaVu Sans", 10)).pack(side="left", padx=4)

    tk.Label(ctl, text="replay fps", bg="#f4f4f6",
             font=("DejaVu Sans", 10)).pack(side="left")
    sp = tk.Scale(ctl, from_=5, to=120, orient="horizontal", length=170,
                  bg="#f4f4f6", highlightthickness=0,
                  command=lambda v: state.__setitem__("fps", int(float(v))))
    sp.set(args.fps); sp.pack(side="left", padx=4)

    tk.Label(ctl, text="slow training", bg="#f4f4f6",
             font=("DejaVu Sans", 10)).pack(side="left", padx=(12, 0))
    tk.Scale(ctl, from_=0, to=600, orient="horizontal", length=140, resolution=25,
             bg="#f4f4f6", highlightthickness=0,
             command=lambda v: state.__setitem__("throttle", int(float(v)))
             ).pack(side="left", padx=4)

    hint = tk.Label(ctl, bg="#f4f4f6", fg="#555", font=("DejaVu Sans", 9),
                    justify="left", anchor="w")
    hint.pack(side="left", padx=10)
    hint.config(text="every replay now runs to completion before the next one "
                     "starts.\ntick 'watch greedy policy' to see the deterministic "
                     "drive instead of a wobbly exploring one")

    def on_close():
        state["quit"] = True; state["run"] = False
        root.after(120, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)
    draw_static(); poll(); animate()
    stat.config(text="press  ▶ Train  to start learning")
    if args.autostart:
        toggle()
    if args.screenshot_after:
        root.after(int(args.screenshot_after * 1000), on_close)
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-eps", type=int, default=8)
    ap.add_argument("--fps", type=int, default=45)
    ap.add_argument("--no-baseline", action="store_true",
                    help="ablate the value baseline: weight grad-log-pi by the raw "
                         "return instead of the advantage. Run it and compare.")
    ap.add_argument("--throttle", type=int, default=0,
                    help="ms to sleep after each update, to slow training down")
    ap.add_argument("--autostart", action="store_true",
                    help="begin training immediately instead of waiting for the button")
    ap.add_argument("--screenshot-after", type=float, default=0,
                    help="close the window after N seconds (used for automated checks)")
    a = ap.parse_args()
    run_headless(a) if a.headless else run_gui(a)
