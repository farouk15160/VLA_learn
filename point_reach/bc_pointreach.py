"""2D PointReach: behavioral cloning from a scripted expert, end to end in one file.

Everything is written by hand -- environment, expert, episode split, normalizer,
MLP, training loop, evaluation -- so no framework hides the concepts.

    observation  o_t = [x_t, y_t, x_g, y_g]
    action       a_t = [dx_t, dy_t]
    dynamics     p_{t+1} = p_t + a_t + eps_t,      eps_t ~ N(0, sigma^2 I)
    expert       a_t* = clip(K (p_g - p_t), -a_max, +a_max)

Two task variants share all of that machinery:

  open field (default)  no obstacles, the expert above exactly. The expert map is
                        piecewise linear, so an MLP nails it *and* extrapolates
                        it -- BC matches the expert, and there is no drift to see.
  obstacle course       --obstacles adds three fixed discs. The expert steers to
                        the tangent point around the first blocking disc, which
                        makes its map nonlinear and switching. Now imitation error
                        matters, and BC visibly drifts. See README.md.

Run:
    .venv/bin/python point_reach/bc_pointreach.py
    .venv/bin/python point_reach/bc_pointreach.py --obstacles --demo-band
    .venv/bin/python point_reach/dagger_pointreach.py      # the DAgger GUI
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Three fixed discs. Fixed, not randomized, because the observation is only
# [x, y, x_g, y_g] -- the policy has to memorize the map from experience rather
# than read obstacle positions out of its input.
OBSTACLE_COURSE: tuple[tuple[tuple[float, float], float], ...] = (
    ((0.00, 0.00), 0.28),
    ((-0.55, 0.55), 0.20),
    ((0.55, -0.55), 0.20),
)


@dataclass
class PointReachConfig:
    """Everything that defines the task. Shared by the BC and DAgger scripts."""

    bounds: float = 1.0            # workspace is [-bounds, +bounds]^2
    a_max: float = 0.10            # per-axis action limit
    noise_std: float = 0.02        # std of eps_t, the process noise
    goal_radius: float = 0.05      # inside this distance the episode is a success
    max_steps: int = 80            # episode horizon
    expert_gain: float = 0.60      # K in the proportional expert
    min_start_dist: float = 0.80   # reject start/goal pairs that are already close

    # Obstacle variant: empty tuple == the plain open-field task.
    obstacles: tuple[tuple[tuple[float, float], float], ...] = ()
    obstacle_margin: float = 0.08  # clearance the expert keeps around each disc

    # Restricts where *starts* are sampled, e.g. (-1.0, -0.6) for a left corridor.
    # This is how limited demonstration coverage is simulated; goals stay uniform.
    start_band: tuple[float, float] | None = None

    @classmethod
    def obstacle_course(cls, **overrides) -> "PointReachConfig":
        return cls(obstacles=OBSTACLE_COURSE, **overrides)

    def replace(self, **changes) -> "PointReachConfig":
        return dataclasses.replace(self, **changes)

    def blocked(self, p) -> bool:
        """True if position p is inside any obstacle disc."""
        return any(np.linalg.norm(np.asarray(p) - np.asarray(c)) < r
                   for c, r in self.obstacles)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


class PointReachEnv:
    """A point mass that must reach a goal. Fully observed, continuous actions.

    The agent commands a displacement directly. The complications are the process
    noise, the workspace walls, and -- in the obstacle variant -- discs that block
    motion: a step that would end inside a disc is rejected and the agent stays put.
    """

    def __init__(self, cfg: PointReachConfig | None = None, seed: int | None = None):
        self.cfg = cfg or PointReachConfig()
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(2)
        self.goal = np.zeros(2)
        self.steps = 0

    # -- core API ----------------------------------------------------------

    def reset(self, pos=None, goal=None) -> np.ndarray:
        """Start a new episode. Pass pos/goal to override the random sampling."""
        if pos is None or goal is None:
            p, g = self.sample_start_goal()
            pos = p if pos is None else np.asarray(pos, dtype=float)
            goal = g if goal is None else np.asarray(goal, dtype=float)
        self.pos = np.asarray(pos, dtype=float).copy()
        self.goal = np.asarray(goal, dtype=float).copy()
        self.steps = 0
        return self.observation()

    def sample_start_goal(self) -> tuple[np.ndarray, np.ndarray]:
        """Rejection-sample a start/goal pair that is legal and non-trivial."""
        cfg, b = self.cfg, self.cfg.bounds
        lo, hi = cfg.start_band if cfg.start_band else (-b, b)
        for _ in range(1000):
            p = np.array([self.rng.uniform(lo, hi), self.rng.uniform(-b, b)])
            g = self.rng.uniform(-b, b, size=2)
            if (np.linalg.norm(g - p) >= cfg.min_start_dist
                    and not cfg.blocked(p) and not cfg.blocked(g)):
                return p, g
        return p, g  # pathological config: return the last draw rather than hang

    def step(self, action) -> tuple[np.ndarray, bool, dict]:
        """Apply one clipped action plus noise. Returns (obs, done, info)."""
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=float), -cfg.a_max, cfg.a_max)
        eps = self.rng.normal(0.0, cfg.noise_std, size=2)
        candidate = np.clip(self.pos + a + eps, -cfg.bounds, cfg.bounds)
        if not cfg.blocked(candidate):        # a blocked step wastes the timestep
            self.pos = candidate
        self.steps += 1
        dist = self.distance()
        success = dist < cfg.goal_radius
        done = success or self.steps >= cfg.max_steps
        return self.observation(), done, {"distance": dist, "success": success}

    # -- helpers -----------------------------------------------------------

    def observation(self) -> np.ndarray:
        return np.concatenate([self.pos, self.goal])

    def distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.pos))

    def expert_action(self) -> np.ndarray:
        return expert_action(self.observation(), self.cfg)


# --------------------------------------------------------------------------
# The scripted expert
# --------------------------------------------------------------------------


def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def expert_waypoint(p, g, cfg: PointReachConfig) -> np.ndarray:
    """What the expert steers at: the goal, or a tangent point around a disc.

    With no obstacles this is just the goal, and the expert below reduces to the
    plain proportional law. With obstacles: if the straight segment p->g clips an
    inflated disc, aim at the tangent point on the side away from the disc centre.
    The left/right choice is a discrete switch -- that is what makes the expert's
    state->action map nonlinear, and what a cloned policy can get wrong.
    """
    p, g = np.asarray(p, float), np.asarray(g, float)
    blocking = None
    for c, r in cfg.obstacles:
        c = np.asarray(c, float)
        R = r + cfg.obstacle_margin
        d = c - p
        L = float(np.linalg.norm(d))
        if L < R:                                   # already inside the margin ring
            u = (p - c) / (L + 1e-9)
            return c + u * R * 1.05                 # step straight back out
        v = g - p
        vl = float(np.linalg.norm(v)) + 1e-9
        u = v / vl
        t = float(np.clip(np.dot(c - p, u), 0.0, vl))   # closest approach along path
        if t > 0 and np.linalg.norm(p + u * t - c) < R:
            if blocking is None or t < blocking[0]:      # nearest one wins
                blocking = (t, c, R, L, d)
    if blocking is None:
        return g
    _, c, R, L, d = blocking
    theta = float(np.arcsin(min(R / L, 1.0)))
    side = -np.sign(_cross(g - p, c - p)) or 1.0        # pass on the far side
    ct, st = np.cos(side * theta), np.sin(side * theta)
    direction = np.array([ct * d[0] - st * d[1], st * d[0] + ct * d[1]]) / L
    return p + direction * float(np.sqrt(max(L * L - R * R, 1e-6)))


def expert_action(obs, cfg: PointReachConfig) -> np.ndarray:
    """The scripted expert: clipped proportional control toward its waypoint.

        a* = clip(K (target - p), -a_max, +a_max)

    With cfg.obstacles empty, target == the goal and this is exactly the expert
    from the task statement. This function is also the DAgger labelling oracle:
    it answers "what would I have done here?" for *any* state, including states
    the expert itself would never visit.
    """
    obs = np.asarray(obs, dtype=float)
    p, g = obs[..., :2], obs[..., 2:4]
    target = expert_waypoint(p, g, cfg) if cfg.obstacles else g
    return np.clip(cfg.expert_gain * (target - p), -cfg.a_max, cfg.a_max)


def expert_policy(cfg: PointReachConfig):
    """The expert as a `policy(obs) -> action` callable."""
    return lambda obs: expert_action(obs, cfg)


# --------------------------------------------------------------------------
# Data collection: whole episodes, kept whole
# --------------------------------------------------------------------------


@dataclass
class Episode:
    """One complete trajectory. Kept intact so the split can be episode-wise."""

    obs: np.ndarray                       # (T, 4)  states visited
    act: np.ndarray                       # (T, 2)  labels stored for those states
    positions: np.ndarray                 # (T+1, 2) path, including the final point
    goal: np.ndarray                      # (2,)
    success: bool = False
    final_distance: float = float("nan")

    def __len__(self) -> int:
        return len(self.obs)


def rollout(env: PointReachEnv, policy, record_expert: bool = False) -> Episode:
    """Run one episode under `policy`. `policy(obs) -> action`.

    The caller must have reset the environment already -- that is what lets the
    same start/goal pair be replayed under several different policies.

    With record_expert=True the stored actions are the *expert's* labels for the
    visited states rather than the actions actually executed. That single flag is
    the whole difference between collecting demonstrations and DAgger relabelling.
    """
    obs = env.observation()   # the caller decides where the episode starts
    obs_list, act_list, pos_list = [], [], [env.pos.copy()]
    done, info = False, {"distance": env.distance(), "success": False}
    while not done:
        action = np.asarray(policy(obs), dtype=float)
        label = expert_action(obs, env.cfg) if record_expert else action
        obs_list.append(obs.copy())
        act_list.append(label)
        obs, done, info = env.step(action)
        pos_list.append(env.pos.copy())
    return Episode(
        obs=np.asarray(obs_list, dtype=np.float32).reshape(-1, 4),
        act=np.asarray(act_list, dtype=np.float32).reshape(-1, 2),
        positions=np.asarray(pos_list, dtype=np.float32),
        goal=env.goal.copy(),
        success=bool(info["success"]),
        final_distance=float(info["distance"]),
    )


def collect_episodes(n: int, cfg: PointReachConfig, policy=None, seed: int = 0,
                     record_expert: bool = False) -> list[Episode]:
    """Run n fresh episodes under `policy` (default: the expert)."""
    env = PointReachEnv(cfg, seed=seed)
    policy = policy or expert_policy(cfg)
    out = []
    for _ in range(n):
        env.reset()
        out.append(rollout(env, policy, record_expert=record_expert))
    return out


def collect_expert_episodes(n: int, cfg: PointReachConfig, seed: int = 0) -> list[Episode]:
    """Generate n complete expert demonstrations."""
    return collect_episodes(n, cfg, policy=expert_policy(cfg), seed=seed)


def split_by_episode(episodes: list[Episode], seed: int = 0,
                     fractions=(0.70, 0.15, 0.15)) -> tuple[list[Episode], ...]:
    """70/15/15 split at the *episode* level.

    Splitting per transition would leak: consecutive states within one trajectory
    are near-duplicates, so train and test would overlap almost perfectly and the
    test MSE would be meaningless.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(episodes))
    n_train = int(round(fractions[0] * len(episodes)))
    n_val = int(round(fractions[1] * len(episodes)))
    chunks = (idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:])
    return tuple([episodes[i] for i in chunk] for chunk in chunks)


def flatten(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate episodes into (N, 4) observations and (N, 2) actions."""
    if not episodes:
        return np.zeros((0, 4), np.float32), np.zeros((0, 2), np.float32)
    return (np.concatenate([e.obs for e in episodes]),
            np.concatenate([e.act for e in episodes]))


# --------------------------------------------------------------------------
# Normalization -- fitted on the training split only
# --------------------------------------------------------------------------


class Normalizer:
    """Zero-mean / unit-std scaling for observations and actions.

    Fitted on training data only. Using validation or test statistics would leak
    information about held-out episodes into the model's inputs and targets.
    """

    def __init__(self, obs: np.ndarray, act: np.ndarray, eps: float = 1e-6):
        self.obs_mean, self.obs_std = obs.mean(0), obs.std(0) + eps
        self.act_mean, self.act_std = act.mean(0), act.std(0) + eps

    def norm_obs(self, o):
        return ((np.asarray(o, np.float32) - self.obs_mean) / self.obs_std).astype(np.float32)

    def norm_act(self, a):
        return ((np.asarray(a, np.float32) - self.act_mean) / self.act_std).astype(np.float32)

    def denorm_act(self, a):
        return (np.asarray(a, np.float32) * self.act_std + self.act_mean).astype(np.float32)

    def state_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in vars(self).items()}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, np.asarray(v, dtype=np.float32))


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


class MLPPolicy(nn.Module):
    """A deliberately small network: 4 -> hidden -> hidden -> 2, plain regression."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def act(self, obs, norm: Normalizer, cfg: PointReachConfig) -> np.ndarray:
        """Numpy in, numpy out: normalize, forward, denormalize, clip."""
        was_training = self.training
        self.eval()
        x = torch.as_tensor(norm.norm_obs(obs)).reshape(1, 4)
        a = norm.denorm_act(self(x).numpy().reshape(2))
        if was_training:
            self.train()
        return np.clip(a, -cfg.a_max, cfg.a_max)


def make_policy_fn(model: MLPPolicy, norm: Normalizer, cfg: PointReachConfig):
    """Wrap a model into the `policy(obs) -> action` callable rollout() expects."""
    return lambda obs: model.act(obs, norm, cfg)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def train_bc(model: MLPPolicy, norm: Normalizer,
             train: tuple[np.ndarray, np.ndarray],
             val: tuple[np.ndarray, np.ndarray],
             epochs: int = 150, batch_size: int = 256, lr: float = 1e-3,
             seed: int = 0, optimizer: torch.optim.Optimizer | None = None,
             on_epoch=None, should_stop=None) -> TrainHistory:
    """Plain supervised regression of normalized action on normalized observation.

    Loss is mean squared error -- the maximum-likelihood objective for a Gaussian
    around the expert's action, which is the standard behavioral-cloning choice.

    on_epoch(epoch, train_loss, val_loss) fires after each epoch, and should_stop()
    is polled between epochs so a GUI can pause or interrupt training cleanly.
    Pass `optimizer` to keep Adam's moment estimates across DAgger iterations.
    """
    torch.manual_seed(seed)
    x = torch.as_tensor(norm.norm_obs(train[0]))
    y = torch.as_tensor(norm.norm_act(train[1]))
    xv = torch.as_tensor(norm.norm_obs(val[0]))
    yv = torch.as_tensor(norm.norm_act(val[1]))

    opt = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = TrainHistory()
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        if should_stop is not None and should_stop():
            break
        model.train()
        perm = torch.randperm(len(x), generator=generator)
        total = 0.0
        for start in range(0, len(x), batch_size):
            batch = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        train_loss = total / max(len(x), 1)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item() if len(xv) else float("nan")

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        if on_epoch is not None:
            on_epoch(epoch, train_loss, val_loss)
    return history


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@torch.no_grad()
def action_mse(model: MLPPolicy, norm: Normalizer,
               data: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    """Open-loop error on held-out transitions: (normalized MSE, raw MSE).

    This is the *supervised* metric. It says nothing about what happens once the
    policy's own errors start choosing which states it sees next.
    """
    model.eval()
    obs, act = data
    if len(obs) == 0:
        return float("nan"), float("nan")
    pred_n = model(torch.as_tensor(norm.norm_obs(obs))).numpy()
    mse_norm = float(((pred_n - norm.norm_act(act)) ** 2).mean())
    mse_raw = float(((norm.denorm_act(pred_n) - act) ** 2).mean())
    return mse_norm, mse_raw


@dataclass
class RolloutStats:
    success_rate: float
    mean_final_distance: float
    median_final_distance: float
    mean_steps: float
    episodes: list[Episode]


def evaluate_closed_loop(policy, cfg: PointReachConfig, n_episodes: int = 200,
                         seed: int = 1234) -> RolloutStats:
    """Actually drive the environment with the policy and measure what happens."""
    eps = collect_episodes(n_episodes, cfg, policy=policy, seed=seed)
    dists = np.array([e.final_distance for e in eps])
    return RolloutStats(
        success_rate=float(np.mean([e.success for e in eps])),
        mean_final_distance=float(dists.mean()),
        median_final_distance=float(np.median(dists)),
        mean_steps=float(np.mean([len(e) for e in eps])),
        episodes=eps,
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def draw_obstacles(ax, cfg: PointReachConfig) -> None:
    from matplotlib.patches import Circle
    for c, r in cfg.obstacles:
        ax.add_patch(Circle(c, r, color="0.35", zorder=3))
        ax.add_patch(Circle(c, r + cfg.obstacle_margin, fill=False,
                            ls=":", ec="0.6", zorder=3))


def plot_results(history: TrainHistory, expert_eps: list[Episode],
                 learner_eps: list[Episode], cfg: PointReachConfig,
                 show: bool = True) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(history.train_loss, label="train")
    ax.plot(history.val_loss, label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalized units)")
    ax.set_title("BC training curve")
    ax.legend()
    ax.grid(alpha=0.3)

    for ax, eps, title, color in (
        (axes[1], expert_eps, "Expert rollouts", "tab:green"),
        (axes[2], learner_eps, "Learned-policy rollouts", "tab:blue"),
    ):
        for e in eps[:40]:
            ax.plot(e.positions[:, 0], e.positions[:, 1], color=color, alpha=0.5, lw=1.0)
            ax.plot(*e.positions[0], "o", color="black", ms=3)
            ax.plot(*e.goal, "*", color="tab:red", ms=9)
        draw_obstacles(ax, cfg)
        rate = np.mean([e.success for e in eps]) if eps else float("nan")
        ax.set_title(f"{title}\nsuccess {rate:.0%}")
        ax.set_xlim(-cfg.bounds * 1.05, cfg.bounds * 1.05)
        ax.set_ylim(-cfg.bounds * 1.05, cfg.bounds * 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Behavioral cloning on 2D PointReach")
    p.add_argument("--episodes", type=int, default=800, help="expert demos to collect (500-1000)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--obstacles", action="store_true",
                   help="use the obstacle course (nonlinear expert; BC then drifts)")
    p.add_argument("--demo-band", action="store_true",
                   help="demos start only in the left corridor, deployment starts anywhere")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gui", action="store_true", help="print metrics without opening plots")
    args = p.parse_args()

    # Deployment config: what the policy is graded on.
    cfg = (PointReachConfig.obstacle_course(noise_std=args.noise_std)
           if args.obstacles else PointReachConfig(noise_std=args.noise_std))
    # Demonstration config: may cover only part of the state space.
    demo_cfg = cfg.replace(start_band=(-1.0, -0.6)) if args.demo_band else cfg

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Task: {'obstacle course' if args.obstacles else 'open field'}"
          f"{', demos restricted to the left corridor' if args.demo_band else ''}")
    print(f"Collecting {args.episodes} expert episodes ...")
    episodes = collect_expert_episodes(args.episodes, demo_cfg, seed=args.seed)
    train_eps, val_eps, test_eps = split_by_episode(episodes, seed=args.seed)
    train, val, test = flatten(train_eps), flatten(val_eps), flatten(test_eps)
    print(f"  episodes  train/val/test = {len(train_eps)}/{len(val_eps)}/{len(test_eps)}")
    print(f"  transitions              = {len(train[0])}/{len(val[0])}/{len(test[0])}")
    print(f"  expert success rate      = {np.mean([e.success for e in episodes]):.1%}")

    norm = Normalizer(*train)                     # training statistics only
    model = MLPPolicy(hidden=args.hidden)

    print(f"Training for {args.epochs} epochs ...")
    history = train_bc(model, norm, train, val, epochs=args.epochs,
                       batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    print(f"  final train / val loss   = {history.train_loss[-1]:.5f} / {history.val_loss[-1]:.5f}")

    mse_norm, mse_raw = action_mse(model, norm, test)
    learner = evaluate_closed_loop(make_policy_fn(model, norm, cfg), cfg,
                                   args.eval_episodes, seed=args.seed + 1234)
    expert = evaluate_closed_loop(expert_policy(cfg), cfg,
                                  args.eval_episodes, seed=args.seed + 1234)

    print("\n" + "=" * 64)
    print("RESULTS" + ("  (evaluated on the full state space)" if args.demo_band else ""))
    print("=" * 64)
    print(f"Test-set action MSE (normalized)  {mse_norm:.6f}")
    print(f"Test-set action MSE (raw units)   {mse_raw:.3e}"
          f"   -> RMS {np.sqrt(mse_raw):.5f} vs a_max {cfg.a_max}")
    print(f"{'':34}{'expert':>10}{'cloned':>12}")
    print(f"{'Closed-loop success rate':34}{expert.success_rate:>10.1%}{learner.success_rate:>12.1%}")
    print(f"{'Mean final distance':34}{expert.mean_final_distance:>10.4f}"
          f"{learner.mean_final_distance:>12.4f}")
    print(f"{'Median final distance':34}{expert.median_final_distance:>10.4f}"
          f"{learner.median_final_distance:>12.4f}")
    print(f"{'Mean steps per episode':34}{expert.mean_steps:>10.1f}{learner.mean_steps:>12.1f}")
    print("=" * 64)
    gap = expert.success_rate - learner.success_rate
    if gap > 0.02:
        print("Per-step imitation looks excellent, yet the rollout is measurably worse:")
        print("small residuals push the agent off the expert's state distribution, where")
        print("its predictions were never trained. That is compounding error, and it is")
        print("exactly what DAgger fixes -- see dagger_pointreach.py.")
    else:
        print("Here BC essentially matches the expert. In the open field the expert map")
        print("clip(K(g-p)) is piecewise linear, so a ReLU MLP reproduces *and*")
        print("extrapolates it, and there is no drift to observe. Re-run with")
        print("--obstacles --demo-band to make the expert nonlinear and the coverage")
        print("partial; the compounding-error gap then appears.")

    plot_results(history, expert.episodes, learner.episodes, cfg, not args.no_gui)


if __name__ == "__main__":
    main()
