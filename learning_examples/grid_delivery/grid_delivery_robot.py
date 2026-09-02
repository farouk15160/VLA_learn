"""
GRID DELIVERY ROBOT — a robot learns to reach a goal YOU pick, on a 128x128 map.
================================================================================
Run:
    .venv/bin/python learning_examples/grid_delivery/grid_delivery_robot.py
    .venv/bin/python learning_examples/grid_delivery/grid_delivery_robot.py --headless

Full write-up, including the MDP answer sheet: this folder's README.md
--------------------------------------------------------------------------------

WHAT IS DIFFERENT FROM reinforcement_learning.py
    That file learns ONE route to ONE fixed goal. This one learns a
    GOAL-CONDITIONED policy: the goal is an input to the network, you move it by
    clicking, and the same trained weights must handle wherever you put it. That
    is the difference between a robot that memorised a path and a robot that can
    navigate -- and it is exactly the "language-conditioned" structure of a VLA,
    where the instruction is another input alongside the image.

THE PROBLEM, STATED AS AN MDP
    S  STATE .................. the full truth is (robot cell, goal cell, step
         counter, wall map, hazard map). The first three change during an
         episode; the two maps are fixed for the life of the map.

    O  OBSERVATION ............ 334 floats, and NOT equal to S:
         [0:81]    9x9 egocentric WALL patch, 1 = blocked. Off-map counts as
                   wall. The robot sees 4 cells in each direction, no further.
         [81:162]  9x9 egocentric HAZARD patch, 1 = dangerous.
         [162:243] the SAME 9x9 wall view, zoomed out: the map max-pooled into
                   7x7 tiles and sampled every 7th cell, so this one spans 57
                   cells instead of 9. Without it a robot pressed against a
                   14-cell wall block cannot see either end of it, and "go
                   around the left" and "go around the right" are literally the
                   same observation -- so it grinds into the wall forever. Two
                   scales is the grid-world version of why a real robot gets a
                   wrist camera AND a scene camera.
         [243:324] the same, for hazards.
         [324]     (goal_row - row) / 128        signed offset to the goal
         [325]     (goal_col - col) / 128
         [326]     unit direction to goal, row component
         [327]     unit direction to goal, col component
         [328]     Manhattan distance to goal / (2*128)
         [329]     steps remaining / max_steps
         [330:334] one-hot of the PREVIOUS action, all zeros on the first step

         Four deliberate choices, each with a reason:
         - The ABSOLUTE cell (row, col) is not in there. Feed it and the network
           memorises this one map; withhold it and the only thing it can learn is
           "walk toward the goal, dodge what is next to me", which transfers to
           a map it has never seen. Same argument as the lidar car.
         - The goal enters as a RELATIVE offset, never as an absolute cell. "The
           parcel is 40 cells north of me" is directly actionable; "the parcel is
           at (12, 96)" requires the robot to also know where it is standing.
         - Steps remaining IS included, and it has to be. Truncation at 512 steps
           makes the task non-stationary: with 200 steps left a detour around a
           hazard is correct, with 3 steps left nothing is worth doing. A value
           function cannot be right about both unless it is told the clock.
         - The PREVIOUS ACTION is included, and this one I added only after
           watching it fail. A memoryless policy in a symmetric pocket has no
           way to break a tie, and it deadlocks: the trained robot would reach 3
           cells from a goal tucked against a wall and then step RIGHT, LEFT,
           RIGHT, LEFT until the clock ran out. Two cells, forever. Feeding back
           what it just did makes "arrived here going left" a different
           observation from "arrived here going right", which is enough to break
           the symmetry. This is the cheapest possible form of memory, and it is
           the honest answer to the exercise's question "do you need the
           previous position?" -- you need something, and the previous action is
           smaller and works.

    A  ACTIONS ................ 4 discrete, one per step:
         0 = UP (row-1)   1 = RIGHT (col+1)   2 = DOWN (row+1)   3 = LEFT (col-1)
         There is no WAIT action. Adding one only creates a way to burn the
         clock, and the step penalty means it could never be optimal anyway.

    P  TRANSITION ............. STOCHASTIC, as in Gymnasium's FrozenLake:
         0.8  the intended direction
         0.1  one perpendicular direction
         0.1  the other perpendicular direction
         A move into a wall or off the map leaves the robot where it was. The
         list of actions above is in rotational order, so the two perpendicular
         slips of action a are simply (a+1) % 4 and (a-1) % 4.

         Worked example, robot at (0,0), the top-left corner, action RIGHT:
             P( (0,1) | (0,0), RIGHT ) = 0.8     the move works
             P( (1,0) | (0,0), RIGHT ) = 0.1     slips DOWN
             P( (0,0) | (0,0), RIGHT ) = 0.1     slips UP, off the map, stays put
         Note the self-loop. Slip probability does not vanish at a wall; it turns
         into a wasted step, which is why hugging walls is quietly expensive.

    R  REWARD .................. per step:
         +10.00  and END, on reaching G
         -10.00  and END, on entering H
          -0.01  every step                       <- time cost
          -0.05  extra, if the step was blocked    <- do not scrape the walls
          +0.10 * (previous Manhattan distance - current)   <- progress shaping

         The shaping term is what makes this learnable. On a 128x128 map a random
         walk reaches a distant goal with probability approximately zero, so a
         pure +10-on-arrival reward gives a gradient of exactly zero forever.
         Shaping by a DIFFERENCE OF DISTANCES is the safe kind (potential-based,
         Ng et al. 1999): it cannot be farmed by pacing back and forth, because
         a step away costs precisely what the step back pays.

         Why is the hazard penalty -10 and not -1? Because the shaping term pays
         +0.1 per cell of progress. A hazard sitting 12 cells nearer the goal is
         worth +1.2 of banked shaping, so at -1.0 "walk into the hazard" is a
         PROFITABLE shortcut and the robot learns to take it. The failure looks
         like a stupid agent and is really an arithmetic bug in the reward.

    gamma  DISCOUNT ............ 0.995, because episodes run to 512 steps.

    terminated = reached G (success) or entered H (failure) -- the task itself
                 ended.
    truncated  = the 512-step budget ran out -- an external limit ended it, the
                 task did not. They are reported separately because the value of
                 the last state is 0 for the first case and NOT 0 for the second.

THE ALGORITHM
    REINFORCE with a learned value baseline, same as the sibling file:

        grad J = E[ sum_t  grad log pi(a_t|o_t) * A_t ],   A_t = G_t - V(o_t)

    Policy : 334 -> 256 -> 256 -> 4   (tanh, categorical)
    Value  : 334 -> 256 -> 256 -> 1   (tanh, MSE onto observed returns)
    Advantage: GAE(lambda=0.95) rather than the raw Monte-Carlo return -- at 256
    steps per episode the MC estimator's variance is what stops it training.
    plus a small entropy bonus, because 4 discrete actions collapse to a
    deterministic policy early and then stop exploring.

    CURRICULUM. Training samples a random start and a random goal within a
    radius that GROWS as the success rate rises, and shortens the episode budget
    to match. Without it, early episodes are 256 steps of noise around a goal
    128 cells away and nothing is learned. With it, the robot first solves
    "goal 10 cells away", then 15, then 40. The user's clicked goal is never
    trained on directly -- it has to generalise, which is the whole point.

HOW THE GUI IS MEANT TO BE USED
    It trains ITSELF the moment you launch it, in two phases:

      (1) TRAINING -- the robot practises on curriculum goals you never chose,
          and the canvas replays those practice episodes. The banner shows the
          progress. It stops on its own when 16 TEST DELIVERIES -- greedy runs
          from S to random cells, i.e. the thing you are about to click -- clear
          --target-success TWICE IN A ROW, or after --train-updates, whichever
          comes first.
          "Stop & use it" cuts it short at any time.

          Note the banner shows TWO success numbers, and they disagree on
          purpose. "practice" is the curriculum success rate, which passes 90%
          by update ~120; "test deliveries" is the real task, and lags far
          behind it. Stopping on the first would hand you a policy that looks
          trained and then misses half the cells you click.

      (2) READY -- training has stopped. CLICK ANY CELL and the trained policy
          drives from S to it, scored live. Nothing is learned in this phase, so
          what you are watching is the finished policy rather than one that is
          quietly still improving. "Train more" goes back to phase 1.

    The window sizes itself to your screen and SCROLLS (wheel, Shift+wheel for
    sideways, PgUp/PgDn/Home/End) if the layout does not fit. On a small screen
    the map and the V(s) panel stack instead of sitting side by side, and the
    map shrinks -- pass --map-px to override it.

WHAT YOU SEE IN THE GUI
    left   the map. Grey = wall, red = hazard, green = start, red ring = goal.
           An episode replays live on top.
    right  V(s) for your current goal over every cell of the map, plus arrows
           showing the action the policy would actually take -- the learned
           behaviour, drawn. Watch the route grow backwards from the goal.
    bottom success rate, mean return, policy entropy, curriculum radius.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import maximum_filter

# ---------------------------------------------------------------- world ------
GRID = 128                 # the map is GRID x GRID cells
START = (0, 0)             # S, the depot, top-left
MAX_STEPS = 512            # truncation limit. NOT 256: the far corner of a
                           # 128x128 map is 254 cells away, slipping inflates
                           # any route by ~1.3x, so it needs ~330 steps. At 256
                           # the far third of the map is unreachable BY
                           # CONSTRUCTION, and "click anywhere and the robot
                           # goes there" quietly stops being true. Use
                           # --steps 256 for the exercise as literally stated.
PATCH = 9                  # the robot sees a PATCH x PATCH window around itself
PAD = PATCH // 2
COARSE = 7                 # ...plus a second PATCH x PATCH window at 1/COARSE
CPAD = PATCH // 2 * COARSE # resolution, so it can also see 28 cells out

# actions, in ROTATIONAL order. That ordering is not cosmetic: it makes the two
# perpendicular slips of action a exactly (a+1)%4 and (a-1)%4.
DIRS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
ACT_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
N_ACT = 4
OBS_DIM = 4 * PATCH * PATCH + 6 + N_ACT

R_GOAL = 10.0              # reach G
R_HAZARD = -10.0           # enter H
R_STEP = -0.01             # every step. Halved when the budget went from 256
                           # to 512: the design rule is |R_STEP| < R_GOAL /
                           # horizon (below which detouring still pays), and
                           # doubling the horizon halves the bound to 10/512 =
                           # 0.0195. 0.01 keeps the same ~2x margin as before.
R_BLOCKED = -0.05          # extra, when the move was refused by a wall/edge
R_PROGRESS = 0.10          # per cell of Manhattan progress
GAMMA = 0.995


def make_map(n=GRID, n_walls=16, n_hazards=14, block=14, seed=0):
    """Scatter n_walls wall blocks and n_hazards hazard blocks over an n x n grid.

    The exercise says "10-20 X placed randomly". Taken literally as 10-20 single
    CELLS on a 128x128 map that is 16384 cells, which is an empty room -- the
    optimal policy is "walk diagonally" and nothing interesting is ever learned.
    So each obstacle here is a random RECTANGLE up to `block` cells on a side:
    still 10-20 obstacles, but each one large enough to actually be in the way.
    Pass --block 1 to get the literal single-cell reading back.

    (0, 0) is always kept clear, or the robot would start inside a wall.
    """
    rng = np.random.RandomState(seed)
    wall = np.zeros((n, n), bool)
    haz = np.zeros((n, n), bool)

    def scatter(grid, count):
        for _ in range(count):
            h = rng.randint(1, block + 1)
            w = rng.randint(1, block + 1)
            r = rng.randint(0, n - h + 1)
            c = rng.randint(0, n - w + 1)
            grid[r:r + h, c:c + w] = True

    scatter(wall, n_walls)
    scatter(haz, n_hazards)
    haz &= ~wall                       # a cell cannot be both
    wall[START] = False
    haz[START] = False
    return wall, haz


class GridDeliveryEnv:
    """The 128x128 stochastic delivery MDP. Gymnasium-shaped interface.

    Deliberately NOT a gymnasium.Env subclass -- writing reset/step by hand is
    the point of the exercise. The signatures match so it could be dropped in.
    """

    def __init__(self, n=GRID, n_walls=16, n_hazards=14, block=14, seed=0,
                 max_steps=MAX_STEPS):
        self.n = n
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.wall, self.haz = make_map(n, n_walls, n_hazards, block, seed)
        # Padded copies so the egocentric patch can be sliced without any
        # bounds checking. Off-map reads as WALL, which is the truth: the robot
        # cannot go there.
        self.wpad = np.pad(self.wall, PAD, constant_values=True)
        self.hpad = np.pad(self.haz, PAD, constant_values=False)
        # SECOND SCALE. A 9x9 view sees 4 cells out, and the wall blocks are up
        # to 14 cells long, so a robot pressed against one cannot see either end
        # of it -- there is no observation that distinguishes "go around left"
        # from "go around right", and the policy has no choice but to grind into
        # it. The fix is a second, zoomed-out 9x9 view: max-pool the map into
        # COARSE x COARSE tiles ("is there ANY wall in this tile?") and sample it
        # every COARSE cells, giving a 57-cell-wide field of view for 81 more
        # numbers. Cheap, and it is what turns detours into a learnable thing.
        wpool = maximum_filter(self.wall, size=COARSE)
        hpool = maximum_filter(self.haz, size=COARSE)
        self.wpad_c = np.pad(wpool, CPAD, constant_values=True)
        self.hpad_c = np.pad(hpool, CPAD, constant_values=False)
        self.free = np.argwhere(~self.wall & ~self.haz)
        self.dmax = 2.0 * n
        self.goal = (n - 1, n - 1)
        self.pos = START
        self.t = 0
        self.prev_a = -1
        self.done_reason = None

    # -- map helpers ---------------------------------------------------------
    def set_goal(self, rc):
        """Place G. Refuses walls; a hazard cell is allowed but is a bad idea."""
        r, c = int(rc[0]), int(rc[1])
        if self.wall[r, c]:
            return False
        self.goal = (r, c)
        return True

    def random_free_cell(self, rng=None):
        rng = rng or self.rng
        return tuple(self.free[rng.randint(len(self.free))])

    def sample_goal_near(self, start, radius, rng=None):
        """A free cell within `radius` Manhattan of start -- the curriculum knob."""
        rng = rng or self.rng
        for _ in range(200):
            d = rng.randint(1, max(2, int(radius) + 1))
            dr = rng.randint(-d, d + 1)
            dc = (d - abs(dr)) * (1 if rng.rand() < .5 else -1)
            r, c = start[0] + dr, start[1] + dc
            if 0 <= r < self.n and 0 <= c < self.n and not self.wall[r, c] \
                    and (r, c) != tuple(start):
                return (r, c)
        return self.random_free_cell(rng)

    # -- observation ---------------------------------------------------------
    def _obs(self):
        r, c = self.pos
        # +PAD converts a map coordinate into a padded-array coordinate, so this
        # slice is always in bounds and always PATCH x PATCH.
        wp = self.wpad[r:r + PATCH, c:c + PATCH].astype(np.float32)
        hp = self.hpad[r:r + PATCH, c:c + PATCH].astype(np.float32)
        # Coarse view: stride COARSE through the padded pooled map. Padded index
        # r + k*COARSE is map row r + (k-4)*COARSE, i.e. centred on the robot.
        sl = slice(r, r + (PATCH - 1) * COARSE + 1, COARSE)
        sl2 = slice(c, c + (PATCH - 1) * COARSE + 1, COARSE)
        wc = self.wpad_c[sl, sl2].astype(np.float32)
        hc = self.hpad_c[sl, sl2].astype(np.float32)
        prev = np.zeros(N_ACT, np.float32)
        if self.prev_a >= 0:
            prev[self.prev_a] = 1.0
        dr = self.goal[0] - r
        dc = self.goal[1] - c
        dist = abs(dr) + abs(dc)
        norm = np.hypot(dr, dc) + 1e-8
        return np.concatenate([
            wp.ravel(), hp.ravel(), wc.ravel(), hc.ravel(),
            [dr / self.n, dc / self.n,          # signed offset to the goal
             dr / norm, dc / norm,              # unit direction: scale-free, so
                                                # "which way" survives when the
                                                # offset itself is tiny
             dist / self.dmax,                  # how far, normalized
             1.0 - self.t / self.max_steps],    # clock: how much budget is left
            prev,                               # what I just did -- see docstring
        ]).astype(np.float32)

    # -- dynamics ------------------------------------------------------------
    def reset(self, start=None, goal=None, max_steps=None, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        if max_steps is not None:
            self.max_steps = max_steps
        if goal is not None:
            self.goal = tuple(int(v) for v in goal)
        self.pos = tuple(start) if start is not None else START
        self.t = 0
        self.prev_d = abs(self.goal[0] - self.pos[0]) + abs(self.goal[1] - self.pos[1])
        self.done_reason = None
        self.blocked = False
        self.prev_a = -1               # "no previous action" -> all-zero one-hot
        return self._obs()

    def _slip(self, a):
        """80% intended, 10% each perpendicular. Returns the direction ACTUALLY taken."""
        u = self.rng.rand()
        if u < 0.8:
            return a
        return (a + 1) % 4 if u < 0.9 else (a - 1) % 4

    def step(self, a):
        self.prev_a = int(a)           # the INTENDED action, not the slipped one:
                                       # that is what the policy actually chose
        eff = self._slip(a)
        dr, dc = DIRS[eff]
        r, c = self.pos[0] + dr, self.pos[1] + dc

        # Off the map or into a wall -> the robot does not move. It still burns
        # a step, and it still burned its 20% slip chance getting here.
        blocked = not (0 <= r < self.n and 0 <= c < self.n) or self.wall[r, c]
        if blocked:
            r, c = self.pos
        self.pos = (r, c)
        self.blocked = blocked
        self.t += 1

        d = abs(self.goal[0] - r) + abs(self.goal[1] - c)

        # --- REWARD, and why each number is what it is -----------------------
        # Progress shaping first: without it the +10 at the goal is unreachable
        # by random walk on a 128x128 grid and the gradient is zero forever.
        # Written as a DIFFERENCE of distances, so it is potential-based and
        # cannot be farmed by pacing: stepping away costs what stepping back pays.
        rew = R_PROGRESS * (self.prev_d - d)
        rew += R_STEP                       # time cost -> prefer the short route
        if blocked:
            rew += R_BLOCKED                # scraping a wall wastes a step
        self.prev_d = d

        terminated = False
        if (r, c) == self.goal:
            rew += R_GOAL
            terminated = True
            self.done_reason = "goal"
        elif self.haz[r, c]:
            # Must OUTWEIGH the shaping already banked on the way in. A hazard 12
            # cells nearer the goal has paid +1.2 of progress, so anything
            # smaller than that makes "walk into the hazard" a profitable
            # shortcut -- and the robot will find it.
            rew += R_HAZARD
            terminated = True
            self.done_reason = "hazard"

        truncated = self.t >= self.max_steps
        if truncated and not terminated:
            self.done_reason = "timeout"
        return self._obs(), rew, terminated, truncated

    # -- an oracle, for reporting only ---------------------------------------
    def bfs_distance(self, goal=None):
        """Shortest wall-avoiding path length from every cell to the goal.

        Used ONLY to tell you whether your clicked goal is even reachable in the
        step budget, and to score the policy against optimal. The agent never
        sees it -- it would be cheating, and on a real robot it does not exist.
        """
        goal = tuple(goal or self.goal)
        INF = np.iinfo(np.int32).max
        dist = np.full((self.n, self.n), INF, np.int32)
        # A goal inside a wall is unreachable from everywhere, and saying so is
        # the point. Seeding the search at a wall cell anyway (my first version)
        # happily reports a route to a cell the robot can never occupy, and then
        # a perfectly good policy looks broken because it "fails" to get there.
        if self.wall[goal]:
            return dist
        dist[goal] = 0
        frontier = [goal]
        while frontier:
            nxt = []
            for (r, c) in frontier:
                d = dist[r, c] + 1
                for dr, dc in DIRS:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.n and 0 <= cc < self.n \
                            and not self.wall[rr, cc] and dist[rr, cc] > d:
                        dist[rr, cc] = d
                        nxt.append((rr, cc))
            frontier = nxt
        return dist


def feasibility(env, goal, max_steps=MAX_STEPS, start=START):
    """Is this goal reachable at all, and is the step budget big enough?

    Slipping costs steps: 20% of moves go sideways, so a route of `opt` cells
    takes roughly opt/0.8 = 1.25*opt steps in expectation, and more when the
    detours land the robot against a wall. Anything past ~1.35*opt of budget is
    a coin flip no matter how good the policy is -- which is worth knowing
    BEFORE blaming the agent.
    """
    if env.wall[tuple(goal)]:
        return {"ok": False, "why": "that cell is a wall", "opt": None}
    opt = int(env.bfs_distance(goal)[start])
    if opt >= np.iinfo(np.int32).max:
        return {"ok": False, "why": "walled off from the depot", "opt": None}
    need = 1.3 * opt
    if opt > max_steps:
        why = f"needs {opt} steps, budget is {max_steps}"
        return {"ok": False, "why": why, "opt": opt}
    if need > max_steps:
        return {"ok": True, "tight": True, "opt": opt,
                "why": f"optimal {opt}, slipping needs ~{need:.0f}, budget {max_steps}"}
    return {"ok": True, "tight": False, "opt": opt,
            "why": f"optimal {opt} steps, budget {max_steps}"}


# ---------------------------------------------------------------- nets -------
class Policy(nn.Module):
    def __init__(self, obs=OBS_DIM, hid=256, act=N_ACT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(),
                                 nn.Linear(hid, act))

    def forward(self, x):
        return self.net(x)

    def dist(self, x):
        return torch.distributions.Categorical(logits=self.net(x))


class ValueNet(nn.Module):
    def __init__(self, obs=OBS_DIM, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(),
                                 nn.Linear(hid, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def gae(rew, vals, last_val, gamma, lam):
    """Generalized Advantage Estimation (Schulman 2015).

        delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)          one-step TD error
        A_t     = delta_t + (gamma*lam) * A_{t+1}          exponentially smoothed

    lam = 1 recovers the plain Monte-Carlo advantage used in the car file:
    unbiased, but its variance grows with the episode length, and episodes here
    are 256 steps rather than 60. lam = 0.95 accepts a little bias from the
    value net in exchange for far less noise, which is the trade that makes a
    long-horizon task like this one train at all.

    Returns (advantages, value targets).
    """
    T = len(rew)
    adv = np.empty(T, np.float32)
    acc = 0.0
    nxt = last_val
    for i in range(T - 1, -1, -1):
        delta = rew[i] + gamma * nxt - vals[i]
        acc = delta + gamma * lam * acc
        adv[i] = acc
        nxt = vals[i]
    return adv, adv + vals


def returns_to_go(rew, gamma):
    """G_t = r_t + gamma*r_{t+1} + ... computed backwards in one pass."""
    out = np.empty_like(rew)
    acc = 0.0
    for i in range(len(rew) - 1, -1, -1):
        acc = rew[i] + gamma * acc
        out[i] = acc
    return out


def obs_grid(env, goal, stride=2, steps_left=1.0):
    """Build the observation of EVERY cell at once, for the V(s) heatmap.

    Same maths as env._obs, vectorised over the whole map. Returns the batch and
    the (rows, cols) it corresponds to. Walls are skipped -- the robot can never
    stand there, so V is undefined and the map draws them grey.
    """
    n = env.n
    rr, cc = np.meshgrid(np.arange(0, n, stride), np.arange(0, n, stride),
                         indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    keep = ~env.wall[rr, cc]
    rr, cc = rr[keep], cc[keep]

    # Gather every PATCH x PATCH window with one fancy-index, no Python loop.
    off = np.arange(PATCH)
    ridx = rr[:, None, None] + off[None, :, None]
    cidx = cc[:, None, None] + off[None, None, :]
    wp = env.wpad[ridx, cidx].reshape(len(rr), -1).astype(np.float32)
    hp = env.hpad[ridx, cidx].reshape(len(rr), -1).astype(np.float32)
    coff = off * COARSE
    cri = rr[:, None, None] + coff[None, :, None]
    cci = cc[:, None, None] + coff[None, None, :]
    wc = env.wpad_c[cri, cci].reshape(len(rr), -1).astype(np.float32)
    hc = env.hpad_c[cri, cci].reshape(len(rr), -1).astype(np.float32)

    dr = goal[0] - rr
    dc = goal[1] - cc
    dist = np.abs(dr) + np.abs(dc)
    norm = np.hypot(dr, dc) + 1e-8
    # No action history exists for a hypothetical cell, so the heatmap is drawn
    # for "the robot has just arrived here", i.e. an all-zero previous action.
    tail = np.stack([dr / n, dc / n, dr / norm, dc / norm, dist / env.dmax,
                     np.full(len(rr), steps_left)], 1).astype(np.float32)
    tail = np.concatenate([tail, np.zeros((len(rr), N_ACT), np.float32)], 1)
    return np.concatenate([wp, hp, wc, hc, tail], 1), rr, cc


# ---------------------------------------------------------------- trainer ----
class Trainer:
    """REINFORCE + value baseline + a distance curriculum, one update at a time.

    `update()` runs a batch of episodes, takes ONE policy-gradient step, and
    returns a snapshot. The GUI calls it from a worker thread and draws the
    snapshot; headless mode loops over it and prints. Identical code either way.
    """

    def __init__(self, env=None, seed=0, gamma=GAMMA, lr=5e-4, batch_eps=16,
                 hid=256, use_baseline=True, ent_coef=0.03, curriculum=True,
                 max_steps=MAX_STEPS, lam=0.95):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.policy = Policy(hid=hid)
        self.value = ValueNet(hid=hid)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.vopt = torch.optim.Adam(self.value.parameters(), lr=1e-2)
        self.env = env if env is not None else GridDeliveryEnv(seed=seed)
        self.gamma, self.batch_eps = gamma, batch_eps
        self.use_baseline = use_baseline
        self.ent_coef = ent_coef
        self.lam = lam
        self.curriculum = curriculum
        self.max_steps = max_steps
        # Used only when the curriculum is ablated: the one goal to train on.
        # Held separately because env.goal is overwritten by every reset().
        self.fixed_goal = self.env.goal
        # The curriculum radius. Start small: on a 128x128 map, a goal 128 cells
        # away is 256 steps of pure noise and teaches nothing. Grow it as the
        # robot starts succeeding.
        self.radius = 10.0 if curriculum else 2.0 * self.env.n
        self.updates = 0
        self.episodes = 0
        self.hist = {"ret": [], "success": [], "hazard": [], "steps": [],
                     "entropy": [], "radius": []}

    # -- one episode ---------------------------------------------------------
    def rollout(self, greedy=False, start=None, goal=None, max_steps=None,
                seed=None, record=True):
        """Run one episode. `record` keeps the path so the GUI can replay it."""
        o = self.env.reset(start=start, goal=goal,
                           max_steps=max_steps or self.max_steps, seed=seed)
        O, A, R = [], [], []
        traj = {"path": [], "probs": [], "goal": self.env.goal,
                "start": self.env.pos}
        while True:
            with torch.no_grad():
                d = self.policy.dist(torch.from_numpy(o))
                a = int(d.probs.argmax()) if greedy else int(d.sample())
            if record:
                traj["path"].append(self.env.pos)
                traj["probs"].append(d.probs.numpy().copy())
            O.append(o)
            A.append(a)
            o, r, term, trunc = self.env.step(a)
            R.append(r)
            if term or trunc:
                break
        if record:
            traj["path"].append(self.env.pos)
        traj["reason"] = self.env.done_reason
        traj["greedy"] = greedy
        traj["ret"] = float(sum(R))
        # `terminated` and `last_obs` are what makes the bootstrap in update()
        # correct. If the episode TERMINATED, the future after it is worth
        # exactly 0. If it was TRUNCATED at the step limit, the future is worth
        # V(last state) -- the robot was mid-journey and we cut it off. Treating
        # those two the same teaches the value net that running out of clock is
        # as final as dying, and it is the single most common bug in hand-rolled
        # RL code.
        traj["terminated"] = self.env.done_reason in ("goal", "hazard")
        traj["last_obs"] = o
        return np.array(O, np.float32), np.array(A), np.array(R, np.float32), traj

    # -- one gradient step ---------------------------------------------------
    def update(self):
        # Episode length is tied to the curriculum radius: while the goals are
        # 10 cells away there is no reason to simulate 256 steps of wandering.
        ep_steps = int(min(self.max_steps, max(30, 4 * self.radius + 30)))
        logps, coefs, ents, Os, Gs = [], [], [], [], []
        rets, succ, haz, steps = [], [], [], []
        last = None

        for k in range(self.batch_eps):
            if self.curriculum:
                s = self.env.random_free_cell()
                g = self.env.sample_goal_near(s, self.radius)
            else:
                s, g = START, self.fixed_goal
            # Only the final episode of the batch keeps its path: the GUI shows
            # one training episode per update, and recording all 8 is wasted work.
            O, A, R, traj = self.rollout(start=s, goal=g, max_steps=ep_steps,
                                         record=(k == self.batch_eps - 1))
            Ot = torch.from_numpy(O)
            d = self.policy.dist(Ot)
            if self.use_baseline:
                with torch.no_grad():
                    v = self.value(Ot).numpy()
                    # Bootstrap only if the episode was cut off by the clock.
                    last = 0.0 if traj["terminated"] else float(
                        self.value(torch.from_numpy(traj["last_obs"][None]))[0])
                a_t, G = gae(R, v, last, self.gamma, self.lam)
            else:
                # Ablation: no baseline, no bootstrap -- weight grad-log-pi by
                # the raw discounted return, as REINFORCE was first written.
                G = returns_to_go(R, self.gamma)
                a_t = G
            logps.append(d.log_prob(torch.from_numpy(A)))
            coefs.append(torch.from_numpy(np.asarray(a_t, np.float32)))
            ents.append(d.entropy())
            Os.append(Ot)
            Gs.append(torch.from_numpy(np.asarray(G, np.float32)))
            rets.append(traj["ret"])
            succ.append(traj["reason"] == "goal")
            haz.append(traj["reason"] == "hazard")
            steps.append(len(R))
            self.episodes += 1
            last = traj

        logp = torch.cat(logps)
        adv = torch.cat(coefs).detach()
        # Normalize: returns here range over roughly [-15, +15] depending on how
        # far the goal was, and unnormalized advantages make the effective step
        # size depend on that. Standard practice in A2C/PPO for the same reason.
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        ent = torch.cat(ents).mean()
        # The entropy bonus matters more here than in the car file: with only 4
        # actions the softmax saturates within a few dozen updates, and a policy
        # that has stopped exploring cannot discover the way around a wall.
        loss = -(logp * adv).mean() - self.ent_coef * ent
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 2.0)
        self.opt.step()

        S, Gt = torch.cat(Os), torch.cat(Gs)
        for _ in range(5 if self.use_baseline else 0):
            vl = nn.functional.mse_loss(self.value(S), Gt)
            self.vopt.zero_grad()
            vl.backward()
            self.vopt.step()

        sr = float(np.mean(succ))
        if self.curriculum:
            # Promote only on SUSTAINED competence -- the mean over the last 10
            # updates, not one lucky batch of 8 episodes. Promoting on a single
            # batch (my first version) walks the radius up to the map diagonal
            # inside 150 updates on pure noise, and the robot then trains
            # forever on goals it cannot yet reach.
            recent = self.hist["success"][-10:]
            if len(recent) >= 10 and float(np.mean(recent)) > 0.80:
                self.radius = min(self.radius * 1.10, 2.0 * self.env.n)

        self.updates += 1
        self.hist["ret"].append(float(np.mean(rets)))
        self.hist["success"].append(sr)
        self.hist["hazard"].append(float(np.mean(haz)))
        self.hist["steps"].append(float(np.mean(steps)))
        self.hist["entropy"].append(float(ent.detach()))
        self.hist["radius"].append(self.radius)
        return {"update": self.updates, "episodes": self.episodes,
                "ret": self.hist["ret"][-1], "success": sr,
                "hazard": self.hist["hazard"][-1], "steps": self.hist["steps"][-1],
                "entropy": self.hist["entropy"][-1], "radius": self.radius,
                "traj": last, "hist": self.hist}

    def spot_check(self, n=16, rng=None):
        """Greedy deliveries from S to random reachable goals. Fraction won.

        This exists because the training success rate is NOT the thing the user
        cares about. It is measured on curriculum goals -- a random start, a
        goal sampled near it -- and it hits 0.9 long before the robot can cross
        the map from the depot. Stopping training on it hands over a policy that
        looks trained and then misses half the cells you click.

        So the stopping criterion measures the actual task: start at S, go to a
        cell chosen at random, greedily. It is still a noisy estimate -- which is
        why the caller requires two consecutive passes -- but it is noisy about
        the right quantity, and that beats a precise measurement of the wrong one.
        """
        rng = rng or np.random.RandomState(np.random.randint(1 << 30))
        wins = 0
        for _ in range(n):
            for _ in range(50):                    # find a goal worth scoring
                g_ = tuple(int(v) for v in self.random_goal_cell(rng))
                if feasibility(self.env, g_, self.max_steps)["ok"]:
                    break
            _, _, _, tr = self.rollout(greedy=True, start=START, goal=g_,
                                       max_steps=self.max_steps, record=False)
            wins += tr["reason"] == "goal"
        return wins / n

    def random_goal_cell(self, rng):
        return self.env.random_free_cell(rng)

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, goal, n=20, start=START, greedy=True, max_steps=None):
        """Roll the current policy `n` times to a FIXED goal and count outcomes."""
        if self.env.wall[tuple(goal)]:
            raise ValueError(f"goal {tuple(goal)} is inside a wall")
        out = {"goal": 0, "hazard": 0, "timeout": 0}
        lens = []
        for i in range(n):
            _, _, R, tr = self.rollout(greedy=greedy, start=start, goal=goal,
                                       max_steps=max_steps or self.max_steps,
                                       seed=7000 + i, record=False)
            out[tr["reason"]] = out.get(tr["reason"], 0) + 1
            if tr["reason"] == "goal":
                lens.append(len(R))
        out["mean_len"] = float(np.mean(lens)) if lens else float("nan")
        return out


import argparse
import queue
import threading
import time


# ============================================================== headless ======
def run_headless(args):
    env = GridDeliveryEnv(n=args.grid, n_walls=args.walls, n_hazards=args.hazards,
                          block=args.block, seed=args.seed, max_steps=args.steps)
    t = Trainer(env=env, seed=args.seed, lr=args.lr, batch_eps=args.batch_eps,
                use_baseline=not args.no_baseline, ent_coef=args.ent,
                curriculum=not args.no_curriculum, max_steps=args.steps)

    goal = tuple(int(v) for v in args.goal.split(",")) if args.goal else (args.grid - 1, args.grid - 1)
    if env.wall[goal]:
        # Whatever the user asked for is a wall: slide to the nearest legal cell
        # rather than silently training toward somewhere unreachable.
        free = env.free
        k = int(np.argmin(np.abs(free - np.array(goal)).sum(1)))
        goal = tuple(int(v) for v in free[k])
        print(f"requested goal is a wall; using the nearest free cell {goal}")
    env.set_goal(goal)

    f = feasibility(env, goal, args.steps)
    print(f"map {args.grid}x{args.grid}   {int(env.wall.sum())} wall cells "
          f"({args.walls} blocks)   {int(env.haz.sum())} hazard cells ({args.hazards} blocks)")
    print(f"goal {goal}   {f['why']}"
          + ("   << not reachable" if not f["ok"] else
             "   << tight: slipping alone may run out the clock" if f["tight"] else ""))
    print()
    print(f"{'upd':>5} {'eps':>7} {'return':>8} {'success':>8} {'hazard':>7} "
          f"{'steps':>7} {'entropy':>8} {'radius':>7}")
    print("-" * 64)

    t0 = time.time()
    for i in range(1, args.updates + 1):
        s = t.update()
        if i % max(1, args.updates // 20) == 0 or i == 1:
            print(f"{s['update']:>5} {s['episodes']:>7} {s['ret']:>8.2f} "
                  f"{s['success']:>8.2f} {s['hazard']:>7.2f} {s['steps']:>7.1f} "
                  f"{s['entropy']:>8.3f} {s['radius']:>7.0f}")
    print(f"\ntrained in {time.time() - t0:.0f}s")

    if f["ok"]:
        r = t.evaluate(goal, n=args.eval_eps)
        print(f"greedy eval to {goal}: {r['goal']}/{args.eval_eps} delivered, "
              f"{r['hazard']} destroyed, {r['timeout']} timed out"
              + (f", mean {r['mean_len']:.0f} steps vs {f['opt']} optimal"
                 f" ({r['mean_len']/f['opt']:.2f}x)" if r["goal"] else ""))

    # Generalisation is the real question: this policy was NEVER trained on any
    # specific goal, so score it on goals it has never been given, by distance.
    print("\ngeneralisation to unseen goals (20 random valid goals per band):")
    rng = np.random.RandomState(123)
    for lo, hi in [(1, 30), (30, 80), (80, 150), (150, 256)]:
        picked, wins, ratios = 0, 0, []
        for _ in range(2000):
            if picked >= 20:
                break
            g_ = env.random_free_cell(rng)
            fb = feasibility(env, g_, args.steps)
            if not fb["ok"] or not (lo <= fb["opt"] < hi):
                continue
            picked += 1
            r = t.evaluate(g_, n=3)
            wins += r["goal"]
            if r["goal"]:
                ratios.append(r["mean_len"] / fb["opt"])
        if picked:
            print(f"  optimal {lo:>3}-{hi:<3} steps : {wins}/{picked*3} delivered"
                  + (f"   mean {np.mean(ratios):.2f}x optimal" if ratios else ""))


# ============================================================== the GUI =======
def run_gui(args):
    import tkinter as tk
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    N = args.grid
    C_FREE, C_WALL, C_HAZ = "#ffffff", "#4a4a55", "#f0a9a4"
    BG = "#f4f4f6"

    root = tk.Tk()
    root.title(f"Grid delivery robot — {N}x{N}, stochastic, goal-conditioned "
               f"(REINFORCE + GAE)")
    root.configure(bg=BG)

    # ------------------------------------------------------- fit to the screen
    # The full layout is ~1600x1030. That is taller than a 1080p laptop screen
    # once the window bar and taskbar are taken out, so the bottom plots were
    # simply cut off with no way to reach them. Everything below therefore sizes
    # itself against the actual screen, and the whole window scrolls.
    SW = root.winfo_screenwidth()
    SH = root.winfo_screenheight()
    map_px = args.map_px or int(min(620, SH * 0.52))
    SC = max(1, int(map_px / N))              # cell size in pixels
    CW = CH = N * SC
    # matplotlib figure sizes are in inches at 100 dpi, i.e. hundreds of pixels.
    fig_w = max(4.0, min(5.6, (SW - CW - 120) / 100.0))
    fig_h = max(3.0, CH / 100.0 - 0.6)
    strip_w = max(7.0, min(13.5, (SW - 110) / 100.0))
    strip_h = max(1.9, min(2.6, SH / 420.0))
    # On a narrow screen the map and the V(s) panel will not fit side by side,
    # so stack them instead of forcing a horizontal scroll for normal use.
    narrow = (CW + int(fig_w * 100) + 90) > (SW - 40)

    env = GridDeliveryEnv(n=N, n_walls=args.walls, n_hazards=args.hazards,
                          block=args.block, seed=args.seed, max_steps=args.steps)

    # THE USER'S GOAL LIVES HERE, not in env.goal. env.goal is scratch space:
    # every training episode calls env.reset(goal=<curriculum goal>) and
    # overwrites it. Reading env.goal to draw the star meant the star tracked
    # whatever random goal the trainer had just used, and the replayed episode
    # walked to that instead of to the cell you clicked.
    # The GUI has two phases and the whole layout follows from them:
    #   "training"  -- the robot practises on curriculum goals you never see,
    #                  and the canvas replays those practice episodes.
    #   "ready"     -- training has stopped. You click a cell, and the trained
    #                  policy drives to it. Nothing is learned in this phase, so
    #                  what you see is the finished policy, not a policy that is
    #                  quietly still improving while you watch.
    state = {"run": False, "phase": "training", "traj": None, "frame": 0,
             "quit": False, "fps": args.fps, "throttle": args.throttle,
             "snap": None, "vgrid": None, "field": None, "pending": None,
             "target": args.train_updates, "deliveries": [], "auto_stop": True,
             "shown_phase": None, "last_check": 0, "spot": None, "passes": 0,
             "goal": (N - 1, N - 1) if not env.wall[N - 1, N - 1]
                     else env.random_free_cell()}
    watch_greedy = tk.BooleanVar(value=True)
    show_arrows = tk.BooleanVar(value=True)
    show_heat = tk.BooleanVar(value=True)
    q = queue.Queue()
    lock = threading.Lock()
    trainer = {"t": Trainer(env=env, seed=args.seed, lr=args.lr,
                            batch_eps=args.batch_eps, ent_coef=args.ent,
                            use_baseline=not args.no_baseline,
                            curriculum=not args.no_curriculum,
                            max_steps=args.steps)}

    # ---------------------------------------------------------------- layout
    # A Tk window does not scroll on its own: you scroll a Canvas, and put the
    # real layout inside it as a single embedded window. `content` is what every
    # widget below is packed into; `scroller` is the viewport onto it.
    shell = tk.Frame(root, bg=BG); shell.pack(fill="both", expand=True)
    scroller = tk.Canvas(shell, bg=BG, highlightthickness=0)
    vbar = tk.Scrollbar(shell, orient="vertical", command=scroller.yview)
    hbar = tk.Scrollbar(shell, orient="horizontal", command=scroller.xview)
    scroller.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
    # grid, not pack: with pack, the expanding canvas claims the space first and
    # a later-packed horizontal bar gets squeezed into the leftover corner.
    scroller.grid(row=0, column=0, sticky="nsew")
    vbar.grid(row=0, column=1, sticky="ns")
    hbar.grid(row=1, column=0, sticky="ew")
    shell.rowconfigure(0, weight=1)
    shell.columnconfigure(0, weight=1)
    content = tk.Frame(scroller, bg=BG)
    scroller.create_window((0, 0), window=content, anchor="nw")

    def _resize(_=None):
        # The scrollable area is exactly as big as the content wants to be, and
        # each scrollbar appears only when there is something to scroll to.
        box = scroller.bbox("all")
        if box is None:
            return
        scroller.configure(scrollregion=box)
        need_v = box[3] - box[1] > scroller.winfo_height() + 2
        need_h = box[2] - box[0] > scroller.winfo_width() + 2
        (vbar.grid if need_v else vbar.grid_remove)()
        (hbar.grid if need_h else hbar.grid_remove)()
    content.bind("<Configure>", _resize)
    scroller.bind("<Configure>", _resize)

    def _wheel(ev):
        # X11 sends wheel as Button-4/5; Windows and macOS send <MouseWheel>
        # with a delta. Handle both so this scrolls wherever it is run.
        if ev.num == 4 or getattr(ev, "delta", 0) > 0:
            scroller.yview_scroll(-3, "units")
        elif ev.num == 5 or getattr(ev, "delta", 0) < 0:
            scroller.yview_scroll(3, "units")

    def _shift_wheel(ev):
        if ev.num == 4 or getattr(ev, "delta", 0) > 0:
            scroller.xview_scroll(-3, "units")
        elif ev.num == 5 or getattr(ev, "delta", 0) < 0:
            scroller.xview_scroll(3, "units")

    for seq, fn in (("<Button-4>", _wheel), ("<Button-5>", _wheel),
                    ("<MouseWheel>", _wheel),
                    ("<Shift-Button-4>", _shift_wheel),
                    ("<Shift-Button-5>", _shift_wheel),
                    ("<Shift-MouseWheel>", _shift_wheel)):
        root.bind_all(seq, fn)
    root.bind_all("<Prior>", lambda e: scroller.yview_scroll(-1, "pages"))
    root.bind_all("<Next>", lambda e: scroller.yview_scroll(1, "pages"))
    root.bind_all("<Home>", lambda e: scroller.yview_moveto(0))
    root.bind_all("<End>", lambda e: scroller.yview_moveto(1))

    # The banner is a full-width strip of its own. Inside the map column it was
    # the widest thing there, so the column grew to ~740px around a 400px map
    # and left a dead gap before the V(s) panel.
    phase_lbl = tk.Label(content, text="", bg=BG, anchor="w", justify="left",
                         font=("DejaVu Sans", 12, "bold"))
    phase_lbl.pack(side="top", fill="x", padx=12, pady=(8, 2))

    top = tk.Frame(content, bg=BG); top.pack(side="top", fill="both", expand=True)
    side = "top" if narrow else "left"
    left = tk.Frame(top, bg=BG); left.pack(side=side, padx=10, pady=(0, 6),
                                           anchor="nw")
    canvas = tk.Canvas(left, width=CW, height=CH, bg="#ffffff",
                       highlightthickness=1, highlightbackground="#c9c9d0")
    canvas.pack()
    goal_lbl = tk.Label(left, text="", bg=BG, fg="#333", justify="left",
                        anchor="w", font=("DejaVu Sans", 10))
    goal_lbl.pack(fill="x", pady=(6, 0))

    right = tk.Frame(top, bg=BG); right.pack(side=side, padx=(0, 10), pady=10,
                                             anchor="nw")
    figv = Figure(figsize=(fig_w, fig_h), dpi=100)
    axv = figv.add_subplot(111)
    cv = FigureCanvasTkAgg(figv, master=right)
    # No expand: an expanding canvas inside a scrolled frame just grows to the
    # widest sibling and leaves a dead gap next to the map.
    cv.get_tk_widget().pack(anchor="nw")
    stat = tk.Label(right, text="", font=("DejaVu Sans Mono", 10), justify="left",
                    bg=BG, anchor="w")
    stat.pack(fill="x", padx=6)

    figc = Figure(figsize=(strip_w, strip_h), dpi=100)
    ax1 = figc.add_subplot(141); ax2 = figc.add_subplot(142)
    ax3 = figc.add_subplot(143); ax4 = figc.add_subplot(144)
    cc = FigureCanvasTkAgg(figc, master=content)
    cc.get_tk_widget().pack(side="top", fill="both", expand=True, padx=10)
    # Two control rows. One row of every button and slider is ~1300px wide and
    # was the single widest thing in the window, forcing a horizontal scrollbar
    # on a 1366-wide laptop all by itself.
    ctl = tk.Frame(content, bg=BG); ctl.pack(side="top", fill="x", pady=(6, 0))
    ctl2 = tk.Frame(content, bg=BG); ctl2.pack(side="top", fill="x", pady=(2, 6))

    # ------------------------------------------------------------ map drawing
    # The map is 16384 cells. Drawing one Tk rectangle per cell makes the canvas
    # crawl, so the static map is painted ONCE into a PhotoImage and blitted.
    photo = {"img": None}

    def draw_static():
        img = tk.PhotoImage(width=N, height=N)
        rows = []
        for r in range(N):
            rows.append("{" + " ".join(
                C_WALL if env.wall[r, c] else (C_HAZ if env.haz[r, c] else C_FREE)
                for c in range(N)) + "}")
        img.put(" ".join(rows), to=(0, 0))
        photo["img"] = img.zoom(SC) if SC > 1 else img
        canvas.delete("static")
        canvas.create_image(0, 0, anchor="nw", image=photo["img"], tags="static")
        canvas.tag_lower("static")

    def cell_box(rc, pad=0):
        r, c = rc
        return (c * SC - pad, r * SC - pad, (c + 1) * SC + pad, (r + 1) * SC + pad)

    def draw_markers():
        canvas.delete("mark")
        if state["phase"] == "training":
            # During training the goal that matters is the episode's own
            # curriculum goal, drawn per frame in draw_frame(). Showing the
            # user's G as well just invites "why is it not going there?".
            return
        x0, y0, x1, y1 = cell_box(START, 3)
        canvas.create_rectangle(x0, y0, x1, y1, outline="#2e9e2e", width=2,
                                tags="mark")
        canvas.create_text((x0 + x1) / 2, y1 + 9, text="S", fill="#2e9e2e",
                           font=("DejaVu Sans", 10, "bold"), tags="mark")
        gx0, gy0, gx1, gy1 = cell_box(state["goal"], 5)
        canvas.create_oval(gx0, gy0, gx1, gy1, outline="#d43d3d", width=3,
                           tags="mark")
        canvas.create_text((gx0 + gx1) / 2, (gy0 + gy1) / 2, text="G",
                           fill="#d43d3d", font=("DejaVu Sans", 11, "bold"),
                           tags="mark")

    def draw_frame():
        tr = state["traj"]
        canvas.delete("dyn")
        if tr is None or not tr["path"]:
            return
        n = len(tr["path"])
        i = min(state["frame"], n - 1)
        if state["phase"] == "training":
            # this practice episode's own start and goal
            sr, sc = tr["start"]
            canvas.create_rectangle(*cell_box((sr, sc), 3), outline="#2e9e2e",
                                    width=2, tags="dyn")
            canvas.create_oval(*cell_box(tr["goal"], 5), outline="#d43d3d",
                               width=2, tags="dyn")
        pts = []
        for (r, c) in tr["path"][:i + 1]:
            pts += [c * SC + SC / 2, r * SC + SC / 2]
        if len(pts) >= 4:
            canvas.create_line(*pts, fill="#3355bb", width=2, tags="dyn")
        r, c = tr["path"][i]
        col = {"goal": "#2e9e2e", "hazard": "#d43d3d"}.get(
            tr["reason"] if i == n - 1 else None, "#1b3fa0")
        x0, y0, x1, y1 = cell_box((r, c), 3)
        canvas.create_oval(x0, y0, x1, y1, fill=col, outline="#ffffff",
                           width=2, tags="dyn")
        done = i >= n - 1
        txt = {"goal": "DELIVERED", "hazard": "DESTROYED (hazard)",
               "timeout": "OUT OF STEPS"}.get(tr["reason"], "") if done else "moving…"
        canvas.create_rectangle(0, 0, CW, 22, fill="#ffffff", outline="", tags="dyn")
        canvas.create_text(6, 11, anchor="w", tags="dyn",
                           font=("DejaVu Sans Mono", 10),
                           fill={"DELIVERED": "#2e9e2e",
                                 "DESTROYED (hazard)": "#d43d3d"}.get(txt, "#444"),
                           text=f"step {i}/{n - 1}   {txt}")
        if CW >= 520:
            # On a small screen the map is ~380px wide and this label runs into
            # the step counter on the left. The banner already says which phase
            # we are in, so drop it rather than overlap.
            canvas.create_text(CW - 6, 11, anchor="e", tags="dyn",
                               font=("DejaVu Sans Mono", 9), fill="#777",
                               text=("practice episode"
                                     if state["phase"] == "training"
                                     else "trained policy, greedy"))

    def refresh_phase():
        t = trainer["t"]
        if state["shown_phase"] != state["phase"]:
            # The user's G is hidden during training (draw_markers returns
            # early), so it has to be drawn the moment the phase flips --
            # otherwise the ready phase starts with no visible goal.
            state["shown_phase"] = state["phase"]
            draw_markers()
        if state["phase"] == "training":
            pct = 100.0 * min(1.0, t.updates / max(1, state["target"]))
            sr = (np.mean(t.hist["success"][-20:]) if t.hist["success"] else 0.0)
            spot = ("" if state["spot"] is None
                    else f"   test deliveries {state['spot']*100:.0f}%")
            phase_lbl.config(
                fg="#c46a1e",
                text=f"① TRAINING — update {t.updates}/{state['target']} "
                     f"({pct:.0f}%)   practice {sr*100:.0f}%   "
                     f"reach {t.radius:.0f}/{2*env.n}{spot}"
                     + ("   [paused]" if not state["run"] else ""))
            btn.config(text="⏸  Pause" if state["run"]
                       else ("▶  Train" if t.updates == 0 else "▶  Resume"))
        else:
            d = state["deliveries"][-20:]
            tally = (f"   delivered {sum(d)}/{len(d)}" if d else "")
            phase_lbl.config(
                fg="#2e7d32",
                text="② READY — click any cell and the trained policy will "
                     "drive to it" + tally)
            btn.config(text="＋ Train more")

    def refresh_goal_label():
        g_ = state["goal"]
        f = feasibility(env, g_, args.steps)
        if not f["ok"]:
            msg, col = f"goal {g_}: UNREACHABLE — {f['why']}", "#d43d3d"
        elif f["tight"]:
            msg, col = f"goal {g_}: TIGHT — {f['why']}", "#c46a1e"
        else:
            msg, col = f"goal {g_}: {f['why']}", "#2e7d32"
        goal_lbl.config(text=msg + "     (click the map to move G)", fg=col)

    def on_click(ev):
        r, c = int(ev.y // SC), int(ev.x // SC)
        if not (0 <= r < N and 0 <= c < N):
            return
        if env.wall[r, c]:
            goal_lbl.config(text=f"({r}, {c}) is a wall — pick a free cell",
                            fg="#d43d3d")
            return
        state["goal"] = (r, c)
        state["traj"] = None; state["frame"] = 0
        state["vgrid"] = None
        state["deliveries"] = []          # a new goal is a new scoreboard
        draw_markers(); refresh_goal_label(); refresh_phase()
        if state["phase"] == "ready":
            # Redraw V(s) for the cell just clicked, so the heatmap and the
            # arrows describe the journey about to be animated.
            with lock:
                ob, rr, ccx = obs_grid(env, (r, c), stride=2)
                with torch.no_grad():
                    v = trainer["t"].value(torch.from_numpy(ob)).numpy()
                ob2, r2, c2 = obs_grid(env, (r, c), stride=max(4, N // 20))
                with torch.no_grad():
                    a2 = trainer["t"].policy.dist(
                        torch.from_numpy(ob2)).probs.argmax(1).numpy()
            m = (env.n + 1) // 2
            grid = np.full((m, m), np.nan, np.float32)
            grid[rr // 2, ccx // 2] = v
            snap = dict(state["snap"] or {})
            snap.update({"vgrid": grid, "stride": 2, "field": (r2, c2, a2),
                         "hist": trainer["t"].hist})
            state["snap"] = snap
            redraw_plots(snap)

    canvas.bind("<Button-1>", on_click)

    # -------------------------------------------------------------- worker ---
    def training_finished(t):
        """Stop when the robot is competent, or when the budget runs out.

        Competence, not a step count, because how long this takes depends on the
        map: the curriculum has to reach the far end of the grid AND hold a high
        success rate there. A fixed number of updates either stops early on a
        hard map or wastes minutes on an easy one.
        """
        if t.updates >= state["target"]:
            return True
        if not state["auto_stop"]:
            # "Train more" asked for a specific number of extra updates. Letting
            # the competence check fire here would end it after one update,
            # since the policy is by definition already competent.
            return False
        if t.radius < 2.0 * env.n - 1e-6:
            return False                          # curriculum still expanding
        if t.updates - state["last_check"] < 25:
            return False                          # spot-check at most every 25
        state["last_check"] = t.updates
        state["spot"] = t.spot_check(16)
        # Require TWO consecutive passes. With a single 8-episode check the rule
        # fired at update 117 on a policy that then delivered 3/6 to a
        # 165-step goal: 7-of-8 happens by luck often enough when the true rate
        # is ~0.7. Sixteen episodes twice in a row is a much harder coincidence,
        # and costs about two seconds.
        if state["spot"] >= args.target_success:
            state["passes"] += 1
        else:
            state["passes"] = 0
        return state["passes"] >= 2

    def worker():
        while not state["quit"]:
            if not state["run"]:
                time.sleep(0.05); continue
            with lock:
                # With the curriculum ablated, training targets the clicked cell.
                trainer["t"].fixed_goal = state["goal"]
                snap = trainer["t"].update()
                done = training_finished(trainer["t"])
            goal = state["goal"]
            snap["done"] = done
            # The heatmap is a 4096-row forward pass. Doing it every update
            # roughly halves the training rate for a picture that barely changes
            # between consecutive updates, so during training it runs every 5th.
            heavy = done or snap["update"] % 5 == 0
            # V(s) and the policy field are drawn for the CURRENT goal, so they
            # change the instant you click somewhere else -- that is the whole
            # point of a goal-conditioned value function, made visible.
            if show_heat.get() and heavy:
                stride = 2
                with lock:
                    ob, rr, ccx = obs_grid(env, goal, stride=stride)
                    with torch.no_grad():
                        v = trainer["t"].value(torch.from_numpy(ob)).numpy()
                # Scatter into an array AT THE STRIDE, not at full resolution.
                # Writing 64x64 samples into a 128x128 canvas leaves every other
                # cell NaN, and NaN renders as the "wall" colour -- the heatmap
                # came out as a grey checkerboard with the signal hidden inside it.
                m = (env.n + stride - 1) // stride
                grid = np.full((m, m), np.nan, np.float32)
                grid[rr // stride, ccx // stride] = v
                snap["vgrid"] = grid
                snap["stride"] = stride
            if show_arrows.get() and heavy:
                with lock:
                    ob, rr, ccx = obs_grid(env, goal, stride=max(4, N // 20))
                    with torch.no_grad():
                        a = trainer["t"].policy.dist(torch.from_numpy(ob)).probs.argmax(1).numpy()
                snap["field"] = (rr, ccx, a)
            q.put(snap)
            if done:
                # Do NOT clear state["traj"] here. This is the worker thread and
                # animate() runs on the Tk thread: clearing it between that
                # function's None-check and its next use crashed the GUI with
                # "'NoneType' object is not subscriptable" exactly when training
                # finished. The phase flip is enough -- the current replay plays
                # out, and next_trajectory() then returns a ready-phase episode.
                state["run"] = False
                state["phase"] = "ready"
            time.sleep(state["throttle"] / 1000.0)

    wthread = threading.Thread(target=worker, daemon=True)
    wthread.start()

    # -------------------------------------------------------------- plots ----
    def redraw_plots(snap):
        h = snap["hist"]
        if snap.get("vgrid") is not None:
            axv.clear()
            v = snap["vgrid"]
            st = snap.get("stride", 1)
            state["vgrid"] = v
            lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
            cmap = matplotlib.colormaps["viridis"].copy()
            cmap.set_bad("#4a4a55")            # walls: V is undefined there
            axv.imshow(v, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            if snap.get("field") is not None:
                rr, ccx, a = snap["field"]
                d = DIRS[a]
                # quiver's y axis points up, the image's points down -> negate
                axv.quiver(ccx / st, rr / st, d[:, 1], -d[:, 0],
                           color="#ffffff", alpha=.9,
                           scale=40, width=.003, headwidth=4)
            g_ = state["goal"]
            axv.plot(g_[1] / st, g_[0] / st, "*", color="#ff4d4d", ms=16, mec="k")
            axv.plot(START[1] / st, START[0] / st, "o", color="#7CFC00", ms=8,
                     mec="k")
            axv.set_title("V(s) for YOUR goal, and the action the policy would take",
                          fontsize=9)
            axv.set_xticks([]); axv.set_yticks([])
            figv.tight_layout(); cv.draw_idle()

        for a_, key, ttl, col in ((ax1, "success", "success rate (curriculum goals)", "#2e9e2e"),
                                  (ax2, "ret", "mean return", "#3355bb"),
                                  (ax3, "entropy", "policy entropy", "#c46a1e"),
                                  (ax4, "radius", "curriculum radius (cells)", "#7a4fbf")):
            a_.clear(); a_.grid(alpha=.3)
            a_.plot(h[key], color=col)
            a_.set_title(ttl, fontsize=9)
            a_.set_xlabel("update", fontsize=8)
        ax1.set_ylim(-.05, 1.05)
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
            state["pending"] = got.get("traj")
            redraw_plots(got)
            stat.config(text=(
                f"update {got['update']:>5}   episodes {got['episodes']:>7}\n"
                f"success {got['success']*100:>5.1f}%   hazard {got['hazard']*100:>5.1f}%"
                f"   mean return {got['ret']:>7.2f}\n"
                f"avg steps {got['steps']:>6.1f}   entropy {got['entropy']:>5.3f}"
                f"   radius {got['radius']:>5.0f}"))
        refresh_phase()
        if not state["quit"]:
            root.after(80, poll)

    HOLD = 25                       # frames to linger on the final frame

    def next_trajectory():
        """What to animate next, which depends entirely on the phase.

        While TRAINING: replay a practice episode the trainer just ran. Rolling
        a fresh episode here instead would compete with the trainer for the lock
        and slow the very thing you are waiting for.

        When READY: roll the trained policy from S to YOUR goal. The policy was
        never trained on that goal, so this is a generalisation test, not a
        rehearsal -- and nothing is learned from it.
        """
        if state["phase"] == "training":
            tr, state["pending"] = state["pending"], None
            return tr
        g_ = state["goal"]
        if env.wall[g_]:
            return None
        with lock:
            tr = trainer["t"].rollout(greedy=watch_greedy.get(), start=START,
                                      goal=g_, max_steps=args.steps)[3]
        state["deliveries"].append(tr["reason"] == "goal")
        return tr

    def animate():
        # Read once into a local: other callbacks (and, historically, the worker
        # thread) can swap state["traj"] out from under a multi-step function.
        tr = state["traj"]
        if tr is None:
            state["traj"] = next_trajectory(); state["frame"] = 0
        else:
            draw_frame()
            state["frame"] += 1
            if state["frame"] >= len(tr["path"]) + HOLD:
                state["traj"] = next_trajectory(); state["frame"] = 0
        if not state["quit"]:
            root.after(int(1000 / max(1, state["fps"])), animate)

    # ------------------------------------------------------------ controls ---
    def toggle():
        """Pause/resume while training; add another block of updates when ready."""
        if state["phase"] == "ready":
            state["target"] = trainer["t"].updates + args.more_updates
            state["phase"] = "training"
            state["auto_stop"] = False
            state["traj"] = None
            state["run"] = True
        else:
            state["run"] = not state["run"]
        refresh_phase()

    def reset_map():
        state["run"] = False; btn.config(text="▶  Train")
        with lock:
            seed = np.random.randint(10000)
            new = GridDeliveryEnv(n=N, n_walls=args.walls, n_hazards=args.hazards,
                                  block=args.block, seed=seed, max_steps=args.steps)
            trainer["t"] = Trainer(env=new, seed=seed, lr=args.lr,
                                   batch_eps=args.batch_eps, ent_coef=args.ent,
                                   use_baseline=not args.no_baseline,
                                   curriculum=not args.no_curriculum,
                                   max_steps=args.steps)
            env.__dict__.update(new.__dict__)
        state["goal"] = tuple(int(v) for v in env.random_free_cell())
        state["traj"] = None; state["frame"] = 0; state["snap"] = None
        state["phase"] = "training"; state["target"] = args.train_updates
        state["auto_stop"] = True; state["shown_phase"] = None
        state["last_check"] = 0; state["spot"] = None; state["passes"] = 0
        state["deliveries"] = []; state["pending"] = None
        for a_ in (ax1, ax2, ax3, ax4):
            a_.clear()
        axv.clear(); cv.draw_idle(); cc.draw_idle()
        draw_static(); draw_markers(); refresh_goal_label()
        stat.config(text="new map, fresh weights — training from scratch")
        state["run"] = True
        refresh_phase()

    def random_goal():
        state["goal"] = tuple(int(v) for v in env.random_free_cell())
        state["traj"] = None; state["frame"] = 0; state["vgrid"] = None
        state["deliveries"] = []
        draw_markers(); refresh_goal_label(); refresh_phase()

    def skip_training():
        """Stop training now and go straight to click-to-deliver."""
        state["run"] = False
        state["phase"] = "ready"
        state["traj"] = None; state["frame"] = 0
        draw_markers(); refresh_phase()

    btn = tk.Button(ctl, text="▶  Train", command=toggle, width=11,
                    font=("DejaVu Sans", 11, "bold"))
    btn.pack(side="left", padx=8)
    tk.Button(ctl, text="Stop & use it", command=skip_training,
              font=("DejaVu Sans", 10)).pack(side="left", padx=4)
    tk.Button(ctl, text="Random goal", command=random_goal,
              font=("DejaVu Sans", 10)).pack(side="left", padx=4)
    tk.Button(ctl, text="New map (fresh weights)", command=reset_map,
              font=("DejaVu Sans", 10)).pack(side="left", padx=4)
    tk.Checkbutton(ctl2, text="greedy", variable=watch_greedy, bg=BG,
                   font=("DejaVu Sans", 10, "bold")).pack(side="left", padx=(8, 4))
    tk.Checkbutton(ctl2, text="V(s) heat", variable=show_heat, bg=BG,
                   font=("DejaVu Sans", 10)).pack(side="left")
    tk.Checkbutton(ctl2, text="policy arrows", variable=show_arrows, bg=BG,
                   font=("DejaVu Sans", 10)).pack(side="left")
    tk.Label(ctl2, text="replay fps", bg=BG,
             font=("DejaVu Sans", 10)).pack(side="left", padx=(10, 0))
    sp = tk.Scale(ctl2, from_=5, to=120, orient="horizontal", length=130, bg=BG,
                  highlightthickness=0,
                  command=lambda v: state.__setitem__("fps", int(float(v))))
    sp.set(args.fps); sp.pack(side="left", padx=4)
    tk.Label(ctl2, text="slow training", bg=BG,
             font=("DejaVu Sans", 10)).pack(side="left", padx=(10, 0))
    tk.Scale(ctl2, from_=0, to=600, orient="horizontal", length=110, resolution=25,
             bg=BG, highlightthickness=0,
             command=lambda v: state.__setitem__("throttle", int(float(v)))
             ).pack(side="left", padx=4)

    def on_close():
        # Let the worker finish the forward pass it is in the middle of before
        # tearing down the interpreter. Killing a daemon thread while it is
        # inside torch aborts the process with "terminate called without an
        # active exception" -- an ugly crash on every clean exit.
        state["quit"] = True; state["run"] = False

        def finish():
            wthread.join(timeout=3.0)
            root.destroy()
        root.after(50, finish)

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Open at the size the layout wants, capped to what the screen can show.
    # Whatever does not fit is reachable by scrolling rather than lost.
    root.update_idletasks()
    _resize()
    root.geometry("%dx%d+20+20" % (
        min(content.winfo_reqwidth() + vbar.winfo_reqwidth() + 4, SW - 40),
        min(content.winfo_reqheight() + hbar.winfo_reqheight() + 4, SH - 90)))
    root.minsize(560, 400)

    draw_static(); draw_markers(); refresh_goal_label()
    # Training starts by itself. There is nothing useful to click at before a
    # policy exists, so making you press a button first was busywork.
    state["run"] = not args.no_autostart
    refresh_phase(); poll(); animate()
    stat.config(text="training from scratch — the map you see is being practised on")
    if args.demo_click:
        # Used by the automated GUI check: fire a synthetic click at a cell once
        # training has had a moment, so the ready-phase path gets exercised
        # without a mouse.
        dr, dc = (int(v) for v in args.demo_click.split(","))

        def fake_click():
            canvas.event_generate("<Button-1>", x=int(dc * SC + SC / 2),
                                  y=int(dr * SC + SC / 2))
        root.after(int(args.demo_click_after * 1000), fake_click)
    if args.screenshot_after:
        root.after(int(args.screenshot_after * 1000), on_close)
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Goal-conditioned delivery robot on a stochastic grid.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=int, default=GRID, help="map is GRID x GRID")
    ap.add_argument("--walls", type=int, default=16, help="number of wall blocks (10-20)")
    ap.add_argument("--hazards", type=int, default=14, help="number of hazard blocks (10-20)")
    ap.add_argument("--block", type=int, default=14,
                    help="max side length of a block; --block 1 gives single cells")
    ap.add_argument("--steps", type=int, default=MAX_STEPS,
                    help="truncation limit. Far goals need more: slipping costs "
                         "about 30%% over the optimal path length")
    ap.add_argument("--goal", type=str, default=None, help="headless goal, as row,col")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-eps", type=int, default=16)
    ap.add_argument("--ent", type=float, default=0.03, help="entropy bonus")
    ap.add_argument("--eval-eps", type=int, default=20)
    ap.add_argument("--map-px", type=int, default=0,
                    help="width of the map in pixels; 0 fits it to your screen")
    ap.add_argument("--fps", type=int, default=45)
    ap.add_argument("--throttle", type=int, default=0)
    ap.add_argument("--no-baseline", action="store_true",
                    help="ablate the value baseline AND the GAE bootstrap: weight "
                         "grad-log-pi by the raw return, as REINFORCE was first "
                         "written. Run it and compare.")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="ablate the curriculum: train on the far goal from the "
                         "start. Run it and watch nothing happen.")
    ap.add_argument("--train-updates", type=int, default=800,
                    help="GUI: updates to train before switching to "
                         "click-to-deliver. It stops earlier if it gets good "
                         "(see --target-success)")
    ap.add_argument("--target-success", type=float, default=0.85,
                    help="GUI: stop training early once this fraction of 16 test "
                         "deliveries -- from S to random cells, the thing you "
                         "will actually click -- succeeds, twice in a row")
    ap.add_argument("--more-updates", type=int, default=200,
                    help="GUI: how many updates the 'Train more' button adds")
    ap.add_argument("--no-autostart", action="store_true",
                    help="GUI: wait for the button instead of training on launch")
    ap.add_argument("--autostart", action="store_true",
                    help="accepted for symmetry; training now autostarts anyway")
    ap.add_argument("--screenshot-after", type=float, default=0,
                    help="close the window after N seconds (automated checks)")
    ap.add_argument("--demo-click", type=str, default=None,
                    help="fire a synthetic click at row,col (automated checks)")
    ap.add_argument("--demo-click-after", type=float, default=20.0,
                    help="when to fire --demo-click, in seconds")
    a = ap.parse_args()
    run_headless(a) if a.headless else run_gui(a)
