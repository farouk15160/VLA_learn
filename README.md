# VLA_learn — the things every VLA paper assumes you already know

Three runnable programs, each a single file, each with a GUI you can watch learn.

```
VLA_learn/
├── supervised_learning.py       SL: recognise digits in real RGB photographs
├── reinforcement_learning.py    RL: a car learns to drive from A to B
├── grid_delivery_robot.py       RL: a robot delivers to a goal YOU click, on a
│                                    128x128 stochastic grid
├── docs/
│   ├── supervised_learning.md   the data spec, the model, and the 8 ways SL lies to you
│   ├── reinforcement_learning.md  RL from zero: the MDP, policy gradients, baselines
│   └── grid_delivery_robot.md   MDP DESIGN: state vs observation, reward exploits,
│                                    and the two exercise answer sheets
└── data/svhn_test_32x32.mat     61 MB, downloaded on first run
```

Both GUIs render live rather than writing files, so there is nothing to clean up
after a run.

## Setup

```bash
git clone https://github.com/farouk15160/VLA_learn.git && cd VLA_learn
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Tkinter is not on PyPI — it ships with Python. On Debian/Ubuntu:
`sudo apt install python3-tk`. A GPU is optional; the SL file uses CUDA if it
finds one, and the RL file deliberately runs on CPU (its bottleneck is the
Python physics loop, not matrix maths).

The 62 MB SVHN dataset is **not** in this repo — `supervised_learning.py`
downloads it into `data/` on first run.

---

## Run

```bash
# Supervised — GUI: watch 40 photos turn green, then upload your own image
.venv/bin/python supervised_learning.py

# Reinforcement — GUI: watch a car learn to drive, and watch V(s) grow a route
.venv/bin/python reinforcement_learning.py

# Grid delivery — GUI: CLICK ANYWHERE to place the goal, watch it generalise
.venv/bin/python grid_delivery_robot.py
```

Press **▶ Train** in either window. Nothing trains until you do.

```bash
# terminal-only versions
.venv/bin/python supervised_learning.py    --headless --epochs 12
.venv/bin/python reinforcement_learning.py --headless --updates 120

.venv/bin/python grid_delivery_robot.py    --headless --updates 800 --goal 100,110

# the ablations quoted in the docs, so you can reproduce every number
.venv/bin/python supervised_learning.py    --sweep
.venv/bin/python reinforcement_learning.py --headless --updates 120 --no-baseline
.venv/bin/python grid_delivery_robot.py    --headless --updates 400 --no-curriculum
```

---

## What each one is

### `supervised_learning.py` — a teacher gives the answer

**Data:** SVHN, 26,032 32×32 RGB crops of house numbers photographed from Google
Street View. Real photos: blurry, badly lit, with neighbouring digits intruding.
**Model:** a 667k-parameter CNN. **Result:** ~95% test accuracy.

This is the shape of a VLA action head. OpenVLA and RT-2 discretize each robot
action dimension into 256 bins and train it with cross-entropy on the resulting
tokens — image in, class logits out. **Behavior cloning *is* supervised
learning**, so this file is closer to how a real VLA is trained than the RL one.

Answers, with measurements, in `docs/supervised_learning.md`: how the RGB
channels are laid out and normalized (and whether colour actually helps — it
does not), why 32×32, what `CrossEntropyLoss` computes, what dropout does, why
`zero_grad()` is mandatory, what an epoch is and how many to run, and the
difference between validation and test.

### `reinforcement_learning.py` — a critic gives a score

**The MDP**, stated exactly (full detail in `docs/reinforcement_learning.md` §2):

| | |
|---|---|
| **State** `S` | 10 floats: 7 lidar distances + `cos`/`sin` of the bearing to the goal + range to goal. **No absolute `(x,y)`** — the policy must drive by what it senses. |
| **Actions** `A` | 3: steer left, straight, steer right. Speed is constant. |
| **Transition** `P` | `heading += (a−1)·0.30`, then move 1.6 units. Deterministic, and **not differentiable by the agent** — which is why we need a policy gradient. |
| **Reward** `R` | `+0.10 × (progress toward goal)` per step, `−0.002` time cost, `−3.0` and end on crash, `+5.0` and end on arrival. |
| **γ** | 0.99 |

**Algorithm:** REINFORCE with a learned value baseline. **Result:** 0% → 100%
success in ~60 updates (~480 episodes, ~15 seconds), 30/30 on greedy evaluation.

The reward is *shaped* because a sparse "+1 for reaching B" is unlearnable here —
random steering never reaches B, so the gradient would be zero forever. It is
shaped as a **difference of distances**, which cannot be farmed by circling.

### `grid_delivery_robot.py` — the goal moves, and the policy has to cope

The other two files learn **one** thing. This one learns a **goal-conditioned**
policy: you click a cell, and the same weights have to get there. That is the
difference between a robot that memorised a route and a robot that can navigate,
and it is exactly how a VLA is conditioned — swap "goal cell" for "language
instruction" and the structure is identical.

**The MDP** (full detail in `docs/grid_delivery_robot.md` §1, which is the
completed exercise answer sheet):

| | |
|---|---|
| **State** `s_t` | robot cell, goal cell, step counter, and the two maps. |
| **Observation** `o_t` | **≠ `s_t`.** 334 floats: a 9×9 occupancy patch, a blurred 57×57 one, the goal as a *relative* offset, the clock, and the previous action. **No absolute `(row, col)`** — so it must navigate rather than memorise. |
| **Actions** `A` | 4: `UP, RIGHT, DOWN, LEFT`. Discrete. No `WAIT`, on purpose. |
| **Transition** `P` | FrozenLake-style: **0.8** intended, **0.1** each perpendicular. Into a wall or off-map → stay put. |
| **Reward** `R` | `+10` goal (end), `−10` hazard (end), `−0.02`/step, `−0.05` extra when blocked, `+0.10 ×` Manhattan progress. |
| **terminated** | reached G, or entered H — *the task ended*. |
| **truncated** | 256 steps elapsed — *a stopwatch ended it*. Bootstrapped differently, and §1 explains why that matters. |
| **γ / λ** | 0.995 / 0.95 |

**Algorithm:** REINFORCE + value baseline + **GAE(0.95)** + a distance
**curriculum**. The clicked goal is never trained on, so what the GUI replays is
a real generalisation test.

**Result** (800 updates, 12,800 episodes, ~6 min on CPU), on goals it was never
trained on: **59/60** delivered for routes under 30 cells, **57/60** for 30–80,
**41/60** for 80–150 — at **≈1.3× the optimal path length**, which is
essentially the floor, since 20% of moves slip and `1/0.8 = 1.25×`.

The failures are the step budget, not the policy: **254 cells needs ~320 steps
and the budget is 256.** The GUI labels every goal you click green / orange /
red before you press Train, so you know whether you are watching a bad policy or
an impossible ask — and in the run quoted in the doc, that label predicted the
failure correctly in advance. Use `--steps 400` if you want the whole map
reachable.

The doc also carries **Exercise 2** — the UR5e pick-and-place design: privileged
state vs. deployable observation, why the action is relative TCP and not joint
velocities, the seven reward terms with coefficients, and three ways the robot
can farm that reward without ever doing the task.

---

## The one-paragraph summary

Supervised learning has a teacher who gives the right answer; reinforcement
learning has a critic who gives a score and leaves you to work out which of your
80 steering decisions earned it. Today's VLAs (RT-2, OpenVLA, π₀) are trained by
**supervised learning** — behavior cloning on human demonstrations — so the SL
file is closer to real VLA training than the RL one. But behavior cloning has a
structural failure called **covariate shift**, where the policy's own small
errors carry it into states it was never trained on and errors compound
quadratically with the horizon, and RL is the vocabulary for understanding and
fixing that. **Learn BC to build a VLA; learn RL to understand why it fails.**

---

## Suggested order

1. Run `supervised_learning.py`. Watch the tiles go green, upload something.
2. Skim `docs/supervised_learning.md` §2 (the data spec) and §4 (the eight
   failure modes). §2.3 is the RGB question, answered with an ablation.
3. Read `docs/reinforcement_learning.md` §1–§4 **before** running the RL file.
   §2 is this exact MDP; §4 is the catalogue of what makes RL hard.
4. Run `reinforcement_learning.py`. Tick **watch greedy policy** to see a clean
   drive rather than a wobbly exploring one.
5. Read §6 (the policy gradient derivation — three lines, worth doing by hand
   once) and §8 (what the baseline is actually worth: 3/3 seeds vs 1/3).
6. Read §10 for how all of it connects to VLA, DAgger and action chunking.
7. Run `grid_delivery_robot.py`, click a far corner, and read the coloured
   feasibility line before judging it. Then read
   `docs/grid_delivery_robot.md` §1 (the MDP answer sheet), §2.3 (why the robot
   needs one step of memory — found by watching it deadlock), §3.3 (the policy
   that stopped reading the goal at all) and §6 (three bugs that produced
   perfectly plausible training curves).

Every number quoted in the docs was measured on this machine and is reproducible
with a flag. Where a result contradicted what I expected — the RGB ablation, the
dropout sweep — the docs say so.
