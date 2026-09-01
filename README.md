# VLA_learn — the things every VLA paper assumes you already know

Four runnable programs, each with a GUI you can watch learn — and the last one
drives a car in Gazebo through ROS 2.

```
VLA_learn/
├── supervised_learning.py       SL: recognise digits in real RGB photographs
├── reinforcement_learning.py    RL: a car learns to drive from A to B
├── grid_delivery_robot.py       RL: a robot delivers to a goal YOU click, on a
│                                    128x128 stochastic grid
├── behavioral_cloning/          BC: clone a human driver, then drive a Gazebo
│                                    car with the result, through ROS 2
│   ├── train.py                     dataset, NVIDIA CNN, live dashboard
│   ├── drive_node.py                the policy as a ROS 2 node
│   ├── collect.py / evaluate.py     a Gazebo expert, and lane-keeping scores
│   └── README.md                    the full write-up for that project
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

# Grid delivery — trains itself on launch, then you click where it should go
.venv/bin/python grid_delivery_robot.py

# Behavioral cloning — clone a human driver (needs --fetch once, 498 MB)
.venv/bin/python -m behavioral_cloning.train --fetch
.venv/bin/python -m behavioral_cloning.train

# ...then let the clone drive a car in Gazebo, headless
./behavioral_cloning/run_demo.sh all
```

Press **▶ Train** in the first two windows — nothing trains until you do.
`grid_delivery_robot.py` is the exception: it trains itself on launch, then
switches to click-to-deliver (pass `--no-autostart` to wait for the button).

```bash
# terminal-only versions
.venv/bin/python supervised_learning.py    --headless --epochs 12
.venv/bin/python reinforcement_learning.py --headless --updates 120

.venv/bin/python grid_delivery_robot.py    --headless --updates 800 --goal 127,127

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
| **Reward** `R` | `+10` goal (end), `−10` hazard (end), `−0.01`/step, `−0.05` extra when blocked, `+0.10 ×` Manhattan progress. |
| **terminated** | reached G, or entered H — *the task ended*. |
| **truncated** | 512 steps elapsed — *a stopwatch ended it*. Bootstrapped differently, and §1 explains why that matters. |
| **γ / λ** | 0.995 / 0.95 |

**Algorithm:** REINFORCE + value baseline + **GAE(0.95)** + a distance
**curriculum**. The clicked goal is never trained on, so what the GUI replays is
a real generalisation test.

**The GUI runs in two phases, and starts itself.** There is nothing useful to
click at before a policy exists, so it begins training the moment you launch it:

1. **TRAINING** — the robot practises on curriculum goals you never chose, and
   the canvas replays those practice episodes. The banner tracks progress. It
   stops on its own when **16 test deliveries** — greedy runs from S to random
   cells, i.e. the thing you are about to click — clear `--target-success`
   (default 0.85) **twice in a row**, or after `--train-updates` (default 800),
   whichever comes first. **Stop & use it** cuts it short whenever you like.

   The banner shows *two* success numbers and they disagree on purpose:
   `practice` (the curriculum success rate) passes 90% by update ~120, while
   `test deliveries` — the real task — lags well behind it. Stopping on the
   first would hand you a policy that looks trained and then misses half the
   cells you click. Picking the right thing to measure is the whole lesson.
The window sizes itself to your screen and scrolls if it does not fit — mouse
wheel, Shift+wheel sideways, PgUp/PgDn/Home/End. On a small screen the map and
the V(s) panel stack rather than sitting side by side; `--map-px` overrides the
map size if you want it bigger or smaller than the automatic choice.

2. **READY** — training has stopped. **Click any cell** and the trained policy
   drives from S to it, with a running delivered-N-of-M tally. Nothing is
   learned in this phase, so you are watching the finished policy rather than
   one that is quietly still improving. **Train more** goes back to phase 1.

**Result** (800 updates, 12,800 episodes, 427 s on CPU), on goals it was never
trained on, from the depot: **60/60** delivered for routes under 30 cells,
**60/60** for 30–80, **44/60** for 80–150, **49/60** for 150–256 — all at
**≈1.3× the optimal path length**, which is essentially the floor, since 20% of
moves slip and `1/0.8 = 1.25×`.

**The step budget is 512, not the 256 the exercise suggests**, and that is the
single change that made long routes work (the 150–256 band went from 36/60 at
256 steps to 49/60 at 512). The far corner is 254 cells away and slipping
inflates that to ~330 steps, so at 256 a third of the map is unreachable *by
construction* — which would quietly break the "click anywhere" promise. Every
goal you click is still labelled green / orange / red, so an impossible ask is
visible before you make it. Use `--steps 256` for the exercise as literally
stated.

The doc also carries **Exercise 2** — the UR5e pick-and-place design: privileged
state vs. deployable observation, why the action is relative TCP and not joint
velocities, the seven reward terms with coefficients, and three ways the robot
can farm that reward without ever doing the task.

### `behavioral_cloning.py` — copying a driver, and watching the copy break

The first three files argue that **behavior cloning *is* supervised learning**
and that it fails by **covariate shift**. This one makes both claims executable,
and then drives a real robot stack with the result.

Built on [seraj94ai's behavioral-cloning
project](https://github.com/seraj94ai/A-self-driving-car-using-behavioral-cloning)
and NVIDIA's *End to End Learning for Self-Driving Cars* (Bojarski et al. 2016).
That repo has **no licence file**, so none of its code is copied here — this is
an independent PyTorch implementation and `--fetch` clones their repository
locally for the recordings. It also could not run as-is: it is Keras 1.x, and
its `driving_log.csv` stores Windows paths that `os.path.basename` cannot split
on Linux. Both fixed; the 252,219-parameter count from NVIDIA's table is
asserted by a test.

**The statistic that decides the whole problem:** 78.1% of the recorded steering
angles are *exactly zero*. Train on that and MSE is minimised by predicting ~0
forever — a great validation loss, and a car that drives into the first corner.
Capping each steering bin is worth more than any architectural choice in the
file.

**On the original data** (30 epochs, CPU, 273 s): validation MSE **0.0837** vs
0.1756 for predicting the mean, steering correlation **0.657**.

**Then it drives.** `ros2_bc_driver.py` turns the network into a ROS 2 node —
`sensor_msgs/Image` in, `geometry_msgs/Twist` out — and `gazebo/` provides a
figure-eight track (an oval would let a constant steering angle "win"), a
pure-pursuit expert with privileged ground truth, and an evaluator that scores
lane-keeping in metres. Cloning 3,428 expert frames, then driving at 0.9 m/s on
a 3 m lane:

| policy | mean cross-track error | left the road? |
|---|---|---|
| null baseline — ignores the camera | 36.5 m | **after 5.5 s** |
| the expert (privileged ground truth) | 0.075 m | no |
| **the clone (camera only)** | **0.117 m** | no |
| the same net trained on **Udacity** data | 0.49 m | no |

**The clone drives, within 1.6× of its teacher** — and the null baseline is what
makes that claim mean anything: a car that ignores the camera leaves this track
in five and a half seconds.

Training opens a **live dashboard** that updates on every weight update: the
camera frame *and* the 66×200 tensor the network actually sees, the human's
action against the model's, per-batch loss with the gradient norm beside it, and
a timestamped event log. Its README also covers what else could drive this car —
DAgger, inverse RL, GAIL, why RL is not used on real roads, and how production
self-driving stacks are actually built.

The doc is equally clear about what did **not** happen: the compounding-error
failure covariate shift predicts never materialised, because the lane is wide,
the expert is slow and smooth, and six laps cover the states densely. Saying so
is more useful than claiming a failure that was not observed. The last row is
**domain shift**, and it is the interesting one — a policy trained on a
photoreal canyon road still keeps the Gazebo car between the lines while being
four times worse, which is exactly the "works, but wrong" outcome that makes
sim-to-real dangerous.

`behavioral_cloning/README.md` keeps **six bugs** on the record, including wheels
that spun about a vertical axis (an SDF joint axis is expressed in the child
link's frame) and an evaluator that silently ignored `--model` and scored the
same checkpoint four times — which nearly published a wrong headline.

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
7. Run `grid_delivery_robot.py`. Let it train itself (a few minutes — watch the
   banner), then click cells and see where it goes. Click a far corner and read
   the coloured feasibility line before judging it. Then read
   `docs/grid_delivery_robot.md` §1 (the MDP answer sheet), §2.3 (why the robot
   needs one step of memory — found by watching it deadlock), §3.2b (why the
   training success rate is the wrong thing to stop on), §3.3 (the policy that
   stopped reading the goal at all) and §6 (three bugs that produced perfectly
   plausible training curves).

Every number quoted in the docs was measured on this machine and is reproducible
with a flag. Where a result contradicted what I expected — the RGB ablation, the
dropout sweep — the docs say so.
