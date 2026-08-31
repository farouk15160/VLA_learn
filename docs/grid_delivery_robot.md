# Grid delivery robot — a policy that goes wherever you point

Companion to `grid_delivery_robot.py`. This is the third file in the repo and
the one that answers the two MDP-design exercises. Read
`docs/reinforcement_learning.md` first if the words *policy gradient* are new;
this document assumes them and spends its time on **problem design** instead —
what you put in the state, what you pay for, and how each of those choices came
back to bite me.

---

## 0. The task, exactly as posed

A robot delivers a parcel on a grid warehouse.

```
S . . .          S  start, cell (0, 0)
. X . .          G  the delivery goal — YOU choose it
. . . H          X  wall, cannot be entered
. . . G          H  hazardous cell
                 .  normal floor
```

* one action per step, and the robot knows its own cell;
* a move works **80%** of the time, and slips to each perpendicular direction
  **10%** of the time;
* a move into a wall or off the map leaves the robot where it stood;
* the episode is cut off after a fixed number of steps.

That is the FrozenLake transition model. The version in this repo is scaled up
to what you asked for: **128×128**, **10–20 walls**, **10–20 hazards** placed at
random, a goal you pick by clicking, and a **512-step** budget (§5 explains
why not 256).

### What scaling up actually changed

Not the maths — the maths is identical. What it changed is which *solution
methods* remain available, and that is the interesting part.

| | 4×4 as written | 128×128 as built |
|---|---|---|
| states (robot cell only) | 16 | 16,384 |
| states (cell **×** goal) | 256 | 268 million |
| solve it with a table? | yes, instantly | not per goal, no |
| solve it with value iteration? | yes | yes, but you must re-solve for every new goal |
| sparse "+1 at G" reward | works, random walk finds G | **gradient is zero forever** |
| the hard part | nothing | the goal moves, and the map is bigger than the view |

The middle row is the one that forces the design. Value iteration would solve
any *single* goal on this map in about a second — but you move the goal by
clicking, and re-solving from scratch on every click is not learning, it is
planning with a known model. So this file learns a **goal-conditioned policy**:
the goal is an *input* to the network, and one set of weights has to handle
wherever you put it.

That is the same structure as a VLA. Swap "goal cell" for "language
instruction" and "occupancy patch" for "camera image" and you have the
conditioning pattern that RT-2 and OpenVLA use: one policy, many tasks,
the task specified as part of the observation.

---

## 1. Answer sheet — Exercise 1

Filled in for the implemented 128×128 version. Where the original 4×4 would
differ I say so.

### State

```
s_t = ( robot_cell, goal_cell, t, wall_map, hazard_map )
    = ( (row, col) ∈ [0,128)² , (grow, gcol) , t ∈ [0,512] , W , H )
```

Taking the questions in order:

* **Is the robot position enough?** No — but not for the reason people expect.
  It is enough to make the *dynamics* Markov: where you go next depends only on
  where you are and what you press. It is not enough to make the *task* defined,
  because the goal moves. `(row, col)` alone cannot tell you whether to walk
  north or south.
* **Do you need the previous position?** For the dynamics, no. For a policy that
  has to act without deadlocking, **yes, something like it** — and I only found
  that out by watching it fail. See §4.3. The implementation carries the
  previous *action*, which is smaller and does the same job.
* **Do you need the elapsed steps?** **Yes.** Truncation at 512 makes the value
  of a state depend on the clock: with 200 steps left, walking around a hazard
  is correct; with 3 steps left, nothing is worth doing. Leave `t` out and the
  value function is being asked to predict two different numbers from one input,
  and it splits the difference. This is the standard "time-limit bootstrapping"
  trap (Pardo et al. 2018).
* **Do you need the complete map?** It belongs in `s_t` — the *environment*
  needs it to compute transitions. It does not belong in `o_t`. See below.

### Observation

```
o_t = [ wall_patch_9×9 , hazard_patch_9×9 ,          # 162 — what is next to me
        wall_coarse_9×9 , hazard_coarse_9×9 ,        # 162 — a 57-cell-wide blur
        Δrow/128 , Δcol/128 ,                        #   2 — signed offset to G
        Δrow/‖Δ‖ , Δcol/‖Δ‖ ,                        #   2 — unit direction to G
        manhattan(pos,G)/256 ,                       #   1 — how far
        1 − t/512 ,                                  #   1 — clock
        onehot(previous action) ]                    #   4 — one step of memory
                                                     # ─── 334 floats
```

**`o_t ≠ s_t`. This is a POMDP, deliberately.** Three things are hidden:

1. **The absolute cell `(row, col)`.** Give it to the network and it memorises
   *this* map — a lookup table with extra steps. Withhold it and the only thing
   it *can* learn is "walk toward the goal, dodge what is beside me", which
   transfers to a map it has never seen. Same argument as the lidar car in the
   sibling file, and the same argument for why robot policies get camera images
   rather than ground-truth object poses.
2. **The map beyond 28 cells.** The robot sees a 9×9 window and a blurred 57×57
   one. It cannot see a wall on the far side of the warehouse, so it cannot plan
   around one. It has to discover it.
3. **Which way it will actually slip.** Nothing observes the 80/10/10 draw. This
   is irreducible: it is what makes the problem stochastic rather than a maze.

The goal is given as a **relative offset, never an absolute cell.** "The parcel
is 40 cells north" is directly actionable; "the parcel is at (12, 96)" requires
the robot to also know where *it* is standing and do the subtraction — which we
just refused to tell it.

### Actions

```
A = { 0 UP, 1 RIGHT, 2 DOWN, 3 LEFT }        |A| = 4
```

**Discrete.** One action per step, no diagonals, and deliberately **no WAIT**:
a wait action only creates a way to burn the clock, and with a negative step
cost it could never be optimal anyway. Why give the optimiser a rope?

The list is in **rotational order**, which is not cosmetic — it makes the two
perpendicular slips of action `a` exactly `(a+1) % 4` and `(a-1) % 4`, and the
slip model becomes one line instead of a lookup table.

### Rewards

| Event | Reward |
|---|---|
| Reach **G** | **+10.00**, and terminate |
| Enter **H** | **−10.00**, and terminate |
| Normal step | **−0.01** |
| Hit wall/boundary | **−0.01 − 0.05 = −0.06** |
| *(shaping)* each cell of Manhattan progress | **+0.10 × (d_prev − d_now)** |

Discount **γ = 0.995**.

Now the four traps the exercise asks about, answered with the numbers above.

* **Could the robot earn more by waiting forever?** No. Every step costs −0.01
  and there is no action with non-negative reward, so the value of dithering for
  512 steps is −5.12 against +10 for delivering. Standing still is strictly
  dominated. This is exactly why the step penalty must be *negative* rather than
  zero: at zero, "never risk the hazard, just idle" ties with success under any
  discount, and a risk-averse policy will take the tie.
* **Is the hazard penalty strong enough?** It has to beat the shaping the robot
  banked walking in. A hazard sitting 12 cells nearer the goal has already paid
  `12 × 0.1 = +1.2`. At `R_H = −1.0` the net is **+0.2 — walking into the
  hazard is profitable**, and the robot learns to do it. At −10 it is −8.8 and
  the shortcut is dead. **The rule is `|R_H| > R_progress × (map diameter)` for
  the hazards you actually care about.** A reward-design bug like this does not
  look like a reward-design bug from outside; it looks like a stupid agent.
* **Will a step penalty encourage the shortest safe route?** Yes, and the
  shaping term does most of that work — `−0.01` alone is a weak preference over
  512 steps (5.12 total, half a delivery). Its real job is breaking ties between
  equally-short routes and killing idling.
* **Could an excessively large step penalty encourage dangerous shortcuts?**
  Yes, and this is the sharp edge. Suppose the safe route is 40 cells longer
  than the route through a hazard. With step penalty `c`, the detour costs
  `40c`; the shortcut costs `0.2 × (extra hazard risk) × 10` per risky step from
  slipping. Push `c` to −0.5 and the detour costs 20 — comparable to the +10
  delivery itself, and cutting the corner starts to win. **The safe range is
  `|R_step| < R_goal / typical_path_length`**, i.e. under 10/512 ≈ 0.0195 here.
  0.01 is a little under half of that — a 1.95× margin, which is why it is the
  number. Note the bound **moves with the horizon**: doubling the step budget
  from 256 to 512 halved it, so the step penalty had to halve with it. `tests/test_grid_delivery_robot.py` asserts exactly this, so the
  margin cannot be edited away by accident.

**Why shaping at all?** On 4×4 you do not need it: a random walk hits G in a few
dozen steps and a sparse +1 is perfectly learnable. On 128×128 a random walk
essentially never arrives, every episode returns the same 0, the gradient is
*exactly* zero, and nothing is learned — ever. So the reward is shaped by a
**difference of distances**. That form is potential-based (Ng, Harada & Russell
1999) and therefore does not change which policy is optimal — and it cannot be
farmed by pacing back and forth, because a step away costs precisely what the
step back pays.

### Termination and truncation

```
terminated = True   iff  robot_cell == G          (success)
                    or   hazard[robot_cell]        (failure — the robot is lost)

truncated  = True   iff  t >= 512   and not terminated
```

They are tracked separately because **they mean different things to the value
function**, and conflating them is the most common bug in hand-rolled RL:

* `terminated` — the future after this state is worth **exactly 0**. Nothing
  more can happen.
* `truncated` — the future is worth **V(s_last)**. The robot was mid-journey and
  an external stopwatch cut it off; the journey still had value.

Bootstrap the truncated case with 0 and you teach the value net that *running
out of clock is as bad as dying*, which makes it pessimistic about every long
route and therefore unwilling to attempt them. `Trainer.update()` handles this
explicitly — `traj["terminated"]` picks between `0.0` and `V(last_obs)`.

### Transition probabilities

From the corner, action RIGHT — the case with a self-loop, which is the one
worth writing:

$$P\big(s_{t+1}=(0,1)\ \big|\ s_t=(0,0),\ a_t=\text{RIGHT}\big)=0.8$$
$$P\big(s_{t+1}=(1,0)\ \big|\ s_t=(0,0),\ a_t=\text{RIGHT}\big)=0.1 \quad\text{(slips DOWN)}$$
$$P\big(s_{t+1}=(0,0)\ \big|\ s_t=(0,0),\ a_t=\text{RIGHT}\big)=0.1 \quad\text{(slips UP, off-map, stays)}$$

The general rule, with `N(s,d)` the cell you reach moving in direction `d`
from `s` (equal to `s` itself if that is a wall or off-map):

$$P(s'\mid s,a)=0.8\,[\,s'=N(s,a)\,]+0.1\,[\,s'=N(s,a{+}1)\,]+0.1\,[\,s'=N(s,a{-}1)\,]$$

with the bracket terms *summed*, not selected — that is what produces the
self-loop when two of the three outcomes land on the same cell.

**The consequence that matters:** slip probability does not disappear next to a
wall, it turns into a wasted step. Hugging a wall is quietly expensive, and a
route one cell further from the wall is often *faster in expectation* despite
being longer. The robot works this out on its own; you can watch it in the
policy arrows.

**The consequence that matters more:** a 254-cell route does not take 254 steps.
20% of moves go sideways, so it takes roughly `254 / 0.8 ≈ 318` — **more than
the 256 steps the exercise suggests**, which is why the default is 512. See §5.

---

## 2. Why the observation is shaped like that

### 2.1 One view is not enough

The first version gave the robot a single 9×9 window — four cells of vision in
each direction. The wall blocks are up to 14 cells long. A robot pressed against
the middle of one **cannot see either end of it**: the observation for "go
around to the left" and "go around to the right" is *identical*, and no policy,
however good, can distinguish them. It ground into the wall, because that is all
the information supported.

The fix is a second, zoomed-out view of the same map: max-pool into 7×7 tiles
("is there *any* wall in this tile?") and sample every 7th cell. That is a
57-cell-wide field of view for 81 more numbers, and it is what makes a detour a
*learnable* thing rather than an unobservable one.

This is the grid-world version of a real design decision. It is why robot
policies get a wrist camera **and** a scene camera: one for precision, one for
context, and neither is sufficient alone.

### 2.2 The clock has to be in there

Covered above, but the symptom is worth naming: without `1 − t/512` the value
net's predictions for a state early and late in an episode are forced to be the
same number, the GAE residuals stay large no matter how long you train, and the
advantage estimates stay noisy. It looks like a learning-rate problem.

### 2.3 The robot needs one step of memory

This one I did not predict. With the observation above minus the previous
action, the trained policy would walk to within **three cells** of a goal tucked
against a wall and then do this, forever:

```
(42,43) → (42,44) → (42,43) → (42,44) → …    until the clock ran out
```

A memoryless policy is a *function* of the observation. In a symmetric pocket,
the observation at the two cells maps to actions that undo each other, and the
policy has no way to break the tie — it is not confused, it is **deterministic
and trapped**. Sampling instead of arg-max does not save it either, because by
then the policy is confident.

Feeding back the previous action makes "arrived here going left" a different
observation from "arrived here going right", which is enough to break the
symmetry. It is the cheapest possible memory, one one-hot vector, and it is the
honest answer to the exercise's *"do you need the previous position?"* — you
need **something**, and the previous action is smaller and works.

The general lesson generalises past grids: **if your agent oscillates, ask
whether the two states it oscillates between are distinguishable in the
observation.** Usually they are not.

---

## 3. The algorithm

REINFORCE with a learned value baseline, same as `reinforcement_learning.py`,
plus two things the longer horizon forced.

### 3.1 GAE instead of the raw return

The car file weights `∇log π` by the Monte-Carlo return minus a baseline. That
is unbiased, but its variance grows with episode length, and episodes here run
to 512 steps rather than 60. So this file uses **GAE(λ=0.95)**:

```
δ_t = r_t + γ V(s_{t+1}) − V(s_t)          one-step TD error
A_t = δ_t + (γλ) A_{t+1}                   exponentially-weighted sum of them
```

`λ = 1` recovers the Monte-Carlo estimator exactly; `λ = 0.95` accepts a little
bias from the value net in exchange for far less noise. On this task that trade
is what makes it train at all rather than a nicety.

### 3.2 A curriculum, or nothing happens

Sampling a goal uniformly on a 128×128 map means the average goal is ~85 cells
away, which early on means hundreds of steps of noise and no arrival. So training
samples a **random start and a goal within a radius that grows**, and shortens
the episode budget to match:

```
radius starts at 10
if mean success over the last 10 updates > 0.80:   radius ×= 1.10   (cap 256)
episode budget = clamp(4 × radius + 30, 30, 512)
```

Two details that are the difference between working and not:

* **Promote on the mean of the last 10 updates, not on one batch.** My first
  version promoted whenever a single batch of 8 episodes cleared 75%. Noise
  alone clears that regularly, the radius walked to the map diagonal within 150
  updates on essentially no skill, and the robot then trained forever on goals
  it could not yet reach. Success flatlined at 0.5.
* **Shrink the episode budget with the radius.** While goals are 10 cells away
  there is no reason to simulate 512 steps of wandering. It is a 5× speedup for
  free, early on, when it matters most.

**The clicked goal is never trained on.** Training runs on curriculum goals you
never see; when the GUI reaches its ready phase and you click, the episode runs
from S to *your* goal. So what you are watching is a genuine generalisation
test, not a rehearsal.

### 3.2b When to stop training — and why the obvious signal is the wrong one

The GUI trains itself and then hands you the policy, so something has to decide
when it is done. The obvious candidate is the training success rate, and it is
**wrong**: it is measured on curriculum goals (a random start, a goal sampled
near it), and it passes 0.90 by update ~120 — long before the robot can cross
the map from the depot. Stop on it and you hand over a policy that looks trained
and then misses half the cells you click.

So the stopping rule measures the actual task instead: every 25 updates, once
the curriculum has reached the far end of the map, run **16 greedy deliveries
from S to random reachable cells** and stop when `--target-success` of them
arrive (default 0.85) **twice in a row**.

The "twice in a row" is not defensive padding, it is a bug fix. The first
version used a single 8-episode check, and it fired at **update 117** on a
policy that then delivered 3 of 6 to a 165-step goal. Seven-of-eight comes up
often enough by chance when the true rate is ~0.7, and a noisy test that stops
training early is worse than no test — it hands over a bad policy *with
confidence*. Sixteen episodes passing twice is a far harder coincidence and
costs about two seconds.

The banner shows both numbers, `practice` and `test deliveries`, and watching
them diverge is the clearest demonstration in this repo of why *what* you
measure matters more than how carefully you measure it.

### 3.3 Hyperparameters, and one that mattered a lot

```
policy / value : 334 → 256 → 256 → {4, 1},  tanh
optimiser      : Adam, 5e-4 (policy) / 1e-2 (value)
batch          : 16 episodes per gradient step, advantages normalised
entropy bonus  : 0.03
γ = 0.995,  λ = 0.95,  gradient clip 2.0
```

The first version ran at `lr = 3e-3, batch = 8, ent = 0.01` — the car file's
settings. It trained to a **success rate of 0.5 that would not move**, and the
reason was not obvious until I probed the policy directly:

```
robot at (60,60), goal to the NORTH  → policy says RIGHT   [0.00 1.00 0.00 0.00]
robot at (60,60), goal to the WEST   → policy says DOWN    [0.00 0.00 0.90 0.10]
```

It had **stopped reading the goal at all** and collapsed onto a constant
"south-east" drift, which reaches ~half of nearby goals by luck. Classic
premature entropy collapse: a large step size on normalised advantages saturates
a 4-way softmax within a few dozen updates, and a policy that has stopped
exploring cannot discover that the goal input is worth using.

Lowering the learning rate to 5e-4, doubling the batch, and tripling the entropy
bonus fixed it. The metric worth watching is not the reward — it is **direction
accuracy**: over random (position, goal) pairs, how often does the arg-max
action reduce the distance? Collapsed policy ≈ 0.25 (chance). Fixed: **0.98**.
Reward curves hide this failure; that one number does not.

---

## 4. Results

Measured on the default map (seed 0, 16 wall blocks / 746 cells, 14 hazard
blocks / 578 cells), 800 updates ≈ 12,800 episodes, 427 s on a CPU.

Reproduce with:

```bash
.venv/bin/python grid_delivery_robot.py --headless --updates 800 --goal 127,127
```

Final training batch: **94% success on curriculum goals, 0% hazard deaths**,
curriculum radius at the cap.

Generalisation to 20 random goals per band that the policy was **never trained
on** (3 episodes each, greedy, from the depot at S):

| optimal route | delivered | steps taken vs optimal |
|---|---|---|
| 1–30 cells | **60/60** | 1.41× |
| 30–80 cells | **60/60** | 1.31× |
| 80–150 cells | **44/60** | 1.33× |
| 150–256 cells | **49/60** | 1.31× |

Three things in that table are worth stopping on.

**The ≈1.3× ratio is not a flaw — it is close to the floor.** With 20% of moves
going sideways, *no* policy can average much better than `1/0.8 = 1.25×` the
shortest path. The robot is running within a few percent of the physical limit
of the transition model, and it holds that ratio at every distance.

**The short-route ratio is the worst one (1.41×), and that is not a mistake
either.** A 6-cell route with one unlucky slip is 8 steps — a 33% overhead that
no amount of skill removes. Percentages are unkind to small denominators; the
delivery rate for short goals is 60/60.

**Raising the budget from 256 to 512 is what fixed the long routes.** The same
training run at 256 steps delivered only 36/60 in the 150–256 band; at 512 it is
49/60. Those failures were never a policy problem — the routes needed more steps
than existed. §5 has the arithmetic.

The worst case on this map is the exact far corner, and the doc should say so:

```
goal (127, 127)   optimal 254 steps, budget 512
greedy eval: 2/20 delivered, 0 destroyed, 18 timed out,
             mean 510 steps vs 254 optimal (2.01x)
```

2/20 — far below the 49/60 its distance band manages. A corner is the hardest
goal on the board: walls on two sides, so every slip near it is a wasted step
with no way to overshoot and come back cheaply, and 254 is the longest route the
map contains. **Zero hazard deaths in all 20 runs**, though — it had learned the
hazards perfectly and simply ran out of clock. `--steps 700` delivers it.

## 5. Why the budget is 512 and not 256

You suggested 256 steps and left the door open ("or more, im not sure"). It has
to be more, and here is the arithmetic that decides it:

* the far corner (127,127) is **254 cells** from S by the shortest wall-avoiding
  route;
* slipping inflates any route by **≈1.25–1.3×** — 20% of moves go sideways;
* so that corner needs **≈330 steps in expectation**.

**At 256, no policy can reliably reach the far third of the map.** Not a
training failure — arithmetic. And since the whole point of this file is "click
anywhere and the robot goes there", a budget that makes a third of the map
unreachable by construction quietly breaks the promise. So **the default is
512**, which clears 330 with room for a bad run.

Raising the horizon is not free, and it changed one other number. The step
penalty obeys `|R_step| < R_goal / horizon` (§1) — below that bound, walking the
long way round a hazard still pays. Doubling the horizon halves the bound, from
`10/256 = 0.039` to `10/512 = 0.0195`, so `R_STEP` had to come down from −0.02
to **−0.01** to keep the same ~2× margin. `tests/test_grid_delivery_robot.py`
asserts that relationship, so the two constants cannot drift apart unnoticed.

The GUI still tells you where you stand before you blame the robot — every goal
you click is labelled:

* green — `optimal 120 steps, budget 512` — comfortable,
* orange — `TIGHT — optimal 254, slipping needs ~330, budget 256` — which at
  the default 512 you will not see on this map at all, since the longest route
  it contains is 254; drop to `--steps 256` and the far third turns orange,
* red — `UNREACHABLE — that cell is a wall`.

Practical guidance:

| you want | use |
|---|---|
| every cell on the map reachable | `--steps 512` (the default) |
| the exercise exactly as literally stated | `--steps 256`, and expect the far corner to time out |
| faster training, near goals only | `--steps 200` |

`--steps` changes the observation (the clock is normalised by it) and the
truncation bootstrap, so it is a genuine part of the MDP, not a display setting.
Change it and you are training a different agent.

---

## 6. Three bugs, kept on the record

Every one of these produced a *plausible-looking* training curve.

1. **The user's goal was being overwritten by training.** `env.goal` is scratch
   space — every training episode calls `env.reset(goal=<curriculum goal>)` and
   clobbers it. The GUI read `env.goal` to draw the star and to roll the replay,
   so the star tracked whatever random goal the trainer had just used, and the
   "replay to your goal" walked somewhere else entirely. Fix: the user's goal
   lives in the GUI's own state and is passed explicitly. **Shared mutable state
   between a trainer and a viewer will find you.**
2. **A wall cell accepted as a goal.** `bfs_distance` seeded its search *at* the
   goal without checking it was floor, so asking for a route to a cell inside a
   wall block returned a confident "82 steps". The policy then "failed" that
   goal 20/20 times and I spent a while debugging a policy that was behaving
   perfectly. Fix: BFS returns unreachable-everywhere for a walled goal, and
   `feasibility()` says so in words.
3. **A clean exit that crashed.** The training thread is a daemon; killing it
   mid-`torch` forward pass aborted the interpreter with `terminate called
   without an active exception` on *every* window close. Fix: signal the thread,
   `join(timeout=3)`, then destroy the window.

---

## 7. Answer sheet — Exercise 2: UR5e pick-and-place

No code for this one — it is a design exercise, and the point is that every
choice below is the same *kind* of choice as §1, at a scale where getting it
wrong costs a real robot.

### True environment state

```
s_t = ( q, q̇, τ                      robot: 6 joint angles, velocities, torques
      , T_tcp, w_gripper, ẇ           tool pose (SE(3)), gripper width and rate
      , p_cube, R_cube, v_cube, ω_cube        exact cube pose and twist
      , {p_i, R_i, v_i}_{i=1,2}       the two distractors, likewise
      , p_bin, R_bin, bin_extent      exact bin pose and geometry
      , C = {(body_a, body_b, f_n, μ)}        every contact and its normal force
      , grasp_flag, table_plane, joint_limits, μ_surfaces, m_cube
      , t )
```

This is *privileged*: it is everything the simulator knows. It is used for the
reward, for success and failure checks, for a critic if you train asymmetric
actor-critic, and for domain randomisation. **It is not what the policy sees.**

### Policy observation

```
o_t = ( I_scene^{RGB-D}    128×128×4, fixed camera
      , I_wrist^{RGB-D}    128×128×4, wrist camera
      , q, q̇               12
      , T_tcp              9  (position + 6D rotation, not Euler, not quaternion)
      , w_gripper, ẇ       2
      , contact_bit        1
      , a_{t-1}            7
      , stacked over the last 3 frames at 10 Hz )
```

**Information deliberately hidden:** exact cube pose and velocity, exact bin
pose, object identities and which of the three is the blue one, contact forces,
masses and frictions, the table plane. All of it is *available* in the simulator
and none of it exists on the real robot — training on it produces a policy that
cannot be deployed. The policy must read the cube's pose out of pixels, which is
the entire point.

**Fully or partially observable?** **Partially**, unavoidably and in three
distinct ways:

1. **Occlusion** — the gripper hides the cube exactly when the grasp matters,
   which is what the wrist camera is for;
2. **Velocity is not in a single frame** — one image cannot tell a cube being
   lifted from a cube being dropped. Hence the 3-frame stack;
3. **Grasp quality is barely observable at all** — a cube held by one corner and
   a cube held squarely look nearly identical. The contact bit plus gripper
   width plus their recent history is the best proxy available, and it is why
   they are in there.

**Is one image sufficient?** No. Three stacked frames (300 ms of history) is the
cheap fix and covers velocity and the "am I still holding it" question. A
recurrent core or a short transformer over frames is the better fix if the task
grows a memory requirement — e.g. "the bin was visible before the arm occluded
it". Note the *same* argument as §2.3, one scale up: a memoryless policy
oscillates, and here it would open and close the gripper.

### Action vector

```
a_t = [ Δx, Δy, Δz, Δr_x, Δr_y, Δr_z, g ]      dimension 7
```

| component | meaning | range | unit |
|---|---|---|---|
| `Δx, Δy, Δz` | TCP translation for this control step, in the base frame | `[−1, 1]` → ±25 mm | metres |
| `Δr_x, Δr_y, Δr_z` | TCP rotation, axis-angle, in the tool frame | `[−1, 1]` → ±0.15 rad | radians |
| `g` | target gripper width | `[−1, 1]` → `[0, 85]` mm | metres |

**Relative for the pose, absolute for the gripper.** The reasoning:

* **Relative TCP** (not joint positions, not joint velocities). The task is
  defined in the space the cube lives in, so a Cartesian action lets the network
  express "move 2 cm toward the cube" as one number instead of a six-way
  coordinated joint change it has to learn from scratch. It also makes the
  policy nearly invariant to where in the workspace the cube spawned, which is
  the randomisation you are training against. **Joint velocities** would be the
  worst of the options: the mapping from a velocity command to "did I get closer
  to the cube" changes with the arm's configuration, so the policy has to learn
  the Jacobian implicitly.
* **Absolute gripper.** Grasping is a *state*, not an increment. `g = −1` should
  mean closed regardless of history; a relative gripper action lets integration
  error accumulate into a half-open hand.
* **±25 mm per step at 10 Hz = 0.25 m/s** — fast enough to cross the table in
  ~2 s of a 20 s episode, slow enough that a single bad action cannot slam the
  table.

**What sends them to the robot:** the policy output is scaled from `[−1,1]` to
the ranges above, added to the *current commanded* TCP pose (not the measured
one — feeding back the measured pose closes a loop the policy did not ask for
and oscillates), converted to joint targets by a damped-least-squares IK step,
and handed to the UR5e's joint position controller at 500 Hz with the 10 Hz
policy output held between updates.

**Safety filtering, applied in this order** — the policy never talks to the arm
directly:

1. clamp the action to its box (the network can emit anything);
2. reject the IK solution if any joint would exceed 95% of its limit, or if the
   solution jumps more than 0.3 rad in one step (branch flip);
3. clamp the target TCP into a workspace box, with `z ≥ table_z + 5 mm` except
   during a commanded grasp descent;
4. run a self-collision and table-collision check on the candidate configuration
   and reject it if it fails;
5. a force-torque watchdog: any contact above 25 N halts and terminates the
   episode.

Rejected actions hold the previous pose. **The robot must be safe under a
*random* policy**, because for the first several thousand steps that is exactly
what it has.

### Reward function

$$r_t = r_{\text{success}} + r_{\text{approach}} + r_{\text{grasp}} + r_{\text{transport}} + r_{\text{safety}} + r_{\text{time}} + r_{\text{smooth}}$$

with `d_go = ‖p_gripper − p_cube‖` and `d_ob = ‖p_cube − p_bin‖`:

| term | formula | why |
|---|---|---|
| `r_success` | `+20` once, on the verified success condition below | the only term that defines the task |
| `r_approach` | `+2.0 · (d_go^{t−1} − d_go^{t})`, **only while not holding** | potential-based; switched off after grasp so it stops paying for hovering |
| `r_grasp` | `+5` **once per episode**, latched, on first verified stable grasp | the sub-goal that unlocks everything |
| `r_transport` | `+3.0 · (d_ob^{t−1} − d_ob^{t})`, **only while holding** | potential-based; gated on the grasp so pushing earns nothing |
| `r_safety` | `−5` and terminate on collision >25 N; `−2` per step at 10–25 N; `−10` and terminate if the cube leaves the table | keeps the arm alive |
| `r_time` | `−0.01` per step | prefers the short trajectory; 200 steps × 0.01 = 2, small against +20 |
| `r_smooth` | `−0.05 ‖a_t − a_{t−1}‖²` | punishes jerk; without it the policy learns a buzzing chatter that is fine in sim and destroys a real gear |
| `r_drop` | `−5` on losing a verified grasp above the table | otherwise dropping the cube mid-transport is free |

Every shaping term is a **difference of distances**, i.e. potential-based, for
the same reason as §1: it cannot be farmed, and it provably does not move the
optimal policy.

### Success condition

Not "cube near bin". Success requires **all** of, held continuously for **1.0 s
(10 consecutive steps)**:

* `p_cube` horizontally inside the bin footprint, with 1 cm margin;
* `p_cube,z` **below the bin rim** and within 3 cm of the bin floor — this is the
  clause that rejects a cube balanced on the rim or held above the bin;
* `‖v_cube‖ < 0.02 m/s` and `‖ω_cube‖ < 0.2 rad/s` — it is at rest, not in
  flight through the bin;
* **the gripper has released** (`w > 60 mm`) and `d_go > 5 cm` — the arm has let
  go and backed off;
* no contact between the cube and the gripper for the full second.

### Failure conditions

* contact force > 25 N against the table, bin, or a distractor;
* any self-collision;
* any joint outside its limit, or the IK filter rejecting >10 consecutive steps
  (the policy is wedged against the workspace boundary);
* the cube leaves the table surface (`z < table_z − 5 cm`);
* a distractor is knocked off the table — *the task is to pick the blue cube,
  not to clear the table*;
* **truncation:** `t ≥ 200 steps` (20 s at 10 Hz). Truncation, not termination —
  bootstrap `V(s_200)`, exactly as in §1.

### Two ways this reward can be exploited

Inspecting the terms above, not the generic list:

1. **The grasp bonus, farmed by re-grasping.** `r_grasp = +5` on a *verified
   stable grasp*. If that is paid every time the condition becomes true, the
   optimal policy is to close the gripper, open it, close it — +5 every few
   steps, forever, worth far more than the +20 for finishing. The mitigation is
   in the spec above and is easy to forget: **latch it, once per episode.**
   Even latched, there is a residue — a policy can grasp, then dither near the
   cube collecting nothing but losing only `r_time`, which is why `r_time` must
   be strictly negative.
2. **The transport term, farmed by dragging.** `r_transport` pays for reducing
   `d_ob`. If it is *not* gated on holding the cube, the cheapest way to earn it
   is to **push** the cube across the table with the closed gripper — no grasp
   needed, no lift, and it collects the full potential all the way to the bin.
   It then stalls at the bin wall, because pushing cannot get the cube *in*, and
   sits there having banked most of the shaping. The gate on a verified grasp is
   what kills it. (A subtler variant survives even the gate: grasp, lift 1 mm,
   and *drag* the cube along the table. `r_drop` and the lift clause in the
   success condition are what make that unprofitable.)
3. *(a third, because it is the one that actually bites in practice)* **Approach
   shaping, farmed by hovering.** `r_approach` pays for reducing `d_go`. A
   policy that parks the gripper 1 mm from the cube and vibrates collects
   nothing further — the potential is exhausted — but it also risks nothing.
   Against a *positive* step reward it would be optimal. Against `r_time = −0.01`
   it slowly bleeds, and the +5 grasp latch is the only thing that pays for
   committing. **Every hovering exploit is a sign that the step penalty is too
   close to zero.**

### Why this state, action and reward

The observation is what a real UR5e cell actually has — two cameras, joint
encoders, a gripper width, a contact bit — and nothing it does not, so the
policy is deployable rather than merely trainable. The action is Cartesian and
relative because the task is defined in the cube's space, not the arm's, and
because it makes the policy invariant to the randomisation you are training
against. The reward is one sparse term that *defines* the task plus a set of
strictly potential-based shaping terms that only make it *findable* — gated so
that each one can only be earned in the phase where it means something, and
sized so that no shaping term can ever outbid the +20 for finishing the job.

That last clause is the whole discipline, and it is the same one as §1: **write
down the total shaping the agent can bank on the way to a shortcut, and make
sure the penalty is bigger.**
