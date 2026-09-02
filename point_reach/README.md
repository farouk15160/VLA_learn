# 2D PointReach — Behavioral Cloning and DAgger

Two self-contained files. No RL framework, no gym wrapper: the environment, the
expert, the dataset, the normalizer, the network and the training loop are all
written out so nothing about the method is hidden.

| File | What it is |
|---|---|
| `bc_pointreach.py` | The whole behavioral-cloning pipeline in one file, CLI only. Also the shared library that the second file imports. |
| `dagger_pointreach.py` | DAgger on top of it, with a Tk GUI: live training, live stats, pause/resume/stop, interactive click-to-test, and five live plots. |

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python point_reach/bc_pointreach.py                        # plain task
.venv/bin/python point_reach/bc_pointreach.py --obstacles --demo-band # where BC fails
.venv/bin/python point_reach/dagger_pointreach.py                    # the GUI
```

---

## 1. The environment

### State

The world state is the agent's position and the goal, both in a square workspace
$\mathcal{W} = [-1, 1]^2$:

$$p_t = (x_t, y_t) \in \mathcal{W}, \qquad p_g = (x_g, y_g) \in \mathcal{W}$$

The task is fully observed — the policy sees the state exactly, with no history:

$$o_t = [\,x_t,\; y_t,\; x_g,\; y_g\,] \in \mathbb{R}^4$$

The goal is *inside* the observation, which is what makes one network able to
serve every goal instead of one network per goal.

### Action

A commanded displacement, bounded per axis by $a_{\max} = 0.1$:

$$a_t = [\Delta x_t,\; \Delta y_t], \qquad a_t \in [-a_{\max}, a_{\max}]^2$$

Bounding matters: it is what makes the task take many steps, and therefore what
lets errors accumulate over a trajectory.

### Dynamics

$$p_{t+1} = \operatorname{clip}_\mathcal{W}\big(p_t + a_t + \epsilon_t\big),
\qquad \epsilon_t \sim \mathcal{N}(0,\, \sigma^2 I),\ \ \sigma = 0.02$$

The noise $\epsilon_t$ is why open-loop planning is not enough: every step the
agent is pushed somewhere it did not ask to be, and it must react to that.

In the **obstacle variant** (see §6) the workspace also contains three fixed
discs $\{(c_i, r_i)\}$. A step whose endpoint lands inside a disc is rejected,
the agent stays where it was, and the timestep is spent anyway. Obstacle
positions are *not* in the observation — the policy has to learn the map.

### Episode termination

$$\text{success} \iff \lVert p_g - p_t \rVert < r_\text{goal} = 0.05$$

An episode ends on success or after $T_{\max} = 80$ steps, whichever first.

### Metrics

- **success rate** — fraction of episodes that reach the goal region.
- **final distance** — $\lVert p_g - p_T\rVert$ at the end of the episode, averaged.
  Sensitive in a way the binary success rate is not: a policy that ends 0.06 away
  scores 0 successes but is nearly there, and this number says so.

Code: `PointReachConfig`, `PointReachEnv` in `bc_pointreach.py`.

---

## 2. The scripted expert

A proportional controller with gain $K = 0.6$, saturated at the action limit:

$$a_t^\* = \operatorname{clip}\big(K\,(p_g - p_t),\; -a_{\max},\; +a_{\max}\big)$$

It is a *feedback* law: it is defined at **every** state, not just the ones it
visits. That property is what makes DAgger possible at all — the expert can be
queried at states the expert itself would never have produced.

Far from the goal the clip is active, so the expert moves at full speed; inside
$\lVert p_g - p_t \rVert < a_{\max}/K \approx 0.167$ it decelerates
proportionally and converges geometrically:

$$\lVert p_g - p_{t+1}\rVert \approx (1 - K)\,\lVert p_g - p_t\rVert$$

**With obstacles**, the same clipped-proportional law is used, but it steers at a
waypoint instead of the goal: if the straight segment $p \to p_g$ clips a disc
inflated by a safety margin, the expert aims at the *tangent point* around the
disc, on the side away from the disc centre.

$$a_t^\* = \operatorname{clip}\big(K\,(w(p_t, p_g) - p_t),\ -a_{\max}, +a_{\max}\big)$$

The left/right tangent choice is a **discrete switch**, which is precisely what
makes this expert's state→action map nonlinear — and therefore hard to clone
(§6). Code: `expert_waypoint`, `expert_action`.

---

## 3. The data

### Collection

Each episode is run to termination and stored **whole** — observations, labels,
the full position path, the goal, and the outcome (`Episode`). Nothing is
flattened until after the split.

$$\tau^{(k)} = \big( (o_0, a_0^\*), (o_1, a_1^\*), \dots, (o_{T_k-1}, a^\*_{T_k-1}) \big)$$

500–1000 episodes at roughly 15–25 steps each gives on the order of 10–20k
transitions.

### Splitting — by episode, never by transition

70% train / 15% validation / 15% test, **at the episode level**:

```python
train_eps, val_eps, test_eps = split_by_episode(episodes, fractions=(0.70, 0.15, 0.15))
```

Consecutive states inside one trajectory are near-duplicates ($p_{t+1}$ differs
from $p_t$ by at most $a_{\max}$). Shuffling transitions would put nearly the
same sample in train and test, and the test MSE would measure memorization
instead of generalization. Splitting whole trajectories keeps the test set
genuinely unseen.

### Normalization — training statistics only

$$\tilde o = \frac{o - \mu_\text{train}^{o}}{\sigma_\text{train}^{o} + \varepsilon},
\qquad
\tilde a = \frac{a - \mu_\text{train}^{a}}{\sigma_\text{train}^{a} + \varepsilon}$$

Fitted on the training split alone (`Normalizer`). Using validation or test
statistics would leak information about held-out episodes into the inputs. The
network is trained entirely in normalized space and its output is mapped back
with $a = \tilde a\,\sigma^a + \mu^a$, then clipped to $[-a_{\max}, a_{\max}]$
before it ever touches the environment.

Actions here have small magnitudes ($\lesssim 0.1$); without rescaling the
targets, MSE values are ~$10^{-4}$ and gradients are correspondingly tiny.

---

## 4. The policy and its training

A deliberately small MLP:

$$\pi_\theta:\ \mathbb{R}^4 \rightarrow \mathbb{R}^2, \qquad
4 \rightarrow 128 \xrightarrow{\text{ReLU}} 128 \xrightarrow{\text{ReLU}} 2$$

Behavioral cloning is plain supervised regression on the demonstration pairs:

$$\mathcal{L}(\theta) = \frac{1}{|D|} \sum_{(o, a^\*) \in D}
\big\lVert \pi_\theta(\tilde o) - \tilde a^\* \big\rVert_2^2$$

MSE is the maximum-likelihood objective for a Gaussian around the expert's
action, which is the standard BC choice for continuous control. Adam, learning
rate $10^{-3}$, batch size 256, ~150 epochs; validation loss is computed every
epoch on the held-out episodes. `train_bc` takes `on_epoch` and `should_stop`
callbacks, which is how the GUI streams the loss curve and pauses cleanly
between epochs.

**Deployment.** At each step: normalize $o_t$, forward, denormalize, clip, apply.
The policy is a *reactive controller* — it re-reads the true state every step, so
it corrects for noise the same way the expert does.

---

## 5. Evaluating: three numbers that do not agree

| Metric | What it measures | Distribution it is measured on |
|---|---|---|
| **Test action MSE** | how well single expert actions are predicted | the **expert's** states $d_{\pi^\*}$ |
| **Final position error** | how close a rollout ends up | the **policy's own** states $d_{\pi_\theta}$ |
| **Closed-loop success rate** | did the rollout finish the task | the **policy's own** states $d_{\pi_\theta}$ |

The first is open-loop and cheap; the last two require actually driving the
environment. They disagree because supervised learning assumes i.i.d. samples,
and a rollout violates that assumption: the policy's own errors decide which
states it sees next.

The standard bound makes the shape of the problem explicit. If the policy makes a
mistake with probability at most $\varepsilon$ under the expert's own state
distribution, over a horizon $T$ its regret grows as

$$J(\pi_\theta) - J(\pi^\*) = \mathcal{O}(\varepsilon T^2)$$

— quadratic in the horizon, because one mistake moves the agent to a state the
training data never covered, where the next mistake is *more* likely. DAgger's
guarantee is $\mathcal{O}(\varepsilon T)$ instead.

---

## 6. Result 1 — where BC does and does not break

### Open field (`bc_pointreach.py`, no flags)

```
Test-set action MSE (raw units)   1.75e-06   -> RMS 0.00132 vs a_max 0.1
                                      expert      cloned
Closed-loop success rate              100.0%      100.0%
Mean final distance                   0.0319      0.0311
```

**BC matches the expert exactly, and there is no drift to see.** This is worth
understanding rather than hiding: in the open field the expert is
$\operatorname{clip}(K(p_g - p))$, a *piecewise-linear* function of the
observation. A ReLU network represents that class exactly and extrapolates it
approximately linearly, so even states far outside the demonstrations get
sensible actions. Compounding error needs something to compound.

### Obstacle course with partial coverage (`--obstacles --demo-band`)

Two changes, both of which mirror how imitation learning actually fails in
practice:

1. **A nonlinear expert.** Three fixed discs; the expert's tangent-point
   switching makes its map nonlinear and discontinuous at the decision boundary.
   Near that boundary a small regression error selects the *wrong side* of an
   obstacle.
2. **Partial demonstration coverage** (`--demo-band`). Demonstrations start only
   in the left corridor $x_0 \in [-1.0, -0.6]$; deployment starts anywhere. The
   demonstration state distribution is a thin set of tangent arcs, and the
   deployment distribution is the whole arena.

```
Test-set action MSE (raw units)   5.41e-04   -> RMS 0.02325 vs a_max 0.1
                                      expert      cloned
Closed-loop success rate              100.0%       76.0%
Mean final distance                   0.0315      0.2592
Mean steps per episode                  19.4        35.8
```

**This is the headline result.** Per-step prediction is good — RMS error is a
quarter of one action limit — yet the rollout loses 24 points of success rate and
its mean final distance is eight times worse. The failure mode is visible in the
GUI: the policy grazes an obstacle, ends up pressed against a disc it never saw
in training, and dithers there until the horizon runs out.

Both variants are in the same file on purpose: the contrast *is* the lesson.

---

## 7. DAgger

Behavioral cloning learns the expert's actions. DAgger learns the expert's
**corrections**. One iteration:

$$
\begin{aligned}
&\textbf{1. train } && \pi_i = \arg\min_\pi \textstyle\sum_{(o,a) \in D} \lVert \pi(o) - a \rVert^2 \\
&\textbf{2. roll out } && \text{run } \pi_i \text{ in the environment, collect the states } o \text{ it visits} \\
&\textbf{3. label } && \text{ask the expert } a^\* = \pi^\*(o) \text{ at each of those states} \\
&\textbf{4. aggregate } && D \leftarrow D \cup \{(o, a^\*)\} \\
\end{aligned}
$$

Two details are the whole method:

- The action the learner actually **executed is thrown away**. Only the state it
  reached is kept, paired with what the *expert* would have done there. The
  learner chooses the questions; the expert gives the answers.
- The dataset is **aggregated, never replaced**. $\pi_i$ is trained on the union
  of all iterations, so it must satisfy the expert everywhere seen so far. This
  is a no-regret online-learning procedure (Follow-the-Leader on the aggregated
  loss), which is where the $\mathcal{O}(\varepsilon T)$ guarantee comes from.

Optionally the *behaviour* policy mixes in the expert,
$\pi^{\text{behave}}_i = \beta_i \pi^\* + (1-\beta_i)\pi_i$ with
$\beta_i = \beta^i$ — the GUI's "Expert mix beta" spinbox. Default $\beta = 0$:
pure DAgger, the learner drives and the expert only comments.

Iteration 0 is exactly plain behavioral cloning, so every chart has its own
baseline built in.

Code: `DaggerEngine.train_once / evaluate / aggregate` in `dagger_pointreach.py`.

---

## 8. Result 2 — what DAgger buys

Default configuration (500 demos, 8 iterations, 60 epochs and 60 relabelled
rollouts per iteration, 100 evaluation episodes, obstacle course, banded demos):

|  | BC (iteration 0) | DAgger (final) | expert |
|---|---|---|---|
| success rate | 87.0% | **98.0%** | 100.0% |
| mean final distance | 0.141 | **0.048** | 0.032 |
| held-out expert action MSE | 9.44e-04 | 5.11e-04 | — |
| dataset size | 7,247 | 20,197 | — |

(One run. The BC baseline moves by a few points between runs — 76–87% across the
runs recorded here — because floating-point non-determinism in training is
amplified by the rollout; the DAgger end point is consistently 96–99%.)

The supervised metric moves by a factor of two. The *closed-loop* metric goes
from failing roughly one episode in eight to failing one in fifty. That gap between the two
columns is the entire point: DAgger does not make the policy a better imitator of
expert trajectories, it makes it a policy that knows how to recover from its own
imperfect states.

The state-distribution plot shows the mechanism directly: the BC dataset is a set
of clean tangent arcs starting in the left corridor, while the aggregated DAgger
dataset is smeared over the whole arena and piles up exactly where the learner
gets into trouble — pressed against obstacle boundaries and workspace walls.
Those are the states BC never had a label for.

---

## 9. The GUI

```bash
.venv/bin/python point_reach/dagger_pointreach.py
```

**Controls (left panel).** Expert demos, DAgger iterations, epochs/iteration,
rollouts/iteration, eval episodes, hidden units, seed, execution noise, start
perturbation, expert-mix beta, plus toggles for the obstacle course and the
banded demonstrations. They are read when a fresh run starts.

- **Start** — collects demonstrations (first time) and runs the loop in a worker
  thread; the UI stays responsive.
- **Pause / Resume** — blocks the worker between epochs. Nothing is lost.
- **Stop** — ends the loop after the current epoch but **keeps the model, the
  optimizer state and the aggregated dataset**. Press Start again to resume from
  the iteration where you stopped; raise the iteration count first to go further.
- **Reset** — discards everything.
- **Save checkpoint / Load** — the model, normalizer, frozen BC policy, dataset
  and history in one `.pt` file, so a run survives closing the window.

**Live stats.** Phase, iteration, epoch, train and validation loss, dataset size,
last evaluation success and distance, held-out MSE, the frozen BC baseline, the
expert's own score, and elapsed time. Below them, a scrolling log of every phase
transition and per-iteration result.

**Tabs.**

1. **Live** — the latest evaluation rollouts in the arena (green = reached, red =
   failed) beside the loss curve, updated during training; vertical lines mark
   iteration boundaries.
2. **Progress** — the four requested curves: success rate vs iteration (with the
   BC baseline and the expert as reference lines), final distance vs iteration,
   collected samples per iteration, and the held-out action MSE, which barely
   moves while the success rate climbs.
3. **State distribution** — BC demonstration states versus aggregated DAgger
   states, with episode starts marked, on the same arena.
4. **Trajectories** — expert, BC and DAgger run on the *same* start/goal pairs.
5. **Interactive test** — left-click sets the goal, right-click sets the start,
   then Run animates the expert, the frozen BC policy and the current DAgger
   policy side by side, with live distance and step count. The model is snapshot
   under a lock, so you can test while training continues.

Plots update continuously in the GUI and are not written as image files.

---

## 10. Things worth trying

| Change | What you should see |
|---|---|
| `--no-obstacles` | BC already matches the expert; DAgger's curves are flat. The failure needs a nonlinear expert. |
| `--no-demo-band` | Full demonstration coverage; the BC gap shrinks a lot. Coverage, not quantity, is what BC lacks. |
| Raise execution noise | Counter-intuitively BC improves — noise makes the demonstrations dither around the expert's path, which is a crude form of the corrective data DAgger collects deliberately. |
| Beta > 0 | Early iterations stay near the expert's distribution; smoother early curves, slower to see the learner's real failures. |
| Fewer demos, more DAgger rollouts | Usually beats the reverse at equal total samples. |
