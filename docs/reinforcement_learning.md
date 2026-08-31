# Reinforcement Learning — a car that learns to drive

Companion to `reinforcement_learning.py`. RL was new to you, so this starts from
nothing and assumes only that you are comfortable with supervised learning.

---

## 1. The one sentence that explains why RL is hard

> In supervised learning a teacher tells you the right answer.
> In reinforcement learning a critic tells you your score, and you have to work
> out for yourself which of the 300 things you did earned it.

Everything difficult about RL follows from that.

| | `supervised_learning.py` | `reinforcement_learning.py` |
|---|---|---|
| Where data comes from | a file on disk, fixed | **the car generates it by driving** |
| Supervision signal | the correct label, per sample | a scalar reward, often much later |
| Loss | distance from the right answer | there is none — we build a *surrogate* |
| Data distribution | fixed (i.i.d. assumption) | **changes every time you update** |
| If the model is bad | you get a bad score | you collect bad data, and get worse |
| Gradient of the objective | exact, via backprop | estimated from noisy samples |

That last row is the crux. You cannot backpropagate through a physics simulator,
a real robot, or a human rating a chatbot reply. The environment is a black box.
So RL needs a gradient **without** differentiating the thing that produced the
reward. §6 shows the trick.

---

## 2. This file's MDP, written out exactly

§3 gives the general notation. This section is the specific instance you are
running, so you can match each symbol to a line of code.

### 2.1 State space `S` — what the car perceives

**10 floats**, all scaled to roughly `[-1, 1]`:

| index | quantity | how it is computed |
|---|---|---|
| `0..6` | seven lidar distances | `raycast(...) / RAY_MAX` — walls ahead, over a 135° fan |
| `7` | `cos(bearing − heading)` | direction of B *relative to where the car points* |
| `8` | `sin(bearing − heading)` | the other half of that angle |
| `9` | range to B | `‖GOAL − pos‖ / diagonal` |

Three deliberate choices worth understanding:

- **The absolute position `(x, y)` is NOT in the state.** The policy gets only
  what a real robot's sensors would give it, so it must learn to drive by what
  it sees rather than by memorising coordinates. Add `(x, y)` and it solves the
  task faster — and learns nothing transferable.
- **The bearing is split into `cos`/`sin`** rather than passed as a raw angle.
  A raw angle jumps from `+π` to `−π` at the same physical heading, and a
  network cannot learn across that discontinuity. `cos`/`sin` is smooth
  everywhere. The same trick is used for joint angles in real robot policies.
- **The bearing is *relative* to the car's heading**, not absolute. "The goal is
  30° to my left" is directly actionable; "the goal is at compass bearing 95°"
  requires the car to also know its own heading and do the subtraction itself.

**Is it Markov?** Nearly. Speed is constant and the map is static, so the
current reading does determine the best action. A real car would need velocity
and slip, and a camera-only robot would not have lidar at all — that is a POMDP,
and the usual fix is a frame stack or a recurrent memory.

### 2.2 Action space `A` — what the car can do

**3 discrete actions**, chosen every step:

| action | effect |
|---|---|
| `0` | `heading -= STEER` (0.30 rad) |
| `1` | go straight |
| `2` | `heading += STEER` |

Speed is **constant** at `SPEED = 1.6`. The car cannot brake, reverse, or
accelerate — it can only turn. That keeps the action space small enough for a
plain categorical policy, so the interesting part stays the *learning* rather
than continuous control.

### 2.3 Transition `P` — the physics

```python
self.th  += (a - 1) * STEER
self.pos += SPEED * [cos(self.th), sin(self.th)]
```

Deterministic, and **the agent never sees this function**. That is the whole
reason this is RL: you cannot differentiate the reward with respect to the
policy weights through code the learner has no access to.

The only stochasticity is the start state: position is jittered ±4 units and
heading ±0.5 rad on every reset. That is the cheapest form of **domain
randomization** — without it the policy can memorise one trajectory instead of
learning a rule, and it is the same idea used when training robot policies in
simulation for transfer to hardware.

### 2.4 Reward `R` — the part you will get wrong first

```python
r  =  0.10 * (prev_distance_to_goal - distance_to_goal)   # progress
r -=  0.002                                               # time cost
if crashed:  r -= 3.0 ; end episode
if arrived:  r += 5.0 ; end episode
```

**Why shaped and not just `+1` for reaching B?** Because a sparse reward is
unlearnable here. Random steering essentially never reaches B, so almost every
episode would return exactly 0, the gradient would be 0, and nothing would ever
be learned (§4.5). The progress term gives feedback on *every single step*.

**Why is it safe to shape?** Because it is a **potential-based** shaping — a
difference of distances. Driving away costs exactly what coming back pays, so
the car cannot farm reward by circling. Shaping that is *not* a difference (say,
`+0.1` for merely being near the goal) is exactly how you get an agent that
learns to hover next to the goal forever without entering it.

**Why is the crash penalty 3.0?** Because I first used 1.0 and it was wrong. A
car that crashed 20 units closer to B banked `0.1 × 20 = 2.0` of progress and
lost only 1.0 — **net +0.9**. "Drive straight into the wall" was a *profitable*
policy. This is the single most instructive bug in the file: reward-design
errors look exactly like agent stupidity, and they are your fault, not the
algorithm's.

For calibration: a clean run from A to B earns about `0.1 × 80 = 8.0` of
progress plus the `+5.0` arrival bonus, so **~12.5 total** — which is the number
you see the return curve converge to.

### 2.5 Discount `γ` and episode end

`γ = 0.99`. Episodes end on **crash**, on **arrival**, or on **truncation** at
`MAX_STEPS = 400`. Crash and arrival are true terminal states; truncation is an
artificial cutoff, and conflating the two biases value learning, because a
truncated state is *not* worth zero future reward.

### 2.6 The map

World is 100 × 60. Start `A = (10, 30)`, goal `B = (90, 30)`, arrival radius 5.
Two staggered walls — one rising from the floor to y=40 at x∈[30,38], one
hanging from the ceiling down to y=20 at x∈[60,68] — mean **the straight line
from A to B is blocked**. The car must go over the first and under the second.
That is deliberate: a policy that simply drives at the goal cannot succeed, so
success is real evidence of learning.

---

## 3. The same thing, in the general notation

Every RL paper writes the problem as the 5-tuple $(S, A, P, R, \gamma)$ you just
saw filled in. The remaining pieces of vocabulary:

**The policy** $\pi_\theta(a \mid s)$ is what we are learning: a network mapping
state to a *distribution* over actions. It is stochastic, and that is not a
defect — sampling **is** the exploration mechanism. A car that always takes its
current best guess can never discover that its guess is wrong.

**The return** $G_t$ is what we maximize:

$$G_t = r_{t} + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

$\gamma$ keeps an infinite-horizon sum finite and encodes "sooner is better". It
is also a variance knob: small $\gamma$ is short-sighted but much less noisy.

**The objective** is the expected return of trajectories drawn from the policy:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ G_0 \right]$$

Note the subscript: the expectation is over trajectories generated by *the very
policy we are optimizing*. That self-reference is the moving-target problem, and
it has no counterpart in supervised learning.

**Model-free** means we never try to learn $P$. Everything here is model-free.

---

## 4. The problems in RL (this is the section you asked for)

These are the standard, named difficulties. I've marked which ones you can
actually watch happen in this script.

### 4.1 Temporal credit assignment ← *visible in the script*
You steered 80 times and then hit a wall. Which of those 80 turns was the
mistake? The reward gives you one number for the whole episode. This is
**the** defining problem of RL. REINFORCE's answer is blunt: credit *every*
action in an episode with the return that followed it. It works, but it is
extremely noisy — good actions inside a bad episode get punished, and vice
versa. Better credit assignment (TD learning, GAE, advantage estimation) is
most of what separates modern RL algorithms from this one.

### 4.2 Exploration vs exploitation ← *visible in the script*
To find a better policy you must try actions you currently believe are worse.
But every step spent exploring is a step not spent earning reward. An agent that
always plays its best guess can never discover its guess is wrong.
In our code this is why the policy is **stochastic** and why we log **entropy**:
entropy high = still exploring; entropy → 0 = the policy has committed, and if
it committed too early you are stuck in whatever local optimum it found. Watch
the entropy column drop across training.

### 4.3 Sample inefficiency
Our tiny car needs ~500 episodes ≈ 40,000 simulated steps to learn a route a
human would drive first time. Real robots make this brutal: 200,000 steps at
10 Hz is ~5.5 hours of *continuous* hardware time, plus resets, plus wear. This
is the single biggest reason robotics leans on imitation learning (file 01's
paradigm) instead of pure RL. Sample inefficiency has several causes — large
observation/action spaces, poor exploration, badly shaped rewards
([survey](https://arxiv.org/abs/2502.01558)).

### 4.4 High gradient variance ← *ablatable with `--no-baseline`*
The policy gradient is estimated from a handful of noisy episodes, so each
update is a guess. Weighting by the raw return instead of the advantage makes
that guess far noisier — run this file with `--no-baseline` and one seed in
three collapses to a policy that never reaches the goal at all (§8). Since the
variance of an average falls as $1/n$, a noisier estimator simply needs more
episodes to reach the same confidence. This is what §7 fixes.

### 4.5 Sparse rewards
Our reward is *dense* by design (§2.4): progress is paid every single step.
Delete that term and only arrival pays — try it, and the car never learns. Now
imagine "put the mug in the dishwasher", where reward arrives only on success. Random exploration will
essentially never stumble on success, so the gradient is zero almost always and
there is nothing to learn from. Fixes include reward shaping (dangerous — see
3.7), curiosity/intrinsic motivation, curricula, demonstrations to bootstrap
from, and hindsight relabeling. ([intrinsic reward
methods](https://arxiv.org/abs/2601.21391), [curiosity-driven
exploration](https://arxiv.org/pdf/2302.10825))

### 4.6 Non-stationarity / the moving target ← *visible in the script*
Update the policy and the distribution of states you visit changes, so your
value function is now fitted to data from a policy that no longer exists. SL
never faces this: the digits dataset does not rewrite itself when you take an
optimizer step. This is why RL training curves are jagged and why an agent can
be at 500 reward and collapse to 50 an episode later — you will see exactly this
with `--no-baseline`.

### 4.7 Reward hacking / specification gaming
The agent optimizes the reward you *wrote*, not the goal you *meant*. Reward a
cleaning robot for dirt collected and it may learn to dump dirt out and
re-collect it. Writing a reward that cannot be gamed is genuinely hard, and it
is the direct reason RLHF exists for language models: nobody can write down a
scalar "helpfulness" function, so a reward *model* is learned from human
comparisons instead.

### 4.8 Instability and catastrophic collapse ← *visible in the script*
One over-large policy update can destroy a good policy, and unlike SL there is
no fixed validation set to catch it — the environment changes with the policy.
This is precisely why **PPO** exists: it clips the update so the new policy
cannot move too far from the old one in a single step. In our runs, vanilla
REINFORCE reached 500 reward and then collapsed; the baseline version was much
steadier (std 54 vs 129 over the second half of training).

---

## 5. Why you cannot just use backprop

Write out the objective:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int p_\theta(\tau)\, R(\tau)\, d\tau$$

To do gradient ascent we need $\nabla_\theta J$. The obvious move — differentiate
$R(\tau)$ — is impossible: $R$ comes from the simulator, which is not
differentiable and whose internals we do not have. The parameters $\theta$ do
not even appear inside $R$.

But look again: $\theta$ appears in $p_\theta(\tau)$, the *probability* of the
trajectory. That is differentiable, because it is built out of our own network's
output probabilities. So we differentiate the probability instead of the reward.

---

## 6. The policy gradient, derived

The whole derivation is three lines. It rests on one identity, the
**log-derivative trick**, which is just the chain rule on $\log$:

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)$$

Now:

$$\nabla_\theta J = \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau
= \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau
= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \nabla_\theta \log p_\theta(\tau)\, R(\tau) \right]$$

The last step is just the definition of an expectation — which means **we can
estimate it by sampling episodes**. That is the entire idea.

One more simplification. The trajectory probability factorizes:

$$p_\theta(\tau) = p(s_0) \prod_t P(s_{t+1} \mid s_t, a_t)\, \pi_\theta(a_t \mid s_t)$$

Take the log and it becomes a sum; then take $\nabla_\theta$ and **every term
without $\theta$ vanishes** — including the unknown dynamics $P$. This is the
beautiful part: *we never needed to know the physics*. What survives is:

$$\boxed{\;\nabla_\theta J = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t \right]\;}$$

In English: **increase the log-probability of actions that were followed by a
high return; decrease it for actions followed by a low return.**

In code that is exactly one line (`02_reinforcement_learning.py`):

```python
loss = -(logp * adv).mean()   # adv = G_t - V(s_t); minimizing = ascending J
```

The minus sign turns PyTorch's descent into ascent. `coef` is `.detach()`ed
because it is a *weight* on the log-probability, not something to backprop
through.

Two refinements are already in the code:

- **Reward-to-go.** Use $G_t$ (return *from t onward*), not the whole episode
  total. An action at time $t$ cannot have caused a reward at $t-5$, so
  including earlier rewards adds pure noise. Removing them reduces variance and
  introduces **no bias**. Free win.
- **Batching episodes.** We average over 5 episodes per update instead of 1.
  Averaging $n$ samples cuts variance by $n$.

This algorithm — sample episodes, weight log-probs by return, ascend — is
**REINFORCE**, from Williams (1992).

---

## 7. The baseline, and what it is worth here

### The problem

Look at what multiplies $\nabla \log \pi$ if you use the raw return $G_t$. In this
task a successful episode banks about **+12.5**, and even a mediocre one that
crawls halfway to B banks a solidly positive number. So the update pushes *up*
the probability of nearly every action it ever took — including the ones that
steered into a wall — and only makes progress because good actions get pushed up
*harder* than bad ones. All the useful signal lives in small differences between
large positive numbers, which is a numerically terrible place to keep it.

### The fix

Subtract a **baseline** $b(s)$: a state-dependent estimate of "how good is this
situation on average". We learn it with a second network $V(s)$, trained by
plain supervised regression onto the observed returns — look at the code, it is
`mse_loss`, ordinary SL sitting inside an RL loop:

$$A_t = G_t - V(s_t)$$

This is the **advantage**: *was this action better or worse than what I normally
get from here?* Now a below-average action gets a **negative** coefficient and is
actively pushed down. The signal becomes the whole number rather than a
difference between big ones.

### Why it is free (unbiased)

Subtracting something sounds like it should change what we optimize. It does
not, as long as $b$ does not depend on the **action**:

$$\mathbb{E}_{a \sim \pi}\left[ b(s)\, \nabla_\theta \log \pi_\theta(a \mid s) \right]
= b(s) \sum_a \pi_\theta(a\mid s) \frac{\nabla_\theta \pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}
= b(s)\, \nabla_\theta \!\!\sum_a \pi_\theta(a\mid s) = b(s)\, \nabla_\theta 1 = 0$$

The subtracted term has expectation exactly zero. Mean unchanged, variance
reduced.

---

## 8. What actually happened when I ran it

The file has a `--no-baseline` flag so you can run this ablation yourself:

```bash
for s in 0 1 2; do
  .venv/bin/python reinforcement_learning.py --headless --updates 120 --seed $s
  .venv/bin/python reinforcement_learning.py --headless --updates 120 --seed $s --no-baseline
done
```

Measured on this machine, 120 updates (960 episodes) per run:

| seed | | final return | success rate | greedy eval |
|---|---|---|---|---|
| 0 | with baseline | 12.35 | 100% | **30/30** |
| 0 | `--no-baseline` | 12.37 | 100% | 30/30 |
| 1 | with baseline | 12.30 | 100% | **30/30** |
| 1 | `--no-baseline` | **−1.22** | **0%** | **0/30** |
| 2 | with baseline | 12.42 | 100% | **30/30** |
| 2 | `--no-baseline` | 8.17 | 62% | 30/30 |

Read this honestly. **The baseline is not what makes the task solvable** — on
seed 0 the ablated run matched it exactly, 12.37 vs 12.35. What the baseline
buys is **reliability**: 3 of 3 seeds versus 1 of 3 fully working, with one run
collapsing to a policy that never reaches the goal at all.

That is the real lesson of variance reduction, and it generalises: these
techniques usually buy you *consistency and sample efficiency*, not a higher
ceiling. Given how expensive RL samples are on real hardware — where every
episode is wall-clock time on a physical robot — reliability is the thing you
actually want.

It is also why **you must never trust a single-seed RL result**. A single run on
seed 0 would have "shown" that the baseline does nothing.

---

## 9. Where this leads (the actual map)

You now have the seed of every modern policy-gradient method:

```
REINFORCE  (raw returns: --no-baseline)
   │  + learned V(s) as baseline     ← what this file does
   ▼
Actor-Critic  — the actor picks actions, the critic scores states
   │  + bootstrap V from V (TD) instead of waiting for the episode to end
   ▼
A2C / A3C
   │  + GAE: a tunable bias/variance dial for the advantage
   │  + clipped updates so one bad step can't wreck the policy
   ▼
PPO  — the workhorse: robotics, game agents, and RLHF for LLMs
   │  + drop the critic, use group-relative advantages
   ▼
GRPO  — used for reasoning-model post-training
```

Every one of those arrows is a **variance or stability fix**. That is why this
file spends its effort on variance rather than on a fancier environment:
variance *is* the subject.

---

## 10. How this connects to VLA and imitation learning

This matters for where you're heading, so it's worth being precise.

**Today's VLAs are trained with supervised learning, not RL.** RT-2, OpenVLA and
π₀ are trained by **behavior cloning**: collect human teleoperated
demonstrations, then do supervised learning to predict the human's action from
the observation. OpenVLA states it plainly — it discretizes *"each dimension of
the robot actions separately into one of 256 bins"* and is *"trained with a
standard next-token prediction objective, evaluating the cross-entropy loss on
the predicted action tokens only"* ([OpenVLA §
tokenization](https://arxiv.org/html/2406.09246v3)). RT-2 does the same, casting
actions as text tokens ([RT-2](https://arxiv.org/abs/2307.15818)).

**So `01_supervised_learning.py` is closer to a real VLA than this file is.**
Image → class logits → cross-entropy *is* the VLA action head, at 1/1000 scale.

**Then why learn RL at all?** Because behavior cloning has a specific,
structural failure, and RL is the vocabulary for understanding and fixing it:

**Covariate shift / compounding error.** BC trains on states the *expert*
visited. At deployment the policy visits states *it* reaches. Its first small
error moves it slightly off the expert's distribution, where it was never
trained, so its next error is bigger — and errors compound. In the worst case
imitation error scales **quadratically** with the episode horizon
([Three Regimes of Covariate Shift](https://arxiv.org/pdf/2102.02872)).
This is the RL problem 3.6 (non-stationary data distribution) showing up inside
a supervised method — and it is precisely why a BC policy can score 99% on its
held-out validation set and still fail on the robot.

Fixes all involve putting the *learner's own* state distribution into training:
- **DAgger** — roll out the student, ask the expert what it *would* have done in
  the states the student actually reached, aggregate, retrain
  ([Ross, Gordon & Bagnell 2011](https://arxiv.org/abs/1011.0686)).
- **DART** — inject noise during demonstration collection so the expert data
  already covers off-distribution states
  ([Laskey et al. 2017](http://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf)).
- **RL fine-tuning** — start from the BC policy, improve it with real reward.
  Same structure as LLM post-training: pretrain by imitation, refine with RL.

**Action chunking**, ubiquitous in modern VLAs (π₀, OpenVLA-OFT), is also a
credit-assignment/compounding-error mitigation: predict a whole *sequence* of
future actions per forward pass instead of one, which shortens the effective
decision horizon and improves temporal consistency
([PD-VLA](https://arxiv.org/pdf/2503.02310),
[adaptive chunking](https://arxiv.org/pdf/2510.12392)).

So the honest summary: **learn BC to build a VLA; learn RL to understand why
your VLA fails and what the literature is doing about it.**

---

## 11. Exercises

1. Delete the `- 0.002` time penalty. Does the car still take the short route?
2. Set the crash penalty back to `-1.0` and watch "drive into the wall" become
   attractive again. This is §2.4 in action.
3. Remove the progress term, leaving only `+5` on arrival — the sparse-reward
   case. It should fail completely. That is §4.5.
4. Add absolute `(x, y)` to the observation. It will learn faster and generalise
   worse; move the goal afterwards and see.
5. Turn off `randomize_start`. Watch it memorise one trajectory.
6. Set `batch_eps=1` in `Trainer` and watch variance wreck the training curve.
7. Widen `RAY_SPREAD` to 360°, or cut `N_RAYS` to 3. How little can it see and
   still drive?

---

## Sources

Foundations
- [Sutton & Barto, *Reinforcement Learning: An Introduction* — MDP chapter slides](https://web.stanford.edu/class/cme241/lecture_slides/rich_sutton_slides/5-6-MDPs.pdf) — the standard reference for §2–3.
- [Reinforcement Learning and Markov Decision Processes (Wiering & van Otterlo)](https://www.ai.rug.nl/~mwiering/Intro_RLBOOK.pdf)
- [An Introduction to Deep Reinforcement and Imitation Learning](https://arxiv.org/pdf/2512.08052)
- [Gymnasium — CartPole-v1 documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/) — the classic benchmark this task replaces; still the clearest statement of termination vs truncation.

Policy gradients and baselines (§6–8)
- [Policy Gradients: REINFORCE with Baseline](https://medium.com/nerd-for-tech/policy-gradients-reinforce-with-baseline-6c871a3a068) — the empirical claim that a baseline reaches 500 in ~86k steps vs ~120k without.
- [Policy Gradient Methods Explained: REINFORCE Step by Step](https://medium.com/@digitalconsumer777/policy-gradient-methods-explained-reinforce-algorithm-step-by-step-eae488d7ccd5)
- [Policy Gradient Methods in Reinforcement Learning](https://vizuara.substack.com/p/policy-gradient-methods-in-reinforcement)
- [The Role of Baselines in Policy Gradient Optimization](https://arxiv.org/pdf/2301.06276)
- [Variance Reduction for Policy-Gradient Methods via Empirical Variance Minimization](https://arxiv.org/pdf/2206.06827)

The problems (§4)
- [Decoupled Exploration and Exploitation Policies for Sample-Efficient RL](https://arxiv.org/pdf/2101.09458)
- [Search-Based Adversarial Estimates for Improving Sample Efficiency](https://arxiv.org/abs/2502.01558)
- [An Information-Theoretic Perspective on Credit Assignment in RL](https://arxiv.org/pdf/2103.06224)
- [Intrinsic Reward Policy Optimization for Sparse-Reward Environments](https://arxiv.org/abs/2601.21391)
- [Curiosity-driven Exploration in Sparse-reward MARL](https://arxiv.org/pdf/2302.10825)
- [Neuron-level Balance between Stability and Plasticity in Deep RL](https://arxiv.org/pdf/2504.08000)

Imitation learning and VLA (§10)
- [Ross, Gordon & Bagnell — DAgger](https://arxiv.org/abs/1011.0686) · [PMLR version](https://proceedings.mlr.press/v15/ross11a.html)
- [Feedback in Imitation Learning: The Three Regimes of Covariate Shift](https://arxiv.org/pdf/2102.02872)
- [DART: Noise Injection for Robust Imitation Learning](http://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf)
- [MEGA-DAgger: Imitation Learning with Multiple Imperfect Experts](https://arxiv.org/html/2303.00638v3)
- [Behavior Cloning — overview](https://www.emergentmind.com/topics/behavior-cloning)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) · [full text, action tokenization](https://arxiv.org/html/2406.09246v3)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [Vision-Language-Action Models: Concepts, Progress, Applications and Challenges](https://arxiv.org/html/2505.04769v1)
- [PD-VLA: Accelerating VLA with Action Chunking via Parallel Decoding](https://arxiv.org/pdf/2503.02310)
- [Improving Generative Behavior Cloning via Self-Guidance and Adaptive Chunking](https://arxiv.org/pdf/2510.12392)
- [Bridging Language and Action: A Survey of Language-Conditioned Robot Manipulation](https://arxiv.org/pdf/2312.10807)
