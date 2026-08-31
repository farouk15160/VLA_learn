# VLA_learn — the two things every VLA paper assumes you already know

Two runnable programs, each a single file, each with a GUI you can watch learn.

```
VLA_learn/
├── supervised_learning.py       SL: recognise digits in real RGB photographs
├── reinforcement_learning.py    RL: a car learns to drive from A to B
├── docs/
│   ├── supervised_learning.md   the data spec, the model, and the 8 ways SL lies to you
│   └── reinforcement_learning.md  RL from zero: the MDP, policy gradients, baselines
└── data/svhn_test_32x32.mat     61 MB, already downloaded
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
```

Press **▶ Train** in either window. Nothing trains until you do.

```bash
# terminal-only versions
.venv/bin/python supervised_learning.py    --headless --epochs 12
.venv/bin/python reinforcement_learning.py --headless --updates 120

# the ablations quoted in the docs, so you can reproduce every number
.venv/bin/python supervised_learning.py    --sweep
.venv/bin/python reinforcement_learning.py --headless --updates 120 --no-baseline
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

Every number quoted in the docs was measured on this machine and is reproducible
with a flag. Where a result contradicted what I expected — the RGB ablation, the
dropout sweep — the docs say so.
