# Supervised Learning — digits in real RGB photographs

Companion to `supervised_learning.py`. This concentrates on three things: **the exact
specification of what this file does**, **the discipline that keeps your numbers
honest**, and **how this pipeline becomes a robot policy**.

---

## 1. What this file is really building

| this file | a Vision-Language-Action policy |
|---|---|
| 32×32×3 RGB photo | camera frame, 224×224×3 |
| → small CNN | → ViT + language model, billions of params |
| → 10 class logits | → 256 action-bin logits, per action dimension |
| cross-entropy loss | cross-entropy loss |
| `argmax` = predicted digit | `argmax` = the action token to execute |
| labelled by a human annotator | labelled by a human teleoperator |

This is not a loose analogy. OpenVLA discretizes *"each dimension of the robot
actions separately into one of 256 bins"* and is *"trained with a standard
next-token prediction objective, evaluating the cross-entropy loss on the
predicted action tokens only"* ([OpenVLA](https://arxiv.org/html/2406.09246v3)).
RT-2 does the same ([RT-2](https://arxiv.org/abs/2307.15818)).

**Behavior cloning — the way essentially every VLA is trained today — is exactly
this file.** Supervised classification, with "which action did the human take"
as the label.

---

## 2. The data, specified

### 2.1 What SVHN is

**S**treet **V**iew **H**ouse **N**umbers: 26,032 crops of house numbers
photographed from Google Street View
([source](http://ufldl.stanford.edu/housenumbers/)). Real photographs — blurry,
badly lit, shot at angles, and frequently with a *neighbouring digit* intruding
at the edge of the crop. Several of the model's remaining errors are cases where
you would hesitate too.

### 2.2 Resolution: 32 × 32 × 3

Each image is **32 pixels wide, 32 tall, 3 colour channels = 3,072 numbers**.
The digit is centred, but its neighbours often are not.

Why 32? Small enough that an epoch takes ~2.5 s so you can watch it learn, big
enough that a digit is legible and three rounds of 2× max-pooling still leave a
4×4 feature map. That is not a guess — `--sweep` measures it:

```
  input           test acc
  32x32x3           0.9294
  16x16x3           0.9193      <- halving resolution costs ~1 point
  8x8x3             0.8149      <- quartering it costs 11 points
```

Halving to 16×16 costs about one point, which is *nearly free* — worth knowing
if you ever need to make a model 4× cheaper. Going to 8×8 falls off a cliff,
because a stroke of a digit stops being more than a pixel wide. This is the same
8×8 resolution as the classic toy digits dataset, and the difference is that
those were clean renderings while these are photographs — at 8×8 a photographed
digit is mush.

The network uses `nn.AdaptiveAvgPool2d((4,4))` before the classifier head
precisely so all three of those runs use an otherwise identical architecture.

### 2.3 The RGB channels — how they are used, and whether they help

**Layout.** PyTorch is channels-first: the tensor is `(N, 3, 32, 32)` =
`[batch, channel, height, width]`. The raw `.mat` file is `(32, 32, 3, N)`, so
`svhn` transposes it. Values go `0..255 → 0..1`.

**Normalization is per channel, not per pixel:**

```python
mu = X_train.mean(axis=(0, 2, 3))   # one number for R, one for G, one for B
sd = X_train.std(axis=(0, 2, 3))
```

Averaging over axes `(0, 2, 3)` collapses batch, height and width, leaving one
statistic per colour channel. **Per-pixel statistics would be wrong here**: a
digit can sit anywhere in the crop, so per-pixel means would bake in *position*,
which is not what we want to normalize away. Measured on this split:

```
  channel means   R 0.452   G 0.452   B 0.468
  channel stds    R 0.219   G 0.226   B 0.228
```

The channels genuinely differ — daylight colour temperature and sensor response
push blue up slightly. And as always, the statistics come from the **training
split only**; using the whole dataset leaks test information into training.

**Where RGB enters the network.** The first layer is `Conv2d(3, 32, 3)`. Each of
its 32 filters is a **3×3×3 cube** — 3 wide, 3 tall, 3 deep across the colour
channels — so a filter *can* respond to a red-on-white edge differently from a
blue-on-white one. After that first layer colour is gone as a concept: layer 2
sees 32 abstract feature maps, not R/G/B.

**Does colour actually help? I measured it, and no.**

```
  input                      test acc
  RGB (3 real channels)        0.9262
  greyscale, copied to 3ch     0.9299
```

Identical architecture, identical parameter count — only the *information*
differs. That is a tie (one standard error on 4,032 test images is ~0.4pp, and
the gap is smaller than that). **I expected colour to help before I ran this.**

The honest reading: on SVHN, digit identity is carried by **shape and local
contrast**, not hue. House numbers come in every colour, so colour is mostly
nuisance variation the network must learn to *ignore*. Keeping RGB costs almost
nothing and is the right default when you do not know, but "it is in colour so
colour must be useful" is an assumption, and this is what checking it looks like.

### 2.4 Labels, and a trap

Labels are digits **0–9**. But SVHN stores the digit `0` as class **10**, not 0.
Forget to remap and you get a model that is confidently wrong about every zero.
The file does `y % 10`.

### 2.5 Class balance — genuinely skewed

Printed at startup:

```
  class counts (train): [1219, 3569, 2818, 1944, 1769, 1660, 1363, 1398, 1157, 1103]
                           0     1     2     3     4     5     6     7     8     9
```

**3,569 examples of `1` but only 1,103 of `9`.** House numbers really are like
that — low digits appear more often. Plain accuracy is therefore slightly
flattering, and per-class recall is the honest measure. See §4.5.

### 2.6 The three-way split — and what val vs test actually means

```
train  (18,000) -> the optimizer sees it; gradients are computed from it
val     (4,000) -> YOU see it; you make decisions with it
test    (4,032) -> touched exactly once, at the very end
```

**The one-line version:** *validation is for choosing, test is for reporting.*

That distinction matters because **you** are part of the training loop.
Validation is data the *optimizer* never sees but that *you* look at constantly,
to choose epochs, width, learning rate, architecture. Each of those choices is
fitted to the validation set — by you, by hand, one experiment at a time. Try 40
configurations and keep the best validation score and you have partly selected
for *luck on those 4,000 images*. Test is data neither the optimizer nor you has
used for any decision; the moment you look at it and change something, it has
become a second validation set.

| you want to… | use |
|---|---|
| pick the number of epochs | val |
| pick architecture / dropout / lr | val |
| detect overfitting while training | val |
| report a number to anyone | test |

**Size caveat.** 4,032 test images means one image is worth 0.025%. The standard
error of an accuracy estimate is roughly $\sqrt{p(1-p)/n}$, here
$\sqrt{0.94 \times 0.06 / 4032} \approx 0.37\%$. Treat anything inside ±0.75%
(two standard errors) as a tie — which is exactly why §2.3 calls the RGB result
a tie rather than a win for greyscale.

**The robotics version, where people actually get burned.** The natural thing is
a random split over frames. That is wrong: two frames 33 ms apart are nearly
identical, so a random split puts near-duplicates of test data into training.
Your accuracy looks wonderful and the robot fails. Hold out **entire episodes**,
and ideally entire scenes, objects and lighting conditions.

---

## 3. The model, specified

```
  Conv(3→32)  BN ReLU   Conv(32→32) BN ReLU   MaxPool     32×32 → 16×16
  Conv(32→64) BN ReLU   Conv(64→64) BN ReLU   MaxPool     16×16 → 8×8
  Conv(64→128) BN ReLU                        MaxPool     8×8   → 4×4
  AdaptiveAvgPool(4×4)
  Flatten(2048)  Dropout  Linear(256) ReLU  Dropout  Linear(10)
```

**667,178 parameters.** Loss: `CrossEntropyLoss` on raw logits. Optimiser: Adam,
`lr=1e-3`, `weight_decay=1e-4`. Batch 128. ~2.5 s/epoch on the RTX 2060.
**Result: ~94% test accuracy in 8–13 epochs.**

### Why a CNN and not a flat MLP

Flattening a 32×32×3 photo into 3,072 independent inputs throws away the fact
that neighbouring pixels are related. A convolution slides the **same** small
filter across every position, which buys two things:

1. **Far fewer parameters.** A dense layer from 3,072 inputs to 3,072 outputs is
   9.4M weights. Our entire network is 667k.
2. **Translation equivariance.** It recognises a `7` wherever it appears in the
   crop, instead of learning "7-in-the-top-left" and "7-in-the-middle" as
   unrelated patterns. Since SVHN digits are centred but not perfectly, this
   matters.

`BatchNorm` after each convolution normalizes activations per channel across the
batch, which keeps gradients well-scaled and lets you train faster; it is why
this network reaches 76% validation accuracy after a *single* epoch.

### 3.1 What `CrossEntropyLoss` actually computes

The model outputs 10 raw numbers called **logits** — unbounded scores, one per
class. They are not probabilities: they can be negative, and they do not sum
to 1. Cross-entropy does two things.

**Step 1: softmax**, turning logits into probabilities:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Exponentiate (everything becomes positive), then divide by the total (now sums
to 1).

**Step 2: negative log-likelihood** of the correct class:

$$\mathcal{L} = -\log p_{y}$$

That is the whole loss. Only the *true* class's probability appears.

Why the log? Look at its shape:

```
   p_correct    loss = -log(p)
      0.99         0.01      confident and right   -> almost no gradient
      0.50         0.69
      0.10         2.30
      0.01         4.61      confident and wrong   -> large gradient
      0.001        6.91
```

The log makes confident mistakes enormously expensive while confident correct
answers cost nearly nothing. That is the behaviour you want: the gradient
concentrates on the examples the model is getting badly wrong. (Squared error
would punish a confident mistake far more gently, which is why it trains poorly
for classification.)

**Why no softmax in the model.** `nn.CrossEntropyLoss` does the softmax
*internally*, for numerical stability — `log(softmax(x))` computed as one
operation avoids overflow when logits are large. Verified against the installed
torch 2.13:

```
CrossEntropyLoss(logits)            = 0.417030
NLLLoss(log_softmax(logits))        = 0.417030   <- identical => log_softmax is internal
CrossEntropyLoss(softmax(logits))   = 0.802121   <- the double-softmax bug
```

Applying softmax yourself as well does not crash. It silently squashes your
logits into 0–1, which flattens the gradients and makes training mysteriously
slow. It is one of the most common beginner bugs in PyTorch precisely because
nothing complains.

**The VLA connection:** this same loss, over 256 action bins instead of 10
digits, is how OpenVLA and RT-2 are trained.

### 3.2 What dropout is, and why use it

During training, dropout randomly sets a fraction `p` of a layer's activations
to zero — a different random set every forward pass. At evaluation time it is
switched off entirely and nothing is dropped. That switch is what
`model.train()` and `model.eval()` control, and forgetting `model.eval()` is a
classic bug: your validation scores become noisy and pessimistic for no visible
reason.

**Why deliberately break your own network?** Because it prevents *co-adaptation*.
Without dropout, the network can build a fragile chain — "unit 12 detects the
top-left stroke, and unit 30 only works if unit 12 fires". That chain memorizes
training data and shatters on anything new. If unit 12 might vanish on any given
step, no other unit can afford to depend on it. Every unit must be independently
useful, and the network is forced into redundant, distributed representations.

A second way to see it: with dropout you are training an ensemble of
exponentially many sub-networks that share weights, and averaging them at test
time. Ensembles generalize better than their members.

Measured on this CNN with `--sweep` (6 epochs, 2 seeds, test accuracy):

```
  dropout   test acc
  0.00        0.9267
  0.15        0.9272
  0.30        0.9257     <- what the file uses
  0.50        0.9177
```

Honest reading: **at this training length dropout buys nothing measurable** —
the first three rows are a tie well inside the ±0.75% noise band from §2.6. Only
0.50 is clearly worse, because dropping half the activations removes real
capacity. Dropout is *insurance*, and a 6-epoch run is not a risky enough
situation to need it. Train for 60 epochs, as in §4.1, and the regularizers earn
their place.

That is worth internalising as a general point: **a regularizer that does
nothing in a short run is not useless, it is untested.** You are measuring it in
a regime where nothing was going wrong anyway.

**Practical rule:** 0.1–0.3 on the dense head, and usually none on the
convolutional layers (BatchNorm already regularizes them). Never on the output
layer. If you are not overfitting, dropout is not what you need.

### 3.3 What weight decay is

The other regularizer in the script, `weight_decay=1e-4` in the optimizer. It
adds a penalty proportional to the squared size of every weight, so each step
pulls all weights slightly toward zero unless the data pushes back.

Why that helps: large weights mean the output changes sharply for a small change
in input — a jagged decision boundary that carves out individual training points.
Keeping weights small keeps the function smooth, and a smoother function is more
likely to be right between the training points.

Dropout and weight decay attack the same problem from different directions:
dropout prevents *dependence between units*, weight decay prevents *any single
weight from becoming dominant*. Using both is standard.

### 3.4 Why `opt.zero_grad()` is not optional

```python
opt.zero_grad()   # clear the previous step's gradients
loss.backward()   # ADD this step's gradients into .grad
opt.step()        # update weights using .grad
```

**PyTorch accumulates gradients by default.** `loss.backward()` does not
*assign* to `param.grad`, it does `param.grad += ...`. So if you omit
`zero_grad()`, step 5 uses the sum of the gradients from steps 1–5, step 100
uses the sum of 100 gradients, and your effective learning rate grows without
bound. Training appears to work for a moment and then diverges to NaN.

**Why would a framework do that?** Because accumulation is a feature. It lets
you simulate a large batch on a small GPU:

```python
for i, (xb, yb) in enumerate(loader):
    loss = loss_fn(model(xb), yb) / accum_steps
    loss.backward()                     # accumulate, don't step
    if (i + 1) % accum_steps == 0:
        opt.step(); opt.zero_grad()     # step once per accum_steps batches
```

This is called *gradient accumulation* and it is how people train large models
on modest hardware. It is also standard when fine-tuning a VLA — you want an
effective batch of 256 and your GPU holds 8. The cost of that flexibility is
that you must clear the gradients yourself in the normal case.

Order matters slightly: `zero_grad()` must come before `backward()`, not between
`backward()` and `step()` — that would erase the gradients you just computed and
`step()` would do nothing.

### 3.5 What an epoch is, and how many to run

**One epoch = one full pass over the training set.** With 18,000 images and
`batch=128`, that is `ceil(18000/128) = 141` optimizer steps. 10 epochs = 1,410
weight updates.

Two details in the loop:

- **Reshuffle every epoch** (`torch.randperm`). Without it the model sees
  identical batches in identical order, which correlates gradient noise across
  epochs.
- **Batch size trades noise for speed.** Small batches give noisy gradients
  (mildly regularizing) and poor GPU utilization; large batches are smoother and
  faster per epoch. 32–256 is the normal range.

**How many epochs?** You do not choose this number — validation chooses it. The
file keeps the **best-validation checkpoint**, so overshooting is harmless:

```
  ep  1  tr_acc 0.3641  va_acc 0.7605      <- BatchNorm doing its work
  ep  2  tr_acc 0.7817  va_acc 0.8832
  ep  3  tr_acc 0.8527  va_acc 0.9040
  ...
  ep 12  tr_acc 0.9497  va_acc 0.9460  (best 0.9460 @ ep 12)
```

The procedure: set `epochs` generously, keep the best-val checkpoint, and check
that the best epoch is comfortably below your budget. **If `best epoch` equals
`epochs`, you stopped too early** — that one check costs nothing.

### 3.6 So how do you choose all of this?

In order, and not all at once:

1. **Get the data right first.** Correct splits, no leakage, look at the images.
   No hyperparameter recovers from a broken split.
2. **Get a baseline running end to end** with boring defaults.
3. **Read the generalization gap** (§5). It tells you which problem you have:
   train accuracy low → underfitting (more capacity, train longer); train high
   and val much lower → overfitting (more data, more regularization, stop
   earlier); both good and gap small → you are done.
4. **Change one thing at a time**, average over ≥2 seeds, and only believe
   differences bigger than the noise (§2.6). The dropout sweep below is a good
   example of a result that looks like a trend and is actually a tie:

```
  dropout   test acc
  0.00        0.9267
  0.15        0.9272
  0.30        0.9257      <- the default
  0.50        0.9177      <- only this one is really worse
```

5. **Touch the test set once**, at the end.

---

## 4. The problems in supervised learning

### 4.1 Overfitting ← *reproduce with `--headless --epochs 60`*

The model memorizes training examples instead of learning the pattern. This
network is well regularized, so it takes a while to show — but it does:

```
  ep   1  tr_acc 0.3502  va_acc 0.7575  gap -0.4073   tr_loss 1.8450  va_loss 0.8825
  ep  10  tr_acc 0.9358  va_acc 0.9457  gap -0.0100   tr_loss 0.2110  va_loss 0.2085
  ep  20  tr_acc 0.9669  va_acc 0.9443  gap +0.0226   tr_loss 0.1083  va_loss 0.2461
  ep  40  tr_acc 0.9846  va_acc 0.9517  gap +0.0329   tr_loss 0.0505  va_loss 0.2435
  ep  60  tr_acc 0.9904  va_acc 0.9575  gap +0.0329   tr_loss 0.0302  va_loss 0.2761
```

Read the two loss columns. **Training loss falls by 60× (1.845 → 0.030) while
validation loss bottoms out at 0.2054 around epoch 19 and then climbs 34% to
0.2761.** Two curves moving in opposite directions is the definition of
overfitting: after epoch ~19 the network is no longer learning about digits, it
is memorising these particular 18,000 photographs.

Notice something subtle though: **validation *accuracy* keeps creeping up**
(0.9443 → 0.9575) even while validation *loss* gets worse. That is not a
contradiction. The model is becoming more confident on the examples it already
gets right and more confidently wrong on the ones it does not — accuracy only
counts the argmax, but cross-entropy also prices the confidence. When those two
disagree, loss is telling you about calibration (§4.7) and accuracy is telling
you about decisions. Pick the one that matches what you actually care about.

Note also `gap` in the first epochs is **negative** — validation accuracy is
*higher* than training accuracy. That is not a bug: dropout is active during
training and disabled during evaluation, and the training figure is averaged
over an epoch during which the model was still improving.

**The defences**, all present in the file: more data (the best one), a smaller
model, **dropout**, **weight decay**, and **best-checkpoint selection** — which
is why the run above still returns the epoch-38 weights (val 0.9595) rather than
the overfit epoch-60 ones.

### 4.2 Underfitting
The opposite: train *and* val are both bad, and the gap is ~0. The model is too
small, the learning rate is wrong, or you stopped too early. Diagnose by the
gap: large gap = overfitting, no gap but bad scores = underfitting.

### 4.3 Data leakage
Any path by which information about evaluation data reaches training. Whole-
dataset normalization (§2.2), duplicated rows across splits, a feature computed
using future information, random frame splits in robotics (§2.1). Leakage
produces *great* offline numbers and total deployment failure, which makes it
the most expensive bug class in ML — you only find out at the end.

### 4.4 Distribution shift
Train and deployment data differ. Digits from a different scanner; a robot
trained in the lab and deployed in a kitchen. Your test set only measures
generalization to data drawn the *same way* as training. It says nothing about a
new lighting condition.

For imitation learning there is a sharper version, **covariate shift**, and it is
the single most important idea to carry from this file into VLA work: a cloned
policy is trained on states the *expert* visited, but at deployment it visits
states *it* reaches. Its first small error moves it off-distribution, where it
was never trained, so the next error is larger — errors compound, and in the
worst case imitation error grows **quadratically** with the horizon
([Three Regimes of Covariate Shift](https://arxiv.org/pdf/2102.02872)).

**This is why a BC policy can score 99% on held-out frames and still fail on the
robot.** The fixes (DAgger, DART, RL fine-tuning) are covered in
[`../reinforcement_learning/README.md` §10](../reinforcement_learning/README.md).

### 4.5 Class imbalance ← *real in this dataset*

If 95% of your data is one class, 95% accuracy means "always guess that class".
Our data is genuinely imbalanced (§2.5): 3,569 examples of `1` against 1,103 of
`9`. Measured per-class recall on the test set:

```
  digit 4: recall 0.985  (n=392)      digit 7: recall 0.948  (n=305)
  digit 1: recall 0.973  (n=745)      digit 3: recall 0.939  (n=492)
  digit 2: recall 0.967  (n=643)      digit 9: recall 0.936  (n=249)
  digit 0: recall 0.957  (n=258)      digit 6: recall 0.929  (n=324)
  digit 5: recall 0.955  (n=357)      digit 8: recall 0.873  (n=267)
```

Overall accuracy is 0.9521, but **digit 8 is only recognised 87.3% of the
time** — five points below the headline number, and the worst class by a clear
margin. The single aggregate number hides that completely.

This matters far more in robotics than it does here. Real teleoperation data is
brutally imbalanced: most timesteps are "move slightly forward", and very few
are the critical grasp. A policy that is 95% accurate overall but fails the
grasp is worthless. Use per-class recall, or reweight the loss.

### 4.6 Label noise
Human annotators disagree and make mistakes; teleoperators demonstrate a task
several different ways. A big unregularized net will happily memorize the noise —
that is what the rising validation loss in §4.1 is. Note that "two valid ways to do
the task" is *worse* than random noise for a robot: averaging two good
trajectories can give one that hits the obstacle between them. This
multimodality is exactly why modern VLAs use diffusion or flow-matching action
heads instead of predicting one mean action.

### 4.7 Overconfidence / miscalibration ← *measured by the script*

Actual output from a run:

```
mean confidence when RIGHT: 0.985
mean confidence when WRONG: 0.758      (193 errors out of 4,032)
```

A well-calibrated model should be *unsure* when it is wrong. Ours is **76%
confident on its mistakes** — you can watch this directly in the GUI's upload
panel, where a wrong answer still shows one tall bar. Neural nets trained with cross-entropy are
systematically overconfident, and for a robot this is what makes "the policy
knows when to stop and ask for help" hard to build. Temperature scaling, deep
ensembles and evidential methods all attack this.

### 4.8 Accuracy hides everything

`0.9521` is one number over 4,032 photographs. Which mistakes does it make?

```
  most confused pairs:
    true 8 -> predicted 3   (9x)
    true 7 -> predicted 1   (9x)
    true 6 -> predicted 0   (8x)
    true 6 -> predicted 5   (8x)
```

Every one of those is a pair that genuinely looks alike at 32×32 in a blurry
photograph — 8 and 3 share a right half, 7 and 1 share a vertical stroke. That
is reassuring: the model is failing the way a human squinting at the same crop
would fail, not in some arbitrary way that would suggest a bug.

For a robot this distinction is everything: a policy 95% accurate overall but
always wrong on "close gripper" is useless. Always look at *which* errors.

---

## 5. Reading the output

**In the terminal / left panel:**

```
epoch      12
train acc  0.9497
val   acc  0.9460   (best 0.9460 @ ep 12)
gap        +0.0037   <- grows = overfitting
```

`gap = train_acc − val_acc` is the single most useful diagnostic in the whole
readout. Small and stable is healthy; growing means overfitting.

**In the GUI:** the 40 tiles are fixed validation photos, re-predicted after
*every* epoch, with borders turning **green** when correct and **red** when
wrong. It is the same information as the accuracy curve, but you can see *which*
images it wins and where it still fails — and the failures are informative
(usually a neighbouring digit intruding into the crop).

**The upload panel** shows the full softmax distribution, not just the guess. A
tall single bar means confident; several similar bars mean it is torn. Watch how
confident it is when it is **wrong** — that is the calibration problem (§4.7).
It also shows the downsampled 32×32 the network actually receives, which usually
explains any surprising answer, plus the three colour channels split out.

**Uploading your own image:** crop tightly to a *single* digit. The model was
trained on tight 32×32 crops, so a wide photo of a whole door will fail — not
because the model is bad, but because that is distribution shift (§4.4).

---

## 6. Exercises

1. `--sweep` — reproduce the RGB, resolution and dropout tables yourself.
2. Train on 16×16 (`_variant(d, "res16")`) and note it costs ~1 point for 4× less
   compute. When would you take that trade?
3. Set `--epochs 60`. Watch the gap grow and confirm best-checkpointing saves you.
4. Delete the `y % 10` remap and look at the confusion on zeros.
5. Train on digits 0–7 only, then evaluate on 8 and 9. That is distribution
   shift (§4.4) at its most brutal, and it is what a robot meeting a new object
   feels like.
6. The bridge to RL: keep this architecture, but make the label "which action did
   the expert take". You have just written behavior cloning.

---

## Sources

- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) · [full text — action tokenization into 256 bins, cross-entropy objective](https://arxiv.org/html/2406.09246v3)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [Vision-Language-Action Models: Concepts, Progress, Applications and Challenges](https://arxiv.org/html/2505.04769v1)
- [Feedback in Imitation Learning: The Three Regimes of Covariate Shift](https://arxiv.org/pdf/2102.02872) — the quadratic-in-horizon compounding error result.
- [Behavior Cloning — overview](https://www.emergentmind.com/topics/behavior-cloning)
- [Ross, Gordon & Bagnell — DAgger](https://arxiv.org/abs/1011.0686)
- [Improving Generative Behavior Cloning via Self-Guidance and Adaptive Chunking](https://arxiv.org/pdf/2510.12392) — on multimodal action distributions (§4.6).
- [scikit-learn `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) — the dataset used here.
- [PyTorch `CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — the log-softmax-is-internal claim in §2.3 was verified numerically against the installed torch 2.13, not just cited.
