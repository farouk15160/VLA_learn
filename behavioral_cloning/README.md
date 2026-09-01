# Behavioral cloning — a car that learns to drive by watching one

```
behavioral_cloning/
├── train.py         the policy: dataset, NVIDIA CNN, trainer, live dashboard
├── drive_node.py    the policy as a ROS 2 node: Image in, Twist out
├── make_track.py    generates the Gazebo world (a figure-eight) + centre line
├── collect.py       a pure-pursuit expert drives and records demonstrations
├── evaluate.py      drives with a trained policy and scores lane-keeping
├── track.py         centre-line helpers shared by expert and evaluator
├── run_demo.sh      collect -> train -> drive, headless
└── README.md        this file
```

| | |
|---|---|
| **Method** | behavioral cloning (imitation learning), *not* reinforcement learning |
| **Input** | one 320×160 RGB camera frame |
| **Output** | one continuous steering angle in `[-1, 1]`, + is right |
| **Model** | NVIDIA CNN, 252,219 parameters ([Bojarski et al. 2016](https://arxiv.org/abs/1604.07316)) |
| **Loss** | MSE against the recorded human/expert action |
| **Deployment** | ROS 2 Humble node → Gazebo 11 differential-drive car |
| **Result** | drives the track within **1.6× of the expert** it copied |


The complete project: a driving policy learned by imitation, deployed as a
ROS 2 node, and scored driving a car in Gazebo.

`README.md` claims that **behavior cloning *is* supervised learning**, that
today's VLAs are trained with it, and that it has a structural failure called
covariate shift. This is that claim made executable: the same NVIDIA network
trained twice, deployed through ROS 2 into a Gazebo car, and **measured driving**
rather than measured on a held-out file.

---

## 0. Credit, and what was changed

The dataset and the original project are by **seraj94ai**:
<https://github.com/seraj94ai/A-self-driving-car-using-behavioral-cloning>.
The architecture is NVIDIA's, from *End to End Learning for Self-Driving Cars*
(Bojarski et al., 2016).

That repository **carries no licence file**, so none of its code is copied into
this one. `train.py` is an independent PyTorch implementation;
`--fetch` clones their repository onto your machine to obtain the recordings.
Cloning someone's data locally and redistributing it are different acts, and
only the first is defensible without a licence. The 253 MB of JPEG is
`.gitignore`d for the same reason it would be anyway: this repo does not commit
datasets.

It also could not have been run as-is. Three things were wrong before a single
epoch was possible:

| | |
|---|---|
| **Keras 1.x API** | `Convolution2D(24, 5, 5, subsample=(2, 2))`, `Adam(lr=…)`, `fit_generator` — all removed from modern Keras. Reimplemented in PyTorch, which is what the rest of this repo uses. |
| **Windows paths in the CSV** | `driving_log.csv` stores `C:\Users\seraj\…\center.jpg`. `os.path.basename` splits on `/` only, so on Linux it returns the *entire string* as the filename and every image lookup fails. Fixed with `PureWindowsPath`, which understands both separators. |
| **cv2 everywhere** | The preprocessing used `cv2.cvtColor`/`GaussianBlur`/`resize`. None of it needs OpenCV, and dropping the dependency is what lets the ROS 2 node run in this venv at all (see §4). |

The parameter count is the check that the reimplementation is faithful: NVIDIA's
table says **252,219**, and `tests/test_behavioral_cloning.py` asserts exactly
that number.

---

## 1. The data, and the one statistic that decides everything

5,186 timesteps of a human driving a simulated track. Each has **three** camera
frames — left, centre, right, 320×160 RGB — and the steering angle the human was
holding, in `[-1, 1]`. 15,558 images.

### The three-camera trick

The side cameras are not just free extra data. They are free **recovery** data.
The left camera sees roughly what the car would see if it had drifted left, so
it is labelled with the human's steering **plus** a correction back toward the
centre (`CAM_CORRECTION = 0.20`).

This matters because of a circular problem at the heart of behavioral cloning:
a *good* demonstrator is never off-centre, so the demonstrations contain no
examples of recovering from being off-centre — which is precisely the situation
the learner will create for itself the moment it makes a small error. The side
cameras manufacture that missing data for free.

### 78.1% of the labels are exactly zero

| | |
|---|---|
| frames with steering exactly 0.0 | **78.1%** |
| non-zero frames | 1,134 of 5,186 |
| mean \|steering\| when non-zero | 0.358 |

The track is mostly straight. Train on this as-is and MSE is minimised by a
model that predicts approximately zero for everything. It posts a respectable
validation loss, and it **drives straight into the first corner**.

`balance()` caps how many samples any single steering bin may contribute (25
bins, 400 each), taking 15,558 frames down to 3,971. This one function is worth
more than any architectural choice in the file, and it is data curation, not
modelling.

That is also why the reported metrics here are **correlation** and
**turn-direction agreement**, not just MSE. MSE can be beaten by a constant;
correlation cannot. Every run prints the MSE of "just predict the training
mean" as the bar to clear.

---

## 2. Preprocessing and model

```
crop rows 60:135      drop sky and bonnet — neither says anything about steering
RGB -> YUV            luma separated from chroma (BT.601)
3x3 Gaussian blur     kills JPEG speckle
resize to 200x66      NVIDIA's input geometry
scale to [0, 1]
```

```
conv 24@5x5 s2 -> 36@5x5 s2 -> 48@5x5 s2 -> 64@3x3 -> 64@3x3
  -> flatten(1152) -> 100 -> 50 -> 10 -> 1        ELU throughout
```

No pooling: all downsampling is done by strided convolutions. The head is a
regression to one float, where `supervised_learning.py` classifies into ten. That
is the only structural difference between the two files, and it is the same
choice RT-2 and OpenVLA face — they go the *other* way, discretising each action
dimension into 256 bins and training with cross-entropy.

**`preprocess()` is imported by the ROS 2 node, not reimplemented there.**
Training and serving must apply byte-identical preprocessing. The standard way
this breaks is that someone tunes the training pipeline, the serving copy
silently diverges, the robot drives worse, and no metric anywhere moves.

---

## 3. Results on the original dataset

```bash
.venv/bin/python -m behavioral_cloning.train --fetch
.venv/bin/python -m behavioral_cloning.train --headless --epochs 30
```

30 epochs, 3,971 balanced samples, CPU, 273 s:

| | |
|---|---|
| baseline MSE (predict the mean) | 0.1756 |
| **best validation MSE** | **0.0837** — 2.10× better than the baseline |
| steering correlation | 0.657 |
| turn-direction agreement | 79.5% |

The GUI shows the part a table cannot: each validation frame with the human's
steering and the model's drawn as two arrows. Where they diverge is where the
car would leave the human's line.

---

## 4. Deployment: ROS 2, and two environment problems worth writing down

`drive_node.py` subscribes to `sensor_msgs/Image`, runs the network, and
publishes `geometry_msgs/Twist`. That is the interface a real robot exposes,
which is why it is worth doing this way rather than through the original
project's Unity simulator — which is a 1 GB download that cannot be tested
headless anyway.

**Problem 1: torch is in the venv, rclpy is in ROS.** `rclpy` is a normal Python
package with extensions built for python3.10, and this venv is python3.10, so
putting ROS's `site-packages` on `PYTHONPATH` lets the venv interpreter import
`rclpy` *and* torch. The alternative — installing torch into the system
interpreter — duplicates 600 MB to work around a path.

**Problem 2: cv_bridge is broken here.** It is compiled against NumPy 1.x and
raises `AttributeError: _ARRAY_API not found` under the NumPy 2 in this venv,
and `cv2` is not in the venv at all. But a `sensor_msgs/Image` is a height, a
width, a row stride and a byte buffer, so `image_to_rgb()` decodes it by hand in
about ten lines and both dependencies disappear. The subtlety worth knowing:
**`step` is a byte stride and is not always `width*3`** — rows can be padded, and
a reshape that assumes otherwise silently skews the image.

Verified: publishing 10 dataset frames over real ROS 2 topics and comparing the
resulting `/cmd_vel` against direct inference gives a **maximum difference of
0.00e+00**.

---

## 5. The Gazebo track, and why the demo is not the obvious one

The obvious demo — train on Udacity, drive in Gazebo — **does not work, and
cannot**. Section 8 measures it. The Udacity track is a photoreal canyon road;
this one is grey slabs on green. Behavioral cloning copies a demonstrator inside
the distribution it was demonstrated in, and nothing in the method crosses that
gap. So the demonstrations are recorded *here*:

```bash
./behavioral_cloning/run_demo.sh all        # collect -> train -> drive, headless
```

* **The track is a figure-eight, not an oval,** and that is not decoration. On
  an oval every corner turns the same way: recorded steering has mean −0.094 and
  std 0.047, so a network that ignores the camera and emits one constant drives
  it perfectly. The demo would "work" and prove nothing. The figure-eight gives
  mean +0.004, std 0.161, both signs — the only way to stay on it is to look at
  the road.
* **The expert is a pure-pursuit controller** with ground-truth odometry and the
  true centre line — privileged information the network never gets. The network
  sees only the camera. This is the same teacher/student asymmetry as the UR5e
  answer sheet in `docs/grid_delivery_robot.md` §7: the teacher may cheat, the
  student may not.
* **The robot has one camera**, so the three-camera recovery trick of §1 is
  simply unavailable. `load_log()` handles the blank side columns.

Expert quality, over 6 laps: **mean cross-track error 0.075 m, max 0.157 m** on
a 3 m lane. That is the bar.

---

## 6. Does the clone drive?

Cloning 3,428 expert frames, 25 epochs:

| offline metric | value |
|---|---|
| validation MSE | 0.0004 — **94.6×** better than the baseline |
| steering correlation | **0.995** |
| turn-direction agreement | **100%** |

Offline numbers do not tell you whether a car drives, so `evaluate.py`
puts the policy in the loop and measures metres from the centre line. All runs
at 0.9 m/s on a 3 m lane (half-width 1.5 m):

| policy | mean cross-track | max | left the road? |
|---|---|---|---|
| **null baseline** — steer ≡ 0, ignores the camera | 36.5 m | 76.8 m | **after 5.5 s / 4.8 m** |
| **the expert** — pure pursuit, privileged ground truth | **0.075 m** | 0.157 m | no |
| **the clone** — camera only, 3.0 laps | **0.117 m** | 0.369 m | no |
| the same network trained on **Udacity** data (§7) | 0.49 m | 1.28 m | no |

**The clone drives, and it is within 1.6× of the teacher it copied.**

The null baseline is not a formality. It is the control that says the test can
tell good from bad: a car that ignores the camera leaves this track in five and
a half seconds. Without that row, "stayed on the road" would be an unfalsifiable
claim about a track that might simply be undemanding.

### What did NOT happen, and saying so

`README.md` warns that behavioral cloning fails by **covariate shift**: the
clone's own small errors carry it into states the expert never demonstrated, its
next action is slightly worse, and errors compound with the horizon rather than
averaging out. Validation frames come from the *expert's* trajectory; driving
frames come from the *policy's own*.

**That failure did not appear here**, and the honest thing is to say so and ask
why not:

* the lane is 3 m wide and the clone's worst excursion was 0.37 m — there is a
  great deal of room to be wrong in;
* the expert is smooth and slow (0.9 m/s), so recovering is easy and the states
  the clone drifts into are still close to demonstrated ones;
* the track repeats. Six laps of a figure-eight cover the state distribution
  densely, which is the *opposite* of the usual behavioral-cloning setting
  where each demonstration is a one-off.

So this is behavioral cloning working, under conditions that favour it. To
provoke the compounding failure you would narrow the lane, raise the speed,
record fewer laps, or start the car off-centre — and the fact that it takes
deliberate effort to break is worth knowing too. The 1.6× gap to the expert *is*
the shadow of the effect: the clone is measurably worse than the thing it copied
despite 0.995 correlation offline, and that gap exists entirely because the two
are evaluated on different state distributions.

The standard fix, when the failure does bite, is **DAgger**: drive with the
clone, have the expert label the states the clone actually visits, add them, and
repeat. The three-camera trick of §1 is a poor man's version of the same idea —
and note it was *unavailable* here, since the Gazebo robot has one camera.

---

## 7. Domain shift, measured

Running the **Udacity-trained** checkpoint on the Gazebo car, unchanged — a
photoreal canyon road's policy driving grey slabs on green:

| | mean cross-track | max | left the road? |
|---|---|---|---|
| trained in Gazebo | 0.117 m | 0.369 m | no |
| **trained on Udacity** | **0.49 m** | 1.28 m | no |

(five runs, 0.463 / 0.487 / 0.499 / 0.527 / 0.530 m — tight.)

It does not fall over, and it is **four times worse**. That is a more
interesting outcome than a clean failure. After cropping to the road band and
converting to YUV, both worlds reduce to "a dark region with bright edges,
bounded by something greener", and enough of that survives the change of world
to keep the car roughly between the lines — while every detail that would make
it precise does not transfer at all.

Partial transfer is the normal case in sim-to-real, and it is the dangerous one:
a policy that visibly *works* while being four times worse than it should be
invites you to ship it. The field's answers — domain randomisation, sim-to-real
fine-tuning, and for VLAs pretraining the visual encoder on internet-scale
images so it has seen every world already — are all attempts to widen exactly
this margin.

---

## 8. Four bugs, kept on the record

Each of these produced confident, plausible-looking output.

1. **The wheels spun about a vertical axis.** In SDF the joint axis is expressed
   in the **child link** frame. Each wheel link is rotated −90° about X so its
   cylinder lies across the car, which maps the link's local `+Z` onto the
   model's `+Y` — so the rolling axis is `0 0 1` *there*, not `0 1 0`. With the
   model-frame answer written in, the wheels turned like turntables: commanded
   0.5 m/s produced 0.08 m/s while the odometry cheerfully reported that
   everything was fine. Every controller looked broken because the robot was.
2. **The gain was applied twice.** The expert computed `steer = -alpha*gain` and
   then `yaw_rate = -steer*gain`, so the effective gain was `gain²`. The car
   oscillated hard enough to average **4.3 m** of cross-track error on a 3 m
   road. Fixed by deriving both the yaw rate and the label from the pure-pursuit
   curvature, and by defining the steering convention **once**, as `KAPPA_MAX` in
   `train.py`, imported by both the expert and the driver. A
   convention defined in two files is a convention that will disagree with
   itself.
3. **The figure-eight crosses itself.** "Nearest point on the centre line" can
   snap to the *other* branch at the crossing, which makes the expert steer onto
   the wrong loop while the cross-track error reads near zero. `track.Progress`
   only searches a window around where the car already was, so the index
   advances instead of teleporting.
4. **The measurement was measuring the spin loop.** Alternating
   `spin_once(driver)` / `spin_once(monitor)` drops callbacks while the other
   node is serviced: it processed 1,072 of ~3,000 frames and published
   `/cmd_vel` at 7 Hz instead of 20, so the car crawled at half the commanded
   speed. A single executor holding both nodes fixed it. The first "result" was
   a fact about my event loop.
5. **`--model` was silently ignored, so every run evaluated the same
   checkpoint.** `BCDriver` declares a `model` parameter and loads the
   checkpoint inside `__init__`; the evaluator constructed `BCDriver()` and set
   parameters *afterwards*, which is too late — the default `bc_model.pt` had
   already been loaded. Four "clone" runs were really the Udacity model, and
   the table in §6 nearly shipped saying the clone was 6× worse than its expert
   when it is 1.6×. Fixed with `parameter_overrides` at construction. The tell
   was there in the node's own startup log the whole time, filtered out of view
   by a `grep -v INFO`. **Never filter the line that says what you loaded.**
6. **Two simulators at once.** A run that killed `gzserver` and started another
   three seconds later sometimes had both alive, publishing to the same topics.
   That produced one spectacular "LEFT THE ROAD after 46 s, max 7.5 m" outlier
   among otherwise tightly clustered runs, plus a stream of
   `sequence size exceeds remaining buffer` deserialisation errors. Waiting for
   `/camera/image_raw` to appear, and for the old process to actually exit,
   removed both.

---

## 9. The live dashboard

`python -m behavioral_cloning.train` opens a dashboard that updates on **every
weight update**, not once an epoch, because the interesting failures are visible
in the first thirty seconds and invisible in a final number.

| panel | what it shows | why it is there |
|---|---|---|
| **STATE + ACTION** | the camera frame, the crop rows as yellow lines, and beneath it the actual 66×200 YUV tensor the network is fed | you cannot debug a vision policy without seeing what it is actually looking at — the second image is the one the network sees, and it is startlingly unlike the first |
| | two arrows on the frame: the human's steering (yellow) and the model's (blue) | where they diverge is where the car would leave the human's line |
| **LEARNING** | per-batch training loss (one point per weight update), smoothed, with per-epoch validation over it, log scale | the shape of the first hundred updates tells you whether the learning rate is sane long before the epoch number does |
| | the **gradient norm** on the right axis | → 0 means learning has stalled; spikes mean it is diverging. Neither is visible in the loss curve, and both are common |
| | a dashed **"predict the mean"** line | the bar. A model that does not clearly beat it has learned nothing, whatever its MSE looks like |
| **SCORE** | model action vs human action, with the identity line | a perfect model lies on the red line. A model that has collapsed to a constant is a horizontal stripe — instantly visible, and invisible in MSE |
| **DIAGNOSTICS** | correlation, turn-direction agreement, MAE per epoch | correlation is the honest headline: a constant predictor scores 0 |
| **DATA** | steering histogram: recorded (grey) vs kept after balancing (green) vs what the model predicts (outline) | this is where you see the 78%-zero spike, and whether the model's output distribution has collapsed narrower than the data's |
| **metrics panel** | device, parameters, epoch, weight updates, images seen, elapsed, every loss, best-so-far, ×baseline, and this frame's two actions | the numbers, aligned, so a screenshot is a complete record of the run |
| **event log** | timestamped epoch results, new bests, checkpoints saved | `What am I looking at?` prints this table into the log |

### There are no rewards here, and that is the point

The dashboard shows **scores, not rewards**. Behavioral cloning is supervised
learning: the signal is the distance to a recorded human action on a fixed
dataset, available for every frame, immediately. Reinforcement learning's reward
is a scalar from an environment, arriving late and telling you only *how well*
you did, never *what you should have done*. The two are not interchangeable, and
`reinforcement_learning.py` in the parent directory is the other half.

The one true *score* for a driving policy is cross-track error with the car in
the loop, and no offline panel can show it — hence `run_demo.sh drive`.

---

## 10. What else could drive this car? (and how a real one does)

### The imitation-learning family

Behavioral cloning is the simplest member, and its weakness is §6: it assumes
the states it will see at test time are the states the expert demonstrated.

| method | idea | what it costs |
|---|---|---|
| **Behavioral cloning** (this file) | supervised regression from state to expert action | nothing beyond a dataset — and it inherits covariate shift |
| **DAgger** (Ross et al. 2011) | roll out the *learner*, ask the expert to label the states the learner actually visits, aggregate, repeat | needs an expert available **during** training, interactively. In this project that is cheap — `collect.py`'s pure-pursuit expert can label any state — which is exactly why a Gazebo expert is more useful than a recorded human |
| **Inverse RL** | infer the *reward function* the expert was optimising, then solve for a policy | far more expensive, but the reward transfers to new tracks and new vehicles in a way a cloned policy does not |
| **GAIL** | a discriminator tries to tell learner trajectories from expert ones; the policy is trained to fool it | no hand-designed reward, but adversarial training is unstable |

### Why not just use RL?

You can, in simulation — and for this track it would work. On a real car it is
close to unusable, for one blunt reason: **RL learns by trying things that do
not work.** An exploring policy on a public road proposes actions that crash,
and there is no way to obtain the reward signal for "drove into the barrier"
without driving into the barrier. This is the standard reason the literature
gives for confining driving RL to simulation or to safety-cage designs where a
classical controller can veto the policy's action.

That, in one line, is why the industry's answer is *imitation from enormous
amounts of human driving*, not RL from scratch — and why RL shows up mainly in
simulation, offline RL from logged fleet data, and as a fine-tuning step on top
of an imitation-learned policy.

### How a real self-driving car actually drives

Two architectures, and the field has been converging.

**The modular pipeline**, the classical answer and still the backbone of most
deployed L4 systems:

```
sensors → perception → prediction → planning → control
(camera,   what and    where will   what should  steering,
 lidar,    where is    they go?     I do?        throttle,
 radar)    everything?                           brake
```

Each stage is separately testable, separately verifiable, and separately
blameable when something goes wrong — which is what makes it defensible to a
regulator. The cost is that hand-designed interfaces between stages throw
information away, and errors compound across them.

**End-to-end learning**, which is what this project is a miniature of: sensor
data in, control out, one network, trained by imitation on very large amounts of
human driving. Tesla's FSD moved to an end-to-end formulation in v12, and by
2024–25 end-to-end stacks trained on fleet-scale data had begun to beat modular
ones on comfort and progress metrics. Waymo, long the standard-bearer for the
modular approach, has been moving the same direction. Both companies launched
commercial robotaxi services in Austin in June 2025.

The practical difference between them is **data**: end-to-end needs enormous,
diverse demonstration coverage, which is why it favours whoever has a large
fleet. Everything else — the 78%-zero imbalance, the recovery-data problem, the
gap between offline metrics and closed-loop driving — is the same problem this
project has, at a different scale.

**Where this connects to VLAs.** An end-to-end driving policy is a
vision-to-action model. A VLA (RT-2, OpenVLA, π₀) is the same thing with an
instruction added to the input and a robot arm on the output, trained the same
way: behavioral cloning on human demonstrations. Everything in §6 about
covariate shift applies to them unchanged, which is what the repository's
one-line summary means by *learn BC to build a VLA, learn RL to understand why
it fails*.

Sources: [Bojarski et al., *End to End Learning for Self-Driving Cars*
(arXiv:1604.07316)](https://arxiv.org/abs/1604.07316) ·
[*End-to-end Autonomous Driving: A Systematic Review*
(arXiv:2311.18636)](https://arxiv.org/pdf/2311.18636) ·
[*Motion Planning for Autonomous Driving: State of the Art*
(arXiv:2303.09824)](https://arxiv.org/pdf/2303.09824) ·
[*Safe RL on Autonomous Vehicles* (arXiv:1910.00399)](https://arxiv.org/pdf/1910.00399) ·
[IDTechEx, *Is End-to-End the Endgame for Level 4 Autonomy?*](https://www.idtechex.com/en/research-article/is-end-to-end-the-endgame-for-level-4-autonomy/33591)

---

## 11. Running it

```bash
# the original dataset (498 MB, cloned from seraj94ai's repo, gitignored)
.venv/bin/python -m behavioral_cloning.train --fetch
.venv/bin/python -m behavioral_cloning.train                # dashboard
.venv/bin/python -m behavioral_cloning.train --headless --epochs 30

# the ablation: keep the 78% zeros and watch a good val loss produce a
# car that cannot turn
.venv/bin/python -m behavioral_cloning.train --headless --unbalanced --epochs 10

# the Gazebo pipeline, all headless (GUI=1 to watch in gzclient)
./behavioral_cloning/run_demo.sh collect     # expert records demonstrations
./behavioral_cloning/run_demo.sh train       # clone them
./behavioral_cloning/run_demo.sh drive       # drive with the clone, scored
./behavioral_cloning/run_demo.sh all
```

```bash
./behavioral_cloning/run_demo.sh stop     # kill a leftover simulator
```

Requires ROS 2 Humble and Gazebo 11 for the simulation parts; the trainer and
its dashboard need neither.

**One operational wrinkle worth knowing.** `gzserver` does not reliably die with
the script that started it — it is a wrapper that execs the real server, and the
survivor is reparented to init. A leftover simulator publishes to the same
topics as the next one, and two of them together produce measurements that look
plausible and are meaningless (§8, bug 6). So `run_demo.sh` **refuses to start**
when one is already running, and `run_demo.sh stop` clears it. If a driving
score ever looks strange, check `pgrep -x gzserver` first.
