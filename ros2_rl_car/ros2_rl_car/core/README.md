# MDP, reward, and PPO math

## Markov decision process

The simulator's internal state contains pose, twist, contacts, the nearest
centre-line segment, unwrapped arc progress, lap count, and simulated time. The
policy deliberately receives less information:

1. 31 lidar ranges resampled across the forward fan, clipped to the physical
   sensor range and divided by maximum range;
2. forward speed divided by the configured maximum speed;
3. `sin(heading_error)` and `cos(heading_error)`, avoiding the discontinuity at
   `-pi/+pi`;
4. signed cross-track error divided by road half-width.

There is no absolute `x` or `y`. Pose is privileged information used only by
reward calculation, evaluation, and the pure-pursuit reference. Camera images
are published but absent from v1 observations. On this host PyTorch reports no
CUDA device. A 35-input, two-layer MLP can learn from lidar on CPU; an
end-to-end CNN policy trained from pixels would require far more samples and can
take overnight without converging. Pixel learning is therefore a stretch goal,
not a hidden dependency.

The action is categorical: `LEFT`, `STRAIGHT`, or `RIGHT`. Forward speed is
fixed at 0.8 m/s and the action selects ROS `angular.z = +0.6, 0, -0.6` rad/s.
Positive angular velocity means a left turn
under REP-103. This convention is declared once in `constants.py`; every
consumer imports it.

## Track geometry

For each point `p`, projection considers every centre-line segment `a -> b`:

```text
t = clip(dot(p-a, b-a) / dot(b-a, b-a), 0, 1)
q = a + t(b-a)
e = cross(b-a, p-q) / ||b-a||
s = cumulative_length(a) + t||b-a||
```

The generated closed curve is intentionally non-oval and its signed curvature
contains both signs. The CSV is the source of truth shared by reward and scorer.
Progress differences unwrap the final-to-first seam and reject implausible jumps
caused by teleport/reset.

## Reward and shortcut bound

For road half-width `h`, one transition receives

```text
r_t = w_p * delta_s / L
    + w_c * (|e_t| - |e_(t+1)|)
    - c_time
    + terminal_bonus_or_penalty
```

The first two terms are differences of bounded potentials. An outward/inward
oscillation that returns to the same `(s, |e|)` earns zero shaping. Over one
lap, progress contributes at most `w_p`; moving from the worst permitted edge
to the centre contributes at most `w_c`. Thus a shortcut that ends in a crash
can bank no more than

```text
MAX_SHORTCUT_SHAPING = w_p + w_c*h
abs(CRASH_PENALTY) > MAX_SHORTCUT_SHAPING
```

With the trained configuration, the bankable maximum is
`10 + 1.5*1.5 = 12.25`, while the crash penalty magnitude is `25`, so
`25 > 12.25`. The strict inequality is a unit test, not just a design
assertion.

## Termination and time limits

Collision or leaving the road produces `terminated=True`; its future value is
zero. Reaching the step limit produces `truncated=True`; it is not failure and
bootstraps the value of the last valid observation. A successful lap is also a
terminal task state.

Generalized advantage estimation uses distinct masks:

```text
delta_t = r_t + gamma (1 - terminated_t) V(s_(t+1)) - V(s_t)
A_t = delta_t + gamma lambda
      (1 - terminated_t)(1 - truncated_t) A_(t+1)
```

The second mask prevents a trace from crossing into the reset episode while
preserving the truncated state's bootstrap in `delta_t`.

PPO consumes this contract but is isolated in the
[`learning` package](../learning/README.md).
