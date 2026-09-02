# Learning package

`ppo.py` contains the ROS-free actor/critic network, generalized advantage
estimation, rollout storage, and clipped PPO update. `trainer.py` collects
Gazebo transitions, updates the model, emits telemetry, and writes atomic
checkpoints.

The actor and critic share a two-layer CPU MLP. For probability ratio
`rho = pi_new(a|s) / pi_old(a|s)`, PPO minimizes

```text
L_policy = -mean(min(rho*A, clip(rho, 1-epsilon, 1+epsilon)*A))
L_total = L_policy + c_v*mean((V-R)^2) - c_H*entropy(pi)
```

Advantages are normalized per rollout and gradients are clipped. Truncated
episodes bootstrap `V(s_last)`; terminated episodes do not. The trainer records
policy/value loss, entropy, gradient norm, approximate KL, and clip fraction on
every update. Greedy-watch mode selects `argmax` only for display and does not
change training weights.

Defaults live in [`../../config/default.json`](../../config/default.json).
