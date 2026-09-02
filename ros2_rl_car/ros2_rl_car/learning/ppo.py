"""Learning primitives: discrete PPO and boundary-correct GAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE while bootstrapping truncations but cutting episode traces."""

    arrays = [np.asarray(item) for item in (rewards, values, next_values, terminated, truncated)]
    if not arrays or any(item.ndim != 1 for item in arrays):
        raise ValueError("GAE inputs must be one-dimensional")
    if len({len(item) for item in arrays}) != 1:
        raise ValueError("GAE inputs must have equal lengths")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    rewards_f, values_f, next_values_f = (item.astype(np.float64, copy=False) for item in arrays[:3])
    terminated_b, truncated_b = (item.astype(bool, copy=False) for item in arrays[3:])
    if np.any(terminated_b & truncated_b):
        raise ValueError("a transition cannot be both terminated and truncated")
    deltas = rewards_f + gamma * next_values_f * (~terminated_b) - values_f
    advantages = np.zeros_like(deltas)
    accumulator = 0.0
    for index in range(len(deltas) - 1, -1, -1):
        trace_continues = not (terminated_b[index] or truncated_b[index])
        accumulator = deltas[index] + gamma * gae_lambda * trace_continues * accumulator
        advantages[index] = accumulator
    output_dtype = np.result_type(np.asarray(rewards).dtype, np.float32)
    advantages = advantages.astype(output_dtype)
    return advantages, (advantages + values_f).astype(output_dtype)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    hidden_size: int = 128

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.clip_coefficient <= 0.0 or self.entropy_coefficient < 0.0:
            raise ValueError("clip coefficient must be positive and entropy coefficient non-negative")
        if self.value_coefficient < 0.0 or self.max_gradient_norm <= 0.0:
            raise ValueError("value coefficient must be non-negative and gradient norm positive")
        if self.update_epochs < 1 or self.minibatch_size < 1 or self.hidden_size < 1:
            raise ValueError("epoch, minibatch, and hidden sizes must be positive")


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    gradient_norm: float
    approximate_kl: float
    clip_fraction: float


try:  # PyTorch intentionally lives in the project venv, not the system Python.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised by non-training environments
    torch = None
    nn = None


if nn is not None:
    class ActorCritic(nn.Module):
        """Shared-feature MLP with categorical policy and scalar value heads."""

        def __init__(self, observation_size: int, action_count: int, hidden_size: int = 128) -> None:
            super().__init__()
            if observation_size < 1 or action_count < 2:
                raise ValueError("invalid observation or action dimensions")
            self.backbone = nn.Sequential(
                nn.Linear(observation_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden_size, action_count)
            self.value_head = nn.Linear(hidden_size, 1)
            self.apply(self._initialize)

        @staticmethod
        def _initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2.0))
                nn.init.zeros_(module.bias)

        def forward(self, observations: Any) -> tuple[Any, Any]:
            features = self.backbone(observations)
            return self.policy_head(features), self.value_head(features).squeeze(-1)

        def action_and_value(self, observations: Any, action: Any | None = None) -> tuple[Any, Any, Any, Any]:
            logits, value = self(observations)
            distribution = torch.distributions.Categorical(logits=logits)
            selected = distribution.sample() if action is None else action
            return selected, distribution.log_prob(selected), distribution.entropy(), value
else:
    class ActorCritic:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is required; activate the project's torch virtualenv")


def parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


def ppo_update(
    model: Any,
    optimizer: Any,
    observations: Any,
    actions: Any,
    old_log_probabilities: Any,
    advantages: Any,
    returns: Any,
    config: PPOConfig,
    *,
    generator: Any | None = None,
) -> PPOUpdateMetrics:
    """Run PPO minibatch epochs and return display-ready mean diagnostics."""

    if torch is None:
        raise RuntimeError("PyTorch is required; activate the project's torch virtualenv")
    sample_count = int(observations.shape[0])
    if sample_count < 1:
        raise ValueError("a PPO update requires at least one sample")
    tensors = (actions, old_log_probabilities, advantages, returns)
    if any(int(tensor.shape[0]) != sample_count for tensor in tensors):
        raise ValueError("all PPO batch tensors must have the same leading dimension")
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    totals = np.zeros(6, dtype=np.float64)
    minibatches = 0
    for _ in range(config.update_epochs):
        permutation = torch.randperm(sample_count, device=observations.device, generator=generator)
        for start in range(0, sample_count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            _, new_log_probability, entropy, value = model.action_and_value(observations[indices], actions[indices])
            log_ratio = new_log_probability - old_log_probabilities[indices]
            ratio = log_ratio.exp()
            batch_advantages = normalized_advantages[indices]
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_coefficient, 1.0 + config.clip_coefficient
            ) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (value - returns[indices]).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_coefficient).float().mean()
            totals += (
                float(policy_loss.detach()),
                float(value_loss.detach()),
                float(entropy_mean.detach()),
                float(gradient_norm.detach()),
                float(approximate_kl.detach()),
                float(clip_fraction.detach()),
            )
            minibatches += 1
    means = totals / minibatches
    return PPOUpdateMetrics(*map(float, means))
