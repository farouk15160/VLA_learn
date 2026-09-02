from __future__ import annotations

import numpy as np

from ros2_rl_car.learning.ppo import compute_gae


def test_terminated_transition_does_not_bootstrap() -> None:
    advantages, returns = compute_gae(
        rewards=np.array([2.0], dtype=np.float32),
        values=np.array([10.0], dtype=np.float32),
        next_values=np.array([50.0], dtype=np.float32),
        terminated=np.array([True]),
        truncated=np.array([False]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert np.isclose(returns[0], 2.0)
    assert np.isclose(advantages[0], -8.0)


def test_truncated_transition_bootstraps_last_value() -> None:
    _, returns = compute_gae(
        rewards=np.array([2.0], dtype=np.float32),
        values=np.array([10.0], dtype=np.float32),
        next_values=np.array([50.0], dtype=np.float32),
        terminated=np.array([False]),
        truncated=np.array([True]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert np.isclose(returns[0], 2.0 + (0.99 * 50.0))


def test_truncation_cuts_gae_trace_between_episodes() -> None:
    advantages, _ = compute_gae(
        rewards=np.array([1.0, 100.0], dtype=np.float32),
        values=np.zeros(2, dtype=np.float32),
        next_values=np.zeros(2, dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([True, False]),
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert np.isclose(advantages[0], 1.0)
