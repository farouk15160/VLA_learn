from __future__ import annotations

import math

from ros2_rl_car.core.mdp import RewardConfig, RewardInput, reward_step


def test_potential_shaping_cannot_be_farmed_by_oscillation() -> None:
    config = RewardConfig()
    outward = reward_step(
        RewardInput(progress_delta=0.0, previous_cross_track=0.1, cross_track=0.8),
        config,
    )
    inward = reward_step(
        RewardInput(progress_delta=0.0, previous_cross_track=0.8, cross_track=0.1),
        config,
    )
    shaping_only = outward.reward + inward.reward + (2.0 * config.time_cost)
    assert math.isclose(shaping_only, 0.0, abs_tol=1e-9)


def test_crash_penalty_exceeds_every_bankable_shortcut_shaping_reward() -> None:
    config = RewardConfig()
    assert abs(config.crash_penalty) > config.max_shortcut_shaping
    assert math.isclose(
        config.max_shortcut_shaping,
        config.progress_weight
        + (config.cross_track_potential_weight * config.road_half_width),
    )


def test_leaving_road_terminates_and_applies_penalty() -> None:
    result = reward_step(
        RewardInput(
            progress_delta=0.2,
            previous_cross_track=0.2,
            cross_track=1.7,
            collision=False,
        ),
        RewardConfig(road_half_width=1.5),
    )
    assert result.terminated is True
    assert result.reason == "off_road"
    assert result.reward < 0.0


def test_completed_lap_is_a_success_terminal_not_a_timeout() -> None:
    result = reward_step(
        RewardInput(
            progress_delta=0.1,
            previous_cross_track=0.1,
            cross_track=0.1,
            lap_completed=True,
            step_count=100,
            max_steps=100,
        ),
        RewardConfig(),
    )
    assert result.terminated is True
    assert result.truncated is False
    assert result.reason == "lap_complete"
