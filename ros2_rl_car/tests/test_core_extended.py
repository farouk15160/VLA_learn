"""Extended tests for the dependency-light training and evaluation core."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from ros2_rl_car.core.constants import LEFT, RIGHT, STRAIGHT
from ros2_rl_car.core.control import (
    action_from_angular_z,
    angular_z_from_action,
    command_from_action,
)
from ros2_rl_car.evaluation.controllers import ConstantController, PurePursuitController, wrap_angle
from ros2_rl_car.evaluation.metrics import EpisodeEvaluation, EvaluationAccumulator
from ros2_rl_car.learning.ppo import ActorCritic, PPOConfig, parameter_count, ppo_update
from ros2_rl_car.dashboard.telemetry import EpisodeTelemetry, TelemetryStore, UpdateTelemetry
from ros2_rl_car.core.track import ParametricTrack, signed_curvature, wrapped_progress_delta
from ros2_rl_car.learning.trainer import TrainingConfig


def test_actor_critic_forward_action_and_ppo_update_have_finite_diagnostics() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    model = ActorCritic(observation_size=7, action_count=3, hidden_size=16)
    observations = torch.randn(12, 7)

    logits, values = model(observations)
    actions, old_log_probabilities, entropy, sampled_values = model.action_and_value(observations)

    assert logits.shape == (12, 3)
    assert values.shape == actions.shape == old_log_probabilities.shape == entropy.shape == (12,)
    assert torch.equal(values, sampled_values)
    assert torch.all((0 <= actions) & (actions < 3))
    assert parameter_count(model) > 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = ppo_update(
        model,
        optimizer,
        observations,
        actions,
        old_log_probabilities.detach(),
        torch.linspace(-1.0, 1.0, 12),
        torch.linspace(0.5, -0.5, 12),
        PPOConfig(update_epochs=2, minibatch_size=5, hidden_size=16),
        generator=torch.Generator().manual_seed(3),
    )

    diagnostics = np.asarray(
        [
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.gradient_norm,
            metrics.approximate_kl,
            metrics.clip_fraction,
        ]
    )
    assert np.all(np.isfinite(diagnostics))
    assert metrics.value_loss >= 0.0
    assert metrics.entropy > 0.0
    assert metrics.gradient_norm >= 0.0
    assert 0.0 <= metrics.clip_fraction <= 1.0


def test_actor_critic_and_ppo_update_reject_invalid_shapes() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="invalid observation"):
        ActorCritic(0, 3)
    with pytest.raises(ValueError, match="invalid observation"):
        ActorCritic(3, 1)

    model = ActorCritic(3, 3, hidden_size=8)
    optimizer = torch.optim.Adam(model.parameters())
    empty_observations = torch.empty(0, 3)
    empty = torch.empty(0)
    with pytest.raises(ValueError, match="at least one sample"):
        ppo_update(model, optimizer, empty_observations, empty.long(), empty, empty, empty, PPOConfig())

    observations = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="same leading dimension"):
        ppo_update(
            model,
            optimizer,
            observations,
            torch.zeros(1, dtype=torch.long),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            PPOConfig(),
        )


def test_pure_pursuit_steers_in_both_directions_on_alternating_track() -> None:
    track = ParametricTrack.generate(samples=600)
    controller = PurePursuitController(track, lookahead_distance=0.25)
    commands: list[float] = []
    actions: set[int] = set()

    # Evaluate poses aligned with each segment. This verifies that the reference
    # controller actually exercises both ROS yaw-rate signs on this track.
    for index in range(0, len(track.points), 3):
        vector = track.segment_vectors[index]
        yaw = math.atan2(vector[1], vector[0])
        x, y = track.points[index] + 0.1 * vector
        command = controller.angular_z(float(x), float(y), yaw)
        commands.append(command)
        actions.add(controller.select_action(float(x), float(y), yaw))

    assert min(commands) < -0.05
    assert max(commands) > 0.05
    assert {LEFT, RIGHT} <= actions


def test_controller_helpers_and_validation() -> None:
    track = ParametricTrack.generate(samples=64)
    assert ConstantController(RIGHT).select_action(np.ones(4)) == RIGHT
    assert wrap_angle(math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi)
    for field in ("lookahead_distance", "wheelbase", "max_angular_z"):
        arguments = {field: 0.0}
        with pytest.raises(ValueError, match="must be positive"):
            PurePursuitController(track, **arguments)


def test_evaluation_accumulator_reports_abs_errors_and_episode_rates() -> None:
    accumulator = EvaluationAccumulator()
    accumulator.record_sensor_frame(-0.25)
    accumulator.record_episode([-0.5, 1.0], laps_completed=2, collision=False)
    accumulator.record_episode([0.75], collision=True, success=False, sensor_frames=4)

    summary = accumulator.summary()

    assert summary.episodes == 2
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.mean_cross_track_error == pytest.approx(0.625)
    assert summary.max_cross_track_error == pytest.approx(1.0)
    assert summary.laps_completed == 2
    assert summary.collision_rate == pytest.approx(0.5)
    assert summary.sensor_frames == 7


@pytest.mark.parametrize(
    "arguments",
    [
        {"sensor_frames": -1, "cross_track_errors": ()},
        {"sensor_frames": 1, "cross_track_errors": (float("nan"),)},
        {"sensor_frames": 1, "cross_track_errors": (), "laps_completed": -1},
    ],
)
def test_episode_evaluation_validates_counts_and_finite_errors(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EpisodeEvaluation(**arguments)  # type: ignore[arg-type]


def test_evaluation_accumulator_rejects_nonfinite_live_and_episode_errors() -> None:
    accumulator = EvaluationAccumulator()
    with pytest.raises(ValueError, match="finite"):
        accumulator.record_sensor_frame(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        accumulator.record_episode([float("nan")])


def test_track_csv_round_trip_and_unit_named_schema(tmp_path) -> None:
    track = ParametricTrack.generate(samples=40)
    standard_csv = tmp_path / "standard.csv"
    track.save_csv(standard_csv)
    loaded = ParametricTrack.load_csv(standard_csv)
    assert np.allclose(loaded.points, track.points)
    assert loaded.points.flags.writeable is False

    unit_csv = tmp_path / "units.csv"
    unit_csv.write_text("progress_m,x_m,y_m\n0,1,2\n1,3,4\n2,5,6\n", encoding="utf-8")
    assert np.array_equal(ParametricTrack.from_csv(unit_csv).points, [[1, 2], [3, 4], [5, 6]])

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("latitude,longitude\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="x/y or x_m/y_m"):
        ParametricTrack.load_csv(bad_csv)


def test_track_wrapped_progress_and_projection_boundaries() -> None:
    length = 10.0
    assert wrapped_progress_delta(0.2, 9.8, length) == pytest.approx(0.4)
    assert wrapped_progress_delta(9.8, 0.2, length) == pytest.approx(-0.4)
    with pytest.raises(ValueError, match="positive"):
        wrapped_progress_delta(1.0, 0.0, 0.0)

    track = ParametricTrack.generate(samples=32)
    assert track.point_at(track.length + 0.5) == pytest.approx(track.point_at(0.5))
    with pytest.raises(ValueError, match="finite"):
        track.project((float("nan"), 0.0))
    with pytest.raises(ValueError, match="finite"):
        track.project((0.0, 1.0, 2.0))
    curvature = signed_curvature(track.points)
    assert curvature.shape == (32,)
    assert np.all(np.isfinite(curvature))


@pytest.mark.parametrize("action", [True, 1.5, "left", None])
def test_control_rejects_non_integer_actions(action: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        angular_z_from_action(action)  # type: ignore[arg-type]


@pytest.mark.parametrize("action", [-1, 3, 100])
def test_control_rejects_out_of_range_actions(action: int) -> None:
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        angular_z_from_action(action)


@pytest.mark.parametrize("angular_z", [float("nan"), float("inf"), -float("inf")])
def test_control_rejects_nonfinite_yaw_rate(angular_z: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        action_from_angular_z(angular_z)


@pytest.mark.parametrize("speed", [-0.01, float("nan"), float("inf")])
def test_control_rejects_invalid_linear_speed(speed: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        command_from_action(STRAIGHT, speed)


def test_telemetry_snapshots_are_isolated_and_history_is_bounded() -> None:
    store = TelemetryStore(history=2)
    hyperparameters = {"lr": 3e-4}
    trajectory = np.array([[1.0, 2.0]])
    rays = np.array([[3.0, 4.0]])
    store.set_hyperparameters(hyperparameters)
    store.set_live(
        trajectory=trajectory,
        ray_endpoints=rays,
        car_pose=(1, 2, 0.5),
        action_probabilities=np.array([0.2, 0.3, 0.5]),
        value_estimate=1.25,
    )
    hyperparameters["lr"] = 9.0
    trajectory[0, 0] = 99.0
    rays[0, 0] = 99.0

    for index in range(3):
        store.add_episode(EpisodeTelemetry(float(index), index, False, str(index)))
        store.add_update(UpdateTelemetry(policy_loss=float(index)))
        store.event(str(index))

    first = store.snapshot()
    assert first.hyperparameters == {"lr": 3e-4}
    assert first.trajectory.tolist() == [[1.0, 2.0]]
    assert first.ray_endpoints.tolist() == [[3.0, 4.0]]
    assert first.car_pose == (1.0, 2.0, 0.5)
    assert first.action_probabilities == pytest.approx((0.2, 0.3, 0.5))
    assert [episode.reason for episode in first.episodes] == ["1", "2"]
    assert [update.policy_loss for update in first.updates] == [1.0, 2.0]
    assert len(first.events) == 2

    first.trajectory[0, 0] = -5.0
    first.ray_endpoints[0, 0] = -5.0
    first.hyperparameters["lr"] = -5.0
    second = store.snapshot()
    assert second.trajectory[0, 0] == 1.0
    assert second.ray_endpoints[0, 0] == 3.0
    assert second.hyperparameters["lr"] == 3e-4


def test_telemetry_flags_validate_names_and_frozen_records() -> None:
    store = TelemetryStore()
    store.set_flags(paused=True, greedy=True, finished=True)
    state = store.snapshot()
    assert state.paused and state.greedy and state.finished
    with pytest.raises(ValueError, match="unknown telemetry flags"):
        store.set_flags(recording=True)
    with pytest.raises(FrozenInstanceError):
        state.paused = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PPOConfig(learning_rate=0.0),
        lambda: PPOConfig(gamma=1.1),
        lambda: PPOConfig(minibatch_size=0),
        lambda: TrainingConfig(total_steps=0),
        lambda: TrainingConfig(torch_threads=0),
    ],
)
def test_training_configs_reject_invalid_boundaries(factory) -> None:
    with pytest.raises(ValueError):
        factory()
