from __future__ import annotations

import pytest

from ros2_rl_car.evaluation.metrics import EvaluationAccumulator


def test_evaluator_refuses_zero_sensor_frames() -> None:
    accumulator = EvaluationAccumulator()
    with pytest.raises(RuntimeError, match="zero sensor frames"):
        accumulator.summary()
