"""Focused regression tests for the grid-delivery learning example."""

import builtins
import runpy

import numpy as np
import pytest

from learning_examples.grid_delivery import grid_delivery_robot as grid


def test_environment_imports_without_optional_torch_dependency(monkeypatch):
    real_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    module = runpy.run_path(grid.__file__, run_name="grid_without_torch")

    environment = module["GridDeliveryEnv"](n=16, seed=0)
    assert environment.reset(goal=(9, 9)).shape == (module["OBS_DIM"],)
    with pytest.raises(ModuleNotFoundError, match="PyTorch is required"):
        module["Policy"]()


def test_observation_matches_declared_dimension():
    env = grid.GridDeliveryEnv(seed=0)

    observation = env.reset(goal=(9, 9))

    assert observation.shape == (grid.OBS_DIM,)


def test_start_cell_is_never_an_obstacle():
    for seed in range(20):
        env = grid.GridDeliveryEnv(seed=seed)

        assert not env.wall[grid.START]
        assert not env.haz[grid.START]


def test_perpendicular_slips_are_orthogonal():
    for action in range(4):
        for slip in ((action + 1) % 4, (action - 1) % 4):
            assert int(np.dot(grid.DIRS[action], grid.DIRS[slip])) == 0


def test_truncation_is_not_termination():
    env = grid.GridDeliveryEnv(n=32, n_walls=0, n_hazards=0, seed=1)
    env.reset(start=(16, 16), goal=(0, 0), max_steps=1)

    _, _, terminated, truncated = env.step(2)

    assert truncated
    assert not terminated
    assert env.done_reason == "timeout"


def test_goal_rejects_walls_and_accepts_free_cells():
    env = grid.GridDeliveryEnv(seed=2)
    wall = tuple(int(value) for value in np.argwhere(env.wall)[0])

    assert env.set_goal(wall) is False
    assert env.set_goal(env.random_free_cell()) is True


def test_gae_with_unit_lambda_matches_discounted_returns():
    rewards = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    values = np.zeros(4, dtype=np.float32)

    advantages, _ = grid.gae(
        rewards,
        values,
        last_val=0.0,
        gamma=0.9,
        lam=1.0,
    )

    assert np.allclose(advantages, grid.returns_to_go(rewards, 0.9), atol=1e-5)
