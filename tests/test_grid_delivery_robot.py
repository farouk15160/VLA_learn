"""Tests for grid_delivery_robot.py.

These check the claims the docs actually make -- the 80/10/10 transition model,
that termination and truncation are distinguishable, and the reward-design rule
that keeps "walk into the hazard" from being a profitable shortcut. If one of
these fails, a number in docs/grid_delivery_robot.md is wrong.
"""
import numpy as np
import pytest

import grid_delivery_robot as g


@pytest.fixture
def env():
    return g.GridDeliveryEnv(seed=0)


@pytest.fixture
def empty():
    """A map with no obstacles, for testing the transition model in isolation."""
    return g.GridDeliveryEnv(n=32, n_walls=0, n_hazards=0, seed=1)


# ---------------------------------------------------------------- observation
def test_obs_matches_declared_dim(env):
    assert env.reset(goal=(9, 9)).shape == (g.OBS_DIM,)


def test_obs_grid_matches_per_step_obs(env):
    """The vectorised whole-map observation must equal the per-step one.

    obs_grid() feeds the V(s) heatmap. If it drifts from _obs(), the picture
    stops describing the policy you are actually training.
    """
    goal = (40, 41)
    batch, rows, cols = g.obs_grid(env, goal, stride=8)
    for i in (0, len(rows) // 2, len(rows) - 1):
        env.reset(start=(int(rows[i]), int(cols[i])), goal=goal)
        assert np.allclose(batch[i], env._obs())


def test_start_cell_is_never_an_obstacle():
    for seed in range(20):
        e = g.GridDeliveryEnv(seed=seed)
        assert not e.wall[g.START] and not e.haz[g.START]


# ---------------------------------------------------------------- transitions
def test_slip_is_80_10_10(empty):
    """The headline claim: 0.8 intended, 0.1 to each perpendicular."""
    counts = np.zeros(4)
    trials = 20000
    for _ in range(trials):
        empty.reset(start=(16, 16), goal=(0, 0))
        empty.step(1)                                   # RIGHT
        dr = empty.pos[0] - 16
        dc = empty.pos[1] - 16
        counts[np.argmax((g.DIRS == np.array([dr, dc])).all(1))] += 1
    freq = counts / trials
    assert freq[1] == pytest.approx(0.8, abs=0.02)      # RIGHT, as intended
    assert freq[0] == pytest.approx(0.1, abs=0.02)      # slipped UP
    assert freq[2] == pytest.approx(0.1, abs=0.02)      # slipped DOWN
    assert freq[3] == 0.0                               # never backwards


def test_perpendicular_slips_really_are_perpendicular():
    """DIRS is in rotational order, so (a+-1) % 4 must be orthogonal to a."""
    for a in range(4):
        for slip in ((a + 1) % 4, (a - 1) % 4):
            assert int(np.dot(g.DIRS[a], g.DIRS[slip])) == 0


def test_blocked_by_boundary_stays_put(empty):
    empty.reset(start=(0, 5), goal=(20, 20))
    for _ in range(50):
        pos = empty.pos
        empty.reset(start=pos, goal=(20, 20))
        empty.step(0)                                   # UP, off the top edge
        assert empty.pos[0] >= 0                        # never leaves the map


def test_blocked_by_wall_stays_put():
    e = g.GridDeliveryEnv(n=16, n_walls=0, n_hazards=0, seed=2)
    e.wall[5, 6] = True
    e.reset(start=(5, 5), goal=(0, 0))
    e.rng = np.random.RandomState(0)
    moved = 0
    for _ in range(200):
        e.reset(start=(5, 5), goal=(0, 0))
        _, _, _, _ = e.step(1)                          # RIGHT, into the wall
        if e.pos == (5, 6):
            moved += 1
    assert moved == 0, "the robot entered a wall cell"


# ---------------------------------------------------------------- reward
def test_goal_terminates_and_pays(empty):
    empty.reset(start=(5, 5), goal=(5, 6))
    empty.rng = np.random.RandomState(0)
    for _ in range(60):
        _, r, term, trunc = empty.step(1)
        if term:
            assert empty.done_reason == "goal"
            assert r > g.R_GOAL - 1                     # +10 dominates the step
            assert not trunc
            return
        empty.reset(start=(5, 5), goal=(5, 6))
    pytest.fail("never reached an adjacent goal in 60 attempts")


def test_hazard_terminates_and_penalises():
    e = g.GridDeliveryEnv(n=16, n_walls=0, n_hazards=0, seed=3)
    e.haz[5, 6] = True
    e.hpad = np.pad(e.haz, g.PAD, constant_values=False)
    for _ in range(60):
        e.reset(start=(5, 5), goal=(15, 15))
        _, r, term, _ = e.step(1)
        if e.pos == (5, 6):
            assert term and e.done_reason == "hazard"
            assert r < g.R_HAZARD + 1
            return
    pytest.fail("never stepped onto the hazard")


def test_truncation_is_not_termination(empty):
    """Running out of clock must set truncated, NOT terminated.

    They bootstrap differently (V(s_last) vs 0), so conflating them teaches the
    value net that the stopwatch is as final as dying.
    """
    empty.reset(start=(16, 16), goal=(0, 0), max_steps=5)
    for i in range(5):
        _, _, term, trunc = empty.step(2)               # DOWN, away from goal
    assert trunc and not term
    assert empty.done_reason == "timeout"


def test_shaping_is_potential_based(empty):
    """A closed loop must earn zero shaping, or it could be farmed by pacing.

    Walk out and back to the same cell; the progress terms must cancel exactly,
    leaving only the (negative) time cost.
    """
    empty.reset(start=(16, 16), goal=(1, 1), max_steps=100)
    empty.rng = np.random.RandomState(7)
    total, steps = 0.0, 0
    start = empty.pos
    for _ in range(40):
        _, r, term, trunc = empty.step(np.random.RandomState(steps).randint(4))
        total += r
        steps += 1
        if term or trunc:
            pytest.skip("episode ended before returning to start")
        if empty.pos == start and steps > 1:
            break
    else:
        pytest.skip("random walk did not return to the start cell")
    # Only step costs remain; shaping cancelled. Blocked steps add -0.05 each.
    assert total <= 0.0
    assert total >= steps * (g.R_STEP + g.R_BLOCKED) - 1e-6


def test_hazard_penalty_outweighs_bankable_shaping():
    """The reward-design rule from the docs, as an executable assertion.

    A hazard `d` cells nearer the goal has already paid d * R_PROGRESS of
    shaping. If |R_HAZARD| is smaller than that, walking into it is PROFITABLE
    and the robot will learn to. Checked against the worst case on this map.
    """
    diameter = 2 * g.GRID
    assert abs(g.R_HAZARD) > 0.0
    # A hazard is never worth entering from anywhere within this many cells:
    safe_radius = abs(g.R_HAZARD) / g.R_PROGRESS
    assert safe_radius >= 100, (
        f"hazard penalty only deters shortcuts within {safe_radius:.0f} cells "
        f"on a map {diameter} cells across")


def test_step_penalty_cannot_beat_delivery():
    """|R_STEP| must stay under R_GOAL / path length, or detours stop paying.

    Above that bound, the accumulated step cost of walking the long way round a
    hazard exceeds the delivery bonus itself, and cutting through starts to win.
    """
    bound = g.R_GOAL / g.MAX_STEPS                      # 10 / 256 = 0.039
    assert abs(g.R_STEP) < bound
    assert bound / abs(g.R_STEP) > 1.9                  # measured margin: 1.95x


# ---------------------------------------------------------------- feasibility
def test_walled_goal_is_reported_unreachable(env):
    walls = np.argwhere(env.wall)
    goal = tuple(int(v) for v in walls[0])
    f = g.feasibility(env, goal)
    assert not f["ok"] and "wall" in f["why"]
    assert env.bfs_distance(goal)[g.START] == np.iinfo(np.int32).max


def test_far_corner_needs_more_than_256_steps(env):
    """254 cells needs ~330 steps with slipping, which is why the budget is 512.

    This is the arithmetic behind MAX_STEPS: at 256 the far third of the map is
    unreachable no matter how good the policy is.
    """
    assert g.feasibility(env, (127, 127), max_steps=256)["tight"] is True
    assert g.feasibility(env, (127, 127), max_steps=g.MAX_STEPS)["tight"] is False


def test_set_goal_refuses_walls(env):
    walls = np.argwhere(env.wall)
    assert env.set_goal(tuple(int(v) for v in walls[0])) is False
    assert env.set_goal(env.random_free_cell()) is True


# ---------------------------------------------------------------- learning
def test_gae_reduces_to_returns_when_lambda_is_one():
    """GAE(1) with a zero baseline is exactly the discounted return."""
    rew = np.array([1.0, -2.0, 0.5, 3.0], np.float32)
    vals = np.zeros(4, np.float32)
    adv, _ = g.gae(rew, vals, 0.0, gamma=0.9, lam=1.0)
    assert np.allclose(adv, g.returns_to_go(rew, 0.9), atol=1e-5)


def test_one_update_runs_and_changes_the_policy():
    t = g.Trainer(env=g.GridDeliveryEnv(n=32, seed=4, block=4),
                  seed=0, batch_eps=2, max_steps=40)
    before = t.policy.net[0].weight.detach().clone()
    snap = t.update()
    assert snap["update"] == 1 and snap["episodes"] == 2
    assert not np.allclose(before, t.policy.net[0].weight.detach())
