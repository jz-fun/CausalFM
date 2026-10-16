"""Contract tests for hidden-label isolation and causal acquisition semantics."""

from dataclasses import fields, replace

import numpy as np
import pytest

from agentic_cate.data import make_task
from agentic_cate.environment import CausalAcquisitionEnv, Observation


def assert_same_observation(left: Observation, right: Observation) -> None:
    for field in fields(Observation):
        lhs, rhs = getattr(left, field.name), getattr(right, field.name)
        if isinstance(lhs, np.ndarray):
            np.testing.assert_array_equal(lhs, rhs)
        else:
            assert lhs == rhs


def test_fixed_scm_and_factual_consistency():
    task = make_task(7)
    np.testing.assert_allclose(
        task._pool_y1 - task._pool_y0,
        task._scm.cate(task.pool_x),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        task.initial_y,
        np.where(task.initial_a == 1, task._initial_y1, task._initial_y0),
    )
    np.testing.assert_allclose(task._query_tau, task._scm.cate(task.query_x))
    np.testing.assert_array_equal(
        task._scm.cate(task.pool_x), task._scm.cate(task.pool_x.copy())
    )
    assert np.bincount(task.initial_a, minlength=2).min() >= 2


@pytest.mark.parametrize("treatment", [0, 1])
def test_acquisition_reveals_only_assigned_factual_outcome(treatment):
    task = make_task(13)
    env = CausalAcquisitionEnv(task, budget=2)
    obs, done, info = env.step(2 * 3 + treatment)
    expected = (task._pool_y0, task._pool_y1)[treatment][3]
    assert obs.y_context[-1] == expected
    assert obs.a_context[-1] == treatment
    np.testing.assert_array_equal(obs.x_context[-1], task.pool_x[3])
    assert not done
    assert set(info) == {"action_type", "unit", "treatment", "cost", "termination"}
    assert set(vars(obs)) == {
        "x_context", "a_context", "y_context", "pool_x", "query_x",
        "available", "remaining_budget", "initial_budget",
        "x_mean", "x_scale", "y_mean", "y_scale",
    }
    assert not env.legal_mask()[6:8].any()
    with pytest.raises(ValueError, match="already"):
        env.step(2 * 3 + (1 - treatment))


def test_hidden_query_label_changes_do_not_change_rollout_observations():
    task = make_task(23)
    permuted = replace(task, _query_tau=task._query_tau[::-1].copy() + 50)
    env_a = CausalAcquisitionEnv(task, budget=2)
    env_b = CausalAcquisitionEnv(permuted, budget=2)
    assert_same_observation(env_a.observation(), env_b.observation())
    for action in [0, 3]:
        obs_a, done_a, info_a = env_a.step(action)
        obs_b, done_b, info_b = env_b.step(action)
        assert_same_observation(obs_a, obs_b)
        assert done_a == done_b
        assert info_a == info_b
    assert not np.array_equal(env_a.terminal_truth(), env_b.terminal_truth())


def test_unacquired_counterfactual_is_not_revealed():
    task = make_task(31)
    altered = replace(task, _pool_y1=task._pool_y1 + 100)
    env_a = CausalAcquisitionEnv(task, budget=2)
    env_b = CausalAcquisitionEnv(altered, budget=2)
    obs_a, _, info_a = env_a.step(0)  # control arm; changed treated outcome hidden
    obs_b, _, info_b = env_b.step(0)
    assert_same_observation(obs_a, obs_b)
    assert info_a == info_b


def test_budget_stop_and_terminal_boundaries():
    task = make_task(5)
    env = CausalAcquisitionEnv(task, budget=1)
    assert env.legal_mask().dtype == np.bool_
    assert env.legal_mask().shape == (2 * len(task.pool_x) + 1,)
    with pytest.raises(RuntimeError, match="before"):
        env.terminal_truth()
    obs, done, info = env.step(1)
    assert done and env.done
    assert env.acquired_count == 1
    assert obs.remaining_budget == 0
    assert info["termination"] == "budget"
    assert not env.legal_mask().any()
    with pytest.raises(RuntimeError, match="terminal"):
        env.step(env.stop_action)
    np.testing.assert_array_equal(env.terminal_truth(), task._query_tau)

    early_stop = CausalAcquisitionEnv(task, budget=3)
    obs, done, info = early_stop.step(early_stop.stop_action)
    assert done and early_stop.acquired_count == 0
    assert obs.remaining_budget == 3 and info["cost"] == 0
    assert info["termination"] == "stop"

    zero = CausalAcquisitionEnv(task, budget=0)
    assert zero.done and not zero.legal_mask().any()
    np.testing.assert_array_equal(zero.terminal_truth(), task._query_tau)


def test_pool_exhaustion_with_larger_budget():
    task = make_task(2, n_pool=1)
    env = CausalAcquisitionEnv(task, budget=3)
    obs, done, info = env.step(0)
    assert done and info["termination"] == "pool_exhausted"
    assert obs.remaining_budget == 2
    assert env.acquired_count == 1


def test_observations_and_initial_snapshot_do_not_alias_environment():
    task = make_task(42)
    env = CausalAcquisitionEnv(task, budget=2)
    expected = env.observation()
    obs = env.observation()
    for field in fields(Observation):
        value = getattr(obs, field.name)
        if isinstance(value, np.ndarray):
            value[...] = 0
    assert_same_observation(expected, env.observation())
    snapshot = env.initial_observation()
    snapshot.y_context[:] = 999
    env.step(0)
    assert_same_observation(expected, env.initial_observation())
    task.pool_x[:] = 999
    assert_same_observation(expected, env.initial_observation())
    env.step(env.stop_action)
    truth = env.terminal_truth()
    truth[:] = 999
    assert not np.all(env.terminal_truth() == 999)


def test_disjoint_seed_reproducibility():
    train = make_task(100)
    held_out = make_task(100_100)
    repeated = make_task(100_100)
    for name in ["initial_x", "initial_a", "initial_y", "pool_x", "query_x", "_query_tau"]:
        np.testing.assert_array_equal(getattr(held_out, name), getattr(repeated, name))
    assert not np.array_equal(train.pool_x, held_out.pool_x)
    assert not np.array_equal(train._scm.effect_weights, held_out._scm.effect_weights)


def test_normalization_uses_initial_factual_support_and_stays_fixed():
    task = make_task(119)
    env = CausalAcquisitionEnv(task, budget=2)
    initial = env.observation()
    np.testing.assert_allclose(initial.x_mean, task.initial_x.mean(axis=0))
    np.testing.assert_allclose(initial.x_scale, task.initial_x.std(axis=0))
    assert initial.y_mean == float(task.initial_y.mean())
    assert initial.y_scale == float(task.initial_y.std())
    after, _, _ = env.step(0)
    for name in ["x_mean", "x_scale", "y_mean", "y_scale"]:
        np.testing.assert_array_equal(getattr(initial, name), getattr(after, name))
    changed_oracles = replace(
        task,
        _pool_y0=task._pool_y0 + 1000,
        _pool_y1=task._pool_y1 + 1000,
        _query_tau=task._query_tau + 1000,
    )
    changed = CausalAcquisitionEnv(changed_oracles, budget=2).observation()
    assert_same_observation(initial, changed)


def test_normalization_guards_constant_initial_data():
    task = make_task(120)
    constant = replace(
        task,
        initial_x=np.zeros_like(task.initial_x),
        initial_y=np.zeros_like(task.initial_y),
    )
    obs = CausalAcquisitionEnv(constant, budget=1).observation()
    assert np.all(np.isfinite(obs.x_scale)) and np.all(obs.x_scale > 0)
    assert np.isfinite(obs.y_scale) and obs.y_scale > 0


@pytest.mark.parametrize("action", [-1, 1000, 0.5, True])
def test_invalid_actions_do_not_mutate_episode(action):
    env = CausalAcquisitionEnv(make_task(8), budget=2)
    initial = env.observation()
    with pytest.raises(ValueError):
        env.step(action)
    assert_same_observation(initial, env.observation())


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        CausalAcquisitionEnv(make_task(8), budget=budget)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_initial": 3}, {"n_pool": 0}, {"n_query": 0}, {"n_features": 0},
     {"noise_std": -1}, {"noise_std": float("nan")}],
)
def test_invalid_task_configuration(kwargs):
    with pytest.raises(ValueError):
        make_task(8, **kwargs)
