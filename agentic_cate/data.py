"""Fixed-mechanism synthetic tasks for active causal data acquisition.

Only factual outcomes belong in an agent observation.  Fields prefixed with
``_`` are simulation oracles, reserved for the environment and final evaluator.
Each unit has one shared additive noise draw across its two potential outcomes;
therefore the analytic CATE is also its individual treatment effect in this MVP.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedSCM:
    """One nonlinear response mechanism, sampled once and held fixed per task."""

    control_weights: np.ndarray
    effect_weights: np.ndarray
    control_intercept: float
    effect_intercept: float

    def control_mean(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        interaction = x[:, 0] * x[:, 1] if x.shape[1] > 1 else x[:, 0] ** 2
        return (
            self.control_intercept
            + x @ self.control_weights
            + 0.5 * np.sin(x[:, 0])
            + 0.25 * interaction
        )

    def cate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return (
            self.effect_intercept
            + x @ self.effect_weights
            + 0.7 * np.tanh(x[:, 0])
        )


@dataclass(frozen=True)
class CausalTask:
    """An episode's public data and private simulation truth.

    Do not pass this object to a policy.  Pass ``env.observation()`` instead.
    Query truth and unobserved pool outcomes must never be policy features.
    """

    initial_x: np.ndarray
    initial_a: np.ndarray
    initial_y: np.ndarray
    pool_x: np.ndarray
    query_x: np.ndarray
    _pool_y0: np.ndarray
    _pool_y1: np.ndarray
    _query_tau: np.ndarray
    _initial_y0: np.ndarray
    _initial_y1: np.ndarray
    _scm: FixedSCM
    seed: int


def _positive_integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def make_task(
    seed: int,
    n_initial: int = 16,
    n_pool: int = 8,
    n_query: int = 8,
    n_features: int = 3,
    noise_std: float = 0.1,
) -> CausalTask:
    """Create a reproducible task with a fixed SCM and analytic query CATE.

    Initial treatments are randomized and balanced (at least two per arm).
    Acquisition actions subsequently choose a unit and its assigned treatment.
    All covariates are pre-treatment, and the noise is independent of treatment.
    Use disjoint seeds for training, development and held-out evaluation; stronger
    mechanism-family shifts should be evaluated separately from this MVP.
    """

    _positive_integer("n_initial", n_initial, minimum=4)
    _positive_integer("n_pool", n_pool)
    _positive_integer("n_query", n_query)
    _positive_integer("n_features", n_features)
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")

    rng = np.random.default_rng(seed)
    scm = FixedSCM(
        control_weights=rng.normal(0.0, 0.6, size=n_features),
        effect_weights=rng.normal(0.0, 0.3, size=n_features),
        control_intercept=float(rng.normal(0.0, 0.3)),
        effect_intercept=float(rng.normal(0.0, 0.5)),
    )
    all_x = rng.normal(size=(n_initial + n_pool + n_query, n_features)).astype(
        np.float32
    )
    noise = rng.normal(0.0, noise_std, size=len(all_x))
    y0 = scm.control_mean(all_x) + noise
    y1 = y0 + scm.cate(all_x)
    initial_a = np.arange(n_initial, dtype=np.int64) % 2
    rng.shuffle(initial_a)
    initial_y = np.where(initial_a == 1, y1[:n_initial], y0[:n_initial])
    pool_slice = slice(n_initial, n_initial + n_pool)
    query_slice = slice(n_initial + n_pool, None)

    return CausalTask(
        initial_x=all_x[:n_initial].copy(),
        initial_a=initial_a,
        initial_y=initial_y.astype(np.float32),
        pool_x=all_x[pool_slice].copy(),
        query_x=all_x[query_slice].copy(),
        _pool_y0=y0[pool_slice].astype(np.float32),
        _pool_y1=y1[pool_slice].astype(np.float32),
        _query_tau=scm.cate(all_x[query_slice]).astype(np.float32),
        _initial_y0=y0[:n_initial].astype(np.float32),
        _initial_y1=y1[:n_initial].astype(np.float32),
        _scm=scm,
        seed=int(seed),
    )
