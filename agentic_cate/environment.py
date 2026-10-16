"""A budgeted factual-outcome acquisition environment.

Action ``2 * i + a`` acquires pool unit ``i`` under treatment ``a``.  The STOP
action is ``2 * n_pool``.  A unit can be acquired only once, so the environment
never returns both potential outcomes of the same unit to the policy.
"""

from dataclasses import dataclass

import numpy as np

from .data import CausalTask


@dataclass(frozen=True)
class Observation:
    """The complete legal policy interface; contains no hidden effect labels."""

    x_context: np.ndarray
    a_context: np.ndarray
    y_context: np.ndarray
    pool_x: np.ndarray
    query_x: np.ndarray
    available: np.ndarray
    remaining_budget: int
    initial_budget: int
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float


def _copy_observation(obs: Observation) -> Observation:
    return Observation(
        x_context=obs.x_context.copy(),
        a_context=obs.a_context.copy(),
        y_context=obs.y_context.copy(),
        pool_x=obs.pool_x.copy(),
        query_x=obs.query_x.copy(),
        available=obs.available.copy(),
        remaining_budget=obs.remaining_budget,
        initial_budget=obs.initial_budget,
        x_mean=obs.x_mean.copy(),
        x_scale=obs.x_scale.copy(),
        y_mean=obs.y_mean,
        y_scale=obs.y_scale,
    )


class CausalAcquisitionEnv:
    """Keep oracle outcomes private and expose copies of factual observations."""

    def __init__(self, task: CausalTask, budget: int = 2):
        if isinstance(budget, (bool, np.bool_)) or not isinstance(
            budget, (int, np.integer)
        ):
            raise ValueError("budget must be a nonnegative integer")
        if budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        self._x_context = task.initial_x.copy()
        self._a_context = task.initial_a.copy()
        self._y_context = task.initial_y.copy()
        self._pool_x = task.pool_x.copy()
        self._query_x = task.query_x.copy()
        self.__pool_y0 = task._pool_y0.copy()
        self.__pool_y1 = task._pool_y1.copy()
        self.__query_tau = task._query_tau.copy()
        self._available = np.ones(len(self._pool_x), dtype=bool)
        self._initial_budget = int(budget)
        # Fixed, factual-only normalization.  Acquisitions never change these
        # statistics, so alternative trajectories share a common prediction scale.
        self._x_mean = self._x_context.mean(axis=0).astype(np.float32)
        x_std = self._x_context.std(axis=0)
        self._x_scale = np.where(x_std > 1e-6, x_std, 1.0).astype(np.float32)
        self._y_mean = float(self._y_context.mean())
        y_std = self._y_context.std()
        self._y_scale = float(y_std if y_std > 1e-6 else 1.0)
        self._spent = 0
        self._done = budget == 0 or len(self._pool_x) == 0
        self._initial_observation = self.observation()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def acquired_count(self) -> int:
        return self._spent

    @property
    def stop_action(self) -> int:
        return 2 * len(self._pool_x)

    def observation(self) -> Observation:
        return Observation(
            x_context=self._x_context.copy(),
            a_context=self._a_context.copy(),
            y_context=self._y_context.copy(),
            pool_x=self._pool_x.copy(),
            query_x=self._query_x.copy(),
            available=self._available.copy(),
            remaining_budget=self._initial_budget - self._spent,
            initial_budget=self._initial_budget,
            x_mean=self._x_mean.copy(),
            x_scale=self._x_scale.copy(),
            y_mean=self._y_mean,
            y_scale=self._y_scale,
        )

    def initial_observation(self) -> Observation:
        """A detached initial snapshot for a fixed, action-independent baseline."""

        return _copy_observation(self._initial_observation)

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.stop_action + 1, dtype=bool)
        if not self._done:
            mask[:-1] = np.repeat(self._available, 2)
            mask[-1] = True
        return mask

    def step(self, action: int) -> tuple[Observation, bool, dict]:
        if self._done:
            raise RuntimeError("episode is terminal; no further actions are legal")
        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise ValueError("action must be an integer")
        action = int(action)
        if action < 0 or action > self.stop_action:
            raise ValueError("action index is outside the action space")
        if action == self.stop_action:
            self._done = True
            return self.observation(), True, {
                "action_type": "stop",
                "cost": 0,
                "termination": "stop",
            }

        unit, treatment = divmod(action, 2)
        if not self._available[unit]:
            raise ValueError("unit has already been acquired")
        factual_y = (
            self.__pool_y1[unit] if treatment == 1 else self.__pool_y0[unit]
        )
        self._x_context = np.concatenate(
            [self._x_context, self._pool_x[unit : unit + 1]], axis=0
        )
        self._a_context = np.append(self._a_context, treatment)
        self._y_context = np.append(self._y_context, factual_y)
        self._available[unit] = False
        self._spent += 1
        termination = None
        if self._spent >= self._initial_budget:
            self._done = True
            termination = "budget"
        elif not self._available.any():
            self._done = True
            termination = "pool_exhausted"
        return self.observation(), self._done, {
            "action_type": "acquire",
            "unit": unit,
            "treatment": treatment,
            "cost": 1,
            "termination": termination,
        }

    def terminal_truth(self) -> np.ndarray:
        """Return CATE labels only to a terminal evaluator, never to a policy."""

        if not self._done:
            raise RuntimeError("query truth is unavailable before termination")
        return self.__query_tau.copy()
