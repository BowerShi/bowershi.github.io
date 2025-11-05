"""Limited-memory BFGS optimizer for FEniCS functions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, Tuple

import numpy as np
from collections import deque
from dolfin import Function

ObjectiveGradFn = Callable[[Function], Tuple[float, Function]]


@dataclass
class LBFGSConfig:
    history: int = 10
    max_iters: int = 200
    c1: float = 1e-4
    step_shrink: float = 0.5
    min_step: float = 1e-8
    tol: float = 1e-6


@dataclass
class LBFGSState:
    iteration: int = 0
    objective: float = 0.0
    gradient_norm: float = 0.0
    converged: bool = False


class LBFGSOptimizer:
    """Simple implementation of the L-BFGS method."""

    def __init__(self, config: Optional[LBFGSConfig] = None):
        self.config = config or LBFGSConfig()
        self.s_history: Deque[np.ndarray] = deque(maxlen=self.config.history)
        self.y_history: Deque[np.ndarray] = deque(maxlen=self.config.history)
        self.rho_history: Deque[float] = deque(maxlen=self.config.history)

    def _two_loop(self, grad: np.ndarray) -> np.ndarray:
        q = grad.copy()
        alpha: list[float] = []
        for s, y, rho in zip(reversed(self.s_history), reversed(self.y_history), reversed(self.rho_history)):
            a = rho * np.dot(s, q)
            alpha.append(a)
            q = q - a * y
        if self.y_history:
            y_last = self.y_history[-1]
            s_last = self.s_history[-1]
            gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y, rho, a) in zip(self.s_history, self.y_history, self.rho_history, reversed(alpha)):
            beta = rho * np.dot(y, r)
            r = r + s * (a - beta)
        return -r

    def step(
        self,
        theta: Function,
        objective: float,
        gradient: Function,
        obj_grad: ObjectiveGradFn,
    ) -> Tuple[Function, float, Function, LBFGSState]:
        config = self.config
        theta_vec = theta.vector().get_local()
        grad_vec = gradient.vector().get_local()
        grad_norm = np.linalg.norm(grad_vec) / np.sqrt(theta.function_space().dim())
        state = LBFGSState(iteration=len(self.s_history), objective=objective, gradient_norm=grad_norm)
        if grad_norm < config.tol:
            state.converged = True
            return theta, objective, gradient, state
        direction = self._two_loop(grad_vec)
        if not np.isfinite(direction).all():
            direction = -grad_vec
        step_len = 1.0
        current_obj = objective
        for _ in range(20):
            trial_values = theta_vec + step_len * direction
            trial_theta = Function(theta.function_space())
            trial_theta.vector().set_local(trial_values)
            trial_theta.vector().apply("insert")
            trial_obj, trial_grad = obj_grad(trial_theta)
            lhs = trial_obj
            rhs = current_obj + config.c1 * step_len * np.dot(grad_vec, direction)
            if lhs <= rhs:
                s = trial_values - theta_vec
                y = trial_grad.vector().get_local() - grad_vec
                ys = np.dot(y, s)
                if ys > 1e-10:
                    rho = 1.0 / ys
                    self.s_history.append(s)
                    self.y_history.append(y)
                    self.rho_history.append(rho)
                theta.assign(trial_theta)
                state.iteration += 1
                return theta, trial_obj, trial_grad, state
            step_len *= config.step_shrink
            if step_len < config.min_step:
                break
        # fall back to gradient descent step
        theta.vector().set_local(theta_vec - config.step_shrink * grad_vec)
        theta.vector().apply("insert")
        new_obj, new_grad = obj_grad(theta)
        state.iteration += 1
        return theta, new_obj, new_grad, state
