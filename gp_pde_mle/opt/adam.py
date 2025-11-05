"""Adam optimizer for coefficient updates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from dolfin import Function


@dataclass
class AdamConfig:
    learning_rate: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_iters: int = 200
    tol: float = 1e-6


class AdamOptimizer:
    """Simple Adam optimizer acting on FEniCS ``Function`` objects."""

    def __init__(self, config: Optional[AdamConfig] = None):
        self.config = config or AdamConfig()
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.iteration = 0

    def step(self, params: Function, grad: Function) -> Function:
        values = params.vector().get_local()
        g = grad.vector().get_local()
        if self.config.weight_decay:
            g = g + self.config.weight_decay * values
        if self.m is None:
            self.m = np.zeros_like(g)
        if self.v is None:
            self.v = np.zeros_like(g)
        self.iteration += 1
        self.m = self.config.beta1 * self.m + (1 - self.config.beta1) * g
        self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * (g * g)
        m_hat = self.m / (1 - self.config.beta1 ** self.iteration)
        v_hat = self.v / (1 - self.config.beta2 ** self.iteration)
        step = self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.epsilon)
        new_values = values - step
        params.vector().set_local(new_values)
        params.vector().apply("insert")
        return params

    def reset(self) -> None:
        self.m = None
        self.v = None
        self.iteration = 0
