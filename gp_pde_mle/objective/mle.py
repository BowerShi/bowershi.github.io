"""Objective and gradient computations for the GP-PDE MLE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
from dolfin import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    assemble,
    div,
    dx,
    grad,
    inner,
)

from ..operators.elliptic import EllipticOperator, project_with_bc
from ..stats.probing import draw_probes


@dataclass
class RegularizationResult:
    value: float
    gradient: Function


def _exp_function(theta: Function) -> Function:
    coeff = Function(theta.function_space())
    coeff.vector()[:] = np.exp(theta.vector().get_local())
    coeff.vector().apply("insert")
    return coeff


def regularizer(
    a: Function,
    reg_type: str,
    alpha: float,
    Va: FunctionSpace,
    bcs: Optional[Iterable[DirichletBC]] = None,
    epsilon: float = 1e-6,
) -> RegularizationResult:
    """Return the regularizer value and gradient."""

    reg_type = reg_type.lower()
    if reg_type == "h1":
        value = 0.5 * alpha * assemble(inner(grad(a), grad(a)) * dx)
        grad_expr = project_with_bc(-alpha * div(grad(a)), Va, bcs)
        return RegularizationResult(value=value, gradient=grad_expr)
    if reg_type == "tv":
        grad_a = grad(a)
        magnitude = (inner(grad_a, grad_a) + Constant(epsilon)) ** 0.5
        value = alpha * assemble(magnitude * dx)
        direction = grad_a / magnitude
        grad_expr = project_with_bc(-alpha * div(direction), Va, bcs)
        return RegularizationResult(value=value, gradient=grad_expr)
    raise ValueError(f"Unknown regularizer type '{reg_type}'")


@dataclass
class ObjectiveConfig:
    alpha: float = 1e-2
    reg_type: str = "h1"
    use_log_parameterization: bool = True
    num_probes: int = 20
    probe_kind: str = "rademacher"
    probe_seed: Optional[int] = 0
    reuse_probes: bool = True


@dataclass
class ObjectiveDiagnostics:
    objectives: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    trace_terms: List[float] = field(default_factory=list)
    logdet_terms: List[float] = field(default_factory=list)


class GaussianMLE:
    """Regularized maximum-likelihood estimation for the GP-PDE model."""

    def __init__(
        self,
        V: FunctionSpace,
        Va: FunctionSpace,
        samples: Sequence[Function],
        operator: EllipticOperator,
        config: ObjectiveConfig,
        bcs: Optional[Iterable[DirichletBC]] = None,
    ) -> None:
        self.V = V
        self.Va = Va
        self.samples = list(samples)
        self.operator = operator
        self.config = config
        self.bcs = list(bcs) if bcs else []
        self.diagnostics = ObjectiveDiagnostics()
        self._probes: List[Function] = []

    def _ensure_probes(self) -> List[Function]:
        if self.config.num_probes <= 0:
            raise ValueError("At least one probe is required")
        if self.config.reuse_probes and self._probes:
            return self._probes
        self._probes = draw_probes(
            self.V,
            self.config.num_probes,
            kind=self.config.probe_kind,
            seed=self.config.probe_seed,
            bcs=self.bcs,
        )
        return self._probes

    def _data_gradient_density(self, a: Function) -> Function:
        accum = Function(self.Va)
        accum.vector()[:] = 0.0
        for u in self.samples:
            w = self.operator.apply_A(u)
            expr = inner(grad(w), grad(u))
            density = project_with_bc(expr, self.Va, self.bcs)
            accum.vector().axpy(1.0 / len(self.samples), density.vector())
        return accum

    def _model_gradient_density(self, a: Function) -> Function:
        probes = self._ensure_probes()
        accum = Function(self.Va)
        accum.vector()[:] = 0.0
        for z in probes:
            y = self.operator.solve_L(z)
            q = self.operator.apply_A(y)
            expr = inner(grad(q), grad(y))
            density = project_with_bc(expr, self.Va, self.bcs)
            accum.vector().axpy(1.0 / len(probes), density.vector())
        return accum

    def _trace_term(self, a: Function) -> float:
        total = 0.0
        for u in self.samples:
            Lu = self.operator.apply_L(u)
            total += assemble(inner(u, Lu) * dx)
        return float(total / len(self.samples))

    def _logdet_term(self, a: Function) -> float:
        probes = self._ensure_probes()
        # Simple log-determinant surrogate: Hutchinson of log spectrum via power iterations
        logdet = 0.0
        for z in probes:
            y = self.operator.solve_L(z)
            logdet += z.vector().inner(y.vector())
        return float(logdet / len(probes))

    def objective_and_gradient(self, theta: Function) -> tuple[float, Function]:
        if self.config.use_log_parameterization:
            a = _exp_function(theta)
        else:
            a = theta
        self.operator.update_coefficient(a)
        data_density = self._data_gradient_density(a)
        model_density = self._model_gradient_density(a)
        reg = regularizer(a, self.config.reg_type, self.config.alpha, self.Va, self.bcs)
        grad_density = Function(self.Va)
        grad_density.vector()[:] = 0.0
        grad_density.vector().axpy(len(self.samples), data_density.vector())
        grad_density.vector().axpy(-len(self.samples), model_density.vector())
        grad_density.vector().axpy(1.0, reg.gradient.vector())
        trace_term = self._trace_term(a)
        logdet_term = self._logdet_term(a)
        objective = 0.5 * len(self.samples) * (trace_term - logdet_term) + reg.value
        self.diagnostics.objectives.append(objective)
        self.diagnostics.grad_norms.append(np.linalg.norm(grad_density.vector().get_local()) / np.sqrt(self.Va.dim()))
        self.diagnostics.trace_terms.append(trace_term)
        self.diagnostics.logdet_terms.append(logdet_term)
        if self.config.use_log_parameterization:
            grad_theta = Function(self.Va)
            grad_theta.vector()[:] = np.exp(theta.vector().get_local()) * grad_density.vector().get_local()
            grad_theta.vector().apply("insert")
            return objective, grad_theta
        return objective, grad_density

    def gradient_only(self, theta: Function) -> Function:
        return self.objective_and_gradient(theta)[1]

    def objective_only(self, theta: Function) -> float:
        return self.objective_and_gradient(theta)[0]

    def diagnostics_dict(self):
        return {
            "objectives": self.diagnostics.objectives,
            "grad_norms": self.diagnostics.grad_norms,
            "trace_terms": self.diagnostics.trace_terms,
            "logdet_terms": self.diagnostics.logdet_terms,
        }
