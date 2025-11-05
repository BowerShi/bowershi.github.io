"""Elliptic operators and helper functionality."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
from dolfin import (
    DirichletBC,
    Function,
    FunctionSpace,
    LinearVariationalProblem,
    LinearVariationalSolver,
    TrialFunction,
    TestFunction,
    assemble,
    div,
    dx,
    grad,
    inner,
)


def _ensure_list(bcs: Optional[Iterable[DirichletBC]]) -> List[DirichletBC]:
    if bcs is None:
        return []
    return list(bcs)


def project_with_bc(expr, V: FunctionSpace, bcs: Optional[Iterable[DirichletBC]] = None,
                    solver_parameters: Optional[Dict] = None) -> Function:
    """Project an expression into *V* while enforcing boundary conditions."""

    u = TrialFunction(V)
    v = TestFunction(V)
    a_form = inner(u, v) * dx
    L_form = inner(expr, v) * dx
    result = Function(V)
    problem = LinearVariationalProblem(a_form, L_form, result, _ensure_list(bcs))
    solver = LinearVariationalSolver(problem)
    if solver_parameters:
        solver.parameters.update(solver_parameters)
    solver.solve()
    return result


@dataclass
class EllipticOperator:
    """Matrix-free access to *A(a)* and *L(a)=A(a)^T A(a).*"""

    V: FunctionSpace
    a: Function
    bcs: List[DirichletBC] = field(default_factory=list)
    solver_parameters: Optional[Dict] = None

    def update_coefficient(self, a: Function) -> None:
        self.a = a

    def apply_A(self, u: Function) -> Function:
        """Compute *w = A(a) u = -div(a grad u)*."""

        expr = -div(self.a * grad(u))
        return project_with_bc(expr, self.V, self.bcs, self.solver_parameters)

    def solve_A(self, rhs: Function) -> Function:
        """Solve the Poisson-type equation *A(a) w = rhs*."""

        w = Function(self.V)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        a_form = self.a * inner(grad(u), grad(v)) * dx
        L_form = inner(rhs, v) * dx
        problem = LinearVariationalProblem(a_form, L_form, w, self.bcs)
        solver = LinearVariationalSolver(problem)
        if self.solver_parameters:
            solver.parameters.update(self.solver_parameters)
        solver.solve()
        return w

    def apply_L(self, u: Function) -> Function:
        """Apply the normal-equation operator *L(a) = A^T A*."""

        return self.apply_A(self.apply_A(u))

    def solve_L(self, rhs: Function) -> Function:
        """Solve *L(a) y = rhs* using two sequential Poisson solves."""

        w = self.solve_A(rhs)
        return self.solve_A(w)

    def energy_inner(self, u: Function, v: Function) -> float:
        """Return the bilinear form \int a grad(u)·grad(v) dx."""

        form = self.a * inner(grad(u), grad(v)) * dx
        return float(assemble(form))

    def l2_inner(self, u: Function, v: Function) -> float:
        """Return the L2 inner product \int u v dx."""

        return float(assemble(inner(u, v) * dx))

    def apply_mass_inverse(self, u: Function) -> Function:
        """Solve the mass matrix problem (I) w = u (utility for diagnostics)."""

        w = Function(self.V)
        trial = TrialFunction(self.V)
        test = TestFunction(self.V)
        a_form = inner(trial, test) * dx
        L_form = inner(u, test) * dx
        problem = LinearVariationalProblem(a_form, L_form, w, self.bcs)
        solver = LinearVariationalSolver(problem)
        if self.solver_parameters:
            solver.parameters.update(self.solver_parameters)
        solver.solve()
        return w


def function_from_array(V: FunctionSpace, values: np.ndarray, bcs: Optional[Iterable[DirichletBC]] = None) -> Function:
    """Create a function with given local values while satisfying boundary conditions."""

    f = Function(V)
    f.vector()[:] = values
    if bcs:
        for bc in bcs:
            bc.apply(f.vector())
    f.vector().apply("insert")
    return f
