"""Mesh and function space helpers for the GP-PDE MLE solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from dolfin import (
    Constant,
    DirichletBC,
    FunctionSpace,
    MeshFunction,
    SubDomain,
    UnitSquareMesh,
)


class Boundary(SubDomain):
    """Marker for the unit-square boundary."""

    def inside(self, x, on_boundary):  # type: ignore[override]
        return bool(on_boundary)


@dataclass
class Spaces:
    """Container for the state and coefficient spaces."""

    state: FunctionSpace
    coeff: FunctionSpace

    @property
    def V(self) -> FunctionSpace:
        return self.state

    @property
    def Va(self) -> FunctionSpace:
        return self.coeff


def make_unit_square(nx: int, ny: int) -> Tuple[UnitSquareMesh, MeshFunction]:
    """Create the unit square mesh and a boundary marker."""

    mesh = UnitSquareMesh(nx, ny)
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)
    Boundary().mark(boundary, 1)
    return mesh, boundary


def build_spaces(
    mesh,
    state_family: str = "P",
    state_degree: int = 1,
    coeff_family: str = "P",
    coeff_degree: int = 1,
) -> Spaces:
    """Return function spaces for the state (u) and coefficient (a)."""

    V = FunctionSpace(mesh, state_family, state_degree)
    Va = FunctionSpace(mesh, coeff_family, coeff_degree)
    return Spaces(state=V, coeff=Va)


def build_dirichlet_bcs(V: FunctionSpace):
    """Return homogeneous Dirichlet boundary conditions on the unit square."""

    return [DirichletBC(V, Constant(0.0), Boundary())]
