#!/usr/bin/env python3
"""Finite-element solver for the 3-D gravity acoustic wave system.

This module re-implements the workflow sketched in the legacy script that
accompanied the assignment.  The new version focuses on readability and
robustness while keeping the overall mathematical model identical:

.. math::
    \rho \partial_t u + \nabla p = 0,\\
    K^{-1} \partial_t p + \nabla \cdot u = 0

supplemented with a free-surface mass term, absorbing walls and a prescribed
bottom uplift rate.  The implementation targets ``dolfin``/``FEniCS``
2019.1 and MPI execution, but it can also be used in serial mode.

Compared to the reference code the solver

* exposes a tidy :class:`AGW3DSolver` class with clear responsibilities;
* isolates the configuration in a :class:`SolverOptions` data class;
* provides a command line interface (``--help`` shows all options);
* keeps all MPI printing and file IO confined to rank zero; and
* documents every public function with doctrings, making it easier for
  students to adapt the script for different experiments.

The script can be executed directly, e.g.::

    python agw3d_solver.py --nx 40 --ny 40 --nz 10 --steps 200

The computation is reasonably lightweight for a laptop mesh but still
benefits from MPI parallelism for high resolutions.  In addition to the
time-marching solver, the script can assemble the *forward increment* map
that sends a discretised seafloor uplift history to bottom pressure sensors::

    python agw3d_solver.py --assemble-forward --forward-output forward_map.npz
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
import tempfile
from typing import List, Optional, Tuple

import numpy as np

try:  # Optional dependency for sparse storage of the forward map
    import scipy.sparse as sp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - scipy may be unavailable
    sp = None  # type: ignore

try:  # pragma: no cover - dolfin is heavy to import during testing
    from fenics import (  # type: ignore
        DOLFIN_EPS,
        FacetNormal,
        Function,
        FunctionSpace,
        HDF5File,
        MeshFunction,
        Point,
        SubDomain,
        Timer,
        UserExpression,
        VectorElement,
        XDMFFile,
        assemble,
        dx,
        ds,
        grad,
        inner,
    )
    import dolfin as dl  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "The FEniCS (dolfin) package is required to run this solver."
    ) from exc

# ---------------------------------------------------------------------------
# MPI helpers
# ---------------------------------------------------------------------------
comm = dl.MPI.comm_world
rank = comm.rank
size = comm.size


def log_once(message: str) -> None:
    """Print ``message`` on rank zero only."""
    if rank == 0:
        print(message)
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SolverOptions:
    """Container that stores all physical and numerical parameters."""

    # Domain extents (metres)
    xmin_km: float = -75.0
    xmax_km: float = 75.0
    ymin_km: float = -75.0
    ymax_km: float = 75.0
    zmin_km: float = -4.5
    zmax_km: float = 0.0

    # Mesh resolution
    nx: int = 40
    ny: int = 40
    nz: int = 10

    # Polynomial orders
    order_u: int = 0  # DG order for velocity
    order_p: int = 1  # CG order for pressure

    # Physical constants
    rho: float = 1025.0
    gravity: float = 9.80665
    bulk_modulus: float = 2.34e9

    # Non-dimensional scales
    l0: float = 1.0e3
    t0: float = 1.0
    p0: float = 1.0e9

    # Forcing profile (see :class:`BottomUpliftRate`)
    profile_index: int = 0

    # Time integration
    total_time: Optional[float] = None
    ramp_time: Optional[float] = None
    amplitude: Optional[float] = None
    num_steps: Optional[int] = None

    # Output
    output_directory: str = "agw3d_out"
    save_stride: int = 0  # 0 -> auto (10 snapshots)
    abc_enabled: bool = True
    sensor_count: int = 25
    assemble_forward: bool = False
    forward_output: Optional[str] = None

    # Diagnostics and solver parameters
    rel_tol: float = 1e-10
    abs_tol: float = 1e-14
    max_iter: int = 2000
    monitor: bool = False

    def scales(self) -> Tuple[float, float, float]:
        """Return characteristic velocity, impedance and scaling factors."""
        u0 = self.l0 / self.t0
        acoustic_speed = math.sqrt(self.bulk_modulus / self.rho)
        impedance = self.rho * acoustic_speed
        return u0, acoustic_speed, impedance


def _configure_jit_cache() -> None:
    """Place FEniCS caches inside the writable temporary directory."""
    tmpdir = os.path.join(os.environ.get("TMPDIR", tempfile.gettempdir()), "fenics_jit_cache")
    os.makedirs(tmpdir, exist_ok=True)
    os.environ.setdefault("DOLFIN_JIT_CACHE", tmpdir)
    os.environ.setdefault("FFC_CACHE_DIR", tmpdir)
    os.environ.setdefault("INSTANT_CACHE_DIR", tmpdir)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Boundary descriptions
# ---------------------------------------------------------------------------
class _Bottom(SubDomain):
    def __init__(self, zmin: float, tol: float) -> None:
        super().__init__()
        self._zmin = zmin
        self._tol = tol

    def inside(self, x, on_boundary):  # noqa: D401
        return on_boundary and abs(x[2] - self._zmin) < self._tol


class _Surface(SubDomain):
    def __init__(self, zmax: float, tol: float) -> None:
        super().__init__()
        self._zmax = zmax
        self._tol = tol

    def inside(self, x, on_boundary):  # noqa: D401
        return on_boundary and abs(x[2] - self._zmax) < self._tol


class _Lateral(SubDomain):
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, tol: float) -> None:
        super().__init__()
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._tol = tol

    def inside(self, x, on_boundary):  # noqa: D401
        if not on_boundary:
            return False
        return (
            abs(x[0] - self._xmin) < self._tol
            or abs(x[0] - self._xmax) < self._tol
            or abs(x[1] - self._ymin) < self._tol
            or abs(x[1] - self._ymax) < self._tol
        )


class BottomUpliftRate(UserExpression):
    """Prescribed bottom velocity (time derivative of uplift)."""

    def __init__(self, options: SolverOptions, degree: int = 2):
        super().__init__(degree=degree)
        self.options = options
        self.t = 0.0
        # lateral footprint of the Gaussian source
        self.xr = 15.0
        self.yr = 15.0
        self.xc = 0.0
        self.yc = 0.0

        # The problem defines four ready-made forcing scenarios.  We keep the
        # values in the exact same order as in the reference script so that
        # existing parameter studies stay valid.
        self._T = [5.0, 6.0, 10.0, 60.0]
        self._Tr = [3.0, 5.0, 10.0, 50.0]
        self._A = [10.0, 10.0, 2.0, 10.0]
        self._steps = [100, 120, 200, 600]

    # ------------------------------------------------------------------
    # Helpers that expose the chosen profile to the solver
    # ------------------------------------------------------------------
    def profile(self) -> Tuple[float, float, float, int]:
        idx = max(0, min(len(self._T) - 1, int(self.options.profile_index)))
        T = self.options.total_time or self._T[idx]
        Tr = self.options.ramp_time or self._Tr[idx]
        amplitude = self.options.amplitude or self._A[idx]
        steps = self.options.num_steps or self._steps[idx]
        return T, Tr, amplitude, steps

    # ------------------------------------------------------------------
    def eval(self, value, x):  # noqa: D401
        l0 = self.options.l0
        xx_km = x[0] * (l0 / 1000.0)
        yy_km = x[1] * (l0 / 1000.0)
        gaussian = math.exp(-((xx_km - self.xc) / self.xr) ** 2 - ((yy_km - self.yc) / self.yr) ** 2)
        T, Tr, amplitude, _ = self.profile()
        t = self.t
        if Tr > 0.0 and t <= Tr:
            rate = amplitude * 0.5 * (math.pi / Tr) * math.sin(math.pi * t / Tr)
        else:
            rate = 0.0
        value[0] = gaussian * rate / self.options.l0  # non-dimensionalised

    def value_shape(self):  # noqa: D401
        return ()


# ---------------------------------------------------------------------------
# Solver core
# ---------------------------------------------------------------------------
class AGW3DSolver:
    """Time integrator for the gravity-acoustic wave system."""

    def __init__(self, options: SolverOptions):
        self.options = options
        self.mesh = None
        self.function_space = None
        self.sub_domains = None
        self.w_prev = None
        self.bottom_rate_prev = None
        self.bottom_rate_curr = None
        self.bottom_displacement = None
        self.u_form = None
        self.a_form = None
        self.L_form = None
        self.A_matrix = None
        self.solver = None
        self.ds = None
        self.normal = None
        self.dt = None
        self.num_steps = None
        self.times = []
        self.sensor_points: List[Point] = []
        self._control_dim = None
        self._state_dim = None
        self._zero_control = None
        self._zero_state = None
        self._zero_displacement = None

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        """Create the mesh, function spaces and variational forms."""
        _configure_jit_cache()
        dl.parameters["mesh_partitioner"] = "ParMETIS"

        opts = self.options
        log_once(f"[agw3d] MPI ranks: {size}")

        # Convert domain limits to the nondimensional frame used internally
        xscale = 1000.0 / opts.l0
        xmin = opts.xmin_km * xscale
        xmax = opts.xmax_km * xscale
        ymin = opts.ymin_km * xscale
        ymax = opts.ymax_km * xscale
        zmin = opts.zmin_km * xscale
        zmax = opts.zmax_km * xscale

        log_once(
            f"[agw3d] domain x=[{xmin:.1f},{xmax:.1f}] km, y=[{ymin:.1f},{ymax:.1f}] km, z=[{zmin:.1f},{zmax:.1f}] km"
        )

        self.mesh = dl.BoxMesh(Point(xmin, ymin, zmin), Point(xmax, ymax, zmax), opts.nx, opts.ny, opts.nz)
        log_once(f"[agw3d] mesh cells={self.mesh.num_cells()} vertices={self.mesh.num_vertices()}")

        DG = VectorElement("DG", self.mesh.ufl_cell(), opts.order_u, dim=3)
        CG = dl.FiniteElement("CG", self.mesh.ufl_cell(), opts.order_p)
        self.function_space = FunctionSpace(self.mesh, dl.MixedElement([DG, CG]))
        log_once(f"[agw3d] dofs={self.function_space.dim()}")

        self.sub_domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1, 3)
        tol = 10 * DOLFIN_EPS
        _Bottom(zmin, tol).mark(self.sub_domains, 0)
        _Surface(zmax, tol).mark(self.sub_domains, 1)
        _Lateral(xmin, xmax, ymin, ymax, tol).mark(self.sub_domains, 2)

        self.ds = dl.Measure("ds", domain=self.mesh, subdomain_data=self.sub_domains)
        self.normal = FacetNormal(self.mesh)

        self._setup_time_parameters()
        self._setup_forms()
        self._setup_solver()
        self._setup_output()
        self.sensor_points = self._build_sensor_points(self.options.sensor_count)
        self._control_dim = self.bottom_rate_prev.vector().size()
        self._state_dim = self.w_prev.vector().size()
        self._zero_control = np.zeros(self._control_dim)
        self._zero_state = np.zeros(self._state_dim)
        self._zero_displacement = np.zeros(self.bottom_displacement.vector().size())

    # ------------------------------------------------------------------
    def _setup_time_parameters(self) -> None:
        rate_expr = BottomUpliftRate(self.options)
        T, Tr, amplitude, steps = rate_expr.profile()
        self.num_steps = steps
        self.dt = T / steps / self.options.t0
        self.bottom_rate_prev = Function(FunctionSpace(self.mesh, "CG", 1))
        self.bottom_rate_curr = Function(FunctionSpace(self.mesh, "CG", 1))
        self.bottom_displacement = Function(FunctionSpace(self.mesh, "CG", 1))
        self.rate_expr_prev = rate_expr
        self.rate_expr_curr = BottomUpliftRate(self.options)
        self.rate_expr_prev.t = 0.0
        self.rate_expr_curr.t = 0.0
        self.rate_expr_prev.xr = rate_expr.xr
        self.rate_expr_curr.xr = rate_expr.xr
        self.rate_expr_prev.yr = rate_expr.yr
        self.rate_expr_curr.yr = rate_expr.yr
        self.rate_expr_prev.xc = rate_expr.xc
        self.rate_expr_curr.xc = rate_expr.xc
        self.rate_expr_prev.yc = rate_expr.yc
        self.rate_expr_curr.yc = rate_expr.yc
        self.rate_expr_prev.profile()
        self.rate_expr_curr.profile()

    # ------------------------------------------------------------------
    def _coefficients(self) -> Tuple[float, float, float, float, float]:
        opts = self.options
        u0 = opts.l0 / opts.t0
        c1 = opts.p0 * opts.t0**2 / (opts.rho * opts.l0**2)
        c2 = opts.bulk_modulus / opts.p0
        c3 = opts.rho * opts.gravity * opts.l0 / opts.p0
        impedance_dimless = opts.rho * math.sqrt(opts.bulk_modulus / opts.rho)
        Z0 = (u0 / opts.p0) * impedance_dimless
        return c1, c2, c3, Z0, math.sqrt(c1 * c2)

    # ------------------------------------------------------------------
    def _setup_forms(self) -> None:
        V = self.function_space
        (u, p) = dl.TrialFunctions(V)
        (tau, v) = dl.TestFunctions(V)

        self.w_prev = Function(V)

        c1, c2, c3, Z0, sqrt_c1c2 = self._coefficients()
        dt = self.dt

        self.a_form = (
            inner(u, tau) + p * v + 0.5 * dt * c1 * inner(grad(p), tau) - 0.5 * dt * c2 * inner(u, grad(v))
        ) * dx

        self.L_form = (
            inner(self.w_prev.sub(0), tau)
            + self.w_prev.sub(1) * v
            - 0.5 * dt * c1 * inner(grad(self.w_prev.sub(1)), tau)
            + 0.5 * dt * c2 * inner(self.w_prev.sub(0), grad(v))
        ) * dx

        # Free surface coupling
        self.a_form += (c2 / c3) * p * v * self.ds(1)
        self.L_form += (c2 / c3) * self.w_prev.sub(1) * v * self.ds(1)

        if self.options.abc_enabled:
            self.a_form += 0.5 * dt * sqrt_c1c2 * p * v * self.ds(2)
            self.L_form += -0.5 * dt * sqrt_c1c2 * self.w_prev.sub(1) * v * self.ds(2)

        self.L_form += (0.5 * dt * c2) * (self.bottom_rate_prev + self.bottom_rate_curr) * v * self.ds(0)

    # ------------------------------------------------------------------
    def _setup_solver(self) -> None:
        dl.parameters["form_compiler"]["representation"] = "uflacs"
        dl.parameters["form_compiler"]["optimize"] = True
        dl.parameters["form_compiler"]["cpp_optimize"] = True

        log_once("[agw3d] assembling system matrix ...")
        self.A_matrix = assemble(self.a_form)

        from dolfin import PETScKrylovSolver  # imported lazily to avoid startup overhead

        self.solver = PETScKrylovSolver("gmres")
        self.solver.set_operator(self.A_matrix)
        ksp = self.solver.ksp()
        pc = ksp.getPC()
        pc.setType("gamg")
        self.solver.parameters["relative_tolerance"] = self.options.rel_tol
        self.solver.parameters["absolute_tolerance"] = self.options.abs_tol
        self.solver.parameters["maximum_iterations"] = self.options.max_iter
        self.solver.parameters["monitor_convergence"] = self.options.monitor

    # ------------------------------------------------------------------
    def _setup_output(self) -> None:
        outdir = os.path.abspath(self.options.output_directory)
        if rank == 0:
            os.makedirs(outdir, exist_ok=True)
            os.makedirs(os.path.join(outdir, "h5"), exist_ok=True)
        dl.MPI.barrier(comm)
        self.outdir = outdir
        self.xdmf_u = XDMFFile(comm, os.path.join(outdir, "u.xdmf"))
        self.xdmf_p = XDMFFile(comm, os.path.join(outdir, "p.xdmf"))
        for f in (self.xdmf_u, self.xdmf_p):
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True

        self.h5dir = os.path.join(outdir, "h5")
        if rank == 0:
            meta = dict(
                xmin_km=float(self.options.xmin_km),
                xmax_km=float(self.options.xmax_km),
                ymin_km=float(self.options.ymin_km),
                ymax_km=float(self.options.ymax_km),
                zmin_km=float(self.options.zmin_km),
                zmax_km=float(self.options.zmax_km),
                l0=float(self.options.l0),
                p0=float(self.options.p0),
                rho=float(self.options.rho),
                gravity=float(self.options.gravity),
            )
            np.savez(os.path.join(outdir, "sampling_meta.npz"), **meta)
        dl.MPI.barrier(comm)

        with HDF5File(comm, os.path.join(self.h5dir, "mesh.h5"), "w") as h5:
            h5.write(self.mesh, "/mesh")

    # ------------------------------------------------------------------
    def advance(self) -> None:
        """Run the time integration."""
        timer = Timer("agw3d time loop")
        dt = self.dt
        t = 0.0
        b_vec = None

        log_once("[agw3d] starting time stepping ...")
        for step in range(1, self.num_steps + 1):
            t += dt
            self.rate_expr_prev.t = t - dt
            self.rate_expr_curr.t = t
            self.bottom_rate_prev.interpolate(self.rate_expr_prev)
            self.bottom_rate_curr.interpolate(self.rate_expr_curr)

            b_vec = assemble(self.L_form, tensor=b_vec)
            self.solver.solve(self.w_prev.vector(), b_vec)

            self.bottom_displacement.vector().axpy(0.5 * dt, self.bottom_rate_prev.vector())
            self.bottom_displacement.vector().axpy(0.5 * dt, self.bottom_rate_curr.vector())

            self._write_outputs(step, t)
            self.times.append((step, t * self.options.t0))
            if rank == 0:
                sys.stdout.write(f"[agw3d] step {step:5d}/{self.num_steps} t={t * self.options.t0:.3f}s\n")
                sys.stdout.flush()

        if rank == 0:
            wall = timer.elapsed()[0]
            log_once(f"[agw3d] done in {wall:.2f}s. data written to {self.outdir}")

    # ------------------------------------------------------------------
    def _write_outputs(self, step: int, time_: float) -> None:
        save_every = self.options.save_stride or max(1, self.num_steps // 10)
        u, p = self.w_prev.split(deepcopy=True)
        if step % save_every == 0 or step == self.num_steps:
            self.xdmf_u.write(u, time_ * self.options.t0)
            self.xdmf_p.write(p, time_ * self.options.t0)

        fname_p = os.path.join(self.h5dir, f"p_{step:05d}.h5")
        fname_xi = os.path.join(self.h5dir, f"xi_{step:05d}.h5")
        with contextlib.ExitStack() as stack:
            h5p = stack.enter_context(HDF5File(comm, fname_p, "w"))
            h5xi = stack.enter_context(HDF5File(comm, fname_xi, "w"))
            h5p.write(p, "/p")
            h5xi.write(self.bottom_displacement, "/xi")

        if rank == 0:
            with open(os.path.join(self.outdir, "times.txt"), "a") as f:
                f.write(f"{step} {time_ * self.options.t0:.9e}\n")

    # ------------------------------------------------------------------
    def _build_sensor_points(self, count: int) -> List[Point]:
        """Return ``count`` uniformly distributed sensors on the bottom."""
        if count <= 0:
            raise ValueError("Sensor count must be positive")
        side = int(round(math.sqrt(count)))
        if side * side != count:
            raise ValueError("Sensor count must be a perfect square (e.g. 25)")

        opts = self.options
        scale = 1000.0 / opts.l0
        xs_km = np.linspace(opts.xmin_km, opts.xmax_km, side)
        ys_km = np.linspace(opts.ymin_km, opts.ymax_km, side)
        z_nd = opts.zmin_km * scale + 1e-8

        sensors: List[Point] = []
        for y in ys_km:
            for x in xs_km:
                sensors.append(Point(x * scale, y * scale, z_nd))

        if rank == 0:
            log_once(
                f"[agw3d] placing {count} bottom sensors on a {side}x{side} grid"
            )

        return sensors

    # ------------------------------------------------------------------
    def _assign_vector(self, vec, values: np.ndarray) -> None:
        """Assign the distributed vector ``vec`` from a global array."""
        start, end = vec.local_range()
        vec.set_local(values[start:end])
        vec.apply("insert")

    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        """Reset mixed solution and control fields to zero."""
        self._assign_vector(self.w_prev.vector(), self._zero_state)
        self._assign_vector(self.bottom_rate_prev.vector(), self._zero_control)
        self._assign_vector(self.bottom_rate_curr.vector(), self._zero_control)
        self._assign_vector(self.bottom_displacement.vector(), self._zero_displacement)

    # ------------------------------------------------------------------
    def _sample_pressure(self) -> np.ndarray:
        """Evaluate the pressure component at sensor locations."""
        _, pressure = self.w_prev.split(deepcopy=True)
        readings = []
        for pt in self.sensor_points:
            value = None
            try:
                value = float(pressure(pt))
            except RuntimeError:
                value = None

            gathered = comm.allgather(value)
            for entry in gathered:
                if entry is not None:
                    readings.append(entry * self.options.p0)
                    break
            else:  # pragma: no cover - indicates an evaluation failure
                raise RuntimeError("Failed to evaluate pressure at a sensor point")

        return np.asarray(readings)

    # ------------------------------------------------------------------
    def _simulate_basis(self, step_index: int, dof_index: int) -> np.ndarray:
        """Return the sensor response for a unit control at ``(step, dof)``."""
        if self._control_dim is None or self._state_dim is None:
            raise RuntimeError("Solver must be prepared before assembling operators")

        unit = np.zeros(self._control_dim)
        unit[dof_index] = 1.0
        zero = self._zero_control

        self._reset_state()
        previous = zero
        b_vec = None
        responses = []

        for step in range(self.num_steps):
            current = unit if step == step_index else zero
            self._assign_vector(self.bottom_rate_prev.vector(), previous)
            self._assign_vector(self.bottom_rate_curr.vector(), current)
            b_vec = assemble(self.L_form, tensor=b_vec)
            self.solver.solve(self.w_prev.vector(), b_vec)
            responses.append(self._sample_pressure())
            previous = current

        return np.concatenate(responses)

    # ------------------------------------------------------------------
    def assemble_forward_increment_matrix(self, sensor_count: Optional[int] = None):
        """Assemble the linear operator mapping uplift to sensor pressures."""
        if sensor_count is not None and sensor_count != self.options.sensor_count:
            self.sensor_points = self._build_sensor_points(sensor_count)
        if self._control_dim is None:
            raise RuntimeError("Solver is not initialised; call prepare() first")

        num_outputs = len(self.sensor_points) * self.num_steps
        total_controls = self._control_dim * self.num_steps

        if sp is not None:
            operator = sp.lil_matrix((num_outputs, total_controls))
        else:
            operator = np.zeros((num_outputs, total_controls))

        if rank == 0:
            log_once(
                f"[agw3d] assembling forward map with {total_controls} control dofs"
            )

        for step in range(self.num_steps):
            for dof in range(self._control_dim):
                response = self._simulate_basis(step, dof)
                column_data = np.asarray(response)
                column = step * self._control_dim + dof
                if sp is not None:
                    operator[:, column] = column_data.reshape(-1, 1)
                else:
                    operator[:, column] = column_data

        if sp is not None:
            operator = operator.tocsr()

        self._reset_state()

        return operator

    # ------------------------------------------------------------------
    def save_forward_operator(self, operator, path: str) -> None:
        """Persist the forward operator to disk."""
        dirname = os.path.dirname(os.path.abspath(path))
        if dirname and rank == 0:
            os.makedirs(dirname, exist_ok=True)
        dl.MPI.barrier(comm)

        metadata = dict(
            num_steps=self.num_steps,
            sensor_count=len(self.sensor_points),
            control_dofs=self._control_dim,
            dt=self.dt * self.options.t0,
        )

        if sp is not None and isinstance(operator, sp.spmatrix):
            sp.save_npz(path, operator.tocsr())
            dl.MPI.barrier(comm)
            if rank == 0:
                with open(f"{path}.json", "w", encoding="utf8") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)
        else:
            np.savez(path, forward_operator=np.asarray(operator), **metadata)

        if rank == 0:
            log_once(f"[agw3d] forward operator stored in {path}")


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[list] = None) -> SolverOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=40, help="Number of cells along x")
    parser.add_argument("--ny", type=int, default=40, help="Number of cells along y")
    parser.add_argument("--nz", type=int, default=10, help="Number of cells along z")
    parser.add_argument("--order-u", type=int, default=0, help="DG polynomial degree for velocity")
    parser.add_argument("--order-p", type=int, default=1, help="CG polynomial degree for pressure")
    parser.add_argument("--profile", type=int, default=0, help="Index of the built-in forcing profile")
    parser.add_argument("--total-time", type=float, help="Override total simulation time in seconds")
    parser.add_argument("--ramp-time", type=float, help="Override uplift ramp time in seconds")
    parser.add_argument("--amplitude", type=float, help="Override uplift amplitude in metres")
    parser.add_argument("--steps", type=int, help="Override number of time steps")
    parser.add_argument("--output", default="agw3d_out", help="Output directory")
    parser.add_argument("--abc", action="store_true", help="Enable absorbing boundary conditions")
    parser.add_argument("--no-abc", action="store_true", help="Disable absorbing boundary conditions")
    parser.add_argument("--monitor", action="store_true", help="Print PETSc convergence information")
    parser.add_argument("--rel-tol", type=float, default=1e-10, help="Relative tolerance for GMRES")
    parser.add_argument("--abs-tol", type=float, default=1e-14, help="Absolute tolerance for GMRES")
    parser.add_argument("--max-iter", type=int, default=2000, help="Maximum iterations for GMRES")
    parser.add_argument(
        "--sensor-count",
        type=int,
        default=25,
        help="Number of uniformly spaced bottom pressure sensors",
    )
    parser.add_argument(
        "--assemble-forward",
        action="store_true",
        help="Assemble the linear map from uplift to sensor pressures",
    )
    parser.add_argument(
        "--forward-output",
        help="Path to save the assembled forward map (npz format)",
    )
    args = parser.parse_args(argv)

    if args.abc and args.no_abc:
        parser.error("Cannot request both --abc and --no-abc")

    options = SolverOptions(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        order_u=args.order_u,
        order_p=args.order_p,
        profile_index=args.profile,
        output_directory=args.output,
        abc_enabled=not args.no_abc,
        monitor=args.monitor,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        max_iter=args.max_iter,
        sensor_count=args.sensor_count,
        assemble_forward=args.assemble_forward,
        forward_output=args.forward_output,
    )

    if args.total_time is not None:
        options.total_time = args.total_time
    if args.ramp_time is not None:
        options.ramp_time = args.ramp_time
    if args.amplitude is not None:
        options.amplitude = args.amplitude
    if args.steps is not None:
        options.num_steps = args.steps

    return options


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    options = _parse_args(argv)
    solver = AGW3DSolver(options)
    solver.prepare()
    if options.assemble_forward:
        operator = solver.assemble_forward_increment_matrix()
        if options.forward_output:
            solver.save_forward_operator(operator, options.forward_output)
        elif rank == 0:
            log_once("[agw3d] forward operator assembled (not saved)")
    else:
        solver.advance()


if __name__ == "__main__":
    main()
