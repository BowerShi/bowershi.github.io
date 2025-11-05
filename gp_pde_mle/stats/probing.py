"""Stochastic probing utilities (Hutchinson, Lanczos)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from dolfin import DirichletBC, Function, FunctionSpace


def draw_probes(
    V: FunctionSpace,
    count: int,
    kind: str = "rademacher",
    seed: Optional[int] = None,
    bcs: Optional[Iterable[DirichletBC]] = None,
) -> List[Function]:
    """Draw random probe vectors in the state space."""

    rng = np.random.default_rng(seed)
    probes: List[Function] = []
    for _ in range(count):
        z = Function(V)
        local = z.vector().get_local()
        if kind.lower() == "gaussian":
            local = rng.standard_normal(local.shape)
        else:  # default rademacher
            local = rng.choice([-1.0, 1.0], size=local.shape)
        z.vector().set_local(local)
        z.vector().apply("insert")
        if bcs:
            for bc in bcs:
                bc.apply(z.vector())
        probes.append(z)
    return probes


def hutchinson_trace(apply_operator: Callable[[Function], Function], probes: Sequence[Function]) -> float:
    """Estimate ``trace(A)`` via the Hutchinson estimator."""

    if not probes:
        raise ValueError("At least one probe is required")
    accum = 0.0
    for z in probes:
        Az = apply_operator(z)
        accum += z.vector().inner(Az.vector())
    return float(accum / len(probes))


@dataclass
class LanczosResult:
    logdet: float
    eigenvalues: np.ndarray
    weights: np.ndarray


def slq_logdet(
    apply_operator: Callable[[Function], Function],
    probes: Sequence[Function],
    krylov_dim: int,
    tol: float = 1e-10,
) -> LanczosResult:
    """Stochastic Lanczos Quadrature approximation of ``log det``."""

    try:
        from scipy.linalg import eigh_tridiagonal
    except Exception as exc:  # pragma: no cover - informative fallback
        raise RuntimeError("scipy is required for SLQ log-det estimation") from exc

    if krylov_dim <= 0:
        raise ValueError("krylov_dim must be positive")
    if not probes:
        raise ValueError("At least one probe is required for SLQ")

    logdet_estimate = 0.0
    eigenvalues_list: List[np.ndarray] = []
    weights_list: List[np.ndarray] = []

    for z in probes:
        q = z.copy(deepcopy=True)
        alphas: List[float] = []
        betas: List[float] = []
        q_norm = np.sqrt(q.vector().inner(q.vector()))
        if q_norm == 0:
            continue
        q.vector()[:] /= q_norm
        w = apply_operator(q)
        alpha = q.vector().inner(w.vector())
        w.vector().axpy(-alpha, q.vector())
        beta = np.sqrt(w.vector().inner(w.vector()))
        alphas.append(alpha)
        betas.append(beta)
        for _ in range(1, krylov_dim):
            if beta < tol:
                break
            q_next = w.copy(deepcopy=True)
            q_next.vector()[:] /= beta
            w = apply_operator(q_next)
            w.vector().axpy(-beta, q.vector())
            alpha = q_next.vector().inner(w.vector())
            w.vector().axpy(-alpha, q_next.vector())
            beta = np.sqrt(max(w.vector().inner(w.vector()), 0.0))
            alphas.append(alpha)
            betas.append(beta)
            q = q_next
        # build tridiagonal
        diag = np.array(alphas, dtype=float)
        offdiag = np.array(betas[:-1], dtype=float)
        evals, vecs = eigh_tridiagonal(diag, offdiag)
        weights = (vecs[0, :] ** 2)
        eigenvalues_list.append(evals)
        weights_list.append(weights)
        dim = z.vector().size()
        logdet_estimate += dim * np.dot(weights, np.log(np.clip(evals, tol, None)))
    logdet_estimate /= len(probes)
    eigenvalues = np.concatenate(eigenvalues_list) if eigenvalues_list else np.array([], dtype=float)
    weights = np.concatenate(weights_list) if weights_list else np.array([], dtype=float)
    return LanczosResult(logdet=logdet_estimate, eigenvalues=eigenvalues, weights=weights)
