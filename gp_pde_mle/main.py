"""Command line interface for GP-PDE coefficient estimation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from dolfin import Function, FunctionSpace

from .core.mesh import build_dirichlet_bcs, build_spaces, make_unit_square
from .objective.mle import GaussianMLE, ObjectiveConfig
from .operators.elliptic import EllipticOperator
from .opt.adam import AdamConfig, AdamOptimizer
from .opt.lbfgs import LBFGSConfig, LBFGSOptimizer
from .io.utils import save_function_xdmf, save_history_csv


def _generate_white_noise(V: FunctionSpace, seed: Optional[int] = None) -> Function:
    rng = np.random.default_rng(seed)
    noise = Function(V)
    noise.vector()[:] = rng.standard_normal(noise.vector().size())
    noise.vector().apply("insert")
    return noise


def _generate_samples(
    operator: EllipticOperator,
    m: int,
    seed: Optional[int] = None,
) -> List[Function]:
    """Sample from the Gaussian field using white-noise forcing."""

    samples: List[Function] = []
    rng = np.random.default_rng(seed)
    for i in range(m):
        white = _generate_white_noise(operator.V, seed=rng.integers(0, 1 << 32))
        sample = operator.solve_L(white)
        samples.append(sample)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GP-PDE MLE for spatial coefficients")
    parser.add_argument("--nx", type=int, default=32, help="Number of cells in x direction")
    parser.add_argument("--ny", type=int, default=32, help="Number of cells in y direction")
    parser.add_argument("--m", type=int, default=10, help="Number of field samples")
    parser.add_argument("--Nz", type=int, default=20, help="Number of Hutchinson probes")
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="lbfgs")
    parser.add_argument("--alpha", type=float, default=1e-2, help="Regularization strength")
    parser.add_argument("--reg", choices=["h1", "tv"], default="h1", help="Regularizer type")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for Adam")
    parser.add_argument("--max-iters", type=int, default=200, help="Maximum optimization iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="Gradient tolerance")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--no-log-param", action="store_true", help="Disable log-parameterization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh, boundary = make_unit_square(args.nx, args.ny)
    spaces = build_spaces(mesh)
    bcs = build_dirichlet_bcs(spaces.V)
    theta0 = Function(spaces.Va)
    theta0.vector()[:] = np.log(np.ones(theta0.vector().size()))
    theta0.vector().apply("insert")
    a0 = Function(spaces.Va)
    a0.vector()[:] = np.exp(theta0.vector().get_local())
    a0.vector().apply("insert")
    operator = EllipticOperator(spaces.V, a0, bcs=bcs)
    samples = _generate_samples(operator, args.m, seed=args.seed)
    config = ObjectiveConfig(
        alpha=args.alpha,
        reg_type=args.reg,
        use_log_parameterization=not args.no_log_param,
        num_probes=args.Nz,
        probe_seed=args.seed,
    )
    mle = GaussianMLE(spaces.V, spaces.Va, samples, operator, config, bcs=bcs)
    theta = theta0 if config.use_log_parameterization else a0

    history_rows: List[List[float]] = []
    if args.optimizer == "adam":
        adam = AdamOptimizer(AdamConfig(learning_rate=args.lr, max_iters=args.max_iters, tol=args.tol))
        for it in range(args.max_iters):
            obj, grad = mle.objective_and_gradient(theta)
            grad_norm = np.linalg.norm(grad.vector().get_local()) / np.sqrt(theta.function_space().dim())
            history_rows.append([it, obj, grad_norm])
            if grad_norm < args.tol:
                break
            theta = adam.step(theta, grad)
    else:
        lbfgs = LBFGSOptimizer(LBFGSConfig(max_iters=args.max_iters, tol=args.tol))
        obj, grad = mle.objective_and_gradient(theta)
        history_rows.append([0, obj, np.linalg.norm(grad.vector().get_local()) / np.sqrt(theta.function_space().dim())])
        for it in range(1, args.max_iters + 1):
            theta, obj, grad, state = lbfgs.step(theta, obj, grad, mle.objective_and_gradient)
            grad_norm = np.linalg.norm(grad.vector().get_local()) / np.sqrt(theta.function_space().dim())
            history_rows.append([it, obj, grad_norm])
            if grad_norm < args.tol or state.converged:
                break

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    result_field = theta if not config.use_log_parameterization else Function(spaces.Va)
    if config.use_log_parameterization:
        result_field.vector()[:] = np.exp(theta.vector().get_local())
        result_field.vector().apply("insert")
    save_function_xdmf(output_dir / "estimated_coefficient.xdmf", result_field, name="a_est")
    save_history_csv(output_dir / "history.csv", ["iter", "objective", "grad_norm"], history_rows)


if __name__ == "__main__":
    main()
