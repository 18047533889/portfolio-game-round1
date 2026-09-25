"""Verify the two closed forms the short-enabled solver relies on.

1. project_box_simplex(v, c) is the exact Euclidean projection onto
   {w : sum(w) = 1, w >= -c}. Checked against cvxpy solving the projection
   problem directly.

2. The duality-gap lower bound min_{v in C} gradient'v is correct on that same
   set. Checked against cvxpy solving the LP.

Both are the load-bearing claims of the short path: a projection that is not
exact makes the projected gradient a different algorithm, and a lower bound
that is too high makes the gap certificate lie about convergence. Neither claim
is safe to leave as algebra on paper.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/portfolio_game_round2/core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("core_r2_proj", CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_projection(v: np.ndarray, c: float) -> np.ndarray:
    """Independent reference by bisection on the KKT multiplier.

    At the projection w, complementary slackness forces w = max(v - lam, -c)
    for the scalar lam that makes the budget bind, so the answer can be found
    without an optimiser and without inheriting an optimiser's tolerance.
    """
    lo = float(v.min()) - 1e8 * max(1.0, abs(float(v.min())))
    hi = float(v.max()) + 1e8 * max(1.0, abs(float(v.max())))
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if float(np.maximum(v - mid, -c).sum()) > 1.0:
            lo = mid
        else:
            hi = mid
    return np.maximum(v - 0.5 * (lo + hi), -c)


def lp_projection(v: np.ndarray, c: float) -> np.ndarray:
    import cvxpy as cp
    n = v.size
    w = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(w - v)),
                         [cp.sum(w) == 1, w >= -c])
    problem.solve(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return np.asarray(w.value).ravel()


def reference_minimum_dot(g: np.ndarray, c: float) -> float:
    import cvxpy as cp
    n = g.size
    v = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(g @ v), [cp.sum(v) == 1, v >= -c])
    problem.solve(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return float(problem.value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    core = load_core()
    rng = np.random.default_rng(20260926)
    projection_errors, lp_errors, bound_errors, checked = [], [], [], 0

    for _ in range(args.trials):
        n = int(rng.integers(2, 14))
        c = float(rng.choice([0.02, 0.05, 0.10, 0.20, 0.35, 0.5]))
        v = rng.normal(0, 1, size=n) * rng.choice([0.05, 0.5, 3.0])
        checked += 1

        got = core.project_box_simplex(v, c)
        projection_errors.append(
            float(np.max(np.abs(got - reference_projection(v, c)))))
        lp_errors.append(float(np.max(np.abs(got - lp_projection(v, c)))))

        # Feasibility of our own answer, independently of the reference.
        assert abs(float(got.sum()) - 1.0) < 1e-9, f"budget {got.sum()!r}"
        assert got.min() >= -c - 1e-10, f"floor {got.min()!r} vs {-c!r}"

        # The gap bound: a linear form over the box-simplex.
        g = rng.normal(0, 1, size=n)
        closed = ((1.0 + n * c) * float(np.min(g)) - c * float(g.sum()))
        lpv = reference_minimum_dot(g, c)
        bound_errors.append(abs(closed - lpv))
        # It must be a genuine lower bound for every feasible point.
        for _ in range(20):
            probe = core.project_box_simplex(rng.normal(0, 1, size=n), c)
            assert float(g @ probe) >= closed - 1e-8, "bound is not a lower bound"

    worst_projection = max(projection_errors)
    worst_lp = max(lp_errors)
    worst_bound = max(bound_errors)
    print(f"trials                        : {checked}")
    print(f"max ||ours - bisection||_inf  : {worst_projection:.3e}")
    print(f"max ||ours - cvxpy||_inf      : {worst_lp:.3e}   (solver tolerance)")
    print(f"max |bound - LP value|        : {worst_bound:.3e}   (solver tolerance)")
    ok = worst_projection < 1e-9 and worst_lp < 1e-5 and worst_bound < 1e-6
    print(f"\nSHORT PROJECTION & GAP: {'VERIFIED' if ok else 'FAILED'}")
    if args.out:
        args.out.write_text(json.dumps({
            "trials": checked,
            "max_projection_error_vs_bisection": worst_projection,
            "max_projection_error_vs_cvxpy": worst_lp,
            "max_gap_bound_error_vs_lp": worst_bound,
            "verified": bool(ok),
        }, indent=2))
        print(f"written -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
