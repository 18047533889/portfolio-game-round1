"""Verify the two closed forms the round-3 short budget relies on.

Round 3 grants gross leverage 1.5, which with full investment means an aggregate
short budget of 0.25. The feasible set is therefore

    C = {w : sum(w) = 1, sum(max(-w, 0)) <= s}

Two claims carry the solver on that set, and neither is safe to leave as algebra:

1. project_gross_budget(v, s) is the exact Euclidean projection onto C. Checked
   against cvxpy solving the projection problem directly, and against an
   independent bisection on the KKT thresholds.

2. The duality-gap lower bound min_{v in C} gradient'v equals
   g_min + s * (g_min - g_max). Checked against cvxpy solving the LP, and -- more
   importantly -- verified to be a genuine lower bound at every sampled feasible
   point, because a bound that is too high would make the convergence
   certificate lie.

A projection that is merely close would make the projected gradient a different
algorithm, and the certificate would then be about that algorithm instead.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/portfolio_game_round3/core.py"


def load_core():
    spec = importlib.util.spec_from_file_location("core_r3_proj", CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_projection_vectorised(v: np.ndarray, s: float) -> np.ndarray:
    """Independent reference by bisection on the KKT thresholds.

    The projection has the form soft(y - lam, theta), equivalently
    max(y - a, 0) - max(b - y, 0) with a >= b, so a and b are two scalars. Here
    they are found by bisection on the two defining sums rather than by the
    water-filling argument under test.
    """
    n = v.size
    if np.maximum(-(v - (v.sum() - 1.0) / n), 0.0).sum() <= s + 1e-12:
        return v - (v.sum() - 1.0) / n
    # a solves sum(max(v - a, 0)) = 1 + s ; b solves sum(max(b - v, 0)) = s
    hi = float(v.max()) + 1e3 * max(1.0, abs(float(v.max())))
    lo = float(v.min()) - 1e3 * max(1.0, abs(float(v.min())))
    a_lo, a_hi = lo, hi
    for _ in range(300):
        mid = 0.5 * (a_lo + a_hi)
        if float(np.maximum(v - mid, 0.0).sum()) > 1.0 + s:
            a_lo = mid
        else:
            a_hi = mid
    a = 0.5 * (a_lo + a_hi)
    b_lo, b_hi = lo, hi
    for _ in range(300):
        mid = 0.5 * (b_lo + b_hi)
        if float(np.maximum(mid - v, 0.0).sum()) > s:
            b_hi = mid
        else:
            b_lo = mid
    b = 0.5 * (b_lo + b_hi)
    return np.maximum(v - a, 0.0) - np.maximum(b - v, 0.0)


def lp_projection(v: np.ndarray, s: float) -> np.ndarray:
    import cvxpy as cp
    w = cp.Variable(v.size)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(w - v)),
        [cp.sum(w) == 1, cp.sum(cp.pos(-w)) <= s])
    problem.solve(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return np.asarray(w.value).ravel()


def lp_minimum_dot(g: np.ndarray, s: float) -> float:
    import cvxpy as cp
    w = cp.Variable(g.size)
    problem = cp.Problem(cp.Minimize(g @ w),
                         [cp.sum(w) == 1, cp.sum(cp.pos(-w)) <= s])
    problem.solve(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return float(problem.value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    core = load_core()
    rng = np.random.default_rng(20261006)
    bisect_err, lp_err, bound_err = [], [], []
    binding = 0

    for _ in range(args.trials):
        n = int(rng.integers(2, 16))
        s = float(rng.choice([0.01, 0.05, 0.10, 0.25, 0.5, 1.0]))
        v = rng.normal(0, 1, size=n) * float(rng.choice([0.05, 0.5, 3.0]))

        got = core.project_gross_budget(v, s)
        if np.maximum(-(v - (v.sum() - 1.0) / n), 0.0).sum() > s + 1e-12:
            binding += 1

        # Independent closed forms, then the LP as a third opinion.
        bisect_err.append(float(np.max(np.abs(got - reference_projection_vectorised(v, s)))))
        lp_err.append(float(np.max(np.abs(got - lp_projection(v, s)))))

        # Feasibility of our own answer, checked without reference to anything.
        assert abs(float(got.sum()) - 1.0) < 1e-9, f"budget {got.sum()!r}"
        assert float(np.maximum(-got, 0.0).sum()) <= s + 1e-10, "short budget broken"

        # The gap bound: exact value against the LP, then the lower-bound
        # property at random feasible points.
        g = rng.normal(0, 1, size=n)
        closed = (float(np.min(g)) + s * (float(np.min(g)) - float(np.max(g))))
        bound_err.append(abs(closed - lp_minimum_dot(g, s)))
        for _ in range(20):
            probe = core.project_gross_budget(rng.normal(0, 1, size=n), s)
            assert float(g @ probe) >= closed - 1e-8, "bound is not a lower bound"

    worst_bisect = max(bisect_err)
    worst_lp = max(lp_err)
    worst_bound = max(bound_err)
    print(f"trials                            : {args.trials}  (constraint binding in {binding})")
    print(f"max ||ours - kkt bisection||_inf  : {worst_bisect:.3e}")
    print(f"max ||ours - cvxpy||_inf          : {worst_lp:.3e}   (solver tolerance)")
    print(f"max |bound - LP value|            : {worst_bound:.3e}   (solver tolerance)")
    ok = worst_bisect < 1e-9 and worst_lp < 1e-5 and worst_bound < 1e-6
    print(f"\nROUND-3 GROSS-BUDGET PROJECTION & GAP: {'VERIFIED' if ok else 'FAILED'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "trials": args.trials,
            "binding_trials": binding,
            "max_projection_error_vs_kkt_bisection": worst_bisect,
            "max_projection_error_vs_cvxpy": worst_lp,
            "max_gap_bound_error_vs_lp": worst_bound,
            "verified": bool(ok),
        }, indent=2), encoding="utf-8")
        print(f"written -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
