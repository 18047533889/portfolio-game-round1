"""Would capping concentration lift Sharpe?

Everything measured so far points one way: the equal-weight book is strong on
this data, and the minimum-variance family is its opposite extreme. The paired
test says raising the anchor penalty only slides along the frontier (return up,
drawdown up, Sharpe flat).

The one mechanism not yet tested is concentration *itself*. Minimum variance
piles weight onto whichever assets happen to have the lowest estimated variance,
and in a 252-row sample a good part of that ranking is estimation noise. Capping
each weight is the standard defence, and it is the same idea as the classic
result that 1/N is hard to beat out of sample.

So: solve the same objective with 0 <= w_i <= cap, using SLSQP as an independent
solver (the shipped projection-gradient code cannot take the extra constraint
without a rewrite, and this is a probe, not a candidate). The no-cap run is also
solved with SLSQP so the comparison is solver-for-solver fair.

Cap is expressed as a multiple of equal weight, so it adapts to n. A cap of
1.0/n *is* equal weight and is prohibited; anything above that is legal.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "portfolio_game" / "core.py"

TRAIN = 252
STEP = 20
ANNUALIZATION = 252
HALF_LIFE = 63.0
RECENT_MIX = 0.25
PENALTY = 0.25

DATASETS = {
    "sp500(20)": "load_sp500_dataset",
    "ftse100(64)": "load_ftse100_dataset",
    "factors(5)": "load_factors_dataset",
}

CAP_MULTIPLES = [1.5, 2.0, 3.0, 5.0, None]


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("core_cap", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def returns_frame(name: str) -> pd.DataFrame:
    import skfolio.datasets as ds

    prices = getattr(ds, DATASETS[name])()
    return prices.pct_change().dropna()


def metrics(returns: np.ndarray) -> dict:
    r = np.asarray(returns, dtype=float)
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    cum = np.cumsum(r)
    return {
        "annualized_mean": float(np.mean(r) * ANNUALIZATION),
        "annualized_sharpe": float(np.mean(r) / sd * math.sqrt(ANNUALIZATION))
        if sd > 1e-15 else 0.0,
        "max_drawdown": float(np.max(np.maximum.accumulate(cum) - cum)),
    }


def solve_capped(core, cov: np.ndarray, anchor: np.ndarray, penalty: float,
                 cap_multiple: float | None) -> np.ndarray | None:
    n = len(anchor)
    if cap_multiple is None:
        bounds = [(0.0, 1.0)] * n
    else:
        cap = min(1.0, cap_multiple / n)
        bounds = [(0.0, cap)] * n
    objective = lambda w: core.relative_objective(w, cov, anchor, penalty)  # noqa: E731
    best = None
    for start in (anchor, np.full(n, 1.0 / n)):
        try:
            res = minimize(objective, start, method="SLSQP", bounds=bounds,
                           constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                           options={"maxiter": 300, "ftol": 1e-12})
        except Exception:  # noqa: BLE001
            continue
        if not res.success or not np.isfinite(res.x).all():
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return None
    w = np.clip(best.x, 0.0, None)
    if w.sum() <= 0 or abs(w.sum() - 1.0) > 1e-6:
        return None
    return w / w.sum()


def walk_forward(core, X: pd.DataFrame, cap_multiple: float | None) -> dict:
    values = X.to_numpy(dtype=float)
    n_obs, n_assets = X.shape
    chunks: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    failed = 0
    for pos in range(TRAIN, n_obs - STEP + 1, STEP):
        train = values[pos - TRAIN:pos]
        try:
            cov = core.estimate_covariance(train, HALF_LIFE, RECENT_MIX)
            anchor = core.hrp_weights(cov)
            w = solve_capped(core, cov, anchor, PENALTY, cap_multiple)
            if w is None:
                raise ValueError("solver")
            if n_assets > 1 and np.allclose(w, 1.0 / n_assets, atol=1e-9, rtol=1e-6):
                raise ValueError("equal weight")
        except BaseException:  # noqa: BLE001
            failed += 1
            continue
        weights.append(w)
        chunks.append(values[pos:pos + STEP] @ w)

    ref = np.concatenate([values[pos:pos + STEP].mean(axis=1)
                          for pos in range(TRAIN, n_obs - STEP + 1, STEP)])
    out = metrics(np.concatenate(chunks)) if chunks else None
    return {
        "metrics": out,
        "reference": metrics(ref),
        "failed_folds": failed,
        "n_folds": len(chunks) + failed,
        "mean_max_weight": float(np.mean([w.max() for w in weights])) if weights else float("nan"),
        "mean_effective_n": float(np.mean([1.0 / (w ** 2).sum() for w in weights]))
        if weights else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="sp500(20),factors(5),ftse100(64)")
    ap.add_argument("--out", default="reports/cap_probe.json")
    args = ap.parse_args()

    core = load_core(CORE)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    report: dict = {"penalty": PENALTY, "cap_multiples": CAP_MULTIPLES, "datasets": {}}

    for name in names:
        frame = returns_frame(name)
        n_assets = frame.shape[1]
        print(f"[{name}] equal weight = {1.0 / n_assets:.4f}", flush=True)
        print(f"  {'cap':>10s} {'sharpe':>8s} {'annual':>9s} {'maxdd':>8s} "
              f"{'fail':>8s} {'meanMaxW':>9s} {'effN':>6s} {'ewSharpe':>9s}", flush=True)
        rows = []
        for mult in CAP_MULTIPLES:
            res = walk_forward(core, frame, mult)
            m = res["metrics"]
            label = "none" if mult is None else f"{mult:g}/n"
            row = {
                "cap": label,
                "cap_multiple": mult,
                "cap_value": None if mult is None else mult / n_assets,
                "sharpe": m["annualized_sharpe"] if m else None,
                "annual": m["annualized_mean"] if m else None,
                "maxdd": m["max_drawdown"] if m else None,
                "failed_folds": res["failed_folds"],
                "n_folds": res["n_folds"],
                "mean_max_weight": res["mean_max_weight"],
                "mean_effective_n": res["mean_effective_n"],
                "reference_sharpe": res["reference"]["annualized_sharpe"],
                "reference_annual": res["reference"]["annualized_mean"],
                "reference_maxdd": res["reference"]["max_drawdown"],
            }
            rows.append(row)
            if m is None:
                print(f"  {label:>10s}  ALL FAILED", flush=True)
                continue
            print(f"  {label:>10s} {row['sharpe']:8.4f} {row['annual']:9.4%} "
                  f"{row['maxdd']:8.4%} {row['failed_folds']:4d}/{row['n_folds']:<4d} "
                  f"{row['mean_max_weight']:9.4f} {row['mean_effective_n']:6.2f} "
                  f"{row['reference_sharpe']:9.4f}", flush=True)
        report["datasets"][name] = {"n_assets": int(n_assets), "rows": rows}
        print(flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"written -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
