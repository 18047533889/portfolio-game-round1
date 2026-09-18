"""Sweep the covariance-estimator dials.

`anchor_penalty` turned out to be a risk-appetite dial, not an efficiency dial:
raising it lifts Sharpe on sp500 and *lowers* it on ftse100, because it only
slides the portfolio along the frontier. Nothing there makes the frontier
itself better.

The dials that can actually move the frontier are the ones inside
`estimate_covariance`: `half_life` (how fast the EWMA volatility layer forgets)
and `recent_mix` (how much weight the near-term risk estimate gets against the
Ledoit-Wolf long covariance). Neither has ever been fitted -- 63 and 0.25 are
plausible defaults, not measured choices.

Selection rule, fixed before looking at anything: a candidate has to improve
Sharpe on *every* dataset, not on average. Optimising a multi-dataset mean is
how a single-period fluke gets shipped; requiring unanimity is how it does not.
Drawdown is reported but is not part of the acceptance test, because it is
already the metric the shipped penalty was chosen for.

Metrics match the grading convention (arithmetic drawdown on cumsum).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "portfolio_game" / "core.py"

TRAIN = 252
STEP = 20
ANNUALIZATION = 252
SHIPPED = {"half_life": 63.0, "recent_mix": 0.25, "anchor_penalty": 0.25}

DATASETS = {
    "sp500(20)": "load_sp500_dataset",
    "ftse100(64)": "load_ftse100_dataset",
    "factors(5)": "load_factors_dataset",
}


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("core_est", path)
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
        if sd > 1e-15
        else None,
        "max_drawdown": float(np.max(np.maximum.accumulate(cum) - cum)),
    }


def walk_forward(core, X: pd.DataFrame, *, half_life: float, recent_mix: float,
                 anchor_penalty: float) -> dict:
    values = X.to_numpy(dtype=float)
    n_obs, n_assets = X.shape
    chunks: list[np.ndarray] = []
    failed = 0
    for pos in range(TRAIN, n_obs - STEP + 1, STEP):
        try:
            result = core.allocate(
                values[pos - TRAIN:pos],
                half_life=half_life, recent_mix=recent_mix,
                anchor_penalty=anchor_penalty, method="regularized",
            )
            w = np.asarray(result["weights"], dtype=float)
            if w.shape != (n_assets,) or not np.isfinite(w).all():
                raise ValueError("shape/finite")
            if (w < -1e-12).any() or abs(float(w.sum()) - 1.0) > 1e-9:
                raise ValueError("floor/sum")
            if n_assets > 1 and np.allclose(w, 1.0 / n_assets, atol=1e-9, rtol=1e-6):
                raise ValueError("equal weight")
        except BaseException:  # noqa: BLE001
            failed += 1
            continue
        chunks.append(values[pos:pos + STEP] @ w)

    ref = np.concatenate([values[pos:pos + STEP].mean(axis=1)
                          for pos in range(TRAIN, n_obs - STEP + 1, STEP)])
    return {
        "metrics": metrics(np.concatenate(chunks)) if chunks else None,
        "reference": metrics(ref),
        "failed_folds": failed,
        "n_folds": len(chunks) + failed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="sp500(20),ftse100(64),factors(5)")
    ap.add_argument("--half-lives", default="21,63,126,252")
    ap.add_argument("--recent-mixes", default="0,0.25,0.5,0.75")
    ap.add_argument("--anchor-penalty", type=float, default=SHIPPED["anchor_penalty"])
    ap.add_argument("--out", default="reports/estimator_sweep.json")
    args = ap.parse_args()

    core = load_core(CORE)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    half_lives = [float(v) for v in args.half_lives.split(",")]
    mixes = [float(v) for v in args.recent_mixes.split(",")]

    report: dict = {
        "anchor_penalty": args.anchor_penalty,
        "shipped": SHIPPED,
        "grid": {"half_life": half_lives, "recent_mix": mixes},
        "datasets": {},
    }
    grid_results: dict[tuple[float, float], dict[str, dict]] = {}

    for name in names:
        frame = returns_frame(name)
        print(f"[{name}] {frame.shape[0]} obs x {frame.shape[1]} assets", flush=True)
        per_cell: dict[str, dict] = {}
        print(f"  {'half_life':>9s} {'recent_mix':>10s} {'sharpe':>8s} {'annual':>9s} "
              f"{'maxdd':>8s} {'fail':>8s} {'ew_sharpe':>10s}", flush=True)
        for hl in half_lives:
            for rm in mixes:
                res = walk_forward(core, frame, half_life=hl, recent_mix=rm,
                                   anchor_penalty=args.anchor_penalty)
                m = res["metrics"]
                cell = {
                    "half_life": hl,
                    "recent_mix": rm,
                    "sharpe": m["annualized_sharpe"] if m else None,
                    "annual": m["annualized_mean"] if m else None,
                    "maxdd": m["max_drawdown"] if m else None,
                    "failed_folds": res["failed_folds"],
                    "n_folds": res["n_folds"],
                    "reference_sharpe": res["reference"]["annualized_sharpe"],
                    "reference_annual": res["reference"]["annualized_mean"],
                    "reference_maxdd": res["reference"]["max_drawdown"],
                }
                per_cell[f"{hl:g}|{rm:g}"] = cell
                grid_results.setdefault((hl, rm), {})[name] = cell
                flag = "  <- shipped" if (hl == SHIPPED["half_life"]
                                          and rm == SHIPPED["recent_mix"]) else ""
                print(f"  {hl:9g} {rm:10g} {cell['sharpe']:8.4f} "
                      f"{cell['annual']:9.4%} {cell['maxdd']:8.4%} "
                      f"{cell['failed_folds']:4d}/{cell['n_folds']:<4d}"
                      f"{cell['reference_sharpe']:10.4f}{flag}", flush=True)
        report["datasets"][name] = per_cell
        print(flush=True)

    print("=" * 78)
    print("cross-dataset unanimity test (Sharpe must beat shipped on EVERY dataset)")
    shipped_key = (SHIPPED["half_life"], SHIPPED["recent_mix"])
    baseline = grid_results[shipped_key]
    survivors = []
    for key, per_ds in sorted(grid_results.items()):
        if key == shipped_key:
            continue
        deltas = {n: per_ds[n]["sharpe"] - baseline[n]["sharpe"] for n in baseline}
        if all(d > 0 for d in deltas.values()):
            survivors.append((sum(deltas.values()), key, deltas))
    if not survivors:
        print("  no configuration beats the shipped one on every dataset.")
        best_avg = max(
            ((sum((per_ds[n]["sharpe"] - baseline[n]["sharpe"]) for n in baseline) / len(baseline),
              key, {n: per_ds[n]["sharpe"] - baseline[n]["sharpe"] for n in baseline})
             for key, per_ds in grid_results.items() if key != shipped_key),
            key=lambda t: t[0],
        )
        print(f"  best on average: half_life={best_avg[1][0]:g} recent_mix={best_avg[1][1]:g} "
              f"mean delta {best_avg[0]:+.4f}")
        for n, d in best_avg[2].items():
            print(f"    {n:14s} delta sharpe {d:+.4f}")
    else:
        for total, key, deltas in sorted(survivors, key=lambda t: -t[0]):
            print(f"  half_life={key[0]:g} recent_mix={key[1]:g} "
                  f"sum delta {total:+.4f}  " +
                  " ".join(f"{n.split('(')[0]}:{d:+.4f}" for n, d in sorted(deltas.items())))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
