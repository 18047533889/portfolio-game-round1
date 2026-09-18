"""Grader-shaped stress test: random asset subsets x random 2-year windows.

Why this exists
---------------
The grader is believed to hand us a *random* subset of names and a *random*
two-year slice of history (possibly very early, e.g. 1990s for the sp500 set)
and then rebalance with the round-1 contract:
``shortselling=False, leverage=1, lookback=252, optimize_every=20``.

Two things therefore have to be true that a single full-history backtest does
NOT establish:

1. no fold may fail (70% of the score) on *any* subset/period combination;
2. no fold may ever return the prohibited equal-weight portfolio, including on
   degenerate subsets (k = 1, k = 2, duplicated or near-constant columns).

This script enumerates that space the way the grader would, checks the exact
contract per fold, and reports counts rather than averages. Everything is
seeded, so a reported failure is reproducible.

Fold accounting follows skfolio's WalkForward, which drops incomplete tail
windows: ``for pos in range(train, n_obs - step + 1, step)``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission" / "portfolio_round1.py"

TRAIN = 252
STEP = 20
EQUAL_ATOL = 1e-9

DATASETS = {
    "sp500(20)": "load_sp500_dataset",
    "ftse100(64)": "load_ftse100_dataset",
    "nasdaq(1455)": "load_nasdaq_dataset",
    "factors(5)": "load_factors_dataset",
    "sp500_index(1)": "load_sp500_index",
}


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("ship_stress", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def returns_frame(name: str) -> pd.DataFrame:
    import skfolio.datasets as ds

    prices = getattr(ds, DATASETS[name])()
    return prices.pct_change().dropna()


def fold_positions(n_obs: int, train: int = TRAIN, step: int = STEP):
    """Positions the grader's WalkForward actually evaluates."""
    return list(range(train, n_obs - step + 1, step))


def check_weights(w, n_assets: int) -> list[str]:
    problems: list[str] = []
    if not isinstance(w, np.ndarray):
        return [f"weights_ is {type(w).__name__}, not ndarray"]
    if w.shape != (n_assets,):
        return [f"weights_ shape {w.shape} != ({n_assets},)"]
    if not np.isfinite(w).all():
        problems.append("weights_ not finite")
        return problems
    if (w < -1e-12).any():
        problems.append("negative weight (short)")
    if abs(float(w.sum()) - 1.0) > 1e-9:
        problems.append(f"sum={w.sum():.12f} != 1")
    if n_assets > 1 and np.allclose(w, 1.0 / n_assets, atol=EQUAL_ATOL, rtol=1e-6):
        problems.append("EQUAL WEIGHT (prohibited)")
    return problems


def fit_once(cls, train_block: np.ndarray, n_assets: int, *, silence: bool = True):
    """Exactly how the grader instantiates and calls us."""
    model = cls(portfolio_params={"name": "CVXPYPortfolio"})
    model.raise_on_failure = False
    with warnings.catch_warnings():
        if silence:
            warnings.simplefilter("ignore")
        model.fit(pd.DataFrame(train_block))
    if getattr(model, "error_", None):
        return None, f"error_={model.error_}"
    w = getattr(model, "weights_", None)
    if w is None:
        return None, "weights_ is None"
    return np.asarray(w, dtype=float), None


def evaluate_slice(cls, block: pd.DataFrame, record_prefix: str) -> dict:
    """Run every fold the grader would run on one subset/period, contract-checked."""
    x = block.to_numpy(dtype=float)
    n_obs, n_assets = x.shape
    positions = fold_positions(n_obs)
    out = {
        "prefix": record_prefix,
        "n_assets": n_assets,
        "n_obs": n_obs,
        "n_folds": len(positions),
        "failed": 0,
        "equal_weight": 0,
        "illegal": 0,
        "exceptions": [],
        "reasons": {},
    }
    for pos in positions:
        train_block = x[pos - TRAIN:pos]
        try:
            w, err = fit_once(cls, train_block, n_assets)
        except BaseException as exc:  # noqa: BLE001 - grader would see this too
            out["failed"] += 1
            out["exceptions"].append(f"{type(exc).__name__}: {str(exc)[:120]}")
            continue
        if err is not None or w is None:
            out["failed"] += 1
            out["reasons"][err or "unknown"] = out["reasons"].get(err or "unknown", 0) + 1
            continue
        problems = check_weights(w, n_assets)
        if problems:
            out["illegal"] += 1
            for p in problems:
                out["reasons"][p] = out["reasons"].get(p, 0) + 1
            if any("EQUAL WEIGHT" in p for p in problems):
                out["equal_weight"] += 1
    return out


def subset_sizes(rng: np.random.Generator, n_total: int) -> int:
    """Deliberately over-sample the small/extreme sizes the grader could pick."""
    menu = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 45, 64, 100, 160, 256]
    menu = [k for k in menu if k <= n_total]
    if not menu:
        menu = [n_total]
    weights = np.array([3.0 if k <= 5 else 1.0 for k in menu], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(menu, p=weights))


def run_dataset(cls, name: str, reps: int, window: int, seed: int) -> dict:
    frame = returns_frame(name)
    n_obs_total, n_total = frame.shape
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # 1) random subset x random window -- the reported scenario
    for i in range(reps):
        k = subset_sizes(rng, n_total)
        cols = rng.choice(n_total, size=k, replace=False)
        if n_obs_total > window:
            start = int(rng.integers(0, n_obs_total - window + 1))
        else:
            start = 0
        block = frame.iloc[start:start + window, cols]
        rows.append(evaluate_slice(cls, block, f"rand#{i}@{frame.index[start].date()}k{k}"))

    # 2) exhaustive early-history sweep: every window start in the first 6 years,
    #    at the full cross-section and at a small random one. Very early years are
    #    exactly where the grader's "random two years" is most likely to land.
    early_span = min(n_obs_total, 6 * 252)
    starts = list(range(0, max(1, early_span - window + 1), max(1, window // 4)))
    for si, start in enumerate(starts[:24]):
        for label, cols in (
            ("all", np.arange(n_total)),
            ("small", rng.choice(n_total, size=min(3, n_total), replace=False)),
        ):
            block = frame.iloc[start:start + window, cols]
            rows.append(evaluate_slice(cls, block, f"early#{si}{label}@{frame.index[start].date()}"))

    # 3) degenerate subsets the grader could still produce by accident
    for label, block in degenerate_blocks(frame, window, n_total):
        rows.append(evaluate_slice(cls, block, label))

    failed = sum(r["failed"] for r in rows)
    folds = sum(r["n_folds"] for r in rows)
    equal = sum(r["equal_weight"] for r in rows)
    illegal = sum(r["illegal"] for r in rows)
    worst = sorted(rows, key=lambda r: (-r["failed"], -r["illegal"], -r["equal_weight"]))[:5]
    return {
        "dataset": name,
        "n_obs_total": int(n_obs_total),
        "n_assets_total": int(n_total),
        "window": window,
        "reps": reps,
        "slices": len(rows),
        "folds": folds,
        "failed_folds": failed,
        "illegal_folds": illegal,
        "equal_weight_folds": equal,
        "failure_rate": failed / folds if folds else float("nan"),
        "worst_slices": worst,
    }


def degenerate_blocks(frame: pd.DataFrame, window: int, n_total: int):
    """Subsets that break naive covariance code without breaking the contract."""
    x = frame.to_numpy(dtype=float)
    n = len(x)
    start = min(n - window, max(0, n - window))
    block = x[max(0, start):max(0, start) + window]
    if block.shape[0] < TRAIN + STEP:
        block = x[:window]
    yield "deg:2identical", pd.DataFrame(np.repeat(block[:, :1], 2, axis=1))
    yield "deg:3identical", pd.DataFrame(np.repeat(block[:, :1], 3, axis=1))
    a = block[:, 0]
    b = block[:, 1] if block.shape[1] > 1 else block[:, 0]
    # two names, strongly correlated but not identical
    yield "deg:2corr", pd.DataFrame(np.column_stack([a, 0.999 * a + 1e-6 * b]))
    # many near-collinear live names (rank-deficient covariance)
    if block.shape[1] > 1:
        basis = block[:, : min(3, block.shape[1])]
        jitter = np.random.default_rng(7).normal(0, 1e-7, size=(len(basis), 24))
        wide = basis @ np.random.default_rng(11).normal(0, 1, size=(basis.shape[1], 24)) + jitter
        yield "deg:24collinear", pd.DataFrame(wide)
    # constant column next to a live one
    yield "deg:constant+live", pd.DataFrame(np.column_stack([np.full(len(block), 1e-4), a]))
    # duplicated block with a scaled copy (perfect collinearity, different scale)
    yield "deg:scaled-copy", pd.DataFrame(np.column_stack([a, a * 1e3, b]))
    # one live asset and 40 dead ones
    dead = np.full((len(block), 40), np.nan)
    yield "deg:1live40nan", pd.DataFrame(np.column_stack([a.reshape(-1, 1), dead]))
    # heavy outlier column
    out = block.copy()
    out[-1, 0] = 5.0
    yield "deg:outlier", pd.DataFrame(out[:, :3])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=60, help="random subset/period draws per dataset")
    ap.add_argument("--window", type=int, default=504, help="trading days per slice (504 ~ 2 years)")
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--datasets", default="sp500(20),ftse100(64),factors(5),sp500_index(1)")
    ap.add_argument("--wide-reps", type=int, default=6, help="reps for the 1455-column dataset")
    ap.add_argument("--out", default="reports/random_windows.json")
    args = ap.parse_args()

    module = load_submission(SUBMISSION)
    cls = module.CVXPYPortfolio

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    results = []
    for name in names:
        reps = args.wide_reps if name == "nasdaq(1455)" else args.reps
        res = run_dataset(cls, name, reps, args.window, args.seed)
        results.append(res)
        print(
            f"{name:16s} slices={res['slices']:4d} folds={res['folds']:5d} "
            f"failed={res['failed_folds']:3d} illegal={res['illegal_folds']:3d} "
            f"equal={res['equal_weight_folds']:3d}",
            flush=True,
        )

    total_folds = sum(r["folds"] for r in results)
    total_failed = sum(r["failed_folds"] for r in results)
    total_equal = sum(r["equal_weight_folds"] for r in results)
    total_illegal = sum(r["illegal_folds"] for r in results)
    report = {
        "submission": str(SUBMISSION.relative_to(ROOT)),
        "window_trading_days": args.window,
        "seed": args.seed,
        "contract": {"train_size": TRAIN, "test_size": STEP, "long_only": True, "leverage": 1},
        "datasets": results,
        "total": {
            "folds": total_folds,
            "failed_folds": total_failed,
            "illegal_folds": total_illegal,
            "equal_weight_folds": total_equal,
            "failure_rate": total_failed / total_folds if total_folds else float("nan"),
        },
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== TOTAL ===")
    print(f"folds           : {total_folds}")
    print(f"failed folds    : {total_failed}  ({report['total']['failure_rate']:.6%})")
    print(f"illegal folds   : {total_illegal}")
    print(f"equal-wt folds  : {total_equal}")
    for res in results:
        f = res["worst_slices"][0]
        print(f"  worst in {res['dataset']}: {f['prefix']} failed={f['failed']}/{f['n_folds']} "
              f"reasons={f['reasons']}")
    print(f"\nwritten -> {out.relative_to(ROOT)}")
    return 1 if (total_failed or total_equal or total_illegal) else 0


if __name__ == "__main__":
    raise SystemExit(main())
