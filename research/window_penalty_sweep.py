"""Does raising the anchor penalty actually help under the grader's own shape?

`research/anchor_sweep.py` sweeps the dial on full history and shows Sharpe and
return rising monotonically with p, drawdown rising with them. But full history
is one path, dominated by the 2000 and 2008 bears, and the grader does not score
one path -- the grader draws a random subset and a random two-year slice.

So this script re-runs the same sweep inside that shape: every configuration is
evaluated on *the same* set of drawn blocks (same seed, same subsets, same
windows), and each block also produces an equal-weight book over the identical
out-of-sample span. Reported per configuration:

    sharpe_win / return_win / drawdown_win   -- win rate against equal weight,
                                                which is the only reference we
                                                can compute that has no estimation
                                                error of its own;
    median sharpe / annual / maxdd           -- absolute level, so a
                                                configuration cannot look good
                                                merely by beating a weak field.

Note equal weight is a *reference* here, not a candidate: it is prohibited in
this round. It is the right yardstick precisely because it is unestimatable-wrong.

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
WINDOW = 504
ANNUALIZATION = 252

DATASETS = {
    "sp500(20)": "load_sp500_dataset",
    "ftse100(64)": "load_ftse100_dataset",
    "factors(5)": "load_factors_dataset",
    "sp500_index(1)": "load_sp500_index",
}

CONFIGS: list[tuple[str, float, str]] = [
    ("regularized", 0.0, "minimum_variance(p=0)"),
    ("regularized", 0.25, "regularized(p=0.25) [shipped]"),
    ("regularized", 1.0, "regularized(p=1)"),
    ("regularized", 4.0, "regularized(p=4)"),
    ("regularized", 16.0, "regularized(p=16)"),
    ("hrp", 0.0, "hrp_anchor(p=inf)"),
]

SIZE_MENU = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 45, 64]


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("core_wsweep", path)
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


def draw_blocks(frame: pd.DataFrame, reps: int, rng: np.random.Generator,
                early_fraction: float) -> list[tuple[np.ndarray, int]]:
    """(column indices, start position) pairs, biased small and partly early."""
    n_obs, n_total = frame.shape
    span = n_obs - WINDOW + 1
    if span <= 0:
        return []
    menu = [k for k in SIZE_MENU if k <= n_total] or [n_total]
    weights = np.array([3.0 if k <= 5 else 1.0 for k in menu], dtype=float)
    weights /= weights.sum()

    blocks = []
    for _ in range(reps):
        k = int(rng.choice(menu, p=weights))
        cols = np.sort(rng.choice(n_total, size=k, replace=False))
        hi = max(1, int(early_fraction * span))
        start = int(rng.integers(0, hi))
        blocks.append((cols, start))
    return blocks


def evaluate(core, frame: pd.DataFrame, cols: np.ndarray, start: int, *,
             method: str, penalty: float) -> dict:
    block = frame.iloc[start:start + WINDOW, cols]
    x = block.to_numpy(dtype=float)
    n_obs, n_assets = x.shape
    chunks: list[np.ndarray] = []
    failed = 0
    for pos in range(TRAIN, n_obs - STEP + 1, STEP):
        try:
            result = core.allocate(x[pos - TRAIN:pos], anchor_penalty=penalty, method=method)
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
        chunks.append(x[pos:pos + STEP] @ w)

    ref_chunks = [x[pos:pos + STEP].mean(axis=1)
                  for pos in range(TRAIN, n_obs - STEP + 1, STEP)]
    out = {
        "n_assets": int(n_assets),
        "failed": failed,
        "n_folds": len(ref_chunks),
        "reference": metrics(np.concatenate(ref_chunks)),
        "portfolio": metrics(np.concatenate(chunks)) if chunks else None,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="sp500(20),factors(5),ftse100(64)")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--early-fraction", type=float, default=0.45,
                    help="share of draws whose window start is forced into the early history")
    ap.add_argument("--out", default="reports/window_penalty_sweep.json")
    args = ap.parse_args()

    core = load_core(CORE)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    report: dict = {
        "contract": {"train_size": TRAIN, "test_size": STEP, "window": WINDOW},
        "reps": args.reps, "seed": args.seed, "datasets": {},
    }

    for name in names:
        frame = returns_frame(name)
        rng = np.random.default_rng(args.seed)
        blocks = draw_blocks(frame, args.reps, rng, args.early_fraction)
        print(f"[{name}] {len(blocks)} drawn blocks", flush=True)

        per_config: dict[str, list[dict]] = {c[2]: [] for c in CONFIGS}
        for cols, start in blocks:
            for method, penalty, lab in CONFIGS:
                per_config[lab].append(
                    evaluate(core, frame, cols, start, method=method, penalty=penalty)
                )

        rows = []
        print(f"  {'config':30s} {'ShWin':>6s} {'RetWin':>7s} {'DdWin':>6s} "
              f"{'medSh':>7s} {'medRet':>8s} {'medDd':>7s} {'fail':>6s}", flush=True)
        for _, _, lab in CONFIGS:
            recs = per_config[lab]
            usable = [r for r in recs if r["portfolio"] is not None]
            sh = sum(1 for r in usable
                     if (r["portfolio"]["annualized_sharpe"] or -9)
                     > (r["reference"]["annualized_sharpe"] or -9))
            rt = sum(1 for r in usable
                     if r["portfolio"]["annualized_mean"] > r["reference"]["annualized_mean"])
            dd = sum(1 for r in usable
                     if r["portfolio"]["max_drawdown"] < r["reference"]["max_drawdown"])
            n = len(usable) or 1
            row = {
                "config": lab,
                "usable_blocks": len(usable),
                "failed_folds": sum(r["failed"] for r in recs),
                "total_folds": sum(r["n_folds"] for r in recs),
                "sharpe_win": sh / n,
                "return_win": rt / n,
                "drawdown_win": dd / n,
                "median_sharpe": float(np.median(
                    [r["portfolio"]["annualized_sharpe"] for r in usable])),
                "median_return": float(np.median(
                    [r["portfolio"]["annualized_mean"] for r in usable])),
                "median_drawdown": float(np.median(
                    [r["portfolio"]["max_drawdown"] for r in usable])),
                "reference_median_sharpe": float(np.median(
                    [r["reference"]["annualized_sharpe"] for r in usable])),
                "reference_median_return": float(np.median(
                    [r["reference"]["annualized_mean"] for r in usable])),
                "reference_median_drawdown": float(np.median(
                    [r["reference"]["max_drawdown"] for r in usable])),
            }
            rows.append(row)
            print(f"  {lab:30s} {row['sharpe_win']:6.1%} {row['return_win']:7.1%} "
                  f"{row['drawdown_win']:6.1%} {row['median_sharpe']:7.4f} "
                  f"{row['median_return']:8.4%} {row['median_drawdown']:7.4%} "
                  f"{row['failed_folds']:3d}/{row['total_folds']:3d}", flush=True)

        report["datasets"][name] = {
            "n_obs": int(frame.shape[0]),
            "n_assets": int(frame.shape[1]),
            "n_blocks": len(blocks),
            "rows": rows,
        }
        print(flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"written -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
