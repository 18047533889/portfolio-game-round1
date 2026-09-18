"""Sweep the risk/return dial.

The submission solves

    min  w'Cw / (b'Cb)  +  p * ||w - b||^2 / ||b||^2      s.t. 1'w = 1, w >= 0

with `b` the HRP anchor. At p = 0 the solution is the pure minimum-variance
portfolio: lowest volatility, lowest return, most concentrated. As p grows the
anchor term dominates and the output converges to HRP: more diversified, more
beta, higher return.

The shipped value is p = 0.25, which was chosen on a *drawdown* criterion
(8 of 9 evaluation blocks improved). Nothing in that choice optimises Sharpe or
return, so this sweep measures what the dial actually does to all three graded
metrics before anyone moves it.

Metric conventions match the grader (arithmetic drawdown on cumsum, not a
geometric net-value curve) and match tools/grader_sample_report.py.
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

DATASETS = {
    "sp500(20)": "load_sp500_dataset",
    "ftse100(64)": "load_ftse100_dataset",
    "factors(5)": "load_factors_dataset",
    "sp500_index(1)": "load_sp500_index",
}


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("core_sweep", path)
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


def walk_forward(core, X: pd.DataFrame, *, penalty: float, method: str) -> dict:
    values = X.to_numpy(dtype=float)
    n_obs, n_assets = X.shape
    chunks: list[np.ndarray] = []
    failures: list[str] = []
    max_weight = 0.0
    for pos in range(TRAIN, n_obs - STEP + 1, STEP):
        train = values[pos - TRAIN:pos]
        try:
            result = core.allocate(train, anchor_penalty=penalty, method=method)
            w = np.asarray(result["weights"], dtype=float)
            if w.shape != (n_assets,):
                raise ValueError(f"shape {w.shape}")
            if not np.isfinite(w).all():
                raise ValueError("non-finite")
            if (w < -1e-12).any():
                raise ValueError("negative")
            if abs(float(w.sum()) - 1.0) > 1e-9:
                raise ValueError(f"sum {w.sum()}")
            if n_assets > 1 and np.allclose(w, 1.0 / n_assets, atol=1e-9, rtol=1e-6):
                raise ValueError("equal weight")
        except BaseException as exc:  # noqa: BLE001
            failures.append(f"{type(exc).__name__}: {str(exc)[:70]}")
            continue
        max_weight = max(max_weight, float(np.max(w)))
        chunks.append(values[pos:pos + STEP] @ w)

    if not chunks:
        return {"metrics": None, "failed_folds": len(failures), "n_folds": len(failures),
                "max_weight": float("nan"), "failures": failures[:3]}
    return {
        "metrics": metrics(np.concatenate(chunks)),
        "failed_folds": len(failures),
        "n_folds": len(chunks) + len(failures),
        "max_weight": max_weight,
        "failures": failures[:3],
    }


def walk_forward_equal_weight(X: pd.DataFrame) -> dict:
    values = X.to_numpy(dtype=float)
    n_obs = len(values)
    chunks = [values[pos:pos + STEP].mean(axis=1)
              for pos in range(TRAIN, n_obs - STEP + 1, STEP)]
    return {
        "metrics": metrics(np.concatenate(chunks)),
        "failed_folds": 0,
        "n_folds": len(chunks),
        "max_weight": float("nan"),
        "failures": [],
    }


def label(args) -> str:
    if args.method != "regularized":
        return args.method
    if args.penalty == 0.0:
        return "minimum_variance(p=0)"
    return f"regularized(p={args.penalty:g})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="sp500(20),ftse100(64),factors(5)")
    ap.add_argument("--penalties", default="0,0.1,0.25,0.5,1,2,4,8,16,1e9")
    ap.add_argument("--out", default="reports/anchor_sweep.json")
    args = ap.parse_args()

    core = load_core(CORE)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    penalties = [float(p) for p in args.penalties.split(",")]

    configs = [{"method": "hrp", "penalty": 0.0}]
    configs.append({"method": "inverse_volatility", "penalty": 0.0})
    configs += [{"method": "regularized", "penalty": p} for p in penalties]

    report: dict = {
        "contract": {"train_size": TRAIN, "test_size": STEP},
        "shipped_penalty": 0.25,
        "datasets": {},
    }

    for name in names:
        frame = returns_frame(name)
        rows: list[dict] = []

        ew = walk_forward_equal_weight(frame)
        rows.append({"model": "reference_equal_weight", **ew})

        for cfg in configs:
            ns = argparse.Namespace(method=cfg["method"], penalty=cfg["penalty"])
            res = walk_forward(core, frame, penalty=cfg["penalty"], method=cfg["method"])
            rows.append({"model": label(ns), **res})
            m = res["metrics"]
            if m is None:
                print(f"  {label(ns):26s}  ALL FOLDS FAILED", flush=True)
                continue
            print(f"  {label(ns):26s} sharpe={m['annualized_sharpe']:7.4f} "
                  f"annual={m['annualized_mean']:8.4%} maxdd={m['max_drawdown']:7.4%} "
                  f"failed={res['failed_folds']:3d}/{res['n_folds']:3d} "
                  f"maxw={res['max_weight']:.3f}", flush=True)

        report["datasets"][name] = {
            "n_obs": int(frame.shape[0]),
            "n_assets": int(frame.shape[1]),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "rows": rows,
        }
        print(f"[{name}] done\n", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"written -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
