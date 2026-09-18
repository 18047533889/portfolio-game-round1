"""Where does the penalty actually help, and does it hurt anywhere?

`research/window_penalty_sweep.py` showed the median drawdown barely moves when
the anchor penalty rises inside a 240-day out-of-sample window, while median
Sharpe and median return both rise. Medians are not the whole story: a drawdown
penalty is only safe if the *tail* is safe, and the score ranks the portfolio
against its peers rather than against a fixed bar.

So this script keeps every drawn block's full triple, then reports

    * the distribution, especially p90/max drawdown, not just the median;
    * a per-block rank across the candidate penalties, so the comparison is
      relative the way the grading formula is relative;
    * a breakdown by subset size and by window start year, because "very early
      years" is exactly where the grader's random two-year pick is most likely
      to land and an average over all years can hide a hole there.

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
}

CONFIGS: list[tuple[str, str, float, float, float]] = [
    ("p=0.25 rm=0.25 SHIPPED", "regularized", 0.25, 63.0, 0.25),
    ("p=4    rm=0.25", "regularized", 4.0, 63.0, 0.25),
    ("p=16   rm=0.25", "regularized", 16.0, 63.0, 0.25),
    ("hrp    rm=0.25", "hrp", 0.0, 63.0, 0.25),
    ("p=0.25 rm=0.75", "regularized", 0.25, 63.0, 0.75),
    ("p=4    rm=0.75", "regularized", 4.0, 63.0, 0.75),
    ("hrp    rm=0.75", "hrp", 0.0, 63.0, 0.75),
]

SIZE_MENU = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 45, 64]


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("core_detail", path)
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


def eval_block(core, x: np.ndarray, method: str, penalty: float,
               half_life: float, recent_mix: float) -> dict | None:
    n_obs, n_assets = x.shape
    chunks: list[np.ndarray] = []
    for pos in range(TRAIN, n_obs - STEP + 1, STEP):
        try:
            result = core.allocate(x[pos - TRAIN:pos], half_life=half_life,
                                   recent_mix=recent_mix, anchor_penalty=penalty,
                                   method=method)
            w = np.asarray(result["weights"], dtype=float)
            if w.shape != (n_assets,) or not np.isfinite(w).all():
                raise ValueError("shape/finite")
            if (w < -1e-12).any() or abs(float(w.sum()) - 1.0) > 1e-9:
                raise ValueError("floor/sum")
            if n_assets > 1 and np.allclose(w, 1.0 / n_assets, atol=1e-9, rtol=1e-6):
                raise ValueError("equal weight")
        except BaseException:  # noqa: BLE001
            return None
        chunks.append(x[pos:pos + STEP] @ w)
    return metrics(np.concatenate(chunks))


def ranks(values: list[float], higher_is_better: bool) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    order = np.argsort(-v if higher_is_better else v, kind="stable")
    out = np.empty(len(v), dtype=float)
    out[order] = np.arange(1, len(v) + 1, dtype=float)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="sp500(20)")
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--early-fraction", type=float, default=0.5)
    ap.add_argument("--out", default="reports/penalty_detail.json")
    args = ap.parse_args()

    core = load_core(CORE)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    report: dict = {"seed": args.seed, "reps": args.reps, "datasets": {}}

    for name in names:
        frame = returns_frame(name)
        n_obs, n_total = frame.shape
        span = n_obs - WINDOW + 1
        rng = np.random.default_rng(args.seed)
        menu = [k for k in SIZE_MENU if k <= n_total] or [n_total]
        weights = np.array([3.0 if k <= 5 else 1.0 for k in menu], dtype=float)
        weights /= weights.sum()

        records = []
        for _ in range(args.reps):
            k = int(rng.choice(menu, p=weights))
            cols = np.sort(rng.choice(n_total, size=k, replace=False))
            hi = max(1, int(args.early_fraction * span))
            start = int(rng.integers(0, hi))
            x = frame.iloc[start:start + WINDOW, cols].to_numpy(dtype=float)

            per = {}
            for lab, method, penalty, hl, rm in CONFIGS:
                per[lab] = eval_block(core, x, method, penalty, hl, rm)
            if any(v is None for v in per.values()):
                continue
            ref = metrics(np.concatenate(
                [x[pos:pos + STEP].mean(axis=1)
                 for pos in range(TRAIN, x.shape[0] - STEP + 1, STEP)]))
            records.append({
                "n_assets": int(k),
                "start_year": int(frame.index[start].year),
                "cells": per,
                "reference": ref,
            })

        print(f"[{name}] {len(records)} usable blocks of {args.reps}", flush=True)

        labels = [c[0] for c in CONFIGS]
        sh = {l: np.array([r["cells"][l]["annualized_sharpe"] for r in records]) for l in labels}
        rt = {l: np.array([r["cells"][l]["annualized_mean"] for r in records]) for l in labels}
        dd = {l: np.array([r["cells"][l]["max_drawdown"] for r in records]) for l in labels}
        ref_sh = np.array([r["reference"]["annualized_sharpe"] for r in records])
        ref_rt = np.array([r["reference"]["annualized_mean"] for r in records])
        ref_dd = np.array([r["reference"]["max_drawdown"] for r in records])

        print(f"  {'config':18s} {'medSh':>7s} {'medRet':>8s} {'medDd':>7s} "
              f"{'p90Dd':>7s} {'maxDd':>7s} {'ShWin':>6s} {'RetWin':>7s} {'DdWin':>6s}",
              flush=True)
        summary = {}
        for l in labels:
            row = {
                "median_sharpe": float(np.median(sh[l])),
                "median_return": float(np.median(rt[l])),
                "median_drawdown": float(np.median(dd[l])),
                "p90_drawdown": float(np.percentile(dd[l], 90)),
                "max_drawdown": float(dd[l].max()),
                "sharpe_win": float((sh[l] > ref_sh).mean()),
                "return_win": float((rt[l] > ref_rt).mean()),
                "drawdown_win": float((dd[l] < ref_dd).mean()),
            }
            summary[l] = row
            print(f"  {l:18s} {row['median_sharpe']:7.4f} {row['median_return']:8.4%} "
                  f"{row['median_drawdown']:7.4%} {row['p90_drawdown']:7.4%} "
                  f"{row['max_drawdown']:7.4%} {row['sharpe_win']:6.1%} "
                  f"{row['return_win']:7.1%} {row['drawdown_win']:6.1%}", flush=True)

        print(f"\n  per-block ranking (1 = best of the {len(labels)} candidates)")
        rank_sh = np.vstack([ranks([r["cells"][l]["annualized_sharpe"] for l in labels], True)
                             for r in records])
        rank_rt = np.vstack([ranks([r["cells"][l]["annualized_mean"] for l in labels], True)
                             for r in records])
        rank_dd = np.vstack([ranks([r["cells"][l]["max_drawdown"] for l in labels], False)
                             for r in records])
        combined = (rank_sh + rank_rt + rank_dd) / 3.0
        print(f"  {'config':18s} {'meanShRank':>11s} {'meanRetRank':>12s} "
              f"{'meanDdRank':>11s} {'combined':>9s} {'wins':>6s}", flush=True)
        ranking = {}
        for i, l in enumerate(labels):
            entry = {
                "mean_sharpe_rank": float(rank_sh[:, i].mean()),
                "mean_return_rank": float(rank_rt[:, i].mean()),
                "mean_drawdown_rank": float(rank_dd[:, i].mean()),
                "combined_rank": float(combined[:, i].mean()),
                "block_wins": int((combined[:, i] == combined.min(axis=1)).sum()),
            }
            ranking[l] = entry
            print(f"  {l:18s} {entry['mean_sharpe_rank']:11.3f} "
                  f"{entry['mean_return_rank']:12.3f} {entry['mean_drawdown_rank']:11.3f} "
                  f"{entry['combined_rank']:9.3f} {entry['block_wins']:6d}", flush=True)

        print(f"\n  by subset size")
        size_buckets = {"k<=5": lambda r: r["n_assets"] <= 5,
                        "6<=k<=10": lambda r: 6 <= r["n_assets"] <= 10,
                        "k>=11": lambda r: r["n_assets"] >= 11}
        strat_size = {}
        for bname, pred in size_buckets.items():
            idx = [i for i, r in enumerate(records) if pred(r)]
            if not idx:
                continue
            strat_size[bname] = {"n": len(idx)}
            line = f"  {bname:10s} n={len(idx):4d}  "
            for l in labels:
                m_s = float(np.median(sh[l][idx]))
                m_r = float(np.median(rt[l][idx]))
                strat_size[bname][l] = {"median_sharpe": m_s, "median_return": m_r}
                line += f"{l.split()[0]}: Sh {m_s:.3f} Ret {m_r:.2%}   "
            print(line, flush=True)

        print(f"\n  by window start year")
        strat_year = {}
        years = np.array([r["start_year"] for r in records])
        qs = sorted({int(v) for v in np.percentile(years, [25, 50, 75])})
        bounds = [int(years.min())] + qs + [int(years.max()) + 1]
        year_buckets = [(lo, hi_, f"{lo}-{hi_ - 1}")
                        for lo, hi_ in zip(bounds[:-1], bounds[1:]) if hi_ > lo]
        for lo, hi_, yname in year_buckets:
            idx = [i for i, r in enumerate(records) if lo <= r["start_year"] < hi_]
            if not idx:
                continue
            strat_year[yname] = {"n": len(idx)}
            line = f"  {yname:10s} n={len(idx):4d}  "
            for l in labels:
                m_s = float(np.median(sh[l][idx]))
                m_r = float(np.median(rt[l][idx]))
                strat_year[yname][l] = {"median_sharpe": m_s, "median_return": m_r}
                line += f"{l.split()[0]}: Sh {m_s:.3f} Ret {m_r:.2%}   "
            print(line, flush=True)

        report["datasets"][name] = {
            "n_blocks": len(records),
            "summary": summary,
            "ranking": ranking,
            "by_size": strat_size,
            "by_year": strat_year,
            "records": records,
        }
        print(flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"written -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
