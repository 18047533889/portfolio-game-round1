"""Measure how far the shipped book is from each prohibited portfolio.

Round 5 is the first round in which compliance is in direct tension with the
score. The score is three risk measures and a failure rate, with no return term,
so lower risk is unconditionally better -- and the prohibited SCM minimum
variance book is the lowest-risk long-only book there is. Every increment of the
anchor penalty that keeps us away from it is now pure score cost, which means
the decision cannot be made by taste and has to be made by measuring the margin.

The margin is reported per fold, not as a pooled statistic, and both the median
and the worst fold are printed. A median distance of 0.4 with one fold at 0.02
would mean one reoptimisation returned the banned book, and an average would
hide exactly that.

Distance is L1 in weight space. For two long-only books the maximum is 2, and
the scale is total absolute weight, so a distance of d means the two books
disagree on d/2 of the capital. Correlation with the banned book is reported
alongside because L1 alone can be large while the books are nearly proportional.

The SCM minimum variance book is rebuilt from the raw sample covariance of the
same training window -- the model's own covariance would measure distance to a
book that is not the banned one.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

LOOKBACK, STEP = 252, 20
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "nasdaq": ("load_nasdaq_dataset", 60),
    "factors": ("load_factors_dataset", 5),
}


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("sub_r5_cmp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submission/portfolio_round5.py")
    ap.add_argument("--datasets", default="sp500:20,ftse100:64,nasdaq:60,factors:5")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports/round5_compliance.json")
    args = ap.parse_args()

    from round2_backtest_report import _ewp, _ivp, _scm_gmvp
    from skfolio.model_selection import WalkForward, cross_val_predict

    cls = load_submission(args.submission).CVXPYPortfolio
    rows = []
    print(f"{'dataset':9s} {'folds':>6s}  {'L1 EWP med/min':>16s}  "
          f"{'L1 IVP med/min':>16s}  {'L1 SCM-GMVP med/min':>21s}  {'corr GMVP max':>14s}")
    print("-" * 96)

    for spec in [d.strip() for d in args.datasets.split(",")]:
        name, _, cap_s = spec.partition(":")
        cap = int(cap_s) if cap_s else None
        from skfolio import datasets as ds
        frame = getattr(ds, DATASETS[name][0])()
        panel = frame.pct_change(fill_method=None).dropna()
        if cap is not None and cap < panel.shape[1]:
            panel = panel.iloc[:, :cap]
        x_all = panel.to_numpy(dtype=float)

        model = cls(portfolio_params={"weight_drift": True, "name": "r5c"})
        model.raise_on_failure = False
        pred = cross_val_predict(model, panel,
                                 cv=WalkForward(train_size=LOOKBACK, test_size=STEP),
                                 n_jobs=1)

        d_ewp, d_ivp, d_gmvp, corr_gmvp = [], [], [], []
        positions = list(range(LOOKBACK, x_all.shape[0] - STEP + 1, STEP))
        weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
                   if getattr(p, "weights", None) is not None
                   and p.__class__.__name__ != "FailedPortfolio"]
        for w, pos in zip(weights, positions):
            train = x_all[pos - LOOKBACK:pos]
            complete = train[np.isfinite(train).all(axis=1)][-LOOKBACK:]
            d_ewp.append(float(np.abs(w - _ewp(w.size)).sum()))
            cov = np.cov(complete, rowvar=False, ddof=1)
            sd = np.sqrt(np.maximum(np.diag(cov), np.finfo(float).tiny))
            ivp = (1.0 / sd) / (1.0 / sd).sum()
            d_ivp.append(float(np.abs(w - ivp).sum()))
            gmvp = _scm_gmvp(complete)
            if gmvp is not None:
                d_gmvp.append(float(np.abs(w - gmvp).sum()))
                if w.std() > 0 and gmvp.std() > 0:
                    corr_gmvp.append(float(np.corrcoef(w, gmvp)[0, 1]))

        def fmt(values):
            if not values:
                return "n/a"
            return f"{np.median(values):.3f} / {min(values):.3f}"

        print(f"{name:9s} {len(weights):6d}  {fmt(d_ewp):>16s}  {fmt(d_ivp):>16s}  "
              f"{fmt(d_gmvp):>21s}  {max(corr_gmvp) if corr_gmvp else float('nan'):14.3f}")
        rows.append({
            "dataset": name, "folds": len(weights),
            "l1_ewp_median": float(np.median(d_ewp)) if d_ewp else None,
            "l1_ewp_min": float(min(d_ewp)) if d_ewp else None,
            "l1_ivp_median": float(np.median(d_ivp)) if d_ivp else None,
            "l1_ivp_min": float(min(d_ivp)) if d_ivp else None,
            "l1_scm_gmvp_median": float(np.median(d_gmvp)) if d_gmvp else None,
            "l1_scm_gmvp_min": float(min(d_gmvp)) if d_gmvp else None,
            "corr_scm_gmvp_max": float(max(corr_gmvp)) if corr_gmvp else None,
            "exact_hits": 0,
        })
        rows[-1]["exact_hits"] = int(sum(1 for d in d_gmvp if d < 1e-9))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "submission": str(args.submission.relative_to(ROOT)),
        "note": ("L1 in weight space, median and worst fold. For long-only books the "
                 "maximum is 2 and the scale is total absolute weight, so a distance "
                 "of d means the books disagree on d/2 of the capital."),
        "datasets": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")
    worst = min((r["l1_scm_gmvp_min"] for r in rows
                 if r["l1_scm_gmvp_min"] is not None), default=None)
    print(f"worst single-fold L1 to the SCM minimum variance book: "
          f"{worst:.4f}" if worst is not None else "no measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
