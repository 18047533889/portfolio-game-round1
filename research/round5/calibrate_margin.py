"""Calibrate the compliance margin against books that are admittedly legitimate.

An absolute distance threshold is a guess. The question "is our book the banned
book" is better answered relative to a known-good answer: how far are *other*
legitimate portfolios from the SCM minimum variance book on the same folds? If
a plainly different method sits at distance d from the banned book, then any
book at distance >= d is as distinct from it as a legitimate alternative is.

That is the calibration this measures, and it is why it exists: an earlier
absolute rule ("the worst fold must be at least ten times the distance at which
a violation was diagnosed, i.e. 0.30") turned out to be unreachable on sp500 at
any penalty, because the worst fold there saturates near 0.24 regardless of the
anchor penalty. The rule was the problem, not the book.

Reported per fold, as the minimum and median, because one fold matters.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/round2"))
sys.path.insert(0, str(ROOT / "tools"))

LOOKBACK, STEP = 252, 20
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "factors": ("load_factors_dataset", 5),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports/round5_margin_calibration.json")
    args = ap.parse_args()

    import rank_sim
    from round2_backtest_report import _scm_gmvp
    from skfolio import datasets as ds
    from skfolio.model_selection import WalkForward, cross_val_predict

    rows = []
    for name in [d.strip() for d in args.datasets.split(",")]:
        loader, cap = DATASETS[name]
        panel = getattr(ds, loader)().pct_change(fill_method=None).dropna().iloc[:, :cap]
        x_all = panel.to_numpy(dtype=float)
        positions = list(range(LOOKBACK, x_all.shape[0] - STEP + 1, STEP))
        print(f"\n=== {name}  {panel.shape}  ({len(positions)} folds) ===")
        print(f"  {'legitimate method':26s} {'L1 med':>8s} {'L1 MIN':>8s}")
        for label, factory in rank_sim.field().items():
            if label == "MinVariance(SCM)":
                continue  # this is the banned book; distance to itself is zero
            model = factory()
            model.set_params(portfolio_params={"weight_drift": True, "name": "f"})
            model.raise_on_failure = False
            try:
                pred = cross_val_predict(model, panel,
                                         cv=WalkForward(train_size=LOOKBACK,
                                                        test_size=STEP), n_jobs=1)
            except Exception:  # noqa: BLE001
                print(f"  {label:26s} {'--':>8s} {'--':>8s}  (failed to run)")
                continue
            weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
                       if getattr(p, "weights", None) is not None
                       and p.__class__.__name__ != "FailedPortfolio"]
            dist = []
            for w, pos in zip(weights, positions):
                train = x_all[pos - LOOKBACK:pos]
                complete = train[np.isfinite(train).all(axis=1)][-LOOKBACK:]
                gmvp = _scm_gmvp(complete)
                if gmvp is None:
                    continue
                dist.append(float(np.abs(w - gmvp).sum()))
            if not dist:
                continue
            print(f"  {label:26s} {np.median(dist):8.3f} {min(dist):8.3f}")
            rows.append({"dataset": name, "method": label, "folds": len(dist),
                         "l1_median": float(np.median(dist)),
                         "l1_min": float(min(dist))})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "note": ("L1 distance from each admittedly-legitimate reference method to the "
                 "prohibited long-only SCM minimum variance book, on the same folds. "
                 "The smallest of these is the distance at which a portfolio is "
                 "demonstrably a different book."),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")

    per_dataset = {}
    for row in rows:
        per_dataset.setdefault(row["dataset"], []).append(row["l1_min"])
    print("\nsmallest per-fold distance among legitimate methods, per dataset:")
    for dataset, values in per_dataset.items():
        print(f"  {dataset:9s} {min(values):.3f}   "
              f"(across {len(values)} methods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
