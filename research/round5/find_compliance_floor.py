"""Find the lowest anchor penalty whose WORST fold stays clear of the banned book.

Why this exists. Round 5's score is three risk measures with no return term, so
lower risk is unconditionally better -- and the prohibited SCM minimum variance
book is the lowest-risk long-only book there is. Lowering the anchor penalty
therefore buys score and spends compliance margin, continuously, and the trade
has to be priced rather than felt.

The pricing is per fold, not pooled. The sweep that selected the parameters
reported the *median* distance and showed a comfortable 0.346 on the worst
dataset; the per-fold version of the same measurement showed one fold at 0.000
with correlation 1.000, i.e. one reoptimisation that returned the banned book
exactly. A median cannot see that, and the median was what the selection was
being made on. Hence this tool: it reports the worst fold, which is the number
that decides whether the submission is compliant, and the corresponding worst
correlation.

Reported for each penalty and each dataset so the floor is set by the data
rather than by a round number.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

LOOKBACK, STEP = 252, 20
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "factors": ("load_factors_dataset", 5),
}


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("core_r5_floor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", type=Path,
                    default=ROOT / "src/portfolio_game_round5/core.py")
    ap.add_argument("--penalties", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--mixes", default="0.5")
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports/round5_compliance_floor.json")
    args = ap.parse_args()

    from round2_backtest_report import _scm_gmvp
    from skfolio import datasets as ds
    from skfolio.model_selection import WalkForward, cross_val_predict
    from skfolio.optimization import BaseOptimization

    core = load_core(args.core)
    penalties = [float(x) for x in args.penalties.split(",")]
    mixes = [float(x) for x in args.mixes.split(",")]

    rows = []
    for name in [d.strip() for d in args.datasets.split(",")]:
        loader, cap = DATASETS[name]
        panel = getattr(ds, loader)().pct_change(fill_method=None).dropna().iloc[:, :cap]
        x_all = panel.to_numpy(dtype=float)
        print(f"\n=== {name}  {panel.shape} ===")
        print(f"  {'p':>6s} {'mix':>5s} {'folds':>6s} {'L1 med':>8s} {'L1 MIN':>8s} "
              f"{'corr MAX':>9s} {'folds<0.10':>11s}")
        for mix in mixes:
            for penalty in penalties:
                class Probe(BaseOptimization):
                    def fit(self, X, y=None):
                        from sklearn.utils.validation import validate_data
                        try:
                            r = validate_data(self, X)
                        except ValueError:
                            r = validate_data(self, X, ensure_all_finite=False)
                        res = core.allocate(np.asarray(r, dtype=float),
                                            half_life=63.0, recent_mix=0.25,
                                            anchor_penalty=penalty,
                                            gross_budget=1.0, downside_mix=mix)
                        self.weights_ = res["weights"]
                        self.diagnostics_ = res["diagnostics"]
                        return self

                model = Probe(portfolio_params={"weight_drift": True, "name": "f"})
                model.raise_on_failure = False
                pred = cross_val_predict(model, panel,
                                         cv=WalkForward(train_size=LOOKBACK,
                                                        test_size=STEP), n_jobs=1)
                positions = list(range(LOOKBACK, x_all.shape[0] - STEP + 1, STEP))
                weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
                           if getattr(p, "weights", None) is not None
                           and p.__class__.__name__ != "FailedPortfolio"]
                d_gmvp, c_gmvp = [], []
                for w, pos in zip(weights, positions):
                    train = x_all[pos - LOOKBACK:pos]
                    complete = train[np.isfinite(train).all(axis=1)][-LOOKBACK:]
                    gmvp = _scm_gmvp(complete)
                    if gmvp is None:
                        continue
                    d_gmvp.append(float(np.abs(w - gmvp).sum()))
                    if w.std() > 0 and gmvp.std() > 0:
                        c_gmvp.append(float(np.corrcoef(w, gmvp)[0, 1]))
                if not d_gmvp:
                    continue
                close = int(sum(1 for d in d_gmvp if d < 0.10))
                print(f"  {penalty:6.2f} {mix:5.2f} {len(d_gmvp):6d} "
                      f"{np.median(d_gmvp):8.3f} {min(d_gmvp):8.3f} "
                      f"{max(c_gmvp):9.4f} {close:11d}")
                rows.append({"dataset": name, "penalty": penalty, "downside_mix": mix,
                             "folds": len(d_gmvp),
                             "l1_median": float(np.median(d_gmvp)),
                             "l1_min": float(min(d_gmvp)),
                             "corr_max": float(max(c_gmvp)) if c_gmvp else None,
                             "folds_below_0.10": close})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "note": ("L1 in weight space against the long-only SCM minimum variance book "
                 "rebuilt from the same training window. The minimum over folds, not "
                 "the median, is what decides compliance."),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
