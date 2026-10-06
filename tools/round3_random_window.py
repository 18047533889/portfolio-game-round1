"""Round-3 grader-shaped stress: random subset x random two-year window.

This is the highest-value gate in every round, because the failure rate is 70%
of the score and a full-history backtest cannot establish it. Round 3 changes
what a fold has to satisfy, so the checks are re-derived rather than inherited:

  * lookback is 126, so a two-year slice now contains far more folds than it
    did at 252 and small panels are exercised harder;
  * leverage is 1.5 and the permission to short actually has room. Net exposure
    is pinned at 1 by full investment and gross exposure is 1 + 2 * (total
    short), so the two rules together cap the aggregate short at 0.25. Gross is
    therefore asserted against the frozen leverage, and the realised short is
    reported because "granted" and "used" are different quantities.

Every fold is validated on its own: shape, finiteness, full investment, the
gross-leverage cap, and that the book is not equal weight. Counts are reported,
never averages, because a failure is a count.
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
LOOKBACK = 126
STEP = 20
WINDOW = 2 * 252  # a two-year slice, as the brief describes
CONFIG = ROOT / "configs/submission_round3.json"


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("sub_r3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATASETS = {
    "sp500": "load_sp500_dataset",
    "ftse100": "load_ftse100_dataset",
    "nasdaq": "load_nasdaq_dataset",
    "factors": "load_factors_dataset",
}


def load_panel(name: str, cap: int | None) -> pd.DataFrame:
    from skfolio import datasets as ds
    frame = getattr(ds, DATASETS[name])()
    returns = frame.pct_change(fill_method=None).dropna()
    if cap is not None and cap < returns.shape[1]:
        return returns.iloc[:, :cap]
    return returns


EXPOSURE_ATOL = 1e-8
LEVERAGE = 1.5  # overwritten from the frozen config at run time


def check(w: np.ndarray, n: int) -> str | None:
    """Per-fold legality under the round-3 contract."""
    if w is None:
        return "no_weights"
    if w.shape != (n,):
        return f"shape{w.shape}"
    if not np.isfinite(w).all():
        return "nonfinite"
    if abs(float(w.sum()) - 1.0) > EXPOSURE_ATOL:
        return "not_fully_invested"
    if float(w.sum()) > 1.0 + EXPOSURE_ATOL:
        return "net_exposure_above_one"
    if float(np.abs(w).sum()) > LEVERAGE + EXPOSURE_ATOL:
        return "gross_exposure_over_leverage"
    if n > 1 and np.allclose(w, 1.0 / n, atol=1e-8, rtol=1e-5):
        return "equal_weight"
    return None


def main() -> int:
    global LEVERAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submission/portfolio_round3.py")
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--datasets", default="sp500:20,ftse100:64,nasdaq:60,factors:5")
    ap.add_argument("--early-frac", type=float, default=0.4,
                    help="share of draws whose window starts in the first 30%% of history")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from skfolio.model_selection import WalkForward, cross_val_predict

    cls = load_submission(args.submission).CVXPYPortfolio
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    LEVERAGE = float(cfg.get("gross_leverage", 1.0))
    print(f"frozen gross_leverage = {LEVERAGE}  "
          f"(aggregate short budget {(LEVERAGE - 1.0) / 2.0})")

    rng = np.random.default_rng(20261006)
    cv = WalkForward(train_size=LOOKBACK, test_size=STEP)

    report: dict = {"contract": {"lookback": LOOKBACK, "step": STEP,
                                 "weight_drift": True, "leverage": LEVERAGE},
                    "datasets": {}}
    grand = {"slices": 0, "folds": 0, "failed": 0, "illegal": 0, "equal": 0}
    for spec in [d.strip() for d in args.datasets.split(",")]:
        name, _, cap_s = spec.partition(":")
        cap = int(cap_s) if cap_s else None
        panel = load_panel(name, cap)
        x_all = panel.to_numpy(dtype=float)
        stats = {"slices": 0, "folds": 0, "failed": 0, "illegal": 0, "equal": 0,
                 "reasons": {}, "max_weight": [], "gross": [], "short": []}
        print(f"\n{'='*74}\n{name}  {panel.shape}  cap={cap}")
        n_all = x_all.shape[1]
        for _ in range(args.reps):
            # Subset size skewed small: the brief says "a portion of the stocks",
            # and small subsets are where risk models break.
            hi = max(2, min(n_all, 25))
            k = int(rng.integers(1, hi + 1))
            cols = rng.choice(n_all, size=k, replace=False)
            n_obs = x_all.shape[0]
            span = WINDOW + LOOKBACK
            if span >= n_obs:
                start = 0
            elif rng.random() < args.early_frac:
                start = int(rng.integers(0, max(1, int(n_obs * 0.3))))
            else:
                start = int(rng.integers(0, n_obs - span + 1))
            block = x_all[start:start + span][:, cols]
            if block.shape[0] < LOOKBACK + STEP:
                continue
            frame = pd.DataFrame(block)
            model = cls(portfolio_params={"weight_drift": True, "name": "r3"})
            model.raise_on_failure = False
            try:
                pred = cross_val_predict(model, frame, cv=cv, n_jobs=1)
            except Exception as exc:  # noqa: BLE001
                stats["slices"] += 1
                stats["failed"] += 1
                key = f"cross_val {type(exc).__name__}"
                stats["reasons"][key] = stats["reasons"].get(key, 0) + 1
                continue
            stats["slices"] += 1
            for ptf in pred.portfolios:
                stats["folds"] += 1
                if ptf.__class__.__name__ == "FailedPortfolio":
                    stats["failed"] += 1
                    stats["reasons"]["failed_portfolio"] = \
                        stats["reasons"].get("failed_portfolio", 0) + 1
                    continue
                weights = getattr(ptf, "weights", None)
                problem = check(np.asarray(weights, dtype=float) if weights is not None
                                else None, k)
                if problem == "equal_weight":
                    stats["equal"] += 1
                elif problem is not None:
                    stats["illegal"] += 1
                    stats["reasons"][problem] = stats["reasons"].get(problem, 0) + 1
                if weights is not None:
                    wv = np.asarray(weights, dtype=float)
                    stats["max_weight"].append(float(np.max(wv)))
                    stats["gross"].append(float(np.abs(wv).sum()))
                    stats["short"].append(float(np.maximum(-wv, 0.0).sum()))
        stats["max_weight_median"] = (float(np.median(stats["max_weight"]))
                                      if stats["max_weight"] else None)
        stats["gross_max"] = max(stats["gross"]) if stats["gross"] else None
        stats["short_max"] = max(stats["short"]) if stats["short"] else None
        stats["gross_within_leverage"] = (
            None if stats["gross_max"] is None
            else bool(stats["gross_max"] <= LEVERAGE + EXPOSURE_ATOL))
        del stats["max_weight"], stats["gross"], stats["short"]
        report["datasets"][name] = stats
        for key in grand:
            grand[key] += stats[key]
        print(f"  slices={stats['slices']:5d} folds={stats['folds']:6d} "
              f"failed={stats['failed']:4d} illegal={stats['illegal']:4d} "
              f"equal={stats['equal']:3d}  reasons={stats['reasons']}")
        print(f"  short_max={stats['short_max']} gross_max={stats['gross_max']} "
              f"within_leverage={stats['gross_within_leverage']}")

    print(f"\n{'='*74}\n=== TOTAL ===")
    print(f"slices          : {grand['slices']}")
    print(f"folds           : {grand['folds']}")
    print(f"failed folds    : {grand['failed']}")
    print(f"illegal folds   : {grand['illegal']}")
    print(f"equal-wt folds  : {grand['equal']}")
    print(f"gross<=leverage : "
          f"{all(d['gross_within_leverage'] for d in report['datasets'].values() if d['gross_within_leverage'] is not None)}"
          f"  (leverage={LEVERAGE})")
    report["total"] = grand
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"written -> {args.out}")
    return 1 if (grand["failed"] or grand["illegal"] or grand["equal"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
