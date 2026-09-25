"""Round-2 grader-shaped stress: random subset x random two-year window.

Round 1 ran this under the round-1 contract. Round 2 changes two things that
affect the result and therefore require a re-run rather than a re-interpretation:

  * weight_drift = TRUE, so each holding period starts at the target weights and
    then drifts with asset returns until the next rebalance. This is the more
    realistic implementation convention and it also exercises a different code
    path in skfolio (previous_weights is threaded between folds).
  * shortselling = TRUE, leverage = 1. Whether that permission is used is a
    frozen configuration decision, so the checks here read
    configs/submission_round2.json instead of assuming long-only. Net exposure is
    pinned at 1 by full investment; gross exposure is 1 + 2 * (total short), so
    the two rules ("you may short" and "gross <= 1") cannot both bind. When a
    short budget is granted, gross is reported against the budget rather than
    asserted against a leverage cap it can no longer satisfy.

Every fold is validated on its own: shape, finiteness, the frozen short floor,
full investment, and that the book is not equal weight. Counts are reported, not
averages, because a failure is a count.
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
LOOKBACK = 252
STEP = 20
WINDOW = 2 * LOOKBACK  # a two-year slice, as the brief describes


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("sub_r2", path)
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
        returns = returns.iloc[:, :cap]
    return returns


LEVERAGE = 1.0
EXPOSURE_ATOL = 1e-8


def load_frozen() -> tuple[bool, float]:
    """Read the shipped short setting rather than assuming long-only.

    Round 2's brief sets shortselling = TRUE. Whether this submission uses that
    permission is a frozen configuration decision, so the validation has to read
    it: a hard-coded `weights < 0 -> illegal` check would pass a long-only build
    and fail a deliberately short-enabled one for no reason.
    """
    path = ROOT / "configs/submission_round2.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    allow = bool(cfg.get("allow_short", False))
    return allow, float(cfg.get("short_cap", 0.0)) if allow else 0.0


SHORT_CAP = 0.0
ALLOW_SHORT = False


def check(w: np.ndarray, n: int) -> str | None:
    if w is None:
        return "no_weights"
    if w.shape != (n,):
        return f"shape{w.shape}"
    if not np.isfinite(w).all():
        return "nonfinite"
    if np.any(w < -SHORT_CAP - EXPOSURE_ATOL):
        return "below_short_budget"
    if abs(float(w.sum()) - 1.0) > EXPOSURE_ATOL:
        return "not_fully_invested"
    # Round 2 sets leverage = 1. skfolio's MultiPeriodPortfolio defines the
    # measurable quantities as Net = sum(w) and Gross = sum|w|. Net is pinned at
    # 1 by full investment. Gross is 1 + 2*(total short), so it stays within the
    # leverage cap only while nothing is shorted: a short budget and a gross cap
    # of 1 cannot both bind. When a budget is granted the gross cap is therefore
    # reported instead of asserted, and the assertion becomes the budget itself.
    if float(w.sum()) > LEVERAGE + EXPOSURE_ATOL:
        return "net_exposure_over_leverage"
    if float(np.abs(w).sum()) > LEVERAGE + 2.0 * SHORT_CAP * n + EXPOSURE_ATOL:
        return "gross_exposure_over_short_budget"
    if n > 1 and np.allclose(w, 1.0 / n, atol=1e-8, rtol=1e-5):
        return "equal_weight"
    return None


def main() -> int:
    global SHORT_CAP, ALLOW_SHORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submission/portfolio_round2.py")
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--datasets", default="sp500:20,ftse100:64,nasdaq:60,factors:5")
    ap.add_argument("--early-frac", type=float, default=0.4,
                    help="share of draws whose window starts in the first N%% of history")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from skfolio.model_selection import WalkForward, cross_val_predict

    cls = load_submission(args.submission).CVXPYPortfolio
    ALLOW_SHORT, SHORT_CAP = load_frozen()
    print(f"frozen config: allow_short={ALLOW_SHORT} short_cap={SHORT_CAP}")
    rng = np.random.default_rng(20260925)
    cv = WalkForward(train_size=LOOKBACK, test_size=STEP)

    report: dict = {"contract": {"lookback": LOOKBACK, "step": STEP,
                                 "weight_drift": True, "leverage": LEVERAGE,
                                 "allow_short": ALLOW_SHORT,
                                 "short_cap": SHORT_CAP},
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
        for rep in range(args.reps):
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
            model = cls(portfolio_params={"weight_drift": True, "name": "r2"})
            model.raise_on_failure = False
            try:
                pred = cross_val_predict(model, frame, cv=cv, n_jobs=1)
            except Exception as exc:
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
                    stats["short"].append(float(-np.minimum(wv, 0.0).sum()))
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
              f"within_leverage_1={stats['gross_within_leverage']}")

    print(f"\n{'='*74}\n=== TOTAL ===")
    print(f"slices          : {grand['slices']}")
    print(f"folds           : {grand['folds']}")
    print(f"failed folds    : {grand['failed']}")
    print(f"illegal folds   : {grand['illegal']}")
    print(f"equal-wt folds  : {grand['equal']}")
    within = [d["gross_within_leverage"] for d in report["datasets"].values()
              if d["gross_within_leverage"] is not None]
    print(f"gross<=1 on all folds : {all(within) if within else 'n/a'} "
          f"(allow_short={ALLOW_SHORT} short_cap={SHORT_CAP})")
    report["total"] = grand
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"written -> {args.out}")
    return 1 if (grand["failed"] or grand["illegal"] or grand["equal"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
