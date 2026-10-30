"""Calibrate the round-6 compliance metric against legitimate reference methods.

Round 7 adds a fourth prohibition -- risk parity on the sample covariance -- and
the shipped anchor is its close relative: a hierarchical tree split whose
between-cluster rule is inverse cluster variance. So the compliance question is
not "is our book risk parity" (it is not, on either count: different allocation
rule, different estimator) but "how close is it to that banned object compared to
books nobody would question".

That comparison cannot be made from our own numbers alone. It needs a scale, and
the only defensible scale is the distance that legitimate methods sit at on the
same folds. Round 5 established the pattern: there, the distance to the banned
SCM minimum variance book was measured for eleven reference methods and NCO
turned out to reach 0.000 on two datasets, which meant a zero on one fold was
not by itself evidence of anything.

This script produces the same table for round 7, for all four banned books, with
the minimum over folds rather than the median.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/round2"))
sys.path.insert(0, str(ROOT / "tools"))

LOOKBACK, STEP = 60, 40
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "factors": ("load_factors_dataset", 5),
}


def load_core(path: Path | None = None):
    spec = importlib.util.spec_from_file_location(
        "core_r7_cal", path or ROOT / "src/portfolio_game_round7/core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel(name: str):
    from skfolio.datasets import __dict__ as ds_dict
    loader, cap = DATASETS[name]
    return ds_dict[loader]().pct_change(fill_method=None).dropna().iloc[:, :cap]


def distances(core, weights, x_all, positions) -> dict[str, list[float]]:
    """Per-fold L1 to each of the four banned books, rebuilt from the raw window."""
    from round2_backtest_report import _ewp, _ivp, _scm_gmvp

    out: dict[str, list[float]] = {"ewp": [], "ivp": [], "scm_gmvp": [], "erc_scm": []}
    for w, pos in zip(weights, positions):
        train = x_all[max(0, pos - LOOKBACK):pos]
        complete = train[np.isfinite(train).all(axis=1)][-LOOKBACK:]
        if len(complete) < 3 or complete.shape[1] != w.size:
            continue
        sample = np.atleast_2d(np.cov(complete, rowvar=False))
        if sample.shape != (w.size, w.size) or not np.isfinite(sample).all():
            continue
        gmvp = _scm_gmvp(complete)
        if gmvp is not None:
            out["scm_gmvp"].append(float(np.abs(w - gmvp).sum()))
        try:
            out["erc_scm"].append(float(np.abs(w - core.erc_weights(sample)).sum()))
        except Exception:  # noqa: BLE001
            pass
        ivp = _ivp(complete)
        if ivp is not None:
            out["ivp"].append(float(np.abs(w - ivp).sum()))
        out["ewp"].append(float(np.abs(w - _ewp(w.size)).sum()))
    return out


def run_method(factory, X) -> tuple[list[np.ndarray], int, int]:
    from skfolio.model_selection import WalkForward, cross_val_predict

    model = factory()
    model.set_params(portfolio_params={"weight_drift": True, "name": "cal"})
    model.raise_on_failure = False
    try:
        pred = cross_val_predict(model, X, cv=WalkForward(train_size=LOOKBACK,
                                                          test_size=STEP), n_jobs=1)
    except Exception:  # noqa: BLE001
        return [], 0, 0
    weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
               if getattr(p, "weights", None) is not None
               and p.__class__.__name__ != "FailedPortfolio"]
    return weights, len(pred.portfolios), int(pred.n_failed_portfolios)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/round7_calibration.json")
    args = ap.parse_args()

    import rank_sim
    core = load_core()
    report = {"lookback": LOOKBACK, "rows": []}

    for name in [d.strip() for d in args.datasets.split(",")]:
        X = panel(name)
        x_all = X.to_numpy(dtype=float)
        positions = list(range(LOOKBACK, x_all.shape[0] - STEP + 1, STEP))
        print(f"\n=== {name}  {X.shape}  ({len(positions)} folds) ===", flush=True)
        print(f"  {'legitimate method':24s} {'fail%':>7s} " +
              " ".join(f"{k:>16s}" for k in ("L1 EWP", "L1 IVP", "L1 GMVP", "L1 ERC")), flush=True)
        for label, factory in rank_sim.field().items():
            weights, n_total, n_failed = run_method(factory, X)
            if not weights:
                print(f"  {label:24s} {'--':>7s}   (no weights)", flush=True)
                continue
            d = distances(core, weights, x_all, positions)
            row = {"dataset": name, "method": label,
                   "failure_rate": n_failed / n_total if n_total else 1.0}
            cells = []
            for key in ("ewp", "ivp", "scm_gmvp", "erc_scm"):
                v = d[key]
                row[f"{key}_median"] = float(np.median(v)) if v else float("nan")
                row[f"{key}_min"] = float(np.min(v)) if v else float("nan")
                cells.append(f"{row[f'{key}_median']:.3f}/{row[f'{key}_min']:.3f}"
                             if v else "   --/--")
            report["rows"].append(row)
            print(f"  {label:24s} {row['failure_rate']:7.1%} " +
                  " ".join(f"{c:>16s}" for c in cells), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}   (median/min per fold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
