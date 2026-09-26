"""Can a downside-risk term buy drawdown without giving up return or compliance?

The round-2 score is annual return 15% + max drawdown 15% + failure rate 70%, and
the penalty sweep showed that lowering anchor_penalty improves drawdown but costs
annual return. That is a trade along the frontier, not an improvement. The
question this script answers is whether a *strictly* better direction exists:
adding a downside-risk term to the objective while holding anchor_penalty fixed.

The objective stays a convex quadratic, so the existing projected-gradient
solver is reused unchanged -- only the matrix it is handed changes:

    minimize  w'(C + lam*S)w / (b'(C + lam*S)b) + p*||w-b||^2/||b||^2

where C is the shipped shrunk-and-denoised covariance, S is a downside
(semi-)covariance estimated from negative deviations, and b is the HRP anchor of
the combined risk measure. At lam = 0 this is exactly the shipped objective.

Also reports the L1 distance to the three prohibited books, because an objective
change can move the book toward a banned one even when it improves the score.

Exposure is deliberately long-only here: the short-budget study
(tools/round2_short_sweep.py) already showed that shorting buys nothing at the
shipped penalty and pushes gross exposure past the leverage cap.
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
sys.path.insert(0, str(ROOT / "research/round2"))
sys.path.insert(0, str(ROOT / "tools"))

LOOKBACK, STEP = 252, 20
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "factors": ("load_factors_dataset", 5),
}


def load_core():
    spec = importlib.util.spec_from_file_location(
        "core_r2_downside", ROOT / "src/portfolio_game_round2/core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def downside_covariance(core, returns: np.ndarray, half_life: float) -> np.ndarray:
    """Semi-covariance of below-mean deviations, normalised like the main estimate.

    Uses the same exponential recency weights as the shipped fast block, so the
    two risk measures differ only in which observations they count -- the
    downside ones -- and not in how they weight time. Normalised through the
    same _stabilise_covariance helper (median diagonal -> 1) so a blend weight is
    comparable rather than arbitrary.
    """
    x = np.asarray(returns, dtype=np.float64)
    ages = np.arange(len(x) - 1, -1, -1, dtype=np.float64)
    a = np.exp2(-ages / half_life)
    a /= a.sum()
    centered = x - a @ x
    down = np.minimum(centered, 0.0)
    weighted = down * np.sqrt(a)[:, None]
    semi = weighted.T @ weighted
    return core._stabilise_covariance(semi)


def combined_weights(core, train: np.ndarray, lam: float, penalty: float,
                     half_life: float, recent_mix: float) -> dict:
    """One fit with the downside blend; lam = 0 reproduces the shipped book."""
    x = np.asarray(train, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    complete = x[-LOOKBACK:]
    covariance = core.estimate_covariance(complete, half_life, recent_mix)
    if lam > 0.0:
        semi = downside_covariance(core, complete, half_life)
        blended = core._stabilise_covariance(covariance + lam * semi)
    else:
        blended = covariance
    anchor = core.hrp_weights(blended)
    solution, info = core.solve_regularized_qp(blended, anchor, penalty)
    return {"weights": solution, "anchor": anchor, "covariance": blended,
            "converged": info["converged"], "gap": info["relative_gap"]}


def l1_prohibited(core, w: np.ndarray, train: np.ndarray, blended: np.ndarray) -> dict:
    """L1 distance to the three prohibited books, rebuilt from the same window."""
    from round2_backtest_report import _ewp, _ivp, _scm_gmvp

    x = np.asarray(train, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)][-LOOKBACK:]
    out = {}
    ref = _ewp(w.size)
    sd = np.sqrt(np.maximum(np.diag(blended), np.finfo(float).tiny))
    ivp = (1.0 / sd) / (1.0 / sd).sum()
    gmvp = _scm_gmvp(x)
    for key, r in (("l1_ewp", ref), ("l1_ivp", ivp), ("l1_gmvp", gmvp)):
        if r is not None:
            out[key] = float(np.abs(w - r).sum())
    return out


def evaluate(core, name: str, cap_n: int, penalties: list[float],
             lams: list[float], half_life: float, recent_mix: float) -> list[dict]:
    from skfolio.datasets import __dict__ as ds_dict
    from skfolio.model_selection import WalkForward, cross_val_predict
    from skfolio.optimization import BaseOptimization

    loader, _ = DATASETS[name]
    panel = ds_dict[loader]().pct_change(fill_method=None).dropna().iloc[:, :cap_n]
    x_all = panel.to_numpy(dtype=float)
    positions = list(range(LOOKBACK, x_all.shape[0] - STEP + 1, STEP))

    rows = []
    for penalty in penalties:
        for lam in lams:
            # No __init__ override: skfolio's weight_drift path calls
            # set_params(previous_weights=...), and a narrowed constructor
            # signature would reject it.
            class Probe(BaseOptimization):
                def fit(self, X, y=None):
                    from sklearn.utils.validation import validate_data
                    try:
                        r = validate_data(self, X)
                    except ValueError:
                        r = validate_data(self, X, ensure_all_finite=False)
                    res = combined_weights(core, np.asarray(r, dtype=float), lam,
                                          penalty, half_life, recent_mix)
                    self.weights_ = res["weights"]
                    self.diagnostics_ = res
                    return self

            model = Probe(portfolio_params={"weight_drift": True, "name": "d"})
            model.raise_on_failure = False
            pred = cross_val_predict(model, panel, cv=WalkForward(train_size=LOOKBACK,
                                                                 test_size=STEP), n_jobs=1)
            weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
                       if getattr(p, "weights", None) is not None]
            dist = {k: [] for k in ("l1_ewp", "l1_ivp", "l1_gmvp")}
            for w, pos in zip(weights, positions):
                train = x_all[pos - LOOKBACK:pos]
                complete = train[np.isfinite(train).all(axis=1)][-LOOKBACK:]
                cov = core.estimate_covariance(complete, half_life, recent_mix)
                if lam > 0.0:
                    cov = core._stabilise_covariance(
                        cov + lam * downside_covariance(core, complete, half_life))
                for k, v in l1_prohibited(core, w, train, cov).items():
                    dist[k].append(v)
            failed = int(pred.n_failed_portfolios)
            row = {
                "dataset": name, "penalty": penalty, "lam": lam,
                "n_portfolios": len(pred.portfolios), "n_failed": failed,
                "failure_rate": failed / max(1, len(pred.portfolios)),
                "annual_return": float(pred.annualized_mean) if failed == 0 else float("nan"),
                "max_drawdown": float(pred.max_drawdown) if failed == 0 else float("nan"),
            }
            for k, v in dist.items():
                row[k] = float(np.median(v)) if v else float("nan")
            rows.append(row)
            print(f"  p={penalty:5.2f} lam={lam:5.2f} fail={failed:3d}/{len(pred.portfolios):4d} "
                  f"annual={row['annual_return']:8.4%} dd={row['max_drawdown']:8.4%} "
                  f"L1gmvp={row['l1_gmvp']:6.3f} L1ivp={row['l1_ivp']:6.3f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", default="4.0")
    ap.add_argument("--lams", default="0.0,0.25,0.5,1.0,2.0")
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--half-life", type=float, default=63.0)
    ap.add_argument("--recent-mix", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/round2_downside_scan.json")
    args = ap.parse_args()

    core = load_core()
    penalties = [float(x) for x in args.penalties.split(",")]
    lams = [float(x) for x in args.lams.split(",")]

    report = {"contract": {"lookback": LOOKBACK, "optimize_every": STEP,
                           "weight_drift": True, "allow_short": False,
                           "half_life": args.half_life,
                           "recent_mix": args.recent_mix},
              "rows": []}
    for name in [d.strip() for d in args.datasets.split(",")]:
        print(f"\n=== {name} ===", flush=True)
        report["rows"].extend(evaluate(core, name, DATASETS[name][1], penalties,
                                       lams, args.half_life, args.recent_mix))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
