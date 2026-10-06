"""Round-3 headline report, computed on the SHIPPED submission file.

Three reasons this is separate from research/round2/scan_penalty_r2.py:

1. the scan runs `src/portfolio_game_round3/core.py`; this runs
   `submission/portfolio_round3.py`, i.e. the exact bytes that get uploaded,
   so the numbers are attached to a hash the user can verify;
2. the round-2 contract is used verbatim (lookback 252, optimize_every 20,
   weight_drift TRUE, leverage 1), and the two exposure quantities the
   leverage rule is about -- Net = sum(w) and Gross = sum|w| -- are recorded
   rather than assumed;
3. the compliance claim ("not one of the three prohibited books") is measured
   on every fold of the shipped artifact against the three prohibited weight
   vectors rebuilt from the same training window.

Reported per dataset: annualized return, max drawdown (skfolio's definition),
failure/fallback counts, and the median L1 distance of the shipped weights to
EWP, IVP and SCM-GMVP. Distances are reported as distributions (median / p05 /
p95) because a single fold is not evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
LOOKBACK = 126
STEP = 20
LEVERAGE = 1.5
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "factors": ("load_factors_dataset", 5),
}


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("shipped_r3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- prohibited books
def _scm_gmvp(train: np.ndarray) -> np.ndarray | None:
    """Long-only global minimum variance on the SAMPLE covariance matrix.

    This is prohibited book (c). It is rebuilt here only to measure how far the
    shipped weights are from it. Solved with SLSQP on the same simplex the
    submission uses, so the comparison is like-for-like.
    """
    from scipy.optimize import minimize

    x = np.asarray(train, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < 3:
        return None
    s = np.cov(x, rowvar=False, ddof=1)
    if s.ndim != 2 or not np.isfinite(s).all():
        return None
    n = s.shape[0]
    if np.allclose(s, 0):
        return None
    s = s + np.eye(n) * max(float(np.trace(s)) / n, 1e-18) * 1e-12
    obj = lambda w: float(w @ s @ w)  # noqa: E731
    jac = lambda w: 2.0 * s @ w  # noqa: E731
    res = minimize(obj, np.full(n, 1.0 / n), jac=jac, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones_like(w)}],
                   options={"maxiter": 500, "ftol": 1e-14})
    if not res.success or not np.isfinite(res.x).all():
        return None
    w = np.clip(res.x, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else None


def _ivp(train: np.ndarray) -> np.ndarray | None:
    """Prohibited book (b): inverse volatility on the training window."""
    x = np.asarray(train, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < 3:
        return None
    sd = np.nanstd(x, axis=0, ddof=1)
    if not np.isfinite(sd).all() or not np.all(sd > 0):
        return None
    inv = 1.0 / sd
    return inv / inv.sum()


def _ewp(n: int) -> np.ndarray:
    """Prohibited book (a)."""
    return np.full(n, 1.0 / n)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def _summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size),
            "median": float(np.median(v)),
            "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
            "min": float(v.min())}


def run_dataset(cls, name: str, loader: str, cap: int) -> dict:
    from skfolio.datasets import __dict__ as ds_dict
    from skfolio.model_selection import WalkForward, cross_val_predict

    prices = ds_dict[loader]()
    panel = prices.pct_change(fill_method=None).dropna().iloc[:, :cap]
    x_all = panel.to_numpy(dtype=float)
    n_obs, n_assets = x_all.shape

    model = cls(portfolio_params={"weight_drift": True, "name": name})
    model.raise_on_failure = False
    pred = cross_val_predict(model, panel, cv=WalkForward(train_size=LOOKBACK,
                                                          test_size=STEP),
                             n_jobs=1)

    positions = list(range(LOOKBACK, n_obs - STEP + 1, STEP))
    exposure = pred.long_short_exposure
    weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios]

    l1 = {"ewp": [], "ivp": [], "scm_gmvp": []}
    exact = {"ewp": 0, "ivp": 0, "scm_gmvp": 0}
    for w, pos in zip(weights, positions):
        train = x_all[pos - LOOKBACK:pos]
        ewp = _ewp(n_assets)
        ivp = _ivp(train)
        gmvp = _scm_gmvp(train)
        for key, ref in (("ewp", ewp), ("ivp", ivp), ("scm_gmvp", gmvp)):
            if ref is None:
                continue
            d = _dist(w, ref)
            l1[key].append(d)
            if d < 1e-9:
                exact[key] += 1

    return {
        "dataset": name,
        "panel": {"observations": int(n_obs), "assets": int(n_assets),
                  "start": str(panel.index[0].date()),
                  "end": str(panel.index[-1].date())},
        "contract": {"lookback": LOOKBACK, "optimize_every": STEP,
                     "weight_drift": True, "leverage": LEVERAGE},
        "n_portfolios": int(len(pred.portfolios)),
        "n_failed_portfolios": int(pred.n_failed_portfolios),
        "n_fallback_portfolios": int(pred.n_fallback_portfolios),
        "failure_rate": float(pred.n_failed_portfolios / len(pred.portfolios)),
        # skfolio exposes the annualised arithmetic mean of the out-of-sample
        # portfolio returns (`annualized_mean`); there is no `annualized_return`
        # attribute on MultiPeriodPortfolio. This is the quantity the round-2
        # scan reported as "annual", so the naming is kept consistent with it.
        "annual_return": float(pred.annualized_mean),
        "annualized_mean": float(pred.annualized_mean),
        "max_drawdown": float(pred.max_drawdown),
        "annual_volatility": float(pred.annualized_standard_deviation),
        "sharpe": float(pred.annualized_sharpe_ratio),
        "exposure": {
            "net_min": float(exposure["Net"].min()),
            "net_max": float(exposure["Net"].max()),
            "gross_max": float(exposure["Gross"].max()),
            "short_max": float(exposure["Short"].max()),
            "leverage_ok": bool(float(exposure["Gross"].max()) <= LEVERAGE + 1e-9),
        },
        "max_weight": {"median": float(np.median([w.max() for w in weights])),
                       "max": float(max(w.max() for w in weights))},
        "prohibited_distance_l1": {k: _summarise(v) for k, v in l1.items()},
        "prohibited_exact_hits": exact,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submission/portfolio_round3.py")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports/round3_backtest.json")
    args = ap.parse_args()

    cls = load_submission(args.submission).CVXPYPortfolio
    sha = hashlib.sha256(args.submission.read_bytes()).hexdigest()

    rows = []
    for name, (loader, cap) in DATASETS.items():
        print(f"running {name} ...", flush=True)
        rows.append(run_dataset(cls, name, loader, cap))

    report = {"submission": str(args.submission.relative_to(ROOT)),
              "submission_sha256": sha, "datasets": rows}
    cfg_path = ROOT / "configs/submission_round3.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        leverage = float(cfg.get("gross_leverage", 1.0))
        report["frozen_leverage_setting"] = {
            "gross_leverage": leverage,
            "aggregate_short_budget": 0.5 * (leverage - 1.0),
        }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [f"shipped submission SHA256: {sha}"]
    if "frozen_leverage_setting" in report:
        s = report["frozen_leverage_setting"]
        lines.append(f"frozen leverage setting  : gross_leverage={s['gross_leverage']} "
                     f"(aggregate short budget {s['aggregate_short_budget']})")
    lines.append("")
    head = (f"{'dataset':10s} {'n':>4s} {'folds':>5s} {'fail':>4s} "
            f"{'annual':>9s} {'vol':>8s} {'maxdd':>8s} "
            f"{'L1_EWP':>7s} {'L1_IVP':>7s} {'L1_GMVP':>8s} {'gross':>6s}")
    lines += [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['dataset']:10s} {r['panel']['assets']:4d} "
            f"{r['n_portfolios']:5d} {r['n_failed_portfolios']:4d} "
            f"{r['annual_return']:8.2%} {r['annual_volatility']:7.2%} "
            f"{r['max_drawdown']:8.2%} "
            f"{r['prohibited_distance_l1']['ewp']['median']:7.3f} "
            f"{r['prohibited_distance_l1']['ivp']['median']:7.3f} "
            f"{r['prohibited_distance_l1']['scm_gmvp']['median']:8.3f} "
            f"{r['exposure']['gross_max']:6.3f}")
    lines.append("")
    lines.append("total failed portfolios : "
                 f"{sum(r['n_failed_portfolios'] for r in rows)} / "
                 f"{sum(r['n_portfolios'] for r in rows)}")
    lines.append("total fallback portfolios: "
                 f"{sum(r['n_fallback_portfolios'] for r in rows)}")
    lines.append("exact hits on a prohibited book: "
                 f"{ {r['dataset']: r['prohibited_exact_hits'] for r in rows} }")
    gross_over = [r["dataset"] for r in rows if not r["exposure"]["leverage_ok"]]
    lines.append(f"max gross exposure       : "
                 f"{max(r['exposure']['gross_max'] for r in rows):.6f}")
    lines.append(f"max short exposure       : "
                 f"{max(r['exposure']['short_max'] for r in rows):.6f}")
    lines.append(f"datasets with gross > 1  : {gross_over if gross_over else 'none'}")
    text = "\n".join(lines)
    (args.out.with_suffix(".txt")).write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    total_failed = sum(r["n_failed_portfolios"] for r in rows)
    any_exact = any(any(v for v in r["prohibited_exact_hits"].values()) for r in rows)
    return 1 if (total_failed or any_exact) else 0


if __name__ == "__main__":
    raise SystemExit(main())
