"""Round-5 evaluation harness and parameter sweep.

Round 5's score is three risk measures and a failure rate -- annual volatility
10%, maximum drawdown 10%, CVaR(95%) 10%, failure rate 70% -- with no return term
at all. Two consequences drive this harness:

  * the object being ranked is the *risk* of the book, so all three measures are
    recorded per setting rather than summarised into one number, because a book
    that trades volatility for CVaR is not obviously better or worse and the
    rank arithmetic is what decides;
  * because lower risk is unconditionally better here, a low-risk book can only
    be rejected on compliance grounds. The distance to the prohibited SCM
    minimum variance book is therefore measured on the training window for every
    setting and reported next to the score, so a setting that ranks well by
    collapsing onto the banned book is visible rather than rewarded.

The reference field is re-evaluated under the round-5 contract (train 252, test
20, weight_drift) rather than reused from an earlier round, because the window
is back to 252 and a stale field would score our settings against the wrong
opponents.
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
CORE = ROOT / "src/portfolio_game_round5/core.py"


def load_core(path: Path | None = None):
    spec = importlib.util.spec_from_file_location("core_r5", path or CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel(name: str) -> pd.DataFrame:
    from skfolio.datasets import __dict__ as ds_dict
    loader, cap = DATASETS[name]
    return ds_dict[loader]().pct_change(fill_method=None).dropna().iloc[:, :cap]


def _compliance(core, weights: list[np.ndarray], x_all: np.ndarray,
                positions: list[int], half_life: float, recent_mix: float,
                downside_mix: float) -> dict:
    """L1 distance to the three prohibited books, rebuilt per training window.

    The SCM minimum variance book is rebuilt from the raw sample covariance of
    the same window, which is what the prohibition names -- using the model's
    own covariance would measure distance to a book that is not the banned one.
    """
    from round2_backtest_report import _ewp, _ivp, _scm_gmvp

    out = {k: [] for k in ("l1_ewp", "l1_ivp", "l1_scm_gmvp")}
    for w, pos in zip(weights, positions):
        train = x_all[pos - LOOKBACK:pos]
        complete = train[np.isfinite(train).all(axis=1)][-LOOKBACK:]
        cov = core.estimate_covariance(complete, half_life, recent_mix)
        if downside_mix > 0.0:
            cov = core.blend_risk(cov, core.downside_covariance(complete, half_life),
                                  downside_mix)
        sd = np.sqrt(np.maximum(np.diag(cov), np.finfo(float).tiny))
        ivp = (1.0 / sd) / (1.0 / sd).sum()
        gmvp = _scm_gmvp(complete)
        if gmvp is not None:
            out["l1_scm_gmvp"].append(float(np.abs(w - gmvp).sum()))
        out["l1_ivp"].append(float(np.abs(w - ivp).sum()))
        out["l1_ewp"].append(float(np.abs(w - _ewp(w.size)).sum()))
    return {k: (float(np.median(v)) if v else float("nan")) for k, v in out.items()}


def measure(core, X: pd.DataFrame, *, half_life: float, recent_mix: float,
            penalty: float, downside_mix: float, leverage: float = 1.0,
            method: str = "regularized", compliance: bool = False) -> dict:
    from sklearn.utils.validation import validate_data
    from skfolio.model_selection import WalkForward, cross_val_predict
    from skfolio.optimization import BaseOptimization

    class Probe(BaseOptimization):
        # No __init__ override: skfolio's weight_drift path calls
        # set_params(previous_weights=...), which a narrowed signature rejects.
        def fit(self, X, y=None):
            try:
                returns = validate_data(self, X)
            except ValueError:
                returns = validate_data(self, X, ensure_all_finite=False)
            result = core.allocate(np.asarray(returns, dtype=float),
                                   half_life=half_life, recent_mix=recent_mix,
                                   anchor_penalty=penalty, method=method,
                                   gross_budget=leverage,
                                   downside_mix=downside_mix)
            self.weights_ = result["weights"]
            self.diagnostics_ = result["diagnostics"]
            return self

    model = Probe(portfolio_params={"weight_drift": True, "name": "r5"})
    model.raise_on_failure = False
    pred = cross_val_predict(model, X, cv=WalkForward(train_size=LOOKBACK,
                                                      test_size=STEP), n_jobs=1)
    n_total = len(pred.portfolios)
    n_failed = int(pred.n_failed_portfolios)
    out = {"n_portfolios": n_total, "n_failed": n_failed,
           "failure_rate": n_failed / n_total if n_total else 1.0,
           "n_fallback": int(pred.n_fallback_portfolios)}
    weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
               if getattr(p, "weights", None) is not None
               and p.__class__.__name__ != "FailedPortfolio"]
    if n_total and n_failed == 0:
        out["annual_volatility"] = float(pred.annualized_standard_deviation)
        out["max_drawdown"] = float(pred.max_drawdown)
        out["cvar_95"] = float(pred.cvar)
        out["annual_return"] = float(pred.annualized_mean)  # reported, not scored
    else:
        for k in ("annual_volatility", "max_drawdown", "cvar_95", "annual_return"):
            out[k] = float("nan")
    if weights:
        out["short_max"] = float(max(np.maximum(-w, 0.0).sum() for w in weights))
        out["gross_max"] = float(max(float(np.abs(w).sum()) for w in weights))
        out["min_weight"] = float(min(float(w.min()) for w in weights))
    if compliance and weights:
        x_all = X.to_numpy(dtype=float)
        positions = list(range(LOOKBACK, x_all.shape[0] - STEP + 1, STEP))
        out.update(_compliance(core, weights, x_all, positions, half_life,
                               recent_mix, downside_mix))
    return out


def evaluate_opponent(factory, X: pd.DataFrame) -> dict:
    from skfolio.model_selection import WalkForward, cross_val_predict

    model = factory()
    model.set_params(portfolio_params={"weight_drift": True, "name": "f"})
    model.raise_on_failure = False
    try:
        pred = cross_val_predict(model, X, cv=WalkForward(train_size=LOOKBACK,
                                                          test_size=STEP), n_jobs=1)
    except Exception as exc:  # noqa: BLE001
        return {"annual_volatility": float("nan"), "max_drawdown": float("nan"),
                "cvar_95": float("nan"), "failure_rate": 1.0,
                "n_portfolios": 0, "n_failed": 0, "error": type(exc).__name__}
    n_total, n_failed = len(pred.portfolios), int(pred.n_failed_portfolios)
    out = {"n_portfolios": n_total, "n_failed": n_failed,
           "failure_rate": n_failed / n_total if n_total else 1.0}
    if n_total and n_failed == 0:
        out["annual_volatility"] = float(pred.annualized_standard_deviation)
        out["max_drawdown"] = float(pred.max_drawdown)
        out["cvar_95"] = float(pred.cvar)
        out["annual_return"] = float(pred.annualized_mean)
    else:
        for k in ("annual_volatility", "max_drawdown", "cvar_95", "annual_return"):
            out[k] = float("nan")
    return out


def r5_score(entries: list[dict], rank_pct) -> list[float]:
    """0.10 * rank(-vol) + 0.10 * rank(-maxDD) + 0.10 * rank(-CVaR) + 0.70 * rank(-fail)."""
    r_vol = rank_pct([e["annual_volatility"] for e in entries], False)
    r_dd = rank_pct([e["max_drawdown"] for e in entries], False)
    r_cv = rank_pct([e["cvar_95"] for e in entries], False)
    r_fail = rank_pct([e["failure_rate"] for e in entries], False)
    return list(0.10 * r_vol + 0.10 * r_dd + 0.10 * r_cv + 0.70 * r_fail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", default="0.05,0.25,1.0,4.0,16.0")
    ap.add_argument("--mixes", default="0.0,0.35,0.65,1.0")
    ap.add_argument("--half-life", type=float, default=63.0)
    ap.add_argument("--recent-mix", type=float, default=0.25)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--compliance", action="store_true", default=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/round5_sweep.json")
    args = ap.parse_args()

    import rank_sim
    core = load_core()
    penalties = [float(x) for x in args.penalties.split(",")]
    mixes = [float(x) for x in args.mixes.split(",")]

    report = {"contract": {"lookback": LOOKBACK, "optimize_every": STEP,
                           "weight_drift": True, "shortselling": False,
                           "budget": args.leverage,
                           "scoring": {"annual_volatility": 0.10,
                                       "max_drawdown": 0.10, "cvar_95": 0.10,
                                       "failure_rate": 0.70},
                           "half_life": args.half_life,
                           "recent_mix": args.recent_mix},
              "settings": [], "field": {}}

    for name in [d.strip() for d in args.datasets.split(",")]:
        X = panel(name)
        print(f"\n{'='*130}\n{name}  {X.shape}", flush=True)
        field = {}
        if not args.skip_field:
            for label, factory in rank_sim.field().items():
                entry = evaluate_opponent(factory, X)
                entry["label"] = label
                field[label] = entry

        rows = []
        for mix in mixes:
            for penalty in penalties:
                res = measure(core, X, half_life=args.half_life,
                              recent_mix=args.recent_mix, penalty=penalty,
                              downside_mix=mix, leverage=args.leverage,
                              compliance=args.compliance)
                row = {"dataset": name, "penalty": penalty, "downside_mix": mix,
                       "leverage": args.leverage, "half_life": args.half_life,
                       "recent_mix": args.recent_mix, **res}
                rows.append(row)
                print(f"  mix={mix:4.2f} p={penalty:6.2f} fail={res['n_failed']:3d}/"
                      f"{res['n_portfolios']:4d} vol={res['annual_volatility']:7.4%} "
                      f"dd={res['max_drawdown']:8.4%} cvar={res['cvar_95']:7.4%} "
                      f"ret={res['annual_return']:8.4%} "
                      f"L1_gmvp={res.get('l1_scm_gmvp', float('nan')):6.3f}", flush=True)

        entries = [dict(e, label=k) for k, e in field.items()]
        for row in rows:
            entries.append({"label": f"OURS p={row['penalty']:.2f} mix={row['downside_mix']:.2f}",
                            "annual_volatility": row["annual_volatility"],
                            "max_drawdown": row["max_drawdown"],
                            "cvar_95": row["cvar_95"],
                            "failure_rate": row["failure_rate"],
                            "n_portfolios": row["n_portfolios"],
                            "n_failed": row["n_failed"]})
        scores = r5_score(entries, rank_sim.rank_pct)
        ordered = sorted(zip(scores, entries), key=lambda t: -t[0])
        print(f"\n  {'setting':28s} {'vol':>8s} {'maxdd':>9s} {'cvar95':>8s} "
              f"{'fail%':>7s} {'SCORE':>7s}")
        for score, e in ordered:
            print(f"  {e['label']:28s} {e['annual_volatility']:8.4%} "
                  f"{e['max_drawdown']:9.4%} {e['cvar_95']:8.4%} "
                  f"{e['failure_rate']:7.1%} {score:7.4f}")
        report["field"][name] = [dict(e, score=s) for s, e in ordered]
        report["settings"].extend(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
