"""Round-6 evaluation harness and parameter sweep.

Round 6 scores Sharpe ratio 20%, maximum drawdown 10% and failure rate 70%.
Two things follow, and they conflict:

  * return is back in the objective after round 5 removed it, but only through
    the return/risk ratio. A book that wins on Sharpe is not the lowest-risk
    book and not the highest-return book, so the interesting axis is no longer
    "minimise risk" -- it is "move along the frontier", and the honest way to
    report that is Sharpe AND the two quantities it is made of, because a
    setting can raise Sharpe by cutting volatility with a bigger cut in return.
  * a fourth prohibition arrives: risk parity on the sample covariance. The
    round-6 anchor is a close relative (a hierarchical tree split whose
    between-cluster rule is inverse cluster variance), so the distance to
    ERC-on-SCM is measured per fold against the sample covariance of the same
    window, MINIMUM over folds rather than median. Round 5 established why: the
    median hid seven folds on which the shipped book WAS the prohibited one.

The reference field is re-evaluated under the round-6 contract (train 100, test
20, weight drift) rather than reused from an earlier round. The window changed
from 252 to 100, which changes which reference methods survive: with 100
observations a 64-asset covariance has n/T = 0.64 and several optimizers start
emitting numerically singular systems.
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

LOOKBACK, STEP = 100, 20
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "nasdaq": ("load_nasdaq_dataset", 60),
    "factors": ("load_factors_dataset", 5),
}
CORE = ROOT / "src/portfolio_game_round6/core.py"


def load_core(path: Path | None = None):
    spec = importlib.util.spec_from_file_location("core_r6", path or CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel(name: str) -> pd.DataFrame:
    from skfolio.datasets import __dict__ as ds_dict
    loader, cap = DATASETS[name]
    return ds_dict[loader]().pct_change(fill_method=None).dropna().iloc[:, :cap]


def _compliance(core, weights: list[np.ndarray], x_all: np.ndarray,
                positions: list[int]) -> dict:
    """Per-fold L1 distance to the FOUR prohibited books.

    Everything here is rebuilt from the raw window, not from the model's own
    regularised matrix: two of the four prohibitions are qualified by "using the
    SCM", so the sample covariance is the object they name. Comparing against the
    shrunk and denoised estimate would understate the distance.

    Reported as (median, min) pairs. The minimum is the decision variable -- a
    book that coincides with a banned portfolio on one fold out of four hundred
    is exactly the failure mode that a median is blind to.
    """
    from round2_backtest_report import _ewp, _ivp, _scm_gmvp

    keys = ("l1_ewp", "l1_ivp", "l1_scm_gmvp", "l1_erc_scm")
    series: dict[str, list[float]] = {k: [] for k in keys}
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
            series["l1_scm_gmvp"].append(float(np.abs(w - gmvp).sum()))
        try:
            series["l1_erc_scm"].append(
                float(np.abs(w - core.erc_weights(sample)).sum()))
        except Exception:  # noqa: BLE001 - a failed reference solve is not a result
            pass
        ivp = _ivp(complete)
        if ivp is not None:
            series["l1_ivp"].append(float(np.abs(w - ivp).sum()))
        series["l1_ewp"].append(float(np.abs(w - _ewp(w.size)).sum()))
    out: dict[str, float] = {}
    for k, v in series.items():
        if v:
            out[f"{k}_median"] = float(np.median(v))
            out[f"{k}_min"] = float(np.min(v))
            out[f"{k}_p05"] = float(np.percentile(v, 5))
        else:
            out[f"{k}_median"] = float("nan")
            out[f"{k}_min"] = float("nan")
            out[f"{k}_p05"] = float("nan")
    return out


def _summarise(pred) -> dict:
    n_total = len(pred.portfolios)
    n_failed = int(pred.n_failed_portfolios)
    out = {"n_portfolios": n_total, "n_failed": n_failed,
           "failure_rate": n_failed / n_total if n_total else 1.0,
           "n_fallback": int(pred.n_fallback_portfolios)}
    if n_total and n_failed == 0:
        out["sharpe"] = float(pred.annualized_sharpe_ratio)
        out["annual_return"] = float(pred.annualized_mean)
        out["annual_volatility"] = float(pred.annualized_standard_deviation)
        out["max_drawdown"] = float(pred.max_drawdown)
        out["cvar_95"] = float(pred.cvar)
    else:
        for k in ("sharpe", "annual_return", "annual_volatility",
                  "max_drawdown", "cvar_95"):
            out[k] = float("nan")
    return out


def measure(core, X: pd.DataFrame, *, half_life: float, recent_mix: float,
            penalty: float, downside_mix: float, budget: float = 1.0,
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
                                   gross_budget=budget,
                                   downside_mix=downside_mix)
            self.weights_ = result["weights"]
            self.diagnostics_ = result["diagnostics"]
            return self

    model = Probe(portfolio_params={"weight_drift": True, "name": "r6"})
    model.raise_on_failure = False
    pred = cross_val_predict(model, X, cv=WalkForward(train_size=LOOKBACK,
                                                      test_size=STEP), n_jobs=1)
    out = _summarise(pred)
    weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
               if getattr(p, "weights", None) is not None
               and p.__class__.__name__ != "FailedPortfolio"]
    if weights:
        out["min_weight"] = float(min(float(w.min()) for w in weights))
        out["gross_max"] = float(max(float(np.abs(w).sum()) for w in weights))
    if compliance and weights and out["n_failed"] == 0:
        x_all = X.to_numpy(dtype=float)
        positions = list(range(LOOKBACK, x_all.shape[0] - STEP + 1, STEP))
        out.update(_compliance(core, weights, x_all, positions))
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
        return {"sharpe": float("nan"), "max_drawdown": float("nan"),
                "failure_rate": 1.0, "n_portfolios": 0, "n_failed": 0,
                "error": type(exc).__name__}
    return _summarise(pred)


def r6_score(entries: list[dict], rank_pct) -> list[float]:
    """0.20 * rank(sharpe) + 0.10 * rank(-maxDD) + 0.70 * rank(-failure rate)."""
    r_sharpe = rank_pct([e["sharpe"] for e in entries], True)
    r_dd = rank_pct([e["max_drawdown"] for e in entries], False)
    r_fail = rank_pct([e["failure_rate"] for e in entries], False)
    return list(0.20 * r_sharpe + 0.10 * r_dd + 0.70 * r_fail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", default="0.0,0.25,0.5,1.0,2.0,4.0,8.0,16.0")
    ap.add_argument("--mixes", default="0.0,0.5")
    ap.add_argument("--half-life", type=float, default=25.0)
    ap.add_argument("--recent-mix", type=float, default=0.25)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--skip-compliance", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/round6_sweep.json")
    args = ap.parse_args()

    import rank_sim
    core = load_core()
    penalties = [float(x) for x in args.penalties.split(",")]
    mixes = [float(x) for x in args.mixes.split(",")]

    report = {"contract": {"lookback": LOOKBACK, "optimize_every": STEP,
                           "weight_drift": True, "shortselling": False,
                           "budget": args.budget,
                           "scoring": {"sharpe": 0.20, "max_drawdown": 0.10,
                                       "failure_rate": 0.70},
                           "half_life": args.half_life,
                           "recent_mix": args.recent_mix},
              "settings": [], "field": {}}

    for name in [d.strip() for d in args.datasets.split(",")]:
        X = panel(name)
        print(f"\n{'='*140}\n{name}  {X.shape}", flush=True)
        field = {}
        if not args.skip_field:
            for label, factory in rank_sim.field().items():
                entry = evaluate_opponent(factory, X)
                entry["label"] = label
                field[label] = entry
                print(f"  [field] {label:24s} fail={entry['failure_rate']:7.1%} "
                      f"sharpe={entry['sharpe']:7.4f} dd={entry['max_drawdown']:8.4%}",
                      flush=True)

        rows = []
        for mix in mixes:
            for penalty in penalties:
                res = measure(core, X, half_life=args.half_life,
                              recent_mix=args.recent_mix, penalty=penalty,
                              downside_mix=mix, budget=args.budget,
                              compliance=not args.skip_compliance)
                row = {"dataset": name, "penalty": penalty, "downside_mix": mix,
                       "budget": args.budget, "half_life": args.half_life,
                       "recent_mix": args.recent_mix, **res}
                rows.append(row)
                print(f"  mix={mix:4.2f} p={penalty:6.2f} fail={res['n_failed']:3d}/"
                      f"{res['n_portfolios']:4d} sharpe={res['sharpe']:7.4f} "
                      f"ret={res['annual_return']:8.4%} vol={res['annual_volatility']:7.4%} "
                      f"dd={res['max_drawdown']:8.4%} "
                      f"L1_gmvp_min={res.get('l1_scm_gmvp_min', float('nan')):6.3f} "
                      f"L1_erc_min={res.get('l1_erc_scm_min', float('nan')):6.3f}",
                      flush=True)

        entries = [dict(e, label=k) for k, e in field.items()]
        for row in rows:
            entries.append({"label": f"OURS p={row['penalty']:.2f} mix={row['downside_mix']:.2f}",
                            "sharpe": row["sharpe"],
                            "max_drawdown": row["max_drawdown"],
                            "failure_rate": row["failure_rate"],
                            "n_portfolios": row["n_portfolios"],
                            "n_failed": row["n_failed"]})
        scores = r6_score(entries, rank_sim.rank_pct)
        ordered = sorted(zip(scores, entries), key=lambda t: -t[0])
        print(f"\n  {'setting':28s} {'sharpe':>8s} {'maxdd':>9s} {'fail%':>7s} {'SCORE':>7s}")
        for score, e in ordered:
            print(f"  {e['label']:28s} {e['sharpe']:8.4f} "
                  f"{e['max_drawdown']:9.4%} {e['failure_rate']:7.1%} {score:7.4f}")
        report["field"][name] = [dict(e, score=s) for s, e in ordered]
        report["settings"].extend(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
