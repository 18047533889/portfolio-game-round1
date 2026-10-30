"""Round-7 evaluation harness and parameter sweep.

Round 7 scores the MEDIAN over folds of Sharpe (10%) and CVaR 95% (20%), plus
the failure rate (70%). Three things about that differ from every earlier
round and each one changes the harness:

  * the metrics are per-fold medians, not metrics of the concatenated path.
    Those are different numbers: a multi-period portfolio's Sharpe is computed
    on the whole chained return series, so one catastrophic fold drags it down,
    while the median over folds simply does not see it. The brief says median,
    so the harness computes the metric on each sub-portfolio and takes the
    median, and it reports both so the difference is visible rather than
    assumed.

  * CVaR carries 20% and no variance or drawdown term is scored at all. The
    scored risk is therefore a left-tail quantity, which is what the
    downside_mix blend targets.

  * lookback = 60 with up to 64 assets means n >= T on the widest panel: the
    sample covariance is singular and the prohibitions that are qualified by
    "using the SCM" no longer define a portfolio. The reference books are still
    built -- np.linalg.solve on a singular matrix raises, and _scm_gmvp returns
    None -- so the compliance columns can be empty on those folds, and an empty
    column must be read as "no reference exists", not as "distance zero".

optimize_every = 40, so there are roughly half as many folds as round 6 and
each one spans twice as much drift.
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

LOOKBACK, STEP = 60, 40
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "nasdaq": ("load_nasdaq_dataset", 60),
    "factors": ("load_factors_dataset", 5),
}
CORE = ROOT / "src/portfolio_game_round7/core.py"


def load_core(path: Path | None = None):
    spec = importlib.util.spec_from_file_location("core_r7", path or CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel(name: str) -> pd.DataFrame:
    from skfolio.datasets import __dict__ as ds_dict
    loader, cap = DATASETS[name]
    return ds_dict[loader]().pct_change(fill_method=None).dropna().iloc[:, :cap]


def _per_fold(pred) -> tuple[list[float], list[float]]:
    """Sharpe and CVaR of each sub-portfolio, skipping failed ones."""
    sharpes, cvars = [], []
    for portfolio in pred.portfolios:
        if portfolio.__class__.__name__ == "FailedPortfolio":
            continue
        try:
            s = float(portfolio.annualized_sharpe_ratio)
            c = float(portfolio.cvar)
        except (AttributeError, TypeError, ValueError):
            continue
        if np.isfinite(s):
            sharpes.append(s)
        if np.isfinite(c):
            cvars.append(c)
    return sharpes, cvars


def _compliance(core, weights: list[np.ndarray], x_all: np.ndarray,
                positions: list[int]) -> dict:
    """Per-fold L1 to the four prohibited books; median and worst fold."""
    from round2_backtest_report import _ewp, _ivp, _scm_gmvp

    keys = ("l1_ewp", "l1_ivp", "l1_scm_gmvp", "l1_erc_scm")
    series: dict[str, list[float]] = {k: [] for k in keys}
    undefined = 0
    for w, pos in zip(weights, positions):
        train = x_all[max(0, pos - LOOKBACK):pos]
        complete = train[np.isfinite(train).all(axis=1)][-LOOKBACK:]
        if len(complete) < 3 or complete.shape[1] != w.size:
            continue
        # The two prohibitions qualified by "using the SCM" require the sample
        # covariance to be invertible. At lookback 60 with 60 or 64 assets it is
        # not: rank 59 of 64, condition number ~1e17. numpy's solve does not
        # raise on that -- it returns a vector -- so the reference book is
        # numerically available and numerically meaningless. Reporting a
        # distance to it would be worse than reporting nothing, so the distance
        # is left undefined and counted.
        rank_deficient = (len(complete) <= w.size
                          or np.linalg.matrix_rank(complete) < w.size)
        if rank_deficient:
            undefined += 1
        else:
            gmvp = _scm_gmvp(complete)
            if gmvp is not None:
                series["l1_scm_gmvp"].append(float(np.abs(w - gmvp).sum()))
            with np.errstate(invalid="ignore", divide="ignore"):
                sample = np.atleast_2d(np.cov(complete, rowvar=False))
            if sample.shape == (w.size, w.size) and np.isfinite(sample).all():
                try:
                    series["l1_erc_scm"].append(
                        float(np.abs(w - core.erc_weights(sample)).sum()))
                except Exception:  # noqa: BLE001 - a failed reference is a result
                    pass
        ivp = _ivp(complete)
        if ivp is not None:
            series["l1_ivp"].append(float(np.abs(w - ivp).sum()))
        series["l1_ewp"].append(float(np.abs(w - _ewp(w.size)).sum()))
    out: dict[str, float] = {}
    for k, v in series.items():
        for stat, fn in (("median", np.median), ("min", np.min)):
            out[f"{k}_{stat}"] = float(fn(v)) if v else float("nan")
        out[f"{k}_n"] = float(len(v))
    out["scm_reference_undefined_folds"] = float(undefined)
    return out


def _summarise(core, pred, X: pd.DataFrame, compliance: bool) -> dict:
    n_total = len(pred.portfolios)
    n_failed = int(pred.n_failed_portfolios)
    out = {"n_portfolios": n_total, "n_failed": n_failed,
           "failure_rate": n_failed / n_total if n_total else 1.0,
           "n_fallback": int(pred.n_fallback_portfolios)}
    sharpes, cvars = _per_fold(pred)
    if sharpes and cvars:
        out["sharpe_median"] = float(np.median(sharpes))
        out["cvar_median"] = float(np.median(cvars))
        out["sharpe_aggregate"] = float(pred.annualized_sharpe_ratio)
        out["cvar_aggregate"] = float(pred.cvar)
        out["sharpe_p05"] = float(np.percentile(sharpes, 5))
        out["cvar_p95"] = float(np.percentile(cvars, 95))
        out["annual_return"] = float(pred.annualized_mean)
        out["annual_volatility"] = float(pred.annualized_standard_deviation)
        out["max_drawdown"] = float(pred.max_drawdown)
    else:
        for k in ("sharpe_median", "cvar_median", "sharpe_aggregate",
                  "cvar_aggregate", "sharpe_p05", "cvar_p95", "annual_return",
                  "annual_volatility", "max_drawdown"):
            out[k] = float("nan")
    weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
               if getattr(p, "weights", None) is not None
               and p.__class__.__name__ != "FailedPortfolio"]
    if weights:
        out["max_weight_median"] = float(np.median([float(w.max()) for w in weights]))
        out["min_weight"] = float(min(float(w.min()) for w in weights))
        out["gross_max"] = float(max(float(np.abs(w).sum()) for w in weights))
    if compliance and weights and out["n_failed"] == 0:
        x_all = X.to_numpy(dtype=float)
        positions = list(range(LOOKBACK, x_all.shape[0] - STEP + 1, STEP))
        out.update(_compliance(core, weights, x_all, positions))
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

    model = Probe(portfolio_params={"weight_drift": True, "name": "r7"})
    model.raise_on_failure = False
    pred = cross_val_predict(model, X, cv=WalkForward(train_size=LOOKBACK,
                                                      test_size=STEP), n_jobs=1)
    return _summarise(core, pred, X, compliance)


def evaluate_opponent(core, factory, X: pd.DataFrame) -> dict:
    from skfolio.model_selection import WalkForward, cross_val_predict

    model = factory()
    model.set_params(portfolio_params={"weight_drift": True, "name": "f"})
    model.raise_on_failure = False
    try:
        pred = cross_val_predict(model, X, cv=WalkForward(train_size=LOOKBACK,
                                                          test_size=STEP), n_jobs=1)
    except Exception as exc:  # noqa: BLE001
        return {"sharpe_median": float("nan"), "cvar_median": float("nan"),
                "failure_rate": 1.0, "n_portfolios": 0, "n_failed": 0,
                "error": type(exc).__name__}
    return _summarise(core, pred, X, compliance=False)


def r7_score(entries: list[dict], rank_pct) -> list[float]:
    """0.10 * rank(median Sharpe) + 0.20 * rank(-median CVaR) + 0.70 * rank(-fail)."""
    r_sharpe = rank_pct([e["sharpe_median"] for e in entries], True)
    r_cvar = rank_pct([e["cvar_median"] for e in entries], False)
    r_fail = rank_pct([e["failure_rate"] for e in entries], False)
    return list(0.10 * r_sharpe + 0.20 * r_cvar + 0.70 * r_fail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", default="0.5,1,2,4,8,16")
    ap.add_argument("--mixes", default="0.0,0.5,1.0")
    ap.add_argument("--half-life", type=float, default=15.0)
    ap.add_argument("--recent-mix", type=float, default=0.25)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--skip-compliance", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/round7_sweep.json")
    args = ap.parse_args()

    import rank_sim
    core = load_core()
    penalties = [float(x) for x in args.penalties.split(",")]
    mixes = [float(x) for x in args.mixes.split(",")]

    report = {"contract": {"lookback": LOOKBACK, "optimize_every": STEP,
                           "weight_drift": True, "shortselling": False,
                           "budget": args.budget,
                           "scoring": {"sharpe_median": 0.10,
                                       "cvar_median": 0.20,
                                       "failure_rate": 0.70},
                           "half_life": args.half_life,
                           "recent_mix": args.recent_mix},
              "settings": [], "field": {}}

    for name in [d.strip() for d in args.datasets.split(",")]:
        X = panel(name)
        print(f"\n{'='*150}\n{name}  {X.shape}  folds~{(X.shape[0]-LOOKBACK)//STEP+1}",
              flush=True)
        field = {}
        if not args.skip_field:
            for label, factory in rank_sim.field().items():
                entry = evaluate_opponent(core, factory, X)
                entry["label"] = label
                field[label] = entry
                print(f"  [field] {label:24s} fail={entry['failure_rate']:7.1%} "
                      f"sharpe_med={entry['sharpe_median']:7.4f} "
                      f"cvar_med={entry['cvar_median']:7.4%}", flush=True)

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
                      f"{res['n_portfolios']:4d} sharpe_med={res['sharpe_median']:7.4f} "
                      f"cvar_med={res['cvar_median']:7.4%} "
                      f"dd={res['max_drawdown']:8.4%} maxw={res.get('max_weight_median', float('nan')):5.3f} "
                      f"gmvp_min={res.get('l1_scm_gmvp_min', float('nan')):6.3f} "
                      f"erc_min={res.get('l1_erc_scm_min', float('nan')):6.3f}", flush=True)

        entries = [dict(e, label=k) for k, e in field.items()]
        for row in rows:
            entries.append({"label": f"OURS p={row['penalty']:.2f} mix={row['downside_mix']:.2f}",
                            "sharpe_median": row["sharpe_median"],
                            "cvar_median": row["cvar_median"],
                            "failure_rate": row["failure_rate"],
                            "n_portfolios": row["n_portfolios"],
                            "n_failed": row["n_failed"]})
        scores = r7_score(entries, rank_sim.rank_pct)
        ordered = sorted(zip(scores, entries), key=lambda t: -t[0])
        print(f"\n  {'setting':28s} {'sharpe_med':>10s} {'cvar_med':>9s} "
              f"{'fail%':>7s} {'SCORE':>7s}")
        for score, e in ordered:
            print(f"  {e['label']:28s} {e['sharpe_median']:10.4f} "
                  f"{e['cvar_median']:9.4%} {e['failure_rate']:7.1%} {score:7.4f}")
        report["field"][name] = [dict(e, score=s) for s, e in ordered]
        report["settings"].extend(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
