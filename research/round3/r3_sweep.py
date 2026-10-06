"""Round-3 evaluation harness and parameter sweep.

Round 3 changes the contract in two ways that both touch the choice of
parameters, so the round-2 selection cannot be inherited:

  * lookback 252 -> 126. The estimation window halved, so half_life -- which is
    a decay scale in days, not a fraction of the window -- has to be re-derived
    rather than scaled blindly.
  * scoring: annual return 20%, annual volatility 10%, failure rate 70%.
    Max drawdown is gone and volatility takes its 10%. Volatility is the one
    quantity this objective already minimises, so the term that replaces
    drawdown is aligned with the method rather than against it.
  * leverage 1 -> 1.5, which grants an aggregate short budget of 0.25. Whether
    the optimizer wants any of it is a measurement, not an assumption: at a high
    anchor penalty the anchor term dominates and the budget goes unused.

Everything is measured through the instructor's own accounting path
(WalkForward over the round's lookback/step with weight_drift on), and each
candidate is scored inside a simulated field so the ranking is computed between
candidates rather than in isolation.
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

LOOKBACK, STEP = 126, 20
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "factors": ("load_factors_dataset", 5),
}
CORE = ROOT / "src/portfolio_game_round3/core.py"


def load_core(path: Path | None = None):
    spec = importlib.util.spec_from_file_location("core_r3", path or CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def panel(name: str) -> pd.DataFrame:
    from skfolio.datasets import __dict__ as ds_dict
    loader, cap = DATASETS[name]
    return ds_dict[loader]().pct_change(fill_method=None).dropna().iloc[:, :cap]


def measure(core, X: pd.DataFrame, *, half_life: float, recent_mix: float,
            penalty: float, leverage: float, method: str = "regularized") -> dict:
    """One walk-forward run under the round-3 contract."""
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
                                   gross_leverage=leverage)
            self.weights_ = result["weights"]
            self.diagnostics_ = result["diagnostics"]
            return self

    model = Probe(portfolio_params={"weight_drift": True, "name": "r3"})
    model.raise_on_failure = False
    pred = cross_val_predict(model, X, cv=WalkForward(train_size=LOOKBACK,
                                                      test_size=STEP), n_jobs=1)

    weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
               if getattr(p, "weights", None) is not None
               and p.__class__.__name__ != "FailedPortfolio"]
    n_total = len(pred.portfolios)
    n_failed = int(pred.n_failed_portfolios)
    out = {
        "n_portfolios": n_total, "n_failed": n_failed,
        "failure_rate": n_failed / n_total if n_total else 1.0,
        "n_fallback": int(pred.n_fallback_portfolios),
    }
    if weights:
        out["short_max"] = float(max(np.maximum(-w, 0.0).sum() for w in weights))
        out["gross_max"] = float(max(float(np.abs(w).sum()) for w in weights))
        out["folds_using_shorts"] = int(sum(int((w < 0).any()) for w in weights))
    else:
        out["short_max"] = out["gross_max"] = float("nan")
        out["folds_using_shorts"] = 0
    if n_total - n_failed > 0:
        out["annual_return"] = float(pred.annualized_mean)
        out["annual_volatility"] = float(pred.annualized_standard_deviation)
        out["max_drawdown"] = float(pred.max_drawdown)
    else:
        out["annual_return"] = float("nan")
        out["annual_volatility"] = float("nan")
        out["max_drawdown"] = float("nan")
    return out


def r3_score(entries: list[dict], rank_pct) -> list[float]:
    """0.20 * rank(annual return) + 0.10 * rank(-volatility) + 0.70 * rank(-failure)."""
    r_ann = rank_pct([e["annual_return"] for e in entries], True)
    r_vol = rank_pct([e["annual_volatility"] for e in entries], False)
    r_fail = rank_pct([e["failure_rate"] for e in entries], False)
    return list(0.20 * r_ann + 0.10 * r_vol + 0.70 * r_fail)


def evaluate_opponent(factory, X: pd.DataFrame) -> dict:
    """Run a reference skfolio optimizer under the round-3 contract.

    Deliberately not reusing rank_sim.evaluate: that one hard-codes
    train_size=252, which is the round-2 lookback and would silently measure
    every opponent on the wrong window.
    """
    from skfolio.model_selection import WalkForward, cross_val_predict

    model = factory()
    model.set_params(portfolio_params={"weight_drift": True, "name": "f"})
    model.raise_on_failure = False
    try:
        pred = cross_val_predict(model, X, cv=WalkForward(train_size=LOOKBACK,
                                                          test_size=STEP), n_jobs=1)
    except Exception as exc:  # noqa: BLE001
        return {"annual_return": float("nan"), "annual_volatility": float("nan"),
                "failure_rate": 1.0, "n_portfolios": 0, "n_failed": 0,
                "error": type(exc).__name__}
    n_total = len(pred.portfolios)
    n_failed = int(pred.n_failed_portfolios)
    out = {"n_portfolios": n_total, "n_failed": n_failed,
           "failure_rate": n_failed / n_total if n_total else 1.0}
    # Performance is only credited to a run in which every fold survived: a
    # partial run measures a different, smaller sample, and the surviving folds
    # of a broken method have repeatedly looked better than clean ones.
    if n_total > 0 and n_failed == 0:
        out["annual_return"] = float(pred.annualized_mean)
        out["annual_volatility"] = float(pred.annualized_standard_deviation)
        out["max_drawdown"] = float(pred.max_drawdown)
    else:
        out["annual_return"] = float("nan")
        out["annual_volatility"] = float("nan")
        out["max_drawdown"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", default="0.25,1.0,4.0,16.0")
    ap.add_argument("--leverages", default="1.0,1.5")
    ap.add_argument("--half-life", type=float, default=30.0)
    ap.add_argument("--recent-mix", type=float, default=0.25)
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--skip-field", action="store_true",
                    help="measure only our own settings; the opponent field does not "
                         "depend on our parameters, so a tuned grid can be re-ranked "
                         "against the field already stored by a full run")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/round3_sweep.json")
    args = ap.parse_args()

    import rank_sim
    core = load_core()
    penalties = [float(x) for x in args.penalties.split(",")]
    leverages = [float(x) for x in args.leverages.split(",")]

    report = {"contract": {"lookback": LOOKBACK, "optimize_every": STEP,
                           "weight_drift": True,
                           "scoring": {"annual_return": 0.20,
                                       "annual_volatility": 0.10,
                                       "failure_rate": 0.70},
                           "half_life": args.half_life,
                           "recent_mix": args.recent_mix},
              "settings": [], "field": {}}

    for name in [d.strip() for d in args.datasets.split(",")]:
        X = panel(name)
        print(f"\n{'='*118}\n{name}  {X.shape}", flush=True)

        field = {}
        if not args.skip_field:
            for label, factory in rank_sim.field().items():
                entry = evaluate_opponent(factory, X)
                entry["label"] = label
                field[label] = entry
        else:
            print("  (field skipped: our settings only)", flush=True)

        rows = []
        for leverage in leverages:
            for penalty in penalties:
                res = measure(core, X, half_life=args.half_life,
                              recent_mix=args.recent_mix, penalty=penalty,
                              leverage=leverage)
                row = {"dataset": name, "penalty": penalty, "leverage": leverage, **res}
                rows.append(row)
                print(f"  lev={leverage:4.2f} p={penalty:6.2f} fail={res['n_failed']:3d}/"
                      f"{res['n_portfolios']:4d} annual={res['annual_return']:8.4%} "
                      f"vol={res['annual_volatility']:7.4%} dd={res['max_drawdown']:8.4%} "
                      f"short={res['short_max']:6.4f} gross={res['gross_max']:6.3f} "
                      f"folds_with_shorts={res['folds_using_shorts']:4d}", flush=True)

        entries = [dict(e, label=k) for k, e in field.items()]
        for row in rows:
            entries.append({"label": f"OURS p={row['penalty']:.2f} lev={row['leverage']:.2f}",
                            "annual_return": row["annual_return"],
                            "annual_volatility": row["annual_volatility"],
                            "failure_rate": row["failure_rate"],
                            "n_portfolios": row["n_portfolios"],
                            "n_failed": row["n_failed"]})
        scores = r3_score(entries, rank_sim.rank_pct)
        ordered = sorted(zip(scores, entries), key=lambda t: -t[0])
        print(f"\n  {'setting':28s} {'annual':>9s} {'vol':>8s} {'fail%':>7s} {'SCORE':>7s}")
        for score, e in ordered:
            print(f"  {e['label']:28s} {e['annual_return']:9.4%} "
                  f"{e['annual_volatility']:8.4%} {e['failure_rate']:7.1%} {score:7.4f}")
        report["field"][name] = [dict(e, score=s) for s, e in ordered]
        report["settings"].extend(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
