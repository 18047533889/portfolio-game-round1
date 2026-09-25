"""Sweep the short budget by hand and score every setting.

The brief allows shortselling. This tool answers three questions with numbers
rather than arguments, for each setting of (anchor_penalty, short_cap):

  1. ACCOUNTING -- how many folds fail, and what annual return / max drawdown
     the teacher's own backtester produces (lookback 252, optimize_every 20,
     weight_drift TRUE, leverage 1).
  2. EXPOSURE -- how much of the granted short budget the optimizer actually
     uses, and what gross exposure that realises, because Net and Gross are the
     measurable quantities the leverage rule is about.
  3. COMPLIANCE -- the L1 distance from the delivered book to each of the three
     prohibited books, rebuilt from the same training window. A setting that
     drifts toward a prohibited book is disqualified no matter how it scores.

The settings are then placed in a simulated field of plausible submissions and
scored under the round-2 rank-percentile rule, so the ranking is computed
between candidates rather than in isolation.

Only the core is driven here, not the shipped file: the shipped file has its
parameters baked in, so a sweep has to run ahead of the freeze. The chosen
setting is frozen into configs/submission_round2.json and re-verified on the
shipped bytes by tools/round2_backtest_report.py.
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
sys.path.insert(0, str(ROOT / "research/round2"))
sys.path.insert(0, str(ROOT / "tools"))

LOOKBACK, STEP, LEVERAGE = 252, 20, 1.0
DATASETS = {
    "sp500": ("load_sp500_dataset", 20),
    "ftse100": ("load_ftse100_dataset", 64),
    "factors": ("load_factors_dataset", 5),
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BACKTEST = _load(ROOT / "tools/round2_backtest_report.py", "r2_backtest_report")
RANK_SIM = _load(ROOT / "research/round2/rank_sim.py", "r2_rank_sim")


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "median": float(np.median(v)),
            "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min())}


def evaluate(X: pd.DataFrame, penalty: float, cap: float) -> dict:
    """One walk-forward run of the core under the round-2 contract.

    The core is pinned through r2_eval.CORE_OVERRIDE rather than an instance
    attribute, because cross_val_predict clones the estimator and a clone does
    not carry attributes set after construction.
    """
    from skfolio.model_selection import WalkForward, cross_val_predict
    from r2_eval import Probe

    model = Probe(penalty=penalty, allow_short=cap > 0.0, short_cap=cap,
                  portfolio_params={"weight_drift": True, "name": "sweep"})
    model.raise_on_failure = False
    pred = cross_val_predict(model, X, cv=WalkForward(train_size=LOOKBACK,
                                                     test_size=STEP), n_jobs=1)
    weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios
               if getattr(p, "weights", None) is not None
               and p.__class__.__name__ != "FailedPortfolio"]
    return {"pred": pred, "weights": weights,
            "n_portfolios": len(pred.portfolios),
            "n_failed": int(pred.n_failed_portfolios),
            "n_fallback": int(pred.n_fallback_portfolios)}


def compliance(res: dict, x_all: np.ndarray, n_assets: int) -> dict:
    """L1 distance of every fold's book to the three prohibited books."""
    positions = list(range(LOOKBACK, x_all.shape[0] - STEP + 1, STEP))
    dist = {"ewp": [], "ivp": [], "scm_gmvp": []}
    exact = {"ewp": 0, "ivp": 0, "scm_gmvp": 0}
    for w, pos in zip(res["weights"], positions):
        train = x_all[pos - LOOKBACK:pos]
        for key, ref in (("ewp", BACKTEST._ewp(n_assets)),
                         ("ivp", BACKTEST._ivp(train)),
                         ("scm_gmvp", BACKTEST._scm_gmvp(train))):
            if ref is None:
                continue
            d = BACKTEST._dist(w, ref)
            dist[key].append(d)
            if d < 1e-9:
                exact[key] += 1
    return {"distance": {k: summarise(v) for k, v in dist.items()},
            "exact_hits": exact}


def exposure_stats(res: dict) -> dict:
    w = res["weights"]
    if not w:
        return {"short_max": float("nan"), "gross_max": float("nan"),
                "min_weight": float("nan"), "folds_with_shorts": 0}
    return {
        "short_max": float(max(np.maximum(-x, 0.0).sum() for x in w)),
        "gross_max": float(max(float(np.abs(x).sum()) for x in w)),
        "min_weight": float(min(float(x.min()) for x in w)),
        "folds_with_shorts": int(sum(int((x < 0).any()) for x in w)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalties", default="4.0,1.0,0.25,0.0")
    ap.add_argument("--caps", default="0.0,0.01,0.02,0.05,0.10,0.20,0.35,0.50")
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/round2_short_sweep.json")
    args = ap.parse_args()

    import r2_eval
    r2_eval.CORE_OVERRIDE = ROOT / "src/portfolio_game_round2/core.py"
    penalties = [float(p) for p in args.penalties.split(",")]
    caps = [float(c) for c in args.caps.split(",")]

    report: dict = {"contract": {"lookback": LOOKBACK, "optimize_every": STEP,
                                 "weight_drift": True, "leverage": LEVERAGE},
                    "settings": [], "field": {}}

    for name in [d.strip() for d in args.datasets.split(",")]:
        loader, cap_n = DATASETS[name]
        from skfolio.datasets import __dict__ as ds_dict
        panel = ds_dict[loader]().pct_change(fill_method=None).dropna().iloc[:, :cap_n]
        x_all = panel.to_numpy(dtype=float)
        print(f"\n{'='*130}\n{name}  {panel.shape}", flush=True)

        # ---- the simulated field, once per dataset
        field: dict[str, dict] = {}
        if not args.skip_field:
            for label, factory in RANK_SIM.field().items():
                field[label] = RANK_SIM.evaluate(factory, panel)
            print(f"  field of {len(field)} reference submissions evaluated", flush=True)

        rows = []
        for p in penalties:
            for c in caps:
                res = evaluate(panel, p, c)
                row = {"dataset": name, "penalty": p, "short_cap": c,
                       "n_portfolios": res["n_portfolios"],
                       "n_failed": res["n_failed"],
                       "n_fallback": res["n_fallback"],
                       "failure_rate": res["n_failed"] / max(1, res["n_portfolios"])}
                row.update(exposure_stats(res))
                row.update(compliance(res, x_all, cap_n))
                if res["n_portfolios"] > res["n_failed"]:
                    row["annual_return"] = float(res["pred"].annualized_mean)
                    row["max_drawdown"] = float(res["pred"].max_drawdown)
                    row["sharpe"] = float(res["pred"].annualized_sharpe_ratio)
                else:
                    row["annual_return"] = row["max_drawdown"] = row["sharpe"] = float("nan")
                rows.append(row)
                print(f"  p={p:6.2f} cap={c:5.2f} fail={row['n_failed']:4d}/"
                      f"{row['n_portfolios']:4d} annual={row['annual_return']:8.3%} "
                      f"dd={row['max_drawdown']:8.3%} short={row['short_max']:7.4f} "
                      f"gross={row['gross_max']:6.3f} "
                      f"L1gmvp={row['distance']['scm_gmvp'].get('median', float('nan')):6.3f}",
                      flush=True)

        # ---- rank-percentile score for every setting inside the field
        labels = list(field)
        results = {k: dict(field[k]) for k in labels}
        for row in rows:
            results[f"OURS p={row['penalty']:.2f} cap={row['short_cap']:.2f}"] = {
                "annual_return": row["annual_return"],
                "max_drawdown": row["max_drawdown"],
                "failure_rate": row["failure_rate"],
                "n_portfolios": row["n_portfolios"], "n_failed": row["n_failed"],
            }
        keys = list(results)
        r_ann = RANK_SIM.rank_pct([results[k]["annual_return"] for k in keys], True)
        r_dd = RANK_SIM.rank_pct([results[k]["max_drawdown"] for k in keys], False)
        r_fail = RANK_SIM.rank_pct([results[k]["failure_rate"] for k in keys], False)
        score = 0.15 * r_ann + 0.15 * r_dd + 0.70 * r_fail
        scored = []
        for i, k in enumerate(keys):
            entry = {**results[k], "rank_annual": float(r_ann[i]),
                     "rank_dd": float(r_dd[i]), "rank_fail": float(r_fail[i]),
                     "score": float(score[i]), "label": k}
            scored.append(entry)
        scored.sort(key=lambda e: -e["score"])
        report["field"][name] = scored
        print(f"\n  {'setting':26s} {'annual':>9s} {'maxdd':>8s} {'fail%':>7s} "
              f"{'r_ann':>6s} {'r_dd':>6s} {'r_fail':>7s} {'SCORE':>7s}")
        for e in scored:
            print(f"  {e['label']:26s} {e['annual_return']:9.4%} "
                  f"{e['max_drawdown']:8.4%} {e['failure_rate']:7.1%} "
                  f"{e['rank_annual']:6.3f} {e['rank_dd']:6.3f} "
                  f"{e['rank_fail']:7.3f} {e['score']:7.4f}")
        report["settings"].extend(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
