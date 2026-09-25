"""Rank-percentile simulation of the round-2 scoring rule.

The round-2 score is

    0.15 * rank_pct(annual_return)
  + 0.15 * rank_pct(-max_drawdown)
  + 0.70 * rank_pct(-failure_rate)

Rank percentiles are relative to the other entrants, so the score cannot be
computed from one candidate in isolation. This simulates the field with a pool
of plausible student submissions: the skfolio optimizers a student would reach
for first, including the prohibited ones (people will submit them anyway), plus
our own candidates.

This is a proxy for the field, not the field itself. The value of the exercise
is not the absolute score but the ranking of our candidates *within a plausible
population*, and whether the choice of anchor_penalty changes that ranking.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_eval import load_datasets, run  # noqa: E402

warnings.filterwarnings("ignore")

from skfolio.model_selection import WalkForward, cross_val_predict  # noqa: E402
from skfolio.optimization import (  # noqa: E402
    EqualWeighted, HierarchicalEqualRiskContribution, HierarchicalRiskParity,
    InverseVolatility, MaximumDiversification, MeanRisk,
    NestedClustersOptimization, Random, RiskBudgeting, SchurComplementary,
)
from skfolio.moments import DenoiseCovariance, ShrunkCovariance  # noqa: E402
from skfolio.prior import EmpiricalPrior  # noqa: E402


def field() -> dict:
    """A plausible population of submissions."""
    return {
        "EqualWeighted": lambda: EqualWeighted(),
        "InverseVolatility": lambda: InverseVolatility(),
        "MinVariance(SCM)": lambda: MeanRisk(),
        "MinVariance(Denoise)": lambda: MeanRisk(
            prior_estimator=EmpiricalPrior(covariance_estimator=DenoiseCovariance())),
        "MinVariance(Shrunk)": lambda: MeanRisk(
            prior_estimator=EmpiricalPrior(covariance_estimator=ShrunkCovariance())),
        "RiskBudgeting": lambda: RiskBudgeting(),
        "MaxDiversification": lambda: MaximumDiversification(),
        "SchurComplementary": lambda: SchurComplementary(),
        "NCO": lambda: NestedClustersOptimization(),
        "HRP": lambda: HierarchicalRiskParity(),
        "HERC": lambda: HierarchicalEqualRiskContribution(),
        "Random(Dirichlet)": lambda: Random(),
    }


def evaluate(factory, X) -> dict:
    cv = WalkForward(train_size=252, test_size=20)
    model = factory()
    model.set_params(portfolio_params={"weight_drift": True, "name": "f"})
    model.raise_on_failure = False
    try:
        pred = cross_val_predict(model, X, cv=cv, n_jobs=1)
    except Exception as exc:
        return {"annual_return": float("nan"), "max_drawdown": float("nan"),
                "failure_rate": 1.0, "n_portfolios": 0, "n_failed": 0,
                "error": f"{type(exc).__name__}"}
    n_total = len(pred)
    n_failed = int(getattr(pred, "n_failed_portfolios", 0) or 0)
    out = {"n_portfolios": n_total, "n_failed": n_failed,
           "failure_rate": n_failed / n_total if n_total else 1.0}
    # A run that failed any fold has its performance measured over a different
    # and smaller sample. HRP/HERC report annual +80% here off a single
    # surviving fold out of 403. Mixing that with clean runs would let the
    # worst entrants top the annual-return ranking, so performance is only
    # credited to a run in which every fold survived.
    if n_total > 0 and n_failed == 0:
        out["annual_return"] = float(pred.annualized_mean)
        out["max_drawdown"] = float(pred.max_drawdown)
    else:
        out["annual_return"] = float("nan")
        out["max_drawdown"] = float("nan")
    return out


def rank_pct(values: list[float], higher_is_better: bool,
             *, rtol: float = 1e-6) -> np.ndarray:
    """Percentile rank in [0,1]; 1.0 is best. NaNs score 0. Ties share a rank.

    Ties MUST share a rank, and "tie" has to include values that differ only in
    their last bits. Two versions of this function have now been wrong in the
    same direction:

      * an argsort version assigned ties in list order, fabricating a spread of
        r_fail from 0.13 to 1.00 across 14 methods that all had a 0% failure
        rate -- pure ordering decided the ranking;
      * rankdata alone still splits values that are equal to within 1e-16,
        which is what two short-cap settings produce when the cap is not
        binding. On the factors panel, cap = 0.00 and cap = 0.20 deliver the
        identical long-only book, yet differ in the solver's last iterate, and
        rankdata scored them 1.000 and 0.000. That is 0.30 of the composite
        score manufactured from floating-point noise.

    Values are therefore quantised to a relative tolerance before ranking, and
    anything inside that window is reported as tied. The window is 1e-6
    relative, far below any economically meaningful difference in a return or a
    drawdown, and far above solver roundoff.
    """
    v = np.asarray(values, dtype=float)
    out = np.zeros(len(v))
    ok = np.isfinite(v)
    if ok.sum() <= 1:
        return out
    sub = v[ok].copy()
    if not higher_is_better:
        sub = -sub
    scale = max(1.0, float(np.max(np.abs(sub))))
    keys = np.round(sub / (scale * rtol))
    # rankdata gives rank 1 to the worst and len(sub) to the best.
    r = rankdata(keys, method="average")
    out[ok] = (r - 1.0) / (len(sub) - 1.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sp500(20),ftse100(64),factors(5)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    panels = load_datasets([d.strip() for d in args.datasets.split(",")])

    report: dict = {"datasets": {}}
    for name, X in panels.items():
        print(f"\n{'='*94}\n{name}  {X.shape}")
        results: dict[str, dict] = {}
        for label, factory in field().items():
            results[label] = evaluate(factory, X)
        for p in (0.25, 1.0, 4.0, 16.0):
            results[f"OURS p={p}"] = run(X, p, "regularized")

        labels = list(results)
        annual = [results[k]["annual_return"] for k in labels]
        dd = [results[k]["max_drawdown"] for k in labels]
        fail = [results[k]["failure_rate"] for k in labels]
        r_ann = rank_pct(annual, True)
        r_dd = rank_pct(dd, False)
        r_fail = rank_pct(fail, False)
        score = 0.15 * r_ann + 0.15 * r_dd + 0.70 * r_fail

        table = {}
        print(f"  {'method':22s} {'annual':>9s} {'maxdd':>9s} {'fail%':>7s} "
              f"{'r_ann':>7s} {'r_dd':>7s} {'r_fail':>7s} {'SCORE':>7s}")
        order = np.argsort(-score)
        for i in order:
            k = labels[i]
            r = results[k]
            table[k] = {**r, "rank_annual": float(r_ann[i]), "rank_dd": float(r_dd[i]),
                        "rank_fail": float(r_fail[i]), "score": float(score[i])}
            print(f"  {k:22s} {r['annual_return']:9.4%} {r['max_drawdown']:9.4%} "
                  f"{r['failure_rate']:7.1%} {r_ann[i]:7.3f} {r_dd[i]:7.3f} "
                  f"{r_fail[i]:7.3f} {score[i]:7.4f}")
        report["datasets"][name] = table

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
