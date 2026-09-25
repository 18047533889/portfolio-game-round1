"""Round-2 cross-section of skfolio's own optimizers, under the grader contract.

Purpose: establish what the *library's* methods achieve on this exact
evaluation, so that a hand-written method is only shipped if it is at least as
good. The three prohibited books are included at the bottom purely as
references -- they are never candidates.

All runs use WalkForward(252, 20) and weight_drift=True.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_eval import load_datasets, run  # noqa: E402

warnings.filterwarnings("ignore")

from skfolio.model_selection import WalkForward, cross_val_predict  # noqa: E402
from skfolio.optimization import (  # noqa: E402
    EqualWeighted, HierarchicalEqualRiskContribution, HierarchicalRiskParity,
    InverseVolatility, MaximumDiversification, MeanRisk,
    NestedClustersOptimization, RiskBudgeting, SchurComplementary,
)
from skfolio.prior import EmpiricalPrior  # noqa: E402
from skfolio.moments import DenoiseCovariance  # noqa: E402

ALLOWED = {
    "HRP": lambda: HierarchicalRiskParity(),
    "HERC": lambda: HierarchicalEqualRiskContribution(),
    "NCO": lambda: NestedClustersOptimization(),
    "MaxDiv": lambda: MaximumDiversification(),
    "RiskBudgeting": lambda: RiskBudgeting(),
    "SchurComp": lambda: SchurComplementary(),
    "MeanRisk+Denoise": lambda: MeanRisk(prior_estimator=EmpiricalPrior(
        covariance_estimator=DenoiseCovariance())),
}
BANNED_REFERENCE = {
    "_ref EqualWeighted": lambda: EqualWeighted(),
    "_ref InverseVol": lambda: InverseVolatility(),
    "_ref MeanRisk(SCM)": lambda: MeanRisk(),
}


def evaluate(factory, X) -> dict:
    cv = WalkForward(train_size=252, test_size=20)
    model = factory()
    model.set_params(portfolio_params={"weight_drift": True, "name": "x"})
    model.raise_on_failure = False
    try:
        pred = cross_val_predict(model, X, cv=cv, n_jobs=1)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    n_total = len(pred)
    n_failed = int(getattr(pred, "n_failed_portfolios", 0) or 0)
    out = {"n_portfolios": n_total, "n_failed": n_failed,
           "failure_rate": n_failed / n_total if n_total else 1.0}
    if n_total - n_failed > 0:
        out["annual_return"] = float(pred.annualized_mean)
        out["max_drawdown"] = float(pred.max_drawdown)
        out["sharpe"] = float(pred.annualized_sharpe_ratio)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sp500(20),ftse100(64),factors(5)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    panels = load_datasets([d.strip() for d in args.datasets.split(",")])
    report: dict = {"datasets": {}}
    for name, X in panels.items():
        print(f"\n{'='*80}\n{name}  {X.shape}")
        block = {"shape": list(X.shape), "methods": {}}
        print(f"  {'method':20s} {'annual':>9s} {'maxdd':>9s} {'sharpe':>7s} {'fail':>10s}")
        for label, factory in {**ALLOWED, **BANNED_REFERENCE}.items():
            m = evaluate(factory, X)
            block["methods"][label] = m
            if "error" in m:
                print(f"  {label:20s} ERROR {m['error'][:50]}")
            else:
                print(f"  {label:20s} {m['annual_return']:9.4%} {m['max_drawdown']:9.4%} "
                      f"{m['sharpe']:7.3f} {m['n_failed']:4d}/{m['n_portfolios']}")
        mine = run(X, 0.25, "regularized")
        block["methods"]["OURS p=0.25"] = mine
        print(f"  {'OURS p=0.25':20s} {mine['annual_return']:9.4%} {mine['max_drawdown']:9.4%} "
              f"{mine['sharpe']:7.3f} {mine['n_failed']:4d}/{mine['n_portfolios']}")
        report["datasets"][name] = block
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
