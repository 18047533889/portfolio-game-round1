"""Scan anchor_penalty under the round-2 contract (weight_drift on).

Question: round 1 selected anchor_penalty=0.25 on a drawdown criterion, when
Sharpe was 10% of the score. Round 2 drops Sharpe entirely and doubles annual
return to 15%, so the same knob now trades annual return against drawdown with
equal weight on each. This measures where that trade actually lands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r2_eval import load_datasets, reference_equal_weight, run  # noqa: E402

CONFIGS: list[tuple[str, float, str]] = [
    ("regularized", 0.0, "p=0(=minvar)"),
    ("regularized", 0.25, "shipped-p=0.25"),
    ("regularized", 1.0, "p=1"),
    ("regularized", 4.0, "p=4"),
    ("regularized", 16.0, "p=16"),
    ("regularized", 64.0, "p=64"),
    ("regularized", 1e6, "p=1e6(ceil)"),
    ("hrp", 0.0, "hrp"),
    ("equal_weight", 0.0, "EQUAL-WEIGHT ref"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sp500(20),ftse100(64),factors(5)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panels = load_datasets([d.strip() for d in args.datasets.split(",")])
    report: dict = {"contract": {"lookback": 252, "step": 20, "weight_drift": True},
                    "datasets": {}}

    for name, X in panels.items():
        print(f"\n{'='*78}\n{name}  {X.shape}  {X.index.min().date()} -> {X.index.max().date()}")
        block: dict = {"shape": list(X.shape), "configs": {}}
        ref = reference_equal_weight(X)
        block["equal_weight_reference"] = ref
        print(f"  {'equal-weight ref':22s} "
              f"ann={ref['annual_return']:8.4%} dd={ref['max_drawdown']:8.4%} "
              f"sh={ref['sharpe']:6.3f}")
        print(f"  {'config':22s} {'annual':>9s} {'maxdd':>9s} {'sharpe':>8s} {'fail':>6s}")
        for method, penalty, label in CONFIGS:
            m = run(X, penalty, method)
            block["configs"][label] = m
            print(f"  {label:22s} {m['annual_return']:9.4%} {m['max_drawdown']:9.4%} "
                  f"{m['sharpe']:8.3f} {m['n_failed']:3d}/{m['n_portfolios']}")
        report["datasets"][name] = block

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
