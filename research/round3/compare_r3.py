"""Cross-dataset comparison of round-3 settings, and the leverage question.

Two decisions have to be made from this sweep and neither is obvious from a
single dataset:

  1. which anchor_penalty to freeze, given that the scoring now weights annual
     return 20% and volatility 10% -- the return end of the frontier is worth
     twice the risk end, which is the opposite incentive from round 2's ;
  2. whether to use the leverage-1.5 short budget at all.

(2) is the more consequential one. Using the budget mechanically lowers both
volatility and return, so whether it helps depends entirely on the weights, and
the weights changed this round. The comparison is therefore made on the score,
per dataset and averaged, rather than on either quantity alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sweep = json.loads((ROOT / "reports/round3_sweep.json").read_text(encoding="utf-8"))
    datasets = list(sweep["field"])

    ours: dict[str, dict[str, dict]] = {}
    for name in datasets:
        for entry in sweep["field"][name]:
            if entry["label"].startswith("OURS"):
                ours.setdefault(entry["label"], {})[name] = entry
    others: dict[str, dict[str, dict]] = {}
    for name in datasets:
        for entry in sweep["field"][name]:
            if not entry["label"].startswith("OURS"):
                others.setdefault(entry["label"], {})[name] = entry

    print("=== our settings, score per dataset ===")
    head = f"{'setting':26s} " + " ".join(f"{d:>9s}" for d in datasets)
    print(head + f" {'mean':>8s} {'worst':>8s}")
    print("-" * (len(head) + 18))
    ranked = []
    for label, per in ours.items():
        if len(per) != len(datasets):
            continue
        scores = [per[d]["score"] for d in datasets]
        ranked.append((sum(scores) / len(scores), min(scores), label, scores))
    ranked.sort(reverse=True)
    for mean, worst, label, scores in ranked:
        cells = " ".join(f"{s:9.4f}" for s in scores)
        print(f"{label:26s} {cells} {mean:8.4f} {worst:8.4f}")

    print("\n=== reference methods, for scale ===")
    ref = []
    for label, per in others.items():
        if len(per) != len(datasets):
            continue
        if max(per[d]["failure_rate"] for d in datasets) > 0:
            continue
        scores = [per[d]["score"] for d in datasets]
        ref.append((sum(scores) / len(scores), label, scores))
    ref.sort(reverse=True)
    for mean, label, scores in ref[:6]:
        cells = " ".join(f"{s:9.4f}" for s in scores)
        print(f"{label:26s} {cells} {mean:8.4f}")

    print("\n=== the leverage decision: same penalty, budget used or not ===")
    print(f"{'penalty':>8s} {'mean(lev=1.0)':>14s} {'mean(lev=1.5)':>14s} "
          f"{'delta':>8s}   per-dataset delta")
    for penalty in sorted({float(e["penalty"]) for e in sweep["settings"]}):
        a, b, deltas = [], [], []
        for name in datasets:
            x = ours.get(f"OURS p={penalty:.2f} lev=1.00", {}).get(name)
            y = ours.get(f"OURS p={penalty:.2f} lev=1.50", {}).get(name)
            if x is None or y is None:
                continue
            a.append(x["score"])
            b.append(y["score"])
            deltas.append(y["score"] - x["score"])
        if not deltas:
            continue
        cells = ", ".join(f"{d}={v:+.4f}" for d, v in zip(datasets, deltas))
        print(f"{penalty:8.2f} {np.mean(a):14.4f} {np.mean(b):14.4f} "
              f"{np.mean(deltas):+8.4f}   {cells}")

    print("\n=== what the budget actually does to the measured quantities ===")
    for name in datasets:
        x = ours.get("OURS p=4.00 lev=1.00", {}).get(name)
        y = ours.get("OURS p=4.00 lev=1.50", {}).get(name)
        if x and y:
            print(f"  {name:9s} p=4.00  annual {x['annual_return']:.4%} -> "
                  f"{y['annual_return']:.4%} ({y['annual_return']-x['annual_return']:+.4%})   "
                  f"vol {x['annual_volatility']:.4%} -> {y['annual_volatility']:.4%} "
                  f"({y['annual_volatility']-x['annual_volatility']:+.4%})")


if __name__ == "__main__":
    main()
