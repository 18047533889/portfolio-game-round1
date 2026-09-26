"""Cross-dataset score comparison: is any candidate consistently better?

The field simulation scores a candidate relative to the other entrants, so a
single dataset cannot tell an edge from noise. This prints each candidate's
score on every dataset side by side, plus the mean and the worst case, because
the shipped choice has to survive whichever of them the grader happens to use.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CANDIDATES = [
    "OURS p=4.00 cap=0.00", "OURS p=1.00 cap=0.00", "OURS p=0.25 cap=0.00",
    "OURS p=0.00 cap=0.00",
    "NCO", "SchurComplementary", "MinVariance(Denoise)", "MinVariance(Shrunk)",
    "RiskBudgeting", "MaxDiversification", "InverseVolatility", "EqualWeighted",
    "MinVariance(SCM)", "Random(Dirichlet)",
]


def main() -> None:
    sweep = json.loads((ROOT / "reports/round2_short_sweep.json").read_text(encoding="utf-8"))
    datasets = list(sweep["field"])

    table: dict[str, dict[str, dict]] = {}
    for name in datasets:
        for entry in sweep["field"][name]:
            table.setdefault(entry["label"], {})[name] = entry

    head = f"{'method':24s} " + " ".join(f"{d:>9s}" for d in datasets)
    print(head + f" {'mean':>8s} {'worst':>8s}  failures")
    print("-" * (len(head) + 22))

    rows = []
    for label in CANDIDATES:
        if label not in table:
            continue
        scores = [table[label][d]["score"] for d in datasets]
        worst_fail = max(table[label][d]["failure_rate"] for d in datasets)
        rows.append((sum(scores) / len(scores), label, scores, worst_fail))
    rows.sort(reverse=True)

    for mean, label, scores, worst_fail in rows:
        cells = " ".join(f"{v:9.4f}" for v in scores)
        fail = "none" if worst_fail == 0 else f"{worst_fail:.0%}"
        print(f"{label:24s} {cells} {mean:8.4f} {min(scores):8.4f}  {fail:>8s}")

    print(f"\nmodel rank of the shipped setting per dataset:")
    for name in datasets:
        ranked = sorted(sweep["field"][name], key=lambda e: -e["score"])
        labels = [e["label"] for e in ranked]
        pos = labels.index("OURS p=4.00 cap=0.00") + 1
        print(f"  {name:9s} {pos:3d} / {len(labels)}   (best: {labels[0]})")


if __name__ == "__main__":
    main()
