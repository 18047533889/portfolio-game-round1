"""Final round-5 selection: every measured setting scored in ONE pool per dataset.

Rank percentiles are relative to the pool, so settings measured in different
runs are not comparable unless they are re-scored together. This rebuilds a
single pool per dataset (the twelve reference submissions plus one candidate at
a time) and reports mean and worst case across datasets, because with three
datasets and a rank-based score a setting that wins on average while losing
badly somewhere is not a safe choice -- round 3 produced exactly that pattern
and it was the reason for switching away from chasing the best single number.

It also prints the compliance margin next to the score, since round 5 is the
first round in which lowering risk and moving toward the prohibited SCM minimum
variance book are the same action.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/round2"))

import rank_sim  # noqa: E402

SOURCES = ["round5_sweep.json", "round5_fine.json"]


def score_against(opponents: list[dict], row: dict) -> float:
    entries = opponents + [{"label": "ours",
                            "annual_volatility": row["annual_volatility"],
                            "max_drawdown": row["max_drawdown"],
                            "cvar_95": row["cvar_95"],
                            "failure_rate": row["failure_rate"]}]
    r_vol = rank_sim.rank_pct([e["annual_volatility"] for e in entries], False)
    r_dd = rank_sim.rank_pct([e["max_drawdown"] for e in entries], False)
    r_cv = rank_sim.rank_pct([e["cvar_95"] for e in entries], False)
    r_fail = rank_sim.rank_pct([e["failure_rate"] for e in entries], False)
    return float(0.10 * r_vol[-1] + 0.10 * r_dd[-1] + 0.10 * r_cv[-1]
                 + 0.70 * r_fail[-1])


def main() -> None:
    reports = ROOT / "reports"
    field: dict[str, list[dict]] = {}
    settings: list[dict] = []
    for name in SOURCES:
        path = reports / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not field and "field" in data:
            for dataset, entries in data["field"].items():
                field[dataset] = [e for e in entries
                                  if not e["label"].startswith("OURS")]
        settings.extend(data.get("settings", []))

    datasets = list(field)
    print(f"opponent pool: { {d: len(v) for d, v in field.items()} }")
    print(f"measured settings: {len(settings)}\n")

    grouped: dict[str, dict[str, dict]] = {}
    for row in settings:
        if row["dataset"] not in field:
            continue
        key = f"p={row['penalty']:.3f} mix={row['downside_mix']:.2f}"
        grouped.setdefault(key, {})[row["dataset"]] = row

    rows = []
    for key, per in grouped.items():
        if len(per) != len(datasets):
            continue
        scores = [score_against(field[d], per[d]) for d in datasets]
        margins = [per[d].get("l1_scm_gmvp", float("nan")) for d in datasets]
        rows.append((float(np.mean(scores)), float(np.min(scores)), key, scores, margins))
    rows.sort(reverse=True)

    head = f"{'setting':22s} " + " ".join(f"{d:>9s}" for d in datasets)
    print(head + f" {'mean':>8s} {'worst':>8s}  worst L1_gmvp")
    print("-" * (len(head) + 32))
    for mean, worst, key, scores, margins in rows:
        cells = " ".join(f"{s:9.4f}" for s in scores)
        margin = min(m for m in margins if np.isfinite(m)) if any(
            np.isfinite(m) for m in margins) else float("nan")
        print(f"{key:22s} {cells} {mean:8.4f} {worst:8.4f}  {margin:.3f}")

    print("\n--- reference methods in the same pool ---")
    for label in ("NCO", "MinVariance(Denoise)", "MinVariance(SCM)",
                  "MinVariance(Shrunk)", "SchurComplementary",
                  "MaxDiversification", "RiskBudgeting", "HRP"):
        per = {d: next((e for e in field[d] if e["label"] == label), None)
               for d in datasets}
        if any(v is None for v in per.values()):
            continue
        scores = [per[d]["score"] for d in datasets]
        cells = " ".join(f"{s:9.4f}" for s in scores)
        fails = max(per[d]["failure_rate"] for d in datasets)
        print(f"{label:22s} {cells} {np.mean(scores):8.4f} {min(scores):8.4f}  "
              f"fail<={fails:.1%}")


if __name__ == "__main__":
    main()
