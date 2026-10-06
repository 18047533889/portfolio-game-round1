"""Final round-3 selection: every measured setting scored in ONE opponent pool.

The sweeps were run in separate processes, so their score columns are not
comparable to each other -- rank percentiles are relative to the pool, and each
process had a different set of our own settings in it. This rebuilds a single
pool (the twelve reference submissions, plus one candidate at a time) and scores
every setting we measured against it, which is the only way the columns become
comparable.

It also reports the worst case across datasets next to the mean, because with
three datasets and a rank-based score a setting that wins on average while
losing badly somewhere is not a safe choice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/round2"))

import rank_sim  # noqa: E402

SOURCES = ["round3_sweep.json", "round3_highp.json", "round3_decay.json"]


def score_against(opponents: list[dict], row: dict) -> float:
    entries = opponents + [{"label": "ours",
                            "annual_return": row["annual_return"],
                            "annual_volatility": row["annual_volatility"],
                            "failure_rate": row["failure_rate"]}]
    r_ann = rank_sim.rank_pct([e["annual_return"] for e in entries], True)
    r_vol = rank_sim.rank_pct([e["annual_volatility"] for e in entries], False)
    r_fail = rank_sim.rank_pct([e["failure_rate"] for e in entries], False)
    return float(0.20 * r_ann[-1] + 0.10 * r_vol[-1] + 0.70 * r_fail[-1])


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

    by_setting: dict[str, dict[str, dict]] = {}
    for row in settings:
        if row["dataset"] not in field:
            continue
        key = (f"p={row['penalty']:.2f} lev={row['leverage']:.2f}"
               f" hl={row['half_life']:.0f} rm={row['recent_mix']:.2f}"
               if "half_life" in row else
               f"p={row['penalty']:.2f} lev={row['leverage']:.2f}")
        by_setting.setdefault(key, {})[row["dataset"]] = row

    rows = []
    for key, per in by_setting.items():
        if len(per) != len(datasets):
            continue
        scores = [score_against(field[d], per[d]) for d in datasets]
        rows.append((float(np.mean(scores)), float(np.min(scores)), key, scores))
    rows.sort(reverse=True)

    head = f"{'setting':34s} " + " ".join(f"{d:>9s}" for d in datasets)
    print(head + f" {'mean':>8s} {'worst':>8s}")
    print("-" * (len(head) + 18))
    for mean, worst, key, scores in rows[:20]:
        cells = " ".join(f"{s:9.4f}" for s in scores)
        print(f"{key:34s} {cells} {mean:8.4f} {worst:8.4f}")

    print("\n--- reference methods in the same pool ---")
    for label in ("MaxDiversification", "MinVariance(Shrunk)", "Random(Dirichlet)",
                  "RiskBudgeting", "MinVariance(SCM)", "NCO", "HRP"):
        per = {d: next((e for e in field[d] if e["label"] == label), None)
               for d in datasets}
        if any(v is None for v in per.values()):
            continue
        scores = [per[d]["score"] for d in datasets]
        cells = " ".join(f"{s:9.4f}" for s in scores)
        fails = max(per[d]["failure_rate"] for d in datasets)
        print(f"{label:34s} {cells} {np.mean(scores):8.4f} {min(scores):8.4f}  "
              f"fail<={fails:.1%}")


if __name__ == "__main__":
    main()
