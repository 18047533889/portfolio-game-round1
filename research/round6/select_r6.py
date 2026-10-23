"""Round-6 setting selection across datasets, in one rank pool.

Two rules, both learned the hard way in earlier rounds.

  * THE POOL MUST CONTAIN ONE ENTRY PER METHOD. Round 2 ranked our own 32
    near-duplicate parameter settings inside the opponent pool, which made each
    of them look mediocre because they were competing with their own siblings.
    Here the pool is the 12 reference methods plus the settings under
    comparison, added once each.

  * A SETTING IS JUDGED ON ITS WORST DATASET, NOT ITS MEAN. Every round so far
    has produced a parameter whose sign flips across datasets -- round 2's
    penalty, round 3's leverage, round 5's downside blend. The mean rewards
    whichever dataset has the widest rank spread; the minimum is what a hidden
    test set is actually able to expose.

A per-fold compliance floor is applied BEFORE ranking, so a setting that scores
well by collapsing onto a prohibited book is excluded rather than rewarded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/round2"))

FLOOR = 0.10  # same floor as round 5: 3.3x the distance at which round 2
              # diagnosed an actual violation, applied to every banned book.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=ROOT / "reports/round6_sweep.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import rank_sim

    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    datasets = list(sweep["field"])
    rows = {(r["penalty"], r["downside_mix"]): {} for r in sweep["settings"]}
    for r in sweep["settings"]:
        rows[(r["penalty"], r["downside_mix"])][r["dataset"]] = r

    print(f"opponent pool: { {k: sum(1 for e in v if not e['label'].startswith('OURS')) for k, v in sweep['field'].items()} }")
    print(f"measured settings: {len(rows)}\n")

    header = f"{'setting':22s} " + " ".join(f"{d:>9s}" for d in datasets) + \
             f" {'mean':>8s} {'worst':>8s}  {'minGMVP':>8s} {'minERC':>7s}"
    print(header)
    print("-" * len(header))

    table = []
    for key, per_ds in rows.items():
        if len(per_ds) != len(datasets):
            continue
        scores, gmvp, erc = [], [], []
        for d in datasets:
            entry = next(e for e in sweep["field"][d]
                         if e["label"] == f"OURS p={key[0]:.2f} mix={key[1]:.2f}")
            scores.append(entry["score"])
            gmvp.append(per_ds[d].get("l1_scm_gmvp_min", np.nan))
            erc.append(per_ds[d].get("l1_erc_scm_min", np.nan))
        table.append({"key": key, "scores": scores,
                      "mean": float(np.mean(scores)),
                      "worst": float(np.min(scores)),
                      "min_gmvp": float(np.nanmin(gmvp)),
                      "min_erc": float(np.nanmin(erc)),
                      "admissible": bool(np.nanmin(gmvp) >= FLOOR
                                         and np.nanmin(erc) >= FLOOR)})

    for row in sorted(table, key=lambda r: -r["worst"]):
        k = row["key"]
        flag = "" if row["admissible"] else "  <- below floor"
        print(f"p={k[0]:<6.2f} mix={k[1]:<4.2f}      " +
              " ".join(f"{v:9.4f}" for v in row["scores"]) +
              f" {row['mean']:8.4f} {row['worst']:8.4f}  "
              f"{row['min_gmvp']:8.3f} {row['min_erc']:7.3f}{flag}")

    print("\n--- reference methods in the same pool ---")
    for d in datasets:
        field = [e for e in sweep["field"][d] if not e["label"].startswith("OURS")]
        field.sort(key=lambda e: -e["score"])
        print(f"  {d:9s} " + "  ".join(
            f"{e['label']}={e['score']:.4f}(fail {e['failure_rate']:.1%})"
            for e in field[:4]))

    if args.out:
        args.out.write_text(json.dumps(
            {"floor": FLOOR, "datasets": datasets, "rows": table}, indent=2),
            encoding="utf-8")
        print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
