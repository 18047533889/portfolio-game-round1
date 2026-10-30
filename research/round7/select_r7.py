"""Round-7 setting selection: one pool per candidate, identically built.

The trap this file exists to avoid has now appeared in three rounds. Rank
percentiles are relative to the pool they are computed in, so putting our own
twenty-five near-duplicate parameter settings into the pool alongside the twelve
reference methods makes every one of them look mediocre -- they spend their rank
slots on each other. Round 2 did exactly that and produced a conclusion that had
to be retracted.

The fix here is stricter than "one entry per method": every candidate, ours or a
reference, is scored in the SAME 13-entry pool -- the twelve references plus that
one candidate. So a candidate's score answers one question, "how does this book
rank against the reference field", and the answers are comparable across
candidates.

A per-fold compliance floor is applied first. Two details matter:

  * the SCM-qualified prohibitions do not exist when n >= T. On the 60- and
    64-asset panels at lookback 60 the sample covariance is singular, so those
    folds are counted as UNDEFINED and contribute no constraint. Silently
    treating them as distance zero would fail every setting; silently treating
    them as satisfied would be permissive. They are counted and printed.
  * the minimum over folds is an extreme order statistic and therefore noisy, so
    the margin over the floor is printed next to it rather than reduced to a
    pass/fail bit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/round2"))

FLOOR = 0.10  # 3.3x the distance at which round 2 diagnosed an actual violation


def score_against_field(entry: dict, field: list[dict], rank_pct) -> float:
    pool = field + [entry]
    return float(0.10 * rank_pct([e["sharpe_median"] for e in pool], True)[-1]
                 + 0.20 * rank_pct([e["cvar_median"] for e in pool], False)[-1]
                 + 0.70 * rank_pct([e["failure_rate"] for e in pool], False)[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+",
                    default=[str(ROOT / "reports/round7_sweep.json"),
                             str(ROOT / "reports/round7_fine.json")])
    ap.add_argument("--out", type=Path, default=ROOT / "reports/round7_selection.json")
    args = ap.parse_args()

    import rank_sim

    sources = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.sweeps]
    datasets = list(sources[0]["field"])
    rows: dict[tuple, dict] = {}
    for src in sources:
        for r in src["settings"]:
            rows[(r["dataset"], round(r["penalty"], 4), round(r["downside_mix"], 4))] = r
    field = {ds: [e for e in sources[0]["field"][ds]
                  if not e["label"].startswith("OURS")] for ds in datasets}

    print(f"reference field per dataset: {len(field[datasets[0]])}")
    print("every candidate is scored against those references PLUS itself (13 entries)\n")

    print("--- reference methods, same 13-entry pool ---")
    for ds in datasets:
        scored = []
        for e in field[ds]:
            others = [x for x in field[ds] if x["label"] != e["label"]]
            scored.append((score_against_field(e, others, rank_sim.rank_pct),
                           e["label"], e.get("failure_rate", 0.0)))
        scored.sort(key=lambda t: -t[0])
        print(f"  {ds:9s} " + "  ".join(f"{n}={v:.4f}(fail {f:.1%})"
                                        for v, n, f in scored[:3]))

    table = []
    for p, m in sorted({(k[1], k[2]) for k in rows if k[0] == datasets[0]}):
        if any((ds, p, m) not in rows for ds in datasets):
            continue
        scores, gmvp, erc, undefined = [], [], [], 0
        for ds in datasets:
            r = rows[(ds, p, m)]
            entry = {"sharpe_median": r["sharpe_median"],
                     "cvar_median": r["cvar_median"],
                     "failure_rate": r["failure_rate"]}
            scores.append(score_against_field(entry, field[ds], rank_sim.rank_pct))
            gmvp.append(r.get("l1_scm_gmvp_min", np.nan))
            erc.append(r.get("l1_erc_scm_min", np.nan))
            undefined += int(r.get("scm_reference_undefined_folds", 0) > 0)
        g = np.asarray(gmvp, float)
        e = np.asarray(erc, float)
        g_ok = g[~np.isnan(g)]
        e_ok = e[~np.isnan(e)]
        table.append({
            "penalty": p, "downside_mix": m, "scores": scores,
            "mean": float(np.mean(scores)), "worst": float(np.min(scores)),
            "min_gmvp": float(np.nanmin(g)) if g_ok.size else float("nan"),
            "min_erc": float(np.nanmin(e)) if e_ok.size else float("nan"),
            "gmvp_margin": float(np.nanmin(g) / FLOOR) if g_ok.size else float("nan"),
            "erc_margin": float(np.nanmin(e) / FLOOR) if e_ok.size else float("nan"),
            "datasets_with_undefined_reference": int(undefined),
            "admissible": bool(g_ok.size and e_ok.size
                               and np.all(g_ok >= FLOOR) and np.all(e_ok >= FLOOR)),
        })

    header = (f"{'p':>7s} {'mix':>4s} " + " ".join(f"{d:>9s}" for d in datasets)
              + f" {'mean':>8s} {'worst':>8s} {'GMVP':>6s} {'ERC':>6s} "
                f"{'xfl':>5s}  admissible")
    print(f"\n{header}")
    print("-" * len(header))
    for t in sorted(table, key=lambda r: -r["worst"]):
        print(f"{t['penalty']:7.3f} {t['downside_mix']:4.1f} "
              + " ".join(f"{v:9.4f}" for v in t["scores"])
              + f" {t['mean']:8.4f} {t['worst']:8.4f} "
                f"{t['min_gmvp']:6.3f} {t['min_erc']:6.3f} "
                f"{min(t['gmvp_margin'], t['erc_margin']):5.2f}  "
                f"{'yes' if t['admissible'] else 'NO'}")

    args.out.write_text(json.dumps(
        {"floor": FLOOR, "datasets": datasets, "rows": table}, indent=2),
        encoding="utf-8")
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
