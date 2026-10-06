"""Re-derive the decay scale for the halved estimation window.

round 2 shipped half_life = 63 with lookback 252, i.e. a quarter of the window.
Round 3 halves the window to 126, and half_life is a decay scale in *days*, not
a fraction -- so the round-2 value cannot be inherited and cannot be rescaled
blindly either: what matters is how many observations carry weight, which also
depends on recent_mix. This measures the pair on the actual contract.

The opponent field is not re-run. It does not depend on our parameters, so the
settings measured here are re-ranked against the field already stored by the
full sweep, which keeps the comparison in one pool instead of silently
comparing across two different ones.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/round3"))
sys.path.insert(0, str(ROOT / "research/round2"))

import rank_sim  # noqa: E402
from r3_sweep import DATASETS, load_core, measure  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-lives", default="10,16,21,30,42,63")
    ap.add_argument("--recent-mixes", default="0.15,0.25,0.40")
    ap.add_argument("--penalty", type=float, default=16.0)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--datasets", default="sp500,ftse100,factors")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/round3_decay.json")
    args = ap.parse_args()

    from skfolio.datasets import __dict__ as ds_dict

    core = load_core()
    panels = {}
    for name in [d.strip() for d in args.datasets.split(",")]:
        loader, cap = DATASETS[name]
        panels[name] = ds_dict[loader]().pct_change(fill_method=None).dropna().iloc[:, :cap]

    rows = []
    for hl in [float(x) for x in args.half_lives.split(",")]:
        for rm in [float(x) for x in args.recent_mixes.split(",")]:
            for name, X in panels.items():
                res = measure(core, X, half_life=hl, recent_mix=rm,
                              penalty=args.penalty, leverage=args.leverage)
                row = {"dataset": name, "penalty": args.penalty,
                       "leverage": args.leverage, "half_life": hl,
                       "recent_mix": rm, **res}
                rows.append(row)
                print(f"  hl={hl:5.1f} rm={rm:4.2f} {name:9s} fail={res['n_failed']:3d}/"
                      f"{res['n_portfolios']:4d} annual={res['annual_return']:8.4%} "
                      f"vol={res['annual_volatility']:7.4%} "
                      f"dd={res['max_drawdown']:8.4%}", flush=True)

    out = {"contract": {"penalty": args.penalty, "leverage": args.leverage},
           "settings": rows}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {args.out}")

    # Re-rank against the stored opponent field so the comparison is in one pool.
    sweep_path = ROOT / "reports/round3_sweep.json"
    if not sweep_path.exists():
        print("no stored field to re-rank against; skipping the score table")
        return 0
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    print(f"\n{'half_life':>9s} {'recent_mix':>10s} "
          + " ".join(f"{d:>9s}" for d in panels) + f" {'mean':>8s} {'worst':>8s}")
    grouped: dict[tuple[float, float], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["half_life"], row["recent_mix"]), {})[row["dataset"]] = row
    summary = []
    for (hl, rm), per in grouped.items():
        if len(per) != len(panels):
            continue
        scores = []
        for name in panels:
            opponents = [e for e in sweep["field"][name]
                         if not e["label"].startswith("OURS")]
            r = per[name]
            entries = opponents + [{"label": "ours",
                                    "annual_return": r["annual_return"],
                                    "annual_volatility": r["annual_volatility"],
                                    "failure_rate": r["failure_rate"]}]
            sc = r3_score(entries, rank_sim.rank_pct)
            scores.append(sc[-1])
        summary.append((float(np.mean(scores)), float(np.min(scores)), hl, rm, scores))
    summary.sort(reverse=True)
    for mean, worst, hl, rm, scores in summary:
        cells = " ".join(f"{s:9.4f}" for s in scores)
        print(f"{hl:9.1f} {rm:10.2f} {cells} {mean:8.4f} {worst:8.4f}")
    return 0


def r3_score(entries: list[dict], rank_pct) -> list[float]:
    r_ann = rank_pct([e["annual_return"] for e in entries], True)
    r_vol = rank_pct([e["annual_volatility"] for e in entries], False)
    r_fail = rank_pct([e["failure_rate"] for e in entries], False)
    return list(0.20 * r_ann + 0.10 * r_vol + 0.70 * r_fail)


if __name__ == "__main__":
    raise SystemExit(main())
