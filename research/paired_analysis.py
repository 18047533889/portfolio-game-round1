"""Are the candidate differences real, or is the sample just small?

Two sweeps disagreed about ftse100: `window_penalty_sweep.py` (100 blocks) said
raising the penalty lowers Sharpe, `penalty_detail.py` (60 blocks) said it raises
Sharpe. Same data, same metric, different draw. That is the signature of an
effect sitting inside sampling noise, and it matters: shipping a configuration
because one draw liked it is exactly the failure mode the factor-tilt work was
rejected for.

So this reads the per-block records and does a *paired* comparison against the
shipped configuration -- every candidate is scored on the identical blocks, and
the test is on the within-block difference, which removes the between-block
variance that dominates the raw averages.

Reported per candidate: mean/median of the paired difference, the share of blocks
that improved, and a Wilcoxon signed-rank p-value (no normality assumption; a
drawdown difference is not normally distributed). A candidate is only worth
shipping if the Sharpe difference is positive with a small p-value on *every*
dataset -- the same unanimity rule the estimator sweep used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = "p=0.25 rm=0.25 SHIPPED"

KEYS = ("annualized_sharpe", "annualized_mean", "max_drawdown")
SIGNS = {"annualized_sharpe": 1, "annualized_mean": 1, "max_drawdown": -1}


def wilcoxon(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; exact for small n is overkill here."""
    d = diff[diff != 0]
    if d.size < 5:
        return float("nan")
    try:
        from scipy.stats import wilcoxon as _w
        return float(_w(d, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="reports/penalty_detail.json")
    args = ap.parse_args()

    data = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    verdicts: dict[str, bool] = {}

    for ds, payload in data["datasets"].items():
        records = payload["records"]
        labels = list(records[0]["cells"].keys())
        if SHIPPED not in labels:
            print(f"[{ds}] shipped configuration missing from records; skipping")
            continue
        others = [l for l in labels if l != SHIPPED]
        n = len(records)
        print("=" * 92)
        print(f"[{ds}]  n={n} blocks  (paired against {SHIPPED})")
        print(f"  {'candidate':24s} {'metric':10s} {'meanDelta':>11s} {'medDelta':>11s} "
              f"{'better':>8s} {'p':>8s}  verdict")

        for cand in others:
            row_ok = True
            for key in KEYS:
                a = np.array([r["cells"][cand][key] for r in records], dtype=float)
                b = np.array([r["cells"][SHIPPED][key] for r in records], dtype=float)
                # orient so that positive always means "better"
                raw = a - b
                oriented = raw * SIGNS[key]
                better = float((oriented > 0).mean())
                p = wilcoxon(oriented)
                favourable = oriented.mean() > 0 and (np.isnan(p) or p < 0.05)
                if key in ("annualized_sharpe", "annualized_mean") and not favourable:
                    row_ok = False
                tag = "better" if favourable else ("worse" if oriented.mean() < 0 else "flat")
                print(f"  {cand:24s} {key:10s} {raw.mean():+11.4f} "
                      f"{np.median(raw):+11.4f} {better:7.1%} {p:8.4f}  {tag}")
            verdicts.setdefault(cand, True)
            verdicts[cand] = verdicts[cand] and row_ok
            print()

    print("=" * 92)
    print("unanimity verdict (Sharpe AND return significantly better on every dataset)")
    for cand, ok in sorted(verdicts.items()):
        print(f"  {cand:24s} {'PASS' if ok else 'fail'}")
    winners = [c for c, ok in verdicts.items() if ok]
    if not winners:
        print("\n  No candidate qualifies. The shipped configuration stands: the")
        print("  measured differences are within sampling noise, and moving would be")
        print("  fitting the draw rather than improving the method.")
    else:
        print(f"\n  qualifying: {', '.join(winners)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
