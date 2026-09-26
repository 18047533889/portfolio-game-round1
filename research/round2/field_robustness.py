"""Is the p = 4 versus p = 1 difference signal, or noise in my simulated field?

The rank-percentile score is computed against a field of other entrants, so a
candidate's score depends on who else is in the pool. My field is 12 skfolio
optimizers, which is a guess at the class. If a claimed improvement survives
only because of who I happened to put in that pool, it is not an improvement.

This perturbs the field -- random subsets of varying size, drawn many times --
recomputes the ranks from scratch each time, and reports how often each setting
wins. A difference that flips sign across resamples is not a finding, and
shipping it would be fitting my own simulation.

Only resampling is done, never re-running the backtest: the per-candidate
annual return, drawdown and failure rate are fixed measurements, and it is the
*ranking* that is uncertain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]

# Our own settings, identified by penalty. Shorting is excluded: the short
# budget was already shown to be score-neutral at the shipped penalty while
# pushing gross exposure past the leverage cap.
OURS = [f"OURS p={p:.2f} cap=0.00" for p in (0.25, 1.0, 4.0)]
LABELS = {f"OURS p={p:.2f} cap=0.00": f"p={p}" for p in (0.25, 1.0, 4.0)}


def rank_pct(values: list[float], higher_is_better: bool, rtol: float = 1e-6) -> np.ndarray:
    """Same tie-tolerant percentile rank used by the sweep, kept in step with it."""
    v = np.asarray(values, dtype=float)
    out = np.zeros(len(v))
    ok = np.isfinite(v)
    if ok.sum() <= 1:
        return out
    sub = v[ok].copy()
    if not higher_is_better:
        sub = -sub
    scale = max(1.0, float(np.max(np.abs(sub))))
    r = rankdata(np.round(sub / (scale * rtol)), method="average")
    out[ok] = (r - 1.0) / (len(sub) - 1.0)
    return out


def score_field(entries: list[dict]) -> dict[str, float]:
    labels = [e["label"] for e in entries]
    r_ann = rank_pct([e["annual_return"] for e in entries], True)
    r_dd = rank_pct([e["max_drawdown"] for e in entries], False)
    r_fail = rank_pct([e["failure_rate"] for e in entries], False)
    total = 0.15 * r_ann + 0.15 * r_dd + 0.70 * r_fail
    return {label: float(total[i]) for i, label in enumerate(labels)}


def main() -> None:
    sweep = json.loads((ROOT / "reports/round2_short_sweep.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260926)
    trials = 400

    # The sweep's own pool is NOT a clean field: it contains 32 near-duplicate
    # variants of our method (four penalties x eight short caps, many of them
    # identical because the cap does not bind). Those siblings occupy rank slots
    # and compress each other's percentiles, which is what made the shipped
    # setting look like 22nd of 44. A clean comparison keeps exactly one entry
    # per method, so here the duplicates are excluded and the field is 12 real
    # opponents plus a small set of genuine candidates.
    print("=== A. our own penalty settings, clean field of 12 opponents ===")
    report(sweep, rng, trials, OURS)

    print("=== B. would switching the method beat the shipped setting? ===")
    runners = ["NCO", "SchurComplementary", "MinVariance(Denoise)",
               "MinVariance(Shrunk)", "MaxDiversification", "RiskBudgeting"]
    report(sweep, rng, trials, [OURS[-1], *runners])


def report(sweep: dict, rng, trials: int, candidates: list[str]) -> None:
    print(f"{'dataset':9s} {'candidate':24s} {'full-field':>11s} {'resample med':>13s} "
          f"{'p05':>8s} {'p95':>8s} {'win rate':>9s}")
    print("-" * 88)

    for name, table in sweep["field"].items():
        # Opponents are the twelve reference submissions and nothing else. Every
        # variant of our own method -- including the ones that are not being
        # tested -- must stay out of the pool, or our own near-duplicates occupy
        # rank slots and compress each other's percentiles.
        others = [e for e in table
                  if not e["label"].startswith("OURS") and e["label"] not in candidates]
        picked = {e["label"]: e for e in table if e["label"] in candidates}
        missing = [c for c in candidates if c not in picked]
        if missing:
            print(f"  {name}: missing {missing}")
            continue

        full = score_field(list(picked.values()) + others)
        wins = {label: 0 for label in picked}
        samples = {label: [] for label in picked}

        for _ in range(trials):
            keep = max(4, int(len(others) * rng.uniform(0.5, 1.0)))
            idx = rng.choice(len(others), size=keep, replace=False)
            entries = list(picked.values()) + [others[i] for i in idx]
            scored = score_field(entries)
            for label in picked:
                samples[label].append(scored[label])
            wins[max(picked, key=lambda k: scored[k])] += 1

        for label in candidates:
            s = np.asarray(samples[label])
            shown = LABELS.get(label, label)
            print(f"{name:9s} {shown:24s} {full[label]:11.4f} {np.median(s):13.4f} "
                  f"{np.percentile(s, 5):8.4f} {np.percentile(s, 95):8.4f} "
                  f"{wins[label]/trials:9.1%}")
        print()


if __name__ == "__main__":
    main()
