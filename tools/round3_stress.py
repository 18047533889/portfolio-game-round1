"""Round-3 stress: synthetic degenerate inputs at the allocate level.

The hostile panels are imported from the round-2 generator rather than retyped,
because the point is to keep applying the same battery -- every one of those
cases has broken a risk model somewhere, and re-typing them would invite a
quietly weakened set. What changes for round 3 is the admissibility test: the
budget is aggregate, so a book may be negative and is judged on gross exposure
and full investment instead of on nonnegativity.

Checked on every case:
  * weights are finite, sum to 1, gross within the frozen leverage
  * weights are NOT equal weight (prohibited this round)
  * weights are NOT the inverse-volatility book (prohibited this round)
  * the call returns rather than raising
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/submission_round3.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases(rng: np.random.Generator):
    """The round-2 hostile battery, reused verbatim through its own module."""
    round2 = load_module(ROOT / "tools/round2_stress.py", "r2_stress_cases")
    yield from round2.cases(rng)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submission/portfolio_round3.py")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    module = load_module(args.submission, "sub_r3_stress")
    cls = module.CVXPYPortfolio
    rng = np.random.default_rng(20261006)

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    leverage = float(cfg.get("gross_leverage", 1.0))
    budget = 0.5 * (leverage - 1.0)
    print(f"frozen config: gross_leverage={leverage} -> aggregate short "
          f"budget {budget}\n")

    failures, equal_hits, ivp_hits, illegal = [], [], [], []
    statuses: dict[str, int] = {}
    short_totals, gross_totals = [], []
    total = 0
    print(f"{'case':26s} {'status':32s} {'n':>3s} {'maxw':>7s} "
          f"{'L1_eq':>7s} {'L1_ivp':>7s}  verdict")
    for name, panel in cases(rng):
        total += 1
        panel = np.asarray(panel, dtype=float)
        model = cls(portfolio_params={"name": "stress"})
        model.raise_on_failure = False
        try:
            model.fit(panel)
            w = model.weights_
            if w is None:
                failures.append((name, "weights_ is None"))
                print(f"{name:26s} {'FAILED':32s} {panel.shape[1]:3d}      -       -       -  "
                      f"FAILED ({model.error_})")
                continue
            status = getattr(model, "solver_status_", "?")
            statuses[status] = statuses.get(status, 0) + 1
            n = panel.shape[1]
            ok_shape = w.shape == (n,)
            ok_finite = bool(np.isfinite(w).all())
            ok_sum = abs(float(w.sum()) - 1.0) < 1e-8
            ok_gross = float(np.abs(w).sum()) <= leverage + 1e-8
            if not (ok_shape and ok_finite and ok_sum and ok_gross):
                illegal.append((name, w))
                print(f"{name:26s} {status:32s} {n:3d} {float(w.max()):7.4f} "
                      f"{'-':>7s} {'-':>7s}  ILLEGAL")
                continue
            short_totals.append(float(np.maximum(-w, 0.0).sum()))
            gross_totals.append(float(np.abs(w).sum()))
            eq = float(np.abs(w - 1.0 / n).sum())
            with np.errstate(invalid="ignore"):
                sd = np.nanstd(panel, axis=0, ddof=1)
            if np.all(np.isfinite(sd)) and np.all(sd > 0):
                ivp = (1.0 / sd) / (1.0 / sd).sum()
                l1_ivp = float(np.abs(w - ivp).sum())
            else:
                l1_ivp = float("nan")
            # With a single asset the only legal book is [1.0], which is
            # simultaneously equal weight and inverse volatility, so both
            # prohibitions are vacuous there and neither is judged.
            is_eq = n > 1 and np.allclose(w, 1.0 / n, atol=1e-8, rtol=1e-5)
            is_ivp = n > 1 and np.isfinite(l1_ivp) and l1_ivp < 1e-9
            if is_eq:
                equal_hits.append(name)
            if is_ivp:
                ivp_hits.append(name)
            verdict = "ok" if not (is_eq or is_ivp) else (
                "EQUAL-WEIGHT!" if is_eq else "INVERSE-VOL!")
            print(f"{name:26s} {status:32s} {n:3d} {float(w.max()):7.4f} "
                  f"{eq:7.4f} {l1_ivp:7.4f}  {verdict}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {str(exc)[:120]}"))
            print(f"{name:26s} {'RAISED':32s} {panel.shape[1]:3d}      -       -       -  "
                  f"RAISED {type(exc).__name__}")

    print(f"\n=== TOTAL {total} cases ===")
    print(f"failures (no weights)  : {len(failures)}")
    print(f"illegal weights        : {len(illegal)}")
    print(f"equal-weight outputs   : {len(equal_hits)}  {equal_hits}")
    print(f"inverse-vol outputs    : {len(ivp_hits)}  {ivp_hits}")
    print(f"final-rule statuses    : {statuses}")
    if short_totals:
        print(f"max total short        : {max(short_totals):.6f} (budget {budget})")
        print(f"max gross exposure     : {max(gross_totals):.6f} (cap {leverage})")
    if failures:
        for name, why in failures:
            print(f"  FAIL {name}: {why}")
    if args.out:
        args.out.write_text(json.dumps({
            "cases": total,
            "failures": [f[0] for f in failures],
            "illegal": [i[0] for i in illegal],
            "equal_weight": equal_hits,
            "inverse_volatility": ivp_hits,
            "statuses": statuses,
            "max_total_short": max(short_totals) if short_totals else None,
            "max_gross": max(gross_totals) if gross_totals else None,
            "gross_leverage": leverage,
        }, indent=2), encoding="utf-8")
        print(f"written -> {args.out}")
    return 1 if (failures or illegal or equal_hits or ivp_hits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
