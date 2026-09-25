"""Round-2 stress: synthetic degenerate inputs at the allocate level.

Round 1's stress test drove the same core, but round 2 changed the fallback
chain, so the hostile-input coverage has to be re-established rather than
assumed. Specifically, every path that used to produce an inverse-volatility
book has been replaced, and the new last-resort rule is risk-free. This checks
that the replacement still survives, and that no path lands on either
prohibited book.

Checked on every case:
  * weights are finite, sum to 1, and respect the short floor from the frozen
    config (long-only when none is granted)
  * weights are NOT equal weight (prohibited this round)
  * weights are NOT the inverse-volatility book (prohibited this round)
  * the call returns rather than raising

The floor is read from configs/submission_round2.json rather than assumed to be
zero, so this stress test keeps testing the shipped setting even if the short
budget is turned on. Gross exposure is reported either way.
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
CONFIG = ROOT / "configs/submission_round2.json"


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("sub2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases(rng: np.random.Generator):
    """Hostile panels. Every one of these has broken some risk model somewhere."""
    n = 4
    yield "zeros", np.zeros((300, n))
    yield "constant", np.full((300, n), 0.001)
    yield "one_live_rest_zero", np.column_stack(
        [rng.normal(0, 0.01, 300), np.zeros((300, 3))])
    yield "identical_columns", np.repeat(rng.normal(0, 0.01, (300, 1)), n, axis=1)
    yield "all_nan", np.full((300, n), np.nan)
    yield "nan_after_10", np.concatenate(
        [rng.normal(0, 0.01, (10, n)), np.full((290, n), np.nan)])
    yield "one_col_all_nan", np.column_stack(
        [rng.normal(0, 0.01, 300), np.full(300, np.nan), rng.normal(0, 0.01, 300),
         rng.normal(0, 0.01, 300)])
    yield "huge_values", np.full((300, n), 1e150)
    yield "tiny_values", np.full((300, n), 1e-200)
    yield "inf_values", np.full((300, n), np.inf)
    yield "mixed_inf_nan", np.column_stack([
        rng.normal(0, 0.01, 300), np.full(300, np.inf),
        np.full(300, np.nan), rng.normal(0, 0.01, 300)])
    yield "outliers", np.column_stack([
        rng.normal(0, 0.01, 300), rng.normal(0, 0.01, 300),
        np.where(rng.random(300) < 0.02, 5.0, rng.normal(0, 0.01, 300)),
        rng.normal(0, 0.01, 300)])
    yield "fat_tails", rng.standard_t(3, (300, n)) * 0.01
    yield "regime_shift", np.vstack([
        rng.normal(0, 0.005, (150, n)), rng.normal(0, 0.05, (150, n))])
    yield "two_rows", np.array([[0.01, -0.02, 0.0, 0.03], [0.0, 0.01, -0.01, 0.0]])
    yield "one_row", np.array([[0.01, -0.02, 0.0, 0.03]])
    yield "single_asset", rng.normal(0, 0.01, (300, 1))
    yield "two_assets", rng.normal(0, 0.01, (300, 2))
    yield "three_assets", rng.normal(0, 0.01, (300, 3))
    yield "wide_5", rng.normal(0, 0.01, (300, 5))
    # non-finite only in the most recent row: coverage screen path
    tail = rng.normal(0, 0.01, (300, n))
    tail[-1, :] = np.nan
    yield "nan_in_last_row", tail
    yield "zero_variance_plus_live", np.column_stack([
        np.zeros(300), np.full(300, 0.002), rng.normal(0, 0.01, 300),
        rng.normal(0, 0.01, 300)])
    yield "perfectly_correlated", np.column_stack([
        rng.normal(0, 0.01, 300), rng.normal(0, 0.01, 300),
        rng.normal(0, 0.01, 300), rng.normal(0, 0.01, 300)])[:, [0, 1, 0, 1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submission/portfolio_round2.py")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    module = load_submission(args.submission)
    cls = module.CVXPYPortfolio
    rng = np.random.default_rng(20260925)

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    allow_short = bool(cfg.get("allow_short", False))
    short_cap = float(cfg.get("short_cap", 0.0)) if allow_short else 0.0
    print(f"frozen config: allow_short={allow_short} short_cap={short_cap} "
          f"-> per-asset floor {-short_cap:.4f}\n")

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
            ok_floor = bool((w >= -short_cap - 1e-10).all())
            ok_sum = abs(float(w.sum()) - 1.0) < 1e-8
            if not (ok_shape and ok_finite and ok_floor and ok_sum):
                illegal.append((name, w))
                print(f"{name:26s} {status:32s} {n:3d} {float(w.max()):7.4f} "
                      f"{'-':>7s} {'-':>7s}  ILLEGAL")
                continue
            short_totals.append(float(np.maximum(-w, 0.0).sum()))
            gross_totals.append(float(np.abs(w).sum()))
            eq = float(np.abs(w - 1.0 / n).sum())
            # inverse volatility on the observed column volatilities
            with np.errstate(invalid="ignore"):
                sd = np.nanstd(panel, axis=0, ddof=1)
            if np.all(np.isfinite(sd)) and np.all(sd > 0):
                ivp = (1.0 / sd) / (1.0 / sd).sum()
                l1_ivp = float(np.abs(w - ivp).sum())
            else:
                l1_ivp = float("nan")
            # With a single asset, the only legal book is [1.0], which is
            # simultaneously the equal-weight and the inverse-volatility book.
            # Both prohibitions become vacuous at n=1, so neither is judged.
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
        except Exception as exc:
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
        print(f"max total short        : {max(short_totals):.6f}")
        print(f"max gross exposure     : {max(gross_totals):.6f} "
              f"(net is 1 by construction; gross = 1 + 2*short)")
        print(f"gross within leverage 1: "
              f"{max(gross_totals) <= 1.0 + 1e-9}")
    if failures:
        for name, why in failures:
            print(f"  FAIL {name}: {why}")
    return 1 if (failures or illegal or equal_hits or ivp_hits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
