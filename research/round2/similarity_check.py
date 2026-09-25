"""Proximity of candidate weights to the three prohibited portfolios.

Round 2 prohibits: equal weight (EWP), inverse volatility (IVP), and global
minimum variance on the sample covariance matrix (SCM-GMVP).

Compliance is a property of the *algorithm*, not of the weights: any low-risk
tilt will correlate with IVP without being IVP. But "conceptually different yet
numerically identical" is a bad place to be, so this measures how close each
candidate sits to each prohibited book. A candidate that reproduces a banned
portfolio to floating-point is a violation; one that merely correlates with it
is a risk-tilt.

Reported per candidate, over many windows:
  L1  = sum |w_candidate - w_prohibited|   (0 = identical, ~1.0+ = unrelated)
  rho = cross-sectional correlation of the weight vectors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_eval import load_core, load_datasets  # noqa: E402

from scipy.optimize import minimize  # noqa: E402


def scm_gmvp(x: np.ndarray) -> np.ndarray:
    """Long-only global minimum variance on the exact sample covariance."""
    c = np.cov(x.T)
    n = c.shape[0]
    r = minimize(lambda w: w @ c @ w, np.ones(n) / n,
                 jac=lambda w: 2 * c @ w, method="SLSQP",
                 bounds=[(0.0, 1.0)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                               "jac": lambda w: np.ones_like(w)}],
                 options={"maxiter": 800, "ftol": 1e-16})
    return np.clip(r.x, 0.0, None) / max(r.x.clip(0).sum(), 1e-12)


def prohibited_books(x: np.ndarray) -> dict[str, np.ndarray]:
    n = x.shape[1]
    vol = x.std(axis=0, ddof=1)
    vol = np.maximum(vol, np.finfo(float).tiny)
    return {
        "EWP": np.ones(n) / n,
        "IVP": (1.0 / vol) / (1.0 / vol).sum(),
        "SCM-GMVP": scm_gmvp(x),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sp500(20),ftse100(64),factors(5)")
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    core = load_core()
    panels = load_datasets([d.strip() for d in args.datasets.split(",")])
    candidates = [("p=0", 0.0), ("p=0.25", 0.25), ("p=1", 1.0), ("p=4", 4.0),
                  ("p=16", 16.0)]

    report: dict = {"datasets": {}}
    for name, full in panels.items():
        x_all = full.to_numpy(dtype=float)
        rng = np.random.default_rng(20260925)
        acc = {label: {"L1": [], "rho": []} for label, _ in candidates}
        acc["hrp"] = {"L1": [], "rho": []}
        print(f"\n{'='*82}\n{name}  {full.shape}")
        print(f"  {'candidate':10s} " + " ".join(f"{b:>21s}" for b in ("EWP", "IVP", "SCM-GMVP")))
        for _ in range(args.windows):
            i = int(rng.integers(0, len(x_all) - 252))
            blk = x_all[i:i + 252]
            banned = prohibited_books(blk)
            books = {lab: core.allocate(blk, anchor_penalty=p, method="regularized")["weights"]
                     for lab, p in candidates}
            books["hrp"] = core.allocate(blk, method="hrp")["weights"]
            for lab, w in books.items():
                for bname, bw in banned.items():
                    acc[lab].setdefault(bname, {"L1": [], "rho": []})
                    acc[lab][bname]["L1"].append(float(np.abs(w - bw).sum()))
                    if np.std(w) > 0 and np.std(bw) > 0:
                        acc[lab][bname]["rho"].append(float(np.corrcoef(w, bw)[0, 1]))
        block = {}
        for lab in [c[0] for c in candidates] + ["hrp"]:
            row = {}
            cells = []
            for bname in ("EWP", "IVP", "SCM-GMVP"):
                d = acc[lab][bname]
                row[bname] = {"L1_median": float(np.median(d["L1"])),
                              "rho_median": float(np.median(d["rho"])) if d["rho"] else None}
                cells.append(f"L1={row[bname]['L1_median']:.3f} r={row[bname]['rho_median']:+.3f}"
                             if row[bname]["rho_median"] is not None
                             else f"L1={row[bname]['L1_median']:.3f} r=  n/a")
            block[lab] = row
            print(f"  {lab:10s} " + " ".join(f"{c:>21s}" for c in cells))
        report["datasets"][name] = block

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
