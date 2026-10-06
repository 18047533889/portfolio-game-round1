"""Isolated check: run the submission in an empty directory that contains it.

The point is to prove the shipped file has no hidden dependency on this repo
(no `src/` import, no config file, no relative data path). The repository's own
verification runs everything from the repo root, where an accidental
`from portfolio_game_round3.core import ...` would still resolve; the grader
will not have that luxury, so the file is exercised in isolation here.

Steps:
  1. create a temporary directory;
  2. copy in exactly one file -- the submission;
  3. write a grader-shaped driver next to it (round-3 contract, weight_drift
     TRUE, WalkForward 126/20) that imports the submission by path;
  4. run that driver as a subprocess with cwd = temp dir and PYTHONPATH cleared;
  5. require a clean exit, zero failed portfolios, and legal weights.

Nothing from the repo is on the child's import path, so a pass here is evidence
the single file is self-contained.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DRIVER = '''\
"""Grader-shaped driver. Runs in an otherwise empty directory."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import BaseOptimization

here = Path(__file__).resolve().parent
mods = sorted(p for p in here.glob("*.py") if p.name != Path(__file__).name)
if len(mods) != 1:
    raise SystemExit(f"expected exactly 1 submission file, found {len(mods)}")

spec = importlib.util.spec_from_file_location("isolated_submission", mods[0])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

cls = getattr(module, "CVXPYPortfolio", None)
if cls is None:
    raise SystemExit("no CVXPYPortfolio class")
if not issubclass(cls, BaseOptimization):
    raise SystemExit("CVXPYPortfolio does not inherit BaseOptimization")

X = load_sp500_dataset().pct_change(fill_method=None).dropna()
# Round-3 contract: lookback 126, optimize every 20, weight drift on.
model = cls(portfolio_params={"name": "CVXPYPortfolio", "weight_drift": True})
model.raise_on_failure = False
pred = cross_val_predict(model, X, cv=WalkForward(train_size=126, test_size=20),
                         n_jobs=1)

n = int(pred.n_failed_portfolios)
bad = []
weights = [np.asarray(p.weights, dtype=float) for p in pred.portfolios]
for w in weights:
    if not np.isfinite(w).all():
        bad.append("nonfinite")
    if abs(w.sum() - 1.0) > 1e-8:
        bad.append("not_fully_invested")
    if abs(w).sum() > __LEVERAGE__ + 1e-8:
        bad.append("gross_over_leverage")
    if w.size > 1 and np.allclose(w, 1.0 / w.size, atol=1e-8, rtol=1e-5):
        bad.append("equal_weight")

print(json.dumps({
    "imported_file": mods[0].name,
    "n_portfolios": len(pred.portfolios),
    "n_failed_portfolios": n,
    "n_fallback_portfolios": int(pred.n_fallback_portfolios),
    "annual_return": float(pred.annualized_mean),
    "annual_volatility": float(pred.annualized_standard_deviation),
    "max_short_total": float(max(np.maximum(-w, 0.0).sum() for w in weights)),
    "max_gross": float(max(np.abs(w).sum() for w in weights)),
    "violations": sorted(set(bad)),
}))
if n or bad:
    raise SystemExit(1)
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submission/portfolio_round3.py")
    ap.add_argument("--keep", action="store_true", help="keep the temp directory")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    cfg = json.loads((ROOT / "configs/submission_round3.json").read_text(encoding="utf-8"))
    leverage = float(cfg.get("gross_leverage", 1.0))

    tmp = Path(tempfile.mkdtemp(prefix="r3_isolated_"))
    try:
        shutil.copy2(args.submission, tmp / args.submission.name)
        driver = DRIVER.replace("__LEVERAGE__", repr(leverage))
        print(f"frozen gross_leverage: {leverage}")
        (tmp / "driver.py").write_text(driver, encoding="utf-8")
        listing = sorted(p.name for p in tmp.iterdir())
        print(f"temp dir : {tmp}")
        print(f"contents : {listing}")

        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
               "HOME": str(Path.home()),
               "PYTHONPATH": "",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MPLBACKEND": "Agg"}
        proc = subprocess.run([args.python, "driver.py"], cwd=tmp, env=env,
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        print(f"exit code: {proc.returncode}")
        print(proc.stdout.strip())
        if proc.returncode != 0:
            print("\nISOLATED CHECK FAILED")
            return 1
        print("\nISOLATED CHECK PASSED (single-file submission is self-contained)")
        return 0
    finally:
        if args.keep:
            print(f"kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
