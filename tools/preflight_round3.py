#!/usr/bin/env python3
"""Round-3 preflight: the instructor's own self-test, on the uploaded file alone.

Round 1 and 2 both used this shape and it is worth keeping identical, because
the grading system runs the submission the way self_test.py does: import the
file by path, instantiate with portfolio_params, call fit, read weights_.

Steps:
  1. fail immediately if a genuine dependency is missing (a missing skfolio is a
     FAILURE here, not a skip -- the whole point is to exercise the real library);
  2. check the artifact is in sync with its core and frozen config;
  3. run the round-3 test module;
  4. copy the submission and teacher_reference/self_test.py into a temp dir and
     run the instructor's script there, so nothing from this repo can rescue it.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission/portfolio_round3.py"
SELF_TEST = ROOT / "teacher_reference/self_test.py"


def run(args: list[str], cwd: Path | None = None) -> int:
    proc = subprocess.run([sys.executable, *args], cwd=cwd or ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout.rstrip())
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("== dependency check ==")
    missing = []
    for name in ("numpy", "scipy", "sklearn", "pandas", "skfolio", "cvxpy"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        print(f"MISSING DEPENDENCIES: {missing}")
        return 2
    import skfolio
    print(f"skfolio {getattr(skfolio, '__version__', 'unknown')} present")

    print("\n== artifact in sync ==")
    code = run(["tools/build_submission_round3.py", "--check"])
    if code != 0:
        print("PREFLIGHT FAILED: submission is stale")
        return 1

    if not args.quick:
        print("\n== round-3 core tests ==")
        code = run(["-m", "pytest", "-q", "tests/test_round3_core.py"])
        if code != 0:
            print("PREFLIGHT FAILED: round-3 tests")
            return 1

    print("\n== instructor self-test on the isolated file ==")
    tmp = Path(tempfile.mkdtemp(prefix="r3_preflight_"))
    try:
        shutil.copy2(SUBMISSION, tmp / SUBMISSION.name)
        if not SELF_TEST.exists():
            print("teacher_reference/self_test.py missing")
            return 2
        shutil.copy2(SELF_TEST, tmp / "self_test.py")
        listing = sorted(p.name for p in tmp.iterdir())
        print(f"isolated directory contents: {listing}")
        proc = subprocess.run([sys.executable, "self_test.py", SUBMISSION.name],
                              cwd=tmp, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
                                   "PYTHONPATH": ""})
        out = proc.stdout
        print(out.rstrip())
        if proc.returncode != 0 or "Basic checks passed" not in out:
            print("\nPREFLIGHT FAILED: instructor self-test did not pass")
            return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nROUND-3 FULL PREFLIGHT PASSED: real skfolio integration and the "
          "original instructor self-test, on the uploaded file alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
