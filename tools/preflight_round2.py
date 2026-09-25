#!/usr/bin/env python3
"""Round-2 preflight: the teacher's own self-test on the isolated uploaded file.

Same idea as the round-1 preflight -- copy *only* the file that will be
uploaded plus the instructor's original self_test.py into a fresh directory and
run it there, so that no source package in this repository can make a missing
standalone dependency disappear. Missing genuine dependencies are a FAILURE,
not a pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-teacher-dataset", action="store_true",
                   help="Synthetic integration only; not full acceptance")
    p.add_argument("--submission", type=Path,
                   default=ROOT / "submission/portfolio_round2.py")
    a = p.parse_args()

    missing = [name for name in ("numpy", "scipy", "sklearn", "pandas", "skfolio")
               if importlib.util.find_spec(name) is None]
    if missing:
        print("BLOCKED: missing genuine dependencies: " + ", ".join(missing),
              file=sys.stderr)
        return 2

    subprocess.run([sys.executable, str(ROOT / "tools/build_submission_round2.py"),
                    "--check"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "pytest",
                    str(ROOT / "tests/test_round2_core.py"), "-q"],
                   cwd=ROOT, check=True)
    if a.skip_teacher_dataset:
        print("SYNTHETIC INTEGRATION ONLY: original instructor self-test was "
              "explicitly not run.")
        return 0

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / a.submission.name
        shutil.copy2(a.submission, target)
        test = Path(folder) / "self_test.py"
        shutil.copy2(ROOT / "teacher_reference/self_test.py", test)
        listing = sorted(x.name for x in Path(folder).iterdir())
        print(f"isolated directory contents: {listing}")
        subprocess.run([sys.executable, str(test), str(target)],
                       cwd=folder, check=True)
    print("ROUND-2 FULL PREFLIGHT PASSED: real skfolio integration and the "
          "original instructor self-test, on the uploaded file alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
