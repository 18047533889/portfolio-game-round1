#!/usr/bin/env python3
"""Strict pre-submission checks: missing dependencies are a FAILURE, not PASS."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-teacher-dataset',action='store_true',help='Synthetic integration only; not full acceptance')
    a=p.parse_args()
    missing=[name for name in ('numpy','scipy','sklearn','pandas','skfolio') if importlib.util.find_spec(name) is None]
    if missing:
        print('BLOCKED: missing genuine dependencies: '+', '.join(missing),file=sys.stderr)
        return 2
    subprocess.run([sys.executable,str(ROOT/'tools/build_submission.py'),'--check'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','pytest',str(ROOT/'tests/test_skfolio_integration.py'),'-q'],cwd=ROOT,check=True)
    if a.skip_teacher_dataset:
        print('SYNTHETIC INTEGRATION ONLY: original instructor self-test was explicitly not run.')
        return 0
    # Copy exactly the teacher file and the uploaded instructor self-test. No
    # research source package can make a missing standalone dependency disappear.
    with tempfile.TemporaryDirectory() as folder:
        target=Path(folder)/'portfolio_round1.py'
        shutil.copy2(ROOT/'submission/portfolio_round1.py',target)
        test=Path(folder)/'self_test.py'
        shutil.copy2(ROOT/'teacher_reference/self_test.py',test)
        subprocess.run([sys.executable,str(test),str(target)],cwd=folder,check=True)
    print('FULL PREFLIGHT PASSED: real skfolio integration and original teacher self-test.')
    return 0

if __name__=='__main__': raise SystemExit(main())
