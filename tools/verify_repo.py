#!/usr/bin/env python3
"""Reproducible evidence report, keeping blocked integration explicit."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def run(name: str,args: list[str]) -> dict:
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    result=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (ROOT/'reports'/f'{name}.txt').write_text(result.stdout,encoding='utf-8')
    print(f'{name}: exit={result.returncode}',flush=True)
    return {'exit_code':result.returncode,'log':f'reports/{name}.txt'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-integration',action='store_true');a=p.parse_args()
    (ROOT/'reports').mkdir(exist_ok=True)
    checks={}
    checks['build']=run('build-check',['tools/build_submission.py','--check'])
    checks['syntax']=run('syntax-check',['-m','compileall','-q','src','research','submission','tools'])
    checks['pytest']=run('pytest',['-m','pytest','-q','--junitxml=reports/pytest.xml'])
    checks['stress']=run('stress-run',['tools/stress.py','--cases','240'])
    checks['preflight']=run('preflight',['tools/preflight.py'])
    totals={'tests':0,'failures':0,'errors':0,'skipped':0}
    if (ROOT/'reports/pytest.xml').exists():
        doc=ET.parse(ROOT/'reports/pytest.xml')
        for suite in doc.findall('.//testsuite'):
            for k in totals:totals[k]+=int(suite.attrib.get(k,0))
    totals['passed']=totals['tests']-totals['failures']-totals['errors']-totals['skipped']
    packages={}
    for name in ['numpy','scipy','scikit-learn','pandas','pytest','skfolio','cvxpy','cvxpy-base']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    full=checks['preflight']['exit_code']==0
    report={'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'python':sys.version,'platform':platform.platform(),'packages':packages,'checks':checks,'pytest':totals,
        'standalone_sha256':hashlib.sha256((ROOT/'submission/portfolio_round1.py').read_bytes()).hexdigest(),
        'core_sha256':hashlib.sha256((ROOT/'src/portfolio_game/core.py').read_bytes()).hexdigest(),
        'real_skfolio_preflight':'PASSED' if full else 'BLOCKED_OR_FAILED',
        'ready_for_teacher_submission':full and all(checks[k]['exit_code']==0 for k in ('build','syntax','pytest','stress')),
        'market_backtest':'NOT_RUN_IN_THIS_DELIVERY','hyperparameter_search':'NOT_RUN_IN_THIS_DELIVERY',
        'github_remote_publication':'NOT_PERFORMED_BY_THIS_VERIFIER',
        'notice':'Synthetic robustness and core optimization checks do not establish hidden-test success or investment performance.'}
    (ROOT/'reports/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if any(checks[k]['exit_code']!=0 for k in ('build','syntax','pytest','stress')):return 1
    return 2 if a.require_integration and not full else 0

if __name__=='__main__':raise SystemExit(main())
