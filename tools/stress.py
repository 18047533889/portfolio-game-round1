#!/usr/bin/env python3
"""Deterministic synthetic robustness tests; NOT financial performance evidence."""
from __future__ import annotations
from collections import Counter
import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from portfolio_game.core import allocate,check_weights


def stress(n_cases: int=240) -> dict:
    rng=np.random.default_rng(20260918)
    failures=[];statuses=Counter();degraded=0;durations=[];gapmax=0.
    attempts=0;converged=0;accepted_gapmax=0.
    dimensions=[2,3,5,10,20,64,128,257]
    lengths=[1,2,5,20,63,252,400]
    for case in range(n_cases):
        n=dimensions[case%len(dimensions)];t=lengths[(case//len(dimensions))%len(lengths)]
        x=rng.normal(0,.006,(t,1))+rng.normal(size=(t,n))*np.linspace(.002,.035,n)
        kind=case%10
        if kind==1: x[:]=0
        elif kind==2: x[:]=x[:,:1]
        elif kind==3: x[rng.random(x.shape)<.05]=np.nan
        elif kind==4: x[:,:max(1,n//4)]=np.nan
        elif kind==5: x[0,0]=1e200
        elif kind==6: x[-1,0]=np.inf
        elif kind==7: x[:]=rng.standard_t(3,x.shape)*.015;np.maximum(x,-1,out=x)
        elif kind==8: x[:t//2]*=.1;x[t//2:]*=3
        before=time.perf_counter()
        try:
            result=allocate(x);check_weights(result['weights'],n)
            d=result['diagnostics'];statuses[d['status']]+=1
            gapmax=max(gapmax,float(d.get('qp_relative_gap',0.)))
            if 'qp_relative_gap' in d:
                attempts+=1
                if d['main_solver_converged']:
                    converged+=1;accepted_gapmax=max(accepted_gapmax,float(d['qp_relative_gap']))
            degraded+=int(d['status'].startswith(('fallback','emergency')) or
                          d['anti_equal_weight_applied'] or d['single_asset_fallback'])
            # Repeated calls must not depend on hidden state or randomness.
            repeated=allocate(x)['weights']
            if not np.array_equal(result['weights'],repeated):
                raise AssertionError('Repeated allocation is nondeterministic')
        except Exception as exc:
            failures.append({'case':case,'rows':t,'assets':n,'kind':kind,
                             'error':f'{type(exc).__name__}: {exc}'})
        durations.append(time.perf_counter()-before)
    return {'dataset_type':'SYNTHETIC_ENGINEERING_ONLY', 'cases':n_cases,
        'final_failures':len(failures),'failure_rate':len(failures)/n_cases,
        'degraded_cases':degraded,'statuses':dict(statuses),'failures':failures,
        'two_fit_seconds_p50':float(np.median(durations)),
        'two_fit_seconds_p95':float(np.quantile(durations,.95)),
        'two_fit_seconds_max':float(max(durations)),'maximum_attempted_qp_relative_gap':gapmax,
        'main_solver_attempts':attempts,'main_solver_converged_cases':converged,
        'main_solver_nonconverged_cases':attempts-converged,
        'maximum_converged_qp_relative_gap':accepted_gapmax,
        'note':'Includes intentionally degenerate data and resource guards. Zero failures here is not a hidden-test guarantee.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=int,default=240)
    p.add_argument('--output',type=Path,default=ROOT/'reports/stress.json');a=p.parse_args()
    if a.cases<1:p.error('--cases must be positive')
    report=stress(a.cases);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    return int(report['final_failures']>0)

if __name__=='__main__':raise SystemExit(main())
