"""Bounded chronological search; 2018+ data never enters candidate fitting."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.data import load_returns,visible_for_tuning,phase_bounds,VALIDATION_END
from research.engine import walk_forward
from research.metrics import performance

ROOT=Path(__file__).resolve().parents[1]


def candidate_grid(include_schur: bool=False) -> list[dict]:
    # The rho=0, lambda=0 candidate already is slow-covariance min variance;
    # do not count an identical renamed minimum-variance strategy twice.
    grid=[{'method':'regularized','half_life':63.,'recent_mix':rho,'anchor_penalty':penalty}
          for rho,penalty in itertools.product([0.,.25,.5],[0.,.3,1.,3.])]
    grid.extend([{'method':'hrp','half_life':63.,'recent_mix':.25,'anchor_penalty':1.},
                 {'method':'inverse_volatility','half_life':63.,'recent_mix':0.,'anchor_penalty':1.}])
    if include_schur:
        grid.extend({'method':'schur','half_life':63.,'recent_mix':.25,'anchor_penalty':1.,'gamma':gamma}
                    for gamma in [0.,.25,.5,.75])
    return grid


def evaluate(frame: pd.DataFrame,config: dict,phase: str,*,weight_drift: bool=False) -> dict:
    start,end=phase_bounds(phase)
    result=walk_forward(frame,config,start=start,end=end,weight_drift=weight_drift)
    series=result['returns']; yearly={}
    if result['summary']['failed_folds']==0:
        for year,part in series.groupby(series.index.year):
            if len(part)>=126: yearly[int(year)]=performance(part.to_numpy())
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    return {'candidate':identity,'config':config,'summary':result['summary'],'yearly':yearly}


def rank_records(records: list[dict],return_metric: str='cagr') -> list[dict]:
    """Ranks internal candidates, not competitors. Failed runs never get a score.

    Priority: eligible/no final failures, median yearly percentile, lower-quartile
    yearly percentile, full-period percentile, lower operational degradation.
    """
    rows=[]
    for record in records:
        stats=record['summary']; perf=stats['performance']
        eligible=(stats['failed_folds']==0 and perf is not None and perf.get('sharpe') is not None
                  and stats.get('degradation_rate',0.)<=.05)
        record['eligible_for_selection']=bool(eligible)
        record['median_year_percentile']=-1.;record['lower_quartile_year_percentile']=-1.
        record['full_period_percentile']=-1.
        if eligible:
            rows.append({'id':record['candidate'],'sharpe':perf['sharpe'],
                         'return':perf[return_metric],'negative_drawdown':-perf['max_drawdown']})
    if rows:
        table=pd.DataFrame(rows).set_index('id')
        scores=table.rank(pct=True,method='average').mean(axis=1)
        by_id={r['candidate']:r for r in records}
        for key,value in scores.items(): by_id[key]['full_period_percentile']=float(value)
        years=sorted({year for r in records if r['eligible_for_selection'] for year in r['yearly']})
        annual={key:[] for key in table.index}
        for year in years:
            entries=[]
            for r in records:
                if not r['eligible_for_selection']: continue
                perf=r['yearly'].get(year)
                if perf and perf.get('sharpe') is not None:
                    entries.append({'id':r['candidate'],'sharpe':perf['sharpe'],'return':perf[return_metric],
                                    'negative_drawdown':-perf['max_drawdown']})
            if entries:
                ranks=pd.DataFrame(entries).set_index('id').rank(pct=True,method='average').mean(axis=1)
                for key,value in ranks.items(): annual[key].append(float(value))
        for key,values in annual.items():
            if values:
                by_id[key]['median_year_percentile']=float(np.median(values))
                by_id[key]['lower_quartile_year_percentile']=float(np.quantile(values,.25))
            else:
                by_id[key]['median_year_percentile']=float(scores[key])
                by_id[key]['lower_quartile_year_percentile']=float(scores[key])
    return sorted(records,key=lambda r:(not r['eligible_for_selection'],
        -r['median_year_percentile'],-r['lower_quartile_year_percentile'],-r['full_period_percentile'],
        r['summary'].get('degradation_rate',1.),r['candidate']))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prices-csv',type=Path);p.add_argument('--returns-csv',type=Path)
    p.add_argument('--include-schur',action='store_true')
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--refine-half-life',action='store_true')
    p.add_argument('--weight-drift',action='store_true')
    p.add_argument('--return-metric',choices=['cagr','annualized_mean'],default='cagr')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/tuning')
    a=p.parse_args()
    if not 1<=a.top_k<=14: p.error('--top-k must be 1..14')
    # Physically truncate before all candidate evaluations.
    frame=visible_for_tuning(load_returns(prices_csv=a.prices_csv,returns_csv=a.returns_csv,end=VALIDATION_END))
    if frame.empty or frame.index[-1]>VALIDATION_END: raise RuntimeError('Invalid tuning time domain')
    a.output.mkdir(parents=True,exist_ok=True)
    development=[]
    for config in candidate_grid(a.include_schur):
        print('Development:',config,flush=True)
        development.append(evaluate(frame,config,'development',weight_drift=a.weight_drift))
    devrank=rank_records(development,a.return_metric)
    (a.output/'development.json').write_text(json.dumps(devrank,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    survivors=[r['config'] for r in devrank if r['eligible_for_selection']][:a.top_k]
    if not survivors: raise SystemExit('No candidate meets failure/degradation gates; no configuration frozen')
    refinements=list(survivors)
    if a.refine_half_life:
        for cfg in survivors:
            # Half life has no effect at rho=0 or for inverse volatility.
            if cfg['recent_mix']>0 and cfg['method']!='inverse_volatility':
                refinements.extend({**cfg,'half_life':h} for h in [42.,126.])
    # Only development survivors/refinements see the validation period.
    validation=[evaluate(frame,cfg,'validation',weight_drift=a.weight_drift) for cfg in refinements]
    ranking=rank_records(validation,a.return_metric)
    (a.output/'validation.json').write_text(json.dumps(ranking,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if not ranking or not ranking[0]['eligible_for_selection']:
        raise SystemExit('No validation winner; default submission not changed')
    chosen={**ranking[0]['config'],'selection_status':'validation_selected',
        'selection_end':'2017-12-31','return_metric':a.return_metric,'weight_drift':a.weight_drift,
        'holdout_used':False,'tuning_rows':len(frame),
        'tuning_data_sha256':hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),
        'note':'Internal candidate percentile, not a predicted teacher score. Rebuild explicitly after review.'}
    (a.output/'selected.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(chosen,ensure_ascii=False,indent=2))
    if chosen['method']=='schur':
        print('Schur is research-only: this result requires a separate validated exporter; the current teacher file is NOT changed.')
    else:
        print('Review selected.json; then explicitly build and run real skfolio preflight. Holdout remains unopened.')


if __name__=='__main__': main()
