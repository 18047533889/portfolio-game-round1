"""Run one frozen configuration; no hyperparameter search in this command."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.data import load_returns,phase_bounds,VALIDATION_END
from research.engine import walk_forward

ROOT=Path(__file__).resolve().parents[1]


def save_result(result: dict,output: Path,metadata: dict) -> None:
    output.mkdir(parents=True,exist_ok=True)
    result['returns'].to_csv(output/'returns.csv')
    result['weights'].to_csv(output/'weights.csv',index=False)
    (output/'summary.json').write_text(json.dumps({**metadata,**result['summary']},ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'folds.json').write_text(json.dumps(result['folds'],ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/submission.json')
    p.add_argument('--phase',choices=['development','validation','holdout','engineering'],default='validation')
    p.add_argument('--allow-holdout',action='store_true',help='Explicitly open the final holdout for one frozen config')
    p.add_argument('--prices-csv',type=Path);p.add_argument('--returns-csv',type=Path)
    p.add_argument('--synthetic',action='store_true',help='Engineering smoke only, not market evidence')
    p.add_argument('--weight-drift',action='store_true')
    p.add_argument('--cost-bps',type=float,default=0.0)
    p.add_argument('--rebalance',type=int,default=20)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/backtest')
    a=p.parse_args();config=json.loads(a.config.read_text(encoding='utf-8'))
    if a.phase=='holdout':
        if not a.allow_holdout: p.error('Holdout is locked; use --allow-holdout only after freezing selection')
        if pd.Timestamp(config.get('selection_end','2017-12-31'))>VALIDATION_END:
            p.error('Config selection overlaps the holdout')
    if a.synthetic:
        if a.phase!='engineering' or a.prices_csv or a.returns_csv:
            p.error('--synthetic requires --phase engineering and no real-data CSV')
        rng=np.random.default_rng(5310)
        x=rng.normal(0,.01,(800,1))*np.linspace(.4,1.2,12)+rng.normal(0,1,(800,12))*np.linspace(.005,.02,12)
        frame=pd.DataFrame(x,index=pd.bdate_range('2000-01-03',periods=800),columns=[f'A{i}' for i in range(12)])
        start=end=None
        kind='SYNTHETIC_ENGINEERING_ONLY_NOT_INVESTMENT_EVIDENCE'
    else:
        if a.phase=='engineering': p.error('engineering phase requires --synthetic')
        start,end=phase_bounds(a.phase)
        frame=load_returns(prices_csv=a.prices_csv,returns_csv=a.returns_csv,end=end)
        kind='USER_LOCAL_CSV' if a.prices_csv or a.returns_csv else 'SKFOLIO_BUNDLED_SP500'
    result=walk_forward(frame,config,start=start,end=end,rebalance=a.rebalance,
                        weight_drift=a.weight_drift,cost_bps=a.cost_bps)
    meta={'dataset_type':kind,'phase':a.phase,'config':config,
          'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
          'is_teacher_grading_result':False,'holdout_opened':a.phase=='holdout'}
    save_result(result,a.output,meta)
    print(json.dumps({**meta,**result['summary']},ensure_ascii=False,indent=2,allow_nan=False))
    if result['summary']['failed_folds']: raise SystemExit(1)


if __name__=='__main__': main()
