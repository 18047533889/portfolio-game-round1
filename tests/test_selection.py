from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]

def tune():
    assert (ROOT/'research/tune.py').exists(),'Tuning script missing'
    return importlib.import_module('research.tune')

def test_grid_small_and_fixed():
    candidates=tune().candidate_grid(False)
    assert len(candidates)==14
    assert len({str(sorted(c.items())) for c in candidates})==14

def test_failed_strategy_cannot_win_selection():
    records=[{'candidate':'good','config':{},'summary':{'failed_folds':0,'failure_rate':0.,'degradation_rate':0.,
        'performance':{'sharpe':.7,'cagr':.1,'max_drawdown':.2}},'yearly':{2013:{'sharpe':.7,'cagr':.1,'max_drawdown':.2}}},
      {'candidate':'failed','config':{},'summary':{'failed_folds':1,'failure_rate':.1,'degradation_rate':0.,'performance':None},'yearly':{}}]
    ranked=tune().rank_records(records,'cagr')
    assert ranked[0]['candidate']=='good'
    assert ranked[-1]['eligible_for_selection'] is False

def test_costs_include_internal_daily_rebalancing():
    from research.engine import holding_returns
    _,_,turnover=holding_returns(np.array([[.1,0],[0,.1]]),np.array([.5,.5]),False)
    assert turnover[1]>0
    _,_,drift=holding_returns(np.array([[.1,0],[0,.1]]),np.array([.5,.5]),True)
    assert drift.sum()==0

def test_end_cut_prevents_future_model_fits():
    from research.engine import walk_forward
    x=pd.DataFrame(np.random.default_rng(10).normal(0,.01,(400,3)),index=pd.bdate_range('2017-01-02',periods=400))
    with pytest.raises(ValueError): walk_forward(x,{},end=pd.Timestamp('2017-02-01'))
