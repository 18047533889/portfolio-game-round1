from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]

def module(name):
    assert (ROOT/f'research/{name}.py').exists(),f'{name} research module not implemented'
    return importlib.import_module('research.'+name)

def test_drawdown_includes_initial_capital():
    m=module('metrics').performance(np.array([-.1, .05]),252)
    assert m['max_drawdown']==pytest.approx(.1)

def test_geometric_return_and_arithmetic_annual_mean_are_distinct():
    m=module('metrics').performance(np.array([.1,-.1]),2)
    assert m['cagr']==pytest.approx(-.01)
    assert m['annualized_mean']==pytest.approx(0.)

def test_metrics_zero_volatility_is_not_infinity():
    m=module('metrics').performance(np.zeros(20),252)
    assert m['sharpe'] is None

def test_failed_observations_are_not_silently_dropped():
    with pytest.raises(ValueError): module('metrics').performance(np.array([.1,np.nan,.2]),252)

def test_input_dates_must_be_ordered():
    frame=pd.DataFrame({'A':[.01,.02],'B':[0,.01]},index=pd.to_datetime(['2000-01-03','2000-01-02']))
    with pytest.raises(ValueError): module('data').validate_returns(frame)

def test_split_protects_holdout():
    mod=module('data')
    frame=pd.DataFrame(np.zeros((2000,3)),index=pd.bdate_range('2011-01-03',periods=2000))
    visible=mod.visible_for_tuning(frame)
    assert visible.index.max()<=pd.Timestamp('2017-12-31')

def test_walk_forward_uses_past_only_and_has_all_test_rows():
    mod=module('engine')
    rng=np.random.default_rng(4)
    x=pd.DataFrame(rng.normal(0,.01,(400,4)),index=pd.bdate_range('2000-01-03',periods=400))
    result=mod.walk_forward(x, {'method':'regularized'},lookback=252,rebalance=20)
    assert len(result['returns'])==148
    for fold in result['folds']:
        assert pd.Timestamp(fold['train_end'])<pd.Timestamp(fold['test_start'])
        assert fold['train_rows']==252
    assert result['summary']['failed_folds']==0

def test_failed_fit_is_counted_and_no_sharpe_is_reported():
    mod=module('engine')
    x=pd.DataFrame(np.full((300,3),.001),index=pd.bdate_range('2000-01-03',periods=300))
    def failure(*args,**kwargs): raise RuntimeError('test failure')
    r=mod.walk_forward(x,{},allocator=failure)
    assert r['summary']['failed_folds']==3
    assert r['summary']['performance'] is None
    assert np.isnan(r['returns']).all()

def test_buy_hold_vs_constant_weight_returns():
    mod=module('engine')
    r=np.array([[.1,0],[0,.1]])
    constant,_,_=mod.holding_returns(r,np.array([.5,.5]),False)
    drifted,_,_=mod.holding_returns(r,np.array([.5,.5]),True)
    np.testing.assert_allclose(constant,[.05,.05])
    np.testing.assert_allclose(drifted,[.05, .05/1.05])

def test_schur_missing_is_not_substituted_by_another_algorithm():
    mod=module('models')
    if importlib.util.find_spec('skfolio') is not None:
        pytest.skip('This test covers absent optional dependency only')
    with pytest.raises(ImportError): mod.run_model(np.ones((252,3)),{'method':'schur'})

def test_loading_tuning_data_excludes_holdout_before_validation(tmp_path):
    mod=module('data')
    csv=tmp_path/'returns.csv'
    pd.DataFrame({'A':[.01,np.nan],'B':[.02,np.nan]},
        index=pd.to_datetime(['2017-12-29','2018-01-02'])).to_csv(csv)
    historical=mod.load_returns(returns_csv=csv,end=pd.Timestamp('2017-12-31'))
    assert len(historical)==1 and historical.index.max().year==2017
