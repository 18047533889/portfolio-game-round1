"""REAL skfolio integration. Missing packages cause SKIP, never mock success."""
from pathlib import Path
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

skfolio=pytest.importorskip('skfolio',reason='BLOCKED: genuine skfolio is not installed')
from sklearn.base import clone
from skfolio.optimization import BaseOptimization
from skfolio.model_selection import WalkForward,cross_val_predict
from portfolio_game.core import allocate,check_weights
ROOT=Path(__file__).resolve().parents[1]


def load():
    path=ROOT/'submission/portfolio_round1.py'
    spec=importlib.util.spec_from_file_location('submission',path)
    module=importlib.util.module_from_spec(spec)
    # Intentionally identical to the teacher: do not register sys.modules entry.
    spec.loader.exec_module(module)
    return module.CVXPYPortfolio


def frame(t=700,n=8):
    rng=np.random.default_rng(5310)
    x=rng.normal(0,.01,(t,1))+rng.normal(size=(t,n))*np.linspace(.005,.02,n)
    return pd.DataFrame(x,index=pd.bdate_range('2000-01-03',periods=t),columns=[f'A{i}' for i in range(n)])


def test_teacher_import_constructor_clone_fit_predict():
    cls=load();assert issubclass(cls,BaseOptimization)
    model=cls(portfolio_params={'name':'CVXPYPortfolio'})
    model.raise_on_failure=False
    cloned=clone(model)
    x=frame()
    assert cloned.fit(x.iloc[:252]) is cloned
    assert not getattr(cloned,'error_',None)
    check_weights(cloned.weights_,x.shape[1])
    prediction=cloned.predict(x.iloc[252:272])
    np.testing.assert_allclose(prediction.returns,x.iloc[252:272].to_numpy()@cloned.weights_,atol=1e-12)


@pytest.mark.parametrize('step',[20,63])
def test_real_walk_forward(step):
    prediction=cross_val_predict(load()(portfolio_params={'name':'Round1'}),frame(),
                    cv=WalkForward(train_size=252,test_size=step),n_jobs=1)
    assert np.isfinite(prediction.returns).all()


def test_adapter_and_numerical_core_are_identical():
    x=frame().iloc[:252]
    model=load()().fit(x)
    np.testing.assert_allclose(model.weights_,allocate(x.to_numpy())['weights'],atol=1e-12)


def test_real_adapter_handles_missing_fit_input():
    x=frame().iloc[:252].copy();x.iloc[0,0]=np.nan
    model=load()().fit(x)
    check_weights(model.weights_,x.shape[1])


def test_standalone_import_without_repository_on_path():
    with tempfile.TemporaryDirectory() as temp:
        target=Path(temp)/'teacher.py';shutil.copy2(ROOT/'submission/portfolio_round1.py',target)
        code=("import importlib.util,numpy as np; "
              "s=importlib.util.spec_from_file_location('s','teacher.py'); "
              "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
              "x=np.random.default_rng(42).normal(0,.01,(252,5)); "
              "p=m.CVXPYPortfolio(portfolio_params={'name':'Round1'}).fit(x); "
              "assert p.weights_.shape==(5,) and np.isfinite(p.weights_).all()")
        subprocess.run([sys.executable,'-I','-c',code],cwd=temp,check=True)
