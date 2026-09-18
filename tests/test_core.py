"""Tests run against the real NumPy/SciPy/sklearn core, not a fake skfolio."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1] / "src/portfolio_game/core.py"

def core():
    assert CORE.is_file(), "Numerical core has not been implemented"
    spec = importlib.util.spec_from_file_location("core_under_test", CORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def sample(t=252, n=20, seed=42):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.01, (t, 1))
    x = market * np.linspace(.3, 1.4, n) + rng.normal(size=(t,n)) * np.linspace(.004,.025,n)
    return x

def valid(w, n):
    assert isinstance(w,np.ndarray) and w.shape==(n,)
    assert np.isfinite(w).all() and (w>=0).all()
    assert abs(w.sum()-1)<1e-12
    if n>1: assert not np.allclose(w, np.full(n,1/n))

@pytest.mark.parametrize("n", [2,3,5,20,63,257])
def test_allocation_contract(n):
    r=core().allocate(sample(n=n))
    valid(r["weights"], n)

@pytest.mark.parametrize("t", [1,2,5,20,63,252,400])
def test_short_and_full_lookback(t):
    r=core().allocate(sample(t=t,n=7))
    valid(r["weights"],7)
    assert r["diagnostics"]["rows_used"] <=252

def test_no_input_mutation():
    x=sample(); previous=x.copy()
    core().allocate(x)
    np.testing.assert_array_equal(x,previous)

def test_last_252_only():
    x=sample(t=504); a=core().allocate(x)["weights"]
    x[:252]=.1
    b=core().allocate(x)["weights"]
    np.testing.assert_array_equal(a,b)

def test_scale_invariance():
    x=sample(); a=core().allocate(x)["weights"]; b=core().allocate(x*2)["weights"]
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-8)

def test_permutation_equivariance_without_ties():
    x=sample(); p=np.random.default_rng(7).permutation(x.shape[1])
    a=core().allocate(x)["weights"]; b=core().allocate(x[:,p])["weights"]
    np.testing.assert_allclose(a[p],b,atol=5e-6)

def test_zero_and_duplicate_data():
    for x in (np.zeros((252,10)),np.full((50,5),.001),np.tile(sample(n=2)[:,0:1],(1,8))):
        r=core().allocate(x); valid(r["weights"],x.shape[1])
        if np.ptp(x, axis=0).max() == 0:
            assert r["diagnostics"]["anti_equal_weight_applied"]

def test_nan_inf_are_not_imputed_as_riskless():
    x=sample(n=6); x[:,0]=np.nan; x[-1,1]=np.inf
    r=core().allocate(x)
    valid(r["weights"],6)
    assert r["weights"][0]==0 and r["weights"][1]==0

def test_all_missing_has_explicit_degraded_status():
    r=core().allocate(np.full((252,4),np.nan)); valid(r["weights"],4)
    assert r["diagnostics"]["status"]=="emergency_no_observations"

def test_sparse_common_rows_uses_diagonal_fallback():
    x=sample(t=252,n=80)
    for j in range(80): x[j::80,j]=np.nan
    r=core().allocate(x); valid(r["weights"],80)

def test_huge_finite_values_do_not_overflow():
    x=sample(n=7); x[0,0]=1e300
    r=core().allocate(x);valid(r["weights"],7)

def test_single_asset_is_flagged():
    r=core().allocate(np.zeros((20,1)))
    np.testing.assert_array_equal(r["weights"],[1.])
    assert r["diagnostics"]["single_asset_rule_ambiguity"] is True

@pytest.mark.parametrize("x",[np.zeros((0,3)),np.zeros((3,0)),np.zeros(3),np.array([["bad","x"]])])
def test_invalid_contract_rejected(x):
    with pytest.raises(ValueError): core().allocate(x)

def test_covariance_symmetric_positive_definite():
    mod=core(); c=mod.estimate_covariance(sample(),63.,.25)
    np.testing.assert_allclose(c,c.T,atol=1e-12)
    assert np.linalg.eigvalsh(c).min()>0

def test_qp_beats_anchor_and_gap_is_small():
    mod=core(); c=mod.estimate_covariance(sample(),63.,.25); b=mod.hrp_weights(c)
    w,info=mod.solve_regularized_qp(c,b,1.)
    assert info["converged"]
    assert mod.relative_objective(w,c,b,1.) <=mod.relative_objective(b,c,b,1.)+1e-9
    assert info["relative_gap"]<1e-7

def test_qp_forced_nonconvergence_is_explicit():
    mod=core(); c=mod.estimate_covariance(sample(),63.,.25); b=mod.hrp_weights(c)
    _,info=mod.solve_regularized_qp(c,b,0.,max_iter=0)
    assert not info["converged"]

def test_failed_solver_falls_back_to_real_anchor(monkeypatch):
    mod=core()
    def fail(*args,**kwargs): raise np.linalg.LinAlgError("injected solver failure")
    monkeypatch.setattr(mod,"solve_regularized_qp",fail)
    r=mod.allocate(sample()); valid(r["weights"],20)
    assert r["diagnostics"]["status"]=="fallback_hrp"
    assert any("injected" in s for s in r["diagnostics"]["events"])

def test_simplex_projection():
    p=core().project_simplex(np.array([-3.,.2,4.]))
    np.testing.assert_allclose(p,[0,0,1])

@pytest.mark.parametrize("params",[{"half_life":0},{"recent_mix":1.2},{"anchor_penalty":-1},{"method":"unknown"}])
def test_bad_hyperparameters_rejected(params):
    with pytest.raises(ValueError): core().allocate(sample(),**params)

@pytest.mark.parametrize("penalty", [0.0, 0.3, 1.0, 3.0])
def test_qp_matches_independent_scipy_slsqp(penalty):
    from scipy.optimize import minimize
    mod=core(); c=mod.estimate_covariance(sample(n=8),63.,.25); b=mod.hrp_weights(c)
    w,info=mod.solve_regularized_qp(c,b,penalty,max_iter=3000)
    reference=minimize(lambda v: mod.relative_objective(v,c,b,penalty), b,
        bounds=[(0,1)]*len(b),constraints={"type":"eq","fun":lambda v:v.sum()-1},
        method="SLSQP", options={"ftol":1e-12,"maxiter":2000})
    assert reference.success and info["converged"]
    assert abs(mod.relative_objective(w,c,b,penalty)-reference.fun)<1e-7
    np.testing.assert_allclose(w,reference.x,atol=5e-5)
