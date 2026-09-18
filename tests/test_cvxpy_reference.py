"""Optional independent QP check using genuine CVXPY, not required by submission."""
import numpy as np
import pytest
cp=pytest.importorskip('cvxpy',reason='BLOCKED: CVXPY independent reference not installed')
from portfolio_game.core import estimate_covariance,hrp_weights,solve_regularized_qp,relative_objective

@pytest.mark.parametrize('penalty',[0.,.3,1.,3.])
def test_qp_matches_cvxpy(penalty):
    rng=np.random.default_rng(11);x=rng.normal(0,.01,(252,10))*np.linspace(.5,2,10)
    c=estimate_covariance(x,63,.25);b=hrp_weights(c)
    w,info=solve_regularized_qp(c,b,penalty,max_iter=3000)
    v=cp.Variable(len(b))
    objective=cp.quad_form(v,cp.psd_wrap(c))/(b@c@b)+penalty*cp.sum_squares(v-b)/(b@b)
    problem=cp.Problem(cp.Minimize(objective),[v>=0,cp.sum(v)==1]);problem.solve()
    assert problem.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE)
    assert info['converged']
    assert abs(relative_objective(w,c,b,penalty)-problem.value)<1e-6
