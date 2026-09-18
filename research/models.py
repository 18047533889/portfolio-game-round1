"""Comparable numerical-core models plus explicitly optional true skfolio Schur."""
from __future__ import annotations
import numpy as np
from portfolio_game.core import allocate, check_weights, estimate_covariance


def run_model(returns: np.ndarray, config: dict) -> dict:
    method = config.get("method", "regularized")
    params = {k: config[k] for k in ("half_life", "recent_mix", "anchor_penalty") if k in config}
    if method != "schur":
        return allocate(returns, method=method, **params)
    # No pretend Schur or silent substitution when the optional library is absent.
    from sklearn.utils.validation import validate_data
    from skfolio.moments import BaseCovariance
    from skfolio.prior import EmpiricalPrior
    from skfolio.distance import CovarianceDistance
    from skfolio.optimization import SchurComplementary

    class ConfiguredCovariance(BaseCovariance):
        def __init__(self, half_life=63.0, recent_mix=0.25):
            self.half_life = half_life
            self.recent_mix = recent_mix

        def fit(self, X, y=None):
            values = validate_data(self, X)
            self.covariance_ = estimate_covariance(values, self.half_life, self.recent_mix)
            return self

    kwargs = dict(half_life=config.get("half_life",63.0), recent_mix=config.get("recent_mix",.25))
    estimator = SchurComplementary(
        gamma=config.get("gamma",.5), keep_monotonic=True,
        prior_estimator=EmpiricalPrior(covariance_estimator=ConfiguredCovariance(**kwargs)),
        distance_estimator=CovarianceDistance(covariance_estimator=ConfiguredCovariance(**kwargs)),
        min_weights=0.0, max_weights=1.0,
    )
    x = np.asarray(returns,dtype=float)[-252:]
    estimator.fit(x)
    w = np.asarray(estimator.weights_,dtype=float)
    # Fairly report a noncompliant Schur output as failure, not a different model.
    check_weights(w,x.shape[1])
    return {"weights":w,"diagnostics":{"status":"schur","main_solver_converged":True,
        "anti_equal_weight_applied":False,"single_asset_fallback":False,
        "effective_gamma":float(estimator.effective_gamma_)}}
