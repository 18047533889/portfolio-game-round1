"""
Example portfolio submission for the student_portfolios folder.

This file shows the expected structure for your submission: a class named
``CVXPYPortfolio`` that subclasses skfolio's ``BaseOptimization`` and implements
``fit(self, X, y=None)`` by setting ``self.weights_`` (a numpy array of shape
(n_assets,)) and returning ``self``. Call ``validate_data(self, X)`` first to
turn the returns DataFrame into a numpy array.
"""

import cvxpy as cp
import numpy as np
from sklearn.utils.validation import validate_data
from skfolio.optimization import BaseOptimization


class CVXPYPortfolio(BaseOptimization):
    """Example: Minimum Variance Portfolio."""

    def fit(self, X, y=None):
        X = validate_data(self, X)
        n_assets = X.shape[1]

        # Minimize portfolio variance, long-only, fully invested
        cov_matrix = np.cov(X.T)
        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [
            cp.sum(w) == 1,  # Budget constraint
            w >= 0,          # Long-only constraint
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.solver_status_ = prob.status

        if prob.status == cp.OPTIMAL:
            self.weights_ = w.value
        else:
            # Fallback to equal weights if optimization fails
            self.weights_ = np.ones(n_assets) / n_assets
        return self
