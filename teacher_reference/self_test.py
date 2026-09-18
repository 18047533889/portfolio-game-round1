"""
Quick self-check script to run before submitting.

Usage:
    python self_test.py path/to/your_portfolio.py

The file must define a class named exactly CVXPYPortfolio that inherits from
skfolio's BaseOptimization and implements fit() setting self.weights_.

If the submission does not satisfy the basic requirements (class name,
inheritance, constructor signature, fit / weights_), the script raises an error.

Note: this check only covers the basic submission format. It does NOT check
leverage, short selling, or any other extra rules specific to each portfolio
game round — those are enforced by the grading system for that round.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import BaseOptimization


def load_submission(path):
    """Import a submission file and return its CVXPYPortfolio class."""
    spec = importlib.util.spec_from_file_location("submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cls = getattr(module, "CVXPYPortfolio", None)
    if cls is None:
        raise NameError("The file must define a class named CVXPYPortfolio (exact match)")
    if not (isinstance(cls, type) and issubclass(cls, BaseOptimization)):
        raise TypeError("CVXPYPortfolio must inherit from skfolio.optimization.BaseOptimization")
    return cls


def main(path):
    cls = load_submission(path)

    # Fit once and check the basic format of weights_ only.
    # Leverage, short selling, and other per-round rules are NOT checked here.
    X = load_sp500_dataset().pct_change().dropna()
    model = cls(portfolio_params={"name": "CVXPYPortfolio"})  # same instantiation as the grader
    model.raise_on_failure = False
    model.fit(X)
    if getattr(model, "error_", None):
        raise RuntimeError(f"fit failed: {model.error_}")

    w = model.weights_
    if not isinstance(w, np.ndarray) or w.shape != (X.shape[1],):
        raise ValueError(f"weights_ must be a numpy array of shape ({X.shape[1]},)")
    if not np.isfinite(w).all():
        raise ValueError("weights_ contains NaN or infinity")

    # Run a real backtest on skfolio's built-in dataset
    pred = cross_val_predict(model, X, cv=WalkForward(train_size=252, test_size=63), n_jobs=1)
    print("Basic checks passed. Backtest summary:")
    print(pred.summary())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python self_test.py path/to/your_portfolio.py")
        sys.exit(1)
    main(Path(sys.argv[1]))
