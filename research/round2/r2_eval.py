"""Round 2 evaluation harness.

Round 2 changes three things relative to round 1:

  * shortselling = TRUE   (round 1: FALSE)
  * weight_drift = TRUE   (round 1: not mentioned, so it defaulted to False)
  * scoring: annual return 15%, max drawdown 15%, failure rate 70%
    (round 1: Sharpe 10%, max drawdown 10%, annual 10%, failure 70%)
    -- Sharpe is no longer scored, and annual return now carries the same
    weight as drawdown.

Everything else is unchanged: lookback 252, optimize_every 20, leverage 1.

This module provides the shared plumbing so that every candidate is measured
under exactly the same contract. It deliberately does NOT contain a portfolio
method; it only measures one.
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.validation import validate_data
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import BaseOptimization

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
LOOKBACK = 252
STEP = 20


def load_core(path: str | Path | None = None):
    """Import a portfolio core module by file path."""
    target = Path(path) if path else REPO / "src" / "portfolio_game" / "core.py"
    spec = importlib.util.spec_from_file_location("r2core", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import core from {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_datasets(names: list[str]) -> dict[str, pd.DataFrame]:
    """Load skfolio datasets as return panels (the only input the grader gives)."""
    from skfolio.datasets import (load_factors_dataset, load_ftse100_dataset,
                                  load_nasdaq_dataset, load_sp500_dataset)
    registry = {
        "sp500": load_sp500_dataset,
        "ftse100": load_ftse100_dataset,
        "nasdaq": load_nasdaq_dataset,
        "factors": load_factors_dataset,
    }
    out: dict[str, pd.DataFrame] = {}
    for raw in names:
        name, _, size = raw.partition("(")
        n = int(size[:-1]) if size else None
        frame = registry[name.strip()]()
        returns = frame.pct_change(fill_method=None).dropna()
        if n is not None and n < returns.shape[1]:
            returns = returns.iloc[:, :n]
        out[raw] = returns
    return out


class Probe(BaseOptimization):
    """Thin sklearn-style wrapper so skfolio's backtester can drive a core.

    The only thing added over the core function is the estimator contract the
    grader uses: validate_data, weights_, and a clean fit.
    """

    def __init__(self, penalty: float = 0.25, method: str = "regularized",
                 half_life: float = 63.0, recent_mix: float = 0.25,
                 portfolio_params: dict | None = None, fallback=None,
                 previous_weights=None, raise_on_failure: bool = True):
        super().__init__(portfolio_params=portfolio_params, fallback=fallback,
                         previous_weights=previous_weights,
                         raise_on_failure=raise_on_failure)
        self.penalty = penalty
        self.method = method
        self.half_life = half_life
        self.recent_mix = recent_mix

    def fit(self, X, y=None):
        X = validate_data(self, X)
        if self.method == "equal_weight":
            # Reference only -- equal weight is prohibited as a strategy in
            # round 2. Routed through the identical backtest path so the
            # comparison cannot be contaminated by a different accounting.
            n = np.asarray(X).shape[1]
            self.weights_ = np.ones(n, dtype=float) / n
            self.diagnostics_ = {"status": "equal_weight_reference"}
            return self
        result = self._core().allocate(
            np.asarray(X, dtype=float), half_life=self.half_life,
            recent_mix=self.recent_mix, anchor_penalty=self.penalty,
            method=self.method,
        )
        self.weights_ = np.asarray(result["weights"], dtype=float)
        self.diagnostics_ = result["diagnostics"]
        return self

    def _core(self):
        if not hasattr(self, "_core_cache"):
            self._core_cache = load_core()
        return self._core_cache


def run(X: pd.DataFrame, penalty: float, method: str = "regularized",
        *, weight_drift: bool = True, half_life: float = 63.0,
        recent_mix: float = 0.25) -> dict:
    """One walk-forward run under the round-2 contract."""
    cv = WalkForward(train_size=LOOKBACK, test_size=STEP)
    model = Probe(penalty=penalty, method=method, half_life=half_life,
                  recent_mix=recent_mix,
                  portfolio_params={"weight_drift": weight_drift, "name": "probe"})
    model.raise_on_failure = False
    pred = cross_val_predict(model, X, cv=cv, n_jobs=1)

    n_total = len(pred)
    n_failed = int(getattr(pred, "n_failed_portfolios", 0) or 0)
    out = {
        "n_portfolios": n_total,
        "n_failed": n_failed,
        "failure_rate": n_failed / n_total if n_total else 1.0,
        "n_fallback": int(getattr(pred, "n_fallback_portfolios", 0) or 0),
    }
    if n_total - n_failed > 0:
        out["annual_return"] = float(pred.annualized_mean)
        out["max_drawdown"] = float(pred.max_drawdown)
        out["sharpe"] = float(pred.annualized_sharpe_ratio)
    else:
        out["annual_return"] = float("nan")
        out["max_drawdown"] = float("nan")
        out["sharpe"] = float("nan")
    return out


def reference_equal_weight(X: pd.DataFrame) -> dict:
    """Equal-weight book over the identical out-of-sample dates.

    Equal weight is prohibited as a strategy in this round, so this is a
    reference point only -- never a submission. It is driven through the same
    Probe/backtest path as every candidate so that the comparison uses one
    accounting and not two.
    """
    return run(X, 0.0, "equal_weight")
