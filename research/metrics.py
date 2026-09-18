"""Explicit metrics: include initial wealth; never drop failed return rows."""
from __future__ import annotations
import math
import numpy as np


def performance(returns: np.ndarray, annualization: int = 252) -> dict:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 1 or not np.isfinite(r).all() or np.any(r < -1):
        raise ValueError("Performance needs complete, finite simple returns; failed rows must not be dropped")
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    if not np.isfinite(wealth).all():
        raise FloatingPointError("Wealth overflow; result is not reportable")
    high = np.maximum.accumulate(wealth)
    drawdown = 1.0 - wealth / high
    vol = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(np.mean(r) / vol * math.sqrt(annualization)) if vol > 1e-15 else None
    cagr = float(wealth[-1] ** (annualization / len(r)) - 1.0)
    return {
        "n_observations": len(r), "cumulative_return": float(wealth[-1] - 1.0),
        "cagr": cagr, "annualized_mean": float(np.mean(r) * annualization),
        "annualized_volatility": vol * math.sqrt(annualization),
        "sharpe": sharpe, "max_drawdown": float(np.max(drawdown)),
        "risk_free_rate_assumption": 0.0,
    }
