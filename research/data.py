"""Public instructor dataset or explicit local input; chronological guardrails."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DEVELOPMENT_END = pd.Timestamp("2012-12-31")
VALIDATION_END = pd.Timestamp("2017-12-31")


def validate_returns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[1] < 2:
        raise ValueError("Research needs a nonempty DataFrame with >=2 asset columns")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Use a DatetimeIndex; first CSV column must contain dates")
    if frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Dates must be valid, unique and increasing; no silent sort")
    if not frame.columns.is_unique:
        raise ValueError("Asset names must be unique")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < -1):
        raise ValueError("Research backtest requires finite simple returns >= -1; clean upstream explicitly")
    return frame.astype(float)


def load_returns(*, prices_csv: str | Path | None = None,
                 returns_csv: str | Path | None = None,
                 end: pd.Timestamp | None = None) -> pd.DataFrame:
    if prices_csv and returns_csv:
        raise ValueError("Specify prices OR returns, not both")
    if prices_csv or returns_csv:
        frame = pd.read_csv(prices_csv or returns_csv, index_col=0, parse_dates=True)
        if end is not None:
            frame = frame.loc[frame.index <= end]
        if prices_csv:
            frame = frame.pct_change(fill_method=None).dropna(how="any")
    else:
        try:
            from skfolio.datasets import load_sp500_dataset
        except ImportError as exc:
            raise ImportError("Install requirements.txt or supply an explicit local CSV; no synthetic market results substituted") from exc
        # Same data entry as teacher_reference/self_test.py. No fill is needed
        # for the complete bundled dataset; fill_method=None avoids hidden padding.
        prices = load_sp500_dataset()
        if end is not None:
            prices = prices.loc[prices.index <= end]
        frame = prices.pct_change(fill_method=None).dropna(how="any")
    return validate_returns(frame)


def visible_for_tuning(frame: pd.DataFrame) -> pd.DataFrame:
    """Never expose 2018+ returns to the tuning stage."""
    return frame.loc[frame.index <= VALIDATION_END].copy()


def phase_bounds(phase: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if phase == "development": return None, DEVELOPMENT_END
    if phase == "validation": return DEVELOPMENT_END + pd.Timedelta(days=1), VALIDATION_END
    if phase == "holdout": return VALIDATION_END + pd.Timedelta(days=1), None
    raise ValueError("phase must be development, validation or holdout")
