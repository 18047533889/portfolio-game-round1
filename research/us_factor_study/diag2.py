#!/usr/bin/env python3
"""Is the US panel's return level real, or an artefact?

The universe book came out at ~32%/yr over 2004-2026, which is roughly three
times what an equal-weighted S&P 500 book actually did. Before any factor is
believed, find out whether the panel returns are real: check known tickers
against known total returns, and find which names produce the extreme 20-day
moves.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "pg"
panel = pd.read_parquet(ROOT / "cache" / "sp500_panel.parquet")
panel = panel.drop_duplicates(subset=["date", "ticker"], keep="last")


def wide(col):
    return panel[["date", "ticker", col]].pivot(index="date", columns="ticker", values=col).sort_index()


P = wide("adj_close")
R = wide("ret")
live = P.notna() & P.gt(0)

print("=== membership breadth ===")
print("unique tickers:", P.shape[1], " dates:", P.shape[0])
n_live = live.sum(axis=1)
print("live names/date: min %d p10 %d p50 %d max %d" % (
    n_live.min(), n_live.quantile(.1), n_live.median(), n_live.max()))

print("\n=== known tickers: panel cumulative return vs reality ===")
checks = [
    ("AAPL", "2004-01-02", "2004-12-31", 2.01),
    ("AAPL", "2004-01-02", "2024-12-31", 300.0),
    ("MSFT", "2004-01-02", "2024-12-31", 15.0),
    ("XOM", "2004-01-02", "2024-12-31", 4.0),
    ("GE", "2004-01-02", "2018-12-31", 0.4),
]
for tic, lo, hi, expect in checks:
    if tic not in P.columns:
        print(f"  {tic}: absent")
        continue
    s = P[tic].loc[lo:hi].dropna()
    if len(s) < 20:
        print(f"  {tic}: too few rows ({len(s)})")
        continue
    got = s.iloc[-1] / s.iloc[0] - 1
    print(f"  {tic} {lo}..{hi}: panel={got*100:>10.1f}%   reality~{expect*100:>10.1f}%")

print("\n=== extreme 20d forward returns: who and when ===")
fwd = (P.shift(-20) / P - 1.0).where(live)
fv = fwd.to_numpy()
mask = np.isfinite(fv) & (np.abs(fv) > 3)
idx = np.argwhere(mask)
print(f"  |20d ret| > 300%: {idx.shape[0]} observations")
rows = []
for i, j in idx[:20]:
    rows.append((str(fwd.index[i].date()), fwd.columns[j], float(fv[i, j])))
for r in rows:
    print("   ", r)
if idx.shape[0]:
    tick = pd.Series([fwd.columns[j] for i, j in idx])
    print("  worst offenders:", tick.value_counts().head(10).to_dict())

print("\n=== daily |ret| > 100% count per year ===")
Rv = R.where(live).to_numpy()
for y in sorted(set(R.index.year)):
    sel = (R.index.year == y)
    v = Rv[sel]
    print(f"  {y}: |r|>1: {(np.abs(v) > 1).sum():5d}   |r|>5: {(np.abs(v) > 5).sum():4d}   "
          f"n={np.isfinite(v).sum()}")

print("\n=== equal-weight book excluding |ret|>1 days ===")
Rc = R.where(live)
Rc = Rc.where(Rc.abs() <= 1.0)
Pc = (1.0 + Rc.fillna(0.0)).cumprod()
step = (Pc.shift(-20) / Pc - 1.0).where(live)
rebal = P.index[252::20]
rs = []
for d in rebal:
    r = step.loc[d].to_numpy()
    r = r[np.isfinite(r)]
    if r.size:
        rs.append(r.mean())
rs = np.asarray(rs)
ann = rs.mean() * 252 / 20
print(f"  n={rs.size} ann={ann:+.4f} sharpe={ann/(rs.std(ddof=1)*np.sqrt(252/20)):.4f}")
