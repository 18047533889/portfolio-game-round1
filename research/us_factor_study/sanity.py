#!/usr/bin/env python3
"""Sanity probes before trusting any factor result on the US panel."""
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
C = wide("close")
F = wide("adj_factor")
live = P.notna() & P.gt(0)

print("=== adj_factor distribution (non-null) ===")
af = F.to_numpy()
af = af[np.isfinite(af)]
print("min %.3e p1 %.4f p50 %.4f p99 %.4f max %.3e" % (
    af.min(), np.quantile(af, .01), np.quantile(af, .5), np.quantile(af, .99), af.max()))

print("\n=== per-year stats of the 20d forward return, equal-weight universe ===")
fwd = (P.shift(-20) / P - 1.0).where(live)
rows = []
for y, grp in fwd.groupby(fwd.index.year):
    v = grp.to_numpy()
    v = v[np.isfinite(v)]
    if v.size < 100:
        continue
    rows.append((y, v.size, float(np.mean(v)), float(np.median(v)), float(np.quantile(v, .99))))
for y, n, m, med, p99 in rows:
    print(f"  {y}: n={n:7d} mean20d={m:+.4f} med20d={med:+.4f} p99={p99:+.3f}  "
          f"annualised_mean={m*252/20:+.4f}")

print("\n=== ret identity check: ret vs adj_close pct_change ===")
pct = P.pct_change().where(live)
diff = (pct - R.where(live)).abs()
d = diff.to_numpy()
d = d[np.isfinite(d)]
print(f"  n={d.size} median|diff|={np.median(d):.3e} p99={np.quantile(d,.99):.3e} max={d.max():.3e}")

print("\n=== valid-name counts per rebalance date for three rolling windows ===")
Rl = R.where(live)
for w in (10, 20, 60):
    f = Rl.rolling(w).std()
    rebal = P.index[252::20]
    cnt = [int(np.isfinite(f.loc[d]).sum()) for d in rebal if d in f.index]
    cnt = np.asarray(cnt)
    print(f"  rv{w}: rebalances={cnt.size} valid_min={cnt.min()} valid_p50={int(np.median(cnt))}")

print("\n=== equal-weight book over the whole universe (reference) ===")
step = (P.shift(-20) / P - 1.0).where(live)
rebal = P.index[252::20]
rs = []
for d in rebal:
    r = step.loc[d].to_numpy()
    r = r[np.isfinite(r)]
    if r.size:
        rs.append(r.mean())
rs = np.asarray(rs)
ann = rs.mean() * 252 / 20
sd = rs.std(ddof=1)
print(f"  n={rs.size} ann={ann:+.4f} sharpe={ann/(sd*np.sqrt(252/20)):.4f}")
cum = np.cumsum(rs)
print(f"  maxdd={np.max(np.maximum.accumulate(cum)-cum):.4f}")

print("\n=== top-20% by rv20, per-period detail ===")
rv = Rl.rolling(20).std()
for lo, hi in [(2003, 2010), (2011, 2018), (2019, 2026)]:
    sel = [d for d in rebal if lo <= d.year <= hi]
    rs, sizes, years = [], [], []
    for d in sel:
        a = rv.loc[d].to_numpy()
        r = step.loc[d].to_numpy()
        good = np.flatnonzero(np.isfinite(a) & np.isfinite(r))
        if good.size < 30:
            continue
        k = max(1, int(round(good.size * .2)))
        ch = good[np.argsort(-a[good])[:k]]
        rs.append(r[ch].mean())
        sizes.append(k)
        years.append(d.year)
    rs = np.asarray(rs)
    if rs.size < 5:
        print(f"  {lo}-{hi}: too few periods ({rs.size})")
        continue
    ann = rs.mean() * 252 / 20
    print(f"  {lo}-{hi}: n={rs.size} names~{int(np.mean(sizes))} ann={ann:+.4f} "
          f"sharpe={ann/(rs.std(ddof=1)*np.sqrt(252/20)):.3f} "
          f"mean20d={rs.mean():+.4f} max20d={rs.max():+.3f} min20d={rs.min():+.3f}")
