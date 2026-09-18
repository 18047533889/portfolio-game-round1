#!/usr/bin/env python3
"""Which A-share factor families survive on a point-in-time S&P 500 universe? (v2)

v1 reported an equal-weighted universe book of ~32%/yr, which is ~3x what an
equal-weighted S&P 500 basket actually did. The cause was found and is worth
recording: the COS adjusted-close column carries a handful of split/reverse-
split days whose adjusted return was not neutralised, so ~200 observations show
|1-day return| between 2x and 19x (Citigroup's 1:10 reverse split, AIG's 1:20,
BRK.B's 50:1, AAPL's 7:1, MS/WB in 2005 and 2008 ...). A cross-sectional
*arithmetic mean* is defenceless against those, and it dragged the whole
reported level up by 20 percentage points a year. After dropping |r| > 50%
(never a real one-day move in a US large cap outside an acquisition) the same
book lands at ~9.6%/yr with Sharpe ~0.49, which is right for the period.

Everything below runs on the cleaned returns, and every rolling window states
its own min_periods instead of relying on pandas' default (which silently
requires a fully dense window and, on a gap-carrying membership panel, is the
difference between a factor and an empty column).

Two independent yardsticks per factor:
  1. RankIC vs the next 20-day forward return -- the A-share library's own
     yardstick -- split into disjoint sub-periods so a sign flip is visible
     instead of averaged away;
  2. a long-only top-quintile book rebalanced every 20 days, scored with the
     portfolio game's own annualised Sharpe / max drawdown / annual return.
Equal weighting inside the quintile is a factor-evaluation convention here
(the standard long-only quantile test); it is NOT the submitted portfolio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "pg"
PANEL = ROOT / "cache" / "sp500_panel.parquet"
OUT = ROOT / "factor_study_v2.json"

WARMUP = 252
STEP = 20
HOLD = 20
MIN_NAMES = 60
GLITCH = 0.5          # |1-day return| above this is a corporate-action artefact
QUANTILE = 0.2
PERIODS = [("2004-2010", "2004-01-01", "2010-12-31"),
           ("2011-2017", "2011-01-01", "2017-12-31"),
           ("2018-2026", "2018-01-01", "2026-12-31")]

RETURN_ONLY = {
    "rv10", "rv20", "rv60", "rv252", "dabs10", "dabs20", "downside_vol60",
    "upside_vol60", "skew60", "kurt60", "volofvol60", "autocorr20", "mom231",
    "mom21", "rev5", "rev10", "beta252", "corr_mkt252", "residvol252",
    "winmaxdd60", "mom120_vol", "corr_vol_mkt",
}


def wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    frame = panel[["date", "ticker", col]].drop_duplicates(
        subset=["date", "ticker"], keep="last")
    return frame.pivot(index="date", columns="ticker", values=col).sort_index()


def minp(w: int) -> int:
    return max(10, int(round(w * 0.6)))


def roll(a: pd.DataFrame, w: int) -> pd.core.window.rolling.Rolling:
    return a.rolling(w, min_periods=minp(w))


def rcov(a: pd.DataFrame, b: pd.DataFrame, w: int) -> pd.DataFrame:
    return a.rolling(w, min_periods=minp(w)).cov(b)


def rcorr(a: pd.DataFrame, b: pd.DataFrame, w: int) -> pd.DataFrame:
    return a.rolling(w, min_periods=minp(w)).corr(b)


def clean_returns(R: pd.DataFrame, live: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Returns with corporate-action artefacts removed, plus a usable price path."""
    Rl = R.where(live)
    bad = Rl.abs() > GLITCH
    n_bad = int(bad.to_numpy().sum())
    Rf = Rl.mask(bad).fillna(0.0)      # flat on missing/artefact days
    P = (1.0 + Rf).cumprod()
    return Rl.mask(bad), P, n_bad


def build_factors(P: pd.DataFrame, Rf: pd.DataFrame, live: pd.DataFrame,
                  V, H, L, O, A) -> dict[str, pd.DataFrame]:
    mkt = Rf.mean(axis=1)
    mkt_wide = pd.DataFrame(
        np.repeat(mkt.to_numpy()[:, None], Rf.shape[1], axis=1),
        index=Rf.index, columns=Rf.columns)

    f: dict[str, pd.DataFrame] = {}
    # ---- returns-only arm: what the submission could actually compute ----
    f["rv10"] = roll(Rf, 10).std()
    f["rv20"] = roll(Rf, 20).std()
    f["rv60"] = roll(Rf, 60).std()
    f["rv252"] = roll(Rf, 252).std()
    f["dabs10"] = Rf.abs().ewm(halflife=10, min_periods=30).mean()
    f["dabs20"] = Rf.abs().ewm(halflife=20, min_periods=60).mean()
    f["downside_vol60"] = roll(Rf.clip(upper=0), 60).std()
    f["upside_vol60"] = roll(Rf.clip(lower=0), 60).std()
    f["skew60"] = roll(Rf, 60).skew()
    f["kurt60"] = roll(Rf, 60).kurt()
    f["volofvol60"] = roll(roll(Rf, 20).std(), 60).std()
    f["autocorr20"] = rcorr(Rf, Rf.shift(1), 20)
    f["mom231"] = (P / P.shift(231) - 1.0).where(live)
    f["mom21"] = (P / P.shift(21) - 1.0).where(live)
    f["rev5"] = -(P / P.shift(5) - 1.0).where(live)
    f["rev10"] = -(P / P.shift(10) - 1.0).where(live)
    f["mom120_vol"] = ((P / P.shift(120) - 1.0).where(live)) / roll(Rf, 120).std()
    var_m = roll(mkt_wide, 252).var()
    beta = rcov(Rf, mkt_wide, 252) / var_m.replace(0, np.nan)
    f["beta252"] = beta.where(live)
    f["corr_mkt252"] = rcorr(Rf, mkt_wide, 252).where(live)
    f["residvol252"] = roll(Rf - beta * mkt_wide, 252).std().where(live)
    f["corr_vol_mkt"] = rcorr(Rf.abs(), mkt_wide.abs(), 60).where(live)
    f["winmaxdd60"] = -((Rf.cumsum() - Rf.cumsum().rolling(60, min_periods=36).max()))

    # ---- OHLCV arm: the A-share winners, mechanism check only ----
    f["range_intensity"] = (H - L) / (H + L).replace(0, np.nan)
    f["range_decay"] = f["range_intensity"].ewm(halflife=5, min_periods=10).mean()
    f["hl_ratio"] = (H - L) / L.replace(0, np.nan)
    f["cov_dhigh_vol10"] = rcov(H.diff(), V, 10)
    f["cov_dhigh_dvol10"] = rcov(H.diff(), V.diff(), 10)
    f["cov_range_vol20"] = rcov(H - L, V, 20)
    f["cov_ret_vol20"] = rcov(Rf, V, 20)
    f["cov_ret_amt20"] = rcov(Rf, A, 20)
    f["absret_amt20"] = (Rf.abs() * A).ewm(halflife=15, min_periods=20).mean()
    f["intraday_var10"] = roll((H - O) / O.replace(0, np.nan), 10).var()
    f["shadows"] = roll((H - np.maximum(O, P)) / (H - L).replace(0, np.nan), 10).mean()
    f["vwap_dist10"] = roll((O - P) / P.replace(0, np.nan), 10).mean()
    f["vol_z20"] = (V - roll(V, 20).mean()) / roll(V, 20).std().replace(0, np.nan)
    f["turn_z20"] = V / roll(V, 60).mean() - 1.0
    f["amihud20"] = roll(Rf.abs() / A.replace(0, np.nan), 20).mean()
    return f


def rank_ic(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    good = m & np.isfinite(a) & np.isfinite(b)
    if good.sum() < MIN_NAMES:
        return np.nan
    ra = pd.Series(a[good]).rank().to_numpy()
    rb = pd.Series(b[good]).rank().to_numpy()
    # pandas 3 returns read-only arrays, so centre out of place
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


def evaluate(fac: pd.DataFrame, fwd: pd.DataFrame, step: pd.DataFrame,
             live: pd.DataFrame, dates: pd.DatetimeIndex, top: float) -> dict:
    fa, fw, st, ms = fac.to_numpy(), fwd.to_numpy(), step.to_numpy(), live.to_numpy()
    ics, rets, turns = [], [], []
    prev: set[int] = set()
    for d in dates:
        if d not in fac.index:
            continue
        row = fac.index.get_loc(d)
        if row < WARMUP or row + HOLD >= len(fac.index):
            continue
        ics.append(rank_ic(fa[row], fw[row], ms[row]))
        a, m, r = fa[row], ms[row], st[row]
        good = np.flatnonzero(m & np.isfinite(a) & np.isfinite(r))
        if good.size < MIN_NAMES:
            continue
        k = max(1, int(round(good.size * top)))
        ch = good[np.argsort(-a[good])[:k]]
        rets.append(float(r[ch].mean()))
        cur = set(ch.tolist())
        turns.append(1.0 - len(cur & prev) / max(1, len(cur)))
        prev = cur
    ics = np.asarray(ics, dtype=float)
    ics = ics[np.isfinite(ics)]
    r = np.asarray(rets)
    ppy = 252.0 / HOLD
    out = {"n_ic": int(ics.size),
           "ic": float(ics.mean()) if ics.size else None,
           "ic_std": float(ics.std(ddof=1)) if ics.size > 1 else None,
           "icir": (float(ics.mean() / ics.std(ddof=1))
                    if ics.size > 1 and ics.std(ddof=1) > 0 else None),
           "ic_t": (float(ics.mean() / ics.std(ddof=1) * np.sqrt(ics.size))
                    if ics.size > 1 and ics.std(ddof=1) > 0 else None),
           "ic_pos_share": float((ics > 0).mean()) if ics.size else None,
           "n_book": int(r.size)}
    if r.size >= 8:
        ann = float(r.mean() * ppy)
        sd = float(r.std(ddof=1))
        cum = np.cumsum(r)
        out.update({
            "sharpe": float(ann / (sd * np.sqrt(ppy))) if sd > 0 else None,
            "maxdd": float(np.max(np.maximum.accumulate(cum) - cum)),
            "annual": ann,
            "turnover": float(np.mean(turns)),
        })
    else:
        out.update({"sharpe": None, "maxdd": None, "annual": None, "turnover": None})
    return out


def main() -> int:
    panel = pd.read_parquet(PANEL)
    P_raw = wide(panel, "adj_close")
    R = wide(panel, "ret")
    live = P_raw.notna() & P_raw.gt(0)
    Rl, P, n_bad = clean_returns(R, live)
    print(f"panel {R.shape}  {R.index.min().date()}..{R.index.max().date()}")
    print(f"corporate-action artefacts removed: {n_bad} observations "
          f"(|1-day ret| > {GLITCH:.0%})")

    step = (P.shift(-HOLD) / P - 1.0).where(live)
    fwd = step
    rebal = P.index[WARMUP::STEP]

    # reference: the whole-universe equal-weight book on cleaned returns
    ref = evaluate(pd.DataFrame(1.0, index=P.index, columns=P.columns), fwd, step,
                   live, rebal, top=1.0)
    print(f"REFERENCE equal-weight universe: annual={ref['annual']:.4f} "
          f"sharpe={ref['sharpe']:.4f} maxdd={ref['maxdd']:.4f} n={ref['n_book']}")
    if not (0.03 <= ref["annual"] <= 0.20):
        print("WARNING: reference book outside a plausible S&P 500 EW range")

    factors = build_factors(P, Rl.fillna(0.0), live, wide(panel, "volume"),
                            wide(panel, "high"), wide(panel, "low"),
                            wide(panel, "open"), wide(panel, "amount"))
    print(f"factors: {len(factors)}")

    results: dict[str, dict] = {}
    for name, fac in factors.items():
        fac = fac.where(live)
        entry = {"returns_only": name in RETURN_ONLY,
                 "full": evaluate(fac, fwd, step, live, rebal, QUANTILE),
                 "periods": {}}
        for label, lo, hi in PERIODS:
            sel = rebal[(rebal >= pd.Timestamp(lo)) & (rebal <= pd.Timestamp(hi))]
            entry["periods"][label] = evaluate(fac, fwd, step, live, sel, QUANTILE)
        results[name] = entry
        fl = entry["full"]
        per = [entry["periods"][l]["ic"] for l, _, _ in PERIODS]
        print(f"{name:20s} ic={_f(fl['ic'])} t={_f(fl['ic_t'],1)} "
              f"sharp={_f(fl['sharpe'],3)} ann={_f(fl['annual'])} "
              f"| per-period ic {[None if v is None else round(v,4) for v in per]}")

    OUT.write_text(json.dumps({
        "universe": "S&P 500 point-in-time membership",
        "panel": {"rows": int(len(panel)), "dates": int(R.shape[0]),
                  "tickers": int(R.shape[1]),
                  "start": str(R.index.min().date()), "end": str(R.index.max().date())},
        "hygiene": {"glitch_threshold": GLITCH, "glitch_observations": n_bad},
        "reference_equal_weight_universe": ref,
        "n_rebalances": int(len(rebal)),
        "ic_horizon_days": HOLD, "rebalance_days": STEP,
        "quantile": QUANTILE, "min_names": MIN_NAMES,
        "factors": results,
    }, indent=2, default=str))
    print(f"written -> {OUT}")
    return 0


def _f(v, nd=4):
    return "  None" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    raise SystemExit(main())
