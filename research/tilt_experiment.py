"""Would a factor tilt actually help, under the grader's own contract?

The A-share study (COS `mining_outputs`, 263 mined factors) ranks its winners by
RankIC ~0.05, and the mechanism that carries them is price-volume covariance and
range intensity. Ported to a point-in-time S&P 500 universe, the OHLCV versions
keep t-stats around -2, but the grader never gives us OHLCV: round 1 hands over
a 252 x N matrix of daily returns. Only the returns-only arm can ship.

So the question this script answers is the only one that matters for the
deliverable: over the *grader's own* datasets and rebalance contract, does a
returns-only composite signal, used as a linear tilt on the existing convex QP,

    min  w'Cw/(b'Cb) + penalty * ||w-b||^2/||b||^2 - gamma * s'w   s.t. w in simplex

out-rank plain gamma = 0? It is evaluated three ways, all of which must agree
before anything is shipped:

  A. full-history walk-forward on 4 skfolio datasets, teacher metrics;
  B. the teacher-shaped score (10/10/10/70 rank percentile blend);
  C. the grader-shaped stress: many random asset subsets x random 2-year
     windows, counting how often the tilt beats the baseline on Sharpe, on max
     drawdown, and on annual return, plus any contract violation.

Nothing here writes to the submission. gamma = 0 reproduces the shipped file.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "src" / "portfolio_game" / "core.py"
TRAIN, STEP, HOLD = 252, 20, 20


def core():
    spec = importlib.util.spec_from_file_location("core_tilt", CORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# returns-only signal: only features that were sign-stable in the US study
# --------------------------------------------------------------------------- #
def _z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    good = np.isfinite(v)
    if good.sum() < 3:
        return np.zeros_like(v)
    z = np.zeros_like(v)
    mu, sd = v[good].mean(), v[good].std(ddof=1)
    if sd > 0:
        z[good] = (v[good] - mu) / sd
    return z


def composite_signal(x: np.ndarray, kind: str = "stable") -> np.ndarray | None:
    """Cross-sectionally standardised blend of return-only signals.

    Oriented so that a higher value means a higher expected return, using the
    signs estimated on a point-in-time S&P 500 universe. Features were kept only
    if the sign agreed in all three disjoint sub-periods there:
      corr_mkt252, corr_vol_mkt, kurt60, volofvol60, mom231 (weak).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 60 or x.shape[1] < 3:
        return None
    if not np.isfinite(x).all():
        return None
    n = x.shape[1]
    market = x.mean(axis=1)

    if kind == "ashare_vol":
        # The literal transfer the request asks for: the A-share library's
        # winning families are the volatility / amplitude complex (Var(returns),
        # Decay(Abs(returns)), range intensity), all oriented positive -- i.e.
        # "higher realised risk -> higher expected return". Range intensity needs
        # OHLCV, so the returns-only remnant of that family is realised vol.
        rv20 = np.array([x[i:i + 20].std(axis=0, ddof=1) for i in range(len(x) - 19)])
        rv60 = np.array([x[i:i + 60].std(axis=0, ddof=1) for i in range(len(x) - 59)])
        early = x[: max(1, len(x) // 2)]
        late = x[len(x) // 2:]
        parts = [
            _z(rv20[-1]),
            _z(rv60[-1]),
            _z(early.std(axis=0, ddof=1)),
            _z(late.std(axis=0, ddof=1)),
            _z(np.abs(x[-20:]).mean(axis=0)),
            _z(rv20[-1] / np.where(rv60[-1] > 0, rv60[-1], np.nan)),
        ]
        stack = np.vstack(parts)
        signal = np.nanmean(stack, axis=0)
        signal = np.where(np.isfinite(signal), signal, 0.0)
        return signal - signal.mean()

    parts: list[np.ndarray] = []

    def corr_with(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            a = np.repeat(a[:, None], n, axis=1)
        am = a - a.mean(axis=0, keepdims=True)
        bm = b - b.mean()
        num = (am * bm[:, None]).sum(axis=0)
        den = np.sqrt((am ** 2).sum(axis=0) * (bm ** 2).sum())
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        return out

    # corr_mkt252 : co-movement with the equal-weighted book
    parts.append(_z(corr_with(x, market)))
    # corr_vol_mkt : co-movement of absolute moves
    parts.append(_z(corr_with(np.abs(x), np.abs(market))))

    # kurt60 and volofvol60
    tail = x[-60:]
    mu = tail.mean(axis=0, keepdims=True)
    sd = tail.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 0, sd, np.nan)
    kurt = np.nanmean(((tail - mu) / sd) ** 4, axis=0) - 3.0
    parts.append(_z(kurt))

    rv = np.array([x[i:i + 20].std(axis=0, ddof=1) for i in range(len(x) - 19)])
    parts.append(_z(rv.std(axis=0, ddof=1)))

    # mom231 : 12-1 momentum on the return-only price path
    if x.shape[0] >= 232:
        path = (1.0 + x).cumprod(axis=0)
        mom = path[-21] / path[-231] - 1.0
        parts.append(_z(mom))

    stack = np.vstack(parts)
    signal = np.nanmean(stack, axis=0)
    signal = np.where(np.isfinite(signal), signal, 0.0)
    return signal - signal.mean()


# --------------------------------------------------------------------------- #
# tilted QP: identical to the shipped solver except for the linear term
# --------------------------------------------------------------------------- #
def solve_tilted(core_mod, cov, anchor, penalty, signal, gamma, *, max_iter=600,
                 tol=1e-8):
    c = np.asarray(cov, float)
    b = core_mod._normalise(anchor)
    risk, norm = float(b @ c @ b), float(b @ b)
    q = c / risk + np.eye(len(b)) * (penalty / norm)
    p = penalty * b / norm
    if signal is not None and gamma:
        # the anchor term is O(penalty) and the risk term O(1), so the tilt must
        # be on that same scale: p = 0.5 * gamma * s, not gamma * s / n
        p = p + 0.5 * gamma * np.asarray(signal, float)
    lip = 2.0 * float(np.max(np.sum(np.abs(q), axis=1)))

    def obj(v):
        return float(v @ q @ v - 2 * (p @ v) + penalty)

    def gap(v):
        g = 2.0 * (q @ v - p)
        return max(0.0, float(g @ v - g.min())) / (1.0 + abs(obj(v)))

    w = b.copy()
    y, mom, val = w.copy(), 1.0, obj(w)
    for it in range(1, max_iter + 1):
        cand = core_mod.project_simplex(y - (2.0 / lip) * (q @ y - p))
        nv = obj(cand)
        if nv > val + 1e-13:
            y, mom = w.copy(), 1.0
            cand = core_mod.project_simplex(y - (2.0 / lip) * (q @ y - p))
            nv = obj(cand)
        rg = gap(cand)
        prev, w, val = w, cand, nv
        if rg <= tol:
            return core_mod._normalise(w), True, rg
        nm = (1.0 + math.sqrt(1.0 + 4.0 * mom ** 2)) * .5
        y = w + ((mom - 1.0) / nm) * (w - prev)
        mom = nm
    return core_mod._normalise(w), False, rg


def allocate_tilted(mod, x: np.ndarray, gamma: float,
                    kind: str = "stable") -> np.ndarray:
    """Same pipeline as the shipped file, with the tilt switched on at the end.

    Falls back exactly like the shipped file does: the tree anchor, then the
    inverse-risk book, then the deterministic last-resort budget. Equal weight is
    never a candidate.
    """
    x = np.array(x[-TRAIN:], dtype=float)
    n = x.shape[1]
    if n == 1:
        return np.ones(1)
    valid = np.isfinite(x) & (x >= -1.0)
    counts = valid.sum(axis=0)
    required = min(len(x), max(2, math.ceil(.8 * len(x))))
    eligible = (counts >= required) & valid[-1]
    risks = mod._column_variances(x, valid)
    if not eligible.any():
        eligible = (counts > 0) & valid[-1]
        if not eligible.any():
            eligible = counts == counts.max()
    idx = np.flatnonzero(eligible)
    if idx.size <= 1 or idx.size > mod.MAX_DENSE_ASSETS:
        w = np.zeros(n)
        w[idx] = mod._inverse_risk(risks[idx]) if idx.size else 0.0
        return mod._normalise(w) if idx.size else np.full(n, 1.0 / n)
    common = np.all(valid[:, idx], axis=1)
    complete = x[np.ix_(common, idx)]
    if len(complete) < max(2, min(32, len(x) // 2)):
        w = np.zeros(n)
        w[idx] = mod._inverse_risk(risks[idx])
        return mod._normalise(w)
    try:
        cov = mod.estimate_covariance(complete, 63.0, 0.25)
        anchor = mod.hrp_weights(cov)
        sig = composite_signal(complete, kind) if gamma else None
        sol, ok, _ = solve_tilted(mod, cov, anchor, 0.25, sig, gamma)
        if not ok:
            raise ArithmeticError("no gap certificate")
        w = np.zeros(n)
        w[idx] = sol
        return mod._normalise(w)
    except Exception:
        w = np.zeros(n)
        try:
            w[idx] = mod.hrp_weights(mod.estimate_covariance(complete, 63.0, 0.25))
            return mod._normalise(w)
        except Exception:
            w[idx] = mod._inverse_risk(risks[idx])
            return mod._normalise(w)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def datasets():
    import skfolio.datasets as ds
    return {
        "sp500(20)": ds.load_sp500_dataset,
        "ftse100(64)": ds.load_ftse100_dataset,
        "factors(5)": ds.load_factors_dataset,
        "sp500_index(1)": ds.load_sp500_index,
    }


def returns_of(loader) -> pd.DataFrame:
    return loader().pct_change(fill_method=None).dropna()


def walk_forward(mod, X: pd.DataFrame, gamma: float,
                 kind: str = "stable") -> tuple[dict, int]:
    x = X.to_numpy(float)
    n_obs, n = x.shape
    per_period: list[float] = []
    weights: list[np.ndarray] = []
    failed = 0
    for pos in range(TRAIN, n_obs - STEP + 1, STEP):
        try:
            w = allocate_tilted(mod, x[max(0, pos - TRAIN):pos], gamma, kind)
            if w.shape != (n,) or not np.isfinite(w).all() or abs(w.sum() - 1) > 1e-9:
                failed += 1
                continue
            if n > 1 and np.allclose(w, 1.0 / n, atol=1e-9):
                failed += 1
                continue
            r = x[pos:pos + STEP] @ w
            if not np.isfinite(r).all():
                failed += 1
                continue
            per_period.append(float(r.mean()))
            weights.append(w)
        except BaseException:
            failed += 1
    r = np.asarray(per_period)
    if r.size < 8:
        return {"sharpe": None, "maxdd": None, "annual": None, "n": int(r.size)}, failed
    ppy = 252.0 / STEP
    ann = float(r.mean() * ppy)
    sd = float(r.std(ddof=1))
    cum = np.cumsum(r)
    W = np.vstack(weights)
    return {
        "sharpe": float(ann / (sd * np.sqrt(ppy))) if sd > 0 else None,
        "maxdd": float(np.max(np.maximum.accumulate(cum) - cum)),
        "annual": ann,
        "n": int(r.size),
        "max_weight_p50": float(np.median(W.max(axis=1))),
        "effective_n_p50": float(np.median(1.0 / (W ** 2).sum(axis=1))),
    }, failed


def rank_pct(v, higher):
    from scipy.stats import rankdata
    v = np.asarray(v, float)
    r = rankdata(-v if higher else v, method="average")
    return 1.0 - (r - 0.5) / len(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gammas", default="0,0.5,1.0,2.0,4.0")
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--kinds", default="stable")
    ap.add_argument("--out", default="outputs/round1-audit/tilt_experiment.json")
    args = ap.parse_args()
    gammas = [float(g) for g in args.gammas.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    mod = core()

    print("=== A. full-history walk-forward (teacher contract) ===")
    table: dict[str, dict] = {}
    for label, loader in datasets().items():
        X = returns_of(loader)
        table[label] = {}
        for g in gammas:
            m, failed = walk_forward(mod, X, g, kinds[0])
            m["failed"] = failed
            table[label][g] = m
            print(f"{label:14s} gamma={g:<5} sharpe={_f(m['sharpe'])} "
                  f"maxdd={_f(m['maxdd'])} ann={_f(m['annual'])} "
                  f"maxw={_f(m.get('max_weight_p50'),3)} failed={failed}")

    print("\n=== B. teacher-shaped score (10/10/10/70 rank blend) ===")
    labels = [l for l in table if all(table[l][g]["sharpe"] is not None for g in gammas)]
    score = {g: 0.0 for g in gammas}
    for l in labels:
        s = rank_pct([table[l][g]["sharpe"] for g in gammas], True)
        d = rank_pct([table[l][g]["maxdd"] for g in gammas], False)
        a = rank_pct([table[l][g]["annual"] for g in gammas], True)
        f = rank_pct([table[l][g]["failed"] for g in gammas], False)
        for i, g in enumerate(gammas):
            score[g] += 0.1 * s[i] + 0.1 * d[i] + 0.1 * a[i] + 0.7 * f[i]
    for g in gammas:
        print(f"  gamma={g:<5} score={score[g]/len(labels):.4f}")

    print("\n=== C. random subset x random 2-year window (grader stress) ===")
    rng = np.random.default_rng(20260919)
    Xs = {l: returns_of(loader) for l, loader in datasets().items() if l != "sp500_index(1)"}
    wins = {g: {"sharpe": 0, "maxdd": 0, "annual": 0, "n": 0} for g in gammas if g != 0.0}
    for label, X in Xs.items():
        arr = X.to_numpy(float)
        n_total = arr.shape[1]
        for _ in range(args.reps):
            k = int(rng.integers(2, n_total + 1)) if n_total > 2 else 2
            cols = rng.choice(n_total, size=k, replace=False)
            if arr.shape[0] > 504:
                start = int(rng.integers(0, arr.shape[0] - 504 + 1))
            else:
                start = 0
            block = arr[start:start + 504][:, cols]
            if block.shape[0] < TRAIN + 3 * STEP:
                continue
            base, bf = walk_forward(mod, pd.DataFrame(block), 0.0, kinds[0])
            if base["sharpe"] is None or bf:
                continue
            for g in wins:
                m, failed = walk_forward(mod, pd.DataFrame(block), g, kinds[0])
                if m["sharpe"] is None or failed:
                    continue
                wins[g]["n"] += 1
                wins[g]["sharpe"] += int(m["sharpe"] > base["sharpe"])
                wins[g]["maxdd"] += int(m["maxdd"] < base["maxdd"])
                wins[g]["annual"] += int(m["annual"] > base["annual"])
    for g, w in wins.items():
        n = max(1, w["n"])
        print(f"  gamma={g:<5} n={w['n']:4d}  beat-baseline: "
              f"sharpe={w['sharpe']/n:.3f} maxdd={w['maxdd']/n:.3f} "
              f"annual={w['annual']/n:.3f}")

    out = ROOT / args.out
    out.write_text(json.dumps({
        "gammas": gammas, "full_history": {l: {str(g): table[l][g] for g in gammas}
                                           for l in table},
        "teacher_score": {str(g): score[g] / max(1, len(labels)) for g in gammas},
        "stress_wins": {str(g): w for g, w in wins.items()},
        "stress_reps": args.reps,
    }, indent=2, default=str))
    print(f"\nwritten -> {out}")
    return 0


def _f(v, nd=4):
    return "  None" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    raise SystemExit(main())
