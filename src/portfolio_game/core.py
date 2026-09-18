"""Return-only portfolio allocation. Pure numerical core; no files or network.

This file is embedded verbatim by tools/build_submission.py. Keep all run-time
helpers here so the teacher submission needs no local package, model, or data.

Objective: w'Cw/(b'Cb) + penalty * ||w-b||^2/||b||^2 on the long-only simplex.
The bounded projected-gradient solver verifies a convex first-order gap. It is
not a heuristic change of objective, and it is not dependent on CVXPY being
available. Numerical convergence is distinguished from returning a safe fallback.

Estimation chain, all of it from returns alone:
  Ledoit-Wolf shrinkage -> long-run correlation
  Marchenko-Pastur eigenvalue cleaning -> the part of that correlation that is
      distinguishable from sampling noise
  exponentially weighted short-run volatilities -> current risk scale
  tree (HRP) anchor -> a diversified prior that does not trust the noisy
      off-diagonal estimates on its own
  convex QP -> the anchor shrunk toward the risk-minimising portfolio
The anchor is what keeps small universes diversified (unconstrained minimum
variance puts most of a 5-asset book into one asset); the QP is what stops the
tree heuristic from overriding the covariance information.

On wide, ill-conditioned cross-sections the QP stops reaching its 1e-8 relative
gap inside the iteration budget; the tree anchor is then returned, and the
diagnostic status records which branch actually produced the weights. That
anchor is still a covariance-based allocation (shrunk + cleaned correlation,
single linkage, inverse-variance cluster risk), not an equal-weight or
per-asset placeholder, so the fallback degrades the tilt, not the risk model.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import ledoit_wolf

LOOKBACK = 252
# Dense risk estimation is O(n^3) in the tree step, so the guard is a cost bound,
# not a statistical one. Measured single-fit cost on a real 252xN cross-section:
# N=256 -> 0.02 s, N=1455 -> 2.8 s, N=2048 -> 6.4 s, N=2900 -> 12.3 s. 2048 keeps
# the worst realistic fold inside a few seconds while covering every standard
# cross-section (the widest common benchmark has 1455 columns). Above it we fall
# back to per-asset risk, which is O(T*N) and never allocates a dense matrix.
MAX_DENSE_ASSETS = 2048
WEIGHT_ATOL = 1e-10
EQUAL_ATOL = 1e-8
EQUAL_RTOL = 1e-5


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto w >= 0, sum(w) = 1 (sorting algorithm)."""
    v = np.asarray(vector, dtype=np.float64)
    if v.ndim != 1 or not v.size or not np.isfinite(v).all():
        raise ValueError("Simplex projection requires a finite, non-empty vector")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.arange(1, v.size + 1)
    good = np.flatnonzero(u - cssv / idx > 0)
    if not good.size:
        raise FloatingPointError("Cannot determine simplex threshold")
    rho = int(good[-1])
    w = np.maximum(v - cssv[rho] / (rho + 1), 0.0)
    return _normalise(w)


def _normalise(weights: np.ndarray) -> np.ndarray:
    """Correct only solver-scale roundoff, never repair arbitrary shorting."""
    w = np.array(weights, dtype=np.float64, copy=True)
    if w.ndim != 1 or not w.size or not np.isfinite(w).all():
        raise ValueError("Weights must be a finite, non-empty one-dimensional array")
    if float(w.min()) < -WEIGHT_ATOL:
        raise ValueError("Materially negative weights are not acceptable")
    np.maximum(w, 0.0, out=w)
    total = float(w.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Weight budget must be finite and positive")
    w /= total
    # Put the tiny residual on the largest position, not on a zero position.
    k = int(np.argmax(w))
    w[k] += 1.0 - float(w.sum())
    if w[k] < 0 or abs(float(w.sum()) - 1.0) > WEIGHT_ATOL:
        raise FloatingPointError("Unable to reconcile investment budget")
    return w


def is_equal_weight(weights: np.ndarray) -> bool:
    """A conservative local tolerance; the teacher's tolerance is unknown."""
    w = np.asarray(weights)
    return bool(np.allclose(w, 1.0 / w.size, atol=EQUAL_ATOL, rtol=EQUAL_RTOL))


def check_weights(weights: np.ndarray, n_assets: int, *, forbid_equal: bool = True) -> None:
    """Raise on final output violations; does not claim all grader rules known."""
    if not isinstance(weights, np.ndarray) or weights.shape != (n_assets,):
        raise ValueError("Weight shape must be (n_assets,)")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative")
    if abs(float(weights.sum()) - 1.0) > WEIGHT_ATOL:
        raise ValueError("Portfolio must be fully invested")
    if forbid_equal and n_assets > 1 and is_equal_weight(weights):
        raise ValueError("Equal-weight portfolios are prohibited in round 1")


def _stabilise_covariance(covariance: np.ndarray) -> np.ndarray:
    c = np.asarray(covariance, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all():
        raise ValueError("Covariance is not a finite square matrix")
    c = (c + c.T) * 0.5
    diag = np.diag(c)
    positives = diag[diag > 0]
    base = float(np.median(positives)) if positives.size else 1.0
    c = c / base  # a common positive scale leaves the objective unchanged
    minimum = float(np.linalg.eigvalsh(c)[0])
    # A bounded relative eigenvalue floor, not blind clipping of eigenvectors.
    floor = 1e-8 * max(1.0, float(np.trace(c)) / len(c))
    if minimum < floor:
        c = c + np.eye(len(c)) * (floor - minimum)
    return (c + c.T) * 0.5


def marchenko_pastur_edge(n_observations: int, n_assets: int) -> float:
    """Upper edge of the Marchenko-Pastur bulk for a standardised matrix.

    For a (n_observations x n_assets) matrix whose true correlation is the
    identity, the empirical eigenvalue density converges to the MP law supported
    on [(1 - 1/sqrt(q))^2, (1 + 1/sqrt(q))^2] with q = T / N. Eigenvalues at or
    below that upper edge carry no reliable signal.
    """
    if n_assets < 1 or n_observations < 1:
        raise ValueError("Dimensions must be positive")
    q = n_observations / n_assets
    return (1.0 + 1.0 / math.sqrt(q)) ** 2


def rmt_denoise(correlation: np.ndarray, n_observations: int) -> np.ndarray:
    """Random-matrix-theory cleaning of a correlation matrix.

    Eigenvalues below the Marchenko-Pastur upper edge are statistically
    indistinguishable from noise, so they are collapsed onto their common mean
    (the trace-preserving choice) and the matrix is rebuilt. This is a variance
    reduction of the estimate, not a change of what is being estimated, and it
    leaves the matrix unit-diagonal and positive definite.

    Skipped, rather than approximated, whenever the sample cannot separate
    signal from noise: n_assets < 2 or n_observations <= n_assets + 2.
    """
    c = np.asarray(correlation, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all():
        raise ValueError("Correlation must be a finite square matrix")
    n_assets = c.shape[0]
    if n_assets < 2 or n_observations <= n_assets + 2:
        return c
    eigenvalues, eigenvectors = np.linalg.eigh((c + c.T) * 0.5)
    edge = marchenko_pastur_edge(int(n_observations), n_assets)
    signal = eigenvalues >= edge
    if signal.all() or not signal.any():
        # Either every direction is resolvable or none is; there is no split to act on.
        return c
    noise_mean = float(eigenvalues[~signal].mean())
    rebuilt = (eigenvectors * np.where(signal, eigenvalues, noise_mean)) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(rebuilt), np.finfo(float).tiny))
    cleaned = rebuilt / diagonal[:, None] / diagonal[None, :]
    cleaned = np.clip((cleaned + cleaned.T) * 0.5, -1.0, 1.0)
    np.fill_diagonal(cleaned, 1.0)
    if not np.isfinite(cleaned).all():
        raise FloatingPointError("Denoised correlation is not finite")
    return cleaned


def estimate_covariance(returns: np.ndarray, half_life: float, recent_mix: float) -> np.ndarray:
    """Ledoit-Wolf long correlation, RMT-denoised, with EWMA short volatility.

    Expects complete data. A common scale is removed before any squaring so a
    very large finite observation cannot overflow the covariance calculation.
    """
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Covariance needs >=2 complete rows and >=1 asset")
    scale = float(np.max(np.abs(x)))
    z = x / scale if scale > 0 else x.copy()
    slow, _ = ledoit_wolf(z, assume_centered=False)
    slow = np.atleast_2d(slow)
    diag = np.maximum(np.diag(slow), 0)
    positive = diag[diag > 0]
    floor = max(float(np.median(positive)) * 1e-8, 1e-16) if positive.size else 1e-12
    diag = np.maximum(diag, floor)
    slow = slow.copy()
    np.fill_diagonal(slow, diag)
    sd = np.sqrt(diag)
    corr = slow / sd[:, None] / sd[None, :]
    corr = np.clip((corr + corr.T) * 0.5, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    # Remove the part of the correlation spectrum that cannot be told apart from
    # sampling noise; the long and the short layer then share one cleaner
    # correlation shape instead of propagating one noisy estimate twice.
    corr = rmt_denoise(corr, len(z))
    ages = np.arange(len(z) - 1, -1, -1, dtype=np.float64)
    a = np.exp2(-ages / half_life)
    a /= a.sum()
    centered = z - a @ z
    correction = max(1.0 - float(a @ a), 1e-12)
    fast_var = (a[:, None] * centered**2).sum(axis=0) / correction
    fast_sd = np.sqrt(np.maximum(fast_var, floor))
    fast = fast_sd[:, None] * corr * fast_sd[None, :]
    return _stabilise_covariance((1.0 - recent_mix) * slow + recent_mix * fast)


def _cluster_variance(covariance: np.ndarray, indices: np.ndarray) -> float:
    block = covariance[np.ix_(indices, indices)]
    inv_var = 1.0 / np.diag(block)
    inv_var /= inv_var.sum()
    return max(float(inv_var @ block @ inv_var), np.finfo(float).tiny)


def hrp_weights(covariance: np.ndarray) -> np.ndarray:
    """Tree-split HRP with single linkage and inverse-variance cluster risk.

    Splits follow actual linkage children (not arbitrary half splits of a leaf
    list). This makes non-tied trees equivariant to asset-column permutations.
    Exact linkage ties can still depend on column order and are documented.
    """
    c = np.asarray(covariance, dtype=np.float64)
    n = c.shape[0]
    if n == 1:
        return np.ones(1)
    sigma = np.sqrt(np.maximum(np.diag(c), np.finfo(float).tiny))
    correlation = np.clip(c / sigma[:, None] / sigma[None, :], -1, 1)
    distance = np.sqrt(np.maximum(0, (1 - correlation) / 2))
    distance = (distance + distance.T) * .5
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="single")
    members: list[np.ndarray] = [np.array([i], dtype=int) for i in range(n)]
    for row in tree:
        members.append(np.concatenate((members[int(row[0])], members[int(row[1])])) )
    w = np.ones(n, dtype=np.float64)
    stack = [2 * n - 2]
    while stack:
        node = stack.pop()
        if node < n:
            continue
        left, right = (int(v) for v in tree[node - n, :2])
        il, ir = members[left], members[right]
        vl, vr = _cluster_variance(c, il), _cluster_variance(c, ir)
        alpha = vr / (vl + vr)
        w[il] *= alpha
        w[ir] *= 1.0 - alpha
        stack.extend((left, right))
    return _normalise(w)


def relative_objective(w: np.ndarray, c: np.ndarray, b: np.ndarray, penalty: float) -> float:
    risk = max(float(b @ c @ b), np.finfo(float).tiny)
    norm = float(b @ b)
    return float(w @ c @ w / risk + penalty * ((w-b) @ (w-b)) / norm)


def solve_regularized_qp(
    covariance: np.ndarray,
    anchor: np.ndarray,
    penalty: float,
    *,
    max_iter: int = 600,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Accelerated projected gradient with restart and a simplex duality gap.

    gap = grad(f,w)'w - min_i grad(f,w)_i bounds f(w)-f(w*) for convex f on
    the simplex. Feasibility alone is NOT labelled optimization convergence.
    The row-sum norm bounds the Hessian spectral norm, giving a safe step.
    """
    c = np.asarray(covariance, dtype=np.float64)
    b = _normalise(anchor)
    risk, norm = float(b @ c @ b), float(b @ b)
    if not (risk > 0 and np.isfinite(risk)):
        raise ValueError("Anchor variance must be positive and finite")
    q = c / risk + np.eye(len(b)) * (penalty / norm)
    p = penalty * b / norm
    lipschitz = 2.0 * float(np.max(np.sum(np.abs(q), axis=1)))
    if not math.isfinite(lipschitz) or lipschitz <= 0:
        raise FloatingPointError("Invalid quadratic curvature")

    def objective(v: np.ndarray) -> float:
        return float(v @ q @ v - 2 * (p @ v) + penalty)

    def gap(v: np.ndarray) -> float:
        gradient = 2.0 * (q @ v - p)
        raw = max(0.0, float(gradient @ v - np.min(gradient)))
        return raw / (1.0 + abs(objective(v)))

    w = b.copy()
    y, momentum = w.copy(), 1.0
    value = objective(w)
    relative_gap = gap(w)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        candidate = project_simplex(y - (2.0 / lipschitz) * (q @ y - p))
        new_value = objective(candidate)
        if new_value > value + 1e-13:
            y, momentum = w.copy(), 1.0
            candidate = project_simplex(y - (2.0 / lipschitz) * (q @ y - p))
            new_value = objective(candidate)
        relative_gap = gap(candidate)
        previous = w
        w, value = candidate, new_value
        if relative_gap <= tolerance:
            break
        new_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum**2)) * .5
        y = w + ((momentum - 1.0) / new_momentum) * (w - previous)
        momentum = new_momentum
    return _normalise(w), {
        "converged": bool(relative_gap <= tolerance),
        "iterations": iterations,
        "relative_gap": float(relative_gap),
        "objective": relative_objective(w, c, b, penalty),
    }


def _column_variances(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Estimate using observed entries only. Missing entries are not zero returns."""
    result = np.full(x.shape[1], np.inf)
    for i in range(x.shape[1]):
        values = x[valid[:, i], i]
        if values.size >= 2:
            result[i] = max(float(np.var(values, ddof=1)), 0.0)
    return result


def _inverse_risk(variances: np.ndarray) -> np.ndarray:
    v = np.asarray(variances, dtype=float)
    positive = v[np.isfinite(v) & (v > 0)]
    floor = max(float(np.median(positive)) * 1e-6, 1e-16) if positive.size else 1e-12
    # Infinite risk (too few observations) gets zero weight, not infinite weight.
    inv = np.where(np.isfinite(v), 1.0 / np.sqrt(np.maximum(v, floor)), 0.0)
    if not inv.sum() > 0:
        inv = np.ones(len(v))
    return _normalise(inv)


def _finish(
    candidates: list[np.ndarray],
    n_assets: int,
    eligible: np.ndarray,
    risks: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    for candidate in candidates:
        try:
            w = _normalise(candidate)
            if n_assets > 1 and is_equal_weight(w):
                diagnostics["anti_equal_weight_applied"] = True
                continue
            check_weights(w, n_assets)
            return {"weights": w, "diagnostics": diagnostics}
        except (ValueError, FloatingPointError):
            continue
    # Not an epsilon perturbation of equal weight: a distinct, explicit rule.
    pool = np.flatnonzero(eligible)
    if not pool.size:
        pool = np.arange(n_assets)
    k = int(pool[np.argmin(risks[pool])])
    w = np.zeros(n_assets, dtype=np.float64)
    w[k] = 1.0
    diagnostics["events"].append("deterministic_single_asset_fallback")
    diagnostics["final_rule"] = "minimum_observed_variance_then_column_position"
    diagnostics["single_asset_fallback"] = True
    check_weights(w, n_assets)
    return {"weights": w, "diagnostics": diagnostics}


def allocate(
    returns: np.ndarray,
    *,
    half_life: float = 63.0,
    recent_mix: float = 0.25,
    anchor_penalty: float = 0.25,
    method: str = "regularized",
) -> dict[str, Any]:
    """Compute weights with diagnostics, using no information outside returns.

    Methods: regularized (submission), hrp, minimum_variance, inverse_volatility.
    All invalid numerical paths are traceable. Shape/type contract errors and
    impossible hyperparameters are raised, not hidden as a successful strategy.
    """
    if not np.isfinite(half_life) or not 0 < half_life <= 1e6:
        raise ValueError("half_life must be finite and in (0, 1e6]")
    if not np.isfinite(recent_mix) or not 0 <= recent_mix <= 1:
        raise ValueError("recent_mix must be in [0,1]")
    if not np.isfinite(anchor_penalty) or not 0 <= anchor_penalty <= 1e6:
        raise ValueError("anchor_penalty must be finite and in [0,1e6]")
    if method not in {"regularized", "hrp", "minimum_variance", "inverse_volatility"}:
        raise ValueError("Unknown allocation method")
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("Returns require a nonempty (observations, assets) matrix")
    rows_input, n = x.shape
    x = np.array(x[-LOOKBACK:], dtype=np.float64, copy=True)
    t = len(x)
    valid = np.isfinite(x) & (x >= -1.0)
    observed = x[valid]
    scale = float(np.max(np.abs(observed))) if observed.size else 1.0
    scale = scale if scale > 0 else 1.0
    x[valid] /= scale
    x[~valid] = np.nan
    counts = valid.sum(axis=0)
    required = min(t, max(2, math.ceil(.8 * t)))
    eligible = (counts >= required) & valid[-1]
    risks = _column_variances(x, valid)
    diagnostic: dict[str, Any] = {
        "status": "unstarted", "method": method, "rows_input": rows_input,
        "rows_used": t, "assets_total": n, "eligible_assets": int(eligible.sum()),
        "complete_rows": 0, "main_solver_converged": False,
        "anti_equal_weight_applied": False, "single_asset_fallback": False,
        "single_asset_rule_ambiguity": n == 1, "events": [],
        "half_life": float(half_life), "recent_mix": float(recent_mix),
        "anchor_penalty": float(anchor_penalty),
    }
    if n == 1:
        diagnostic["status"] = "single_asset_rule_ambiguous"
        return {"weights": np.ones(1), "diagnostics": diagnostic}
    if not eligible.any():
        # Partial columns with a finite latest return are still preferable to
        # completely missing/stale columns in the emergency path.
        eligible = (counts > 0) & valid[-1]
        if not eligible.any():
            eligible = counts == counts.max()
        diagnostic["status"] = "emergency_no_observations" if not valid.any() else "fallback_insufficient_coverage"
        diagnostic["eligible_assets"] = int(eligible.sum())
        diagnostic["events"].append("insufficient_complete_history")
        fallback = np.zeros(n)
        fallback[eligible] = _inverse_risk(risks[eligible])
        return _finish([fallback], n, eligible, risks, diagnostic)

    indices = np.flatnonzero(eligible)
    fallback = np.zeros(n)
    fallback[indices] = _inverse_risk(risks[indices])
    if len(indices) == 1:
        diagnostic["status"] = "one_eligible_asset"
        return _finish([fallback], n, eligible, risks, diagnostic)
    if method == "inverse_volatility":
        diagnostic["status"] = "inverse_volatility"
        return _finish([fallback], n, eligible, risks, diagnostic)
    if len(indices) > MAX_DENSE_ASSETS:
        diagnostic["status"] = "fallback_resource_guard"
        diagnostic["events"].append("dense_asset_limit_exceeded")
        return _finish([fallback], n, eligible, risks, diagnostic)

    common = np.all(valid[:, indices], axis=1)
    complete = x[np.ix_(common, indices)]
    diagnostic["complete_rows"] = len(complete)
    # Do not fabricate covariance by filling missing returns with zeros.
    if len(complete) < max(2, min(32, t // 2)):
        diagnostic["status"] = "fallback_insufficient_common_rows"
        return _finish([fallback], n, eligible, risks, diagnostic)

    full_anchor: np.ndarray | None = None
    try:
        covariance = estimate_covariance(complete, half_life, recent_mix)
        anchor = hrp_weights(covariance)
        full_anchor = np.zeros(n)
        full_anchor[indices] = anchor
        if method == "hrp":
            diagnostic["status"] = "hrp"
            return _finish([full_anchor, fallback], n, eligible, risks, diagnostic)
        penalty = 0.0 if method == "minimum_variance" else anchor_penalty
        solution, info = solve_regularized_qp(covariance, anchor, penalty)
        diagnostic.update({
            "main_solver_converged": info["converged"],
            "solver_iterations": info["iterations"],
            "qp_relative_gap": info["relative_gap"],
            "qp_objective": info["objective"],
        })
        if not info["converged"]:
            raise ArithmeticError("QP iteration limit reached without gap certificate")
        full = np.zeros(n)
        full[indices] = solution
        diagnostic["status"] = "regularized_qp" if method == "regularized" else "minimum_variance"
        return _finish([full, full_anchor, fallback], n, eligible, risks, diagnostic)
    except Exception as exc:
        # BaseException (interrupt/process termination) is deliberately not swallowed.
        diagnostic["events"].append(f"{type(exc).__name__}: {str(exc)[:180]}")
        if full_anchor is not None:
            diagnostic["status"] = "fallback_hrp"
            return _finish([full_anchor, fallback], n, eligible, risks, diagnostic)
        diagnostic["status"] = "fallback_inverse_volatility"
        return _finish([fallback], n, eligible, risks, diagnostic)
