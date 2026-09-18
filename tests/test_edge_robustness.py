"""Low-dimension and RMT-cleaning regression tests.

Two things are pinned here, both learned from a real incident:

1. skfolio's own hierarchical family (HierarchicalRiskParity,
   HierarchicalEqualRiskContribution, SchurComplementary,
   NestedClustersOptimization) cannot produce weights for n_assets <= 2: its
   PearsonDistance returns a scalar at n_assets == 1 and the condensed linkage
   is degenerate at n_assets == 2, so skfolio sets weights_ = None and the
   grader records a failed dataset. Our file must never depend on that path, so
   the low-dimension contract is asserted here directly.
2. The Marchenko-Pastur cleaning step must leave a valid correlation matrix:
   unit diagonal, symmetric, positive definite, and a no-op when the sample
   cannot separate signal from noise.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/portfolio_game/core.py"


def core():
    spec = importlib.util.spec_from_file_location("core_under_test", CORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def valid(w, n):
    assert isinstance(w, np.ndarray) and w.shape == (n,)
    assert np.isfinite(w).all() and (w >= 0).all()
    assert abs(w.sum() - 1.0) < 1e-12
    if n > 1:
        assert not np.allclose(w, np.full(n, 1 / n), atol=1e-10, rtol=1e-8)


def returns(t=252, n=3, seed=0, scale=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0004, 0.01, (t, 1)) * np.linspace(0.4, 1.3, n)
    x = x + rng.normal(size=(t, n)) * 0.006
    if scale is not None:
        x = x * scale
    return x


# --------------------------------------------------------------------------- #
# low dimension: the exact regime where skfolio's own optimizers return None
# --------------------------------------------------------------------------- #
def test_single_asset_returns_full_weight_without_raising():
    r = core().allocate(returns(n=1))
    np.testing.assert_array_equal(r["weights"], [1.0])
    assert r["diagnostics"]["single_asset_rule_ambiguity"] is True


@pytest.mark.parametrize("seed", range(5))
def test_two_assets_always_yield_a_non_equal_valid_portfolio(seed):
    r = core().allocate(returns(n=2, seed=seed))
    valid(r["weights"], 2)


def test_two_identical_columns_still_produce_a_valid_portfolio():
    column = returns(t=300, n=1, seed=3)
    r = core().allocate(np.tile(column, (1, 2)))
    valid(r["weights"], 2)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_low_dimension_short_windows(n):
    for t in (1, 2, 5, 20, 252):
        r = core().allocate(returns(t=t, n=n))
        valid(r["weights"], n)


def test_our_hrp_handles_two_assets_where_skfolio_cannot():
    """Our own tree code must not inherit skfolio's n_assets == 2 defect."""
    mod = core()
    c = mod.estimate_covariance(returns(n=2, seed=11), 63.0, 0.25)
    w = mod.hrp_weights(c)
    valid(w, 2)


def test_documented_skfolio_defect_is_still_real():
    """If skfolio fixes this, the guard above stays harmless; assert the premise."""
    skfolio = pytest.importorskip("skfolio")
    from skfolio.optimization import HierarchicalRiskParity

    rng = np.random.default_rng(2)
    for n in (1, 2):
        x = rng.normal(0, 0.01, (300, n))
        model = HierarchicalRiskParity()
        model.raise_on_failure = False
        with pytest.warns(UserWarning):
            model.fit(x)
        assert model.weights_ is None, (
            f"skfolio now handles n_assets={n}; the low-dimension guard can be revisited"
        )


# --------------------------------------------------------------------------- #
# Marchenko-Pastur cleaning
# --------------------------------------------------------------------------- #
def test_mp_edge_formula_and_monotonicity():
    mod = core()
    # q = T / N, edge = (1 + 1/sqrt(q))^2
    assert mod.marchenko_pastur_edge(252, 252) == pytest.approx(4.0)
    assert mod.marchenko_pastur_edge(1008, 252) == pytest.approx((1 + 0.5) ** 2)
    assert mod.marchenko_pastur_edge(1000, 10) < mod.marchenko_pastur_edge(100, 10)


def test_denoised_correlation_is_a_valid_correlation_matrix():
    mod = core()
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.01, (252, 40))
    raw = np.corrcoef(x, rowvar=False)
    cleaned = mod.rmt_denoise(raw, len(x))
    np.testing.assert_allclose(cleaned, cleaned.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(cleaned), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(cleaned).min() > 0
    assert np.isfinite(cleaned).all()
    assert cleaned.shape == raw.shape


def test_denoising_flattens_the_noise_band_and_preserves_the_trace():
    """Signal eigenvalues survive; the noise band collapses onto one level."""
    mod = core()
    rng = np.random.default_rng(9)
    n_assets, n_obs = 60, 252
    factor = rng.normal(0, 0.01, (n_obs, 1))
    x = factor * np.linspace(0.5, 1.5, n_assets) + rng.normal(0, 0.006, (n_obs, n_assets))
    raw = np.corrcoef(x, rowvar=False)
    cleaned = mod.rmt_denoise(raw, n_obs)

    ev_raw = np.linalg.eigvalsh(raw)
    ev_new = np.linalg.eigvalsh(cleaned)
    edge = mod.marchenko_pastur_edge(n_obs, n_assets)

    band_new, band_raw = ev_new[ev_new < edge], ev_raw[ev_raw < edge]
    assert band_new.size >= 2, "expected a non-empty noise band in this regime"
    # the diagonal rescaling perturbs the spectrum slightly, so the band is not
    # exactly constant; it must nevertheless be drastically flattened
    assert band_new.std() < 0.3 * band_raw.std()
    # the dominant signal direction is kept, the near-singular ones are lifted
    assert ev_new.max() == pytest.approx(ev_raw.max(), rel=1e-2)
    assert ev_new.min() > ev_raw.min()
    assert np.linalg.cond(cleaned) < np.linalg.cond(raw)


def test_denoising_is_the_identity_when_every_direction_is_resolvable():
    mod = core()
    rng = np.random.default_rng(13)
    x = rng.normal(0, 0.01, (4000, 4))
    raw = np.corrcoef(x, rowvar=False)
    np.testing.assert_array_equal(mod.rmt_denoise(raw, len(x)), raw)


def test_denoising_preserves_a_strong_real_factor():
    mod = core()
    rng = np.random.default_rng(21)
    factor = rng.normal(0, 0.01, (600, 1))
    x = factor * np.linspace(0.8, 1.2, 12) + rng.normal(0, 0.002, (600, 12))
    raw = np.corrcoef(x, rowvar=False)
    cleaned = mod.rmt_denoise(raw, len(x))
    assert cleaned[np.triu_indices(12, 1)].mean() > 0.9


@pytest.mark.parametrize("n,t", [(1, 300), (2, 3), (5, 7), (10, 12)])
def test_denoising_is_skipped_when_the_sample_cannot_resolve_signal(n, t):
    mod = core()
    raw = np.eye(n)
    np.testing.assert_array_equal(mod.rmt_denoise(raw, t), raw)


def test_denoising_rejects_non_square_input():
    with pytest.raises(ValueError):
        core().rmt_denoise(np.zeros((3, 4)), 100)


def test_estimate_covariance_stays_positive_definite_after_cleaning():
    mod = core()
    for n in (2, 5, 40, 120):
        c = mod.estimate_covariance(returns(t=252, n=n, seed=n), 63.0, 0.25)
        np.testing.assert_allclose(c, c.T, atol=1e-12)
        assert np.linalg.eigvalsh(c).min() > 0


def test_cleaning_does_not_depend_on_asset_order():
    """Permuting columns must permute the estimate, not change it."""
    mod = core()
    x = returns(t=252, n=15, seed=4)
    perm = np.random.default_rng(1).permutation(15)
    a = mod.estimate_covariance(x, 63.0, 0.25)
    b = mod.estimate_covariance(x[:, perm], 63.0, 0.25)
    np.testing.assert_allclose(b, a[np.ix_(perm, perm)], rtol=1e-9, atol=1e-14)


# --------------------------------------------------------------------------- #
# dense-asset cost guard
# --------------------------------------------------------------------------- #
def test_wide_cross_section_uses_the_covariance_path():
    """The guard bounds cost; it must not swallow the widest standard universe.

    A 1455-column cross-section (the widest standard benchmark) costs ~2.8 s per
    fit on the dense path and 0.01 s on the per-asset fallback, but the fallback
    ignores the correlation structure entirely. On one real 1455-asset
    walk-forward that difference was Sharpe 0.604 -> 1.189 with zero failed
    folds, so the limit is kept above every standard cross-section.
    """
    mod = core()
    assert mod.MAX_DENSE_ASSETS >= 1455, "the dense limit must cover standard universes"
    result = mod.allocate(returns(t=252, n=300, seed=7))
    valid(result["weights"], 300)
    diagnostics = result["diagnostics"]
    assert diagnostics["status"] in {"regularized_qp", "fallback_hrp"} or \
        diagnostics["status"] == "minimum_variance", diagnostics["status"]
    assert "dense_asset_limit_exceeded" not in diagnostics["events"]


def test_dense_asset_guard_still_fires_above_its_limit(monkeypatch):
    """Lowering the limit must still yield a valid, non-equal portfolio."""
    mod = core()
    monkeypatch.setattr(mod, "MAX_DENSE_ASSETS", 8)
    result = mod.allocate(returns(t=252, n=20, seed=8))
    assert result["diagnostics"]["status"] == "fallback_resource_guard"
    assert "dense_asset_limit_exceeded" in result["diagnostics"]["events"]
    valid(result["weights"], 20)
