"""The round-1 prohibition, enforced as a total invariant.

Round 1 forbids the equally weighted portfolio, and it forbids it *absolutely*:
it is not "avoid it when the optimizer succeeds", it is "never emit 1/n".
That makes it a property of the whole public entry point, not of the happy
path, so this file brute-forces the space of inputs that could plausibly drag
an implementation back to 1/n and asserts the property on every one of them:

* n = 1..12 at several window lengths;
* every-degenerate column layouts (all-zero, all-constant, duplicated,
  exactly collinear, one live column among dead ones, alternating blocks);
* NaN patterns that leave one, two, or no usable columns;
* explosive-but-finite magnitudes and infinities;
* the same matrix submitted twice (no hidden state) and with a permuted
  column order (the prohibition must not depend on column position).

The last-resort rule is also pinned: when the entire candidate chain is
unusable the file must still spread risk over the investable instruments
instead of disappearing into a single name.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/portfolio_game/core.py"


def core():
    spec = importlib.util.spec_from_file_location("core_equal_wt", CORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assert_legal(w, n):
    assert isinstance(w, np.ndarray) and w.shape == (n,)
    assert np.isfinite(w).all(), "weights must be finite"
    assert (w >= -1e-12).all(), "long-only violated"
    assert abs(float(w.sum()) - 1.0) < 1e-12, "must be fully invested"
    if n > 1:
        assert not np.allclose(w, np.full(n, 1.0 / n), atol=1e-10, rtol=0.0), (
            f"equal weight 1/{n} is prohibited in round 1"
        )


def live(t=300, n=3, seed=0):
    """A plain, well-behaved panel."""
    rng = np.random.default_rng(seed)
    x = rng.normal(5e-4, 1e-2, (t, 1)) * np.linspace(0.5, 1.4, n)
    return x + rng.normal(size=(t, n)) * 6e-3


def hostile_panels(t=300):
    """Panels engineered so that *every* risk-based candidate in the chain is
    either invalid or exactly 1/n."""
    x = live(t=t)
    yield "all_zero", np.zeros((t, 4))
    yield "all_constant", np.full((t, 5), 1e-4)
    yield "identical_columns", np.tile(x[:, :1], (1, 4))
    yield "scaled_copies", np.column_stack([x[:, 0], 2 * x[:, 0], 1e3 * x[:, 0]])
    yield "sign_flipped_copies", np.column_stack([x[:, 0], -x[:, 0], x[:, 0]])
    yield "collinear_span", x[:, :2] @ np.array([[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 0.0]])
    yield "one_live_rest_constant", np.column_stack([x[:, 0], np.full((t, 3), 2e-4)])
    yield "one_live_rest_nan", np.column_stack([x[:, 0], np.full((t, 3), np.nan)])
    yield "alternating_blocks", np.tile(np.repeat(x[: t // 2, :1], 2, axis=1), (2, 1))
    yield "zero_plus_live", np.column_stack([np.zeros(t), x[:, 0], x[:, 1]])
    yield "tiny_scale", np.column_stack([x[:, 0] * 1e-300, np.zeros(t), 1e-300 * x[:, 1]])
    yield "huge_scale", np.column_stack([x[:, 0] * 1e150, x[:, 0] * 1e150, x[:, 1] * 1e150])
    yield "one_column_split_by_nan", np.where(
        np.arange(t)[:, None] < t // 2, x[:, :2], np.nan
    )
    yield "single_finite_row", np.where(np.arange(t)[:, None] == t - 1, x[:, :3], np.nan)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 12])
def test_no_degenerate_layout_ever_yields_equal_weight(n):
    """The prohibition is total: no column layout may drag us back to 1/n."""
    mod = core()
    x = live(t=300, n=max(n, 1))
    panels = [x[:, :n]] if n <= x.shape[1] else [np.tile(x[:, :1], (1, n))]
    panels = list(panels)
    if n > 1:
        base = x[:, :1]
        panels += [
            np.tile(base, (1, n)),
            np.zeros((300, n)),
            np.full((300, n), 3e-4),
            np.column_stack([base, np.zeros((300, n - 1))]),
            np.column_stack([base, np.full((300, n - 1), np.nan)]),
        ]
    for panel in panels:
        for t in (1, 2, 5, 20, 252, 300):
            block = panel[:t] if t <= panel.shape[0] else panel
            if block.shape[0] < 1:
                continue
            w = mod.allocate(block)["weights"]
            assert_legal(w, n)


def test_hostile_panels_never_yield_equal_weight():
    mod = core()
    for label, panel in hostile_panels():
        n = panel.shape[1]
        w = mod.allocate(panel)["weights"]
        assert_legal(w, n)
        # and again through the grader's own entry point, on the last 252 rows
        w2 = mod.allocate(panel[-252:])["weights"]
        assert_legal(w2, n)


def test_last_resort_spreads_over_investable_names_instead_of_one_name():
    """When the whole candidate chain is unusable we must not go all-in on one
    instrument, and we must not solve the problem with an epsilon nudge."""
    mod = core()
    panel = np.zeros((300, 6))
    result = mod.allocate(panel)
    w = result["weights"]
    d = result["diagnostics"]
    assert_legal(w, 6)
    assert d["last_resort_fallback"] is True
    assert d["last_resort_names"] == 6
    assert d["single_asset_fallback"] is False
    assert (w > 0).sum() == 6
    assert not np.allclose(w, np.full(6, 1 / 6.0), atol=1e-3), "not an epsilon nudge"
    # strictly decreasing with the risk rank (all risks tie -> column order)
    assert np.all(np.diff(w) < 0)


def test_last_resort_collapses_to_one_name_only_when_one_name_is_investable():
    """With a single usable column there is nothing to spread, so a fully
    invested long-only book is necessarily 100% in that column -- and that is
    reached by the eligibility rule, not by the last-resort budget."""
    mod = core()
    panel = np.column_stack([np.zeros(300), np.full((300, 3), np.nan)])
    result = mod.allocate(panel)
    np.testing.assert_array_equal(result["weights"], [1.0, 0.0, 0.0, 0.0])
    assert result["diagnostics"]["status"] == "one_eligible_asset"
    assert result["diagnostics"]["last_resort_fallback"] is False


def test_prohibition_is_independent_of_column_order_and_repeatable():
    mod = core()
    panel = live(t=300, n=4)
    first = mod.allocate(panel)["weights"]
    assert_legal(first, 4)
    np.testing.assert_array_equal(first, mod.allocate(panel)["weights"])
    perm = np.array([2, 0, 3, 1])
    shuffled = mod.allocate(panel[:, perm])["weights"]
    assert_legal(shuffled, 4)
    # weights on the permuted panel are the original weights resampled by perm
    np.testing.assert_allclose(shuffled, first[perm], atol=1e-12)


def test_exact_risk_ties_may_break_on_column_order_but_must_stay_legal():
    """Exact ties have no risk-based tie-break, so the documented rule is that
    the *order* may differ; the prohibition and the contract may not."""
    mod = core()
    panel = np.tile(live(t=300, n=1), (1, 4))
    first = mod.allocate(panel)["weights"]
    assert_legal(first, 4)
    np.testing.assert_array_equal(first, mod.allocate(panel)["weights"])
    perm = np.array([2, 0, 3, 1])
    assert_legal(mod.allocate(panel[:, perm])["weights"], 4)


def test_infinity_and_nan_sentinels_still_land_on_a_legal_book():
    mod = core()
    panel = live(t=300, n=4)
    panel[0, 0] = np.inf
    panel[5, 1] = -np.inf
    panel[10:, 2] = np.nan
    panel[20, 3] = 1e200
    assert_legal(mod.allocate(panel)["weights"], 4)
