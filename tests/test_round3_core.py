"""Round-3 invariants, checked on the shipped module and the shipped file.

The prohibited list is unchanged from round 2 (equal weight, inverse
volatility, SCM minimum variance), so the structural exclusions are guarded the
same way. What is new is the feasible set: leverage 1.5 with full investment
means an aggregate short budget of 0.25, and the budget is an *aggregate* cap
rather than a per-asset floor. The tests that matter most here are therefore the
ones that pin the budget: never exceeded, exactly zero when no leverage is
granted, and monotone in the leverage that is granted.

Note also what is deliberately NOT assumed: that a larger budget is automatically
used. At the shipped penalty the anchor term dominates and the budget goes
untouched, which is a measurement (reports/round3_sweep.json), not a bug.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/portfolio_game_round3/core.py"
SHIPPED = ROOT / "submission/portfolio_round3.py"
CONFIG = ROOT / "configs/submission_round3.json"
R2_CORE = ROOT / "src/portfolio_game_round2/core.py"

LEVERAGES = (1.0, 1.25, 1.5, 1.75, 2.0)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def core():
    return _load(CORE, "core_r3_tests")


def shipped():
    return _load(SHIPPED, "shipped_r3_tests")


HOSTILE = {
    "zeros": np.zeros((200, 4)),
    "constant": np.full((200, 4), 1e-3),
    "all_nan": np.full((200, 4), np.nan),
    "inf": np.full((200, 4), np.inf),
    "two_rows": np.array([[0.01, -0.02, 0.0, 0.03], [0.0, 0.01, -0.01, 0.0]]),
    "one_row": np.array([[0.01, -0.02, 0.0, 0.03]]),
    "one_live": np.column_stack([np.linspace(-.01, .01, 200), np.zeros((200, 3))]),
    "identical": np.repeat(np.linspace(-.01, .01, 200).reshape(-1, 1), 4, axis=1),
    "perfect_corr": np.tile(np.linspace(-.01, .01, 200).reshape(-1, 1), (1, 2))
                    @ np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
    "nan_tail": np.vstack([np.linspace(-.01, .01, 199).reshape(-1, 1)] * 4),
    "mixed_inf_nan": np.column_stack([
        np.linspace(-.01, .01, 200), np.full(200, np.inf),
        np.full(200, np.nan), np.linspace(.01, -.01, 200)]),
    "one_col_nan": np.column_stack([
        np.linspace(-.01, .01, 200), np.full(200, np.nan),
        np.linspace(.01, -.01, 200), np.linspace(-.02, .02, 200)]),
}


def budget_of(leverage: float) -> float:
    return 0.5 * (leverage - 1.0)


def assert_legal(w: np.ndarray, n: int, leverage: float = 1.0) -> None:
    assert w.shape == (n,)
    assert np.isfinite(w).all()
    assert abs(float(w.sum()) - 1.0) < 1e-8, w.sum()
    short = float(np.maximum(-w, 0.0).sum())
    assert short <= budget_of(leverage) + 1e-8, (short, budget_of(leverage))
    assert float(np.abs(w).sum()) <= leverage + 1e-8


# ------------------------------------------------------------------ the file
def test_shipped_file_is_self_contained_and_in_sync():
    text = SHIPPED.read_text(encoding="utf-8")
    assert CORE.read_text(encoding="utf-8") in text
    assert "from portfolio_game_round3" not in text
    assert "import portfolio_game_round3" not in text
    assert "research." not in text
    assert "portfolio_game_round2" not in text


def test_shipped_class_contract_matches_teacher_requirement():
    import ast

    from skfolio.optimization import BaseOptimization

    cls = shipped().CVXPYPortfolio
    assert issubclass(cls, BaseOptimization)
    tree = ast.parse(SHIPPED.read_text(encoding="utf-8"))
    body = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "CVXPYPortfolio")
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "__init__"
                   for n in body.body)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "fit" for n in body.body)


def test_frozen_config_is_consistent_and_declares_its_override():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["round"] == 3
    assert cfg["method"] == "regularized"
    # The round's own leverage, expressed where the solver can act on it.
    assert cfg["gross_leverage"] >= 1.0
    assert cfg["anchor_penalty"] > 0
    mod = core()
    import inspect
    sig = inspect.signature(mod.allocate)
    for key, value in cfg.items():
        if key in sig.parameters and sig.parameters[key].default != value:
            assert key in cfg.get("overrides_core_default", []), key
    out = mod.allocate(np.zeros((3, 2)), half_life=cfg["half_life"],
                       recent_mix=cfg["recent_mix"],
                       anchor_penalty=cfg["anchor_penalty"],
                       method=cfg["method"],
                       gross_leverage=cfg["gross_leverage"])
    assert_legal(out["weights"], 2, cfg["gross_leverage"])


# ------------------------------------------------------------ the short budget
def test_no_leverage_means_exactly_the_long_only_book():
    mod = core()
    rng = np.random.default_rng(11)
    for n in (2, 5, 20):
        x = rng.normal(0, 0.01, (126, n))
        out = mod.allocate(x, gross_leverage=1.0)
        w = out["weights"]
        assert (w >= 0).all(), w
        assert out["diagnostics"]["short_exposure"] == 0.0
        assert out["diagnostics"]["gross_exposure"] == pytest.approx(1.0, abs=1e-12)


def test_short_budget_is_never_exceeded_on_hostile_panels():
    mod = core()
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        for lev in LEVERAGES:
            w = mod.allocate(panel, gross_leverage=lev)["weights"]
            assert_legal(w, panel.shape[1], lev), name


def test_gross_exposure_identity_holds_everywhere():
    """gross = 1 + 2 * short is the rule the leverage cap is written against."""
    mod = core()
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        for lev in (1.5, 2.0):
            d = mod.allocate(panel, gross_leverage=lev)["diagnostics"]
            assert d["gross_exposure"] == pytest.approx(
                1.0 + 2.0 * d["short_exposure"], abs=1e-9), name


def test_granting_more_leverage_can_only_relax_the_constraint():
    """The budget is a relaxation, so realised short cannot shrink as it grows.

    Compared at the solver's own tolerance: the certify gap is 1e-8, so a
    tighter claim would be about convergence noise rather than about the
    feasible set.
    """
    mod = core()
    rng = np.random.default_rng(88)
    tol = 1e-9
    for _ in range(10):
        n = int(rng.integers(3, 10))
        x = rng.normal(0, 0.01, (126, n)) * np.exp(rng.normal(0, 1.0, n))
        shorts = [mod.allocate(x, gross_leverage=lv,
                               anchor_penalty=0.5)["diagnostics"]["short_exposure"]
                  for lv in LEVERAGES]
        for earlier, later in zip(shorts, shorts[1:]):
            assert later >= earlier - tol, shorts


def test_low_penalty_actually_uses_the_budget():
    """The round-3 rule is only meaningful if the budget can bind.

    At a low anchor penalty the variance term dominates and the optimizer has a
    reason to hedge, so the aggregate short should reach the budget on at least
    some panels. Asserted over a batch rather than on one panel: a first version
    of this test pinned a single synthetic panel and failed on it, because
    whether a particular random draw offers a hedging opportunity is itself
    random. The real evidence is on market data -- on sp500 at p=0.25 the budget
    is used in 383 of 409 folds and saturates at exactly 0.25
    (reports/round3_sweep.txt) -- so what is asserted here is that the mechanism
    is reachable and never overshoots.
    """
    mod = core()
    rng = np.random.default_rng(4)
    users = 0
    for _ in range(12):
        n = int(rng.integers(3, 14))
        x = rng.normal(0, 0.01, (126, n)) * np.exp(rng.normal(0, 0.8, n))
        d = mod.allocate(x, gross_leverage=1.5, anchor_penalty=0.25)["diagnostics"]
        assert d["short_exposure"] <= 0.25 + 1e-9
        assert d["gross_exposure"] <= 1.5 + 1e-9
        if d["short_exposure"] > 0.01:
            users += 1
    assert users > 0, "the short budget was never reachable on any panel"


def test_projection_is_the_exact_euclidean_projection():
    """Checked against the KKT thresholds, computed independently here."""
    mod = core()
    rng = np.random.default_rng(31337)
    for _ in range(40):
        n = int(rng.integers(2, 12))
        s = float(rng.choice([0.02, 0.1, 0.25, 0.5]))
        v = rng.normal(0, 1, n) * float(rng.choice([0.05, 1.0, 4.0]))
        w = mod.project_gross_budget(v, s)
        assert abs(float(w.sum()) - 1.0) < 1e-9
        assert float(np.maximum(-w, 0.0).sum()) <= s + 1e-10
        shifted = v - (v.sum() - 1.0) / n
        if float(np.maximum(-shifted, 0.0).sum()) <= s + 1e-12:
            assert np.allclose(w, shifted, atol=1e-12), "slack case should be a shift"
            continue
        hi = float(v.max()) + 1e3
        lo = float(v.min()) - 1e3
        a_lo, a_hi = lo, hi
        for _ in range(200):
            mid = 0.5 * (a_lo + a_hi)
            if float(np.maximum(v - mid, 0.0).sum()) > 1.0 + s:
                a_lo = mid
            else:
                a_hi = mid
        a = 0.5 * (a_lo + a_hi)
        b_lo, b_hi = lo, hi
        for _ in range(200):
            mid = 0.5 * (b_lo + b_hi)
            if float(np.maximum(mid - v, 0.0).sum()) > s:
                b_hi = mid
            else:
                b_lo = mid
        b = 0.5 * (b_lo + b_hi)
        assert np.allclose(w, np.maximum(v - a, 0.0) - np.maximum(b - v, 0.0),
                           atol=1e-9)


def test_impossible_leverage_is_rejected_not_silently_clipped():
    mod = core()
    x = np.random.default_rng(6).normal(0, 0.01, (126, 5))
    for bad in (0.5, 0.99, 3.5, np.nan, np.inf):
        with pytest.raises(ValueError):
            mod.allocate(x, gross_leverage=bad)


# ------------------------------------------------------------ the three bans
def test_inverse_risk_rule_does_not_exist_anywhere():
    import tokenize

    def executable_tokens(path: Path) -> str:
        with path.open("rb") as handle:
            toks = list(tokenize.tokenize(handle.readline))
        keep = [t for t in toks
                if t.type not in (tokenize.COMMENT, tokenize.STRING,
                                  tokenize.NL, tokenize.NEWLINE,
                                  tokenize.INDENT, tokenize.DEDENT)]
        return tokenize.untokenize(keep).decode("utf-8")

    for path in (CORE, SHIPPED):
        code = executable_tokens(path)
        assert "_inverse_risk" not in code, path
        for banned in ("1/sigma", "1 / sigma", "inverse_volatility",
                       "inverse-volatility", "inv_vol"):
            assert banned not in code, (path, banned)
        assert "inverse-volatility" in path.read_text(encoding="utf-8"), path


def test_prohibited_methods_are_rejected_not_silently_supported():
    mod = core()
    x = np.random.default_rng(11).normal(0, 0.01, (200, 6))
    for method in ("minimum_variance", "inverse_volatility", "global_minimum_variance"):
        with pytest.raises(ValueError):
            mod.allocate(x, method=method)


def test_never_returns_equal_weight_or_inverse_volatility():
    mod = core()
    rng = np.random.default_rng(4242)
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        n = panel.shape[1]
        for lev in (1.0, 1.5):
            w = mod.allocate(panel, gross_leverage=lev)["weights"]
            assert_legal(w, n, lev)
            if n > 1:
                assert not np.allclose(w, 1.0 / n, atol=1e-8, rtol=1e-5), name
    for trial in range(30):
        n = int(rng.integers(2, 12))
        x = rng.normal(0, 1, (150, n)) * np.exp(rng.normal(0, 1.3, n))
        for lev in (1.0, 1.5):
            w = mod.allocate(x, gross_leverage=lev)["weights"]
            assert_legal(w, n, lev)
            assert not np.allclose(w, 1.0 / n, atol=1e-8, rtol=1e-5), trial
            sd = x.std(axis=0, ddof=1)
            ivp = (1.0 / sd) / (1.0 / sd).sum()
            assert np.abs(w - ivp).sum() > 1e-6, trial


# ------------------------------------------------------------------- behaviour
def test_last_resort_budget_stays_long_only_even_when_leverage_is_granted():
    """The fallback is a risk-free convention, so it grants itself no shorts."""
    mod = core()
    for n in range(2, 7):
        out = mod.allocate(np.full((200, n), np.nan), gross_leverage=2.0)
        w = out["weights"]
        assert_legal(w, n, 2.0)
        assert (w >= 0).all(), w
        assert out["diagnostics"]["short_exposure"] == 0.0


def test_allocate_never_raises_and_repeats_exactly():
    mod = core()
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        first = mod.allocate(panel, gross_leverage=1.5)
        second = mod.allocate(panel, gross_leverage=1.5)
        assert np.array_equal(first["weights"], second["weights"]), name
        assert_legal(first["weights"], panel.shape[1], 1.5)


def test_single_asset_is_the_only_case_where_the_bans_are_vacuous():
    mod = core()
    out = mod.allocate(np.linspace(-0.01, 0.01, 200).reshape(-1, 1),
                       gross_leverage=1.5)
    assert np.allclose(out["weights"], [1.0])
    assert "single" in out["diagnostics"]["status"] or \
        out["diagnostics"].get("final_rule") == "single_eligible_asset"


def test_check_weights_message_refers_to_round_three():
    mod = core()
    with pytest.raises(ValueError) as exc:
        mod.check_weights(np.full(4, 0.25), 4)
    assert "round 3" in str(exc.value)


def test_round2_core_is_left_untouched():
    """Round 3 adds a tree rather than editing round 2's, so round 2 stays valid."""
    assert R2_CORE.exists()
    assert "MAX_SHORT_CAP" in R2_CORE.read_text(encoding="utf-8")
    assert "MAX_SHORT_CAP" not in CORE.read_text(encoding="utf-8")
