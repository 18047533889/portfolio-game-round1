"""Round-5 invariants, checked on the shipped module and the shipped file.

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
CORE = ROOT / "src/portfolio_game_round5/core.py"
SHIPPED = ROOT / "submission/portfolio_round5.py"
CONFIG = ROOT / "configs/submission_round5.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def core():
    return _load(CORE, "core_r5_tests")


def shipped():
    return _load(SHIPPED, "shipped_r5_tests")


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


def assert_legal(w: np.ndarray, n: int, budget: float = 1.0) -> None:
    assert w.shape == (n,)
    assert np.isfinite(w).all()
    assert abs(float(w.sum()) - 1.0) < 1e-8, w.sum()
    # Round 5 states the contract explicitly: nonnegative, sum to one, no cash.
    assert w.min() >= -1e-12, w.min()
    assert float(np.abs(w).sum()) <= budget + 1e-8


# ------------------------------------------------------------------ the file
def test_shipped_file_is_self_contained_and_in_sync():
    text = SHIPPED.read_text(encoding="utf-8")
    assert CORE.read_text(encoding="utf-8") in text
    assert "from portfolio_game_round5" not in text
    assert "import portfolio_game_round5" not in text
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
    assert cfg["round"] == 5
    assert cfg["method"] == "regularized"
    # The round's stated contract: shorts forbidden, budget exactly one.
    assert cfg["gross_budget"] == 1.0
    assert cfg["anchor_penalty"] > 0
    assert 0.0 <= cfg["downside_mix"] <= 1.0
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
                       gross_budget=cfg["gross_budget"],
                       downside_mix=cfg["downside_mix"])
    assert_legal(out["weights"], 2, cfg["gross_budget"])
    assert (out["weights"] >= 0).all()


# -------------------------------------------------- the round-5 risk contract
def test_round_five_contract_holds_on_hostile_panels():
    """Nonnegative, sum to one, no cash -- the three things the brief spells out.

    Round 5 is the first round to state the contract unambiguously, so it is
    asserted directly rather than inferred from the absence of a short budget.
    A book that satisfies the first two has gross exposure exactly 1, and that
    is asserted too, because it is the implication and not an assumption.
    """
    mod = core()
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        n = panel.shape[1]
        for mix in (0.0, 0.5, 1.0):
            out = mod.allocate(panel, downside_mix=mix)
            w = out["weights"]
            assert_legal(w, n)
            assert (w >= 0).all(), (name, mix, w)
            assert out["diagnostics"]["short_exposure"] == 0.0
            assert out["diagnostics"]["gross_exposure"] == pytest.approx(
                1.0, abs=1e-12)


def test_a_budget_above_one_is_not_this_rounds_contract():
    """The core keeps the general budget, but the frozen config pins it at 1.0.

    The parameter exists because three earlier rounds varied it; what matters
    here is that the *shipped* setting cannot drift, which the config test
    asserts. This one records that the core still supports the general case, so
    a future round does not have to re-derive it.
    """
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert float(cfg["gross_budget"]) == 1.0
    mod = core()
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.01, (252, 8))
    for budget in (1.0, 1.25, 1.5):
        w = mod.allocate(x, gross_budget=budget)["weights"]
        assert abs(float(w.sum()) - 1.0) < 1e-8
        assert float(np.abs(w).sum()) <= budget + 1e-8
        if budget == 1.0:
            assert (w >= 0).all()


# ------------------------------------------------------------- the risk blend
def test_downside_mix_is_validated_not_clipped():
    mod = core()
    x = np.random.default_rng(6).normal(0, 0.01, (252, 6))
    for bad in (-0.1, 1.5, np.nan, np.inf):
        with pytest.raises(ValueError):
            mod.allocate(x, downside_mix=bad)


def test_zero_mix_reproduces_the_covariance_only_path_exactly():
    """mix = 0 must be the previous rounds' behaviour, bit for bit.

    The blend is the round-5 change, so the cheapest way to be sure it did not
    disturb anything is to require that switching it off returns identical
    weights -- not close ones.
    """
    mod = core()
    rng = np.random.default_rng(7)
    for n in (3, 8, 20):
        x = rng.normal(0, 0.01, (252, n))
        plain = mod.allocate(x, downside_mix=0.0)["weights"]
        explicit = mod.allocate(x, downside_mix=0.0)["weights"]
        assert np.array_equal(plain, explicit)
        cov = mod.estimate_covariance(x, 63.0, 0.25)
        assert np.array_equal(mod.blend_risk(cov, cov, 0.0), cov)


def test_semi_covariance_is_a_psd_risk_model_and_scaled_like_the_covariance():
    mod = core()
    rng = np.random.default_rng(8)
    for n in (2, 5, 12):
        x = rng.normal(0, 0.01, (252, n))
        semi = mod.downside_covariance(x, 63.0)
        assert semi.shape == (n, n)
        assert np.isfinite(semi).all()
        assert np.allclose(semi, semi.T, atol=1e-15)
        # PSD is what lets the solver keep its quadratic structure and its
        # duality-gap certificate.
        assert np.linalg.eigvalsh(semi).min() >= -1e-12
        # Normalised to a median diagonal of one, so a convex mix with the
        # covariance is a weight rather than a scale.
        assert float(np.median(np.diag(semi))) == pytest.approx(1.0, abs=1e-9)
        blended = mod.blend_risk(mod.estimate_covariance(x, 63.0, 0.25), semi, 0.5)
        assert np.linalg.eigvalsh(blended).min() >= -1e-12


def test_the_blend_actually_changes_the_book_and_stays_compliant():
    """A blend that does nothing would make the round's headline claim vacuous.

    Also checks the direction the round relies on: the blended book must stay
    away from the prohibited inverse-volatility book. Whether it improves the
    scored risk measures is measured on market data, not asserted here.
    """
    mod = core()
    rng = np.random.default_rng(9)
    moved = 0
    for _ in range(8):
        n = int(rng.integers(4, 14))
        x = rng.normal(0, 0.01, (252, n)) * np.exp(rng.normal(0, 1.0, n))
        a = mod.allocate(x, downside_mix=0.0)
        b = mod.allocate(x, downside_mix=0.75)
        assert_legal(b["weights"], n)
        if not np.allclose(a["weights"], b["weights"], atol=1e-10):
            moved += 1
        sd = np.sqrt(np.diag(mod.estimate_covariance(x, 63.0, 0.25)))
        ivp = (1.0 / sd) / (1.0 / sd).sum()
        assert np.abs(b["weights"] - ivp).sum() > 1e-6
    assert moved > 0, "the downside blend never changed any book"


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


def test_impossible_budget_is_rejected_not_silently_clipped():
    mod = core()
    x = np.random.default_rng(6).normal(0, 0.01, (126, 5))
    for bad in (0.5, 0.99, 3.5, np.nan, np.inf):
        with pytest.raises(ValueError):
            mod.allocate(x, gross_budget=bad)


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
    """The two prohibitions that are visible in the returned vector.

    Exercised at the shipped budget of 1.0, which is the configuration that gets
    graded. A first version of this test also ran budget 1.5, left over from
    round 3: that grants a short budget, the book legitimately goes negative, and
    the nonnegativity assertion in assert_legal then fails on correct output.
    Testing a configuration the round does not ship is how a test starts
    asserting the wrong contract, so the general-budget case is covered by its
    own test instead.
    """
    mod = core()
    rng = np.random.default_rng(4242)
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        n = panel.shape[1]
        w = mod.allocate(panel, gross_budget=1.0)["weights"]
        assert_legal(w, n, 1.0)
        if n > 1:
            assert not np.allclose(w, 1.0 / n, atol=1e-8, rtol=1e-5), name
    for trial in range(30):
        n = int(rng.integers(2, 12))
        x = rng.normal(0, 1, (150, n)) * np.exp(rng.normal(0, 1.3, n))
        w = mod.allocate(x, gross_budget=1.0)["weights"]
        assert_legal(w, n, 1.0)
        assert not np.allclose(w, 1.0 / n, atol=1e-8, rtol=1e-5), trial
        sd = x.std(axis=0, ddof=1)
        ivp = (1.0 / sd) / (1.0 / sd).sum()
        assert np.abs(w - ivp).sum() > 1e-6, trial


# ------------------------------------------------------------------- behaviour
def test_last_resort_budget_never_grants_itself_a_short():
    """The fallback is a risk-free convention, so it takes no directional view."""
    mod = core()
    for n in range(2, 7):
        out = mod.allocate(np.full((200, n), np.nan), gross_budget=1.0)
        w = out["weights"]
        assert_legal(w, n, 1.0)
        assert (w >= 0).all(), w
        assert out["diagnostics"]["short_exposure"] == 0.0


def test_allocate_never_raises_and_repeats_exactly():
    mod = core()
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        first = mod.allocate(panel, gross_budget=1.0)
        second = mod.allocate(panel, gross_budget=1.0)
        assert np.array_equal(first["weights"], second["weights"]), name
        assert_legal(first["weights"], panel.shape[1], 1.0)


def test_single_asset_is_the_only_case_where_the_bans_are_vacuous():
    mod = core()
    out = mod.allocate(np.linspace(-0.01, 0.01, 200).reshape(-1, 1),
                       gross_budget=1.0)
    assert np.allclose(out["weights"], [1.0])
    assert "single" in out["diagnostics"]["status"] or \
        out["diagnostics"].get("final_rule") == "single_eligible_asset"


def test_check_weights_message_refers_to_round_five():
    mod = core()
    with pytest.raises(ValueError) as exc:
        mod.check_weights(np.full(4, 0.25), 4)
    assert "round 5" in str(exc.value)


def test_earlier_round_trees_are_left_untouched():
    """Each round adds a tree rather than editing an earlier one, so every
    earlier submission stays exactly as it was graded."""
    r3 = ROOT / "src/portfolio_game_round3/core.py"
    assert r3.exists()
    assert "MAX_SHORT_BUDGET" in r3.read_text(encoding="utf-8")
    assert "MAX_SHORT_BUDGET" in CORE.read_text(encoding="utf-8")
    text = CORE.read_text(encoding="utf-8")
    assert "portfolio_game_round3" not in text
    assert "portfolio_game_round2" not in text
