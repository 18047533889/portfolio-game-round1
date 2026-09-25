"""Round-2 invariants, checked on the shipped module and the shipped file.

Round 1's test suite still guards the round-1 module, which is correct -- that
module is still in the repository and its prohibition still applies to it. This
file guards the round-2 artifact, which is a different artifact with three
prohibitions instead of one and a different numerical core. The distinction
matters because round 1's prohibition (equal weight) is a property of the
returned vector, whereas two of round 2's three prohibitions (inverse
volatility, SCM minimum variance) cannot be detected from a weight vector
alone -- so they are enforced structurally and asserted as such here:

  * EWP      -- property of the output: brute-forced over hostile panels.
  * IVP      -- structural: the inverse-risk rule must not exist at all.
  * SCM-GMVP -- structural + measured: the objective carries an anchor, and the
                proximity to the SCM book is bounded away from zero.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/portfolio_game_round2/core.py"
SHIPPED = ROOT / "submission/portfolio_round2.py"
CONFIG = ROOT / "configs/submission_round2.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def core():
    return _load(CORE, "core_r2_tests")


def shipped():
    return _load(SHIPPED, "shipped_r2_tests")


def scm_gmvp(train: np.ndarray) -> np.ndarray:
    """Prohibited book (c), rebuilt so the distance can be measured."""
    from scipy.optimize import minimize

    x = np.asarray(train, dtype=float)
    s = np.cov(x, rowvar=False, ddof=1)
    n = s.shape[0]
    s = s + np.eye(n) * max(float(np.trace(s)) / n, 1e-18) * 1e-12
    res = minimize(lambda w: float(w @ s @ w), np.full(n, 1.0 / n),
                   jac=lambda w: 2.0 * s @ w, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones_like(w)}],
                   options={"maxiter": 500, "ftol": 1e-14})
    assert res.success
    return np.clip(res.x, 0.0, None) / np.clip(res.x, 0.0, None).sum()


HOSTILE = {
    "zeros": np.zeros((300, 4)),
    "constant": np.full((300, 4), 1e-3),
    "all_nan": np.full((300, 4), np.nan),
    "inf": np.full((300, 4), np.inf),
    "two_rows": np.array([[0.01, -0.02, 0.0, 0.03], [0.0, 0.01, -0.01, 0.0]]),
    "one_row": np.array([[0.01, -0.02, 0.0, 0.03]]),
    "one_live": np.column_stack([np.linspace(-.01, .01, 300), np.zeros((300, 3))]),
    "identical": np.repeat(np.linspace(-.01, .01, 300).reshape(-1, 1), 4, axis=1),
    "perfect_corr": np.tile(np.linspace(-.01, .01, 300).reshape(-1, 1), (1, 2))
                    @ np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
    "nan_tail": np.vstack([np.linspace(-.01, .01, 299).reshape(-1, 1)] * 4),
}


def assert_legal(weights: np.ndarray, n: int, *, short_cap: float = 0.0) -> None:
    assert weights.shape == (n,)
    assert np.isfinite(weights).all()
    assert (weights >= -short_cap - 1e-12).all(), weights.min()
    assert abs(float(weights.sum()) - 1.0) < 1e-8


# ------------------------------------------------------------------ shipped file
def test_shipped_file_is_self_contained_and_in_sync():
    text = SHIPPED.read_text(encoding="utf-8")
    assert CORE.read_text(encoding="utf-8") in text
    assert "from portfolio_game_round2" not in text
    assert "import portfolio_game_round2" not in text
    assert "research." not in text


def test_shipped_class_contract_matches_teacher_requirement():
    import ast

    from skfolio.optimization import BaseOptimization

    mod = shipped()
    cls = mod.CVXPYPortfolio
    assert issubclass(cls, BaseOptimization)
    tree = ast.parse(SHIPPED.read_text(encoding="utf-8"))
    body = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "CVXPYPortfolio")
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "__init__"
                   for n in body.body)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "fit" for n in body.body)


def test_config_matches_module_defaults_and_is_not_the_minimum_variance_limit():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["round"] == 2
    assert cfg["method"] == "regularized"
    # penalty = 0 is the SCM-GMVP limit on low-dimensional panels and was
    # explicitly rejected; the shipped value must not silently drift to it.
    assert cfg["anchor_penalty"] == pytest.approx(4.0)
    assert cfg["anchor_penalty"] > 0
    # The short settings must be frozen in the same way, and the pair must be
    # self-consistent: a budget nobody granted is a configuration error.
    assert isinstance(cfg["allow_short"], bool)
    assert 0.0 <= cfg["short_cap"] <= 0.5
    if cfg["short_cap"] > 0.0:
        assert cfg["allow_short"] is True
    text = SHIPPED.read_text(encoding="utf-8")
    assert f"anchor_penalty={cfg['anchor_penalty']!r}" in text
    assert f"allow_short={cfg['allow_short']!r}" in text
    assert f"short_cap={cfg['short_cap']!r}" in text


# ------------------------------------------------------------ the three bans
def test_inverse_risk_rule_does_not_exist_anywhere():
    """Prohibited book (b) is excluded structurally, not by reweighting.

    Round 1 turned column variances into 1/sigma weights in six places. Any
    reappearance of that helper -- under any name -- would make inverse
    volatility reachable again, so the symbol is asserted absent.

    Only *executable* tokens are inspected. The module docstring deliberately
    names the prohibited books while arguing why none of them is formed, and
    `allocate` compares the method name against the banned strings in order to
    raise; neither is a code path that builds an inverse-volatility book. So
    comments and string literals are stripped before the check, which leaves
    exactly the identifiers, operators and numbers that can actually compute
    something.
    """
    import io
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
        # The documentation must keep saying so, or the exclusion becomes
        # invisible to the next reader.
        assert "inverse-volatility" in path.read_text(encoding="utf-8"), path


def test_prohibited_methods_are_rejected_not_silently_supported():
    mod = core()
    rng = np.random.default_rng(11)
    x = rng.normal(0, 0.01, (300, 6))
    for method in ("minimum_variance", "inverse_volatility", "global_minimum_variance"):
        with pytest.raises(ValueError):
            mod.allocate(x, method=method)


def test_never_returns_equal_weight_on_hostile_input():
    mod = core()
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        n = panel.shape[1]
        for method in ("regularized", "hrp"):
            w = mod.allocate(panel, method=method)["weights"]
            assert_legal(w, n)
            if n > 1:
                assert not np.allclose(w, 1.0 / n, atol=1e-8, rtol=1e-5), \
                    (name, method, w)


def test_never_returns_inverse_volatility_on_random_panels():
    mod = core()
    rng = np.random.default_rng(4242)
    for trial in range(40):
        n = int(rng.integers(2, 12))
        scales = np.exp(rng.normal(0, 1.4, n))
        x = rng.normal(0, 1, (320, n)) * scales
        out = mod.allocate(x)
        w = out["weights"]
        assert_legal(w, n)
        sd = x.std(axis=0, ddof=1)
        ivp = (1.0 / sd) / (1.0 / sd).sum()
        assert np.abs(w - ivp).sum() > 1e-6, (trial, w, ivp)


def test_distance_to_scm_minimum_variance_is_bounded_away_from_zero():
    """Prohibited book (c) is not reachable at the shipped penalty.

    A panel is constructed on which the sample-covariance minimum variance book
    is decisively different from a diversified one, so "did we accidentally ship
    the SCM solution" becomes a measurable question rather than a claim.
    """
    mod = core()
    rng = np.random.default_rng(7)
    n = 6
    scales = np.array([0.004, 0.010, 0.020, 0.035, 0.050, 0.080])
    common = rng.normal(0, 0.01, (320, 1))
    x = common * rng.normal(1.0, 0.05, (1, n)) + rng.normal(0, 1, (320, n)) * scales
    ref = scm_gmvp(x)
    w = mod.allocate(x, anchor_penalty=4.0)["weights"]
    assert_legal(w, n)
    assert np.abs(w - ref).sum() > 0.1, (w, ref)


def test_shipped_penalty_zero_would_be_near_the_scm_book():
    """Why the penalty is not zero -- the excluded setting is measured, not assumed.

    This test documents the finding rather than defending the choice: at
    penalty 0 the objective loses its anchor and the answer collapses onto the
    minimum variance book. It is asserted so that a future edit cannot quietly
    reintroduce penalty 0 while leaving the compliance claim in the docstring.
    """
    mod = core()
    rng = np.random.default_rng(7)
    n = 6
    scales = np.array([0.004, 0.010, 0.020, 0.035, 0.050, 0.080])
    common = rng.normal(0, 0.01, (320, 1))
    x = common * rng.normal(1.0, 0.05, (1, n)) + rng.normal(0, 1, (320, n)) * scales
    ref = scm_gmvp(x)
    w0 = mod.allocate(x, anchor_penalty=0.0)["weights"]
    w4 = mod.allocate(x, anchor_penalty=4.0)["weights"]
    assert np.abs(w0 - ref).sum() < np.abs(w4 - ref).sum()


# ------------------------------------------------------------------- behaviour
def test_last_resort_budget_carries_no_risk_information():
    """The fallback that replaced the inverse-risk rule must be risk-free.

    If it were risk-aware it would be re-deriving an inverse-risk ordering from
    data, which is the prohibited book. It therefore depends only on column
    position, and that is exactly what is asserted: on an all-NaN panel (no
    observations at all) the result is the fixed (k, k-1, ..., 1) pattern.
    """
    mod = core()
    for n in range(2, 9):
        out = mod.allocate(np.full((300, n), np.nan))
        w = out["weights"]
        assert_legal(w, n)
        raw = np.arange(n, 0, -1, dtype=float)
        assert np.allclose(w, raw / raw.sum()), (n, w)


def test_symmetry_and_asymmetry_of_the_estimator():
    """Column order carries no information; time order does.

    Permuting columns must permute the weights and nothing else -- if it did
    not, the book would depend on the arbitrary order the grader happens to
    hand us the names in. Permuting rows, by contrast, *must* change the
    answer, because the covariance is estimated with a 63-day half-life and a
    recent-block mix; time order is genuine information and discarding it would
    make the estimator worse, not more correct.
    """
    mod = core()
    rng = np.random.default_rng(99)
    x = rng.normal(0, 0.01, (300, 5))

    a = mod.allocate(x)["weights"]
    cols = [2, 0, 4, 1, 3]
    c = mod.allocate(x[:, cols])["weights"]
    assert np.allclose(a[cols], c, atol=1e-10), (a[cols], c)

    rows = rng.permutation(300)
    b = mod.allocate(x[rows])["weights"]
    assert_legal(b, 5)
    assert not np.allclose(a, b, atol=1e-9), \
        "row order must matter: the covariance is recency-weighted"


def test_allocate_never_raises_and_repeats_exactly():
    mod = core()
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        first = mod.allocate(panel)
        second = mod.allocate(panel)
        assert np.array_equal(first["weights"], second["weights"]), name
        assert_legal(first["weights"], panel.shape[1])
        assert isinstance(first["diagnostics"].get("status"), str)


def test_single_asset_is_the_only_case_where_the_bans_are_vacuous():
    """n = 1: the only legal book is [1.0], which is also 1/n and also 1/sigma.

    Both prohibitions are therefore vacuous at n = 1, and the code says so in
    its status string instead of pretending one of them was satisfied.
    """
    mod = core()
    out = mod.allocate(np.linspace(-0.01, 0.01, 300).reshape(-1, 1))
    assert np.allclose(out["weights"], [1.0])
    assert "single" in out["diagnostics"]["status"] or \
        out["diagnostics"].get("final_rule") == "single_eligible_asset"


def test_check_weights_message_refers_to_round_two():
    mod = core()
    with pytest.raises(ValueError) as exc:
        mod.check_weights(np.full(4, 0.25), 4)
    assert "round 2" in str(exc.value)


# ------------------------------------------------------------- short selling
SHORT_CAPS = (0.01, 0.05, 0.20, 0.50)


def test_short_is_off_by_default_and_that_is_a_real_choice():
    mod = core()
    rng = np.random.default_rng(5)
    x = rng.normal(0, 0.01, (300, 6))
    a = mod.allocate(x)
    b = mod.allocate(x, allow_short=False)
    c = mod.allocate(x, allow_short=True, short_cap=0.0)
    assert np.array_equal(a["weights"], b["weights"])
    assert np.array_equal(a["weights"], c["weights"])
    assert a["diagnostics"]["allow_short"] is False
    assert a["diagnostics"]["short_cap"] == 0.0
    assert a["diagnostics"]["short_exposure"] == 0.0
    assert a["diagnostics"]["gross_exposure"] == pytest.approx(1.0, abs=1e-10)


def test_a_short_budget_nobody_granted_is_rejected():
    mod = core()
    x = np.random.default_rng(6).normal(0, 0.01, (300, 5))
    with pytest.raises(ValueError, match="requires allow_short"):
        mod.allocate(x, short_cap=0.1)
    for bad in (-0.1, 0.6, np.nan, np.inf):
        with pytest.raises(ValueError):
            mod.allocate(x, allow_short=True, short_cap=bad)
    with pytest.raises(ValueError, match="boolean"):
        mod.allocate(x, allow_short="yes")


def test_short_budget_is_respected_and_full_investment_is_kept():
    """The two claims the box constraint makes, on every hostile panel."""
    mod = core()
    for name, panel in HOSTILE.items():
        panel = np.asarray(panel, dtype=float)
        n = panel.shape[1]
        for cap in SHORT_CAPS:
            w = mod.allocate(panel, allow_short=True, short_cap=cap)["weights"]
            assert_legal(w, n, short_cap=cap)
            # Gross must be exactly the accounting identity, not an estimate.
            diag = mod.allocate(panel, allow_short=True, short_cap=cap)["diagnostics"]
            assert diag["gross_exposure"] == pytest.approx(
                1.0 + 2.0 * diag["short_exposure"], abs=1e-9), name


def test_short_never_produces_a_prohibited_book():
    """Opening the short dimension must not open a route to a banned book.

    Equal weight is the only one of the three prohibitions that is visible in
    the returned vector, so it is the one checked here -- and shorting cannot
    reach it, because 1/n has no negative entries while a book with a short
    position necessarily has one.
    """
    mod = core()
    rng = np.random.default_rng(2026)
    for trial in range(30):
        n = int(rng.integers(2, 12))
        scales = np.exp(rng.normal(0, 1.3, n))
        x = rng.normal(0, 1, (320, n)) * scales
        for cap in (0.05, 0.20, 0.50):
            w = mod.allocate(x, allow_short=True, short_cap=cap)["weights"]
            assert_legal(w, n, short_cap=cap)
            assert not np.allclose(w, 1.0 / n, atol=1e-8, rtol=1e-5), (trial, cap)
            sd = x.std(axis=0, ddof=1)
            ivp = (1.0 / sd) / (1.0 / sd).sum()
            assert np.abs(w - ivp).sum() > 1e-6, (trial, cap)


def test_short_projection_is_the_exact_euclidean_projection():
    """The solver's projection step must project, not merely land nearby.

    Verified against the KKT characterisation directly: at the projection the
    optimum is w = max(v - lam, -cap) for the single scalar that makes the
    budget bind, and complementary slackness must hold position by position.
    """
    mod = core()
    rng = np.random.default_rng(31337)
    for trial in range(50):
        n = int(rng.integers(2, 12))
        cap = float(rng.choice(SHORT_CAPS))
        v = rng.normal(0, 1, n) * float(rng.choice([0.05, 1.0, 4.0]))
        w = mod.project_box_simplex(v, cap)
        assert abs(float(w.sum()) - 1.0) < 1e-9
        assert w.min() >= -cap - 1e-10
        # w is the projection iff it is closest among the KKT candidates, so
        # compare against the sorted closed form computed independently here.
        lo, hi = float(v.min()) - 1e6, float(v.max()) + 1e6
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(np.maximum(v - mid, -cap).sum()) > 1.0:
                lo = mid
            else:
                hi = mid
        assert np.allclose(w, np.maximum(v - 0.5 * (lo + hi), -cap), atol=1e-9), trial


def test_short_budget_is_the_only_thing_that_changes_with_the_cap():
    """Enlarging the cap can only relax the constraint, so shorts cannot shrink.

    The comparison is made at the solver's own tolerance rather than exactly.
    A first version demanded 1e-12 and failed on a panel where the caps 0.01
    and 0.05 both left the constraint slack and delivered shorts differing by
    1.4e-11 -- two orders of magnitude below the 1e-8 duality gap the solver is
    asked to certify, so the difference is convergence noise, not a violation.
    """
    mod = core()
    rng = np.random.default_rng(88)
    tol = 1e-9  # not tighter than the 1e-8 gap the solver certifies
    for trial in range(12):
        n = int(rng.integers(3, 10))
        x = rng.normal(0, 0.01, (320, n)) * np.exp(rng.normal(0, 1.0, n))
        shorts = [mod.allocate(x, allow_short=True, short_cap=c)["diagnostics"]
                  ["short_exposure"] for c in (0.0, 0.01, 0.05, 0.20, 0.50)]
        for earlier, later in zip(shorts, shorts[1:]):
            assert later >= earlier - tol, (trial, shorts)


def test_last_resort_budget_stays_long_only_even_when_shorting_is_allowed():
    """The fallback is a risk-free convention, so it grants itself no shorts."""
    mod = core()
    for n in range(2, 7):
        out = mod.allocate(np.full((300, n), np.nan), allow_short=True,
                           short_cap=0.5)
        w = out["weights"]
        assert_legal(w, n, short_cap=0.5)
        assert (w >= 0).all(), w
        assert out["diagnostics"]["short_exposure"] == 0.0
