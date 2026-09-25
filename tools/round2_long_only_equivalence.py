"""Prove that the short-enabled core did not change the long-only behaviour.

Adding a parameter is only free if the pre-existing code path is untouched.
This gate compares the current core, driven with the Round-2 frozen settings
and shorting off, against a frozen copy of the core as it stood before shorting
existed (research/round2/frozen/core_r2_long_only_80bbd559.py, sha256
80bbd559...). The comparison is exact: weights must be equal bit for bit, and
every numeric diagnostic field must be equal too.

A tolerance-based comparison would hide exactly the kind of drift this gate
exists to catch, so there is no tolerance here.

One behaviour change is intended and is therefore asserted rather than
forgiven: the frozen reference discarded a converged optimum and returned the
HRP anchor whenever any column was ineligible, because a diagnostic helper
subtracted a k-vector from an n-vector and raised. Those cases are expected to
differ, and the gate requires that they differ in exactly one way -- the new
book must come from the certified QP path, with no internal error logged and a
legal weight vector. Anything else is reported as drift.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/portfolio_game_round2/core.py"
FROZEN = ROOT / "research/round2/frozen/core_r2_long_only_80bbd559.py"
FROZEN_SHA = "80bbd5594bf98e97642afbdb4107c3edfb0aecce0a6cf0a663882740ae3849cd"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panels() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20260926)
    out: list[tuple[str, np.ndarray]] = []
    for n in (1, 2, 3, 5, 12, 40):
        out.append((f"gauss45x{n}", rng.normal(0, 0.01, size=(45, n))))
        out.append((f"gauss252x{n}", rng.normal(0, 0.01, size=(252, n))))
    # Anisotropic, correlated, and recency-structured panels: these are the ones
    # that actually separate an estimator change from a refactor.
    base = rng.normal(0, 0.012, size=(300, 8))
    loadings = rng.normal(0, 1, size=(8, 8))
    out.append(("correlated300x8", base @ loadings))
    regime = np.vstack([rng.normal(0, 0.006, size=(200, 6)),
                        rng.normal(0, 0.03, size=(100, 6))])
    out.append(("regime300x6", regime))
    out.append(("constant3", np.ones((60, 3)) * 0.001))
    out.append(("zero3", np.zeros((60, 3))))
    dup = rng.normal(0, 0.01, size=(120, 4))
    out.append(("duplicated4", np.column_stack([dup[:, 0], dup[:, 0], dup[:, 1], dup[:, 2]])))
    nan = rng.normal(0, 0.01, size=(252, 6))
    nan[10:40, 0] = np.nan
    nan[100:252, 4] = np.nan
    nan[5, 2] = np.nan
    out.append(("missing252x6", nan))
    allnan = np.full((30, 4), np.nan)
    out.append(("allnan30x4", allnan))
    tiny = rng.normal(0, 0.01, size=(3, 4))
    out.append(("tiny3x4", tiny))
    out.append(("single_row1x5", rng.normal(0, 0.01, size=(1, 5))))
    out.append(("extreme5x3", np.array([[1e6, -1e6, 1.0], [2.0, 3.0, -1.0],
                                        [0.0, 0.0, 0.0], [1e-9, 1e-9, 1e-9],
                                        [5.0, 5.0, 5.0]])))
    return out


PARAM_GRID = [
    {},
    {"anchor_penalty": 0.0},
    {"anchor_penalty": 4.0},
    {"recent_mix": 0.0},
    {"recent_mix": 1.0},
    {"half_life": 1.0},
    {"half_life": 252.0},
    {"method": "hrp"},
    {"allow_short": False, "short_cap": 0.0},
    {"allow_short": True, "short_cap": 0.0},
]


def compare(a: dict, b: dict) -> list[str]:
    problems: list[str] = []
    wa, wb = np.asarray(a["weights"]), np.asarray(b["weights"])
    if wa.shape != wb.shape:
        return [f"shape {wa.shape} != {wb.shape}"]
    if not np.array_equal(wa, wb):
        delta = float(np.max(np.abs(wa - wb)))
        problems.append(f"weights differ (max |dw| = {delta:.3e})")
    da, db = a["diagnostics"], b["diagnostics"]
    shared = set(da) & set(db)
    if set(da) - set(db):
        problems.append(f"diagnostics missing in new: {sorted(set(da) - set(db))}")
    for key in sorted(shared):
        va, vb = da[key], db[key]
        if isinstance(va, float) and isinstance(vb, float):
            if not (va == vb or (np.isnan(va) and np.isnan(vb))):
                problems.append(f"diagnostic {key!r}: {va!r} != {vb!r}")
        elif isinstance(va, (int, str, bool)) and isinstance(vb, (int, str, bool)):
            if va != vb:
                problems.append(f"diagnostic {key!r}: {va!r} != {vb!r}")
        elif key in {"status", "final_rule"} and va != vb:
            problems.append(f"diagnostic {key!r}: {va!r} != {vb!r}")
    return problems


# The one intended behaviour change of the short-selling work. In the frozen
# reference, _prohibited_proximity() subtracted a k-vector from an n-vector and
# raised whenever any column was ineligible; allocate() caught that, discarded an
# already converged optimum, and silently returned the HRP anchor instead. Cases
# that hit it are expected to differ, and must differ in exactly this way.
BROADCAST_DEFECT = "could not be broadcast"


def is_pre_fix_fallback(result: dict) -> bool:
    diagnostics = result["diagnostics"]
    if not any(BROADCAST_DEFECT in str(e) for e in diagnostics.get("events", [])):
        return False
    return diagnostics.get("status") == "fallback_hrp"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    actual_frozen_sha = sha(FROZEN)
    print(f"frozen reference : {FROZEN.name}")
    print(f"  sha256         : {actual_frozen_sha}")
    if actual_frozen_sha != FROZEN_SHA:
        print("  !! frozen reference has been modified -- this gate is void")
        return 2

    old = load(FROZEN, "core_r2_frozen")
    new = load(CORE, "core_r2_current")
    print(f"current core     : {sha(CORE)}")

    cases, failures, fixes = 0, [], []
    for label, x in panels():
        for params in PARAM_GRID:
            cases += 1
            # The frozen reference predates the short parameters, so a set that
            # only toggles them is compared against the reference's own default
            # call. With short_cap = 0 the two must agree exactly; that is the
            # whole point of including these sets.
            reference_params = {k: v for k, v in params.items()
                                if k not in {"allow_short", "short_cap"}}
            try:
                ra = old.allocate(np.array(x, copy=True), **reference_params)
                ea = None
            except Exception as exc:  # noqa: BLE001
                ra, ea = None, f"{type(exc).__name__}: {exc}"
            try:
                rb = new.allocate(np.array(x, copy=True), **params)
                eb = None
            except Exception as exc:  # noqa: BLE001
                rb, eb = None, f"{type(exc).__name__}: {exc}"
            if ea or eb:
                if ea != eb:
                    failures.append(f"{label} {params}: raised {eb!r} vs {ea!r}")
                continue
            if is_pre_fix_fallback(ra):
                # Expected to differ. Require that it differs in exactly the
                # intended way, and that the new answer is a legal, certified
                # book rather than merely different.
                diagnostics = rb["diagnostics"]
                status = diagnostics.get("status")
                method = params.get("method", "regularized")
                if diagnostics.get("events"):
                    failures.append(f"{label} {params}: fixed path still logs "
                                    f"{diagnostics['events']}")
                    continue
                if method == "hrp":
                    # The HRP path returns the anchor either way, so only the
                    # status label may change -- the weights must not move.
                    if status != "hrp":
                        failures.append(f"{label} {params}: expected status 'hrp', "
                                        f"got {status!r}")
                        continue
                    moved = compare({"weights": ra["weights"], "diagnostics": {}},
                                    {"weights": rb["weights"], "diagnostics": {}})
                    if moved:
                        failures.append(f"{label} {params}: HRP weights moved: {moved}")
                        continue
                    fixes.append(f"{label} {params}: status fallback_hrp -> hrp "
                                 f"(weights unchanged)")
                    continue
                if status != "regularized_qp":
                    failures.append(f"{label} {params}: expected the QP path after "
                                    f"the fix, got status={status!r}")
                    continue
                if diagnostics.get("main_solver_converged") is not True:
                    failures.append(f"{label} {params}: fixed path is not certified")
                    continue
                w = np.asarray(rb["weights"])
                if w.min() < -1e-12 or abs(float(w.sum()) - 1.0) > 1e-8:
                    failures.append(f"{label} {params}: fixed path returns an "
                                    f"illegal book")
                    continue
                fixes.append(f"{label} {params}: fallback_hrp -> regularized_qp")
                continue
            for problem in compare(ra, rb):
                failures.append(f"{label} {params}: {problem}")

    print(f"\ncases compared   : {cases}")
    print(f"differences      : {len(failures)}")
    for line in failures[:40]:
        print(f"  - {line}")
    if len(failures) > 40:
        print(f"  ... {len(failures) - 40} more")
    print(f"intended fixes   : {len(fixes)}  (ineligible-column fallback defect)")
    for line in fixes:
        print(f"  * {line}")
    verdict = "IDENTICAL" if not failures else "DRIFTED"
    print(f"\nLONG-ONLY EQUIVALENCE: {verdict}")
    if args.out:
        args.out.write_text(json.dumps({
            "frozen_sha256": actual_frozen_sha,
            "current_core_sha256": sha(CORE),
            "cases": cases,
            "differences": failures,
            "intended_fixes": fixes,
            "verdict": verdict,
        }, indent=2))
        print(f"written -> {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
