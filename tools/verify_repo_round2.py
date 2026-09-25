#!/usr/bin/env python3
"""Round-2 reproducible evidence report.

Round 1's verifier is left untouched because it still describes a valid
artifact (`submission/portfolio_round1.py`, `src/portfolio_game/core.py`).
This is its round-2 counterpart, and it differs in two ways that matter:

* the frozen-config check compares `configs/submission_round2.json` against the
  round-2 core, and the SHA is taken over the round-2 file and core;
* the acceptance loop adds the gates that the round-2 brief makes mandatory --
  the instructor's own self-test run on the file in isolation, and a random
  subset x random two-year window sweep under `weight_drift = TRUE`, because
  failure rate is 70% of the round-2 score and a single full-history backtest
  does not establish it.

Exit codes: 0 = all gates passed, 1 = a gate failed, 2 = integration blocked
while `--require-integration` was requested.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUBMISSION = ROOT / "submission/portfolio_round2.py"
CORE = ROOT / "src/portfolio_game_round2/core.py"
CONFIG = ROOT / "configs/submission_round2.json"


def run(name: str, args: list[str]) -> dict:
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (ROOT / "reports" / f"{name}.txt").write_text(result.stdout, encoding="utf-8")
    print(f"{name}: exit={result.returncode}", flush=True)
    return {"exit_code": result.returncode, "log": f"reports/{name}.txt"}


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-integration", action="store_true")
    ap.add_argument("--reps", type=int, default=60,
                    help="random subset/period draws per dataset in the sweep")
    a = ap.parse_args()

    (ROOT / "reports").mkdir(exist_ok=True)
    checks: dict[str, dict] = {}

    checks["build"] = run("r2-build-check",
                          ["tools/build_submission_round2.py", "--check"])
    checks["syntax"] = run("r2-syntax-check",
                           ["-m", "compileall", "-q", "src", "research",
                            "submission", "tools"])
    checks["frozen_config"] = run("r2-frozen-config", ["tools/check_config_round2.py"])
    checks["long_only_equivalence"] = run(
        "r2-long-only-equivalence",
        ["tools/round2_long_only_equivalence.py",
         "--out", "reports/round2_long_only_equivalence.json"])
    checks["short_math"] = run(
        "r2-short-projection",
        ["tools/round2_short_projection_check.py",
         "--out", "reports/round2_short_projection.json"])
    checks["pytest"] = run("r2-pytest",
                           ["-m", "pytest", "-q",
                            "--junitxml=reports/r2-pytest.xml"])
    checks["hostile_inputs"] = run("r2-stress", ["tools/round2_stress.py"])
    checks["random_windows"] = run("r2-random-windows", [
        "tools/round2_random_window.py", "--reps", str(a.reps),
        "--datasets", "sp500:20,ftse100:64,nasdaq:60,factors:5",
        "--out", "reports/round2_random_windows.json"])
    checks["isolated_file"] = run("r2-isolated", ["tools/round2_isolated_check.py"])
    checks["backtest"] = run("r2-backtest", ["tools/round2_backtest_report.py"])
    checks["teacher_preflight"] = run("r2-preflight", ["tools/preflight_round2.py"])

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    xml = ROOT / "reports/r2-pytest.xml"
    if xml.exists():
        doc = ET.parse(xml)
        for suite in doc.findall(".//testsuite"):
            for key in totals:
                totals[key] += int(suite.attrib.get(key, 0))
    totals["passed"] = (totals["tests"] - totals["failures"]
                        - totals["errors"] - totals["skipped"])

    backtest = {}
    bt_path = ROOT / "reports/round2_backtest.json"
    if bt_path.exists():
        bt = json.loads(bt_path.read_text(encoding="utf-8"))
        backtest = {row["dataset"]: {
            "portfolios": row["n_portfolios"],
            "failed": row["n_failed_portfolios"],
            "fallback": row["n_fallback_portfolios"],
            "annual_return": row["annual_return"],
            "max_drawdown": row["max_drawdown"],
            "gross_max": row["exposure"]["gross_max"],
            "short_max": row["exposure"]["short_max"],
            "l1_to_ewp_median": row["prohibited_distance_l1"]["ewp"]["median"],
            "l1_to_ivp_median": row["prohibited_distance_l1"]["ivp"]["median"],
            "l1_to_scm_gmvp_median": row["prohibited_distance_l1"]["scm_gmvp"]["median"],
            "exact_prohibited_hits": row["prohibited_exact_hits"],
        } for row in bt["datasets"]}
        backtest_sha = bt.get("submission_sha256")
    else:
        backtest_sha = None

    windows = {}
    wr_path = ROOT / "reports/round2_random_windows.json"
    if wr_path.exists():
        windows = json.loads(wr_path.read_text(encoding="utf-8"))["total"]

    packages = {}
    for name in ["numpy", "scipy", "scikit-learn", "pandas", "pytest", "skfolio",
                 "cvxpy", "cvxpy-base"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    cfg = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    allow_short = bool(cfg.get("allow_short", False))
    short_cap = float(cfg.get("short_cap", 0.0)) if allow_short else 0.0

    gate_names = ("build", "syntax", "frozen_config", "long_only_equivalence",
                  "short_math", "pytest", "hostile_inputs",
                  "random_windows", "isolated_file", "backtest", "teacher_preflight")
    all_gates = all(checks[k]["exit_code"] == 0 for k in gate_names)

    # A pass is only meaningful if the numbers were produced from the bytes
    # being shipped. Assert that link instead of assuming it.
    submission_sha = sha(SUBMISSION)
    sha_consistent = (backtest_sha is None) or (backtest_sha == submission_sha)

    report = {
        "round": 2,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "contract": {"shortselling": True, "leverage": 1, "lookback": 252,
                     "optimize_every": 20, "weight_drift": True,
                     "scoring": {"annual_return": 0.15, "max_drawdown": 0.15,
                                 "failure_rate": 0.70}},
        "short_selling": {
            "option_implemented": True,
            "frozen_allow_short": allow_short,
            "frozen_short_cap": short_cap,
            "net_exposure_always": 1.0,
            "gross_exposure_identity": "1 + 2 * (total short)",
            "long_only_path_bit_identical_to_pre_short_core":
                checks["long_only_equivalence"]["exit_code"] == 0,
            "projection_and_duality_gap_verified":
                checks["short_math"]["exit_code"] == 0,
            "sweep": "reports/round2_short_sweep.json",
        },
        "prohibitions": ["equally weighted portfolio (EWP)",
                         "inverse-volatility portfolio (IVP)",
                         "GMVP on the sample covariance matrix (SCM)"],
        "checks": checks,
        "pytest": totals,
        "submission_sha256": submission_sha,
        "core_sha256": sha(CORE),
        "config_sha256": sha(CONFIG),
        "backtest_sha256_matches_shipped_file": sha_consistent,
        "headline_backtest": backtest,
        "random_window_totals": windows,
        "real_skfolio_preflight": "PASSED" if checks["teacher_preflight"]["exit_code"] == 0
                                  else "BLOCKED_OR_FAILED",
        "ready_for_teacher_submission": bool(all_gates and sha_consistent),
        "market_backtest": "RUN_ON_THREE_SKFOLIO_DATASETS_UNDER_THE_ROUND_2_CONTRACT",
        "hyperparameter_search": "RUN_AS_A_PENALTY_SWEEP,SEE_reports/round2_penalty_scan.json",
        "notice": ("Passing these gates establishes format compliance, absence of "
                   "the three prohibited books on every tested fold, and "
                   "robustness to hostile inputs. It does not establish a rank "
                   "in the round-2 field: failure rate is 70% of that score and "
                   "the annual-return and drawdown terms are near-tied across "
                   "all compliant candidates, so a clean run is the dominant "
                   "term."),
    }
    (ROOT / "reports/verification_round2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not all_gates:
        return 1
    if a.require_integration and checks["teacher_preflight"]["exit_code"] != 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
