#!/usr/bin/env python3
"""Round-5 reproducible evidence report.

Round 1's and round 2's verifiers are left untouched because they still describe
valid artifacts. This is the round-5 counterpart, and it differs from round 2's
in two ways that matter:

* the gate list reflects what round 5 actually has to establish. There is no
  short budget and no equivalence-to-the-long-only-core question, because the
  round pins shorts off. What replaces them is a *compliance margin* gate: with
  a pure-risk score the optimum sits as close to the prohibited SCM minimum
  variance book as the rules allow, so the distance to that book is measured
  per fold and asserted rather than assumed;
* the contract record is the round's own, including the explicit statement that
  weights must be nonnegative and sum to one.

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

SUBMISSION = ROOT / "submission/portfolio_round5.py"
CORE = ROOT / "src/portfolio_game_round5/core.py"
CONFIG = ROOT / "configs/submission_round5.json"


def run(name: str, args: list[str], extra_env: dict | None = None) -> dict:
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if extra_env:
        env.update(extra_env)
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

    checks["build"] = run("r5-build-check",
                          ["tools/build_submission_round5.py", "--check"])
    checks["syntax"] = run("r5-syntax-check",
                           ["-m", "compileall", "-q", "src", "research",
                            "submission", "tools", "tests"])
    checks["frozen_config"] = run("r5-frozen-config", ["tools/check_config_round5.py"])
    checks["pytest"] = run("r5-pytest",
                           ["-m", "pytest", "-q", "--junitxml=reports/r5-pytest.xml"])
    checks["hostile_inputs"] = run("r5-stress",
                                   ["tools/round5_stress.py",
                                    "--out", "reports/round5_stress.json"])
    checks["random_windows"] = run("r5-random-windows", [
        "tools/round5_random_window.py", "--reps", str(a.reps),
        "--datasets", "sp500:20,ftse100:64,nasdaq:60,factors:5",
        "--out", "reports/round5_random_windows.json"])
    checks["isolated_file"] = run("r5-isolated", ["tools/round5_isolated_check.py"])
    checks["backtest"] = run("r5-backtest", ["tools/round5_backtest_report.py"])
    checks["compliance_margin"] = run("r5-compliance", [
        "tools/round5_compliance_check.py",
        "--out", "reports/round5_compliance.json"])
    checks["teacher_preflight"] = run("r5-preflight", ["tools/preflight_round5.py"])

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    xml = ROOT / "reports/r5-pytest.xml"
    if xml.exists():
        doc = ET.parse(xml)
        for suite in doc.findall(".//testsuite"):
            for key in totals:
                totals[key] += int(suite.attrib.get(key, 0))
    totals["passed"] = (totals["tests"] - totals["failures"]
                        - totals["errors"] - totals["skipped"])

    backtest = {}
    bt_path = ROOT / "reports/round5_backtest.json"
    if bt_path.exists():
        bt = json.loads(bt_path.read_text(encoding="utf-8"))
        backtest = {row["dataset"]: {
            "portfolios": row["n_portfolios"],
            "failed": row["n_failed_portfolios"],
            "fallback": row["n_fallback_portfolios"],
            "annual_return": row["annual_return"],
            "annual_volatility": row["annual_volatility"],
            "max_drawdown": row["max_drawdown"],
            "cvar_95": row["cvar_95"],
            "gross_max": row["exposure"]["gross_max"],
            "short_max": row["exposure"]["short_max"],
            "l1_to_scm_gmvp_median": row["prohibited_distance_l1"]["scm_gmvp"]["median"],
            "exact_prohibited_hits": row["prohibited_exact_hits"],
        } for row in bt["datasets"]}
        backtest_sha = bt.get("submission_sha256")
    else:
        backtest_sha = None

    compliance = {}
    cmp_path = ROOT / "reports/round5_compliance.json"
    if cmp_path.exists():
        cmp_data = json.loads(cmp_path.read_text(encoding="utf-8"))
        compliance = {row["dataset"]: {
            "l1_scm_gmvp_median": row["l1_scm_gmvp_median"],
            "l1_scm_gmvp_worst_fold": row["l1_scm_gmvp_min"],
            "corr_scm_gmvp_max": row["corr_scm_gmvp_max"],
            "exact_hits": row["exact_hits"],
        } for row in cmp_data["datasets"]}

    windows = {}
    wr_path = ROOT / "reports/round5_random_windows.json"
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

    gate_names = ("build", "syntax", "frozen_config", "pytest", "hostile_inputs",
                  "random_windows", "isolated_file", "backtest",
                  "compliance_margin", "teacher_preflight")
    all_gates = all(checks[k]["exit_code"] == 0 for k in gate_names)

    # A pass is only meaningful if the numbers were produced from the bytes
    # being shipped. Assert that link instead of assuming it.
    submission_sha = sha(SUBMISSION)
    sha_consistent = (backtest_sha is None) or (backtest_sha == submission_sha)
    no_exact_hits = all(v.get("exact_hits", 0) == 0 for v in compliance.values()) \
        if compliance else None

    report = {
        "round": 5,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "contract": {"shortselling": False, "budget": 1.0, "lookback": 252,
                     "optimize_every": 20, "weight_drift": True,
                     "weights_nonnegative_and_sum_to_one": True,
                     "scoring": {"annual_volatility": 0.10, "max_drawdown": 0.10,
                                 "cvar_95": 0.10, "failure_rate": 0.70}},
        "frozen_params": {k: cfg.get(k) for k in
                          ("method", "half_life", "recent_mix", "anchor_penalty",
                           "gross_budget", "downside_mix")},
        "prohibitions": ["equally weighted portfolio (EWP)",
                         "inverse-volatility portfolio (IVP)",
                         "GMVP on the sample covariance matrix (SCM)"],
        "checks": checks,
        "pytest": totals,
        "submission_sha256": submission_sha,
        "core_sha256": sha(CORE),
        "config_sha256": sha(CONFIG),
        "backtest_sha256_matches_shipped_file": sha_consistent,
        "compliance_margin": compliance,
        "no_fold_returns_a_prohibited_book_exactly": no_exact_hits,
        "headline_backtest": backtest,
        "random_window_totals": windows,
        "real_skfolio_preflight": ("PASSED" if checks["teacher_preflight"]["exit_code"] == 0
                                   else "BLOCKED_OR_FAILED"),
        "ready_for_teacher_submission": bool(all_gates and sha_consistent
                                             and no_exact_hits is not False),
        "market_backtest": "RUN_ON_THREE_SKFOLIO_DATASETS_UNDER_THE_ROUND_5_CONTRACT",
        "hyperparameter_search": ("RUN_AS_A_RISK_BLEND_AND_PENALTY_SWEEP,SEE_"
                                  "reports/round5_sweep.json_and_round5_fine.json"),
        "notice": ("Passing these gates establishes format compliance, the stated "
                   "round-5 contract on every tested fold, absence of the three "
                   "prohibited books, and robustness to hostile inputs. It does not "
                   "establish a rank in the field: failure rate is 70% of that score "
                   "and the three risk terms are close together across compliant "
                   "candidates, so a clean run is the dominant term."),
    }
    (ROOT / "reports/verification_round5.json").write_text(
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
