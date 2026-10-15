#!/usr/bin/env python3
"""Frozen-config gate for round 5.

Four ways the shipped artifact can silently stop matching the configuration the
selection was made on, all of which have to be closed:

1. `configs/submission_round5.json` drifts from the core's own defaults, so the
   sweep that chose the parameters no longer describes what runs;
2. the shipped file's embedded `allocate(...)` call drifts from the config, so
   the file the teacher runs is not the configuration that was evaluated;
3. the shipped file is regenerated from an older core, so the tested code and
   the uploaded code diverge while both still look fine in isolation;
4. the frozen parameters stop satisfying the round's *stated* contract. Round 5
   is the first round to state it unambiguously -- weights nonnegative and
   summing to one, budget 1.0, no cash -- so that is checked directly here
   rather than inferred from the absence of a short budget.

A deliberate deviation from a core default is allowed only if the config
declares it in `overrides_core_default`; otherwise a drifted default would pass
unnoticed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/submission_round5.json"
CORE = ROOT / "src/portfolio_game_round5/core.py"
SHIPPED = ROOT / "submission/portfolio_round5.py"

PARAMS = ("method", "half_life", "recent_mix", "anchor_penalty", "gross_budget",
          "downside_mix")


def fail(message: str) -> None:
    print(f"CONFIG GATE FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("round") != 5:
        fail(f"config round is {cfg.get('round')!r}, expected 5")
    missing = [k for k in PARAMS if k not in cfg]
    if missing:
        fail(f"config is missing {missing}")

    sys.path.insert(0, str(ROOT / "src"))
    import inspect

    import numpy as np

    from portfolio_game_round5.core import allocate

    sig = inspect.signature(allocate)
    declared = set(cfg.get("overrides_core_default", []))
    undeclared = [k for k in declared if k not in PARAMS]
    if undeclared:
        fail(f"overrides_core_default names unknown parameters {undeclared}")
    override_notes = []
    for key in PARAMS:
        default = sig.parameters[key].default
        if default != cfg[key]:
            if key not in declared:
                fail(f"core default {key}={default!r} != config {cfg[key]!r} "
                     f"and {key!r} is not listed in overrides_core_default")
            override_notes.append(f"  override         : {key}={cfg[key]!r} "
                                  f"(core default {default!r})")

    # The round's stated contract, asserted rather than assumed.
    if float(cfg["gross_budget"]) != 1.0:
        fail(f"round 5 sets budget = 1.0; config says {cfg['gross_budget']!r}")

    allocated = allocate(np.zeros((3, 2)), **{k: cfg[k] for k in PARAMS})
    weights = allocated["weights"]
    if not np.isfinite(weights).all():
        fail("the frozen parameter set returns non-finite weights")
    if weights.min() < -1e-12:
        fail(f"round 5 forbids shorts; min weight is {weights.min()!r}")
    if abs(float(weights.sum()) - 1.0) > 1e-8:
        fail(f"round 5 requires full investment; sum is {weights.sum()!r}")

    shipped = SHIPPED.read_text(encoding="utf-8")

    call = re.search(r"allocate\(\s*returns,(.*?)\)\s*\n", shipped, re.S)
    if call is None:
        fail("could not find the embedded allocate(...) call in the shipped file")
    body = call.group(1)
    for key in PARAMS:
        value = cfg[key]
        pattern = (f"{key}=['\"]{re.escape(str(value))}['\"]" if isinstance(value, str)
                   else rf"{key}={re.escape(repr(value))}")
        if re.search(pattern, body) is None:
            fail(f"shipped file does not pass {key}={value!r}")

    digest = hashlib.sha256(CORE.read_bytes()).hexdigest()
    if f"# Numerical core SHA256: {digest}" not in shipped:
        fail("shipped file was built from a different core than the one on disk")
    if CORE.read_text(encoding="utf-8") not in shipped:
        fail("shipped file does not contain the core verbatim")

    print("CONFIG GATE PASSED")
    print("  round            : 5")
    print(f"  frozen params    : { {k: cfg[k] for k in PARAMS} }")
    print(f"  contract         : long-only, fully invested, budget={cfg['gross_budget']}")
    for note in override_notes:
        print(note)
    print(f"  core sha256      : {digest}")
    print(f"  shipped sha256   : "
          f"{hashlib.sha256(SHIPPED.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
