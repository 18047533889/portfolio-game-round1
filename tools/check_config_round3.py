#!/usr/bin/env python3
"""Frozen-config gate for round 3.

Three ways the shipped artifact can silently stop matching the configuration the
selection was made on, all of which have to be closed:

1. `configs/submission_round3.json` drifts from the core's own defaults, so the
   sweep that chose the parameters no longer describes what runs;
2. the shipped file's embedded `allocate(...)` call drifts from the config, so
   the file the teacher runs is not the configuration that was evaluated;
3. the shipped file is regenerated from an older core, so the tested code and
   the uploaded code diverge while both still look fine in isolation.

A deliberate deviation from a core default is allowed, but only if the config
declares it in `overrides_core_default`; otherwise a drifted default would pass
unnoticed. Round 3 relies on this: gross_leverage is frozen at the brief's 1.5
while the core default is 1.0, so the deviation is asserted rather than implied.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/submission_round3.json"
CORE = ROOT / "src/portfolio_game_round3/core.py"
SHIPPED = ROOT / "submission/portfolio_round3.py"

PARAMS = ("method", "half_life", "recent_mix", "anchor_penalty", "gross_leverage")


def fail(message: str) -> None:
    print(f"CONFIG GATE FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("round") != 3:
        fail(f"config round is {cfg.get('round')!r}, expected 3")
    missing = [k for k in PARAMS if k not in cfg]
    if missing:
        fail(f"config is missing {missing}")

    sys.path.insert(0, str(ROOT / "src"))
    import inspect

    import numpy as np

    from portfolio_game_round3.core import allocate

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

    allocated = allocate(np.zeros((3, 2)), **{k: cfg[k] for k in PARAMS})
    weights = allocated["weights"]
    if not np.isfinite(weights).all() or abs(float(weights.sum()) - 1.0) > 1e-8:
        fail("the frozen parameter set does not produce a legal book")
    budget = 0.5 * (float(cfg["gross_leverage"]) - 1.0)
    short = float(np.maximum(-weights, 0.0).sum())
    if short > budget + 1e-9:
        fail(f"the frozen parameter set exceeds its own short budget "
             f"({short!r} > {budget!r})")

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
    print("  round            : 3")
    print(f"  frozen params    : { {k: cfg[k] for k in PARAMS} }")
    print(f"  short budget     : {budget}  (gross leverage {cfg['gross_leverage']})")
    for note in override_notes:
        print(note)
    print(f"  core sha256      : {digest}")
    print(f"  shipped sha256   : "
          f"{hashlib.sha256(SHIPPED.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
