#!/usr/bin/env python3
"""Frozen-config gate for round 2.

Three ways the shipped artifact can silently stop matching the configuration the
selection was made on, all of which have to be closed:

1. `configs/submission_round2.json` drifts from the core's own defaults, so the
   sweep that chose the parameters no longer describes what runs;
2. the shipped file's embedded `allocate(...)` call drifts from the config, so
   the file the teacher runs is not the configuration that was evaluated;
3. the shipped file is regenerated from an older core, so the tested code and
   the uploaded code diverge while both still look fine in isolation.

(2) and (3) are why the round-1 verifier compared a hash rather than trusting
that the builder had been re-run; the same check is applied here.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/submission_round2.json"
CORE = ROOT / "src/portfolio_game_round2/core.py"
SHIPPED = ROOT / "submission/portfolio_round2.py"

PARAMS = ("method", "half_life", "recent_mix", "anchor_penalty",
          "allow_short", "short_cap")


def fail(message: str) -> None:
    print(f"CONFIG GATE FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("round") != 2:
        fail(f"config round is {cfg.get('round')!r}, expected 2")
    missing = [k for k in PARAMS if k not in cfg]
    if missing:
        fail(f"config is missing {missing}")

    sys.path.insert(0, str(ROOT / "src"))
    from portfolio_game_round2.core import allocate
    import inspect

    sig = inspect.signature(allocate)
    declared = set(cfg.get("overrides_core_default", []))
    undeclared = [k for k in declared if k not in PARAMS]
    if undeclared:
        fail(f"overrides_core_default names unknown parameters {undeclared}")
    override_notes = []
    for key in PARAMS:
        default = sig.parameters[key].default
        if default != cfg[key]:
            # A deliberate override is allowed, but only if the config says so.
            # The point of the check is that a drifted setting cannot pass
            # unnoticed, not that the default is the only permitted value.
            if key not in declared:
                fail(f"core default {key}={default!r} != config {cfg[key]!r} "
                     f"and {key!r} is not listed in overrides_core_default")
            override_notes.append(f"  override         : {key}={cfg[key]!r} "
                                  f"(core default {default!r})")

    import numpy as np

    allocated = allocate(np.zeros((3, 2)), **{k: cfg[k] for k in PARAMS})
    weights = allocated["weights"]
    if not np.isfinite(weights).all() or abs(float(weights.sum()) - 1.0) > 1e-8:
        fail("the frozen parameter set does not produce a legal book")
    floor = -float(cfg["short_cap"]) if cfg["allow_short"] else 0.0
    if float(weights.min()) < floor:
        fail(f"the frozen parameter set returns a weight below its own floor "
             f"({weights.min()!r} < {floor!r})")

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
    print(f"  round            : 2")
    print(f"  frozen params    : { {k: cfg[k] for k in PARAMS} }")
    for note in override_notes:
        print(note)
    print(f"  core sha256      : {digest}")
    print(f"  shipped sha256   : "
          f"{hashlib.sha256(SHIPPED.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
