"""The competitive ratio gate must fail on every way its verdict can be wrong.

A gate that cannot be shown to fail is not evidence. The gate's own decision is
driven here from measured ratios alone -- no pydantic, no timer -- so each
refusal is exercised: a ratio past the recorded ceiling, a shape measured with
no baseline, and a baseline naming a shape no longer measured.

The last two are the ones a ceiling-only reading misses. A shape added without
re-recording has no ceiling to breach, so it would sail through unmeasured; a
baseline key whose shape was removed is a ceiling with nothing behind it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "compare_gate.py"


def _load_gate() -> ModuleType:
    """Import the gate by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("compare_gate", GATE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

TOLERANCE = 0.35
BASE = {"scalars": 1.000, "records": 2.000}


def test_a_ratio_inside_the_tolerance_passes() -> None:
    failures, disagree = gate.judge(
        {"scalars": 1.300, "records": 2.600}, BASE, TOLERANCE
    )
    assert failures == []
    assert not disagree


def test_a_ratio_past_the_ceiling_fails() -> None:
    failures, disagree = gate.judge(
        {"scalars": 1.351, "records": 2.000}, BASE, TOLERANCE
    )
    assert failures == ["scalars"]
    assert not disagree


def test_every_regressing_shape_is_named() -> None:
    failures, _ = gate.judge({"scalars": 9.0, "records": 9.0}, BASE, TOLERANCE)
    assert failures == ["records", "scalars"]


def test_a_measured_shape_with_no_baseline_is_refused() -> None:
    # A shape added without re-recording has no ceiling to breach, so a
    # ceiling-only reading passes it having measured nothing against anything.
    _, disagree = gate.judge(
        {"scalars": 1.0, "records": 2.0, "unions": 1.0}, BASE, TOLERANCE
    )
    assert disagree


def test_a_baseline_shape_no_longer_measured_is_refused() -> None:
    _, disagree = gate.judge({"scalars": 1.0}, BASE, TOLERANCE)
    assert disagree


def test_the_committed_baseline_carries_a_tolerance_and_shapes() -> None:
    # The gate reads both from the file; an empty one would make every
    # comparison vacuous rather than failing.
    recorded = json.loads(
        (ROOT / "scripts" / "perf_compare.json").read_text(encoding="utf-8")
    )
    assert float(recorded["tolerance"]) > 0
    assert len(recorded["ratios"]) >= 2
