"""The instruction-count gate must fail on every way a measurement can be wrong.

A gate that cannot be shown to fail is not evidence. These drive the decision
logic of ``scripts/perf_gate.py`` directly -- no cachegrind, no build -- so each
refusal is exercised: a count over the ceiling, a count under the floor (the
shape a workload that stopped doing the work produces), a workload checksum that
does not match, and output the gate cannot read at all.

The under-floor case is the one a one-sided budget misses: a hollow workload
measures low and reads as an improvement it never earned.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "perf_gate.py"


def _load_gate() -> ModuleType:
    """Import the gate by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("perf_gate", GATE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

BUDGET = 1_000_000
TOLERANCE = 0.10


def test_a_count_inside_the_band_passes() -> None:
    assert gate.check_against_budget(BUDGET, BUDGET, TOLERANCE) == 0
    assert gate.check_against_budget(1_050_000, BUDGET, TOLERANCE) == 0
    assert gate.check_against_budget(950_000, BUDGET, TOLERANCE) == 0


def test_a_count_over_the_ceiling_fails() -> None:
    assert gate.check_against_budget(1_100_001, BUDGET, TOLERANCE) == 1


def test_a_count_under_the_floor_fails() -> None:
    # The shape a workload that stopped doing the work produces. A ceiling-only
    # budget calls this "within budget" and publishes an improvement it never
    # earned, which is the failure this direction exists to catch.
    assert gate.check_against_budget(899_999, BUDGET, TOLERANCE) == 1
    assert gate.check_against_budget(0, BUDGET, TOLERANCE) == 1


def test_a_mismatched_checksum_fails() -> None:
    assert gate.check_checksum(134_000, 134_000, "core workload") == 0
    assert gate.check_checksum(0, 134_000, "core workload") == 1
    assert gate.check_checksum(133_999, 134_000, "core workload") == 1


def test_an_unreadable_instruction_count_is_not_a_pass() -> None:
    with pytest.raises(SystemExit) as excinfo:
        gate.parse_measurement("checksum=1\n", "cachegrind died before its banner")
    assert excinfo.value.code == 2


def test_an_unreadable_checksum_is_not_a_pass() -> None:
    # The workload ran under cachegrind and printed nothing the gate can tie to
    # the work; a count with no checksum behind it is not a verdict.
    with pytest.raises(SystemExit) as excinfo:
        gate.parse_measurement("", "==1== I   refs:      1,234,567")
    assert excinfo.value.code == 2


def test_a_readable_measurement_carries_both_halves() -> None:
    measured = gate.parse_measurement(
        "checksum=134000\n", "==1== I   refs:      252,026,154"
    )
    assert measured.irefs == 252026154
    assert measured.checksum == 134000
    # The binding workload prints a bare number rather than a labelled one.
    bare = gate.parse_measurement("150000\n", "==1== I   refs:      1,000")
    assert bare.checksum == 150000


def test_the_committed_core_checksum_matches_the_workload() -> None:
    # The recorded checksum is a constant of the workload's fixed corpus and
    # iteration count, so it belongs in the tree beside the budget. If the
    # workload is edited without re-recording, this fails before CI spends a
    # cachegrind run finding out.
    import json  # noqa: PLC0415

    budget = json.loads(
        (ROOT / "scripts" / "perf_budget.json").read_text(encoding="utf-8")
    )
    assert budget["core_workload_checksum"] == 134000
