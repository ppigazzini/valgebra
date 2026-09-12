"""The comparison gate's drift ratchet, driven in every direction it can answer.

A ceiling is a claim the project makes and a shape can sit far under one for a
long time. The JSON document measured 0.87 against a ceiling of 1.00 while a
commit message published 0.78 for it, and nothing was red until somebody re-ran
the gate for an unrelated reason -- a number wrong on `main` for as long as it
took to notice by accident.

So the file carries a recorded ratio beside the ceiling, and this drives the
verdict that reads it. The gate's own timing is not involved: `drifted` takes
measured ratios and a recorded block and returns an answer, which is what lets a
failure be shown here rather than argued for.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The repository's gates are not the product suite: this reads a script.
pytestmark = pytest.mark.repository


def _gate():
    spec = importlib.util.spec_from_file_location(
        "compare_gate", ROOT / "scripts" / "compare_gate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _gate()


def _block(environment: str) -> dict[str, object]:
    return {
        "environment": environment,
        "ratios": {"scalar": 0.230, "json_document": 0.780},
        "tolerance": {"scalar": 0.020, "json_document": 0.050},
    }


def test_a_shape_that_holds_its_ratio_does_not_drift() -> None:
    here = GATE.environment()
    moved, unarmed = GATE.drifted(
        {"scalar": 0.235, "json_document": 0.790}, _block(here)
    )
    assert unarmed is None, "the ratchet should arm against its own environment"
    assert moved == []


def test_a_shape_that_loses_more_than_its_spread_drifts() -> None:
    here = GATE.environment()
    # 0.87 against a recorded 0.78 and a spread of 0.05: the move that went
    # unseen under a ceiling of 1.00, which is why this file exists.
    moved, unarmed = GATE.drifted(
        {"scalar": 0.230, "json_document": 0.870}, _block(here)
    )
    assert unarmed is None
    assert moved == ["json_document"]


def test_a_shape_at_exactly_its_tolerance_does_not_drift() -> None:
    # The boundary is the half a test usually leaves out, and the half that
    # decides whether the ratchet fires on noise.
    here = GATE.environment()
    moved, _ = GATE.drifted({"json_document": 0.780 + 0.050}, _block(here))
    assert moved == []
    moved, _ = GATE.drifted({"json_document": 0.780 + 0.0501}, _block(here))
    assert moved == ["json_document"]


def test_a_shape_that_gains_never_drifts() -> None:
    here = GATE.environment()
    moved, _ = GATE.drifted({"json_document": 0.400}, _block(here))
    assert moved == [], "a ratchet reads one direction"


def test_the_ratchet_does_not_arm_in_another_environment() -> None:
    """A recorded ratio read where it was not taken is a number about elsewhere.

    The interpreter, the global lock and the version of the library on the other
    side all move a ratio with nothing here changing, so judging against a
    recording from another of them would manufacture exactly the false signal
    the ceilings file warns about. It reports, and does not fail.
    """
    moved, unarmed = GATE.drifted(
        {"json_document": 0.990}, _block("cpython3.9-gil-pydantic_core-0.0.1")
    )
    assert moved == []
    assert unarmed is not None
    assert "cpython3.9" in unarmed


def test_an_unrecorded_gate_does_not_arm_and_does_not_fail() -> None:
    moved, unarmed = GATE.drifted({"json_document": 0.990}, {})
    assert moved == []
    assert unarmed is not None
    assert "--update" in unarmed


def test_a_shape_with_no_recorded_ratio_is_not_judged() -> None:
    # A shape added to the gate before anyone records it must not read as a
    # drift from zero.
    here = GATE.environment()
    moved, _ = GATE.drifted({"newcomer": 5.0}, _block(here))
    assert moved == []


def test_the_environment_names_what_moves_a_ratio() -> None:
    # Both locks, not one: the free-threaded build is the environment this
    # fingerprint exists to tell apart, and asserting "gil" here failed on
    # exactly that lane -- a test for naming the environment that assumed one.
    here = GATE.environment()
    assert here.startswith(f"cpython{sys.version_info.major}.{sys.version_info.minor}-")
    assert "-gil-" in here or "-freethreaded-" in here
    assert "pydantic_core-" in here


def test_the_verdict_reports_a_drift_as_a_failure() -> None:
    # `drifted` finding a shape and the gate exiting non-zero are two claims;
    # this is the second.
    here = GATE.environment()
    stored = _block(here)
    code = GATE._verdict(  # noqa: SLF001
        {"json_document": 0.870},
        {"json_document": 1.0},
        stored,
        GATE.Findings(over=[], disagree=False, moved=["json_document"]),
    )
    assert code == GATE.EXIT_FAIL


def test_a_ceiling_outranks_a_drift() -> None:
    # Both are failures, so the exit code cannot tell them apart; the report
    # must, and the ceiling is the louder one.
    stored = _block(GATE.environment())
    code = GATE._verdict(  # noqa: SLF001
        {"json_document": 1.5},
        {"json_document": 1.0},
        stored,
        GATE.Findings(over=["json_document"], disagree=False, moved=["json_document"]),
    )
    assert code == GATE.EXIT_FAIL


def test_every_shape_is_ratcheted_or_argued_out_of_it() -> None:
    """A shape carries a tolerance or a reason, and never neither or both.

    A shape with no tolerance is simply not judged, which is the right answer for
    one whose spread is a third of its own value -- and an invisible one, if it
    reaches that state by nobody writing a number down. `unratcheted` is where
    such a shape is named with the measurement that put it there, the same way
    `.cargo/mutants.toml` names an equivalent mutant with its argument.
    """
    document = json.loads((ROOT / "scripts" / "perf_compare.json").read_text())
    ceilings = set(document["ceilings"])
    recorded = document["recorded"]
    judged = set(recorded["tolerance"])
    excused = set(recorded["unratcheted"])

    assert judged & excused == set(), f"both ratcheted and excused: {judged & excused}"
    assert judged | excused == ceilings, (
        f"shapes with neither a tolerance nor a reason: {ceilings - judged - excused}; "
        f"named for a shape that is gone: {(judged | excused) - ceilings}"
    )
    for shape, reason in recorded["unratcheted"].items():
        assert len(reason) > 80, f"{shape} is excused without an argument"


def test_a_tolerance_is_a_spread_rather_than_a_wish() -> None:
    # A tolerance wider than the shape's ceiling judges nothing; one at zero
    # fires on the first run. Both are ways of having a ratchet that is not one.
    document = json.loads((ROOT / "scripts" / "perf_compare.json").read_text())
    for shape, tolerance in document["recorded"]["tolerance"].items():
        assert 0.0 < tolerance < document["ceilings"][shape], shape
