"""The profile-guided build's training reaches each timed shape, or says why not.

The published wheels are laid out by a profile taken over
`scripts/pgo_workload.py`, and a path the training never enters is laid out by
the inliner's guess: a list nested twenty-five deep took 64% longer in the
profiled wheel while the training read homogeneous lists only, and the
instruction gate, which builds without a profile, read 0.9%. Each shape a gate
measures is a claim about the shipped wheel, so the training names, for every
one, the functions whose calls take its reading (`TRAINED`) -- or what the shape
reads and why no call does (`UNTRAINED`).

Held in both directions over the two gates' own registries: a shape either gate
adds with no row fails, a row for a shape neither gate measures fails, and a row
naming a function the workload does not define, or one `main` never runs,
fails. Whether a named function's calls do take the reading is the reviewer's
half, as it is for every claim a table makes about code.

LEDGER: every shape the comparison and instruction gates measure is trained or
excused
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from _reason import is_a_reason

if TYPE_CHECKING:
    from types import ModuleType

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKLOAD = ROOT / "scripts" / "pgo_workload.py"

#: The module-level names the tables are built from: the tables and the reason
#: strings they share.
_TABLE_NAMES = frozenset({"TRAINED", "UNTRAINED", "_JSON_UNTRAINED"})


def _perf_gate() -> ModuleType:
    """Import the instruction gate by path; ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "perf_gate_for_training", ROOT / "scripts" / "perf_gate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _module() -> ast.Module:
    return ast.parse(WORKLOAD.read_text(encoding="utf-8"))


def _tables() -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Read the two tables from the workload's source.

    Built from the assignments alone rather than by importing the workload,
    which imports `valgebra` at its top and would make this check depend on the
    extension being built.
    """
    namespace: dict[str, object] = {}
    for node in _module().body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        if _TABLE_NAMES.intersection(targets):
            body = ast.Module(body=[node], type_ignores=[])
            exec(compile(body, str(WORKLOAD), "exec"), {}, namespace)  # noqa: S102
    trained, untrained = namespace.get("TRAINED"), namespace.get("UNTRAINED")
    assert isinstance(trained, dict), "the workload defines no TRAINED table"
    assert isinstance(untrained, dict), "the workload defines no UNTRAINED table"
    # The parse is the detector: a table it could not read would pass every
    # check below having compared nothing.
    assert len(trained) >= 20, f"TRAINED reads {len(trained)} row(s)"
    return trained, untrained


def _measured() -> set[str]:
    """Every shape the two gates measure, read from their own registries."""
    ceilings = json.loads(
        (ROOT / "scripts" / "perf_compare.json").read_text(encoding="utf-8")
    )["ceilings"]
    return set(ceilings) | set(_perf_gate().MODES)


def _reached() -> set[str]:
    """Name the workload's functions `main` runs, directly or through another."""
    functions = {
        node.name: node for node in _module().body if isinstance(node, ast.FunctionDef)
    }
    reached, pending = set(), ["main"]
    while pending:
        name = pending.pop()
        if name in reached or name not in functions:
            continue
        reached.add(name)
        pending.extend(
            call.func.id
            for call in ast.walk(functions[name])
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        )
    return reached


def test_every_measured_shape_is_trained_or_excused() -> None:
    trained, untrained = _tables()
    both = set(trained) & set(untrained)
    assert not both, f"shapes both trained and excused: {sorted(both)}"
    measured = _measured()
    rows = set(trained) | set(untrained)
    assert rows == measured, (
        f"shapes a gate measures with no row: {sorted(measured - rows)}; rows for "
        f"a shape neither gate measures: {sorted(rows - measured)}. A shape added "
        "to a gate names the training that takes its reading, or says why none "
        "does, in scripts/pgo_workload.py."
    )


def test_every_training_row_names_a_function_main_runs() -> None:
    trained, _ = _tables()
    reached = _reached()
    assert "main" in reached, "the workload defines no main"
    unrun = {
        shape: sorted(set(functions) - reached)
        for shape, functions in trained.items()
        if not functions or set(functions) - reached
    }
    assert not unrun, (
        f"rows naming no function, or one main never runs: {unrun}. A function "
        "the training does not call trains nothing."
    )


def test_every_untrained_shape_says_what_it_reads() -> None:
    _, untrained = _tables()
    unargued = sorted(
        shape for shape, reason in untrained.items() if not is_a_reason(reason)
    )
    assert not unargued, (
        f"shapes excused from the training without a reason: {unargued}"
    )
