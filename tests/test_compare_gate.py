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

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from _tree import ROOT, load_script

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _load_gate() -> ModuleType:
    """Import the gate by path; ``scripts/`` is not an importable package."""
    return load_script("compare_gate")


gate = _load_gate()

CEILINGS = {"scalars": 1.000, "records": 2.000}


def test_a_run_names_one_mode() -> None:
    """A recording, a window, a count and a comparison are four things to do.

    Naming two would leave one of them undone without a word, so the command
    line refuses the pair with exit 2, the gate's "could not run".
    """
    for argv in (
        ["--update", "--count", "scalar", "3"],
        ["--record-window", "window.json", "--update"],
        ["--count", "scalar", "3", "--record-window", "window.json"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            gate._arguments(argv)  # noqa: SLF001
        assert excinfo.value.code == gate.EXIT_CANNOT_RUN, argv


def test_a_ratio_under_its_ceiling_passes() -> None:
    over, disagree = gate.judge({"scalars": 0.999, "records": 2.000}, CEILINGS)
    assert over == []
    assert not disagree


def test_a_ratio_over_its_ceiling_fails() -> None:
    over, disagree = gate.judge({"scalars": 1.001, "records": 2.000}, CEILINGS)
    assert over == ["scalars"]
    assert not disagree


def test_every_shape_over_its_ceiling_is_named() -> None:
    over, _ = gate.judge({"scalars": 9.0, "records": 9.0}, CEILINGS)
    assert over == ["records", "scalars"]


def test_a_measured_shape_with_no_ceiling_is_refused() -> None:
    # A shape added without a ceiling has nothing to breach, so a ceiling-only
    # reading passes it having measured nothing against anything.
    _, disagree = gate.judge({"scalars": 1.0, "records": 2.0, "unions": 1.0}, CEILINGS)
    assert disagree


def test_a_ceiling_for_a_shape_that_is_gone_is_refused() -> None:
    _, disagree = gate.judge({"scalars": 1.0}, CEILINGS)
    assert disagree


def test_every_shape_the_gate_measures_carries_a_ceiling() -> None:
    # Held to the file rather than to a count: the shapes are built from
    # pydantic and the extension, so this reads the names the gate would
    # measure and the names the file claims, in both directions.
    recorded = json.loads(
        (ROOT / "scripts" / "perf_compare.json").read_text(encoding="utf-8")
    )
    ceilings = recorded["ceilings"]
    assert len(ceilings) >= 4
    assert all(float(value) > 0 for value in ceilings.values())
    # The four paths a claim about speed has to cover: the accept walk, the
    # JSON walk, compilation, and the report a failure builds.
    assert {"scalar", "json_document", "build", "error_report"} <= set(ceilings)


def test_an_unreadable_ceiling_file_is_could_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A gate that could not read its ceilings compared nothing. Exit 2 keeps that
    # distinguishable from a regression, which is exit 1.
    monkeypatch.setattr(gate, "CEILING_FILE", tmp_path / "absent.json")
    assert gate.main() == gate.EXIT_CANNOT_RUN


def test_a_missing_benchmark_dependency_is_could_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "CEILING_FILE", ROOT / "scripts" / "perf_compare.json")

    def no_pydantic() -> dict[str, object]:
        raise ImportError("No module named 'pydantic'")

    monkeypatch.setattr(gate, "shapes", no_pydantic)
    assert gate.main() == gate.EXIT_CANNOT_RUN


def test_a_payload_the_validator_rejects_is_a_rig_fault() -> None:
    # The gate times the ACCEPT path; a shape whose payload is rejected takes the
    # fast reject path and would read as a speed-up. `warm_up` refuses it.
    shape = {
        "valgebra": lambda _data: False,
        "pydantic": lambda _data: None,
        "data": 1,
        "number": 1,
    }
    assert not gate.warm_up({"broken": shape})
    ok = {**shape, "valgebra": lambda _data: True}
    assert gate.warm_up({"fine": ok})


def test_the_benchmarks_load_the_gate_where_they_run(tmp_path: Path) -> None:
    """The benchmarks' conftest loads the gate by path, in pytest's own process.

    `scripts/` is not on that import path, so the conftest puts it there before
    loading, as `load_script` does here: without it the gate cannot import
    `_lib`, and every benchmark errored at setup. So the conftest's loader is
    run in a child interpreter whose path does not hold `scripts/`.
    """
    load = (
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location('conftest', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "print(module._compare_gate().provenance.__name__)"
    )
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, "-c", load, str(ROOT / "benches" / "conftest.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "provenance"
