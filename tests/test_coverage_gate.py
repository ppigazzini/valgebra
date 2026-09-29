"""The per-file coverage floor must fail on every way its verdict can be wrong.

A gate that cannot be shown to fail is not evidence. ``scripts/coverage_gate.py``
reads llvm-cov's export, and each of its verdicts is driven here from a
synthetic report rather than an instrumented build: a file under the floor, an
excuse for a file over it, an excuse for a file the report does not carry, and
a report naming too few files to be believed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "coverage_gate.py"


def _load_gate() -> ModuleType:
    """Import the gate by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("coverage_gate", GATE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

#: Well over either scope's floor.
HIGH = (99.0, 99.0)

#: Under either scope's floor, on both counts.
LOW = (10.0, 10.0)


def _report(path: Path, *exports: dict[str, tuple[float, float]]) -> Path:
    """Write an llvm-cov export: one object per binary, each file's summary."""
    data = [
        {
            "files": [
                {
                    "filename": str(ROOT / name),
                    "summary": {
                        "lines": {"percent": lines},
                        "regions": {"percent": regions},
                    },
                }
                for name, (lines, regions) in export.items()
            ]
        }
        for export in exports
    ]
    path.write_text(json.dumps({"data": data}), encoding="utf-8")
    return path


def _files(
    count: int, reading: tuple[float, float] = HIGH
) -> dict[str, tuple[float, float]]:
    return {f"crates/planted/src/file{index}.rs": reading for index in range(count)}


def _verdict(
    monkeypatch: pytest.MonkeyPatch,
    report: Path,
    excused: dict[str, str] | None = None,
) -> int:
    """Run the gate's `main` on `report` under the core scope and `excused`."""
    monkeypatch.setattr(gate, "BELOW_THE_FLOOR", {"core": excused or {}})
    try:
        return gate.main(["--json", str(report)])
    except SystemExit as stop:
        return int(stop.code or 0)


def test_every_file_over_the_floor_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path / "cov.json", _files(5))
    assert _verdict(monkeypatch, report) == gate.EXIT_OK


def test_a_file_under_the_floor_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _files(5) | {"crates/planted/src/driven_by_nothing.rs": LOW}
    report = _report(tmp_path / "cov.json", files)
    assert _verdict(monkeypatch, report) == gate.EXIT_FAIL


def test_a_file_under_one_floor_alone_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lines and regions are two floors, and a file clearing one is not excused
    # the other: the regions are where an arm nothing reaches shows.
    files = _files(5) | {"crates/planted/src/arms.rs": (99.0, 10.0)}
    report = _report(tmp_path / "cov.json", files)
    assert _verdict(monkeypatch, report) == gate.EXIT_FAIL


def test_an_excused_file_under_the_floor_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "crates/planted/src/excused.rs"
    report = _report(tmp_path / "cov.json", _files(5) | {name: LOW})
    assert _verdict(monkeypatch, report, {name: "a reason"}) == gate.EXIT_OK


def test_an_excuse_for_a_file_over_the_floor_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The excuse has outlived what it excused, and a list that may only shrink
    # has to say so rather than carry it.
    name = "crates/planted/src/climbed.rs"
    report = _report(tmp_path / "cov.json", _files(5) | {name: HIGH})
    assert _verdict(monkeypatch, report, {name: "a reason"}) == gate.EXIT_FAIL


def test_an_excuse_for_a_file_the_report_does_not_carry_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path / "cov.json", _files(5))
    excused = {"crates/planted/src/renamed.rs": "a reason"}
    assert _verdict(monkeypatch, report, excused) == gate.EXIT_FAIL


def test_a_report_naming_too_few_files_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A filter that matched almost nothing writes a report every file of which
    # clears the floor: green, having read nothing.
    report = _report(tmp_path / "cov.json", _files(gate.LEAST_FILES - 1))
    assert _verdict(monkeypatch, report) == gate.EXIT_CANNOT_RUN


def test_a_file_two_binaries_measure_reads_the_higher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A binary that never linked the file reports it uncovered; the merged
    # profile supports the other binary's reading.
    name = "crates/planted/src/shared.rs"
    report = _report(tmp_path / "cov.json", _files(5) | {name: LOW}, {name: HIGH})
    assert _verdict(monkeypatch, report) == gate.EXIT_OK


@pytest.mark.parametrize(
    "text",
    ["not json", json.dumps({"data": [{"files": []}]})],
    ids=["unreadable", "no-summaries"],
)
def test_a_report_it_cannot_read_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    report = tmp_path / "cov.json"
    report.write_text(text, encoding="utf-8")
    assert _verdict(monkeypatch, report) == gate.EXIT_CANNOT_RUN


def test_the_exit_code_is_the_verdict(tmp_path: Path) -> None:
    # The lane reads the process's exit code, so the wiring from `main` to it
    # is held once end to end: a scope the gate does not know cannot run.
    result = subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [
            sys.executable,
            str(GATE),
            "--json",
            str(tmp_path / "cov.json"),
            "--scope",
            "none",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == gate.EXIT_CANNOT_RUN, result.stderr
