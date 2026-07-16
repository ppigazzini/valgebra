"""The mutation ratchet must fail on a new survivor and pass on a known one.

A gate that cannot be shown to fail is not evidence. These drive
``scripts/mutation_gate.py`` against synthetic ``mutants.out`` fixtures so the
ratchet's pass/fail behaviour is itself tested, without running a real sweep.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "mutation_gate.py"


def _run(
    work: Path, missed: list[str], baseline: list[str]
) -> subprocess.CompletedProcess[str]:
    out = work / "mutants.out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "caught.txt").write_text("some/file.rs:1:1: caught mutant\n")
    (out / "missed.txt").write_text("".join(line + "\n" for line in missed))
    (work / "scripts").mkdir(exist_ok=True)
    (work / "scripts" / "mutation_gate.py").write_text(GATE.read_text())
    (work / "scripts" / "mutation_baseline.json").write_text(
        json.dumps({"survivors": baseline}) + "\n"
    )
    return subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(work / "scripts" / "mutation_gate.py")],
        cwd=work,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_new_survivor_fails_the_gate(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        missed=["crates/x/src/a.rs:10:5: replace + with - in f"],
        baseline=[],
    )
    assert result.returncode == 1
    assert "NEW SURVIVOR" in result.stdout


def test_a_baselined_survivor_passes(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        missed=["crates/x/src/a.rs:10:5: replace + with - in f"],
        # Same identity, different line: the ratchet keys on file + mutation,
        # not on the drifting line:col.
        baseline=["crates/x/src/a.rs: replace + with - in f"],
    )
    assert result.returncode == 0
    assert "no new survivors" in result.stdout


def test_a_killed_baseline_survivor_still_passes(tmp_path: Path) -> None:
    # A baseline survivor the tests catch is an improvement, not a failure.
    result = _run(
        tmp_path,
        missed=[],
        baseline=["crates/x/src/a.rs: replace + with - in f"],
    )
    assert result.returncode == 0
    assert "killed (was in baseline)" in result.stdout


def test_an_empty_output_dir_refuses_to_pass(tmp_path: Path) -> None:
    (tmp_path / "mutants.out").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "mutation_gate.py").write_text(GATE.read_text())
    (tmp_path / "scripts" / "mutation_baseline.json").write_text('{"survivors": []}\n')
    result = subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(tmp_path / "scripts" / "mutation_gate.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    # A broken detector must not read as a clean tree.
    assert result.returncode != 0
    assert "did not run" in result.stderr
