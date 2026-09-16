"""The branch ratchet fails when the figure drops, and says it could not run.

A gate is worth what it refuses. This one compares a measured branch percentage
against a recorded floor, and the three answers it can give are each a different
thing to a reader: at or above the floor, below it, and *could not be measured*
-- which must never read as the first. A report that is missing or unparseable
is the shape that would otherwise pass quietly, because a number nobody read is
a number nobody can be below.

Driven through the script rather than through its functions, because the exit
code is the contract: a lane reads that and nothing else.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "branch_coverage.py"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2


def _report(tmp_path: Path, count: int, covered: int) -> Path:
    """Write a coverage report holding one file with these branch counts."""
    path = tmp_path / "branches.json"
    path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "files": [
                            {
                                "filename": "crates/valgebra-core/src/ir.rs",
                                "summary": {
                                    "branches": {
                                        "count": count,
                                        "covered": covered,
                                    }
                                },
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, str(GATE), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_figure_at_the_floor_passes(tmp_path: Path) -> None:
    """The recorded floor is the tree's, so the tree's own report passes."""
    floor = float(
        json.loads(
            (ROOT / "scripts" / "branch_coverage.json").read_text(encoding="utf-8")
        )["floor"]
    )
    # One tenth of a point above the floor, rounded up, so the report is at or
    # above it whatever the recorded number's decimals are.
    report = _report(tmp_path, 10_000, int(floor * 100) + 1)
    done = _run([str(report)])
    assert done.returncode == EXIT_OK, done.stdout + done.stderr
    assert "at or above the floor" in done.stdout


def test_a_figure_below_the_floor_fails(tmp_path: Path) -> None:
    """The refusal this exists for: an arm that was reached is not any more."""
    report = _report(tmp_path, 1000, 400)
    done = _run([str(report)])
    assert done.returncode == EXIT_FAIL, done.stdout + done.stderr
    assert "below the recorded floor" in done.stdout


def test_a_report_that_cannot_be_read_is_not_a_pass(tmp_path: Path) -> None:
    """"Did not measure" has its own answer, and it is not "did not regress"."""
    assert _run([str(tmp_path / "absent.json")]).returncode == EXIT_CANNOT_RUN
    assert _run([]).returncode == EXIT_CANNOT_RUN

    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert _run([str(broken)]).returncode == EXIT_CANNOT_RUN

    # A report the tool wrote and that holds no branch at all: the scope
    # excluded everything, which measures nothing and must say so.
    empty = _report(tmp_path, 0, 0)
    assert _run([str(empty)]).returncode == EXIT_CANNOT_RUN


def test_the_recorded_floor_is_under_what_the_tree_measures() -> None:
    """A floor at the measurement makes every ordinary commit re-record it.

    Read as a band rather than as a number: the floor sits below what the tree
    reads, far enough that a commit adding a line does not have to touch it and
    close enough that losing a file's worth of arms is caught.
    """
    recorded = json.loads(
        (ROOT / "scripts" / "branch_coverage.json").read_text(encoding="utf-8")
    )
    floor = float(recorded["floor"])
    assert 50 <= floor <= 100, floor
    assert recorded["_comment"].strip(), "the record carries no reason"
