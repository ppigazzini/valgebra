"""Every wall-clock assertion in the suite is one somebody argued for.

A bound on *growth* is a bound on the algorithm, and the tree says so in two
places: the performance page, and the core's own step-count instrument, whose
doc records that a wall-clock assertion "measures the machine as much as the
algorithm -- it passes on a quiet laptop and fails on a loaded runner for
reasons that have nothing to do with the code". One of the three below did
exactly that during a session that had a compiler running beside it.

Each survives for a reason, and the reason is that no counter in this tree
measures what it measures: two count work the decision's own step counter does
not see, because the cost they pin is oracle calls and rendered strings rather
than decision steps, and the third pins a determinisation that really is a
question about time. Each reads the fastest of several runs, which is the
estimator a loaded machine cannot spoil -- another process steals time from a
reading and never gives any back.

The list is closed. A fourth clock arriving in this suite is a fourth argument
to make, and this ledger is where it is made.

LEDGER: no test measures time except the three that argue for it
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# A repository check: it reads the suite's own source, which ships in no wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent

#: A reading of the clock, in the forms this suite writes one.
CLOCK = re.compile(r"\b(?:perf_counter|monotonic|process_time)\s*\(")

_EQUIVALENCE = "test_equivalence.py::test_two_wide_literal_sets"
_DEEP_VALUE = "test_adversarial_bounds.py::test_explaining_a_deep_value"
_PATTERN = "test_adversarial_bounds.py::test_a_pattern_whose_determinisation"

#: The tests that may read one, by the prefix that names them, each with why no
#: counter in this tree replaces it.
ARGUED = {
    _EQUIVALENCE: (
        "the defect was a call per pair into the bindings, and the decision's "
        "step counter does not count an oracle call: it reads the same five "
        "steps for five hundred members and for eight thousand"
    ),
    _DEEP_VALUE: (
        "the defect was a repr built per level and discarded, which no counter "
        "in this tree reports"
    ),
    _PATTERN: (
        "the bound really is about time: a determinisation that explodes is one "
        "that does not finish, and finishing is the property"
    ),
}


def _argument_for(name: str) -> str | None:
    """Return the recorded argument for a clock-reading test, by name prefix."""
    for prefix, why in ARGUED.items():
        if name.startswith(prefix):
            return why
    return None


def _tests_reading_a_clock() -> set[str]:
    """Return every test function in this suite whose body reads the clock."""
    found = set()
    for path in sorted(ROOT.glob("tests/test_*.py")):
        if path.name == Path(__file__).name:
            continue  # this file names the calls in order to refuse them
        current = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("def test_"):
                current = line[4:].split("(", 1)[0]
            if current and CLOCK.search(line):
                found.add(f"{path.name}::{current}")
    return found


def test_only_the_argued_tests_read_a_clock() -> None:
    reading = _tests_reading_a_clock()
    unargued = sorted(name for name in reading if _argument_for(name) is None)
    assert not unargued, (
        f"tests measuring time with no argument for it: {unargued}. A bound on "
        "growth belongs on a counter; if none measures what this one does, add "
        "the reason to ARGUED and say what the counter misses."
    )


def test_every_argued_test_exists_and_reads_a_clock() -> None:
    reading = _tests_reading_a_clock()
    stale = sorted(
        prefix
        for prefix in ARGUED
        if not any(name.startswith(prefix) for name in reading)
    )
    assert not stale, (
        f"arguments for clocks no test reads: {stale}. An excuse must not "
        "outlive the thing it excuses."
    )
