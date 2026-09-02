"""A test the mutation sweep skips must be marked in its own source.

Three Rust tests exist to prove a bound -- the decision budget, the recursion
depth. A mutation that removes the bound makes each of them run without end, so
the whole sweep returns no verdict for that mutant: a rig fault, not a
detection. Each therefore leaves the *sweep* while staying in the test lane,
where it runs on every push.

That is a hole in what the sweep can judge, so it is held to the tree in both
directions rather than living as three strings in a workflow:

* a test marked ``SWEEP-SKIP`` that no sweep skips fails, so a marker cannot be
  written and forgotten;
* a ``--skip`` naming no marked test fails, so the list cannot outlive the test
  it excuses, or name one that never needed excusing.

LEDGER: every SWEEP-SKIP mark is skipped; every skip is marked
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
MARKER = "SWEEP-SKIP"


def _marked_tests() -> set[str]:
    """Every Rust test whose source carries the marker, read from the sources."""
    marked: set[str] = set()
    for path in (ROOT / "crates").rglob("*.rs"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if MARKER not in line:
                continue
            # The marker sits above the test it excuses; the next `fn` is it.
            for follower in lines[i : i + 12]:
                found = re.match(r"\s*fn ([a-z0-9_]+)\(", follower)
                if found:
                    marked.add(found.group(1))
                    break
    return marked


def _skipped_tests() -> set[str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    return set(re.findall(r"--skip ([a-z0-9_]+)", text))


def test_every_marked_test_is_actually_skipped() -> None:
    marked = _marked_tests()
    # The scan is the detector: no markers at all would pass this file having
    # checked nothing.
    assert marked, "no SWEEP-SKIP marker found in any Rust source"

    unskipped = sorted(marked - _skipped_tests())
    assert not unskipped, (
        f"tests marked {MARKER} that no sweep skips: {unskipped}. "
        "Add each to the sweep's --skip list, or drop the marker."
    )


def test_every_skip_names_a_marked_test() -> None:
    skipped = _skipped_tests()
    assert skipped, "no --skip found in the workflow"

    unmarked = sorted(skipped - _marked_tests())
    assert not unmarked, (
        f"--skip names tests that carry no {MARKER} marker: {unmarked}. "
        "A skip is a hole in the sweep; mark the test with its reason, or "
        "remove the skip."
    )


def test_the_marker_carries_a_reason() -> None:
    # A marker with no argument beside it is a hole nobody can review.
    for path in (ROOT / "crates").rglob("*.rs"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if MARKER in line:
                assert len(line.split(MARKER, 1)[1].strip(" :")) > 20, (
                    f"{path.relative_to(ROOT)}: a {MARKER} marker with no reason"
                )


def test_the_documented_rerun_command_carries_the_same_skips() -> None:
    """The command a contributor is told to run must be the command CI runs.

    The inventory in ``CONTRIBUTING.md`` names a rerun command per contract, and
    its whole value is that following it reproduces that one verdict. A sweep
    command without these skips does not: the skipped cases exist to prove a
    bound, so a mutation removing the bound runs without end and the sweep
    returns no verdict at all -- a rig fault the reader then reads as a finding.

    Held here rather than beside the table because this is the same list
    ``ci.yml`` carries, and a second copy of a list is how the first one drifts.
    """
    rows = [
        line
        for line in CONTRIBUTING.read_text(encoding="utf-8").splitlines()
        if "cargo mutants" in line
    ]
    assert rows, "the contract inventory names no mutation sweep"

    documented = set(re.findall(r"--skip ([a-z0-9_]+)", "\n".join(rows)))
    missing = sorted(_skipped_tests() - documented)
    assert not missing, (
        f"the documented sweep command omits skips CI passes: {missing}. "
        "Following it returns no verdict for those mutants."
    )


def test_the_skips_are_few() -> None:
    # Not a gate on the number so much as on the direction: every skip is a
    # mutant the sweep cannot judge, and a list that grows quietly is how a
    # sweep stops measuring what it claims to.
    assert len(_marked_tests()) <= 5, (
        f"{len(_marked_tests())} tests now leave the sweep; each is a hole in "
        "what it can judge. Re-read whether the bound can be proved another way."
    )
