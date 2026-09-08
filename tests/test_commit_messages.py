"""A commit message names nothing a reader outside this tree can't open.

The maintainer's working area is gitignored, so the notes and plans kept there --
the audit write-ups, the iteration files, the milestone ladder -- reach nobody
who clones this repository. A tracked page naming one is caught by the docs
lint. A *commit message* naming one was caught by nothing, and the history is
the part of this project that travels furthest: it is read on a forge, in a
bisect, and in a release note, by people who have none of those files and no way
to get them.

The failure is not that the reference is private. It is that the sentence stops
being an argument. "This was planned as a fix" says what happened; naming a note
that planned it says only that somebody once wrote it down. A message has to
carry its own reason, because the reason is the whole of what a message is for.

Held over every commit the current branch adds to the released tag, which is the
range a contributor can still amend.

The names are assembled from their pieces below rather than written out, for the
reason this file exists: a check that spells the thing it refuses fails itself.

LEDGER: no commit message names the internal working area
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# A repository check: it reads the history, which ships in no wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent

#: The working area's own name, and the notes and codes kept under it, in the
#: forms a message would write them.
AREA = "__" + "DEV"
NOTE = "REPORT" + "-[0-9]+"
ITER = "ITERATION" + "-[0-9]+"
PAGES = r"\b[0-9]-(?:MILESTONES|THEORY|REFERENCES|PROJECT)\b|\b00-CONTRACT\b"
PROMPT = r"\bPROMPT\.md\b"
MILESTONE = r"\bM[0-9]+(?:\.[0-9]+)?\b"

INTERNAL = re.compile("|".join((NOTE, ITER, re.escape(AREA), PAGES, PROMPT, MILESTONE)))

#: The tag the branch is measured from. A commit at or below it is released, so
#: rewriting it is not a repair a contributor makes.
RELEASED = "v0.0.9"


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(ROOT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout


def _unreleased() -> list[str]:
    """Every commit this branch adds to the released tag, or none if it is gone.

    A shallow clone, or one fetched without tags, carries no tag to measure
    from. That is a shape of checkout rather than a defect in the history, so
    the check skips rather than inventing a range.
    """
    if not _git("tag", "--list", RELEASED).strip():
        return []
    return _git("log", "--format=%H", f"{RELEASED}..HEAD").split()


def test_no_commit_message_names_the_working_area() -> None:
    commits = _unreleased()
    if not commits:
        pytest.skip(f"no {RELEASED} tag in this checkout, so there is no range to read")

    named = []
    for sha in commits:
        message = _git("log", "-1", "--format=%B", sha)
        subject = _git("log", "-1", "--format=%s", sha).strip()
        named.extend(
            f"{sha[:7]} names {name}: {subject}"
            for name in sorted(set(INTERNAL.findall(message)))
        )

    assert not named, (
        "a commit message names the internal working area, which no reader "
        "outside this tree can open; the sentence has to carry its own reason "
        "instead:\n  " + "\n  ".join(named)
    )


def test_the_names_it_refuses_are_the_ones_it_matches() -> None:
    """The pattern, held to both directions by example.

    Without this the check could stop matching and still pass, which is the
    shape of a ledger that cannot fail.
    """
    for names_one in (
        f"{'REPORT'}-31 asked for the closed core",
        f"the plan in {'REPORT'}-35 section 5",
        "what M19 left open",
        "M33.2's exit criterion",
        "recorded in 2-MILESTONES",
        "the workflow in PROMPT.md",
        f"see the notes under {AREA}/",
        f"{'ITERATION'}-74 records the slice",
    ):
        assert INTERNAL.search(names_one), names_one

    for stands_alone in (
        "the decision procedure asked for a closed core",
        "the lowering left the class representation open",
        "the exit criterion was not met, and the number says so",
        "the release checklist records it",
        "a bound over floats is a set the descriptor can hold",
    ):
        assert not INTERNAL.search(stands_alone), stands_alone
