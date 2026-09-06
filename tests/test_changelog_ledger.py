"""A change a user can see, with no line in the changelog, is a silent release.

Nine of them accumulated once: every argument became positional-only, `repr`
changed for three forms, twenty-five relations flipped, constants started
pooling by value -- and the changelog recorded four other things. Each commit
was reviewed, each was described in its own message, and nobody was asked
whether the release notes had grown a line. Reviewing a diff and writing the
notes are two jobs, and only one of them has a gate.

So the unreleased section carries a roll of the commits it accounts for, and
this holds that roll to `git log` in both directions. The universe is the
commits typed `feat` or `fix`, because those are the two conventional types
that mean a caller can see the change, and they are what a version number is
read off. Scoping by *path* was tried and is wrong: the descriptor lives in the
core crate and flipping twenty-five relations is as visible as anything the
bindings do.

A `perf` or `refactor` commit that a caller can see is a mislabelled commit, and
this ledger makes that visible rather than papering over it: relabel it, or the
change it carries goes unrolled.

Being on the roll is not a claim that a commit earned a paragraph. It is a
claim that somebody looked at it and decided, which is the step that was
missing. A commit that deserves no line stays on the roll with `-- internal`
after it.

A shallow clone cannot answer any of this: it carries the tip and no tags, so
there is no range to read. That is a property of the clone rather than of the
tree, so the checks skip there -- and because a ledger that skips in every lane
is a ledger nobody runs, one lane takes the whole history and a fourth check
holds it there.

The roll expires in its own direction too: an entry naming a commit that is not
in the range is a stale line, and it fails, because a roll nobody prunes is a
roll nobody reads.

LEDGER: every feat/fix commit since the last release is on the changelog roll
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"

# The conventional types that mean a caller can see the change. `docs`, `test`,
# `ci`, `build`, `refactor` and `perf` are the types that promise they cannot.
VISIBLE = re.compile(r"^(feat|fix)(\(|!|:)")

# The roll: one `- <subject>` line per accounted-for commit, inside the
# unreleased section, written as an HTML comment so it does not render.
ROLL = re.compile(
    r"<!--\s*changelog-roll\s*\n(.*?)-->",
    re.DOTALL,
)
UNRELEASED = re.compile(r"^## \[Unreleased\]$(.*?)^## \[", re.DOTALL | re.MULTILINE)


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", *args],  # noqa: S607 - git resolved from PATH, as every lane does
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _last_tag() -> str | None:
    """Read the last release tag, or `None` where this clone cannot see one.

    A CI checkout is shallow by default: it carries the tip and no tags, so
    there is no range to read and `git describe` exits non-zero. That is a
    property of the *clone*, not of the tree, so it skips rather than fails --
    and one lane takes the full history so the ledger is actually run
    (`test_the_workflow_runs_this_ledger_with_full_history` below holds it
    there).
    """
    try:
        return _git("describe", "--tags", "--abbrev=0").strip() or None
    except (subprocess.CalledProcessError, OSError):
        return None


LAST_TAG = _last_tag()
SHALLOW = pytest.mark.skipif(
    LAST_TAG is None,
    reason="this clone has no history back to the last release tag",
)


def _visible_commits() -> set[str]:
    """Subjects of the `feat`/`fix` commits since the last tag.

    Keyed by subject rather than by hash, because a subject is what a reader of
    the roll matches against a changelog entry, and a hash survives no rebase.
    """
    raw = _git("log", "--format=%s", f"{LAST_TAG}..HEAD")
    return {line for line in raw.splitlines() if VISIBLE.match(line)}


def _roll() -> list[str]:
    section = UNRELEASED.search(CHANGELOG.read_text(encoding="utf-8"))
    assert section, "the changelog has no [Unreleased] section"
    block = ROLL.search(section.group(1))
    assert block, (
        "the [Unreleased] section carries no `<!-- changelog-roll` block; add one "
        "listing every feat/fix commit since the last release"
    )
    return [
        line.strip()[2:].split(" -- ")[0].strip()
        for line in block.group(1).splitlines()
        if line.strip().startswith("- ")
    ]


@SHALLOW
def test_every_visible_commit_is_on_the_roll() -> None:
    missing = sorted(_visible_commits() - set(_roll()))
    assert not missing, (
        "feat/fix commits with no line on the changelog roll: "
        + "; ".join(missing)
        + ". Add a changelog entry and put the subject on the roll, or put it on "
        "the roll with `-- internal` if a caller cannot see it after all."
    )


@SHALLOW
def test_no_roll_entry_is_stale() -> None:
    extra = sorted(set(_roll()) - _visible_commits())
    assert not extra, (
        "roll entries naming no commit in this release: "
        + "; ".join(extra)
        + ". A released roll is emptied with the section; a rebase renames "
        "subjects and the roll follows."
    )


@SHALLOW
def test_the_roll_is_not_empty_while_the_surface_moves() -> None:
    # The detector: an empty roll beside a moved surface passes both checks
    # above having read nothing, which is how a ledger stops being one.
    if _visible_commits():
        assert _roll(), "the surface moved this release and the roll is empty"


def test_the_workflow_runs_this_ledger_with_full_history() -> None:
    """One lane reads the whole history, or this ledger runs nowhere.

    Every check above skips on a shallow clone, and a ledger that skips in every
    lane is a ledger nobody runs -- which is the failure mode this file exists to
    stop, one level up. So the lane that carries the project's own audit checks
    out with `fetch-depth: 0`, and this holds it there.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "fetch-depth: ${{ (matrix.os == 'ubuntu-latest'" in workflow, (
        "the python lane no longer takes the full history on its floor leg, so "
        "the changelog ledger skips in every lane that runs it"
    )
