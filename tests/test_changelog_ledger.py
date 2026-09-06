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
there is no range to read. Nor can the release window, where the entries have
been rolled into a dated section and the tag for it is not pushed yet. Neither
is a property of the tree, so the checks skip in both -- and because a ledger
that skips in every lane is a ledger nobody runs, one lane takes the whole
history and a fourth check holds it there.

The roll expires in its own direction too: an entry naming a commit that is not
in the range is a stale line, and it fails, because a roll nobody prunes is a
roll nobody reads. With one exception, and it is the gate's own reflection: the
commit being made is not in `git log` yet, so its line has nothing to match.
The roll is oldest first, so a pending line sits after every line that matches;
one *before* the last match names a commit that went away, and that is the stale
line this catches.

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


#: The first released section of the changelog, which names the version the
#: unreleased entries are measured against.
RELEASED = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)


def _range_start() -> str | None:
    """Read the tag the roll is measured from, or `None` if this clone lacks it.

    The changelog says which version is released -- its first dated section --
    and the roll accounts for what has happened since. Reading it from the page
    rather than from `git describe` is what makes the *release window* work: the
    bump commit rolls the unreleased entries into a dated section and the tag is
    pushed several steps later, so for that window the section names a version
    no tag resolves yet. There is then nothing to account for, and skipping is
    the honest answer rather than failing on a page that is correct.

    A shallow checkout takes the same path for the same reason: it carries the
    tip and no tags, so the range is not there to read. That is a property of
    the clone, not of the tree, and one lane takes the full history so the
    ledger is actually run (`test_the_workflow_runs_this_ledger_with_full_history`
    below holds it there).
    """
    released = RELEASED.search(CHANGELOG.read_text(encoding="utf-8"))
    if released is None:
        return None
    tag = f"v{released.group(1)}"
    try:
        _git("rev-parse", "--verify", f"{tag}^{{commit}}")
    except (subprocess.CalledProcessError, OSError):
        return None
    return tag


LAST_TAG = _range_start()
SHALLOW = pytest.mark.skipif(
    LAST_TAG is None,
    reason="the tag this roll is measured from is not in this clone",
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
    roll, commits = _roll(), _visible_commits()
    matched = [at for at, entry in enumerate(roll) if entry in commits]
    settled = roll[: max(matched) + 1] if matched else []
    extra = sorted(set(settled) - commits)
    assert not extra, (
        "roll entries naming no commit in this release: "
        + "; ".join(extra)
        + ". A released roll is emptied with the section; a rebase renames "
        "subjects and the roll follows. A line for the commit being made is not "
        "stale -- it belongs at the end, after every line that matches."
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
