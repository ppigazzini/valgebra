"""A commit a tracked file names is a commit a clone can reach.

A file that records *which build* a number, a corpus or a decision came from is
recording provenance, and provenance nobody can resolve is a claim about
nothing. This tree amends a commit that breaks a lane rather than following it
with a fix, and every such rewrite orphans the commits it replays -- so a hash
written into a tracked file goes unreachable without anything saying so. A
reference corpus naming the commit it was taken at is the shape most exposed to
it, because nothing else reads that name.

So every commit-shaped string in a tracked file is held to the one property
that makes it worth writing down: it resolves, and it is an ancestor of the
branch. A string that resolves to nothing is left alone -- a hash in a lockfile
is not a commit, and this cannot tell the two apart except by asking git, which
is what it does.

The check runs only where the whole history is in the clone: a shallow checkout
carries the tip and cannot answer reachability, and answering "unreachable"
there would be a property of the clone rather than of the tree.

A **release tag** is the same claim by another name, and the one a rewrite
costs most. `v0.0.11` says a release was cut at a commit, the changelog roll is
measured from it, and a rebase below it leaves the tag pointing at a commit no
branch reaches -- so the tag still resolves, `git show` still prints it, and
nothing in a diff of the tree is different. It happened here: two tags were
orphaned by an amend two hundred commits down, and what noticed was a ledger
failing on the *changelog*, three steps away from the cause.

LEDGER: every commit a tracked file cites is an ancestor of this branch
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent

#: A commit-shaped string: git's own abbreviation floor is 7, and 40 is a full
#: object name. Longer runs of hex are something else -- a wheel's hash, a
#: base16 payload -- and are not asked about.
_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{7,40}(?![0-9a-fA-F])")

#: Files whose hex is never a commit, and which are large enough that asking
#: git about every run of it costs more than it says. Lock files record
#: *content* hashes by the thousand.
_NOT_COMMITS = ("uv.lock", "Cargo.lock", ".png", ".ico", ".svg")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git in this tree, under an identity of this file's own.

    `commit-tree` refuses without a committer, and a runner has none
    configured: the plant below wrote nothing there and the row failed for the
    machine's git config rather than for anything about the tree.
    `tests/test_ledger_plants.py` passes an identity for the same reason.
    """
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", *args],  # noqa: S607 - git resolved from PATH, as every lane does
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "plant",
            "GIT_AUTHOR_EMAIL": "plant@example.invalid",
            "GIT_COMMITTER_NAME": "plant",
            "GIT_COMMITTER_EMAIL": "plant@example.invalid",
        },
    )


def _is_shallow() -> bool:
    return _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"


SHALLOW = pytest.mark.skipif(
    _is_shallow(), reason="a shallow clone cannot answer reachability"
)


def _tracked() -> list[Path]:
    listed = _git("ls-files", "-z").stdout.split("\0")
    return [ROOT / name for name in listed if name and not name.endswith(_NOT_COMMITS)]


def _cited() -> dict[str, list[str]]:
    """Every commit-shaped string in a tracked file, by the string."""
    found: dict[str, list[str]] = {}
    for path in _tracked():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # a binary file cites nothing
        for match in _HEX.findall(text):
            found.setdefault(match, []).append(path.name)
    return found


def test_the_scan_reads_the_tree_at_all() -> None:
    # A scan that finds no files makes the assertion below vacuous, which is
    # the failure mode a ledger has: it passes loudest when it is broken.
    tracked = _tracked()
    assert len(tracked) > 100, f"the tracked-file scan found only {len(tracked)}"
    assert any(path.name == "metamorphic_reference.json" for path in tracked)


@SHALLOW
def test_every_cited_commit_is_reachable() -> None:
    orphaned: list[str] = []
    for text, files in sorted(_cited().items()):
        if _git("rev-parse", "--verify", "--quiet", f"{text}^{{commit}}").returncode:
            continue  # not a commit: a content hash, a colour, an id
        if _git("merge-base", "--is-ancestor", text, "HEAD").returncode:
            orphaned.append(f"{text} (in {', '.join(sorted(set(files)))})")
    assert not orphaned, (
        "these files name commits that are not ancestors of this branch:\n  "
        + "\n  ".join(orphaned)
        + "\nA rewrite orphans every commit it replays. Re-record the file "
        "against a commit this branch carries, or cite the change by its "
        "subject, which survives a rebase."
    )


@SHALLOW
def test_the_check_would_see_an_orphan() -> None:
    """The plant: an unreachable commit is refused rather than skipped.

    Built rather than searched for: an empty commit is written with no parent
    and never referenced, which is exactly the shape a rewrite leaves behind.
    """
    tree = _git("write-tree").stdout.strip()
    orphan = _git("commit-tree", tree, "-m", "an orphan, for this row").stdout.strip()
    assert orphan, "the plant could not write a commit"
    resolved = _git("rev-parse", "--verify", "--quiet", f"{orphan}^{{commit}}")
    assert not resolved.returncode, "the plant's commit does not resolve"
    assert _git("merge-base", "--is-ancestor", orphan, "HEAD").returncode, (
        "a commit no ref reaches read as an ancestor, so the check above cannot fail"
    )


def _orphaned(refs: dict[str, str]) -> list[str]:
    """Give the named commits this branch does not reach.

    Takes the pairs rather than reading them, so the row below can ask it about
    a commit built for the purpose: a plant that tagged the real repository
    would leave a ref behind, and one that only asserted the predicate would be
    asking about nothing.
    """
    return sorted(
        f"{name} -> {commit[:12]}"
        for name, commit in refs.items()
        if _git("merge-base", "--is-ancestor", commit, "HEAD").returncode
    )


def _release_tags() -> dict[str, str]:
    """Read every `v*` tag, with the commit it names.

    Resolved through `^{commit}` because an annotated tag names a tag object,
    and it is the commit underneath that a branch does or does not reach.
    """
    listed = _git("tag", "--list", "v*").stdout.split()
    found: dict[str, str] = {}
    for tag in listed:
        resolved = _git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
        if not resolved.returncode:
            found[tag] = resolved.stdout.strip()
    return found


@SHALLOW
def test_every_release_tag_is_an_ancestor_of_this_branch() -> None:
    """A rewrite below a tag orphans a release, and no diff of the tree shows it.

    The tag keeps resolving and keeps printing, so every check that reads the
    *content* of the tree passes. What changes is reachability, which only this
    asks about.
    """
    tags = _release_tags()
    # The scan is the detector: no tags is no claim, which would pass silently.
    assert len(tags) >= 5, sorted(tags)
    orphaned = _orphaned(tags)
    assert not orphaned, (
        "release tags this branch does not reach:\n  "
        + "\n  ".join(orphaned)
        + "\nA rewrite below a tag orphans the release it names. Reset to the "
        "commit that carries them and land the change above the last one."
    )


@SHALLOW
def test_the_check_would_see_an_orphaned_tag() -> None:
    """The plant: a tag on a commit no branch reaches is refused.

    Asked of a commit written for the row rather than of a tag written into the
    repository, because a plant that left a ref behind would be a change to the
    tree rather than a test of it.
    """
    tree = _git("write-tree").stdout.strip()
    orphan = _git("commit-tree", tree, "-m", "an orphan, for this row").stdout.strip()
    assert orphan, "the plant could not write a commit"
    assert _orphaned({"v9.9.9": orphan}) == [f"v9.9.9 -> {orphan[:12]}"]
    # And the same predicate says nothing about a tag the branch does reach.
    assert not _orphaned({"HEAD": _git("rev-parse", "HEAD").stdout.strip()})
