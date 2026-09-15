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
