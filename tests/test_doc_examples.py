"""A published example is one something runs, or one that says why not.

`scripts/run_doc_examples.py` executes every fenced `python` block it reaches,
each in its own process, so a snippet a reader copies is one this project has
run. What it reaches is a list, and the list said `docs/*.md` while the
docstring above it said "every page under docs/" -- so a page one directory
down carried an example nothing had ever executed, and the count the lane
prints was the count of what it looked at rather than of what the tree has.

The two halves, held here:

* every `python` block in a tracked page is one the runner reaches, or one
  marked with the reason it is skipped;
* every page the runner lists is a page the tree tracks, so a file that moves
  takes its entry with it rather than leaving a name that reads as covered.

The universe is globbed from `git ls-files` rather than listed, because the
direction that matters is an example arriving on a page nobody thought to
name -- which is how the one above arrived. And the blocks are found with the
runner's own pattern: a fence indented inside a list item is a block it runs,
and a checker with its own stricter pattern would call those unreached and
report a hole the tree does not have.

A skipped block says so on a comment line. The runner reads the marker
anywhere in the block, which is a word a snippet could use in its own prose or
name -- and a block skipped for that reason is one nobody meant to skip. The
rule is the comment, and `test_the_marker_is_read_as_a_comment` drives both
arms of it.

LEDGER: every python example in a tracked page is run or marked with a reason
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

# A repository check: it reads the tree's pages and the script that runs them,
# neither of which ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_doc_examples.py"


def runner() -> ModuleType:
    """Import the example runner, so its own list is what this reads.

    Restating the list here would make two lists, and the one this file could
    not see going stale is the other one.
    """
    spec = importlib.util.spec_from_file_location("valgebra_doc_examples", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tracked_pages() -> list[Path]:
    """Give every Markdown page the tree tracks.

    From `git ls-files` rather than the filesystem, so a build directory, a
    scratch copy or the gitignored working area cannot answer for a page the
    project publishes.
    """
    listed = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        # git from PATH, as every lane resolves it.
        ["git", "-C", str(ROOT), "ls-files", "-z", "*.md"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return [ROOT / name for name in listed.stdout.split("\0") if name]


def test_every_example_in_a_tracked_page_is_run_or_marked() -> None:
    module = runner()
    reached = {path.resolve() for path in module.DOCS}
    pages = _tracked_pages()
    # The glob is the detector: no pages would pass having read nothing.
    assert len(pages) >= 20, f"the tree lists only {pages}"

    unreached = []
    for page in pages:
        blocks = module.BLOCK.findall(page.read_text(encoding="utf-8"))
        if page.resolve() in reached:
            continue
        unreached += [
            f"{page.relative_to(ROOT).as_posix()} block {index}"
            for index, block in enumerate(blocks, start=1)
            if not module.planned(block)
        ]
    assert not unreached, (
        f"examples no lane runs: {unreached}. Add the page to the runner's "
        "list, or mark the block with the reason it cannot run."
    )


def test_every_page_the_runner_lists_is_one_the_tree_tracks() -> None:
    module = runner()
    tracked = {path.resolve() for path in _tracked_pages()}
    stale = sorted(
        path.relative_to(ROOT).as_posix()
        for path in module.DOCS
        if path.resolve() not in tracked
    )
    assert not stale, (
        f"pages the runner lists that the tree does not track: {stale}. A name "
        "that resolves to nothing reads as an example checked and is none."
    )


def test_the_marker_is_read_as_a_comment() -> None:
    """The word in a comment is a marker; the word in the code is code."""
    module = runner()
    assert module.planned("# PLANNED: the API this shows is not built yet\nx = 1\n")
    assert module.planned("x = 1  # PLANNED, so this is skipped\n")
    # A snippet naming the word is a snippet, and skipping it would be a hole
    # nobody wrote down: this is the shape the runner's own reading missed.
    assert not module.planned('status = "PLANNED"\nprint(status)\n')
    assert not module.planned('print("PLANNED work")\n')
    assert not module.planned("x = 1\n")


def test_the_runner_reaches_every_example_the_lane_reports() -> None:
    """The count is the tree's, so a page added is a page the number moves for.

    The lane prints what it checked, and that number is only worth reading if
    it is the number the tree has. Asserted as the two agreeing rather than as
    a figure written down here, which would be a third list to keep.
    """
    module = runner()
    reached = {path.resolve() for path in module.DOCS}
    everything = sum(
        len(module.BLOCK.findall(page.read_text(encoding="utf-8")))
        for page in _tracked_pages()
    )
    listed = sum(
        len(module.BLOCK.findall(page.read_text(encoding="utf-8")))
        for page in _tracked_pages()
        if page.resolve() in reached
    )
    assert everything == listed, (
        f"the tree has {everything} examples and the runner reaches {listed}"
    )
