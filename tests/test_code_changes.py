"""A push sweep reads the files whose code a change touches.

`scripts/code_changes.py` names them for both diff-scoped mutation sweeps. A
file whose diff is comments and blank lines alone compiles to what it compiled
to before, so every mutant in it answers as it did, and sweeping it again
costs the whole file's mutants and finds nothing. The script is driven over a
synthetic history rather than this tree, so the answer does not depend on the
clone the suite runs in.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from _tree import ROOT, load_script, read_workflow

if TYPE_CHECKING:
    from pathlib import Path

# The repository checks are not the product suite: this file drives a CI
# script against a history of its own, none of which ships in a wheel.
pytestmark = pytest.mark.repository

SCRIPT = ROOT / "scripts" / "code_changes.py"

script = load_script("code_changes")

#: A clone inherits no committer, and a commit here must not depend on whether
#: the machine running the suite has one configured.
IDENTITY = {
    "GIT_AUTHOR_NAME": "sweep",
    "GIT_AUTHOR_EMAIL": "sweep@example.invalid",
    "GIT_COMMITTER_NAME": "sweep",
    "GIT_COMMITTER_EMAIL": "sweep@example.invalid",
}


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(tree), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **IDENTITY},
    ).stdout.strip()


def test_a_comment_or_a_blank_line_is_not_code() -> None:
    assert not script.is_code("")
    assert not script.is_code("    ")
    assert not script.is_code("    // a comment")
    assert not script.is_code("/// a doc comment")
    assert not script.is_code("//! a module doc")
    assert script.is_code("    let x = 1; // and a comment")
    assert script.is_code("/* a block comment is read as code */")


def test_a_file_whose_diff_is_comments_alone_is_left_out(tmp_path: Path) -> None:
    """Of three edited files, the one whose edit is a comment is not named.

    One file changes a line of code, one a comment and a blank line, and one
    both; the second compiles as it did and is the one left out.
    """
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    source = "// a\nfn f() -> u8 {\n    1\n}\n"
    for name in ("code.rs", "comment.rs", "both.rs"):
        (tmp_path / name).write_text(source, encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    edits = {
        "code.rs": source.replace("    1", "    2"),
        "comment.rs": source.replace("// a\n", "// b\n\n"),
        "both.rs": source.replace("// a", "// b").replace("    1", "    3"),
    }
    for name, text in edits.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    _git(tmp_path, "commit", "--quiet", "-am", "change")
    named = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, str(SCRIPT), base],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert named.returncode == script.EXIT_OK, named.stderr
    assert sorted(named.stdout.split()) == ["both.rs", "code.rs"]


def _history(tree: Path) -> str:
    """Lay down a history of one commit holding two files, and name its tip."""
    _git(tree, "init", "--quiet", "--initial-branch=main")
    for name in ("kept.rs", "moved.rs"):
        (tree / name).write_text("fn f() -> u8 {\n    1\n}\n", encoding="utf-8")
    _git(tree, "add", ".")
    _git(tree, "commit", "--quiet", "-m", "base")
    return _git(tree, "rev-parse", "HEAD")


def _named(tree: Path, base: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, str(SCRIPT), base],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_renamed_file_is_named_under_its_new_path(tmp_path: Path) -> None:
    """A move changes no line, and still puts the file where a sweep reads it.

    Git prints a rename as the move alone, with no `+++` line and none of the
    file's lines, so a file moved into sweep scope -- out of a test module's
    declaration, say -- read as no change.
    """
    base = _history(tmp_path)
    _git(tmp_path, "mv", "moved.rs", "renamed.rs")
    _git(tmp_path, "commit", "--quiet", "-m", "move")
    named = _named(tmp_path, base)
    assert named.returncode == script.EXIT_OK, named.stderr
    assert named.stdout.split() == ["renamed.rs"]


@pytest.mark.parametrize(
    "setting",
    [("diff.noprefix", "true"), ("diff.mnemonicPrefix", "true")],
    ids=["noprefix", "mnemonic"],
)
def test_the_diff_reads_alike_whatever_git_prints(
    tmp_path: Path, setting: tuple[str, str]
) -> None:
    """A configuration that changes the paths git prints changes nothing here.

    Under `diff.noprefix` every path lost its `b/`, and the script named no file
    and exited 0: a sweep of nothing that read as a clean one.
    """
    base = _history(tmp_path)
    _git(tmp_path, "config", *setting)
    (tmp_path / "kept.rs").write_text("fn f() -> u8 {\n    2\n}\n", encoding="utf-8")
    _git(tmp_path, "commit", "--quiet", "-am", "change")
    named = _named(tmp_path, base)
    assert named.returncode == script.EXIT_OK, named.stderr
    assert named.stdout.split() == ["kept.rs"]


def test_a_path_in_another_shape_is_unreadable_rather_than_unchanged() -> None:
    """A `+++` line without the prefix asked for is not read as no file."""
    diff = "diff --git kept.rs kept.rs\n--- kept.rs\n+++ kept.rs\n@@ -2 +2 @@\n+    2\n"
    with pytest.raises(script.UnreadableDiffError):
        script.code_changes(diff)
    added = (
        "diff --git a/n.rs b/n.rs\n--- /dev/null\n+++ b/n.rs\n"
        "@@ -0,0 +1 @@\n+fn f() {}\n"
    )
    assert script.code_changes(added) == ["n.rs"]


def test_a_base_git_cannot_read_is_not_an_empty_sweep(tmp_path: Path) -> None:
    """A diff that fails exits 2: a sweep of nothing would read as a clean one."""
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    _git(tmp_path, "commit", "--quiet", "--allow-empty", "-m", "base")
    named = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, str(SCRIPT), "no-such-commit"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert named.returncode == script.EXIT_CANNOT_RUN


def test_both_push_sweeps_list_their_files_through_it() -> None:
    """Both diff-scoped sweeps read the script, so neither sweeps a comment."""
    listings = [
        step["run"]
        for job in read_workflow()["jobs"].values()
        for step in job.get("steps", [])
        if "> changed.txt" in str(step.get("run", ""))
    ]
    assert len(listings) == 2, listings
    for run in listings:
        assert 'python3 scripts/code_changes.py "$base" > changed.txt' in run, run
        assert "git diff --name-only" not in run, run
