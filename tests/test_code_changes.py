"""A push sweep reads the functions whose code a change touches.

`scripts/code_changes.py` names them for both diff-scoped mutation sweeps: the
files, and a pattern each that picks the mutants of the touched functions out
of the package's listing. A change of comments and blank lines alone compiles
to what it compiled to before, so every mutant answers as it did, and sweeping
it again finds nothing. The script is driven over a synthetic history and a
listing written for it rather than this tree, so the answer does not depend on
the clone the suite runs in or on cargo-mutants being installed.
"""

from __future__ import annotations

import json
import os
import re
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


def _listing(tree: Path) -> list[dict]:
    """List a mutant on each line of each function of the tree's Rust files.

    In the shape `cargo mutants --list --json` prints: a function starts at a
    line opening with `fn` and ends at the first line that is `}`, or on its
    own line where it closes there. A line outside every function holds none.
    """
    listed = []
    for source in sorted(tree.rglob("*.rs")):
        path = str(source.relative_to(tree)).replace("\\", "/")
        lines = source.read_text(encoding="utf-8").splitlines()
        for start, text in enumerate(lines, 1):
            if not text.startswith("fn "):
                continue
            end = start
            while not lines[end - 1].endswith("}"):
                end += 1
            span = {"start": {"line": start}, "end": {"line": end}}
            name = text.split("(")[0].removeprefix("fn ")
            listed.extend(
                {
                    "file": path,
                    "function": {"function_name": name, "span": span},
                    "span": {"start": {"line": line}, "end": {"line": line}},
                }
                for line in range(start, end + 1)
            )
    return listed


def _named(tree: Path, base: str) -> subprocess.CompletedProcess[str]:
    """Run the script over the tree's own listing, as the lanes run it."""
    (tree / "listed.json").write_text(json.dumps(_listing(tree)), encoding="utf-8")
    return subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [
            sys.executable,
            str(SCRIPT),
            base,
            "--listing",
            str(tree / "listed.json"),
            "--examined",
            str(tree / "examined.txt"),
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
    )


def _examined(tree: Path) -> list[str]:
    """Name each mutant of the tree's listing a pattern the script wrote picks."""
    patterns = (tree / "examined.txt").read_text(encoding="utf-8").split()
    return [
        f"{mutant['file']}:{mutant['span']['start']['line']}"
        for mutant in _listing(tree)
        if any(
            re.match(pattern, f"{mutant['file']}:{mutant['span']['start']['line']}:1:")
            for pattern in patterns
        )
    ]


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
    named = _named(tmp_path, base)
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
    assert list(script.code_changes(added)) == ["n.rs"]


#: A crate laid out as this tree lays its own: a suite the root declares, a
#: module with a test module and a corpus beside a file of code beneath it.
CRATE = {
    "crates/c/src/lib.rs": "mod m;\n#[cfg(test)]\nmod laws;\n",
    "crates/c/src/laws.rs": "fn law() {}\n",
    "crates/c/src/m.rs": (
        "mod inner;\n"
        "#[cfg(test)]\n"
        "mod tests;\n"
        "// Needs a live interpreter.\n"
        '#[cfg(all(test, feature = "interpreter-tests"))]\n'
        "mod interpreter;\n"
        "fn f() -> u8 {\n    1\n}\n"
    ),
    "crates/c/src/m/inner.rs": "fn g() -> u8 {\n    2\n}\n",
    "crates/c/src/m/tests.rs": "fn t() {}\n",
    "crates/c/src/m/interpreter.rs": "fn i() {}\n",
}


def _laid_down(tree: Path, files: dict[str, str]) -> str:
    """Commit `files` as a history's first commit, and name it."""
    _git(tree, "init", "--quiet", "--initial-branch=main")
    for name, text in files.items():
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(text, encoding="utf-8")
    _git(tree, "add", ".")
    _git(tree, "commit", "--quiet", "-m", "base")
    return _git(tree, "rev-parse", "HEAD")


#: Three functions and a constant outside them, one mutant a line in each.
FUNCTIONS = (
    "const N: u8 = 1;\n"
    "fn f() -> u8 {\n    1\n}\n"
    "fn g() -> u8 {\n    2\n}\n"
    "fn h() -> u8 {\n    3\n    4\n}\n"
)


def test_a_change_sweeps_the_functions_it_touches(tmp_path: Path) -> None:
    """A change examines the mutants of the functions it touches, and no others.

    A line changed in `g` examines `g`; a line removed from `h` examines `h`,
    whose remaining lines hold the place it was; a constant outside every
    function, and a comment inside one, touch none, so neither file is named.
    """
    names = ("changed.rs", "removed.rs", "constant.rs", "comment.rs")
    base = _laid_down(tmp_path, dict.fromkeys(names, FUNCTIONS))
    edits = {
        "changed.rs": FUNCTIONS.replace("    2", "    20"),
        "removed.rs": FUNCTIONS.replace("    4\n", ""),
        "constant.rs": FUNCTIONS.replace("N: u8 = 1", "N: u8 = 2"),
        "comment.rs": FUNCTIONS.replace("    1\n", "    // one\n    1\n"),
    }
    for name, text in edits.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    _git(tmp_path, "commit", "--quiet", "-am", "change")
    run = _named(tmp_path, base)
    assert run.returncode == script.EXIT_OK, run.stderr
    assert run.stdout.split() == ["changed.rs", "removed.rs"]
    assert _examined(tmp_path) == [
        "changed.rs:5",
        "changed.rs:6",
        "changed.rs:7",
        "removed.rs:8",
        "removed.rs:9",
        "removed.rs:10",
    ]


def test_a_rewrite_sweeps_only_what_it_changed(tmp_path: Path) -> None:
    """A rewritten history measured from the tip it replaced names the amend alone.

    `root - c1 - c2` is pushed, then rewritten to `root - c1' - c2'`, with `c1'`
    changing one more line, in `h`, deep under `c2`. The trees of the two tips
    differ in that line, so `h` is examined and nothing the two commits changed
    before the amend is: not `f`, which `c1` changed, nor `g` in another file,
    which `c2` changed.
    """
    root = _laid_down(tmp_path, dict.fromkeys(("a.rs", "b.rs"), FUNCTIONS))
    first = FUNCTIONS.replace("    1\n", "    10\n")
    (tmp_path / "a.rs").write_text(first, encoding="utf-8")
    _git(tmp_path, "commit", "--quiet", "-am", "c1")
    (tmp_path / "b.rs").write_text(
        FUNCTIONS.replace("    2", "    20"), encoding="utf-8"
    )
    _git(tmp_path, "commit", "--quiet", "-am", "c2")
    before = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "reset", "--quiet", "--hard", root)
    (tmp_path / "a.rs").write_text(
        first.replace("    4\n", "    40\n"), encoding="utf-8"
    )
    _git(tmp_path, "commit", "--quiet", "-am", "c1, amended")
    (tmp_path / "b.rs").write_text(
        FUNCTIONS.replace("    2", "    20"), encoding="utf-8"
    )
    _git(tmp_path, "commit", "--quiet", "-am", "c2, replayed")
    run = _named(tmp_path, before)
    assert run.returncode == script.EXIT_OK, run.stderr
    assert run.stdout.split() == ["a.rs"]
    assert _examined(tmp_path) == ["a.rs:8", "a.rs:9", "a.rs:10", "a.rs:11"]


@pytest.mark.parametrize(
    ("changed", "named"),
    [
        ("crates/c/src/m/tests.rs", ["crates/c/src/m.rs", "crates/c/src/m/inner.rs"]),
        (
            "crates/c/src/m/interpreter.rs",
            ["crates/c/src/m.rs", "crates/c/src/m/inner.rs"],
        ),
        ("crates/c/src/laws.rs", []),
        ("crates/c/src/m/inner.rs", ["crates/c/src/m/inner.rs"]),
    ],
    ids=["a test module", "a corpus", "a crate-wide suite", "code"],
)
def test_a_test_module_gives_way_to_the_code_it_holds(
    tmp_path: Path, changed: str, named: list[str]
) -> None:
    """A test module that loses a line is swept as the code it holds, whole.

    cargo-mutants mutates no `#[cfg(test)]` file, so a change confined to one
    swept nothing, and an edit that stops it killing a mutant in the code it
    holds pushed green. A suite the crate root declares holds the whole crate,
    which no push sweeps, so it names nothing.
    """
    base = _laid_down(tmp_path, CRATE)
    path = tmp_path / changed
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("() {}", "() { 1; }")
        .replace("    2", "    3"),
        encoding="utf-8",
    )
    _git(tmp_path, "commit", "--quiet", "-am", "change")
    run = _named(tmp_path, base)
    assert run.returncode == script.EXIT_OK, run.stderr
    assert run.stdout.split() == named
    assert _examined(tmp_path) == [
        f"{mutant['file']}:{mutant['span']['start']['line']}"
        for mutant in _listing(tmp_path)
        if mutant["file"] in named
    ]


def test_a_test_module_that_only_gains_tests_names_nothing(tmp_path: Path) -> None:
    """A test added kills every mutant the module killed, so it sweeps nothing.

    A change that only adds lines to a test module adds tests: the ones it held
    still run, and a mutant they killed is still killed.
    """
    base = _laid_down(tmp_path, CRATE)
    for name in ("crates/c/src/m/tests.rs", "crates/c/src/m/interpreter.rs"):
        path = tmp_path / name
        path.write_text(
            path.read_text(encoding="utf-8") + "fn more() {}\n", encoding="utf-8"
        )
    _git(tmp_path, "commit", "--quiet", "-am", "change")
    run = _named(tmp_path, base)
    assert run.returncode == script.EXIT_OK, run.stderr
    assert run.stdout.split() == []
    assert _examined(tmp_path) == []


def test_the_tree_s_test_modules_are_read_as_test_modules() -> None:
    """The reading finds this tree's test modules, every shape of declaration."""
    for name in (
        "crates/valgebra-core/src/decision/tests.rs",
        "crates/valgebra-core/src/descr/lattice_tests.rs",
        "crates/valgebra-py/src/check/walk/interpreter.rs",
        "crates/valgebra-core/src/laws.rs",
    ):
        assert script.is_test_module(ROOT / name), name
    for name in (
        "crates/valgebra-core/src/decision/emptiness.rs",
        "crates/valgebra-py/src/check/walk/record.rs",
        "crates/valgebra-core/src/ir.rs",
    ):
        assert not script.is_test_module(ROOT / name), name


def test_a_base_git_cannot_read_is_not_an_empty_sweep(tmp_path: Path) -> None:
    """A diff that fails exits 2: a sweep of nothing would read as a clean one."""
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    _git(tmp_path, "commit", "--quiet", "--allow-empty", "-m", "base")
    assert _named(tmp_path, "no-such-commit").returncode == script.EXIT_CANNOT_RUN


def test_a_listing_it_cannot_read_is_not_an_empty_sweep(tmp_path: Path) -> None:
    """A listing that is not one exits 2, rather than examining no mutant."""
    base = _laid_down(tmp_path, {"code.rs": FUNCTIONS})
    (tmp_path / "code.rs").write_text(FUNCTIONS.replace("    2", "    20"))
    _git(tmp_path, "commit", "--quiet", "-am", "change")
    named = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [
            sys.executable,
            str(SCRIPT),
            base,
            "--listing",
            str(tmp_path / "missing.json"),
            "--examined",
            str(tmp_path / "examined.txt"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert named.returncode == script.EXIT_CANNOT_RUN, named.stdout


#: How a sweep step hands `cargo mutants` the patterns the script wrote.
EXAMINED = '"${examined[@]}"'


def test_both_push_sweeps_examine_what_it_names() -> None:
    """Both diff-scoped sweeps read the script, so neither sweeps a comment.

    Each lists the package it sweeps for the script, and hands every
    `cargo mutants` it runs -- the sweep, and the listing its shards are named
    against -- the patterns the script wrote, so a sweep examines the touched
    functions and its ratchet places their mutants among the same ones. And
    neither sweeps a dispatch: it changes nothing, and measured from the
    default branch's tip it would sweep the whole release window again.
    """
    swept = {"mutants-diff-core": "valgebra-core", "mutants-diff-walk": "valgebra-py"}
    jobs = read_workflow()["jobs"]
    for name, package in swept.items():
        steps = [str(step.get("run", "")) for step in jobs[name]["steps"]]
        (listing,) = [run for run in steps if "> changed.txt" in run]
        assert (
            f"  cargo mutants --package {package} --list --json > listed.json\n"
            '  python3 scripts/code_changes.py "$base" --listing listed.json \\\n'
            "    --examined examined.txt > changed.txt"
        ) in listing, listing
        assert "git diff --name-only" not in listing, listing
        assert (
            'if [ "$EVENT" = workflow_dispatch ]; then\n'
            "  : > changed.txt\n  : > examined.txt"
        ) in listing, listing
        (sweep,) = [run for run in steps if "--output" in run]
        assert '--re "$pattern"' in sweep, sweep
        assert "done < examined.txt" in sweep, sweep
        calls = re.findall(r"cargo mutants --package (\S+) (.*)", sweep)
        assert calls, sweep
        for called, arguments in calls:
            assert called == package, sweep
            assert EXAMINED in arguments, f"{name} runs `{arguments}` unexamined"
    (step,) = [
        step
        for step in jobs["mutants-diff-core"]["steps"]
        if "> changed.txt" in str(step.get("run", ""))
    ]
    assert step["env"]["EVENT"] == "${{ github.event_name }}", step["env"]
