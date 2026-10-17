"""Name the Rust files whose code a change touches, one per line.

The push sweeps mutate the files a change touches, whole, so an edit that stops
an existing test from killing a mutant elsewhere in the file is found. A file
whose diff touches only comments and blank lines compiles to what it compiled
to before, so every mutant in it answers as it did on the last sweep of it,
and sweeping it again costs a sweep and finds nothing: a comment edit to a
large file is the whole file's mutants on the push. Such a file is left out.

A line is code unless, stripped, it is empty or a `//` comment -- doc comments
included, which document an item and compile to nothing a mutant reads. A
line inside a block comment counts as code, which sweeps a file more often
than it needs to and never less.

A test module is not where its mutants are. cargo-mutants mutates no
`#[cfg(test)]` file, so a file holding only tests -- a long `tests.rs`, an
`interpreter.rs` corpus -- has nothing to sweep, and the edit that matters is
one that stops it killing a mutant in the code it holds: the file that
declares it and the files beneath that module. Those are named in its place.
A suite the crate root declares holds the whole crate, which no push sweeps,
so it names nothing, and the nightly sweep alone judges it.

The diff is asked for in one shape whatever git is configured to print. A
rename is an added file, so a file moved into sweep scope is swept, where git
would print the move and none of its lines; the paths carry `a/` and `b/`,
which `diff.noprefix` and `diff.mnemonicPrefix` would take away or change; and
no external diff driver runs. A path that still arrives in another shape is
not read as no change: the script exits 2.

Usage:
    python scripts/code_changes.py BASE

Exit 0 with the files on stdout, deleted files left out; 2 when git cannot
answer, or answers in a shape the script cannot read.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _lib import EXIT_CANNOT_RUN, EXIT_OK, arguments, git_output

#: The checkout this is asked about: the one it is started in, which is the
#: lane's checkout and, under test, a history built for the question.
CHECKOUT = Path()


def is_code(line: str) -> bool:
    """Whether a changed line is code, as against a comment or a blank."""
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("//")


class UnreadableDiffError(ValueError):
    """A diff whose paths are not in the shape the script asked git for."""


def code_changes(diff: str) -> list[str]:
    """Name each file of a `git diff -U0` whose added or removed lines hold code.

    A `+++` line naming neither `b/<path>` nor `/dev/null` raises
    `UnreadableDiffError`: the shape was asked for, so a path in another one is a
    diff this cannot read, not a file with nothing in it.
    """
    found: list[str] = []
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current = None
        elif line.startswith("+++ "):
            path = line[4:]
            if path != "/dev/null" and not path.startswith("b/"):
                message = f"a path in a shape this does not read: {line!r}"
                raise UnreadableDiffError(message)
            current = path[2:] if path.startswith("b/") else None
        elif line.startswith("--- "):
            continue
        elif (
            current is not None
            and line[:1] in {"+", "-"}
            and is_code(line[1:])
            and current not in found
        ):
            found.append(current)
    return found


#: An attribute that makes the module below it test code: `#[cfg(test)]`, alone
#: or inside an `all(...)`, as an interpreter corpus is declared.
_TEST_ONLY = re.compile(r"#\[cfg\((?:all\()?test\b")


def _declarer(path: Path) -> Path | None:
    """Name the file that declares the module `path` is, `lib.rs` for a top one.

    A module lives under a crate's `src/`; a file anywhere else is declared by
    nothing, and is no module.
    """
    if "src" not in path.parts[:-1]:
        return None
    parent = path.parent
    if parent.name == "src":
        root = parent / "lib.rs"
        return root if root.is_file() and root != path else None
    for candidate in (parent.with_suffix(".rs"), parent / "mod.rs"):
        if candidate.is_file() and candidate != path:
            return candidate
    return None


def is_test_module(path: Path) -> bool:
    """Whether the file that declares `path` declares it under `#[cfg(test)]`."""
    declarer = _declarer(path)
    if declarer is None:
        return False
    found = re.search(
        r"((?:^[ \t]*(?:#\[[^\n]*\]|//[^\n]*)[ \t]*\n)*)"
        rf"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?mod\s+{re.escape(path.stem)}\s*;",
        declarer.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return found is not None and _TEST_ONLY.search(found.group(1)) is not None


def held_code(path: Path) -> list[Path]:
    """Name the code a test module holds, which is what a change to it can move.

    The file that declares it, and every file beneath the module that file is
    that is not a test module itself. A suite the crate root declares holds
    the whole crate, and none is named for it.
    """
    declarer = _declarer(path)
    if declarer is None or (
        declarer.parent.name == "src" and path.parent.name == "src"
    ):
        return []
    module = declarer.parent if declarer.name == "mod.rs" else declarer.with_suffix("")
    beneath = sorted(
        found
        for found in module.rglob("*.rs")
        if found != declarer and not is_test_module(found)
    )
    return [declarer, *beneath]


def swept(changed: list[str], checkout: Path) -> list[str]:
    """Name what a push sweeps for the files whose code changed, in order.

    A test module gives way to the code it holds; every other file is itself.
    """
    named: list[str] = []
    for path in changed:
        source = checkout / path
        held = (
            [
                str(found.relative_to(checkout)).replace("\\", "/")
                for found in held_code(source)
            ]
            if source.is_file() and is_test_module(source)
            else [path]
        )
        named.extend(found for found in held if found not in named)
    return named


def main(argv: list[str]) -> int:
    parser = arguments(__doc__)
    parser.add_argument("base", help="the commit the change is measured from")
    args = parser.parse_args(argv)
    diff = git_output(
        "diff",
        "-U0",
        "--no-color",
        "--no-ext-diff",
        "--no-renames",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--diff-filter=d",
        f"{args.base}...HEAD",
        "--",
        "*.rs",
        cwd=CHECKOUT,
    )
    if diff is None:
        print(f"code_changes: git cannot diff from {args.base}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    try:
        changed = code_changes(diff)
    except UnreadableDiffError as error:
        print(f"code_changes: {error}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    for path in swept(changed, CHECKOUT):
        print(path)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
