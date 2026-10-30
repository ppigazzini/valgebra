"""Name the mutants a change's push sweep examines: its files, and a pattern each.

The push sweeps mutate the functions a change touches, whole. A function holds
the edit's nearest effect: a branch added that makes an older one of the same
function unreachable leaves the older branch's mutants alive, and sweeping the
function finds them. An effect that crosses into another function -- a caller
whose branch an edit made dead -- is the nightly sweep's to find, which reads
every file whole; the push sweep is bounded by the change, so a one-line edit
to a large file costs that function's mutants rather than the file's.

A line is code unless, stripped, it is empty or a `//` comment -- doc comments
included, which document an item and compile to nothing a mutant reads. A
line inside a block comment counts as code, which sweeps a function more often
than it needs to and never less. A function is touched where a line of code is
added inside it or one is removed from it; an edit outside every function -- a
type, a constant -- touches none, and its effect is the nightly's.

A test module is not where its mutants are. cargo-mutants mutates no
`#[cfg(test)]` file, so a file holding only tests -- a long `tests.rs`, an
`interpreter.rs` corpus -- has nothing to sweep, and the edit that matters is
one that stops it killing a mutant in the code it holds: the file that
declares it and the files beneath that module, swept whole. Only an edit that
removes a line of code can do that, so a test module whose change adds lines
alone -- new tests -- sweeps nothing; an added attribute that stops a test
running is the nightly's. A suite the crate root declares holds the whole
crate, which no push sweeps, so it names nothing, and the nightly sweep alone
judges it.

The mutants are read from the listing `cargo mutants --list --json` prints for
the package the sweep examines: each with its file, its line, and the span of
the function it is in. A file with no mutant to examine is not named, so a
change that touches none skips the sweep.

The diff is asked for in one shape whatever git is configured to print. A
rename is an added file, so a file moved into sweep scope is swept, where git
would print the move and none of its lines; the paths carry `a/` and `b/`,
which `diff.noprefix` and `diff.mnemonicPrefix` would take away or change; and
no external diff driver runs. A path that still arrives in another shape is
not read as no change: the script exits 2.

Usage:
    python scripts/code_changes.py BASE --listing LISTED.json --examined OUT

Exit 0 with the files on stdout, one per line, and in `OUT` a pattern for
each, in the same order, for cargo-mutants' `--re`; deleted files are left
out. Exit 2 when git cannot answer, answers in a shape the script cannot read,
or the listing cannot be read.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from _lib import (
    EXIT_CANNOT_RUN,
    EXIT_OK,
    arguments,
    declarer,
    git_output,
    is_test_module,
)

#: The checkout this is asked about: the one it is started in, which is the
#: lane's checkout and, under test, a history built for the question.
CHECKOUT = Path()

#: A hunk's header, read for where its lines start in the new file and how many.
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def is_code(line: str) -> bool:
    """Whether a changed line is code, as against a comment or a blank."""
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("//")


class UnreadableDiffError(ValueError):
    """A diff whose paths are not in the shape the script asked git for."""


@dataclass
class Change:
    """What a change does to the code of one file.

    `lines` are the lines of the new file a change of code is at: each added
    line of code, and for a removed one the lines either side of where it was.
    `removes` is whether any line of code was removed.
    """

    lines: set[int] = field(default_factory=set)
    removes: bool = False


def code_changes(diff: str) -> dict[str, Change]:
    """Read each file of a `git diff -U0` whose added or removed lines hold code.

    A `+++` line naming neither `b/<path>` nor `/dev/null` raises
    `UnreadableDiffError`: the shape was asked for, so a path in another one is a
    diff this cannot read, not a file with nothing in it.
    """
    found: dict[str, Change] = {}
    current: str | None = None
    at = 0
    span: tuple[int, int] = (0, 0)
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
        elif hunk := HUNK.match(line):
            at = int(hunk[1])
            count = 1 if hunk[2] is None else int(hunk[2])
            # A hunk that adds nothing is placed after the line it names, so
            # what it removed stood between that line and the next.
            span = (at, at + 1) if count == 0 else (at, at + count - 1)
        elif current is not None and line[:1] == "+":
            if is_code(line[1:]):
                found.setdefault(current, Change()).lines.add(at)
            at += 1
        elif current is not None and line[:1] == "-" and is_code(line[1:]):
            change = found.setdefault(current, Change())
            change.lines.update(range(span[0], span[1] + 1))
            change.removes = True
    return found


def held_code(path: Path) -> list[Path]:
    """Name the code a test module holds, which is what a change to it can move.

    The file that declares it, and every file beneath the module that file is
    that is not a test module itself. A suite the crate root declares holds
    the whole crate, and none is named for it.
    """
    declared_by = declarer(path)
    if declared_by is None or (
        declared_by.parent.name == "src" and path.parent.name == "src"
    ):
        return []
    module = (
        declared_by.parent
        if declared_by.name == "mod.rs"
        else declared_by.with_suffix("")
    )
    beneath = sorted(
        found
        for found in module.rglob("*.rs")
        if found != declared_by and not is_test_module(found)
    )
    return [declared_by, *beneath]


def swept(changed: dict[str, Change], checkout: Path) -> dict[str, set[int] | None]:
    """Name what a push sweeps for the files whose code changed, in order.

    Each file maps to the lines whose functions it sweeps, or to `None` where
    it is swept whole: the code a test module holds, where the module lost a
    line of code. A test module that only gained lines names nothing.
    """
    named: dict[str, set[int] | None] = {}
    for path, change in changed.items():
        source = checkout / path
        if not (source.is_file() and is_test_module(source)):
            if path not in named:
                named[path] = set()
            held = named[path]
            if held is not None:
                held.update(change.lines)
            continue
        if not change.removes:
            continue
        for found in held_code(source):
            named[str(found.relative_to(checkout)).replace("\\", "/")] = None
    return named


def examined(named: dict[str, set[int] | None], listing: list[dict]) -> dict[str, str]:
    """Give each named file that holds a mutant to examine the pattern naming them.

    A file swept whole is its path; a file swept by its functions is its path
    and the lines its mutants start on, each mutant taken whose function's span
    -- or its own, outside every function -- holds a changed line. The pattern
    is matched against the names `cargo mutants --list` prints,
    `<file>:<line>:<column>: <mutation>`.
    """
    lines: dict[str, set[int]] = {}
    for mutant in listing:
        path = mutant["file"]
        if path not in named:
            continue
        changed = named[path]
        span = (mutant.get("function") or mutant)["span"]
        if changed is None or any(
            span["start"]["line"] <= line <= span["end"]["line"] for line in changed
        ):
            lines.setdefault(path, set()).add(mutant["span"]["start"]["line"])
    patterns: dict[str, str] = {}
    for path, changed in named.items():
        if path not in lines:
            continue
        if changed is None:
            patterns[path] = f"^{re.escape(path)}:"
        else:
            at = "|".join(str(line) for line in sorted(lines[path]))
            patterns[path] = f"^{re.escape(path)}:({at}):"
    return patterns


def main(argv: list[str]) -> int:
    parser = arguments(__doc__)
    parser.add_argument("base", help="the commit the change is measured from")
    parser.add_argument(
        "--listing",
        type=Path,
        required=True,
        help="what `cargo mutants --list --json` prints for the swept package",
    )
    parser.add_argument(
        "--examined",
        type=Path,
        required=True,
        help="where to write the pattern of each file named",
    )
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
    try:
        listing = json.loads(args.listing.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"code_changes: the listing cannot be read: {error}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    patterns = examined(swept(changed, CHECKOUT), listing)
    args.examined.write_text(
        "".join(f"{pattern}\n" for pattern in patterns.values()), encoding="utf-8"
    )
    for path in patterns:
        print(path)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
