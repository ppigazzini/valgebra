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

Usage:
    python scripts/code_changes.py BASE

Exit 0 with the files on stdout, deleted files left out; 2 when git cannot
answer.
"""

from __future__ import annotations

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


def code_changes(diff: str) -> list[str]:
    """Name each file of a `git diff -U0` whose added or removed lines hold code."""
    found: list[str] = []
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current = None
        elif line.startswith("+++ "):
            path = line[4:]
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


def main(argv: list[str]) -> int:
    parser = arguments(__doc__)
    parser.add_argument("base", help="the commit the change is measured from")
    args = parser.parse_args(argv)
    diff = git_output(
        "diff",
        "-U0",
        "--no-color",
        "--diff-filter=d",
        f"{args.base}...HEAD",
        "--",
        "*.rs",
        cwd=CHECKOUT,
    )
    if diff is None:
        print(f"code_changes: git cannot diff from {args.base}", file=sys.stderr)
        return EXIT_CANNOT_RUN
    for path in code_changes(diff):
        print(path)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
