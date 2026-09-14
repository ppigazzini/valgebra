"""Regenerate the table of when each name, and each stdlib module, arrived.

`tests/test_floor_names.py` refuses a name a module reaches at import time
before the release that spells it -- and a standard-library *module* a source
imports before the release that ships it, which is the same mistake one level
up. It reads both answers out of `tests/floor_names.json`. That table was
assembled by hand across six interpreters, which is a thing nobody does twice:
the release that adds a name is a fact about CPython, and a fact about CPython
is read rather than remembered.

Every interpreter in the table's own range is asked what it carries, and the
spans are computed from the answers -- `since` for the first release holding a
name, and `gone` for the first release past it that does not, since a name can
be removed as well as added (3.15 drops one).

    python scripts/floor_names.py --check     # compare, and say what moved
    python scripts/floor_names.py --update    # rewrite the table

**It refuses rather than guessing.** An interpreter in the range that is not
installed makes every name it would have carried unreadable, and a table
assembled from the rest would be wrong in exactly the direction that matters:
it would date names to a release that never answered. Exit code 2 says the
question could not be asked, which is not the same as an answer.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "tests" / "floor_names.json"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

#: The modules the table dates names in.
MODULES = ("typing", "enum")

#: First pass: the public names each module lists.
LISTS = (
    "import json,sys;"
    "print(json.dumps({m: sorted(n for n in dir(__import__(m))"
    " if not n.startswith('_')) for m in json.loads(sys.argv[1])}))"
)

#: Second pass: which of the candidates each module *reaches*.
#:
#: `dir` and `hasattr` are different questions, and the table answers the second
#: one because that is what a module reaching a name at import time does. A
#: deprecated alias `typing` serves through a module `__getattr__` is absent
#: from `dir` and present to `hasattr` -- and the first access writes it into the
#: module, so asking `hasattr` first makes it appear in a later `dir`. Each pass
#: therefore runs in its own interpreter, and the one that decides asks
#: `hasattr` only.
REACHES = (
    "import json,sys;"
    "cands=json.loads(sys.argv[1]);"
    "print(json.dumps({m: sorted(n for n in cands[m]"
    " if hasattr(__import__(m), n)) for m in cands}))"
)


#: Third pass: which top-level modules this release ships.
#:
#: `sys.stdlib_module_names` is the release's own answer and arrives in 3.10,
#: which is the floor, so every release the table spans can be asked. It is
#: membership rather than an import, so asking costs nothing and leaves nothing
#: behind -- and a module that fails to import for its own reasons still counts
#: as shipped, which is the question here.
STDLIB = (
    "import json,sys;"
    "names=sorted(sys.stdlib_module_names);"
    "print(json.dumps({'stdlib': names}))"
)


#: Where a module import is read from, which is the ledger's own scan list.
SCANNED = ("tests/*.py", "python/valgebra/**/*.py", "scripts/*.py")


def _imported(root: Path) -> set[str]:
    """Every top-level module name the scanned sources import.

    The table dates these and not the standard library entire, because
    `sys.stdlib_module_names` is not a property of the release alone: two builds
    of one version disagree on the modules a platform does not need -- this box's
    3.12 lists the Windows-only `_wmi` and a runner's does not -- so a table of
    every name would report a difference between two builds as a moved row, on
    the nightly, forever. What the ledger judges is the imports this tree makes,
    and those are portable by construction: they run on three operating systems
    and two implementations already.

    Judging *which* of them is reached too early is the ledger's business; this
    only decides what to ask the interpreters about.
    """
    found: set[str] = set()
    for pattern in SCANNED:
        for path in sorted(root.glob(pattern)):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.update(
                        alias.name.split(".", maxsplit=1)[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom):
                    if node.level or not node.module:
                        continue  # a relative import names no top-level module
                    found.add(node.module.split(".", maxsplit=1)[0])
    return found


def _versions(table: dict) -> list[str]:
    """Every release the table spans, `floor` through `known_through`."""
    low = int(table["floor"].split(".")[1])
    high = int(table["known_through"].split(".")[1])
    return [f"3.{minor}" for minor in range(low, high + 1)]


def _ask(version: str, program: str, *argv: str) -> dict[str, list[str]] | None:
    """Run `program` on `version`, or `None` where it cannot be asked."""
    try:
        done = subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                version,
                "python",
                "-c",
                program,
                *argv,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout)
    except ValueError:
        return None


def _rows(
    carried: dict[str, set[str]], versions: list[str], every: list[str]
) -> dict[str, dict[str, str]]:
    """Turn "which releases carry it" into a `since`/`gone` span per name."""
    rows: dict[str, dict[str, str]] = {}
    for name in every:
        holding = [version for version in versions if name in carried[version]]
        if not holding:
            continue  # in no release the table spans: not this table's business
        span = {"since": holding[0]}
        after = versions[versions.index(holding[-1]) + 1 :]
        if after:
            span["gone"] = after[0]
        rows[name] = span
    return rows


def _spans(seen: dict[str, dict[str, list[str]]], versions: list[str]) -> dict:
    """Span every watched name, one section per module."""
    modules: dict[str, dict[str, dict[str, str]]] = {}
    for module in MODULES:
        carried = {version: set(seen[version][module]) for version in versions}
        every = sorted(set().union(*carried.values()))
        modules[module] = _rows(carried, versions, every)
    return modules


def _stdlib(
    seen: dict[str, dict[str, list[str]]], versions: list[str], asked: set[str]
) -> dict[str, dict[str, str]]:
    """Span every module this tree imports, out of each release's own list.

    A third-party package is in no release's list and gets no row, which is how
    the ledger tells "this arrives in 3.11" from "this is not the standard
    library's to answer for". See [`_imported`] for why the table is the tree's
    imports rather than every name a release ships.
    """
    carried = {version: set(seen[version]["stdlib"]) for version in versions}
    every = sorted(asked & set().union(*carried.values()))
    return _rows(carried, versions, every)


def _survey(table: dict, versions: list[str]) -> dict[str, dict[str, list[str]]] | None:
    """Ask every release which of the candidate names it reaches.

    The candidates are every name any release lists *and* every name the table
    already dates, so a row for a name no release lists any more is asked about
    rather than quietly dropped.
    """

    def refuse(version: str) -> None:
        print(f"floor_names: CPython {version} did not answer; install it first")
        print("A table assembled from the rest dates names to a release that")
        print("never answered, which is worse than no table at all.")

    candidates: dict[str, set[str]] = {
        module: set(table["modules"].get(module, {})) for module in MODULES
    }
    for version in versions:
        listed = _ask(version, LISTS, json.dumps(MODULES))
        if listed is None:
            refuse(version)
            return None
        for module, names in listed.items():
            candidates[module].update(names)
    asked = {module: sorted(names) for module, names in candidates.items()}

    seen = {}
    for version in versions:
        reached = _ask(version, REACHES, json.dumps(asked))
        if reached is None:
            refuse(version)
            return None
        shipped = _ask(version, STDLIB)
        if shipped is None:
            refuse(version)
            return None
        seen[version] = {**reached, **shipped}
    return seen


def main() -> int:
    args = sys.argv[1:]
    if not ({"--check", "--update"} & set(args)):
        print(__doc__)
        return EXIT_CANNOT_RUN
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    versions = _versions(table)

    seen = _survey(table, versions)
    if seen is None:
        return EXIT_CANNOT_RUN

    rebuilt = dict(table)
    rebuilt["modules"] = _spans(seen, versions)
    rebuilt["stdlib"] = _stdlib(seen, versions, _imported(ROOT))
    if "--update" in args:
        TABLE.write_text(json.dumps(rebuilt, indent=2) + "\n", encoding="utf-8")
        print(f"floor_names: table rewritten from {len(versions)} interpreters")
        return EXIT_OK

    if rebuilt == table:
        print(f"floor_names: the table matches {len(versions)} interpreters")
        return EXIT_OK
    for module, rows in rebuilt["modules"].items():
        was = table["modules"].get(module, {})
        for name, span in sorted(rows.items()):
            if was.get(name) != span:
                print(f"MOVED {module}.{name}: {was.get(name)} -> {span}")
        for name in sorted(set(was) - set(rows)):
            print(f"GONE  {module}.{name}: {was[name]} -> absent from every release")
    print("\nfloor_names: the table disagrees with the interpreters; --update it")
    return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
