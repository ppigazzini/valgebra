"""Regenerate the table of when each public `typing` and `enum` name arrived.

`tests/test_floor_names.py` refuses a name a module reaches at import time
before the release that spells it, and it reads the answer out of
`tests/floor_names.json`. That table was assembled by hand across six
interpreters, which is a thing nobody does twice: the release that adds a name
is a fact about CPython, and a fact about CPython is read rather than
remembered.

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


def _spans(seen: dict[str, dict[str, list[str]]], versions: list[str]) -> dict:
    """Turn "which releases carry it" into a `since`/`gone` span per name."""
    modules: dict[str, dict[str, dict[str, str]]] = {}
    for module in ("typing", "enum"):
        carried = {version: set(seen[version][module]) for version in versions}
        every = sorted(set().union(*carried.values()))
        rows: dict[str, dict[str, str]] = {}
        for name in every:
            holding = [version for version in versions if name in carried[version]]
            span = {"since": holding[0]}
            after = versions[versions.index(holding[-1]) + 1 :]
            if after:
                span["gone"] = after[0]
            rows[name] = span
        modules[module] = rows
    return modules


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
        seen[version] = reached
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
