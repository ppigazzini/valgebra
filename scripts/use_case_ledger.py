"""Count the use cases the tree has, and say how many the suite reaches.

"Every use case is covered" is a claim only once a use case is a thing something
can count, and the count has to come from the tree rather than from a list
somebody wrote beside it. This derives two products and reports the figure:

* **the public surface**, every name a caller reaches through the package, read
  from the type stub it ships;
* **the error codes**, every code the walk can put in a report, read from the
  Rust that emits them.

A cell is covered when the product suite *names* it. What that is worth, and
what it is not, is written at the top of `tests/test_use_case_ledger.py`, which
is where the claim is enforced: this script is the reporting half, so a lane
prints the number rather than a reader running a test to learn it.

The cells no test names are recorded in `scripts/use_case_ledger.json` with the
reason each is empty, the way the mutation baselines record an accepted
survivor. Held in both directions by that test: a cell with no test and no
reason fails, and a reason for a cell that *is* named fails too, so an excuse
cannot outlive the gap it excuses.

Usage:
    python scripts/use_case_ledger.py            # print the count, check the file
    python scripts/use_case_ledger.py --update   # re-record the accepted cells

Two outcomes, two exit codes: **0** the recorded set matches what the tree has,
**1** it does not. A mismatch names the cells either way round.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "scripts" / "use_case_ledger.json"
STUB = ROOT / "python" / "valgebra" / "_valgebra.pyi"
PACKAGE = ROOT / "python" / "valgebra" / "__init__.py"
IR = ROOT / "crates" / "valgebra-core" / "src" / "ir.rs"
#: Where a code is written: the walk that reports one, and the two entries that
#: report a document the parser refused.
_EMITTERS = (
    ROOT / "crates" / "valgebra-py" / "src" / "check",
    ROOT / "crates" / "valgebra-py" / "src" / "input.rs",
    ROOT / "crates" / "valgebra-py" / "src" / "validator.rs",
    ROOT / "crates" / "valgebra-py" / "src" / "errors.rs",
)

EXIT_OK = 0
EXIT_FAIL = 1


def public_surface() -> set[str]:
    """Every name a caller reaches, read from the stub the package ships."""
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and not member.name.startswith(
                    "__"
                ):
                    names.add(f"{node.name}.{member.name}")
                if isinstance(member, ast.AnnAssign) and isinstance(
                    member.target, ast.Name
                ):
                    names.add(f"{node.name}.{member.target.id}")
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    # The package re-exports a marker of its own, which the stub does not carry.
    exported = ast.parse(PACKAGE.read_text(encoding="utf-8"))
    for node in ast.walk(exported):
        if isinstance(node, ast.ImportFrom) and node.module == "_markers":
            names.update(alias.name for alias in node.names)
    return {name for name in names if not name.startswith("__")}


def error_codes() -> set[str]:
    """Every code the walk can put in a report, read from the Rust.

    Two sources, because a code arrives two ways. The node kinds come from the
    table in the IR, which is what a leaf reports for its own kind. The rest are
    written at the site that reports them, and are recognised by shape: a code is
    snake_case with at least one underscore, which is what separates one from the
    words beside it (`"dict"`, `"list"`) that name a kind in a message.
    """
    codes: set[str] = set()
    source = IR.read_text(encoding="utf-8")
    body = source[source.index("fn error_code") :]
    body = body[: body.index("\n    }")]
    codes.update(re.findall(r'"([a-z_]+)"', body))
    for path in _EMITTERS:
        for found in path.rglob("*.rs") if path.is_dir() else [path]:
            if found.name.endswith("interpreter.rs") or found.name.endswith("tests.rs"):
                continue
            text = found.read_text(encoding="utf-8")
            codes.update(re.findall(r'"([a-z]+(?:_[a-z]+)+)"', text))
    # `anything` is the top's label in the same table, and has no failure. It is
    # carried under a name of its own so the reason reads as being about a code.
    codes.discard("anything")
    codes.add("anything_code")
    return codes


#: The nodes a docstring can be the first statement of.
_CARRIES_A_DOCSTRING = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _without_prose(text: str) -> str:
    """Give the code with every docstring and comment cut.

    A cell is covered by a test *doing* something with it, and this suite
    writes a great deal of prose: a name mentioned in a paragraph about why
    something is hard would otherwise read as coverage. No cell is covered that
    way today, and cutting the prose is what keeps it so rather than a note
    saying somebody checked. Unparsing drops the comments on its own.
    """
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, _CARRIES_A_DOCSTRING):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def product_sources(*, prose: bool = False) -> str:
    """Give every product test file's code, as one blob to search.

    Comments and docstrings are cut unless `prose` is set; `_without_prose`
    says why.
    """
    blobs = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "pytestmark = pytest.mark.repository" in text:
            continue
        if prose:
            blobs.append(text)
            continue
        try:
            blobs.append(_without_prose(text))
        except SyntaxError:  # pragma: no cover - the suite parses
            blobs.append(text)
    return "\n".join(blobs)


def markers() -> set[str]:
    """Give every cell a test claims with a `# USE-CASE:` marker.

    The channel for a cell whose name a test cannot spell -- one reached
    through a fixture, or through a spelling the search does not see. Rare by
    design: the search is the primary direction, because a marker is
    bookkeeping a reader has to keep true and a name in the code is not. What
    the ledger holds is that a marker names a cell that exists, so one left
    behind by a rename fails rather than sitting there.
    """
    found: set[str] = set()
    for path in sorted((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "pytestmark = pytest.mark.repository" in text:
            continue
        found.update(re.findall(r"#\s*USE-CASE:\s*(\S+)", text))
    return found


def names_reached(cells: set[str], suite: str) -> set[str]:
    """Give the cells the suite mentions, each looked for the way it is written.

    A **code** is a string a test compares against, so it is looked for quoted:
    the bare word appears in prose about recursion without any test asserting the
    code. A **name** is called or imported, so the word itself is the evidence.
    """
    codes = error_codes()
    reached = set()
    for cell in cells:
        needle = cell.split(".")[-1]
        pattern = (
            rf"[\"']{re.escape(needle)}[\"']"
            if cell in codes
            else rf"\b{re.escape(needle)}\b"
        )
        if re.search(pattern, suite):
            reached.add(cell)
    return reached


def universe() -> set[str]:
    return public_surface() | error_codes()


def accepted() -> dict[str, str]:
    """Give the recorded empty cells, or an empty mapping before the first run."""
    if not RECORD.exists():
        return {}
    return dict(json.loads(RECORD.read_text(encoding="utf-8"))["empty"])


def main() -> int:
    every = universe()
    reached = names_reached(every, product_sources()) | markers()
    recorded = accepted()
    empty = {cell: recorded.get(cell, "") for cell in sorted(every - reached)}
    universe_size = len(every)

    print(f"use cases: {universe_size}")
    print(f"named by the suite: {len(reached)}")
    print(f"empty, each with a reason: {len(empty)}")
    print(f"covered: {100 * len(reached) / universe_size:.1f}%")

    if "--update" in sys.argv[1:]:
        RECORD.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Use cases the product suite does not name, each with "
                        "the reason it does not. Derived by "
                        "scripts/use_case_ledger.py and held to the tree in "
                        "both directions by tests/test_use_case_ledger.py. A "
                        "reason is a sentence about the cell, not a note that "
                        "nobody got to it."
                    ),
                    "empty": empty,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"use_case_ledger: recorded {len(empty)} empty cell(s)")
        return EXIT_OK

    if not RECORD.exists():
        print(f"use_case_ledger: no {RECORD.name}; create one with --update")
        return EXIT_FAIL

    gained = sorted(set(empty) - set(recorded))
    filled = sorted(set(recorded) - set(empty))
    for cell in gained:
        print(f"EMPTY AND NOT RECORDED: {cell}")
    for cell in filled:
        print(f"RECORDED AND NO LONGER EMPTY: {cell}")
    if gained or filled:
        print(
            "\nuse_case_ledger: the recorded empty cells are not the tree's. "
            "Fill each new one or record it with a reason, and drop the ones "
            "a test now names."
        )
        return EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
