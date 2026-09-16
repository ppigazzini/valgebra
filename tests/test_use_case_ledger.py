"""Every use case the tree has is one the suite reaches, or one with a reason.

"100% coverage of all use cases" is a number only once *use case* is a set
something can count. A line count is not it -- a line runs without anything
checking what it did -- and neither is a list somebody wrote beside the code,
which grows a row when a reader remembers to.

So the universe is read out of the tree, in two products:

* **the public surface**, every name a caller can reach through the package: each
  method on the validator and the exception, each module-level function, each
  constant, parsed from the type stub the package ships;
* **the error codes**, every code the walk can put in a report, read from the
  Rust that emits them.

A cell is covered when the product suite *names* it. That is what a rule can
see, and it is worth saying plainly what it cannot: naming a method is not
asserting its documented outcome, and naming a code is not checking its path or
its location. Those are what `tests/test_node_matrix.py`,
`tests/test_error_contract.py` and the matrices beside them are for. What this
catches is the gap no reading catches -- a name the tree grows and the suite
never mentions, which is the state every one of them starts in.

The other direction is the accepted list: a cell the suite does not reach is
written down with a reason, and a reason for a cell that *is* reached fails too,
so an excuse cannot outlive the gap it excuses.

LEDGER: every public name and every error code is named by the suite, or accepted
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
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

#: Cells the product suite does not name, each with the reason it does not.
#:
#: A reason is a sentence about the cell, not a note that nobody got to it. Two
#: kinds qualify: a name that is not a use case at all, and a code the walk
#: cannot reach from any schema a caller can build.
ACCEPTED: dict[str, str] = {
    "anything_code": (
        "`Schema::Anything` is the top, which admits every value, so the walk has "
        "no failure to report against one and the arm is the table's fallback"
    ),
    "intersection_error": (
        "a meet reports its failing member rather than itself, so the arm is "
        "reached only through a node with no members -- which the constructors "
        "fold to the top before a walk sees it"
    ),
    "object_type": (
        "an attribute record is a meet with the class it came from, so a value of "
        "the wrong type is refused by `instance_type` first and the arm stands for "
        "a bare attribute record, which the frontend does not build"
    ),
    "recursion": (
        "a reference reports through the definition it names, so the walk carries "
        "the definition's own code rather than this one"
    ),
    "validation_error": (
        "the exception's own type name, carried where a report is built from no "
        "violation at all: a caller who constructs one by hand. It is the class "
        "rather than a code a walk chooses"
    ),
    "unresolved_recursion": (
        "the marker a builder leaves behind, resolved before a validator is "
        "returned: a compiled schema holds none"
    ),
}


def _public_surface() -> set[str]:
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


def _error_codes() -> set[str]:
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


def _product_sources() -> str:
    """Every product test file's text, as one blob to search."""
    blobs = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "pytestmark = pytest.mark.repository" in text:
            continue
        blobs.append(text)
    return "\n".join(blobs)


def _names_reached(cells: set[str], suite: str) -> set[str]:
    """Give the cells the suite mentions, each looked for the way it is written.

    A **code** is a string a test compares against, so it is looked for quoted:
    the bare word appears in prose about recursion without any test asserting the
    code. A **name** is called or imported, so the word itself is the evidence.
    """
    codes = _error_codes()
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


def _universe() -> set[str]:
    return _public_surface() | _error_codes()


def test_the_universe_is_read_from_the_tree() -> None:
    """The parse is a detector, so it must be shown to have read something."""
    surface, codes = _public_surface(), _error_codes()
    assert len(surface) >= 20, sorted(surface)
    assert len(codes) >= 25, sorted(codes)
    # Names a reader can check by hand, so a parse that drifted is visible.
    for name in ("Validator.ensure", "Validator.relation_to", "union", "Regex"):
        assert name in surface, sorted(surface)
    for code in ("int_type", "missing_key", "recursion_limit", "union_error"):
        assert code in codes, sorted(codes)


def test_the_product_suite_is_what_is_searched() -> None:
    """A repository check naming a cell would cover it with work about the tree."""
    suite = _product_sources()
    assert len(suite) > 100_000, "the product suite blob is too small to be it"
    # This file is a repository check, so its own accepted list is not the
    # evidence: a cell named only here stays uncovered.
    assert "LEDGER: every public name and every error code" not in suite


@pytest.mark.parametrize("cell", sorted(_universe()))
def test_every_use_case_is_named_by_the_suite_or_accepted(cell: str) -> None:
    """A name the tree grows and the suite never mentions fails here."""
    reached = cell in _names_reached({cell}, _product_sources())
    accepted = cell in ACCEPTED
    assert reached or accepted, (
        f"{cell} is a use case no product test names, and has no accepted reason"
    )
    assert not (reached and accepted), (
        f"{cell} is named by the suite and still carries a reason for not being"
    )


def test_every_accepted_reason_is_a_sentence() -> None:
    """An excuse short enough to be a shrug is not one."""
    for cell, reason in ACCEPTED.items():
        assert len(reason) > 40, f"{cell}: {reason!r}"
        assert cell in _universe(), f"{cell} is accepted and is not a use case"


def test_the_count_is_reported() -> None:
    """The number the brief asks for, computed rather than asserted.

    A floor rather than an equality: the universe grows with the tree, and a
    ledger that pinned the count would fail on the commit that adds a name
    rather than on the one that leaves it unreached.
    """
    universe = _universe()
    reached = _names_reached(universe, _product_sources())
    assert reached | set(ACCEPTED) >= universe
    covered = len(reached) / len(universe)
    assert covered > 0.85, f"{len(reached)} of {len(universe)} use cases named"


def _matrix_rows() -> tuple[set[str], dict[str, str]]:
    """Give the error matrix's two tables, read from the file that holds them.

    `tests/test_error_matrix.py` drives every code through both modes, both
    paths and a nested location. It carries its tables as literals rather than
    deriving them, because it runs against an installed wheel and the codes are
    written in the Rust. This is the half that does read the Rust, so it is the
    half that holds the tables to it.

    Parsed rather than imported: importing a product test from a repository
    check runs its module body, and what is wanted is the two lists.
    """
    tree = ast.parse(
        (ROOT / "tests" / "test_error_matrix.py").read_text(encoding="utf-8")
    )
    rows: set[str] = set()
    elsewhere: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys = [
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        if node.target.id == "CASES":
            rows = set(keys)
        elif node.target.id == "ELSEWHERE":
            elsewhere = {
                key: value.value
                for key, value in zip(keys, node.value.values, strict=True)
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            }
    assert rows, "the error matrix's CASES table did not parse"
    assert elsewhere, "the error matrix's ELSEWHERE table did not parse"
    return rows, elsewhere


def _reachable_codes() -> set[str]:
    """Every code a schema a caller can build can reach.

    The accepted list above is the codes no such schema reaches, so a row for
    one would be a row nothing produces. `not_subset` is a *relation* answer
    that shares the shape this derivation looks for -- `relation_to` writes it
    and reports no violation at all -- so it leaves too.
    """
    return _error_codes() - set(ACCEPTED) - {"not_subset"}


def test_every_reachable_code_is_driven_by_the_error_matrix() -> None:
    """A code the walk reports and the matrix does not drive fails here."""
    rows, elsewhere = _matrix_rows()
    missing = sorted(_reachable_codes() - rows - set(elsewhere))
    assert not missing, (
        f"codes the error matrix does not drive: {missing}. Add a row to its "
        "CASES, or, where the code needs a value a table cannot carry, name "
        "its test in ELSEWHERE."
    )


def test_the_error_matrix_drives_no_code_the_walk_cannot_report() -> None:
    """A row for a code the tree no longer writes is a row nothing produces."""
    rows, elsewhere = _matrix_rows()
    stale = sorted((rows | set(elsewhere)) - _reachable_codes())
    assert not stale, f"the error matrix has rows for absent codes: {stale}"


def test_every_test_the_matrix_names_exists() -> None:
    """A name in the matrix's ELSEWHERE that resolves to nothing holds no code."""
    _, elsewhere = _matrix_rows()
    text = (ROOT / "tests" / "test_error_matrix.py").read_text(encoding="utf-8")
    defined = set(re.findall(r"^def (test_\w+)", text, re.MULTILINE))
    missing = sorted(name for name in elsewhere.values() if name not in defined)
    assert not missing, f"the matrix names tests it does not define: {missing}"


def _snapshot_codes() -> set[str]:
    """Give the codes the core's message-format corpus pins.

    The corpus is hand-built: each row is a `Violation` the test constructs, so
    nothing in the core makes a row's code one the library ever writes. Read by
    parsing the Rust for the first argument of each `violation(...)` call.
    """
    path = ROOT / "crates" / "valgebra-core" / "tests" / "error_snapshots.rs"
    text = path.read_text(encoding="utf-8")
    body = text[text.index("let corpus = [") :]
    return set(re.findall(r'violation\(\s*"([a-z_]+)"', body))


def test_the_snapshot_corpus_pins_codes_the_walk_can_write() -> None:
    """A hand-built corpus can pin a code the library does not have.

    The corpus locks the one-line rendering of a violation, and it builds its
    own rows rather than taking them from a walk. That is what makes it a test
    of the *format* -- and what lets a row drift from the library: a code
    nobody writes renders perfectly and pins a format for a failure no caller
    can meet. Read against the codes the tree emits, which is the same list
    every other row here is read against.
    """
    pinned = _snapshot_codes()
    assert len(pinned) >= 6, sorted(pinned)
    unknown = sorted(pinned - _error_codes())
    assert not unknown, (
        f"the core's snapshot corpus pins codes the walk never writes: "
        f"{unknown}. Each row's code is a string the corpus chose, so a code "
        "that was renamed, or never existed, renders and passes."
    )
