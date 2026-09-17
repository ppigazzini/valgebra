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

PRODUCT: every public name a caller reaches
PRODUCT: every error code a report can carry
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

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
#: Read from `scripts/use_case_ledger.json` rather than written here, so the
#: count a lane prints and the set this file enforces come from one place. A
#: reason is a sentence about the cell, not a note that nobody got to it. Two
#: kinds qualify: a name that is not a use case at all, and a code the walk
#: cannot reach from any schema a caller can build.
LEDGER = ROOT / "scripts" / "use_case_ledger.py"


def _derivation() -> ModuleType:
    """Load the script that derives the universe and reports the count.

    One reading, loaded rather than copied: the lane prints a figure and this
    file enforces the set behind it, and two derivations would let them come to
    different answers about what a cell is.
    """
    spec = importlib.util.spec_from_file_location("use_case_ledger", LEDGER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_LEDGER = _derivation()
_public_surface = _LEDGER.public_surface
_error_codes = _LEDGER.error_codes
_product_sources = _LEDGER.product_sources
_names_reached = _LEDGER.names_reached
_universe = _LEDGER.universe
_markers = _LEDGER.markers
ACCEPTED: dict[str, str] = _LEDGER.accepted()


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
    reached = cell in _names_reached({cell}, _product_sources()) | _markers()
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
    reached = _names_reached(universe, _product_sources()) | _markers()
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


def _codes_named_in(name: str) -> set[str]:
    """Give the codes a test file asserts, read as the quoted strings it holds."""
    text = (ROOT / "tests" / name).read_text(encoding="utf-8")
    return {
        code
        for code in re.findall(r"""["']([a-z]+(?:_[a-z]+)+)["']""", text)
        if code in _reachable_codes()
    }


def test_no_code_is_evidenced_by_the_fail_fast_helper_alone() -> None:
    """A code asserted only under `fail_fast` says nothing about the other mode.

    `tests/test_error_codes.py` pins the code and the path for each node kind
    through a helper that always passes `fail_fast=True`. That is one cell of
    four: a report has two modes and two entry paths, and a code that differed
    between them would pass there. The error matrix is what drives all four,
    so every code that file names has to be a code the matrix drives -- and a
    code named there and nowhere else is exactly the gap this refuses.
    """
    rows, elsewhere = _matrix_rows()
    driven = rows | set(elsewhere)
    named = _codes_named_in("test_error_codes.py")
    assert len(named) >= 20, sorted(named)
    alone = sorted(named - driven)
    assert not alone, (
        f"codes asserted in test_error_codes.py and driven by no matrix row: "
        f"{alone}. Each is held under `fail_fast` alone, so nothing says what "
        "the aggregating mode or the JSON path reports for it."
    )


def test_a_cell_is_covered_by_code_rather_than_by_prose() -> None:
    """A name mentioned in a paragraph is not a test doing anything with it.

    The suite writes a great deal of prose, and a cell whose name appeared only
    in a docstring would read as covered. None does, and this is what keeps it
    so: the search runs over the code with the comments and docstrings cut, and
    reading it with them in must find no cell the tighter reading misses.
    """
    universe = _universe()
    with_prose = _names_reached(universe, _LEDGER.product_sources(prose=True))
    code_only = _names_reached(universe, _product_sources())
    only_mentioned = sorted(with_prose - code_only - _markers())
    assert not only_mentioned, (
        f"cells named only in prose: {only_mentioned}. A paragraph about a "
        "name is not a test that reaches it; write the row, or claim the cell "
        "with a `# USE-CASE:` marker if the test cannot spell the name."
    )


def test_every_marker_names_a_cell_that_exists() -> None:
    """A marker left behind by a rename claims a cell nothing has."""
    stale = sorted(_markers() - _universe())
    assert not stale, (
        f"`# USE-CASE:` markers naming no cell: {stale}. A marker is "
        "bookkeeping a reader keeps true, so one that stopped being true fails "
        "here rather than sitting in the file."
    )


#: The testing page's per-product figures, as a row this reads back: the
#: product, the cells it has, and the cells that are empty with a reason.
_FIGURE = re.compile(
    r"^\| (?P<product>every [^|]+?) \| (?P<cells>\d+) \| (?P<empty>\d+) \|$",
    re.MULTILINE,
)

#: What the page calls each of the two products this file derives.
_SURFACE = "every public name a caller reaches"
_CODES = "every error code a report can carry"


def test_the_page_carries_the_figure_each_product_has() -> None:
    """The page states each product's size, and the tree is what it states.

    "Seventy-odd cells" was the page's word for two products added together,
    and it was wrong in both directions at once: it hid which of the two was
    growing, and it aged without ever failing, because an approximation cannot
    be out by one. The lane prints the two figures apart and the page carries
    them, so the reader who wants to know how large the public surface is does
    not have to run anything -- and the day a name is added, the page is what
    fails rather than a reader's memory of it.
    """
    page = (ROOT / "docs" / "dev" / "08-testing.md").read_text(encoding="utf-8")
    figures = {
        match["product"]: (int(match["cells"]), int(match["empty"]))
        for match in _FIGURE.finditer(page)
    }
    assert set(figures) == {_SURFACE, _CODES}, (
        f"the testing page carries figures for {sorted(figures)}, and this file "
        f"derives {sorted((_SURFACE, _CODES))}"
    )

    surface, codes = _public_surface(), _error_codes()
    empty = set(ACCEPTED)
    for product, cells, unreached in (
        (_SURFACE, surface, empty & surface),
        (_CODES, codes, empty & codes),
    ):
        assert figures[product] == (len(cells), len(unreached)), (
            f"the page says {product} has {figures[product]} cells and empty "
            f"cells; the tree has {(len(cells), len(unreached))}"
        )

    # The detector: two products whose empty cells do not add up to the list
    # are two products that between them do not cover it, and the split above
    # would then be checking a partition of something else.
    assert (empty & surface) | (empty & codes) == empty, sorted(empty)


#: A line the lane prints for one product: its label and its three figures.
_PRINTED = re.compile(
    r"^(?P<label>[a-z ]+): (?P<cells>\d+) cells, (?P<named>\d+) named, "
    r"(?P<empty>\d+) empty",
    re.MULTILINE,
)


def test_the_lane_prints_a_figure_for_each_product() -> None:
    """A figure over two products is a figure neither of them has.

    The lane's output is where a reader learns the number without running a
    test, and one total answered for the public surface and the error codes
    together: a name added to the stub and a code retired from the walk cancel
    in it, and the same 72 reads as "nothing moved". The two are separate
    universes with separate derivations, so they are printed apart, and the
    page's table is held to what is printed rather than to a second reading.
    """
    printed = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, str(LEDGER)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    ).stdout
    figures = {
        match["label"]: (int(match["cells"]), int(match["named"]), int(match["empty"]))
        for match in _PRINTED.finditer(printed)
    }
    assert set(figures) >= {"public surface", "error codes"}, printed

    surface, codes = _public_surface(), _error_codes()
    empty = set(ACCEPTED)
    assert figures["public surface"][0] == len(surface), printed
    assert figures["public surface"][2] == len(empty & surface), printed
    assert figures["error codes"][0] == len(codes), printed
    assert figures["error codes"][2] == len(empty & codes), printed

    # And the page is the printed figure rather than a second reading of the
    # tree that happens to agree with it today.
    page = (ROOT / "docs" / "dev" / "08-testing.md").read_text(encoding="utf-8")
    stated = {
        match["product"]: (int(match["cells"]), int(match["empty"]))
        for match in _FIGURE.finditer(page)
    }
    assert stated[_SURFACE] == (
        figures["public surface"][0],
        figures["public surface"][2],
    ), printed
    assert stated[_CODES] == (figures["error codes"][0], figures["error codes"][2]), (
        printed
    )
