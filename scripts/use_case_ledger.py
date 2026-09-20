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
#: The binding's half of the vocabulary, declared rather than scanned for.
#:
#: The codes were recovered by reading four hand-picked files for snake-case
#: strings, which is a guess from a path: it misses a code written in a file
#: nobody thought to name, and invents a cell for any other string of that
#: shape in one that was -- it had `not_subset`, an answer `relation_to` gives
#: and no failure anybody reports. `tests/test_code_table.py` holds every name
#: here to a site that writes it.
CODE_TABLE = ROOT / "crates" / "valgebra-py" / "src" / "codes.rs"

EXIT_OK = 0
EXIT_FAIL = 1


def public_surface() -> set[str]:
    """Every name a caller reaches, read from the stub the package ships."""
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and (
                    not member.name.startswith("__") or member.name in OPERATORS
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


#: The operator surface: the dunders the stub declares that a caller reaches
#: through an operator or a builtin rather than by name, each with the shape
#: the suite spells it in. `__new__` is not here, because `Validator(...)` is
#: every test's first line and a cell that cannot be empty counts nothing.
OPERATORS: dict[str, str] = {
    "__contains__": "`x in v`",
    "__or__": "`a | b`",
    "__ror__": "`a | b`",
    "__eq__": "`a == b`",
    "__hash__": "`hash(v)`",
    "__copy__": "`copy.copy(v)`",
    "__deepcopy__": "`copy.deepcopy(v)`",
    "__reduce__": "`pickle.dumps(v)`",
}


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
    declared = CODE_TABLE.read_text(encoding="utf-8")
    named = r'const [A-Z][A-Z0-9_]*: Code = Code\("([a-z_]+)"\);'
    codes.update(re.findall(named, declared))
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


def assertions_in(text: str) -> str:
    """Give the parts of one module that sit inside an assertion.

    The test of every `assert`, and the body of every `with pytest.raises(...)`.
    A name inside either is a name the suite is asking a question about; a name
    outside both may be a call whose result nothing reads.

    Taken as text rather than as a path so the narrowing can be put to a pair
    of modules and shown to reject one of them. A detector nobody has seen
    reject anything reports the same figure whether or not it works.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:  # pragma: no cover - the suite parses
        return ""
    kept: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            kept.append(ast.unparse(node))
        elif isinstance(node, ast.With) and _raises(node):
            kept.extend(ast.unparse(statement) for statement in node.body)
    return "\n".join(kept)


def asserted_sources() -> str:
    """Give the product suite's assertions, as one blob to search.

    A cell is *named* when the suite spells it and *asserted* when the suite
    says what it answers, and the two are different claims about coverage. The
    first is what a search over the source can see, and it is satisfied by a
    call whose result nothing reads -- `Validator(spec).ensure(value)` on a line
    of its own names `ensure` and checks nothing about it.

    Being inside an assertion is not a proof that the question asked is the
    right one -- nothing mechanical reads that -- but it separates a call from
    a claim, which is the half the mention count cannot see.
    """
    blobs: list[str] = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "pytestmark = pytest.mark.repository" in text:
            continue
        blobs.append(assertions_in(text))
    return "\n".join(blobs)


def _raises(node: ast.With) -> bool:
    """Whether a `with` block is a `pytest.raises`, however it is spelled."""
    for item in node.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        target = call.func
        if isinstance(target, ast.Attribute) and target.attr == "raises":
            return True
        if isinstance(target, ast.Name) and target.id == "raises":
            return True
    return False


def _stub_kinds() -> dict[str, str]:
    """Give each public name's kind: `method`, `attribute`, `function`, `constant`."""
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    kinds: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    kinds[f"{node.name}.{member.name}"] = "method"
                elif isinstance(member, ast.AnnAssign) and isinstance(
                    member.target, ast.Name
                ):
                    kinds[f"{node.name}.{member.target.id}"] = "attribute"
        elif isinstance(node, ast.FunctionDef):
            kinds[node.name] = "function"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            kinds[node.target.id] = "constant"
    return kinds


def raises_aliases() -> frozenset[str]:
    """Give every name the product suite binds a `pytest.raises` context to.

    `ValidationError.value` is the failing value's summary, and `caught.value`
    is pytest's exception: the same spelling, and the second is written on
    every line that reads an error. A reading that counted it would report the
    attribute as asserted by the whole suite, so the aliases are read out of
    the suite and an attribute read off one of them is not the cell.
    """
    found: set[str] = set()
    for path in sorted((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "pytestmark = pytest.mark.repository" in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.With) and _raises(node):
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        found.add(item.optional_vars.id)
    return frozenset(found)


#: The builtin or module function behind each dunder a caller reaches by a call.
_CALLED_AS: dict[str, str] = {
    "__hash__": "hash",
    "__copy__": "copy",
    "__deepcopy__": "deepcopy",
    "__reduce__": "dumps",
}


def _operator_reached(name: str, node: ast.AST) -> bool:
    """Whether this node is the operator or the call behind `name`."""
    match name:
        case "__contains__":
            return isinstance(node, ast.Compare) and any(
                isinstance(op, ast.In | ast.NotIn) for op in node.ops
            )
        case "__or__" | "__ror__":
            return isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
        case "__eq__":
            return isinstance(node, ast.Compare) and any(
                isinstance(op, ast.Eq | ast.NotEq) for op in node.ops
            )
        case _:
            if not isinstance(node, ast.Call):
                return False
            callee = node.func
            spelled = (
                callee.id
                if isinstance(callee, ast.Name)
                else callee.attr
                if isinstance(callee, ast.Attribute)
                else None
            )
            return spelled == _CALLED_AS.get(name)


class _Reading:
    """What one blob of the suite reaches, read once off its syntax tree.

    A **method** is a call on a receiver, an **attribute** a read off one that
    is not a `pytest.raises` alias, a **function** or a **constant** a name or
    a read off the package, and an **operator** the operator itself -- each
    read from the tree rather than from the word, because the word `value` is
    on every line that catches an exception and names the cell on almost none
    of them.
    """

    def __init__(self, suite: str) -> None:
        try:
            self.tree: ast.AST = ast.parse(suite)
        except SyntaxError:  # pragma: no cover - the suite parses
            self.tree = ast.Module(body=[], type_ignores=[])
        aliases = raises_aliases()
        self.called: set[str] = set()
        self.read: set[str] = set()
        self.named: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.called.add(node.func.attr)
            if isinstance(node, ast.Attribute) and not (
                isinstance(node.value, ast.Name) and node.value.id in aliases
            ):
                self.read.add(node.attr)
            if isinstance(node, ast.Name):
                self.named.add(node.id)

    def reaches(self, cell: str, kind: str | None) -> bool:
        """Whether the suite reaches this cell, read the way its kind is written."""
        needle = cell.rsplit(".", maxsplit=1)[-1]
        if needle in OPERATORS:
            return any(_operator_reached(needle, node) for node in ast.walk(self.tree))
        if kind == "method":
            return needle in self.called
        if kind == "attribute":
            return needle in self.read
        return needle in self.named or needle in self.read


def names_reached(cells: set[str], suite: str) -> set[str]:
    """Give the cells the suite reaches, each looked for the way it is written.

    A **code** is a string a test compares against, so it is looked for quoted:
    the bare word appears in prose about recursion without any test asserting
    the code. Every other cell is read off the syntax tree by its kind, which
    `_Reading` says.
    """
    codes = error_codes()
    kinds = _stub_kinds()
    reading = _Reading(suite)
    reached = set()
    for cell in cells:
        if cell in codes:
            needle = re.escape(cell)
            if re.search(rf"[\"']{needle}[\"']", suite):
                reached.add(cell)
        elif reading.reaches(cell, kinds.get(cell)):
            reached.add(cell)
    return reached


def universe() -> set[str]:
    return public_surface() | error_codes()


def accepted() -> dict[str, str]:
    """Give the recorded empty cells, or an empty mapping before the first run."""
    if not RECORD.exists():
        return {}
    return dict(json.loads(RECORD.read_text(encoding="utf-8"))["empty"])


def report(
    label: str,
    cells: set[str],
    reached: set[str],
    asserted: set[str],
    empty: set[str],
) -> None:
    """Print one product's figures, with what is named apart from what is asked.

    Three counts rather than one, because "covered" means two different things
    here and folding them hides the weaker. **Named** is the suite spelling the
    cell, which is what a search can see and is satisfied by a call nothing
    reads. **Asserted** is the cell inside an `assert` or a `pytest.raises`,
    which is the suite saying what it answers. The ratio is the second, because
    a figure that counts mentions reports a suite as covering a name it never
    asks a question about.

    A product with no cell at all would divide by zero on the way to a ratio,
    and it is a derivation that read nothing rather than a product that is
    fully covered -- so it says so instead.
    """
    named, asked, unreached = cells & reached, cells & asserted, cells & empty
    if not cells:
        print(f"{label}: 0 cells, which is a derivation that read nothing")
        return
    print(
        f"{label}: {len(cells)} cells, {len(named)} named, {len(asked)} asserted, "
        f"{len(unreached)} empty with a reason, "
        f"{100 * len(asked) / len(cells):.1f}% covered"
    )


def main() -> int:
    every = universe()
    reached = names_reached(every, product_sources()) | markers()
    # A marker is a claim that a test drives the cell, which is the stronger of
    # the two readings, so it counts on both sides.
    asserted = names_reached(every, asserted_sources()) | markers()
    recorded = accepted()
    empty = {cell: recorded.get(cell, "") for cell in sorted(every - reached)}

    # Per product, and then the total. One figure over both universes is a
    # figure neither of them has: a name added to the stub and a code retired
    # from the walk cancel in it, and the sum reads as "nothing moved". The
    # labels are the testing page's products, which holds this to that table.
    for label, cells in (
        ("public surface", public_surface()),
        ("error codes", error_codes()),
        ("use cases", every),
    ):
        report(label, cells, reached, asserted, set(empty))

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
