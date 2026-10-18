"""The two halves of the suite must not drift into each other.

The suite holds product tests, which exercise the library, and repository
checks, which read this tree: its configuration, its gate scripts, its
documentation, its CI workflows. Only the first half is a claim about the
wheel. A repository check that runs as part of the product suite makes the
product look tested by work that never touched it, and it fails for a reader
who has the library but not the repository -- a wheel, an sdist, a distribution
package.

The marker is the boundary, and the import is the evidence for it:

* a file that never imports ``valgebra`` exercises nothing and must be marked;
* a marked file that imports ``valgebra`` is a product test wearing the marker,
  and would be dropped from the product suite by mistake.

The line is not between needing the checkout and not needing it. A product
test that holds the library to a claim the tree writes down -- a page's table,
the variants of the IR, the stub's signatures -- reads both, so the product
suite runs from a checkout with the library installed beside it, as every
release lane runs it. Those modules are named here with what each reads, read
from their import of ``_tree``, so a product module that starts reading the
tree is one a reader sees.

Reading the import rather than a hand-written list is what makes the check
survive a new file. A test that arrives importing nothing is caught the day it
arrives, rather than at the next audit.

The same parse answers a second question about the suite, for the same reason:
a typing form subscripted with a runtime validator is a collection error on the
oldest Python this project supports, and the newest accepts it. A suite run only
on the newest version cannot see it.

LEDGER: every test file is a product test or a marked repository check
"""

from __future__ import annotations

import ast

import pytest

from _reason import is_a_reason
from _tree import ROOT

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

TESTS = ROOT / "tests"


def _imports_the_library(tree: ast.Module) -> bool:
    """Answer whether the module imports `valgebra` at any depth.

    Parsed rather than grepped: a mention inside a docstring, a comment or a
    string of expected output is not an import, and every one of those occurs in
    this suite.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] == "valgebra" for alias in node.names
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.split(".")[0] == "valgebra"
        ):
            return True
    return False


# The combinators that return a `Validator`. A validator is a runtime object,
# not a type, so `typing` refuses it where a type belongs -- on 3.10 loudly, at
# subscription time, which is a collection error rather than a test failure.
_COMBINATORS = frozenset(
    {
        "union",
        "intersection",
        "complement",
        "recursive",
        "instance_of",
        "Validator",
    }
)


def _validators_in_typing_forms(tree: ast.Module) -> list[str]:
    """List the `Annotated[...]` forms subscripted with a validator call.

    `Annotated[t, ...]` takes a *type*. On 3.10 `t` is checked when the form is
    built, so a validator there raises at import and takes the whole module out
    of collection; on 3.12 and later the check is looser and the same line runs.
    The schema a validator spells has a typing spelling too -- `Union[int, str]`
    for `union(int, str)` -- and that one is a type on every version.

    Only the head of the subscription is read. A validator *inside* a builtin
    generic (`list[nothing]`) builds a `types.GenericAlias`, which every
    supported version accepts.
    """
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id == "Annotated"):
            continue
        head = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
        if (
            isinstance(head, ast.Call)
            and isinstance(head.func, ast.Name)
            and head.func.id in _COMBINATORS
        ):
            found.append(f"line {head.lineno}: Annotated[{head.func.id}(...), ...]")
    return found


def _is_marked(tree: ast.Module, marker: str = "repository") -> bool:
    """Answer whether the module's `pytestmark` carries `pytest.mark.<marker>`.

    The last assignment is read, because it is the one pytest reads, and the
    marker as an attribute of `pytest.mark`, because the word appearing in the
    value is not a mark: a `skipif` whose reason says "needs no repository"
    marks nothing.
    """
    value = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            value = node.value
    return value is not None and any(
        isinstance(node, ast.Attribute)
        and node.attr == marker
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
        for node in ast.walk(value)
    )


def _modules() -> dict[str, tuple[bool, bool]]:
    modules = {}
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules[path.name] = (_imports_the_library(tree), _is_marked(tree))
    return modules


def test_every_test_file_falls_on_one_side_of_the_line() -> None:
    modules = _modules()
    # The glob is the detector: an empty universe would pass having read nothing.
    assert len(modules) >= 40, f"the test glob found only {sorted(modules)}"

    unmarked = sorted(
        name for name, (uses, marked) in modules.items() if not uses and not marked
    )
    assert not unmarked, (
        f"test files that neither import valgebra nor carry the marker: {unmarked}. "
        "Exercise the library, or mark the file `repository`."
    )

    mismarked = sorted(
        name for name, (uses, marked) in modules.items() if uses and marked
    )
    assert not mismarked, (
        f"product tests carrying the repository marker: {mismarked}. "
        "The marker takes them out of the product suite."
    )


#: The product modules that read the tree, each with what it reads there.
READS_THE_TREE: dict[str, str] = {
    "test_boundary_ledger.py": "the decidability and soundness pages it drives",
    "test_closure_ledger.py": "the variants of `Schema` it sorts into columns",
    "test_constraint_matrix.py": "the variants of `Constraint` and `Kind` it crosses",
    "test_documented_idioms.py": "the limits and schema-language pages it runs",
    "test_error_contract.py": "the error-model page whose promises it holds",
    "test_form_ledger.py": "the tables of the schema-language and refinement pages",
    "test_node_matrix.py": "the variants of `Schema`, and the stub's entry points",
    "test_public_surface.py": "the stub's positional-only parameters",
    "test_regex_dialect.py": "the refinement page's account of the pattern dialect",
    "test_relation_ledger.py": "the variants of `Schema`, and the binding's hooks",
}


def _reads_the_tree(tree: ast.Module) -> bool:
    """Answer whether the module imports `_tree`, the one way to the checkout."""
    return any(
        (isinstance(node, ast.ImportFrom) and node.module == "_tree")
        or (
            isinstance(node, ast.Import)
            and any(alias.name == "_tree" for alias in node.names)
        )
        for node in ast.walk(tree)
    )


def test_a_product_module_reading_the_tree_is_named() -> None:
    """The product suite reads the tree where a table says so, and nowhere else.

    Both directions: a product module that starts reading the checkout arrives
    here with no row, and a row whose module stopped reading it, or became a
    repository check, names a dependency the product suite does not have.
    """
    readers = set()
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        product = _imports_the_library(tree) and not _is_marked(tree)
        if product and _reads_the_tree(tree):
            readers.add(path.name)
    # The scan is the detector: no reader at all would pass both directions.
    assert readers, "no product module reads the tree; the import scan read nothing"
    unnamed = sorted(readers - set(READS_THE_TREE))
    assert not unnamed, (
        f"product modules reading the tree with no row: {unnamed}. Name what "
        "each reads in `READS_THE_TREE`, or read it from the library instead."
    )
    stale = sorted(set(READS_THE_TREE) - readers)
    assert not stale, f"rows naming no product module that reads the tree: {stale}"
    for name, what in READS_THE_TREE.items():
        assert is_a_reason(what), f"{name}: say what the module reads"


def test_the_product_suite_is_the_larger_half() -> None:
    # The partition is only worth drawing while the product half is the bulk of
    # the suite. If the repository checks ever outnumber it, the coverage the
    # suite reports is mostly about the project.
    modules = _modules()
    product = sum(1 for uses, _ in modules.values() if uses)
    assert product > len(modules) - product, (
        f"{product} product tests against {len(modules) - product} repository checks"
    )


def _climbs_from_itself(tree: ast.Module) -> list[int]:
    """Give the lines that read `__file__` for anything but the module's name.

    `Path(__file__).name` is a module naming itself, which a check skipping its
    own source does. Any other reading climbs from the file to a directory, and
    the directory a test climbs to is the checkout, which `tests/_tree.py`
    names for every module.
    """
    named = {
        id(node.value.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "name"
        and isinstance(node.value, ast.Call)
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "__file__"
    }
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "__file__"
        and id(node) not in named
    )


def test_the_checkout_is_found_in_one_place() -> None:
    """Every module reaches the tree through `tests/_tree.py`.

    Each module once spelled the climb for itself, in three spellings, and a
    module that climbs one level too few reads `tests/` as the root and finds
    nothing it looks for -- which a check that asserts it read something says
    loudly, and one that does not says as a pass.
    """
    offenders = {
        path.relative_to(TESTS).as_posix(): found
        for path in sorted(TESTS.rglob("*.py"))
        if path.name != "_tree.py"
        and (found := _climbs_from_itself(ast.parse(path.read_text(encoding="utf-8"))))
    }
    assert not offenders, (
        f"modules finding the checkout from their own file: {offenders}. Import "
        "`ROOT` from `_tree`, or a path it names."
    )


def test_a_module_naming_itself_is_not_one_finding_the_tree() -> None:
    # The readings the check turns on: a climb is one at any depth, and a module
    # asking its own name is not.
    climbs = ast.parse("ROOT = Path(__file__).resolve().parent.parent\n")
    assert _climbs_from_itself(climbs) == [1]
    beside = ast.parse("TESTS = Path(__file__).parent\n")
    assert _climbs_from_itself(beside) == [1]
    named = ast.parse("if path.name == Path(__file__).name:\n    pass\n")
    assert _climbs_from_itself(named) == []


def test_the_marker_is_registered() -> None:
    # `--strict-markers` turns an unregistered marker into an error, so the
    # boundary cannot be drawn by a typo that pytest silently accepts.
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "--strict-markers" in config
    assert "repository: " in config
    assert "interpreter: " in config


#: What a test reads off the interpreter running it, as `sys.<name>`.
_RUNNING = frozenset(
    {"version_info", "implementation", "stdlib_module_names", "_is_gil_enabled"}
)


def _carries(node: ast.FunctionDef, marker: str) -> bool:
    """Answer whether a test is decorated with `pytest.mark.<marker>`."""
    return any(
        ast.unparse(decorator).split("(")[0] == f"pytest.mark.{marker}"
        for decorator in node.decorator_list
    )


def _reads_the_interpreter(node: ast.FunctionDef) -> bool:
    """Answer whether a test, or a decorator on it, reads what is running it."""
    return any(
        isinstance(inner, ast.Attribute)
        and isinstance(inner.value, ast.Name)
        and inner.value.id == "sys"
        and inner.attr in _RUNNING
        for inner in ast.walk(node)
    )


def _unmarked_readers(tree: ast.Module) -> list[str]:
    """Name the repository checks that read the interpreter and do not say so."""
    audit, every_leg = _is_marked(tree), _is_marked(tree, "interpreter")
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("test_")
        and (audit or _carries(node, "repository"))
        and not (every_leg or _carries(node, "interpreter"))
        and _reads_the_interpreter(node)
    ]


def test_a_check_reading_the_interpreter_runs_on_every_leg() -> None:
    """The audit is read on one leg, and a check about the release on all of them.

    CI runs the repository checks on the floor alone, since the tree they read
    answers the same everywhere, and the ones marked `interpreter` on every
    leg. A table dated by release and held to the running one is a different
    check on each leg; read on the floor alone, it is the floor's check only,
    and one skipped below 3.12 is read on no leg at all.
    """
    offenders = {
        path.name: found
        for path in sorted(TESTS.glob("test_*.py"))
        if (found := _unmarked_readers(ast.parse(path.read_text(encoding="utf-8"))))
    }
    assert not offenders, (
        "repository checks that read the running interpreter and lack the "
        f"`interpreter` marker: {offenders}. Mark them, so every leg reads them."
    )


def test_a_reading_of_the_interpreter_is_read_from_the_syntax() -> None:
    # The readings the check turns on: an attribute of `sys` in the body or in a
    # decorator is one, the same words in a string are not, and a product test
    # runs on every leg already.
    audit = "pytestmark = pytest.mark.repository\n"
    gated = "@pytest.mark.skipif(sys.version_info < (3, 12), reason='')\n"
    assert _unmarked_readers(ast.parse(f"{audit}{gated}def test_x(): pass\n")) == [
        "test_x"
    ]
    quoted = f"{audit}def test_x(): 'sys.version_info'\n"
    assert _unmarked_readers(ast.parse(quoted)) == []
    said = f"{audit}@pytest.mark.interpreter\ndef test_x(): sys.version_info\n"
    assert _unmarked_readers(ast.parse(said)) == []
    product = "def test_x(): sys.stdlib_module_names\n"
    assert _unmarked_readers(ast.parse(product)) == []


def test_the_mark_is_read_from_pytest_mark_in_the_last_assignment() -> None:
    # The three readings the marker check turns on.
    head = "import pytest\n"
    named = head + 'pytestmark = pytest.mark.skipif(True, reason="no repository")\n'
    assert not _is_marked(ast.parse(named))
    listed = head + "pytestmark = [pytest.mark.slow, pytest.mark.repository]\n"
    assert _is_marked(ast.parse(listed))
    replaced = (
        head + "pytestmark = pytest.mark.repository\npytestmark = pytest.mark.slow\n"
    )
    assert not _is_marked(ast.parse(replaced))


def test_an_import_is_read_from_the_syntax_and_not_the_text() -> None:
    # The two readings this file's answer turns on.
    mentioned = ast.parse('"""valgebra is named here."""\nimport json\n')
    assert not _imports_the_library(mentioned)
    imported = ast.parse("from valgebra import schema\n")
    assert _imports_the_library(imported)
    submodule = ast.parse("import valgebra.testing\n")
    assert _imports_the_library(submodule)
    similar = ast.parse("import valgebra_vtjson\n")
    assert not _imports_the_library(similar)


def test_no_typing_form_is_subscripted_with_a_validator() -> None:
    # The failure this catches is a collection error, not a test failure: the
    # module raises on import and every test in it disappears, which on a green
    # newer interpreter looks like nothing at all.
    offenders = {}
    for path in sorted(TESTS.glob("test_*.py")):
        found = _validators_in_typing_forms(ast.parse(path.read_text(encoding="utf-8")))
        if found:
            offenders[path.name] = found
    assert not offenders, (
        f"typing forms subscripted with a runtime validator: {offenders}. "
        "Spell the schema as a typing form -- `Union[int, str]` for "
        "`union(int, str)` -- which every supported Python accepts."
    )


def test_the_validator_head_is_read_and_a_nested_one_is_not() -> None:
    # The two readings the check turns on. A validator at the head is the
    # failure; one inside a builtin generic builds a `types.GenericAlias`, which
    # every supported version accepts, and flagging it would refuse a spelling
    # the suite relies on.
    head = ast.parse("x = Annotated[union(int, str), Ge(0)]\n")
    assert _validators_in_typing_forms(head) == ["line 1: Annotated[union(...), ...]"]
    nested = ast.parse("x = Annotated[list[nothing], MinLen(1)]\n")
    assert _validators_in_typing_forms(nested) == []
    typed = ast.parse("x = Annotated[Union[int, str], Ge(0)]\n")
    assert _validators_in_typing_forms(typed) == []
    # A single-argument subscription is not a tuple, and the head is the whole
    # slice: the two shapes reach the same read by different paths.
    lone = ast.parse("x = Annotated[complement(int)]\n")
    assert _validators_in_typing_forms(lone) == [
        "line 1: Annotated[complement(...), ...]"
    ]
