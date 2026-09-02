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
from pathlib import Path

import pytest

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
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


def _is_marked(tree: ast.Module) -> bool:
    """Answer whether the module sets `pytestmark` to the repository marker."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            continue
        return "repository" in ast.dump(node.value)
    return False


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


def test_the_product_suite_is_the_larger_half() -> None:
    # The partition is only worth drawing while the product half is the bulk of
    # the suite. If the repository checks ever outnumber it, the coverage the
    # suite reports is mostly about the project.
    modules = _modules()
    product = sum(1 for uses, _ in modules.values() if uses)
    assert product > len(modules) - product, (
        f"{product} product tests against {len(modules) - product} repository checks"
    )


def test_the_marker_is_registered() -> None:
    # `--strict-markers` turns an unregistered marker into an error, so the
    # boundary cannot be drawn by a typo that pytest silently accepts.
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "--strict-markers" in config
    assert "repository: " in config


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
