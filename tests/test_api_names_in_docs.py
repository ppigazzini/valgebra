"""Every method a user page names on a validator is one `Validator` has.

A reader copies a method name off a page and calls it, and a name the class
does not have fails at the first call. The examples cannot catch one: they run,
but a name in a sentence runs nowhere. `docs/10-limits.md` would list
`is_disjoint_from` among the relations a validator answers, and the API has
`relation_to`, `is_subtype_of`, `is_equivalent` and `is_empty`.

A name reads as a method where the page writes it as one -- `.name`,
`Validator.name`, `Validator(...).name(...)`, `v.name`, or a call on a valgebra
export such as `anything.is_empty()` -- and where the page lists it in
backticks beside a name that is one. A list mixes kinds where its sentence
does, so the other bare names of such a list are held to the API at large: a
member of `Validator`, a valgebra export, a parameter of a `Validator` method,
or a builtin. Fenced code is left to the example runner, which executes it.
The API is read where the package declares it: the stub's `Validator` and the
package's `__all__`.

LEDGER: every method a user page names on a validator is one it has
"""

from __future__ import annotations

import ast
import builtins
import re
from pathlib import Path

import pytest

# A claim about the pages, which ship in no wheel, read against the API where
# the package declares it rather than through an import.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent

#: The declared surface: the stub the package ships for its extension, which
#: `tests/test_surface_outcomes.py` and the checker ledgers hold to the binding,
#: and the package's own export list.
STUB = ROOT / "python" / "valgebra" / "_valgebra.pyi"
PACKAGE = ROOT / "python" / "valgebra" / "__init__.py"

#: The user pages: the documentation set a reader of the API reads.
PAGES = sorted((ROOT / "docs").glob("*.md"))

#: A backticked span on one line.
SPAN = re.compile(r"`([^`\n]+)`")

#: Backticked spans joined as a list: by commas, `and` or `or`.
LIST = re.compile(r"`[^`\n]+`(?:(?:,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+)`[^`\n]+`)+")

#: An identifier as a page writes one bare: `name` or `name()`.
BARE = re.compile(r"\.?([a-z_][a-z0-9_]*)(?:\(\))?")


def _declared() -> tuple[set[str], set[str], set[str]]:
    """Read `Validator`'s members and parameters, and the package's exports."""
    stub = ast.parse(STUB.read_text(encoding="utf-8"))
    (validator,) = (
        node
        for node in stub.body
        if isinstance(node, ast.ClassDef) and node.name == "Validator"
    )
    defined = [node for node in validator.body if isinstance(node, ast.FunctionDef)]
    members = {node.name for node in defined} | set(dir(object))
    parameters = {
        arg.arg
        for node in defined
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    } - {"self", "cls"}
    package = ast.parse(PACKAGE.read_text(encoding="utf-8"))
    (exported,) = (
        ast.literal_eval(node.value)
        for node in package.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "__all__" for target in node.targets)
    )
    exports = {name for name in exported if not name.startswith("_")}
    return members, parameters, exports


_MEMBERS, _PARAMETERS, _EXPORTS = _declared()
_METHODS = {name for name in _MEMBERS if not name.startswith("_")}

#: A name written as a method: on the class, a built validator, the name the
#: pages give one, or a valgebra export, called or not -- or after a bare dot
#: and called, since a bare `.name` is as often an attribute or a path.
METHOD = re.compile(
    r"(?:(?:Validator|v|validator|"
    + "|".join(sorted(_EXPORTS))
    + r")(?:\([^`]*\))?\.([a-z_][a-z0-9_]*)(?:\([^`]*\))?"
    + r"|\.([a-z_][a-z0-9_]*)\([^`]*\))"
)


def _prose(page: Path) -> str:
    """Read a page's text with its fenced code removed."""
    return re.sub(
        r"^```.*?^```",
        "",
        page.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )


def _written_as_methods() -> list[tuple[str, str]]:
    """Each name a page writes in method syntax, with the page."""
    return [
        (page.name, found.group(1) or found.group(2))
        for page in PAGES
        for span in SPAN.findall(_prose(page))
        if (found := METHOD.fullmatch(span))
    ]


def _listed_beside_a_method() -> list[tuple[str, str, list[str]]]:
    """Each bare name a list holds beside a method, with the page and list."""
    found = []
    for page in PAGES:
        for listed in LIST.finditer(_prose(page)):
            spans = SPAN.findall(listed.group())
            names = [
                match.group(1) for span in spans if (match := BARE.fullmatch(span))
            ]
            if not any(name in _METHODS for name in names):
                continue
            found += [(page.name, name, spans) for name in names]
    return found


def test_every_name_written_as_a_method_is_one() -> None:
    unknown = sorted(
        f"{page}: .{name}"
        for page, name in _written_as_methods()
        if name not in _MEMBERS
    )
    assert not unknown, (
        f"pages call methods `Validator` does not have: {unknown}. It has "
        f"{sorted(_METHODS)}."
    )


def test_every_name_listed_beside_a_method_is_the_apis() -> None:
    known = _MEMBERS | _EXPORTS | _PARAMETERS | set(dir(builtins))
    unknown = sorted(
        f"{page}: `{name}` in {spans}"
        for page, name, spans in _listed_beside_a_method()
        if name not in known
    )
    assert not unknown, (
        f"pages list names beside `Validator`'s methods that the API does not "
        f"have: {unknown}. Its methods are {sorted(_METHODS)}."
    )


def test_the_scan_reads_the_names_that_are_there() -> None:
    """A scan that read nothing would pass every page, so it reads them all."""
    methods = {name for _, name in _written_as_methods()}
    listed = {name for _, name, _ in _listed_beside_a_method()}
    assert {"is_valid", "is_empty", "open"} <= methods, sorted(methods)
    assert {"relation_to", "is_subtype_of", "is_equivalent"} <= listed, sorted(listed)
    assert {"fail_fast"} <= _PARAMETERS, sorted(_PARAMETERS)
