"""Put the compiled extension's docstrings on the API reference.

`docs/16-api.md` is generated from the package by mkdocstrings, which reads
the package through griffe. griffe reads the built extension by importing it,
and it reads an object whose `__module__` is not the module it was found in as
an import of that object -- an alias to where it says it lives. `Validator`
and `ValidationError` say they live in `valgebra`, the public package, on
purpose: that is the path a caller imports them from, and the one a repr and
a traceback should name. So griffe writes `valgebra._valgebra.Validator` as
an alias to `valgebra.Validator`, whose own definition is an alias back to
the extension, and the two point at each other. griffe then merges the stub
it read beside the extension into the module, and where a stub class meets
an alias it keeps the stub class, which carries the types and no prose. The
page renders every heading with nothing under it, and `mkdocs build --strict`
exits 0 doing it.

`InspectTheCompiledClasses` is a griffe extension that breaks the cycle at
its source: when the inspector writes such an alias inside the extension
module, the extension inspects the object in place instead, as griffe would
have had the class named the module it is defined in. The stub merge that
follows then finds a class on each side: the docstrings and the members are
the runtime's, and the annotations, the overloads and the type parameters are
the stub's. Each stays written once, where it is.

`mkdocs.yml` names the extension under the handler's `extensions`. Run as a
script this checks a built page instead, because a build that exits 0 is not
evidence the page has content:

    python scripts/docs_stubs.py --check    # after `mkdocs build --strict`

The check imports the package and looks for the first line of each documented
object's docstring in `site/16-api/index.html`, so a phrase edited in Rust is
looked for as edited rather than as remembered here.
"""

from __future__ import annotations

import html
import importlib
import re
import sys
from pathlib import Path
from typing import Any

import griffe

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "valgebra"
COMPILED = "_valgebra"
PAGE = ROOT / "site" / "16-api" / "index.html"
#: The objects the page generates a section for, and whose docstrings it must
#: therefore carry.
GENERATED = ("Validator", "union", "intersection", "complement", "recursive")


class InspectTheCompiledClasses(griffe.Extension):
    """Inspect, in place, an extension object griffe would write as an alias."""

    def on_alias_instance(
        self,
        *,
        alias: griffe.Alias,
        node: griffe.ObjectNode,
        agent: griffe.Inspector,
        **_: Any,
    ) -> None:
        # The node is the object being inspected -- the module, or a class in
        # it -- and the alias is the member griffe is about to skip over. Two
        # aliases are the cycle: a member of the extension module pointing at
        # the public package, and a member of one of its classes pointing at
        # its own path, since a method descriptor names the extension as its
        # module while its class names the package.
        if not isinstance(node, griffe.ObjectNode):
            return
        here = agent.current.path
        if not (
            here == f"{PACKAGE}.{COMPILED}" or here.startswith(f"{PACKAGE}.{COMPILED}.")
        ):
            return
        if alias.target_path not in {f"{PACKAGE}.{alias.name}", f"{here}.{alias.name}"}:
            return
        agent.inspect(
            next(child for child in node.children if child.name == alias.name)
        )


def missing(page: str) -> list[str]:
    """Name every generated object whose docstring's first line the page lacks."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
    package = importlib.import_module(PACKAGE)
    absent = []
    for name in GENERATED:
        first = (getattr(package, name).__doc__ or "").strip().splitlines()[0]
        if not first or first not in text:
            absent.append(f"{name}: {first!r}")
    return absent


def main(argv: list[str]) -> int:
    if argv != ["--check"]:
        print(__doc__)
        return 2
    if not PAGE.exists():
        print(
            f"{PAGE.relative_to(ROOT)} is not built: run `mkdocs build --strict` first"
        )
        return 2
    absent = missing(PAGE.read_text(encoding="utf-8"))
    for line in absent:
        print(f"the API reference does not carry the docstring of {line}")
    carried = len(GENERATED) - len(absent)
    print(f"docs_stubs: {carried} of {len(GENERATED)} documented objects on the page")
    return 1 if absent else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
