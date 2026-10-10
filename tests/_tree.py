"""The tree a test reads, found and parsed in one place.

A check that reads the repository -- a page, the IR, a workflow -- names it
from here rather than from its own `__file__`, so the suite has one notion of
where the checkout is, and `tests/test_suite_partition.py` refuses a second.
A workflow is parsed once per run and every caller gets a copy of its own, so
one check filling in a variable leaves the next reading the file as written.
A Rust enum is read by one pattern, so a change to how `rustfmt` lays an enum
out moves this file and no ledger reading the variants.
"""

from __future__ import annotations

import copy
import functools
import re
from pathlib import Path
from typing import Any

import yaml

#: The root of the checkout the suite runs from.
ROOT = Path(__file__).resolve().parent.parent

#: Where the workflows live, and the one most checks read.
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "ci.yml"

#: The core's term and its kind partition, whose enums several ledgers read.
IR = ROOT / "crates" / "valgebra-core" / "src" / "ir.rs"
KIND = ROOT / "crates" / "valgebra-core" / "src" / "kind.rs"


@functools.cache
def _parsed(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_workflow(path: Path = WORKFLOW) -> dict[str, Any]:
    """Parse a workflow or an action, `ci.yml` unless another is named.

    The parse is shared and the copy is the caller's, to fill in or cut down.
    """
    return copy.deepcopy(_parsed(path))


def runs(path: Path = WORKFLOW) -> list[str]:
    """Give the shell of every step in a workflow, in the order the file lists them.

    Read from the parse rather than the text, so a comment quoting a flag is
    not a step passing it.
    """
    return [
        str(step["run"])
        for job in (read_workflow(path).get("jobs") or {}).values()
        for step in job.get("steps") or []
        if isinstance(step, dict) and "run" in step
    ]


#: A variant of an enum `rustfmt` laid out: four spaces, the name, then a
#: payload, a body or the comma. A doc comment or an attribute starts with
#: something else, and a field of a struct variant sits deeper.
_VARIANT = re.compile(r"^    ([A-Z][A-Za-z0-9]*)[ ({,]", re.MULTILINE)

#: The enums the suite reads, each with the file holding it and variants it
#: must find. The scan is the detector: a pattern that matched nothing would
#: leave every check over the variants quantified over nothing and passing.
_ENUMS: dict[str, tuple[Path, frozenset[str]]] = {
    "Schema": (IR, frozenset({"Anything", "Nothing", "Union", "Complement"})),
    "Constraint": (IR, frozenset({"Ge", "Lt", "MinLen", "Regex"})),
    "Kind": (KIND, frozenset({"NoneType", "Int", "Str", "Dict"})),
}


def variants(enum: str) -> set[str]:
    """Give the variants of one of the core's `pub enum`s, read from its source."""
    path, known = _ENUMS[enum]
    body = re.search(
        rf"^pub enum {enum} \{{$(.*?)^\}}$",
        path.read_text(encoding="utf-8"),
        re.DOTALL | re.MULTILINE,
    )
    assert body, f"{path.name} has no `pub enum {enum}`"
    found = set(_VARIANT.findall(body.group(1)))
    assert known <= found, f"the scan of `{enum}` found only {sorted(found)}"
    return found
