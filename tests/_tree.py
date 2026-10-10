"""The tree a test reads, found and parsed in one place.

A check that reads the repository -- a page, the IR, a workflow -- names it
from here rather than from its own `__file__`, so the suite has one notion of
where the checkout is, and `tests/test_suite_partition.py` refuses a second.
A workflow is parsed once per run and every caller gets a copy of its own, so
one check filling in a variable leaves the next reading the file as written.
A Rust enum is read by one pattern, so a change to how `rustfmt` lays an enum
out moves this file and no ledger reading the variants. A gate script is loaded
the way it runs, with its directory on the import path.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from types import ModuleType

#: The root of the checkout the suite runs from.
ROOT = Path(__file__).resolve().parent.parent

#: Where the workflows live, and the one most checks read.
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "ci.yml"

#: The core's term and its kind partition, whose enums several ledgers read.
IR = ROOT / "crates" / "valgebra-core" / "src" / "ir.rs"
KIND = ROOT / "crates" / "valgebra-core" / "src" / "kind.rs"

#: The gate scripts, which are files rather than a package.
SCRIPTS = ROOT / "scripts"


def scripts() -> list[Path]:
    """Give every script under `scripts/`, which is every file but a shared one.

    A leading underscore marks a module the scripts import rather than one a
    lane runs, as it does in `tests/`: it takes no flags and answers in no code.
    """
    return sorted(
        path for path in SCRIPTS.glob("*.py") if not path.name.startswith("_")
    )


def lay_script(name: str, tree: Path) -> Path:
    """Copy `scripts/<name>.py` into `tree`'s `scripts/`, with `_lib.py` beside it.

    A script finds its root where `_lib.py` sits, so the copy reads `tree` as
    its checkout only with the module copied too; without it the copy fails to
    import rather than reading this one.
    """
    (tree / "scripts").mkdir(parents=True, exist_ok=True)
    for module in (name, "_lib"):
        source = SCRIPTS / f"{module}.py"
        (tree / "scripts" / source.name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return tree / "scripts" / f"{name}.py"


def load_script(name: str, module: str | None = None) -> ModuleType:
    """Import `scripts/<name>.py` by path, registered as `module` or as itself.

    A script runs with its own directory first on the import path, which is how
    it finds `_lib` and a sibling, so the directory goes there before it runs
    here too. It is registered before it runs: a script defining a dataclass has
    the class read back out of `sys.modules` under its own `__module__`.
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    registered = module or name
    spec = importlib.util.spec_from_file_location(registered, SCRIPTS / f"{name}.py")
    assert spec is not None, name
    assert spec.loader is not None, name
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[registered] = loaded
    spec.loader.exec_module(loaded)
    return loaded


@functools.cache
def _parsed(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file of the tree's: the parse is shared, the copy the caller's.

    The caller fills in a variable or cuts a list down, and the next caller reads
    the file as written.
    """
    return copy.deepcopy(_parsed(path))


def read_workflow(path: Path = WORKFLOW) -> dict[str, Any]:
    """Parse a workflow or an action, `ci.yml` unless another is named."""
    return read_yaml(path)


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
