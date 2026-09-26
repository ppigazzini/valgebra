"""Read every runnable documentation example with the checkers a reader runs.

`run_doc_examples.py` runs each fenced ```python block and reads the exit code,
which says the example is true and nothing about how it reads: an unused
import, a call a checker refuses, a spelling the page says a checker accepts. A
reader who copies a block into a project that runs ty, pyright or ruff meets
those first. So the same blocks, dedented the same way, are laid out one file
each under a scratch directory and read once by each of the three, at the
newest release the package supports -- the release ty reads the suite at.

Some diagnostics are the example's point: an argument of the wrong type the
page shows refused, a forward reference the frontend does not resolve, a
validator inside `list[...]`, which is this library's extension of the
spelling. Those are rows of `EXPECTED`: the checker, the page, the rule, how
many, and why. Anything else fails, and so does a row nothing reports any more,
because a stale row is a claim about the pages that stopped being true. An
expected diagnostic lives here rather than as an ignore comment in the block: a
reader copies the block and not the reason, and an ignore written for one
checker is noise to the other two.

ruff reads a block under the project's own selection less the rules an example
is exempt from, each named in `EXEMPT` with why.

Exit 0 when every diagnostic is expected and every row is reported.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_doc_examples.py"
PYPROJECT = ROOT / "pyproject.toml"

#: A diagnostic's identity: the checker, the page, and the rule.
Key = tuple[str, str, str]


class Expected(NamedTuple):
    """Diagnostics one page is meant to draw from one checker, and why."""

    checker: str
    page: str
    rule: str
    count: int
    reason: str


#: Every diagnostic the pages draw on purpose.
EXPECTED: tuple[Expected, ...] = (
    Expected(
        "ty",
        "CHANGELOG.md",
        "invalid-type-form",
        4,
        "the entries that introduced a float `Literal` and a validator inside "
        "`list[...]`, which are this library's extensions of the spelling",
    ),
    Expected(
        "ty",
        "README.md",
        "invalid-assignment",
        1,
        '`config.steps = "ten"` is the mutation the example shows caught',
    ),
    Expected(
        "ty",
        "docs/03-schema-language.md",
        "invalid-argument-type",
        1,
        '`Point(1, "y")` is the wrong argument the block shows refused',
    ),
    Expected(
        "ty",
        "docs/03-schema-language.md",
        "invalid-type-form",
        1,
        "`tuple[str, int, ...]` is the older spelling of a prefix and a tail, "
        "kept beside the `*tuple` one for the floor",
    ),
    Expected(
        "ty",
        "docs/03-schema-language.md",
        "unresolved-reference",
        1,
        '`list["Account"]` is the forward reference the frontend refuses to resolve',
    ),
    Expected(
        "ty",
        "docs/04-algebra.md",
        "invalid-type-form",
        1,
        "a float `Literal`, the extension of the spelling the page states",
    ),
    Expected(
        "ty",
        "docs/15-decidability.md",
        "invalid-type-form",
        10,
        "float `Literal`s and validators inside `list[...]` and `dict[...]`, the "
        "two extensions of the spelling the page decides over",
    ),
    Expected(
        "ty",
        "docs/16-api.md",
        "invalid-argument-type",
        1,
        "`validate_json(123)` is the wrong argument the table shows raise `TypeError`",
    ),
    Expected(
        "ty",
        "docs/17-boundaries.md",
        "unresolved-attribute",
        1,
        "`Model.validator = ...` sets an attribute the class does not declare, "
        "which is what the block shows",
    ),
    Expected(
        "pyright",
        "README.md",
        "reportAttributeAccessIssue",
        1,
        '`config.steps = "ten"` is the mutation the example shows caught',
    ),
    Expected(
        "pyright",
        "docs/03-schema-language.md",
        "reportArgumentType",
        1,
        '`Point(1, "y")` is the wrong argument the block shows refused',
    ),
    Expected(
        "pyright",
        "docs/03-schema-language.md",
        "reportInvalidTypeForm",
        1,
        "`tuple[str, int, ...]` is the older spelling of a prefix and a tail, "
        "kept beside the `*tuple` one for the floor",
    ),
    Expected(
        "pyright",
        "docs/03-schema-language.md",
        "reportUndefinedVariable",
        1,
        '`list["Account"]` is the forward reference the frontend refuses to resolve',
    ),
    Expected(
        "pyright",
        "docs/04-algebra.md",
        "reportAttributeAccessIssue",
        2,
        "`int | Validator(str)` and `Validator(int) | str | None`: pyright "
        "resolves a union written from a type through `type.__or__` to "
        "`types.UnionType` and does not read the validator's `__ror__`; ty and "
        "mypy do, and the example runs",
    ),
    Expected(
        "pyright",
        "docs/16-api.md",
        "reportArgumentType",
        1,
        "`validate_json(123)` is the wrong argument the table shows raise `TypeError`",
    ),
    Expected(
        "pyright",
        "docs/17-boundaries.md",
        "reportAttributeAccessIssue",
        1,
        "`Model.validator = ...` sets an attribute the class does not declare, "
        "which is what the block shows",
    ),
    Expected(
        "ruff",
        "docs/03-schema-language.md",
        "F821",
        1,
        '`list["Account"]` is the forward reference the frontend refuses to resolve',
    ),
    Expected(
        "ruff",
        "docs/03-schema-language.md",
        "N803",
        1,
        "the lambda's parameter is `X` because that is how the repr the block "
        "shows spells it",
    ),
    Expected(
        "ruff",
        "docs/03-schema-language.md",
        "UP035",
        1,
        "`NotRequired` comes from `typing_extensions` beside the `TypedDict` that "
        "takes `closed=True`, which `typing` has on no supported release; the "
        "floor's `typing` has neither",
    ),
    Expected(
        "ruff",
        "docs/03-schema-language.md",
        "UP045",
        1,
        "`Optional[int]` is shown read beside `int | None`, as a spelling a "
        "reader's code carries",
    ),
    Expected(
        "ruff",
        "docs/08-error-model.md",
        "S301",
        1,
        "the block pickles a validator to show the refusal",
    ),
    Expected(
        "ruff",
        "docs/16-api.md",
        "UP037",
        1,
        '`next: "Node"` is the quoted forward reference the block shows refused',
    ),
)

#: The rules an example is exempt from, each with why. Passed as `--ignore`,
#: so everything else in the project's selection applies to a block as it
#: applies to the tree.
EXEMPT: dict[str, str] = {
    "D": "a snippet has no docstring to write",
    "INP001": "a snippet is not a package",
    "S101": "an example asserts what its page states; the runner's ledger requires it",
    "PT": "an example is not a test; these rules read its `except ... assert` as one",
    "ANN": "a snippet annotates what its page is about and nothing more",
    "EM": 'an example raises `AssertionError("...")` inline, as a reader writes it',
    "TRY003": 'an example raises `AssertionError("...")` inline, as a reader writes it',
    "FBT": "an example passes a bool where the page shows one",
    "PLR2004": "a constant is the example, not a magic number",
    "T201": "an example prints what it shows",
    "TC": "a `TYPE_CHECKING` gate saves a module an import; a snippet has no module",
}

_TY_LINE = re.compile(r"^(?P<file>.+?):\d+:\d+: (?P<level>\w+)\[(?P<rule>[\w-]+)\]")
_RUFF_LINE = re.compile(r"^(?P<file>.+?):\d+:\d+: (?P<rule>\S+)")


def runner() -> ModuleType:
    """Import the example runner, so its blocks are the blocks read here."""
    spec = importlib.util.spec_from_file_location("valgebra_doc_examples", RUNNER)
    if spec is None or spec.loader is None:
        message = f"cannot import {RUNNER}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def newest_release() -> str:
    """Give the release the examples are read at: the one ty reads the suite at."""
    found = re.search(
        r'^python-version = "(\d+\.\d+)"$', PYPROJECT.read_text("utf-8"), re.MULTILINE
    )
    if found is None:
        message = "pyproject.toml names no [tool.ty.environment] python-version"
        raise RuntimeError(message)
    return found[1]


def _page(scratch: Path, reported: str) -> str:
    """Give the page a scratch file was cut from, as the tree names it."""
    relative = Path(reported).resolve().relative_to(scratch.resolve())
    return relative.parent.with_suffix(".md").as_posix()


def _output(argv: list[str]) -> str:
    return subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout


def read_ty(scratch: Path) -> Counter[Key]:
    """Read the blocks with ty, under the project's own configuration."""
    out = _output(
        [
            sys.executable,
            "-m",
            "ty",
            "check",
            "--project",
            str(ROOT),
            "--output-format",
            "concise",
            str(scratch),
        ]
    )
    found: Counter[Key] = Counter()
    for line in out.splitlines():
        if (match := _TY_LINE.match(line)) and match["level"] != "info":
            found[("ty", _page(scratch, match["file"]), match["rule"])] += 1
    return found


def read_pyright(scratch: Path, release: str) -> Counter[Key]:
    out = _output(
        [
            sys.executable,
            "-m",
            "pyright",
            # The environment pyright reads is the one running this script, so
            # a venv that is not on the path still resolves the package.
            "--pythonpath",
            sys.executable,
            "--pythonversion",
            release,
            "--outputjson",
            str(scratch),
        ]
    )
    found: Counter[Key] = Counter()
    for item in json.loads(out)["generalDiagnostics"]:
        if item["severity"] != "information":
            found[("pyright", _page(scratch, item["file"]), str(item.get("rule")))] += 1
    return found


def read_ruff(scratch: Path, release: str) -> Counter[Key]:
    out = _output(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            str(PYPROJECT),
            "--target-version",
            f"py{release.replace('.', '')}",
            "--output-format",
            "concise",
            "--ignore",
            ",".join(EXEMPT),
            str(scratch),
        ]
    )
    found: Counter[Key] = Counter()
    for line in out.splitlines():
        if match := _RUFF_LINE.match(line):
            found[("ruff", _page(scratch, match["file"]), match["rule"])] += 1
    return found


def read() -> tuple[int, Counter[Key]]:
    """Lay the blocks out and read them: how many, and what each checker says."""
    release = newest_release()
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        count = 0
        for page, index, block in runner().examples():
            target = (
                scratch / page.relative_to(ROOT).with_suffix("") / f"block_{index}.py"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(block, encoding="utf-8")
            count += 1
        found = (
            read_ty(scratch)
            + read_pyright(scratch, release)
            + read_ruff(scratch, release)
        )
    return count, found


def problems(found: Counter[Key]) -> list[str]:
    """Name every diagnostic no row expects, and every row nothing reports."""
    expected: Counter[Key] = Counter()
    for row in EXPECTED:
        expected[(row.checker, row.page, row.rule)] += row.count
    return [
        f"{checker} reports {rule} on {page} {found[key]} time(s); the ledger "
        f"expects {expected[key]}. Fix the example, or add the row with its reason."
        for key in sorted(found | expected)
        if found[key] != expected[key]
        for checker, page, rule in [key]
    ]


def main() -> int:
    count, found = read()
    wrong = problems(found)
    for line in wrong:
        print(line)
    print(
        f"read {count} example(s) under ty, pyright and ruff: {sum(found.values())} "
        f"diagnostic(s), {len(wrong)} unexpected or stale"
    )
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
