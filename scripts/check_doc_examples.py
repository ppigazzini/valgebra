"""Read every runnable documentation example with the checkers a reader runs.

`run_doc_examples.py` runs each fenced ```python block and reads the exit code,
which says the example is true and nothing about how it reads: an unused
import, a call a checker refuses, a spelling the page says a checker accepts. A
reader who copies a block into a project that runs ty, mypy, pyright or ruff
meets those first. So the same blocks, dedented the same way, are laid out one
file each under a scratch directory and read once by each of the four, at the
newest release the package supports -- the release ty reads the suite at.

Some diagnostics are the example's point: an argument of the wrong type the
page shows refused, a forward reference the frontend does not resolve, a
validator inside `list[...]`, which is this library's extension of the
spelling. Those are rows of `EXPECTED`: the checker, the page, the rule, how
many, and why. Anything else fails, and so does a row nothing reports any more,
because a stale row is a claim about the pages that stopped being true. An
expected diagnostic lives here rather than as an ignore comment in the block: a
reader copies the block and not the reason, and an ignore written for one
checker is noise to the other three.

ruff reads a block under the project's own selection less the rules an example
is exempt from, each named in `EXEMPT` with why. mypy reads it in its default
mode: a snippet's unannotated `def` is how a reader writes one, and `--strict`
reports every such line. And mypy parses with the interpreter that runs it,
whatever release it is told to check, where the other three carry parsers of
their own. A syntax error stops its whole run, so it reads the blocks only on an
interpreter that parses every one -- the one the examples run on, or newer --
and says so where it cannot.

Exit 0 when every diagnostic is expected and every row is reported.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if sys.version_info >= (3, 11):
    import tomllib
else:  # the floor, where pytest's own dependency supplies the parser
    import tomli as tomllib

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


# The deliberate diagnostics most pages draw from more than one checker, each
# reason written once.
_MUTATION = '`config.steps = "ten"` is the mutation the example shows caught'
_WRONG_POINT = '`Point(1, "y")` is the wrong argument the block shows refused'
_SHORT_TUPLE = (
    "`tuple[str, int, ...]` is this library's shorter spelling of a prefix and a "
    "tail, which 3.10's `typing` can write, shown beside the spec's `*tuple`"
)
_FORWARD_REFERENCE = (
    '`list["Account"]` is the forward reference the frontend refuses to resolve'
)
_WRONG_JSON = (
    "`validate_json(123)` is the wrong argument the table shows raise `TypeError`"
)
_UNDECLARED = (
    "`Model.validator = ...` sets an attribute the class does not declare, "
    "which is what the block shows"
)
_CLASS_ON_THE_LEFT = (
    "`int | Validator(str)` shows the reflected operand at work: `type.__or__` "
    "declines and the validator's `__ror__` builds the union. mypy and pyright "
    "read the result as `types.UnionType | type[int]` and refuse the method "
    "called on it, once per member; only ty reads the `__ror__`, and "
    "`docs/18-static-checking.md` says to write the validator on the left"
)
_VALIDATOR_IN_A_TYPE = (
    "a validator inside `list[...]`, this library's extension of the spelling; "
    "mypy reads a variable there as no type, and passes a float `Literal`"
)

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
    Expected("ty", "README.md", "invalid-assignment", 1, _MUTATION),
    Expected(
        "ty", "docs/03-schema-language.md", "invalid-argument-type", 1, _WRONG_POINT
    ),
    Expected(
        "ty",
        "docs/03-schema-language.md",
        "invalid-type-form",
        2,
        f"{_SHORT_TUPLE}; and `Literal[positive]` is the constant spelling of a "
        "function this library reads, which the refusal of `Validator(positive)` "
        "names and the spec's `Literal` does not allow",
    ),
    Expected(
        "ty",
        "docs/03-schema-language.md",
        "unresolved-reference",
        1,
        _FORWARD_REFERENCE,
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
        11,
        "float `Literal`s and validators inside `list[...]` and `dict[...]`, the "
        "two extensions of the spelling the page decides over",
    ),
    Expected("ty", "docs/16-api.md", "invalid-argument-type", 1, _WRONG_JSON),
    Expected("ty", "docs/17-boundaries.md", "unresolved-attribute", 1, _UNDECLARED),
    Expected("mypy", "CHANGELOG.md", "valid-type", 2, _VALIDATOR_IN_A_TYPE),
    Expected("mypy", "README.md", "assignment", 1, _MUTATION),
    Expected("mypy", "docs/03-schema-language.md", "arg-type", 1, _WRONG_POINT),
    Expected("mypy", "docs/03-schema-language.md", "misc", 1, _SHORT_TUPLE),
    Expected(
        "mypy", "docs/03-schema-language.md", "name-defined", 1, _FORWARD_REFERENCE
    ),
    Expected("mypy", "docs/04-algebra.md", "union-attr", 2, _CLASS_ON_THE_LEFT),
    Expected(
        "mypy",
        "docs/15-decidability.md",
        "valid-type",
        7,
        "validators inside `list[...]`, and a float `Literal` as a type argument, "
        "which mypy refuses there: the two extensions of the spelling the page "
        "decides over",
    ),
    Expected("mypy", "docs/16-api.md", "arg-type", 1, _WRONG_JSON),
    Expected("mypy", "docs/17-boundaries.md", "attr-defined", 1, _UNDECLARED),
    Expected("pyright", "README.md", "reportAttributeAccessIssue", 1, _MUTATION),
    Expected(
        "pyright", "docs/03-schema-language.md", "reportArgumentType", 1, _WRONG_POINT
    ),
    Expected(
        "pyright",
        "docs/03-schema-language.md",
        "reportInvalidTypeForm",
        1,
        _SHORT_TUPLE,
    ),
    Expected(
        "pyright",
        "docs/03-schema-language.md",
        "reportUndefinedVariable",
        1,
        _FORWARD_REFERENCE,
    ),
    Expected(
        "pyright",
        "docs/04-algebra.md",
        "reportAttributeAccessIssue",
        2,
        _CLASS_ON_THE_LEFT,
    ),
    Expected("pyright", "docs/16-api.md", "reportArgumentType", 1, _WRONG_JSON),
    Expected(
        "pyright", "docs/17-boundaries.md", "reportAttributeAccessIssue", 1, _UNDECLARED
    ),
    Expected("ruff", "docs/03-schema-language.md", "F821", 1, _FORWARD_REFERENCE),
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
        "takes `closed=True`, which `typing` has only from 3.15; the floor's "
        "`typing` has neither",
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
        "docs/05-refinements.md",
        "UP035",
        1,
        "`deprecated` comes from `typing_extensions`, which serves it on every "
        "release the page covers; the floor's `warnings` has it only from 3.13",
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
_MYPY_LINE = re.compile(r"^(?P<file>.+?):\d+: error: .*\[(?P<rule>[\w-]+)\]$")


class Blocks(NamedTuple):
    """The blocks laid out one module each, and the page each was cut from.

    Flat, with the page in the module's name: mypy names a module by its file
    and refuses two of one name, so `block_1.py` under two page directories is
    a duplicate to it, and a directory named for a page is no package name.
    """

    directory: Path
    pages: dict[str, str]
    """A module's name, and the page as the tree names it."""
    unparsed: tuple[str, ...]
    """The blocks the running interpreter cannot parse, by page and index."""

    def page(self, reported: str) -> str:
        """Give the page a checker's reported file was cut from."""
        return self.pages[Path(reported).stem]


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
    with PYPROJECT.open("rb") as handle:
        manifest = tomllib.load(handle)
    return manifest["tool"]["ty"]["environment"]["python-version"]


def _output(argv: list[str], **env: str) -> str:
    return subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    ).stdout


def read_ty(blocks: Blocks) -> Counter[Key]:
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
            str(blocks.directory),
        ]
    )
    found: Counter[Key] = Counter()
    for line in out.splitlines():
        if (match := _TY_LINE.match(line)) and match["level"] != "info":
            found[("ty", blocks.page(match["file"]), match["rule"])] += 1
    return found


def read_mypy(blocks: Blocks, release: str) -> Counter[Key]:
    """Read the blocks with mypy, against the tree's stub rather than an install."""
    out = _output(
        [
            sys.executable,
            "-m",
            "mypy",
            "--python-version",
            release,
            "--no-error-summary",
            # No cache: mypy replays a cached module's errors under the path of
            # the run that wrote them, and each run lays the blocks out anew.
            "--cache-dir",
            os.devnull,
            str(blocks.directory),
        ],
        MYPYPATH=str(ROOT / "python"),
    )
    found: Counter[Key] = Counter()
    for line in out.splitlines():
        if match := _MYPY_LINE.match(line):
            found[("mypy", blocks.page(match["file"]), match["rule"])] += 1
    return found


def read_pyright(blocks: Blocks, release: str) -> Counter[Key]:
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
            str(blocks.directory),
        ]
    )
    found: Counter[Key] = Counter()
    for item in json.loads(out)["generalDiagnostics"]:
        if item["severity"] != "information":
            found[("pyright", blocks.page(item["file"]), str(item.get("rule")))] += 1
    return found


def read_ruff(blocks: Blocks, release: str) -> Counter[Key]:
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
            str(blocks.directory),
        ]
    )
    found: Counter[Key] = Counter()
    for line in out.splitlines():
        if match := _RUFF_LINE.match(line):
            found[("ruff", blocks.page(match["file"]), match["rule"])] += 1
    return found


def lay_out(directory: Path) -> Blocks:
    """Write every block as a module of its own under `directory`."""
    pages: dict[str, str] = {}
    unparsed: list[str] = []
    for page, index, block in runner().examples():
        named = page.relative_to(ROOT).as_posix()
        module = f"{re.sub(r'[^0-9a-z]+', '_', named.lower())}_block_{index}"
        if module in pages:
            message = f"{named} and {pages[module]} lay out as one module"
            raise RuntimeError(message)
        pages[module] = named
        (directory / f"{module}.py").write_text(block, encoding="utf-8")
        try:
            ast.parse(block)
        except SyntaxError:
            unparsed.append(f"{named} block {index}")
    return Blocks(directory, pages, tuple(unparsed))


class Reading(NamedTuple):
    """What the checkers said about the blocks, and which of them could read."""

    count: int
    found: Counter[Key]
    checkers: tuple[str, ...]
    unparsed: tuple[str, ...]
    """The blocks that kept mypy from reading, if it did not."""


def read() -> Reading:
    """Lay the blocks out and read them with every checker that can."""
    release = newest_release()
    with tempfile.TemporaryDirectory() as directory:
        blocks = lay_out(Path(directory))
        found = read_ty(blocks) + read_pyright(blocks, release)
        found += read_ruff(blocks, release)
        checkers = ("ty", "pyright", "ruff")
        if not blocks.unparsed:
            found += read_mypy(blocks, release)
            checkers = ("ty", "mypy", "pyright", "ruff")
    return Reading(len(blocks.pages), found, checkers, blocks.unparsed)


def problems(found: Counter[Key], checkers: tuple[str, ...]) -> list[str]:
    """Name every diagnostic no row expects, and every row nothing reports.

    A row is held only where its checker read the blocks.
    """
    expected: Counter[Key] = Counter()
    for row in EXPECTED:
        if row.checker in checkers:
            expected[(row.checker, row.page, row.rule)] += row.count
    return [
        f"{checker} reports {rule} on {page} {found[key]} time(s); the ledger "
        f"expects {expected[key]}. Fix the example, or add the row with its reason."
        for key in sorted(found | expected)
        if found[key] != expected[key]
        for checker, page, rule in [key]
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.parse_args(argv)
    reading = read()
    wrong = problems(reading.found, reading.checkers)
    for line in wrong:
        print(line)
    if reading.unparsed:
        print(
            f"mypy did not read: Python {sys.version_info.major}."
            f"{sys.version_info.minor} cannot parse {', '.join(reading.unparsed)}"
        )
    print(
        f"read {reading.count} example(s) under {', '.join(reading.checkers)}: "
        f"{sum(reading.found.values())} diagnostic(s), "
        f"{len(wrong)} unexpected or stale"
    )
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
