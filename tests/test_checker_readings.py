"""One typed surface, three checkers, and the readings they do not share.

`tests/typing/consumer.py` puts every public name through `assert_type`, and the
lane runs it under `mypy --strict` and pyright; `ty check` reads the same tree.
An `assert_type` row therefore holds a reading all three agree on, and a
reading they do not agree on cannot be written as one: whichever checker
disagrees fails the lane. Those readings are the ones a downstream user meets
first -- a `NewType` that is typed under one checker and `object` under the
other two -- and until now they lived in a report, which is a reading nobody
holds.

So each disagreement is a fixture under `tests/typing/readings/`: one file, one
expression, one `reveal_type`. This ledger runs the three checkers over the
directory, reads the type each reveals and the diagnostics each reports, and
holds them to the rows below. A checker release that changes its reading
reddens its cell, and the cell's row says what the change means for the stub:
whether an overload can now say more, or whether nothing moves and the row is
rewritten.

The checkers read the stub from `python/`, not from the installed package, so
a change to the tree's stub is what the rows answer for. One run per checker
over every fixture at once: three subprocesses, not three per row.

LEDGER: every reading the checkers do not share is held per checker by a fixture
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

# The three checkers are the dev group's, and the PyPy lane installs only what
# the product suite reads: there this ledger has nothing to run and says so,
# where a checker's absence would otherwise read as an empty reading.
for _checker in ("ty", "mypy", "pyright"):
    pytest.importorskip(_checker)

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
READINGS = ROOT / "tests" / "typing" / "readings"
#: The tree's own package, which every checker is pointed at ahead of the
#: installed one, so the rows answer for the stub as written here.
PACKAGE = ROOT / "python"
#: The release the readings are taken at. A spelling's reading does not move
#: between releases -- the consumer is what runs at both ends -- and the floor
#: is where a caller on the oldest supported release reads it.
FLOOR: str = json.loads((ROOT / "tests" / "floor_names.json").read_text("utf-8"))[
    "floor"
]
CHECKERS = ("ty", "mypy", "pyright")


class Reading(NamedTuple):
    """One fixture, what each checker reads it as, and what a change means."""

    fixture: str
    """The fixture's file stem under `tests/typing/readings/`."""
    ty: str
    mypy: str
    pyright: str
    """The revealed type, then every diagnostic the checker reports, as
    `level[code]`, sorted and joined with `; `. Module qualifiers are stripped."""
    moves: str
    """What a checker changing its cell means for the stub."""


#: Every disagreement, measured against the stub as committed.
ROWS: tuple[Reading, ...] = (
    Reading(
        "any_schema",
        ty="Validator[Unknown]",
        mypy="Validator[Any]",
        pyright="Validator[object]",
        moves=(
            "`Any` as a schema is the top. mypy and ty keep the parameter "
            "gradual; pyright reads `type[Any]` as no type and falls to the "
            "`object` overload. Every reading admits every value, so nothing in "
            "the stub moves on a change here: rewrite the cell."
        ),
    ),
    Reading(
        "bare_receiver",
        ty="Unknown",
        mypy="object",
        pyright="Any",
        moves=(
            "A bare `Validator` is `Validator[Any]`, and the receiver overloads "
            "are ambiguous over it: ty and pyright answer the gradual type by "
            "the spec's ambiguity rule, mypy the first overload. Nothing in the "
            "stub moves on a change here; a caller who wants a type writes the "
            "parameter typed, as `docs/18-static-checking.md` says."
        ),
    ),
    Reading(
        "newtype",
        ty="Validator[object]",
        mypy="Validator[UserId]",
        pyright="Validator[object]",
        moves=(
            "A `NewType` is a callable at runtime and a distinct type to a "
            "checker; mypy matches it against `type[_S]` and the other two do "
            "not. The frontend reads it as its supertype's set, so the typed "
            "reading is the narrower and sound one. When ty or pyright join "
            "mypy, the `NewType` row of `docs/18-static-checking.md` moves."
        ),
    ),
    Reading(
        "protocol",
        ty="Validator[HasX]",
        mypy="Validator[object]",
        pyright="Validator[HasX]",
        moves=(
            "mypy does not match a protocol class against `type[_S]`; ty and "
            "pyright do. A protocol validator is `isinstance` over the "
            "protocol, which is the set the type names, so either reading is "
            "sound. When mypy joins the other two, the protocol row of "
            "`docs/18-static-checking.md` moves."
        ),
    ),
    Reading(
        "typeform_overload",
        ty="Unknown; error[invalid-type-form]",
        mypy="object",
        pyright="object",
        moves=(
            "ty evaluates an `object`-typed argument against a `TypeForm` "
            "overload as a type expression and reports it, where the overload "
            "spec says to record the failure and try the next overload; mypy "
            "and pyright do that. The constructor's `type[_S]` overload in "
            "`_valgebra.pyi` is written that way because of it. When ty's cell "
            "reads `object` with no diagnostic, that overload can become "
            "`TypeForm[_S]`, and a union written with `|`, a `Literal` and an "
            "`Annotated` schema read as the type each names under ty and "
            "pyright; mypy still reads `int | None` as `object` inside an "
            "overloaded call."
        ),
    ),
    Reading(
        "variadic_union",
        ty="Validator[int | str]",
        mypy="Validator[object]",
        pyright="Validator[object]",
        moves=(
            "Only ty solves a variadic `Validator[_S]` over two typed "
            "arguments to the union of their types; mypy and pyright fall to "
            "the `object` overload. Both are sound. When one of the two joins "
            "ty, the consumer can assert the union for `union(a, b)` as it does "
            "for `a | b`."
        ),
    ),
)

_MYPY_LINE = re.compile(r"^(?P<file>.+?):\d+: (?P<level>\w+): (?P<message>.*)$")
_MYPY_CODE = re.compile(r"\s+\[(?P<code>[\w-]+)\]$")
_TY_LINE = re.compile(
    r"^(?P<file>.+?):\d+:\d+: (?P<level>\w+)\[(?P<code>[\w-]+)\] (?P<message>.*)$"
)
_QUALIFIER = re.compile(r"\b(?:\w+\.)+(?=\w)")


def _fixtures() -> list[Path]:
    return sorted(READINGS.glob("*.py"))


def _run(argv: list[str], **env: str) -> str:
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    )
    return done.stdout


def _unqualified(spelling: str) -> str:
    """Strip module qualifiers, which the three checkers print differently."""
    return _QUALIFIER.sub("", spelling)


def _join(
    revealed: dict[str, str], diagnostics: dict[str, list[str]]
) -> dict[str, str]:
    return {
        stem: "; ".join(
            [revealed.get(stem, "nothing revealed"), *sorted(diagnostics.get(stem, []))]
        )
        for stem in {*revealed, *diagnostics}
    }


def _ty() -> dict[str, str]:
    out = _run(
        [
            sys.executable,
            "-m",
            "ty",
            "check",
            "--output-format",
            "concise",
            "--python-version",
            FLOOR,
            "--extra-search-path",
            str(PACKAGE),
            *map(str, _fixtures()),
        ]
    )
    revealed: dict[str, str] = {}
    diagnostics: dict[str, list[str]] = {}
    for line in out.splitlines():
        if not (found := _TY_LINE.match(line)):
            continue
        stem = Path(found["file"]).stem
        if found["code"] == "revealed-type":
            revealed[stem] = _unqualified(found["message"].split("`")[1])
        else:
            diagnostics.setdefault(stem, []).append(
                f"{found['level']}[{found['code']}]"
            )
    return _join(revealed, diagnostics)


def _mypy() -> dict[str, str]:
    out = _run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--python-version",
            FLOOR,
            *map(str, _fixtures()),
        ],
        MYPYPATH=str(PACKAGE),
    )
    revealed: dict[str, str] = {}
    diagnostics: dict[str, list[str]] = {}
    for line in out.splitlines():
        if not (found := _MYPY_LINE.match(line)):
            continue
        stem = Path(found["file"]).stem
        message = found["message"]
        if message.startswith("Revealed type is "):
            revealed[stem] = _unqualified(message.split('"')[1])
        elif code := _MYPY_CODE.search(message):
            diagnostics.setdefault(stem, []).append(f"{found['level']}[{code['code']}]")
    return _join(revealed, diagnostics)


def _pyright() -> dict[str, str]:
    out = _run(
        [
            sys.executable,
            "-m",
            "pyright",
            # The environment pyright reads is the one running this suite, not
            # whichever `python` is first on the path: run as `<venv>/bin/python
            # -m pytest` with no venv activated, it otherwise finds a bare
            # interpreter and reports every import as a stub without source.
            "--pythonpath",
            sys.executable,
            "--pythonversion",
            FLOOR,
            "--outputjson",
            *map(str, _fixtures()),
        ],
        PYTHONPATH=str(PACKAGE),
    )
    revealed: dict[str, str] = {}
    diagnostics: dict[str, list[str]] = {}
    for item in json.loads(out)["generalDiagnostics"]:
        stem = Path(item["file"]).stem
        message = item["message"]
        if item["severity"] == "information" and message.startswith("Type of "):
            revealed[stem] = _unqualified(message.rsplit('"', 2)[1])
        else:
            diagnostics.setdefault(stem, []).append(
                f"{item['severity']}[{item.get('rule')}]"
            )
    return _join(revealed, diagnostics)


@pytest.fixture(scope="module")
def readings() -> dict[str, dict[str, str]]:
    """Give every checker's reading of every fixture, from one run each."""
    return {"ty": _ty(), "mypy": _mypy(), "pyright": _pyright()}


def test_the_fixtures_and_the_rows_are_the_same_set() -> None:
    """A fixture no row reads, or a row no fixture backs, is held by nothing."""
    fixtures = {path.stem for path in _fixtures()}
    rows = {row.fixture for row in ROWS}
    assert fixtures, "no fixtures under tests/typing/readings"
    assert fixtures == rows, (
        f"fixtures without a row: {sorted(fixtures - rows)}; "
        f"rows without a fixture: {sorted(rows - fixtures)}"
    )


@pytest.mark.parametrize("path", _fixtures(), ids=lambda path: path.stem)
def test_a_fixture_reveals_one_expression(path: Path) -> None:
    """One `reveal_type` per file, so a cell is one reading and not a merge."""
    assert path.read_text(encoding="utf-8").count("reveal_type(") == 1, path.name


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.fixture)
def test_a_row_is_a_disagreement(row: Reading) -> None:
    """A reading the checkers share belongs in the consumer as an `assert_type`."""
    assert len({row.ty, row.mypy, row.pyright}) > 1, (
        f"{row.fixture}: every checker reads {row.ty!r}; write it as an "
        "`assert_type` row in tests/typing/consumer.py and drop the fixture"
    )
    assert len(row.moves) > 80, (
        f"{row.fixture}: {row.moves!r} does not say what a change means"
    )


@pytest.mark.parametrize("checker", CHECKERS)
def test_a_checker_read_every_fixture(
    readings: dict[str, dict[str, str]], checker: str
) -> None:
    """The parse is a detector, so it must be shown to have read every file."""
    assert set(readings[checker]) == {path.stem for path in _fixtures()}, readings[
        checker
    ]


@pytest.mark.parametrize("checker", CHECKERS)
@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.fixture)
def test_a_reading_is_the_one_recorded(
    readings: dict[str, dict[str, str]], row: Reading, checker: str
) -> None:
    expected = getattr(row, checker)
    actual = readings[checker][row.fixture]
    assert actual == expected, (
        f"{row.fixture} under {checker} reads {actual!r}; the ledger records "
        f"{expected!r}. {row.moves}"
    )
