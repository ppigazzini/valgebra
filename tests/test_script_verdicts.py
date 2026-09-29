"""Four scripts whose ledgers hold the tree must also be seen to fail.

`run_doc_examples.py`, `check_doc_examples.py`, `floor_names.py` and
`use_case_ledger.py` each back a lane, and a ledger in `tests/` holds the tree
to the same claim; neither says the script's own exit code can go red. Each is
driven here with the expensive half stood in -- the examples, the four
checkers, the six interpreters, the suite's sources -- so the verdict is the
script's own and the input is one that must fail it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# The repository checks are not the product suite: this file drives the gate
# scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str) -> ModuleType:
    """Import a script by path; ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        f"verdict_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("blocks", "verdict"),
    [(["assert True\n"], 0), (["assert True\n", "assert False\n"], 1)],
)
def test_a_failing_example_fails_the_runner(
    monkeypatch: pytest.MonkeyPatch, blocks: list[str], verdict: int
) -> None:
    runner = _load("run_doc_examples")

    def examples() -> Iterator[tuple[Path, int, str]]:
        for index, block in enumerate(blocks):
            yield Path("page.md"), index, block

    monkeypatch.setattr(runner, "examples", examples)
    assert runner.main([]) == verdict


def test_a_diagnostic_no_row_expects_fails_the_checkers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkers = _load("check_doc_examples")
    names = ("ty", "mypy", "pyright", "ruff")
    expected: Counter[tuple[str, str, str]] = Counter()
    for row in checkers.EXPECTED:
        expected[(row.checker, row.page, row.rule)] += row.count

    def reading(found: Counter[tuple[str, str, str]]) -> object:
        return checkers.Reading(count=1, found=found, checkers=names, unparsed=())

    monkeypatch.setattr(checkers, "read", lambda: reading(expected))
    assert checkers.main([]) == 0
    extra = expected + Counter({("ty", "docs/planted.md", "planted-rule"): 1})
    monkeypatch.setattr(checkers, "read", lambda: reading(extra))
    assert checkers.main([]) == 1
    # A row nothing reports is the other half: an expectation outliving its cause.
    short = expected - Counter({next(iter(expected)): 1})
    monkeypatch.setattr(checkers, "read", lambda: reading(short))
    assert checkers.main([]) == 1


def test_a_table_the_interpreters_disagree_with_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    floor = _load("floor_names")
    table = json.loads(floor.TABLE.read_text(encoding="utf-8"))
    monkeypatch.setattr(floor, "_survey", lambda *_: {})
    monkeypatch.setattr(floor, "_imported", lambda *_: set())
    monkeypatch.setattr(floor, "_stdlib", lambda *_: table["stdlib"])
    monkeypatch.setattr(floor, "_spans", lambda *_: table["modules"])
    assert floor.main(["--check"]) == 0
    moved = {module: dict(rows) for module, rows in table["modules"].items()}
    name = next(iter(moved["typing"]))
    moved["typing"][name] = {"since": "3.99"}
    monkeypatch.setattr(floor, "_spans", lambda *_: moved)
    assert floor.main(["--check"]) == 1


def test_an_interpreter_that_does_not_answer_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    floor = _load("floor_names")
    monkeypatch.setattr(floor, "_survey", lambda *_: None)
    assert floor.main(["--check"]) == 2


@pytest.mark.parametrize(("recorded", "verdict"), [({"b": "a reason"}, 0), ({}, 1)])
def test_an_empty_cell_the_record_does_not_carry_fails_the_ledger(
    monkeypatch: pytest.MonkeyPatch, recorded: dict[str, str], verdict: int
) -> None:
    ledger = _load("use_case_ledger")
    for reader, value in (
        ("universe", {"a", "b"}),
        ("markers", set()),
        ("public_surface", set()),
        ("error_codes", set()),
    ):
        monkeypatch.setattr(ledger, reader, lambda value=value: value)
    monkeypatch.setattr(ledger, "product_sources", lambda **_: "")
    monkeypatch.setattr(ledger, "asserted_sources", lambda: "")
    monkeypatch.setattr(ledger, "names_reached", lambda *_: {"a"})
    monkeypatch.setattr(ledger, "accepted", lambda: recorded)
    monkeypatch.setattr(ledger, "report", lambda *_: None)
    assert ledger.main([]) == verdict
