"""The PyPy link check must fail on a form that answers wrongly.

`scripts/pypy_import_check.py` imports the extension and builds the forms whose
compilation reaches a type object `cpyext` may not carry; the PyPy lane reads
its exit code. It passes on this interpreter, and it fails when the module it
checks answers one form wrongly -- driven here through a stand-in for the
package rather than a broken build.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

# The repository checks are not the product suite: this file drives a lane's
# script, which does not ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pypy_import_check.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pypy_import_check", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_forms_build_here() -> None:
    assert _load().main([]) == 0


def test_a_form_answering_wrongly_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    check = _load()
    package = check.vg

    class Refusing:
        """A validator that refuses what `int` admits, and otherwise answers."""

        def __init__(self, schema: object) -> None:
            self._inner = package.Validator(schema)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def __repr__(self) -> str:
            return repr(self._inner)

        def is_valid(self, value: object) -> bool:
            return False if value == 1 else self._inner.is_valid(value)

    stand_in = SimpleNamespace(**{**vars(package), "Validator": Refusing})
    monkeypatch.setattr(check, "vg", stand_in)
    assert check.main([]) == 1
