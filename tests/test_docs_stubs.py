"""The API page check must fail on a page that lost a docstring.

`scripts/docs_stubs.py --check` reads the built API page for the first line of
each documented object's docstring, because a docs build that exits 0 is not
evidence the page has content. Its three verdicts are driven here on synthetic
pages rather than a build: a page carrying every line passes, a page missing
one fails, and a page not built cannot run.
"""

from __future__ import annotations

import html
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# The repository checks are not the product suite: this file drives a docs
# script, which does not ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "docs_stubs.py"


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the script by path, with a stand-in where griffe is absent.

    griffe is the docs group's, which the lanes running this suite do not
    install, and the script needs it only as the base of the extension class
    mkdocs loads; the check below reads the page and the package alone. The
    stand-in lasts the test, so no later `find_spec` meets it.
    """
    if importlib.util.find_spec("griffe") is None:
        stand_in = types.ModuleType("griffe")
        vars(stand_in)["Extension"] = object
        monkeypatch.setitem(sys.modules, "griffe", stand_in)
    spec = importlib.util.spec_from_file_location("docs_stubs", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _page(stubs: ModuleType, *, leave_out: str | None = None) -> str:
    """Write a page carrying each generated object's first docstring line."""
    package = importlib.import_module(stubs.PACKAGE)
    lines = [
        html.escape((getattr(package, name).__doc__ or "").strip().splitlines()[0])
        for name in stubs.GENERATED
        if name != leave_out
    ]
    return (
        "<html><body>" + "".join(f"<p>{line}</p>" for line in lines) + "</body></html>"
    )


def test_a_page_carrying_every_docstring_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = _load(monkeypatch)
    page = tmp_path / "index.html"
    page.write_text(_page(stubs), encoding="utf-8")
    monkeypatch.setattr(stubs, "PAGE", page)
    assert stubs.main(["--check"]) == 0


def test_a_page_missing_a_docstring_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = _load(monkeypatch)
    page = tmp_path / "index.html"
    page.write_text(_page(stubs, leave_out="union"), encoding="utf-8")
    monkeypatch.setattr(stubs, "PAGE", page)
    assert stubs.main(["--check"]) == 1


def test_a_page_not_built_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = _load(monkeypatch)
    monkeypatch.setattr(stubs, "PAGE", tmp_path / "absent.html")
    monkeypatch.setattr(stubs, "ROOT", tmp_path)
    assert stubs.main(["--check"]) == 2
