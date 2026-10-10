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
import subprocess
import sys
import types
from typing import TYPE_CHECKING

import pytest

from _tree import SCRIPTS, load_script

# The repository checks are not the product suite: this file drives a docs
# script, which does not ship in a wheel.
pytestmark = pytest.mark.repository

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


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
    return load_script("docs_stubs")


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


#: Load a file by path in a fresh interpreter and name a class it defines.
_LOAD_BY_PATH = (
    "import importlib.util, sys; "
    "spec = importlib.util.spec_from_file_location('loaded', sys.argv[1]); "
    "module = importlib.util.module_from_spec(spec); "
    "spec.loader.exec_module(module); "
    "print(getattr(module, sys.argv[2]).__name__)"
)


def test_the_extension_loads_where_mkdocs_loads_it(tmp_path: Path) -> None:
    """Griffe loads the extension by path, in a process no script run started.

    There `scripts/` is not on the import path, and the API reference stopped
    building on the import of `_lib` the script shares with the others. So the
    file is loaded the same way, in a child interpreter whose path holds a
    stand-in griffe and not `scripts/`.
    """
    (tmp_path / "griffe.py").write_text("class Extension:\n    pass\n")
    done = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [
            sys.executable,
            "-c",
            _LOAD_BY_PATH,
            str(SCRIPTS / "docs_stubs.py"),
            "InspectTheCompiledClasses",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "InspectTheCompiledClasses"
