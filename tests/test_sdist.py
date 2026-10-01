"""The source distribution carries every build input and nothing else.

The sdist is the install for every platform without a wheel, and the release
compiles one on Linux before it can publish. A source file the build needs and
the archive leaves out fails there, a release late; a file the archive carries
and the tree does not track -- a built extension beside the package, a scratch
file in a crate -- ships to every user who builds from source. Both are decided
by what `maturin sdist` collects, which no other check reads.

Held both ways against the tree: the archive carries every tracked file of the
workspace's crates and of the Python package, the manifests and the lock, the
readme and licences `pyproject.toml` names and the files `[tool.maturin]
include` adds; and it carries nothing besides them but the `PKG-INFO` it
writes.

LEDGER: the source distribution carries every build input and nothing else
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from _toml import load

# The repository checks are not the product suite: this file builds the
# repository's source distribution, which no installed wheel carries.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent

#: What the build writes into the archive rather than taking from the tree.
WRITTEN = {"PKG-INFO"}


def _tracked(*roots: str) -> set[str]:
    listing = subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", *roots],  # noqa: S607
        capture_output=True,
        check=True,
        text=True,
    )
    return {name for name in listing.stdout.split("\0") if name}


@pytest.fixture(scope="module")
def archived(tmp_path_factory: pytest.TempPathFactory) -> set[str]:
    """Build the sdist from the tree and give its files, the top directory cut."""
    if importlib.util.find_spec("maturin") is None or shutil.which("cargo") is None:
        pytest.skip("building an sdist needs maturin and cargo")
    out = tmp_path_factory.mktemp("sdist")
    subprocess.run(  # noqa: S603 - fixed argv, no shell, test-only
        [sys.executable, "-m", "maturin", "sdist", "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    (archive,) = out.glob("*.tar.gz")
    with tarfile.open(archive) as sdist:
        names = [member.name for member in sdist.getmembers() if member.isfile()]
    return {name.split("/", 1)[1] for name in names}


def _inputs() -> set[str]:
    """Name the files the build reads, from the manifests that say so."""
    project = load(ROOT / "pyproject.toml")
    maturin = project["tool"]["maturin"]
    members = load(ROOT / "Cargo.toml")["workspace"]["members"]
    return (
        _tracked(*members, maturin["python-source"])
        | {"Cargo.toml", "Cargo.lock", "pyproject.toml", project["project"]["readme"]}
        | {entry["path"] for entry in maturin.get("include", [])}
        | set(project["project"]["license-files"])
    )


def test_the_sdist_carries_every_build_input(archived: set[str]) -> None:
    missing = sorted(_inputs() - archived)
    assert not missing, (
        f"build inputs the sdist leaves out: {missing}. A build from source "
        "fails without them."
    )


def test_the_sdist_carries_nothing_else(archived: set[str]) -> None:
    extra = sorted(archived - _inputs() - WRITTEN)
    assert not extra, (
        f"files the sdist carries that the tree does not track as build inputs: {extra}"
    )
