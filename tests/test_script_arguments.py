"""Every script answers `--help` and refuses a flag it does not know.

A script that reads its flags by substring treats a flag it does not know as no
flag at all: `perf_gate.py --bindng` measured the core workload and answered
for it, and `perf_gate.py --help` built both workloads before saying nothing
about how to call it. So every script reads its command line through
`argparse` before doing any work, which prints the usage on `--help` and exits
2 -- the gates' "could not run" -- on anything else it does not know.

Held over the directory rather than a list of scripts, so a script added
tomorrow is held the day it lands.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

# The repository checks are not the product suite: this file runs the scripts,
# none of which ship in a wheel.
pytestmark = pytest.mark.repository

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _absent_import(script: Path) -> str | None:
    """Name a module `script` imports at its top that this environment lacks.

    A script of another dependency group -- the docs build's, which imports
    griffe for mkdocs to load its extension -- cannot start where that group is
    not installed, and says nothing about its command line there.
    """
    for node in ast.parse(script.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            # A sibling script is on the path a script runs with.
            sibling = (SCRIPTS / f"{top}.py").is_file()
            if not sibling and importlib.util.find_spec(top) is None:
                return name
    return None


def _run(script: Path, flag: str) -> subprocess.CompletedProcess[str]:
    if (absent := _absent_import(script)) is not None:
        pytest.skip(f"{script.name} imports {absent}, which this environment lacks")
    return subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [sys.executable, str(script), flag],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_the_directory_holds_the_scripts() -> None:
    assert len(list(SCRIPTS.glob("*.py"))) >= 10, "the scripts directory reads empty"


@pytest.mark.parametrize(
    "script", sorted(SCRIPTS.glob("*.py")), ids=lambda path: path.name
)
def test_help_prints_the_usage_and_does_nothing_else(script: Path) -> None:
    done = _run(script, "--help")
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.startswith("usage:"), done.stdout[:400]


@pytest.mark.parametrize(
    "script", sorted(SCRIPTS.glob("*.py")), ids=lambda path: path.name
)
def test_an_unknown_flag_cannot_run(script: Path) -> None:
    done = _run(script, "--no-such-flag")
    # argparse's refusal: the usage, then the reason -- the unknown flag, or a
    # required argument the call left out, whichever it reads first.
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stderr.startswith("usage:"), done.stderr[:400]
