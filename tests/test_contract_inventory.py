"""The contract inventory must answer for every contract, and only real ones.

`CONTRIBUTING.md` carries a table of what this project promises, the file that
owns each promise, and the single command that reproduces that one verdict. Its
value is that a reader who wants to re-check one thing does not have to
reconstruct the command from a workflow -- and that value survives exactly as
long as the table does.

Held in both directions:

* a gate script with no row fails, so a promise cannot arrive unlisted;
* a row naming a script that does not exist fails, so a row cannot outlive what
  it describes.

The rows that are not script invocations -- `cargo test`, `cargo fmt --check` --
are checked only for naming a real source of truth; whether the command still
does what the row claims is the reader's half, as it is for every claim
`scripts/docs_lint.py` cannot reach.

LEDGER: every gate script has a contract row; every row names a real source
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
SCRIPTS = ROOT / "scripts"
HEADING = "## Contract inventory"


def _table() -> list[tuple[str, str, str]]:
    """Read the inventory's rows: (contract, source of truth, rerun command)."""
    text = CONTRIBUTING.read_text(encoding="utf-8")
    assert HEADING in text, "CONTRIBUTING.md carries no contract inventory"
    section = text.split(HEADING, 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in {"Contract", ""} or set(cells[1]) <= {"-"}:
            continue
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def _commands() -> str:
    return "\n".join(command for _, _, command in _table())


def _whole_table() -> str:
    """Every cell. A script can be a row's *subject* rather than its command.

    The profile-guided training workload is driven through the packaging config
    rather than typed, so it is named in the source-of-truth cell and the rerun
    cell is the build that invokes it. It has a row; it does not have a command
    of its own.
    """
    return "\n".join(" ".join(row) for row in _table())


def test_the_inventory_has_rows() -> None:
    # The parse is the detector: a table it could not read would pass every
    # check below having compared nothing.
    rows = _table()
    assert len(rows) >= 15, f"the inventory parse found only {len(rows)} row(s)"


def test_every_gate_script_has_a_row() -> None:
    # A promise nothing in the table reproduces is a promise a reader cannot
    # check on its own.
    table = _whole_table()
    unlisted = sorted(
        script.name for script in SCRIPTS.glob("*.py") if script.name not in table
    )
    assert not unlisted, (
        f"gate scripts with no row in the contract inventory: {unlisted}. "
        "Add a row naming what it promises and the command that reruns it."
    )


def test_every_row_names_a_real_source_of_truth() -> None:
    missing = []
    for contract, source, _ in _table():
        for named in re.findall(r"`([^`]+)`", source):
            candidate = named.split()[0]
            if "/" not in candidate and not candidate.endswith(".toml"):
                continue  # a config key, not a path
            if not (ROOT / candidate.rstrip("/")).exists():
                missing.append(f"{contract}: {candidate}")
    assert not missing, f"rows naming a source of truth that does not exist: {missing}"


def test_every_script_a_row_invokes_exists() -> None:
    named = set(re.findall(r"scripts/([\w.]+\.py)", _commands()))
    assert named, "no row invokes a script; the table cannot be checked this way"
    gone = sorted(name for name in named if not (SCRIPTS / name).exists())
    assert not gone, f"rows invoking scripts that do not exist: {gone}"


def test_every_row_carries_all_three_cells() -> None:
    for contract, source, command in _table():
        assert contract, "a row with no contract"
        assert source, f"{contract}: a row with no source of truth"
        assert command, f"{contract}: a row with no rerun command"
        assert command.startswith("`"), (
            f"{contract}: the rerun cell must be a command in backticks"
        )
        assert command.endswith("`"), (
            f"{contract}: the rerun cell must be a command in backticks"
        )
