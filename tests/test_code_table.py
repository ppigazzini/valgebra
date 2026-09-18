"""A failure code is a name in a table, not a string at a call site.

A code is the part of a failure a caller writes code against: `exc.code ==
"missing_key"` is a branch in somebody's error handler, so the set of them is a
published vocabulary. It was written as `&'static str` literals at the sites
that report them -- twenty-one of the forty-two, scattered across the walk, the
call boundary and the report -- and `scripts/use_case_ledger.py` recovered the
set by scanning four hand-picked paths for snake-case strings.

That derivation is a guess from a path, and it is wrong in both directions. A
code written in a file outside the four is a code the ledger cannot see, so a
cell nothing names passes as a cell nobody has. And a snake-case string in one
of the four that is *not* a code is a cell the ledger invents and then asks the
suite to cover.

So the binding names its codes once, in `crates/valgebra-py/src/codes.rs`, and
this holds the arrangement:

* every constant the table declares is written somewhere, so a name nothing
  reports is a name the vocabulary does not have;
* no violation is built with a code spelled as a literal, so the table is the
  only way to write one;
* the universe the use-case ledger derives is the table's values together with
  the core's own `error_code` arms -- the two places a code can come from, held
  to the two tables rather than to a scan.

The core's table stays where it is: a code for a node is the node's own, and
`Code::of_schema` is the one crossing.

LEDGER: every failure code is a name the table declares, and every name is used
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# A repository check: it reads the crate's sources and the derivation script,
# neither of which ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
BINDING = ROOT / "crates" / "valgebra-py" / "src"
TABLE = BINDING / "codes.rs"

#: A code the table declares: a name, and the string a caller compares against.
_DECLARED = re.compile(
    r"pub\(crate\) const (?P<name>[A-Z][A-Z0-9_]*): Code"
    r" = Code\(\"(?P<code>[a-z_]+)\"\);"
)

#: A violation built with its code spelled out, which the table exists to stop.
_LITERAL = re.compile(r"code: \"[a-z_]+\"")


def _sources() -> list[Path]:
    """Every Rust file of the binding that ships, corpora and tests aside.

    The corpora name codes as the *expected* half of a row, which is a reading
    of the vocabulary rather than a place one is written.
    """
    return [
        path
        for path in sorted(BINDING.rglob("*.rs"))
        if not path.name.endswith(("interpreter.rs", "tests.rs"))
    ]


def _table() -> dict[str, str]:
    """Give the table's constants, as name to code."""
    if not TABLE.exists():
        return {}
    return {
        found["name"]: found["code"]
        for found in _DECLARED.finditer(TABLE.read_text(encoding="utf-8"))
    }


def test_the_binding_names_its_codes_in_one_table() -> None:
    """The table is the detector: without it every rule below reads nothing."""
    table = _table()
    assert TABLE.exists(), f"{TABLE.relative_to(ROOT)} does not exist"
    assert len(table) >= 15, f"the table declares only {sorted(table)}"


def test_every_name_the_table_declares_is_written_somewhere() -> None:
    table = _table()
    used = "\n".join(
        path.read_text(encoding="utf-8") for path in _sources() if path != TABLE
    )
    # A whole word: the sites import the names, so a constant appears bare
    # rather than behind the module, and `TOO_LONG` must not be satisfied by a
    # longer name that ends in it.
    unwritten = sorted(name for name in table if not re.search(rf"\b{name}\b", used))
    assert not unwritten, (
        f"codes the table declares that nothing reports: {unwritten}. A name no "
        "site writes is a name the vocabulary does not have, so delete it or "
        "report it."
    )


def test_no_violation_is_built_from_a_code_spelled_out() -> None:
    spelled = [
        f"{path.relative_to(ROOT).as_posix()}:{index + 1}"
        for path in _sources()
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
        if _LITERAL.search(line)
    ]
    assert not spelled, (
        f"violations built with the code spelled out: {spelled}. Name it in "
        "`codes.rs` and use the constant; a literal here is a code no table "
        "carries and no ledger can find."
    )


def test_the_derived_universe_is_the_two_tables() -> None:
    """The ledger's universe comes from the tables, not from a scan of paths."""
    script = (ROOT / "scripts" / "use_case_ledger.py").read_text(encoding="utf-8")
    assert "codes.rs" in script, (
        "the use-case ledger does not read the binding's code table, so its "
        "universe is still a guess from the paths it scans"
    )
    assert "_EMITTERS" not in script, (
        "the use-case ledger still scans hand-picked paths for snake-case "
        "strings, which is the derivation the table replaces"
    )
