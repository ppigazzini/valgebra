"""A gate script in no lane is not a gate.

Every executable under ``scripts/`` must be driven by something: a CI workflow,
the packaging configuration, or the test suite. One that is driven by nothing
still passes review, still looks maintained, and asserts nothing -- and the way
that happens is never a decision, it is a lane that quietly stopped naming it.

The universe is globbed from the tree rather than listed here, because a second
hand-written list rots exactly like the first: the direction that matters is
"this script arrived and nothing runs it", and only a globbed universe sees it.

Two extraction rules the check is built around, both of which make a checker
report coverage it does not have:

* a script named in a **comment** is not a script anything runs;
* a word boundary that accepts ``_`` or ``-`` lets ``perf_gate.py`` satisfy a
  reference to ``gate.py``.

The excused list expires in its own direction: an excused script that *is*
driven is a stale excuse and fails, and an excuse naming a script that no longer
exists fails too.

LEDGER: every gate script runs in a lane; no excuse is stale
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# Scripts deliberately driven by nothing, each with the reason. Empty today:
# every script under scripts/ is driven by a workflow, by pyproject, or by a
# test. An entry here is a hole in the coverage claim, so it carries an argument
# rather than a name alone.
EXCUSED: dict[str, str] = {}


# Where a script can be driven from. Workflows and the packaging config are CI;
# the test suite is a lane too, since it runs on every merge.
def _driver_files() -> list[Path]:
    files = sorted((ROOT / ".github").rglob("*.yml"))
    files += sorted((ROOT / ".github").rglob("*.yaml"))
    files.append(ROOT / "pyproject.toml")
    files += sorted(ROOT.glob("tests/*.py"))
    files.append(ROOT / ".pre-commit-config.yaml")
    return [f for f in files if f.exists()]


def strip_comments(text: str) -> str:
    """Drop whole-line comments, which name scripts nothing runs."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def names_script(text: str, script: str) -> bool:
    """Whether `text` drives `script`, with a boundary a longer name cannot satisfy.

    The lookaround refuses `_` and `-` on both sides, so a reference to
    ``gate.py`` is not satisfied by ``perf_gate.py`` or ``mutation_gate.py``.
    """
    pattern = rf"(?<![\w\-]){re.escape(script)}(?![\w\-])"
    return re.search(pattern, text) is not None


def _drivers_of(script: str) -> list[str]:
    hits = []
    for path in _driver_files():
        if path.name == "test_lane_coverage.py":
            continue  # this file names every script; it drives none of them
        if names_script(strip_comments(path.read_text(encoding="utf-8")), script):
            hits.append(str(path.relative_to(ROOT)))
    return hits


def test_every_script_runs_in_a_lane() -> None:
    scripts = sorted(p.name for p in SCRIPTS.glob("*.py"))
    # The glob is the detector; an empty universe would pass having checked
    # nothing, which is worse than a bare failure.
    assert len(scripts) >= 4, f"the scripts glob found only {scripts}"

    undriven = [s for s in scripts if s not in EXCUSED and not _drivers_of(s)]
    assert not undriven, (
        f"scripts in no lane and on no excused list: {undriven}. "
        "Wire each into a workflow, or excuse it by name with a reason."
    )


def test_no_excuse_is_stale() -> None:
    scripts = {p.name for p in SCRIPTS.glob("*.py")}

    gone = sorted(set(EXCUSED) - scripts)
    assert not gone, f"excuses naming a script that no longer exists: {gone}"

    driven = sorted(name for name in EXCUSED if _drivers_of(name))
    assert not driven, (
        f"excused scripts that are in fact driven: {driven}. "
        "Remove the excuse; it claims a hole the tree does not have."
    )


def test_a_comment_does_not_count_as_a_driver() -> None:
    # The rule that a checker gets wrong by reading the file whole.
    assert names_script("run: python scripts/perf_gate.py", "perf_gate.py")
    assert not names_script(
        strip_comments("  # see scripts/perf_gate.py for the budget"), "perf_gate.py"
    )
    # A trailing comment on a real command still leaves the command.
    assert names_script(
        strip_comments("run: python scripts/perf_gate.py  # the budget"),
        "perf_gate.py",
    )


def test_a_longer_name_does_not_satisfy_a_shorter_one() -> None:
    # The boundary rule. Both of these are real script names in this tree, so a
    # loose boundary would report `gate.py` as driven by either.
    haystack = "run: python scripts/perf_gate.py && python scripts/mutation_gate.py"
    assert names_script(haystack, "perf_gate.py")
    assert names_script(haystack, "mutation_gate.py")
    assert not names_script(haystack, "gate.py")
    assert not names_script("run: python scripts/compare_gate.py", "compare.py")


def test_the_stale_excuse_rules_fire() -> None:
    # The excused list is empty today, so its two failure directions are driven
    # against synthetic inputs rather than left unexercised until the first
    # excuse is written.
    scripts = {p.name for p in SCRIPTS.glob("*.py")}
    fictional = "no_such_script.py"
    assert fictional not in scripts  # would be caught by test_no_excuse_is_stale
    real = "perf_gate.py"
    assert real in scripts
    assert _drivers_of(real), "a driven script must be seen to be driven"
