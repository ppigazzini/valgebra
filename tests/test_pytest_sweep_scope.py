"""The files excluded from the sweep "because pytest covers them" are swept too.

Seven files of the binding are reached only through the shipped extension.
`cargo mutants` runs `cargo test`, which never loads it, so every mutant of
those files survives while saying nothing about the tests -- and they were
excluded by name with the reason "pytest covers this". That reason was a
sentence, and nothing measured it: a file could lose its Python coverage
entirely and the exclusion would read the same.

A second configuration measures it. `.cargo/mutants-pytest.toml` examines
exactly those files with a test command that builds the extension from the
mutated copy and runs the Python suite against it, so a mutant one of them
carries is caught by that suite or by nothing.

The two configurations partition the binding, and this holds the partition to
the tree in both directions: a file the ordinary sweep excuses for pytest's sake
and the pytest sweep does not examine is a file nothing measures, and a file
both cover is an hour of runner time re-proving what a second already proved.

LEDGER: every file the sweep excuses to pytest is examined under pytest
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Reads the tree, the two configurations and the workflow; none ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
ORDINARY = ROOT / ".cargo" / "mutants.toml"
UNDER_PYTEST = ROOT / ".cargo" / "mutants-pytest.toml"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
WRAPPER = ROOT / "crates" / "valgebra-py" / "tests" / "pytest_sweep.rs"
MANIFEST = ROOT / "crates" / "valgebra-py" / "Cargo.toml"

#: The feature the wrapper hides behind, and the variable it measures through.
FEATURE = "pytest-sweep"
VENV = "VALGEBRA_SWEEP_VENV"


def _array(config: Path, name: str) -> list[str]:
    """Read a TOML array of strings from a configuration, comments dropped.

    A regex rather than `tomllib`, which is 3.11+ while this suite runs from
    3.10, and by the same rule `tests/test_mutation_scope.py` reads the
    exclusion list with: an entry named only in a comment is not an entry.
    """
    text = config.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\s*=\s*\[(.*?)^\]", text, re.DOTALL | re.MULTILINE
    )
    assert match is not None, f"{name} is absent from {config.name}"
    body = "\n".join(
        line
        for line in match.group(1).splitlines()
        if not line.lstrip().startswith("#")
    )
    entries = re.findall(r'"([^"]+)"', body)
    assert entries, f"{name} in {config.name} parsed empty"
    return entries


def _excused_to_pytest() -> set[str]:
    """Give the files the ordinary sweep excludes and the pytest sweep does not.

    Derived rather than listed. The pytest configuration excludes what no test
    of any kind can observe -- the examples, the build script, the instruction
    gate's own instrument -- and what remains of the ordinary sweep's exclusion
    list is precisely the set excused on the grounds that the Python suite
    covers it. A hand-written list here would be a third copy to keep true, and
    the one that went stale would be the one claiming the coverage.
    """
    return set(_array(ORDINARY, "exclude_globs")) - set(
        _array(UNDER_PYTEST, "exclude_globs")
    )


def _examined() -> set[str]:
    return set(_array(UNDER_PYTEST, "examine_globs"))


def _swept_ordinarily() -> set[str]:
    """Give the binding files the walk sweep names, read from its own lane."""
    text = WORKFLOW.read_text(encoding="utf-8")
    swept = set(re.findall(r"--file (crates/valgebra-py/\S+\.rs)", text))
    assert swept, "the workflow names no binding files for the sweep"
    return swept


def _lane(naming: str) -> str:
    """Give the workflow text of the one job naming this, split from the rest.

    Split on the job headers rather than read from parsed YAML, because what is
    asserted below is the text of a shell block -- the flags a sweep is run
    with -- and a parse would hand back the same string through more machinery.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    jobs = re.split(r"^  (?=[a-z][\w-]*:$)", text, flags=re.MULTILINE)
    holding = [job for job in jobs if naming in job]
    assert len(holding) == 1, f"{len(holding)} job(s) name {naming!r}; one does"
    return holding[0]


def test_every_file_excused_to_pytest_is_examined_there() -> None:
    """The sentence and the measurement name the same files."""
    excused, examined = _excused_to_pytest(), _examined()
    unmeasured = sorted(excused - examined)
    assert not unmeasured, (
        f"excluded from the sweep because pytest covers them, and not examined "
        f"under pytest: {unmeasured}. Either add each to examine_globs in "
        f"{UNDER_PYTEST.name}, or exclude it there with a reason of its own."
    )


def test_nothing_is_examined_under_pytest_that_a_second_already_proves() -> None:
    """A file swept both ways costs a suite run per mutant to re-prove a kill."""
    doubled = sorted(_examined() & _swept_ordinarily())
    assert not doubled, (
        f"swept by the walk lane and examined under pytest: {doubled}. The "
        "pytest sweep runs the whole Python suite per mutant; a file with a "
        "Rust corpus is judged in a second by the other lane."
    )
    overreaching = sorted(_examined() - _excused_to_pytest())
    assert not overreaching, (
        f"examined under pytest and not excused to it: {overreaching}"
    )


def test_every_examined_file_is_in_the_tree() -> None:
    """An examine glob naming nothing measures nothing while reading as scope."""
    absent = sorted(name for name in _examined() if not (ROOT / name).is_file())
    assert not absent, f"examine_globs names files that are not here: {absent}"


def test_the_exclusion_points_at_what_measures_it() -> None:
    """A reader of the excuse can reach the thing that checks it."""
    text = ORDINARY.read_text(encoding="utf-8")
    assert "mutants-pytest.toml" in text, (
        "the ordinary configuration excuses files to the Python suite without "
        "naming the configuration that sweeps them under it"
    )


def test_the_wrapper_refuses_rather_than_measuring_nothing() -> None:
    """Without the environment the suite needs, the wrapper fails the sweep.

    The failure this whole slice exists to end is a green number about a suite
    that never ran. A wrapper that skipped where the variable is unset would
    report every mutant caught, which is exactly that number.
    """
    source = WRAPPER.read_text(encoding="utf-8")
    assert f'#![cfg(feature = "{FEATURE}")]' in source, (
        "the wrapper must be behind its feature; a rebuild and a suite run per "
        "`cargo test` is a cost no ordinary lane should pay"
    )
    assert VENV in source, "the wrapper must read the environment it runs in"
    guard = re.search(rf"var\(\"{VENV}\"\)(.*?)\}};", source, re.DOTALL)
    assert guard is not None, f"the wrapper does not branch on {VENV}"
    assert "panic!" in guard.group(1), (
        f"a wrapper that does not fail on an unset {VENV} reports every mutant "
        "caught while running no suite"
    )


def test_the_workers_of_a_sweep_do_not_share_one_environment() -> None:
    """Two workers building into one environment take the whole run down.

    `cargo mutants -j N` runs N workers in N copies of the tree. They share
    every path outside those copies, so an environment named by the variable and
    used as named is one both workers rebuild the extension into -- at the same
    moment, which fails with `File exists` and ends the sweep with no verdict a
    third of the way through. It did, twice.

    So the wrapper derives an environment from the checkout it is running in,
    whose directory name a sweep makes unique per worker. Checked by shape
    rather than by running two workers: the sweep itself is the experiment, and
    what this refuses is the variable being handed to the build unchanged.
    """
    source = WRAPPER.read_text(encoding="utf-8")
    assert "fn worker_venv(" in source, (
        "the wrapper names no per-worker environment; a shared one races"
    )
    assert "file_name()" in source, (
        "the per-worker environment is not keyed by the checkout it runs in"
    )
    build = re.search(r"let venv = (\w+)\(", source)
    assert build is not None, "the wrapper binds no environment for the build"
    assert build.group(1) == "worker_venv", (
        f"the build uses {build.group(1)}, not the derived environment"
    )


def test_the_feature_is_declared_and_off_by_default() -> None:
    manifest = MANIFEST.read_text(encoding="utf-8")
    assert re.search(rf"^{re.escape(FEATURE)}\s*=\s*\[", manifest, re.MULTILINE), (
        f"{FEATURE} is not declared in {MANIFEST.name}"
    )
    default = re.search(r"^default\s*=\s*\[([^\]]*)\]", manifest, re.MULTILINE)
    if default is not None:
        assert FEATURE not in default.group(1), (
            f"{FEATURE} is on by default, so every `cargo test` runs the suite twice"
        )


def test_a_lane_runs_the_sweep_off_the_merge_path() -> None:
    """The measurement runs somewhere: a configuration no lane invokes is a file.

    Off the merge path by its own condition, because the cost is a rebuild and a
    suite run per mutant -- a minute each against the seconds an ordinary mutant
    takes.
    """
    lane = _lane("mutants-pytest.toml")
    assert f"--features {FEATURE}" in lane, "the lane does not enable the wrapper"
    assert VENV in lane, f"the lane does not give the sweep a {VENV}"
    assert re.search(r"^\s*if:.*(schedule|workflow_dispatch)", lane, re.MULTILINE), (
        "the sweep costs a suite run per mutant and must not sit on the merge path"
    )


def test_a_lane_ratchets_what_the_sweep_finds() -> None:
    """Survivors reported nowhere are survivors nobody sees."""
    lane = _lane("--baseline pytest")
    assert "scripts/mutation_gate.py --baseline pytest" in lane, (
        "the ratchet does not run the gate against the pytest baseline"
    )
    assert re.search(r"^\s*if:.*(schedule|workflow_dispatch)", lane, re.MULTILINE), (
        "the ratchet runs where its sweep does"
    )


def test_the_pytest_baseline_is_registered_and_recorded() -> None:
    """The gate knows the sweep by name, and the accepted set is committed."""
    gate = (ROOT / "scripts" / "mutation_gate.py").read_text(encoding="utf-8")
    baselines = re.search(r"^BASELINES = \{(.*?)^\}", gate, re.DOTALL | re.MULTILINE)
    assert baselines is not None, "mutation_gate.py declares no BASELINES"
    assert '"pytest"' in baselines.group(1), (
        "the gate has no pytest baseline, so `--baseline pytest` cannot run"
    )
    recorded = ROOT / "scripts" / "mutation_baseline_pytest.json"
    assert recorded.is_file(), f"{recorded.name} is not committed"
