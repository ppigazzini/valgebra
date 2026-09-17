"""The mutation sweep's scope must be a decision, not an accident.

`cargo mutants --list` is authoritative for what the sweep covers, and the scope
is set by an exclusion list in ``.cargo/mutants.toml``. A crate-wide glob makes
every future file inherit the exclusion silently, so the binding's pytest-only
files are excluded **by name** and this holds that list to the tree in both
directions:

* a binding source file that is neither swept nor excluded fails, so a new file
  joins the sweep by default rather than disappearing from it;
* an exclusion naming a file that no longer exists fails, so the list cannot
  accumulate entries with no subject.

The same holds of ``exclude_re``, which excuses individual mutants rather than
files: each entry must match a mutant `cargo mutants --list` really offers. A
regex matching nothing has outlived the code it argued about, and reads as
coverage the sweep does not have.

The list is a hole in the coverage claim, so each entry carries the reason it is
there -- stated once in the config's own comment, which this checks is present.

LEDGER: every binding file is swept or excluded by name
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".cargo" / "mutants.toml"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
BINDING = ROOT / "crates" / "valgebra-py"


# The binding files inside the sweep: the membership walk, where soundness is
# decided, the context it carries, and the frontend with the three surfaces
# beside it, where an annotation becomes the set it denotes. All are reachable
# from `cargo test` because the
# `interpreter-tests` feature links an embedded Python and each carries its own
# corpus -- the walk drives real values through `member`, the frontend drives
# real annotations through `build_schema` -- and the context's two predicates are
# asserted over every mode by its own tests.
def _swept() -> set[str]:
    """Read the binding files the walk sweep covers, from the lane covering them.

    This list was written out here, which made three copies of one list: the
    two `--file` lists in `ci.yml` (which `tests/test_required_jobs.py` holds
    equal to each other) and this one, which nothing held to either. A file
    brought into the sweep therefore needed the same edit in three places, and
    forgetting this one left the check below claiming the file was excluded
    while the lane swept it. Read from the workflow, the claim is about what
    runs.

    Parsed with a regex rather than a YAML library, as `_excluded_globs` reads
    the TOML: the wanted lines are the `--file` arguments inside one shell
    block, and parsing the whole workflow to find them would be the larger tool.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    swept = set(re.findall(r"--file (crates/valgebra-py/\S+\.rs)", text))
    assert swept, "the workflow names no binding files for the sweep"
    return swept


def _excluded_globs() -> list[str]:
    """Read `exclude_globs` from the config.

    Parsed with a regex rather than a TOML library: `tomllib` is 3.11+ and this
    suite runs from 3.10, and a third-party parser would be a dependency added
    for one array in one file this repository owns. The array's entries are
    quoted strings, and a comment line is dropped before they are read -- an
    entry named only in a comment is not an entry.
    """
    text = CONFIG.read_text(encoding="utf-8")
    array = r"^exclude_globs\s*=\s*\[(.*?)^\]"
    match = re.search(array, text, re.DOTALL | re.MULTILINE)
    if match is None:
        # A single-line form is also valid TOML; accept it rather than reporting
        # an empty exclusion list, which would pass this file having read nothing.
        match = re.search(r"^exclude_globs\s*=\s*\[(.*?)\]", text, re.MULTILINE)
    assert match is not None, "exclude_globs is absent from the mutants config"
    body = "\n".join(
        line
        for line in match.group(1).splitlines()
        if not line.lstrip().startswith("#")
    )
    return re.findall(r'"([^"]+)"', body)


def _excluded_regexes() -> list[str]:
    """Read `exclude_re` from the config, by the same rule as the globs."""
    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(r"^exclude_re\s*=\s*\[(.*?)^\]", text, re.DOTALL | re.MULTILINE)
    assert match is not None, "exclude_re is absent from the mutants config"
    body = "\n".join(
        line
        for line in match.group(1).splitlines()
        if not line.lstrip().startswith("#")
    )
    # TOML escapes a backslash inside a basic string, so `\\(` on the page is the
    # regex `\(`. Undo that one level to recover the pattern cargo-mutants reads.
    return [entry.replace("\\\\", "\\") for entry in re.findall(r'"([^"]+)"', body)]


def _every_mutant() -> list[str]:
    """List every mutant in the workspace, with the exclusions turned off."""
    cargo = shutil.which("cargo")
    assert cargo is not None, "cargo is on the path wherever cargo-mutants is"
    listing = subprocess.run(  # noqa: S603  # fixed argv, no shell, test-only
        [cargo, "mutants", "--list", "--no-config"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert listing.returncode == 0, listing.stderr
    return listing.stdout.splitlines()


def _binding_sources() -> set[str]:
    """Read the binding's source files: what a sweep could mutate, tests aside.

    A test module is not a subject. The long ones live in sibling files now --
    declared `#[cfg(test)] mod tests;` in the file they test -- and mutating one
    says nothing about the product: it makes the harness wrong, not the code
    untested.

    Cargo's own `tests/` directory is the same argument in the shape Cargo
    gives it: every file there is an integration target, compiled beside the
    library and never into it. The sweep wrapper that runs the Python suite
    lives there, and a mutation of it would be a mutation of the instrument.
    """
    test_only = _test_module_files() | _integration_targets()
    return {
        path
        for path in (
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in BINDING.rglob("*.rs")
        )
        if path not in test_only
    }


def _integration_targets() -> set[str]:
    """Every `.rs` file Cargo compiles as an integration test of the binding.

    Read from the directory Cargo reserves for them rather than from a naming
    convention: `tests/` is where an integration target goes, and that is a fact
    about the build rather than about what a file is called.
    """
    tests = BINDING / "tests"
    if not tests.is_dir():
        return set()
    return {
        str(path.relative_to(ROOT)).replace("\\", "/") for path in tests.rglob("*.rs")
    }


def _test_module_files() -> set[str]:
    """Every `.rs` file some parent module declares under a `#[cfg(test)]`.

    A long test module is a sibling file, declared as `#[cfg(test)] mod tests;`
    rather than written inline. The file is
    still test-only -- it is not compiled into the wheel -- and reading the
    declaration is how that is known from the tree rather than from a naming
    convention.
    """
    declared: set[str] = set()
    for source in ROOT.rglob("*.rs"):
        if "target" in source.parts:
            continue
        text = source.read_text(encoding="utf-8")
        for name in re.findall(
            r"#\[cfg\((?:test|all\(test[^\n]*)\)\]\s*\nmod (\w+);", text
        ):
            for candidate in (
                source.parent / f"{name}.rs",
                source.parent / source.stem / f"{name}.rs",
            ):
                if candidate.exists():
                    declared.add(str(candidate.relative_to(ROOT)).replace("\\", "/"))
    return declared


def _matches(glob: str, path: str) -> bool:
    pattern = "^" + re.escape(glob).replace(r"\*\*", ".*").replace(r"\*", "[^/]*") + "$"
    return re.match(pattern, path) is not None


def test_every_binding_file_is_swept_or_excluded_by_name() -> None:
    sources = _binding_sources()
    # The glob is the detector: an empty universe would pass having checked
    # nothing.
    assert len(sources) >= 8, f"the binding source glob found only {sorted(sources)}"

    globs = _excluded_globs()
    unaccounted = sorted(
        path
        for path in sources
        if path not in _swept() and not any(_matches(g, path) for g in globs)
    )
    assert not unaccounted, (
        f"binding files neither swept nor excluded: {unaccounted}. "
        "Add each to exclude_globs with its reason, or bring it into the sweep."
    )


def test_an_integration_target_is_not_a_subject() -> None:
    """The universe is the code that ships, and `tests/` is not it.

    Asserted rather than left to the glob, because the exclusion list is the
    other way to keep a harness out of the sweep -- and an exclusion carries a
    reason a reader must maintain, for a file that was never a subject.
    """
    targets = _integration_targets()
    assert targets, "the binding has no integration target; the wrapper has moved"
    sources = _binding_sources()
    assert not (targets & sources), (
        f"integration targets read as subjects: {sorted(targets & sources)}"
    )
    # The other direction: dropping `tests/` must not have taken the crate's own
    # sources with it, which would leave every claim below passing over nothing.
    assert {path for path in sources if path.startswith("crates/valgebra-py/src/")}, (
        "the universe holds none of the crate's sources"
    )


def test_no_exclusion_names_a_file_that_is_gone() -> None:
    sources = _binding_sources()
    stale = sorted(
        glob for glob in _excluded_globs() if "*" not in glob and glob not in sources
    )
    assert not stale, f"exclusions naming no file: {stale}"


def test_the_walk_is_not_excluded() -> None:
    # The claim the whole slice rests on: the file where membership is decided is
    # inside the sweep. An exclusion that swallowed it would leave the coverage
    # number intact and the mutation number gone.
    globs = _excluded_globs()
    for path in _swept():
        assert not any(_matches(g, path) for g in globs), f"{path} is excluded"
        assert (ROOT / path).exists(), f"{path} does not exist"


def _configured_timeout(config: Path) -> float:
    """Read `timeout_multiplier` from a sweep configuration."""
    match = re.search(
        r"^timeout_multiplier\s*=\s*([0-9.]+)",
        config.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None, f"{config.name} sets no timeout_multiplier"
    return float(match.group(1))


def _lane_timeouts() -> set[float]:
    """Every `--timeout-multiplier` the workflow's sweeps pass."""
    text = WORKFLOW.read_text(encoding="utf-8")
    found = {float(value) for value in re.findall(r"--timeout-multiplier (\S+)", text)}
    assert found, "no lane passes a timeout multiplier"
    return found


def test_the_configured_timeout_is_the_one_the_lanes_run_under() -> None:
    """A local sweep judges a slow mutant the way the lane judges it.

    Every lane passes `--timeout-multiplier` on the command line, which wins
    over the configuration. A configuration holding a different number is one
    that applies to local runs alone: a mutant the lane judges is reported
    timed out here, and a timeout counts as a survivor, so the ratchet fails
    over a machine rather than over the tests. One number, in one place, and
    the lanes spell the same one.
    """
    lanes = _lane_timeouts()
    assert len(lanes) == 1, f"the lanes pass different multipliers: {sorted(lanes)}"
    lane = lanes.pop()
    for config in (CONFIG, ROOT / ".cargo" / "mutants-pytest.toml"):
        assert _configured_timeout(config) == lane, (
            f"{config.name} sets {_configured_timeout(config)} and the lanes run "
            f"at {lane}"
        )


def test_the_scope_carries_its_reason() -> None:
    # An exclusion list with no argument beside it is a list nobody can review.
    text = CONFIG.read_text(encoding="utf-8")
    assert "pytest" in text, "the config must say why the excluded files are excluded"
    assert "interpreter-tests" in text, (
        "the config must say how the walk is reachable from `cargo test`"
    )


def test_the_glob_matcher_distinguishes_the_shapes_it_is_used_with() -> None:
    # `**` crosses directories and `*` does not, which is what makes an
    # `examples/**` exclusion different from a per-file one.
    assert _matches(
        "crates/valgebra-py/examples/**", "crates/valgebra-py/examples/w.rs"
    )
    assert not _matches("crates/valgebra-py/src/*.rs", "crates/valgebra-py/src/a/b.rs")
    assert _matches("crates/valgebra-py/src/lib.rs", "crates/valgebra-py/src/lib.rs")
    assert not _matches(
        "crates/valgebra-py/src/lib.rs", "crates/valgebra-py/src/check/lib.rs"
    )


@pytest.mark.skipif(
    shutil.which("cargo-mutants") is None, reason="cargo-mutants is not installed"
)
def test_no_excused_mutant_has_outlived_its_subject() -> None:
    # `--list` applies the exclusions, so the unfiltered listing is the universe
    # to match against; matching the filtered one would excuse every entry that
    # works and every entry that is dead alike.
    mutants = _every_mutant()
    assert len(mutants) >= 100, f"the mutant listing returned {len(mutants)} lines"

    patterns = _excluded_regexes()
    assert patterns, "the exclusion list parsed empty"
    stale = sorted(
        pattern
        for pattern in patterns
        if not any(re.search(pattern, mutant) for mutant in mutants)
    )
    assert not stale, (
        f"exclusions matching no mutant: {stale}. "
        "Delete each with the argument beside it, or fix the pattern."
    )
