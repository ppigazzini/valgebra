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

The scope is not the whole of the configuration. A sweep's *environment* decides
whether its verdict is about the tests or about the rig: proptest shrinks a
failure for as long as it takes, and a mutant's budget is the baseline's test
time -- measured with no failure to shrink -- times a multiplier, so an
unbounded shrink turns a caught mutant into a timeout and a timeout returns no
verdict at all. The lanes bound it. A reader running a sweep by hand from the
tooling page gets whatever that page's recipe sets, so the two are held equal
here: a variable a lane needs and the page omits is a recipe that hands its
reader a rig fault.

LEDGER: every binding file is swept or excluded by name
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

# The repository checks are not the product suite: this file reads the tree,
# the configuration and the gate scripts, none of which ship in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".cargo" / "mutants.toml"

#: The three sweeps' accepted survivors, one file per sweep. A sweep judges a
#: fragment of the tree and the three fragments are disjoint, so neither
#: baseline can absorb another's survivors.
BASELINES = (
    ROOT / "scripts" / "mutation_baseline.json",
    ROOT / "scripts" / "mutation_baseline_walk.json",
    ROOT / "scripts" / "mutation_baseline_pytest.json",
)
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


#: `cargo mutants --list` prints `path:line:col: description`; a baseline
#: records `path: description`, because a line number moves whenever a comment
#: above it does. The same normalisation as
#: `scripts/mutation_gate.py::_identity`, which is what writes those keys.
_POSITION = re.compile(r"^(?P<path>[^:]+):\d+:\d+:\s*(?P<desc>.*)$")


def _identity(line: str) -> str:
    """Give a listed mutant the name a baseline records it under."""
    stripped = line.strip()
    match = _POSITION.match(stripped)
    return f"{match['path']}: {match['desc']}" if match else stripped


@pytest.mark.skipif(
    shutil.which("cargo-mutants") is None, reason="cargo-mutants is not installed"
)
def test_no_accepted_survivor_has_outlived_its_subject() -> None:
    """An accepted survivor names a mutant the sweep still offers.

    The other half of the exclusion check above, over the other table. An
    `exclude_re` entry excuses a mutant from being *run*; an `_accepted` entry
    excuses one from being *counted*, and both are holes in the coverage claim
    that a reader has to take on the argument written beside them. An argument
    for a mutant that no longer exists is one nobody can check, and it hides
    the case this catches: code that moves from one sweep's fragment to
    another's arrives in the new one with no tests and leaves behind an excuse
    that reads as coverage.

    `scripts/mutation_gate.py` already refuses an entry whose *file* is gone.
    A file that stays while the function moves out of it is the gap, and it is
    the one that happened.
    """
    offered = {_identity(line) for line in _every_mutant()}
    # The listing is the detector: an empty universe would excuse every entry.
    assert len(offered) >= 100, f"the mutant listing returned {len(offered)} names"

    stale = {
        baseline.name: gone
        for baseline in BASELINES
        for gone in [
            sorted(
                key
                for key in json.loads(baseline.read_text(encoding="utf-8"))["_accepted"]
                if key not in offered
            )
        ]
        if gone
    }
    assert not stale, (
        f"accepted survivors naming no mutant the sweep offers: {stale}. "
        "Delete each with the argument beside it, or move it to the baseline "
        "of the sweep that reaches the code now -- and only after that sweep "
        "has reported it, with a reason written for the cause it survives for "
        "there."
    )


# --- The sweep's environment, held between the lane and the recipe -----------

WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
TOOLING = ROOT / "docs" / "dev" / "07-tooling-ci.md"

#: A fenced block of the tooling page, with the language it is tagged.
_FENCED = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _sweep_jobs() -> dict[str, dict[str, str]]:
    """Give each workflow job that runs a sweep, with the environment it sets."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    found: dict[str, dict[str, str]] = {}
    for name, job in workflow["jobs"].items():
        steps = job.get("steps") or []
        if not any("cargo mutants" in (step.get("run") or "") for step in steps):
            continue
        found[name] = {key: str(value) for key, value in (job.get("env") or {}).items()}
    return found


def _recipes() -> list[str]:
    """Give the tooling page's by-hand sweep recipes."""
    blocks = [
        body
        for language, body in _FENCED.findall(TOOLING.read_text(encoding="utf-8"))
        if language == "bash" and "cargo mutants" in body
    ]
    assert blocks, "the tooling page carries no sweep recipe"
    return blocks


def test_every_variable_a_sweep_lane_sets_is_one_the_recipe_sets() -> None:
    """A recipe missing the lane's environment hands its reader a rig fault.

    The variable that matters is the shrink bound, and what it buys is the
    difference between a verdict and no verdict: unbounded, a mutant the tests
    *caught* spends the whole budget shrinking a counterexample nobody reads,
    reports as a timeout, and stops the gate as a run that could not measure.
    The lane sets it and says why. A reader following the page by hand gets the
    page's recipe, so the page's recipe carries what the lane carries.

    Held by name rather than by value: a seed is the runner's to choose and a
    local sweep chooses its own, but a sweep run without one at all explores the
    same draw every time and reports a stable score that is about the draw.
    """
    lanes = _sweep_jobs()
    assert len(lanes) >= 2, sorted(lanes)
    recipes = _recipes()
    wanted = {variable for environment in lanes.values() for variable in environment}
    assert wanted, sorted(lanes)
    unset = sorted(
        variable
        for variable in wanted
        if not any(variable in recipe for recipe in recipes)
    )
    assert not unset, (
        f"variables the sweep lanes set that no by-hand recipe does: {unset}. "
        "A reader runs the page's recipe, so it carries what the lane carries "
        "-- or the verdict it returns is about the rig."
    )
