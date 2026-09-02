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
BINDING = ROOT / "crates" / "valgebra-py"

# The binding files inside the sweep: the membership walk, where soundness is
# decided, and the context it carries. Both are reachable from `cargo test`
# because the `interpreter-tests` feature links an embedded Python and the walk
# carries its own value corpus; the context's two predicates are asserted over
# every mode by its own tests.
SWEPT = {
    "crates/valgebra-py/src/check/walk.rs",
    "crates/valgebra-py/src/check/ctx.rs",
}


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
    return {
        str(path.relative_to(ROOT)).replace("\\", "/") for path in BINDING.rglob("*.rs")
    }


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
        if path not in SWEPT and not any(_matches(g, path) for g in globs)
    )
    assert not unaccounted, (
        f"binding files neither swept nor excluded: {unaccounted}. "
        "Add each to exclude_globs with its reason, or bring it into the sweep."
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
    for path in SWEPT:
        assert not any(_matches(g, path) for g in globs), f"{path} is excluded"
        assert (ROOT / path).exists(), f"{path} does not exist"


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
