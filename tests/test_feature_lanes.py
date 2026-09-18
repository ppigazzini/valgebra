"""A feature that gates tests is run by a lane named for tests.

A Cargo feature can hide a whole corpus. `interpreter-tests` gates the
binding's four frontend corpora -- the tables that say what `build_schema`
makes of a live annotation and what the walk makes of a live value -- and
`cargo test --workspace` does not pass it, so those tables run only where some
lane names the feature.

For one push the only lane that did was `binding coverage`. A corpus row went
stale under a change to the check it describes, and the lane that reddened was
the one whose job is to *measure*: its failure reads as a coverage problem, the
row's own suite stayed green on three operating systems, and a `cargo test
--workspace` on a developer's machine agreed. A test that fails is worth what
its failure says, and "the measurement broke" is not what a stale corpus means.

So the rule, held in both directions:

* every feature a workspace crate declares is passed to a `cargo test` by some
  job that is not itself a measurement -- or it is excused here by name, with
  the argument for why no test lane wants it;
* no excuse names a feature the tree no longer declares, and none names a
  feature a test lane does in fact run.

A **measurement** lane is one whose name says so: coverage lanes and mutation
sweeps. Both run tests, and both run them to produce a number -- a percentage,
a survivor list -- so a red one names that number rather than the test. They
are the lanes this ledger exists to stop being the only witness.

LEDGER: every crate feature runs in a test lane or is excused by name
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# A repository check: it reads the workflows and the manifests, neither of
# which ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

#: Features no test lane runs, each with the reason. An entry is a hole in the
#: claim above, so it carries an argument rather than a name alone.
EXCUSED: dict[str, str] = {
    "pytest-sweep": (
        "It builds the extension and runs the Python suite once per test "
        "binary, which is the cost of letting a mutation sweep observe a "
        "suite it otherwise cannot reach. The suite it runs is the one every "
        "`python` lane already runs against the same extension, so a test "
        "lane passing this feature would run those tests a second time and "
        "report nothing the first run does not."
    ),
}

#: A job whose red says a measurement moved rather than a test failed.
_MEASURES = re.compile(r"coverage|mutants", re.IGNORECASE)

#: A feature list as a `cargo` invocation spells one: comma-separated, and
#: quoted where a lane passes more than one.
_FEATURES = re.compile(r"--features[= ]+[\"']?([\w,./-]+)[\"']?")

#: Where a `cargo test` starts. `cargo mutants` and `cargo clippy` take the
#: same flag and are not test runs: a sweep reports a survivor and a lint
#: reports a lint, so neither answers for a corpus that has gone stale.
_CARGO_TEST = re.compile(r"\bcargo\s+(?:\+\S+\s+)?test\b")


def _features_declared() -> dict[str, str]:
    """Every feature a workspace crate declares, with the manifest that does.

    Parsed with a regex rather than a TOML library, as the sibling ledgers
    read `.cargo/mutants.toml`: `tomllib` is 3.11+ and this suite runs from
    3.10, and a third-party parser would be a dependency added for one table.
    A feature is a bare key at the top of a line inside `[features]`; the
    comments above each one are prose and carry no key.
    """
    declared: dict[str, str] = {}
    for manifest in sorted(ROOT.glob("crates/*/Cargo.toml")):
        inside = False
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith("["):
                inside = line.strip() == "[features]"
                continue
            if not inside or line.startswith("#") or not line.strip():
                continue
            key = line.split("=", 1)[0].strip()
            if key:
                declared[key] = str(manifest.relative_to(ROOT))
    return declared


def _logical_lines(text: str) -> list[str]:
    r"""Give a `run:` block's commands, with comments dropped and `\` joined.

    A lane wraps a long command over several lines, so a matcher reading the
    file line by line sees `cargo test` on one and its `--features` on the
    next and reports a feature nothing passes.
    """
    kept = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    joined = "\n".join(kept).replace("\\\n", " ")
    return joined.splitlines()


def _test_features(job: dict) -> set[str]:
    """Give the features this job's `cargo test` steps enable."""
    found: set[str] = set()
    for step in job.get("steps") or []:
        script = step.get("run") if isinstance(step, dict) else None
        if not script:
            continue
        for line in _logical_lines(script):
            if not _CARGO_TEST.search(line):
                continue
            for group in _FEATURES.findall(line):
                found.update(part.split("/")[-1] for part in group.split(","))
    return found


def _lanes_running(feature: str) -> list[str]:
    """Every job outside a measurement that passes `feature` to `cargo test`."""
    lanes = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, job in (workflow.get("jobs") or {}).items():
            spelled = f"{job_id} {job.get('name', '')}"
            if _MEASURES.search(spelled):
                continue
            if feature in _test_features(job):
                lanes.append(f"{path.name}:{job_id}")
    return lanes


def test_every_feature_is_run_by_a_lane_that_is_not_a_measurement() -> None:
    declared = _features_declared()
    # The glob is the detector: an empty universe would pass having read
    # nothing, which is the failure this ledger exists to rule out.
    assert declared, "no crate declares a feature, so the manifests were misread"

    unrun = sorted(
        name for name in declared if name not in EXCUSED and not _lanes_running(name)
    )
    assert not unrun, (
        f"features no test lane runs: {unrun}. Pass each to a `cargo test` in "
        "a lane whose name says it runs tests, or excuse it here with the "
        "reason no test lane wants it. A coverage or mutation lane running it "
        "does not count: a red there names a number, not the test."
    )


def test_no_excuse_is_stale() -> None:
    declared = _features_declared()

    gone = sorted(set(EXCUSED) - set(declared))
    assert not gone, f"excuses naming a feature no crate declares: {gone}"

    run = sorted(name for name in EXCUSED if _lanes_running(name))
    assert not run, (
        f"excused features a test lane does run: {run}. Remove the excuse; it "
        "claims a hole the tree does not have."
    )


def test_a_sweep_and_a_lint_are_not_test_runs() -> None:
    """The extraction rule a reader would get wrong, driven both ways."""
    sweep = {"steps": [{"run": "cargo mutants --features interpreter-tests"}]}
    lint = {"steps": [{"run": "cargo clippy --all-targets --features fake"}]}
    assert _test_features(sweep) == set()
    assert _test_features(lint) == set()

    # And a wrapped command is one command: the feature is on the second line.
    wrapped = {
        "steps": [
            {
                "run": (
                    'LD_LIBRARY_PATH="$libdir" \\\n'
                    "  cargo test -p valgebra-py --features interpreter-tests\n"
                )
            }
        ]
    }
    assert _test_features(wrapped) == {"interpreter-tests"}

    # A feature named only in a comment is a feature nothing runs.
    commented = {"steps": [{"run": "# cargo test --features fake\ncargo test\n"}]}
    assert _test_features(commented) == set()
