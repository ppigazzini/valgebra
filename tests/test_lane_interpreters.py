"""Every lane that installs an interpreter names which one, and runs there.

A ceiling in `scripts/perf_compare.json`, a band in `scripts/perf_budget.json`
and a mutation sweep are each claims about an interpreter as much as about the
code: a ratio is a property of the pair running, a budget's band is as wide as
the distance between two releases, and a mutant on a version-gated branch
is killable on one interpreter and unviable on the next.

The wrapper action's `python-version` defaults to the empty string, documented
as "left empty, uv picks one itself", and what uv picks is whatever the runner
image ships. So a lane naming none rests those claims on an image's default,
and the day the image moves they all change meaning at once with every lane
green.

Held over both workflows: a `setup-uv` use with no version fails, and the
version each lane names is the one its own comment argues for.

The second half is the implementations. The packaging metadata states CPython
and PyPy, which is a promise that a schema means the same thing on both. PyPy's
C API answers differently as well as exporting differently: `PyTuple_Size` goes
through the object's own `__len__` there, so a `tuple` subclass overriding it
can walk a reader past the end of its storage and kill the process, which an
import check cannot see, because an import is not an answer. So a stated
implementation has a lane that runs the suite, and a lane that runs the suite on
an implementation is one the metadata states.

The third is the build caches. A cache a job saves, restored by the job's next
run or by another job sharing its key, links the interpreter it was built
against: pyo3 resolves it again only when `PYO3_PYTHON` changes, and every lane
names `.venv/bin/python` whichever build stands behind it. So every rust cache's
key names each interpreter its job installs, and the jobs sharing one install
one interpreter: a lane moving its interpreter moves the key.
A lane that builds against the image's own interpreter has the image for its
interpreter, and the key the cache action computes does not name the image: a
cache saved on Ubuntu 24.04 links libpython3.12 on an image that has none. So
the toolchain wrapper exports the image's name under a prefix the key reads,
and every job restores its cache behind the wrapper.

LEDGER: no lane installs an interpreter without naming it
"""

from __future__ import annotations

import re

import pytest

from _toml import load
from _tree import ROOT, WORKFLOWS, read_workflow

# A repository check: it reads the workflows, which ship in no wheel.
pytestmark = pytest.mark.repository


#: The wrapper every lane installs its interpreter through.
WRAPPER = "/.github/actions/setup-uv"

#: The implementation a bare version string means, which nobody spells out.
CPYTHON = "cpython"

#: A `run:` step that invokes pytest, rather than one that installs it: the
#: dependency is named in quotes beside its floor, and a match on the bare word
#: would read the install as a run.
_RUNS_PYTEST = re.compile(r"(?:^|\s)(?:uv run |[\w./-]*python -m )?pytest\b")


#: The toolchain wrapper, which names the runner image for the cache key.
TOOLCHAIN = "/.github/actions/setup-rust"

#: The build cache every Rust lane restores.
RUST_CACHE = "Swatinem/rust-cache@"

#: The prefixes a variable's name starts with for the cache action to read it
#: into the key.
_KEYED_PREFIXES = ("CARGO", "CC", "CFLAGS", "CXX", "CMAKE", "RUST")

#: A `run:` line exporting the image's name to the steps after it.
_EXPORTS_IMAGE = re.compile(
    r'\b([A-Z][A-Z_]*)=\$\{ImageOS\b[^}]*\}.*>>\s*"\$GITHUB_ENV"'
)

#: An interpreter uv is asked for by name: `pypy-3.11`, `graalpy-24`.
_NAMED = re.compile(r"\b([a-z][a-z_]*[a-z])-\d")

#: One asked for by version alone, which is how CPython is spelled.
_BARE = re.compile(r"(?<![\w.-])\d+\.\d+")


def _interpreter_text(job: dict, step: dict) -> str:
    """Read the interpreter a lane installs, with its matrix folded in.

    A matrix leg names its version through an expression, so the step alone
    says `${{ matrix.python-version }}` and nothing about an implementation.
    The matrix's own value -- a list, or the expression that chooses between
    two lists -- is the text that does, so it is read in the expression's
    place rather than the lane being read as CPython by default. An `include`
    row is read for the same key alone: its other keys name a runner, and a
    runner label such as `macos-26` is not an interpreter.
    """
    named = str((step.get("with") or {}).get("python-version") or "")
    if "matrix." not in named:
        return named
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    parts = [named]
    for key, value in matrix.items():
        if key == "include":
            parts.extend(
                str(row[name])
                for row in value
                for name in row
                if f"matrix.{name}" in named
            )
        elif f"matrix.{key}" in named:
            parts.append(str(value))
    return " ".join(parts)


def _implementations_in(text: str) -> set[str]:
    """Which implementations a lane's interpreter text asks uv for.

    uv spells a non-CPython interpreter by naming it and leaves CPython unsaid,
    so a version standing on its own is CPython.
    """
    found = set(_NAMED.findall(text.lower()))
    if _BARE.search(text):
        found.add(CPYTHON)
    return found


def _stated_implementations() -> set[str]:
    """Read the implementations the packaging metadata promises to support."""
    prefix = "Programming Language :: Python :: Implementation :: "
    return {
        row.removeprefix(prefix).lower()
        for row in load(ROOT / "pyproject.toml")["project"]["classifiers"]
        if row.startswith(prefix)
    }


def _implementations_running_the_suite() -> dict[str, list[str]]:
    """Group every lane that runs pytest by the implementation it installs."""
    running: dict[str, list[str]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = read_workflow(path)
        for job_name, job in (workflow.get("jobs") or {}).items():
            steps = job.get("steps") or []
            if not any(_RUNS_PYTEST.search(str(step.get("run", ""))) for step in steps):
                continue
            for step in steps:
                if not str(step.get("uses", "")).endswith(WRAPPER):
                    continue
                for name in _implementations_in(_interpreter_text(job, step)):
                    running.setdefault(name, []).append(f"{path.name}:{job_name}")
    return running


def _uses_without_a_version() -> list[str]:
    """Return `job/step` for every wrapper use that names no interpreter."""
    unnamed = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = read_workflow(path)
        for job_name, job in (workflow.get("jobs") or {}).items():
            for index, step in enumerate(job.get("steps") or []):
                uses = str(step.get("uses", ""))
                if not uses.endswith(WRAPPER):
                    continue
                version = (step.get("with") or {}).get("python-version")
                if not str(version or "").strip():
                    unnamed.append(f"{path.name}:{job_name}:step {index}")
    return unnamed


def test_no_lane_lets_the_image_choose_its_interpreter() -> None:
    unnamed = _uses_without_a_version()
    assert not unnamed, (
        f"lanes installing an interpreter without naming one: {unnamed}. "
        "Add `python-version` with the reason that lane needs that version, or "
        "the claims it makes belong to the runner image rather than to this tree."
    )


def test_the_wrapper_still_defaults_to_letting_uv_choose() -> None:
    """The rule above is about the lanes, and it is worth only what the default is.

    If the wrapper itself started naming a version, every lane would inherit one
    and this ledger would pass while saying nothing. The check is that the hole
    it closes is still open at the wrapper.
    """
    action = read_workflow(ROOT / ".github" / "actions" / "setup-uv" / "action.yml")
    default = action["inputs"]["python-version"].get("default", "")
    assert not str(default).strip(), (
        "the wrapper names a default interpreter, so a lane naming none inherits "
        "it and this ledger checks nothing; either drop the default or hold the "
        "lanes to the wrapper's value instead"
    )


def test_every_stated_implementation_runs_the_suite_somewhere() -> None:
    stated = _stated_implementations()
    running = _implementations_running_the_suite()
    assert stated, "the packaging metadata states no implementation to hold"
    missing = sorted(stated - running.keys())
    assert not missing, (
        f"stated as targets with no lane running the suite: {missing}. "
        "A classifier is a promise that a schema means the same thing there, "
        "and an import check cannot see an answer that differs."
    )


def test_no_lane_runs_the_suite_on_an_implementation_nobody_states() -> None:
    """The other direction: a lane is the evidence for a claim somebody made."""
    unstated = sorted(
        _implementations_running_the_suite().keys() - _stated_implementations()
    )
    assert not unstated, (
        f"lanes run the suite on implementations the metadata does not state: "
        f"{unstated}. Add the classifier, or the lane is spending minutes on a "
        "target no consumer is told about."
    )


def _caches() -> list[tuple[str, str, list[str]]]:
    """Give each rust cache as its job, its key, and the interpreters it installs."""
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job_name, job in (read_workflow(path).get("jobs") or {}).items():
            steps = job.get("steps") or []
            installed = [
                str(step["with"]["python-version"])
                for step in steps
                if "python-version" in (step.get("with") or {})
            ]
            found.extend(
                (
                    f"{path.name}:{job_name}",
                    " ".join(
                        str((step.get("with") or {}).get(name, ""))
                        for name in ("key", "shared-key")
                    ),
                    installed,
                )
                for step in steps
                if str(step.get("uses", "")).startswith(RUST_CACHE)
            )
    return found


def test_every_rust_cache_names_the_interpreter_its_job_installs() -> None:
    caches = _caches()
    # The scan is the detector: a lane that installs no interpreter and builds
    # no binding has nothing to name, and no lane at all would pass.
    keyed = [(job, key, installed) for job, key, installed in caches if installed]
    assert len(keyed) >= 10, f"the scan found {keyed}"
    unnamed = [
        f"{job} keys its cache `{key.strip() or 'nothing'}` and installs {version}"
        for job, key, installed in keyed
        for version in installed
        if version not in key
    ]
    assert not unnamed, (
        f"{unnamed}: a cache that does not move with the interpreter restores a "
        "build configured against the one before"
    )


def _shared_caches() -> dict[str, set[str]]:
    """Give each shared rust cache key, with the interpreters its jobs install."""
    shared: dict[str, set[str]] = {}
    for job in read_workflow()["jobs"].values():
        steps = job.get("steps") or []
        keys = [
            str(step["with"]["shared-key"])
            for step in steps
            if str(step.get("uses", "")).startswith("Swatinem/rust-cache@")
            and "shared-key" in (step.get("with") or {})
        ]
        installed = " ".join(
            str(step["with"]["python-version"])
            for step in steps
            if "python-version" in (step.get("with") or {})
        )
        for key in keys:
            shared.setdefault(key, set()).add(installed or "none")
    return shared


def test_jobs_sharing_a_rust_cache_install_the_interpreter_it_names() -> None:
    shared = _shared_caches()
    # The scan is the detector: no shared cache at all would pass having read
    # nothing.
    assert shared, "no job shares a rust cache"
    for key, interpreters in shared.items():
        assert len(interpreters) == 1, (
            f"`{key}` is restored by jobs on {sorted(interpreters)}: a cache built "
            "against one links the others against it"
        )
        (interpreter,) = interpreters
        assert interpreter in key, (
            f"`{key}` does not name {interpreter}, so moving the interpreter "
            "leaves the cache behind"
        )


def test_the_toolchain_wrapper_names_the_image_in_the_cache_key() -> None:
    wrapper = ROOT / TOOLCHAIN.removeprefix("/") / "action.yml"
    exported = [
        name
        for step in read_workflow(wrapper)["runs"]["steps"]
        for name in _EXPORTS_IMAGE.findall(str(step.get("run", "")))
    ]
    assert any(name.startswith(_KEYED_PREFIXES) for name in exported), (
        f"the toolchain wrapper exports the image as {exported or 'nothing'}, "
        f"and the cache key reads only names starting {_KEYED_PREFIXES}: a cache "
        "saved on one image restores on the next, linking the interpreter the "
        "first image carried"
    )


def test_every_rust_cache_restores_behind_the_toolchain_wrapper() -> None:
    restored = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job_name, job in (read_workflow(path).get("jobs") or {}).items():
            wrapped = False
            for step in job.get("steps") or []:
                uses = str(step.get("uses", ""))
                wrapped = wrapped or uses.endswith(TOOLCHAIN)
                if not uses.startswith(RUST_CACHE):
                    continue
                restored += 1
                where = f"{path.name}:{job_name}"
                assert wrapped, (
                    f"{where} restores a rust cache before the toolchain wrapper "
                    "names the image, so its key is the same on every image"
                )
                hashed = (step.get("with") or {}).get("add-rust-environment-hash-key")
                assert str(hashed).lower() != "false", (
                    f"{where} keeps the environment out of its cache key, and "
                    "the image's name with it"
                )
    # The scan is the detector: no cache at all would pass having read nothing.
    assert restored, "no job restores a rust cache"
