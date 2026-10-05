"""Every wheel the release builds on a runner is smoked on that runner.

`release.yml` builds wheels in one matrix and smokes them in another, joined by
an artifact name each side spells for itself. A build row whose artifact no
smoke row downloads ships a wheel nothing ran; that is how the PyPy wheels
shipped for two releases as profile-guided builds that died at the walk's depth
bound while the push lane stayed green on a plain wheel it built itself. A
smoke row naming an artifact no build row uploads fails the release instead,
late and for the wrong reason.

Held here, both ways: every build row has a smoke row for its artifact, and
every smoke row names an artifact a build row uploads -- the musllinux sets in
Alpine containers, the only place a musl wheel loads. Within a set, every
release a build row names is loaded by its smoke row: the product suite on the
ends, an import and one check on the releases between. Each wheel is a build of
its own, profiled on its own interpreter, and 0.0.15 shipped 36 of its 62
without loading any of them -- every cp311, cp312, cp313 and GIL cp314 wheel,
and every musllinux one. A smoke is the product suite rather than an import, for the
wheels and for the build from source, and the import it starts with fails on a
warning. And the PyPy wheel is built in a row of its own, without the profile:
the rows that carry the profile name their interpreters, and none of them names
PyPy.

LEDGER: every wheel the release builds is loaded, its ends tested; PyPy's is plain
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = yaml.safe_load(
    (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
)
BUILDS: list[dict[str, object]] = WORKFLOW["jobs"]["wheels"]["strategy"]["matrix"][
    "platform"
]
#: The jobs that load a wheel set, each a matrix of rows naming one.
SMOKE_JOBS = ("smoke", "smoke-musl")
SMOKES: list[dict[str, object]] = [
    row
    for job in SMOKE_JOBS
    for row in WORKFLOW["jobs"][job]["strategy"]["matrix"]["wheel"]
]
#: The wheel sets no smoke row downloads, with the reason. None: the musllinux
#: sets, the last ones here, are loaded in Alpine containers. Their cp314t and
#: pp311 wheels are not, since no Alpine image carries a free-threaded CPython
#: or a PyPy, and the image builds what it carries, so no row names them.
NOT_SMOKED: dict[str, str] = {}


def _artifact(build: dict[str, object]) -> str:
    """Spell the artifact name the upload step composes for a build row."""
    flavour = build.get("variant") or build.get("manylinux") or "native"
    return f"wheels-{build['runner']}-{build['target']}-{flavour}"


def test_the_matrices_were_read() -> None:
    """The parse is a detector, so it must be shown to have read both matrices."""
    assert len(BUILDS) >= 8, BUILDS
    assert len(SMOKES) >= 6, SMOKES


def test_every_built_wheel_set_is_smoked_or_excused() -> None:
    smoked = {str(row["artifact"]) for row in SMOKES}
    built = {_artifact(row) for row in BUILDS}
    unsmoked = sorted(built - smoked - set(NOT_SMOKED))
    assert not unsmoked, (
        f"wheel sets the release builds and never runs: {unsmoked}. Add a smoke "
        "row for each, or name it in NOT_SMOKED with the reason."
    )
    stale = sorted(name for name in NOT_SMOKED if name not in built)
    assert not stale, f"excused wheel sets no build row uploads: {stale}"


def test_every_smoke_row_names_a_wheel_set_the_release_builds() -> None:
    built = {_artifact(row) for row in BUILDS}
    phantom = sorted(
        str(row["artifact"]) for row in SMOKES if row["artifact"] not in built
    )
    assert not phantom, f"smoke rows naming an artifact no build row uploads: {phantom}"


def test_pypy_is_built_plain() -> None:
    """A profiled extension dies on PyPy at the depth bound; the plain one does not."""
    profiled = [row for row in BUILDS if row.get("pgo")]
    assert profiled, "no build row carries the profile"
    for row in profiled:
        interpreters = str(row.get("interpreter", row.get("pythons", "")))
        assert "pypy" not in interpreters.lower(), f"a profiled row builds PyPy: {row}"
    plain_pypy = [
        row
        for row in BUILDS
        if not row.get("pgo") and "pypy" in str(row.get("interpreter", ""))
    ]
    assert plain_pypy, "no plain build row names PyPy"
    unsmoked = [
        _artifact(row)
        for row in plain_pypy
        if not any(smoke["artifact"] == _artifact(row) for smoke in SMOKES)
    ]
    assert not unsmoked, f"PyPy wheel sets with no smoke row: {unsmoked}"


def _releases(spelled: str) -> set[str]:
    """Give the CPython releases a row spells, `3.14t` for a free-threaded one."""
    return set(re.findall(r"(?<![\d.@])3\.\d+t?\b", spelled))


def test_every_release_a_build_row_names_is_loaded() -> None:
    """A set's smoke row loads every release the set's build row names.

    Read for the CPython rows that name their interpreters. A PyPy set holds
    one wheel, which the set's own row loads; a row that builds what its image
    carries names none, and the musllinux image's free-threaded and PyPy builds
    are the wheels left unloaded, with the reason `NOT_SMOKED` gives.
    """
    unloaded: dict[str, list[str]] = {}
    for build in BUILDS:
        if str(build.get("variant", "")).startswith("pypy"):
            continue
        named = _releases(f"{build.get('interpreter', '')} {build.get('pythons', '')}")
        if not named:
            continue
        loaded = set().union(
            *(
                _releases(f"{row.get('pythons', '')} {row.get('imports', '')}")
                for row in SMOKES
                if row["artifact"] == _artifact(build)
            )
        )
        if missing := sorted(named - loaded):
            unloaded[_artifact(build)] = missing
    assert not unloaded, (
        f"releases built and never loaded: {unloaded}. Name each in its smoke "
        "row's `imports`, or in `pythons` for the suite."
    )


def _steps(job: str) -> list[dict[str, str]]:
    return WORKFLOW["jobs"][job]["steps"]


@pytest.mark.parametrize("job", ["smoke", "smoke-musl", "sdist-smoke"])
def test_every_smoke_runs_the_product_suite(job: str) -> None:
    """An import is not a smoke: the crash that shipped showed only in the suite.

    A profiled PyPy build died at the walk's depth bound, which no import
    reaches. Every row of the job runs the suite, so the step carries no
    condition a row could leave unset.
    """
    runs = [
        step
        for step in _steps(job)
        if '-m "not repository"' in str(step.get("run", ""))
    ]
    assert runs, f"{job} runs no product suite"
    conditional = [str(step.get("name")) for step in runs if "if" in step]
    assert not conditional, (
        f"{job} runs the suite only where a row says so: {conditional}"
    )


@pytest.mark.parametrize("job", ["smoke", "smoke-musl", "sdist-smoke"])
def test_every_smoke_import_fails_on_a_warning(job: str) -> None:
    imports = [
        str(step["run"])
        for step in _steps(job)
        if "import" in str(step.get("run", "")) and "valgebra as v" in str(step["run"])
    ]
    assert imports, f"{job} imports nothing"
    silent = [run for run in imports if "-W error" not in run]
    assert not silent, f"{job} imports with warnings left as warnings"
