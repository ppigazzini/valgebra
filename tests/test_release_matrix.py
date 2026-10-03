"""Every platform the release builds for has a wheel for every release it names.

`pyproject.toml` names the CPython releases the package supports in its
classifiers, and its free-threaded build as stable; the `wheels` matrix of
`release.yml` names, per platform, the interpreters a wheel is built for.
Nothing else compares the two. A release classified and missing from a
platform's list sends every installer there to the source distribution, which
compiles the extension on the user's machine -- the failure a per-version wheel
exists to spare them -- and the lifecycle ledger, which moves the classifiers on
the calendar, does not read the matrix.

Held both ways: every platform row builds every classified release, and its
free-threaded build from the first release that supports one, or the gap is
named with its reason; every gap is one the matrix has; and a row that leaves
its interpreters to the build image says so by name.

LEDGER: every classified release has a wheel on every platform, or a named gap
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from _toml import load

# The repository checks are not the product suite: this file reads the
# packaging metadata and the release workflow, neither of which ships.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent

#: The first release whose free-threaded build is supported rather than
#: experimental (PEP 779), and the first the installation page names.
FREE_THREADED_FROM = (3, 14)

#: Releases a platform does not build, by platform, each with the reason.
GAPS: dict[tuple[str, str], str] = {
    ("windows-11-arm aarch64", "3.10"): "the Windows arm64 set starts at 3.12",
    ("windows-11-arm aarch64", "3.11"): "the Windows arm64 set starts at 3.12",
    (
        "windows-11-arm aarch64",
        "3.14t",
    ): "no free-threaded row builds for Windows arm64",
    (
        "windows-11-arm aarch64",
        "3.15t",
    ): "no free-threaded row builds for Windows arm64",
}

#: Rows that name no interpreters, leaving them to the build image.
UNNAMED: dict[str, str] = {
    "ubuntu-latest x86_64 musllinux_1_2": (
        "`--find-interpreter` builds for what the musllinux image carries: in "
        "0.0.15, 3.10 through 3.14 and 3.14t"
    ),
    "ubuntu-latest aarch64 musllinux_1_2": (
        "`--find-interpreter` builds for what the musllinux image carries: in "
        "0.0.15, 3.10 through 3.14 and 3.14t"
    ),
}


def _rows() -> list[dict[str, str]]:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    return workflow["jobs"]["wheels"]["strategy"]["matrix"]["platform"]


def _platform(row: dict[str, str]) -> str:
    """Name a row's platform: the runner, the target, and a musl image's tag."""
    image = str(row.get("manylinux", ""))
    musl = f" {image}" if image.startswith("musllinux") else ""
    return f"{row['runner']} {row['target']}{musl}"


def _named(row: dict[str, str]) -> set[str] | None:
    """Give the releases a row builds, `3.14t` for a free-threaded one, or None.

    A row names them as `python3.N` executables or as the `3.N` versions it
    installs first; one naming neither builds whatever its image carries.
    """
    spelled = f"{row.get('interpreter', '')} {row.get('pythons', '')}"
    found = set(re.findall(r"(?<![\d.])3\.\d+t?\b", spelled))
    return found or None


def _cpython_rows() -> list[dict[str, str]]:
    """Give the build rows for CPython; the smoke ledger holds PyPy's."""
    return [
        row for row in _rows() if not str(row.get("variant", "")).startswith("pypy")
    ]


def _built() -> dict[str, set[str]]:
    """Give each platform the releases its rows build between them."""
    built: dict[str, set[str]] = {}
    for row in _cpython_rows():
        named = _named(row)
        if named is not None:
            built.setdefault(_platform(row), set()).update(named)
    return built


def _classified() -> set[str]:
    """Give the releases the classifiers name, free-threaded builds included."""
    classifiers = load(ROOT / "pyproject.toml")["project"]["classifiers"]
    releases = {
        match.group(1)
        for classifier in classifiers
        if (
            match := re.fullmatch(
                r"Programming Language :: Python :: (3\.\d+)", classifier
            )
        )
    }
    stable = any(
        c.startswith("Programming Language :: Python :: Free Threading :: 3")
        for c in classifiers
    )
    threaded = {
        f"{release}t"
        for release in releases
        if stable and tuple(map(int, release.split("."))) >= FREE_THREADED_FROM
    }
    return releases | threaded


def test_the_matrix_and_the_classifiers_were_read() -> None:
    assert len(_classified()) >= 5, sorted(_classified())
    assert len(_built()) >= 4, sorted(_built())


def test_every_platform_builds_every_release_the_classifiers_name() -> None:
    classified = _classified()
    missing = sorted(
        f"{platform}: {release}"
        for platform, releases in _built().items()
        for release in classified - releases
        if (platform, release) not in GAPS
    )
    assert not missing, (
        f"classified releases a platform ships no wheel for: {missing}. Name the "
        "interpreter in the platform's row, or record the gap with its reason."
    )


def test_no_platform_builds_a_release_the_classifiers_do_not_name() -> None:
    classified = _classified()
    extra = sorted(
        f"{platform}: {release}"
        for platform, releases in _built().items()
        for release in releases - classified
    )
    assert not extra, f"wheels for releases the classifiers do not name: {extra}"


def test_every_gap_is_one_the_matrix_has() -> None:
    built = _built()
    stale = sorted(
        f"{platform}: {release}"
        for platform, release in GAPS
        if platform not in built or release in built[platform]
    )
    assert not stale, f"gaps naming a platform or a release the rows build: {stale}"


def test_every_row_names_its_interpreters_or_says_why() -> None:
    unnamed = {_platform(row) for row in _cpython_rows() if _named(row) is None}
    assert unnamed == set(UNNAMED), (
        f"rows leaving their interpreters to the image: {sorted(unnamed)}; "
        f"excused: {sorted(UNNAMED)}"
    )
