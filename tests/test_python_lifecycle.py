"""The interpreters the tree supports are the ones CPython supports, today.

CPython publishes its calendar: one feature release every October (PEP 602),
each with a schedule PEP that dates its first alpha, its first release
candidate, its final release, and the month its security support ends. This
project follows that calendar rather than a list somebody remembers to edit,
and `docs/dev/09-releasing.md` states the rule. Held here against the date the
suite runs on:

* a release is **supported** from its first release candidate, when its ABI
  freezes, to the end of the month its security support ends. Every supported
  release is a blocking leg of the `python` job and a classifier, and none
  other is;
* a release past its first alpha and short of its first candidate is a leg the
  job forgives, so a break shows a year early and blocks nothing;
* every place the tree states its floor names the oldest supported release,
  and every place it states its newest release names the newest.

It fails on the calendar, with no commit: when a release's support ends the
floor has to move, and when the next one's first alpha lands it has to gain a
lane. Each message names the edit that turns it green. The dates are the
schedule PEPs' own, and a PEP amended is a row edited beside its number.

What this does not read: the release workflow's wheels. They are built per
platform, and the platforms set gaps of their own -- Windows arm64 carries 3.12
onward -- so which wheel a release ships is a different question from which
interpreter the tree supports.

LEDGER: every supported interpreter is one CPython's calendar supports today
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

# A repository check: it reads the packaging metadata, the workflow and a table
# under `tests/`, and none of them ships in a wheel.
pytestmark = pytest.mark.repository

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
WORKFLOW_TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
FLOOR_NAMES = json.loads(
    (ROOT / "tests" / "floor_names.json").read_text(encoding="utf-8")
)

#: The day the claims are read against. UTC, because a schedule PEP gives a day
#: and no timezone, and a day of slack either way decides nothing.
TODAY = datetime.datetime.now(datetime.timezone.utc).date()


class Release(NamedTuple):
    """One CPython feature release, as its schedule PEP dates it."""

    minor: int
    pep: int
    first_alpha: datetime.date
    first_candidate: datetime.date
    end_of_life: tuple[int, int]
    """The year and month the PEP gives for the last security release."""

    def supported(self, on: datetime.date) -> bool:
        """Whether the release is supported on `on`: frozen and still secured."""
        return self.first_candidate <= on and (on.year, on.month) <= self.end_of_life

    def in_development(self, on: datetime.date) -> bool:
        """Whether `on` falls between the first alpha and the first candidate."""
        return self.first_alpha <= on < self.first_candidate


def _day(text: str) -> datetime.date:
    return datetime.date.fromisoformat(text)


#: Every release the tree can name, from its schedule PEP. The end of life is
#: the month the PEP gives -- "approximately October" five years after the
#: final release -- and a release is supported through that month's end.
RELEASES = (
    Release(10, 619, _day("2020-10-05"), _day("2021-08-03"), (2026, 10)),
    Release(11, 664, _day("2021-10-05"), _day("2022-08-08"), (2027, 10)),
    Release(12, 693, _day("2022-10-24"), _day("2023-08-06"), (2028, 10)),
    Release(13, 719, _day("2023-10-13"), _day("2024-08-01"), (2029, 10)),
    Release(14, 745, _day("2024-10-15"), _day("2025-07-22"), (2030, 10)),
    Release(15, 790, _day("2025-10-14"), _day("2026-08-04"), (2031, 10)),
    Release(16, 826, _day("2026-10-13"), _day("2027-07-27"), (2032, 10)),
)
BY_MINOR = {release.minor: release for release in RELEASES}

#: PEP 602's cadence: a feature release's first alpha comes a year after the
#: previous one's, give or take the weekday it falls on.
CADENCE = datetime.timedelta(days=372)


def _supported(on: datetime.date) -> set[int]:
    return {release.minor for release in RELEASES if release.supported(on)}


def _in_development(on: datetime.date) -> set[int]:
    return {release.minor for release in RELEASES if release.in_development(on)}


def _minor(version: str) -> int:
    """Read `3.14`, `3.14t` or `py314` as its minor release."""
    found = re.fullmatch(r"(?:3\.|py3)(\d{1,2})t?", version)
    if found is None:
        message = f"{version!r} is not a CPython 3.x release"
        raise ValueError(message)
    return int(found.group(1))


def _one(pattern: str, text: str, where: str) -> int:
    """Read the single release `pattern` captures, refusing none or several."""
    found = re.findall(pattern, text, re.MULTILINE)
    if len(found) != 1:
        message = f"{where}: expected one match of {pattern!r}, found {found!r}"
        raise AssertionError(message)
    return _minor(found[0])


def _matrix() -> dict[int, bool]:
    """Give each release the `python` job runs, and whether its red blocks.

    A leg is forgiven by a `continue-on-error` expression naming its version,
    which is how a prerelease runs, or by the literal `true`, which forgives
    the whole job; `tests/test_version_gates.py` reads the same two shapes.
    """
    job = yaml.safe_load(WORKFLOW_TEXT)["jobs"]["python"]
    matrix = job["strategy"]["matrix"]
    versions = [str(version) for version in matrix["python-version"]]
    versions += [str(row["python-version"]) for row in matrix.get("include", [])]
    forgiven = str(job.get("continue-on-error", False)).strip().lower()
    lanes: dict[int, bool] = {}
    for version in versions:
        blocks = forgiven != "true" and f"'{version}'" not in forgiven
        lanes[_minor(version)] = lanes.get(_minor(version), False) or blocks
    return lanes


def _classifiers() -> set[int]:
    return {
        int(minor)
        for minor in re.findall(
            r'"Programming Language :: Python :: 3\.(\d+)"', PYPROJECT
        )
    }


def _name(minor: int) -> str:
    release = BY_MINOR.get(minor)
    return f"3.{minor}" + (f" (PEP {release.pep})" if release else "")


#: Every place the tree states its floor: the oldest release it supports.
FLOORS = {
    "requires-python in pyproject.toml": _one(
        r'^requires-python = ">=(3\.\d+)"', PYPROJECT, "pyproject.toml"
    ),
    "ruff's target-version": _one(
        r'^target-version = "(py3\d+)"', PYPROJECT, "pyproject.toml"
    ),
    "the ty floor leg in ci.yml": _one(
        r"ty check python/ --python-version (3\.\d+)$", WORKFLOW_TEXT, "ci.yml"
    ),
    "the typed consumer's first mypy target in ci.yml": _one(
        r"for target in (3\.\d+) 3\.\d+; do$", WORKFLOW_TEXT, "ci.yml"
    ),
    "the floor of tests/floor_names.json": _minor(FLOOR_NAMES["floor"]),
    "the python job's oldest leg": min(_matrix()),
}

#: Every place the tree states the newest release it supports.
NEWEST = {
    "[tool.ty.environment] python-version": _one(
        r'^python-version = "(3\.\d+)"', PYPROJECT, "pyproject.toml"
    ),
    "the typed consumer's last mypy target in ci.yml": _one(
        r"for target in 3\.\d+ (3\.\d+); do$", WORKFLOW_TEXT, "ci.yml"
    ),
    "the reach of tests/floor_names.json": _minor(FLOOR_NAMES["known_through"]),
}


def test_the_table_knows_the_release_in_development() -> None:
    """A release the table has no row for is one no claim below can see.

    The newest row's first alpha is at most a year old, so the release after
    it has not reached its own first alpha unseen; its schedule PEP is
    published months before that.
    """
    newest = RELEASES[-1]
    assert newest.first_alpha + CADENCE > TODAY, (
        f"3.{newest.minor + 1} is past its first alpha and has no row: add its "
        "schedule PEP's dates to RELEASES"
    )


def test_every_release_the_tree_names_has_a_row() -> None:
    named = (
        set(_matrix()) | _classifiers() | set(FLOORS.values()) | set(NEWEST.values())
    )
    assert named <= set(BY_MINOR), (
        f"releases with no schedule row: {named - set(BY_MINOR)}"
    )


def test_the_matrix_blocks_on_every_supported_release_and_no_other() -> None:
    supported = _supported(TODAY)
    blocking = {minor for minor, blocks in _matrix().items() if blocks}
    for minor in sorted(supported - blocking):
        pytest.fail(
            f"{_name(minor)} is supported since its first release candidate: give "
            "it a leg of the python job in ci.yml that is not forgiven"
        )
    for minor in sorted(blocking - supported):
        release = BY_MINOR[minor]
        if (TODAY.year, TODAY.month) > release.end_of_life:
            pytest.fail(
                f"{_name(minor)} is past the end of its security support "
                f"({release.end_of_life[0]}-{release.end_of_life[1]:02d}): drop it "
                "from the python job, the classifiers and every statement of the "
                "floor, and raise requires-python to the next release"
            )
        pytest.fail(
            f"{_name(minor)} has not reached its first release candidate: its leg "
            "is forgiven until then, with continue-on-error naming its version"
        )


def test_a_release_in_development_runs_forgiven() -> None:
    lanes = _matrix()
    for minor in sorted(_in_development(TODAY)):
        assert minor in lanes, (
            f"{_name(minor)} is past its first alpha: add it to the python job in "
            "ci.yml, forgiven with continue-on-error naming its version"
        )
        assert not lanes[minor], (
            f"{_name(minor)} is a prerelease: its leg must be forgiven, so a break "
            "in an alpha is news rather than a red merge"
        )


def test_the_classifiers_name_the_supported_releases() -> None:
    assert _classifiers() == _supported(TODAY), (
        "the 'Programming Language :: Python :: 3.N' classifiers in pyproject.toml "
        f"name {sorted(_classifiers())}; the supported releases are "
        f"{sorted(_supported(TODAY))}"
    )


@pytest.mark.parametrize(("where", "stated"), FLOORS.items(), ids=list(FLOORS))
def test_every_statement_of_the_floor_names_the_oldest_supported_release(
    where: str, stated: int
) -> None:
    floor = min(_supported(TODAY))
    assert stated == floor, f"{where} says 3.{stated}; the floor is {_name(floor)}"


@pytest.mark.parametrize(("where", "stated"), NEWEST.items(), ids=list(NEWEST))
def test_every_statement_of_the_newest_release_names_it(
    where: str, stated: int
) -> None:
    newest = max(_supported(TODAY))
    assert stated == newest, (
        f"{where} says 3.{stated}; the newest supported release is {_name(newest)}"
    )


def test_the_calendar_moves_the_floor_and_admits_the_next_release() -> None:
    """The claims above turn on the dates as the schedule PEPs give them.

    Held on fixed days, so a row entered with a transposed date or a month read
    as inclusive the wrong way fails here rather than on the day it matters.
    """
    assert _supported(_day("2026-10-31")) == {10, 11, 12, 13, 14, 15}
    assert _supported(_day("2026-11-01")) == {11, 12, 13, 14, 15}
    assert _in_development(_day("2026-10-12")) == set()
    assert _in_development(_day("2026-10-13")) == {16}
    assert _supported(_day("2027-07-27")) >= {16}
