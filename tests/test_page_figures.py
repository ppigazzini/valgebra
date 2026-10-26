"""The pages' figures are read off a whole run, or not reported at all.

`scripts/page_figures.py` takes every wall-clock figure the performance and JSON
pages state, and its report is what a reader copies into them. Two things make
the report worth copying: each figure is computed as the page says it is -- a
median and a half-range of ten runs, a saving per pair of wheels -- and a run
cut short reports nothing rather than a figure over fewer readings than the page
claims.

Both are driven here from readings written by hand -- no wheels, no timer -- so
each is exercised rather than assumed.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from _tree import ROOT, load_script

if TYPE_CHECKING:
    from pathlib import Path

# The repository checks are not the product suite: this file reads the tree and
# the scripts, neither of which ships in a wheel.
pytestmark = pytest.mark.repository

figures = load_script("page_figures")

SHAPES = (
    "scalar",
    "large_array",
    "wide_record",
    "deep_nesting",
    "json_document",
    "build",
    "error_report",
)

CRITERION_OUTPUT = """\
reindexed_record_width64
                        time:   [1.1745 µs 1.1786 µs 1.1825 µs]
with_records_open_depth32
                        time:   [4.3191 µs 4.3335 µs 4.3537 µs]
"""


def _gate(vg: float = 10.0, ratio: float = 0.2) -> dict[str, Any]:
    shapes = {shape: {"vg": vg, "pyd": vg / ratio, "ratio": ratio} for shape in SHAPES}
    return {"python": "3.14.7", **shapes}


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _bench(path: Path, cells: dict[tuple[str, str], float]) -> None:
    rows = [
        {"group": group, "params": {"lib": lib}, "stats": {"median": seconds}}
        for (group, lib), seconds in cells.items()
    ]
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def _write_run(results: Path) -> None:
    """Write every file a whole run leaves, each holding the readings a run takes."""
    results.mkdir()
    (results / "tree.txt").write_text("0123abcd\n", encoding="utf-8")
    for label, runs in figures.GATE_RUNS.items():
        _jsonl(results / f"gate-{label}.jsonl", [_gate() for _ in range(runs)])
    pairs = figures.MEASURE["pairs"]
    for name, first, second in (
        ("pgo-pair", "plain", "pgo"),
        ("pool-pair", "pool", "pgo"),
    ):
        _jsonl(results / f"{name}-{first}.jsonl", [_gate(vg=100.0)] * pairs)
        _jsonl(results / f"{name}-{second}.jsonl", [_gate(vg=80.0)] * pairs)
    floor = {"in": 16.0, "is_valid": 21.0, "anything": 19.0, "isinstance": 12.0}
    _jsonl(results / "floor.jsonl", [{**floor, "empty_call": 15.0}] * 6)
    for label in ("3.14", "3.12"):
        reading = {"build_50_refined_us": 16.0, "deep_report_us": 60.0}
        _jsonl(results / f"techniques-{label}.jsonl", [reading] * 2)
    runs = [(which, n) for which in ("first", "second") for n in range(1, 6)]
    for index, (which, n) in enumerate(runs, start=1):
        sides = ("valgebra", "pydantic", "jsonschema")
        compared = {
            (shape, lib): index * 1e-6 for shape in figures.COMPARED for lib in sides
        }
        _bench(results / f"compare-{which}-{n}.json", compared)
        parsed = {}
        for shape in figures.JSON_SHAPES:
            parsed[shape, "valgebra-json"] = 2e-6
            parsed[shape, "valgebra-loads"] = 9e-6
            parsed[shape, "pydantic-json"] = 4e-6
        _bench(results / f"json-{which}-{n}.json", parsed)
    _jsonl(results / "json-split.json", [{"checked_us": 59.0, "parse_only_us": 46.0}])
    _jsonl(results / "jiter.json", [{"tree_us": 61.0, "pull_us": 16.0}])
    (results / "criterion.txt").write_text(CRITERION_OUTPUT, encoding="utf-8")


def test_a_whole_run_reports_every_section_it_reads(tmp_path: Path) -> None:
    _write_run(tmp_path / "results")
    lines = figures.report(tmp_path / "results")
    text = "\n".join(lines)
    for heading in (
        "docs/11-performance.md, Baseline matrix",
        "docs/11-performance.md, The cheapest door",
        "docs/11-performance.md, Results",
        "docs/11-performance.md, One of those margins",
        "docs/11-performance.md, The closest races",
        "docs/11-performance.md, Core micro-benchmarks",
        "docs/11-performance.md, Crossing into Rust",
        "docs/11-performance.md, Building a validator",
        "docs/07-json.md, Performance: median of ten runs",
        "docs/07-json.md, Performance: where the document's time goes",
    ):
        assert heading in text, heading
    assert "the walk 13.0 us" in text
    assert "4.50x" in text, "json.loads + is_valid over is_valid_json, 9 over 2"


def test_a_cell_is_the_median_of_its_runs_and_their_half_range(tmp_path: Path) -> None:
    # Ten runs reading 1 to 10 microseconds: the median is 5.5 and the runs part
    # by 9, which the page states as a half-range of 4.5.
    _write_run(tmp_path / "results")
    cells = figures.bench_medians(tmp_path / "results", "compare")
    runs = cells["large_array", "valgebra"]
    assert sorted(runs) == pytest.approx([float(n) for n in range(1, 11)])
    assert figures.cell(runs) == "5.5 +/- 4.5 us"


def test_a_profiled_wheel_taking_less_time_reads_as_a_saving(tmp_path: Path) -> None:
    _write_run(tmp_path / "results")
    plain = figures.read_lines(tmp_path / "results" / "pgo-pair-plain.jsonl")
    profiled = figures.read_lines(tmp_path / "results" / "pgo-pair-pgo.jsonl")
    assert figures.gains(plain, profiled, "scalar") == pytest.approx([20.0] * 6)
    report = "\n".join(figures.report_profile(tmp_path / "results"))
    assert re.search(r"scalar\s+20\.0%", report), report


@pytest.mark.parametrize(
    "cut",
    ["gate-3.12.jsonl", "compare-second-5.json", "json-first-1.json", "jiter.json"],
)
def test_a_run_cut_short_reports_nothing(tmp_path: Path, cut: str) -> None:
    """A figure over fewer readings than the page states is not that figure."""
    results = tmp_path / "results"
    _write_run(results)
    if cut.startswith("gate"):
        kept = (results / cut).read_text(encoding="utf-8").splitlines()[1:]
        (results / cut).write_text("\n".join(kept) + "\n", encoding="utf-8")
    else:
        (results / cut).unlink()
    with pytest.raises(SystemExit) as refused:
        figures.report(results)
    assert refused.value.code == 2


def test_an_estimate_belongs_to_the_last_name_criterion_printed() -> None:
    # A name too long for its column takes a line of its own, and its estimate
    # the next: read line by line alone, it would have none.
    assert figures.criterion_medians(CRITERION_OUTPUT) == {
        "reindexed_record_width64": "1.1786 µs",
        "with_records_open_depth32": "4.3335 µs",
    }


def test_the_probe_builds_the_jiter_the_lock_pins() -> None:
    # Read package by package, not by the script's pattern, so the two readings
    # of the lock file are independent.
    packages = (ROOT / "Cargo.lock").read_text(encoding="utf-8").split("[[package]]")
    (pinned,) = [
        line.split('"')[1]
        for package in packages
        if 'name = "jiter"' in package.splitlines()
        for line in package.splitlines()
        if line.startswith("version = ")
    ]
    assert figures.jiter_version(ROOT / "Cargo.lock") == pinned
    assert f'version = "={pinned}"' in figures.probe_manifest(pinned)


def test_a_build_drops_the_callers_compiler_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A caller's `RUSTFLAGS` replaces the flags `.cargo/config.toml` gives every
    # build, which compile the reference pool out of the wheel the release ships.
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=native")
    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", "-Copt-level=1")
    env = figures.build_environment()
    assert "RUSTFLAGS" not in env
    assert "CARGO_ENCODED_RUSTFLAGS" not in env


def test_the_page_states_the_repeats_the_script_takes() -> None:
    """The Method section's counts are the script's, and it names the script.

    The page states how many readings each figure is taken over, and the script
    takes them; a count changed in one and not the other makes the page describe
    a run nobody took.
    """
    page = (ROOT / "docs" / "11-performance.md").read_text(encoding="utf-8")
    words = {6: "six", 7: "seven", 10: "ten", 12: "twelve", 18: "eighteen"}
    gate = figures.GATE_RUNS
    stated = (
        (
            f"median over {words[gate['3.12']]} runs on 3.12,"
            f" {words[gate['3.14']]} on 3.14 and {words[gate['3.14t']]} on"
            " the free-threaded build"
        ),
        f"**median of {words[2 * figures.MEASURE['benchmarks']]} independent runs**",
        f"{words[figures.MEASURE['pairs']]} readings of each build interleaved",
        "uv run --no-sync python scripts/page_figures.py",
    )
    for phrase in stated:
        assert re.sub(r"\s+", " ", phrase) in re.sub(r"\s+", " ", page), phrase
