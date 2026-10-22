"""A cost reading is read from the files its instrument writes, and nothing else.

`scripts/perf_inspect.py` reads cachegrind's and callgrind's output files, and
`scripts/perf_shapes.py` runs inside an environment that holds the standard
library and valgebra alone. What makes a reading worth reading is driven here
from text written by hand -- no valgrind, no counter -- or from the shape table
itself, so each is exercised rather than assumed.
"""

from __future__ import annotations

import ast
import sys
from typing import TYPE_CHECKING, Any

import pytest

from _tree import ROOT, load_script

if TYPE_CHECKING:
    from collections.abc import Sequence

# The repository checks are not the product suite: this file reads the tree and
# the scripts, neither of which ships in a wheel.
pytestmark = pytest.mark.repository

inspect = load_script("perf_inspect")
shapes = load_script("perf_shapes")

CALLGRIND = """\
# callgrind format
events: Ir
ob=(1) /usr/lib/libpython3.14.so
fl=(1) ???
fn=(1) PyObject_Call
0x1000 0 5
ob=(2) /env/lib/python3.14/site-packages/valgebra/_valgebra.cpython-314.so
fl=(2) ???
fn=(2) member
0x2000 10 3
+4 10 2
* 11 1
cob=(1)
cfn=(1)
calls=1 0x1000
+8 12 100
+2 12 7
ob=(1)
fn=(3) PyDict_GetItem
0x1100 0 9
ob=(2)
fn=(2)
0x2010 10 4
"""


def test_a_compressed_callgrind_file_reads_as_instructions_per_address() -> None:
    # The line after `calls=` is the call's inclusive cost, the callee's; a
    # position is absolute, relative or the last; an object named once by its
    # number is that object again; and only the extension's instructions count.
    assert dict(inspect.callgrind_instructions(CALLGRIND, "_valgebra")) == {
        0x2000: 3,
        0x2004: 3,
        0x200E: 7,
        0x2010: 4,
    }


def test_two_trees_name_one_source_file_alike() -> None:
    def profile(tree: str) -> str:
        return (
            f"fl={tree}/crates/valgebra-py/src/check/walk.rs\n"
            "fn=_valgebra::check::walk::member::h0123456789abcdef\n"
            "501 7\n"
            "596 3\n"
        )

    first = profile("/x/target/perf-inspect/trees/0123456789ab")
    second = profile("/y/elsewhere/trees/ba9876543210")
    by_function = inspect.cachegrind_costs(first, by_line=False)
    assert by_function == inspect.cachegrind_costs(second, by_line=False)
    assert by_function == {
        ("crates/valgebra-py/src/check/walk.rs", "_valgebra::check::walk::member"): 10
    }
    by_line = inspect.cachegrind_costs(second, by_line=True)
    assert by_line[("crates/valgebra-py/src/check/walk.rs", "596")] == 3


@pytest.mark.parametrize(("rise", "status"), [(0.5, 0), (2.0, 1)])
def test_a_rise_past_the_tolerance_exits_1(
    monkeypatch: pytest.MonkeyPatch, rise: float, status: int
) -> None:
    # Each count is a start-up cost plus the calls' cost, so the difference of
    # two call counts is the per-call cost alone, and the second environment
    # rises by `rise` per cent of it.
    per_call = {"base": 1000.0, "head": 1000.0 * (1 + rise / 100)}

    def counted(command: Sequence[str], out: str = "/dev/null", pad: int = 0) -> int:
        del out, pad
        name, calls = command[0], int(command[-1])
        return int(5_000_000 + per_call[name] * calls)

    monkeypatch.setattr(inspect.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(inspect, "worker", lambda name: [name])
    monkeypatch.setattr(inspect, "cachegrind", counted)
    assert inspect.count(["base", "head"], ["deep"], tolerance=1.0) == status
    assert inspect.count(["base", "head"], ["deep"], tolerance=None) == 0


def test_a_pad_adds_one_variable_of_its_size_to_every_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A shape that allocates reads the heap its loop starts from, and the
    # environment's size moves that heap; a padded count differs from a plain
    # one in that size and in nothing else.
    handed: list[dict[str, str]] = []

    def run(command: Sequence[str], **kwargs: Any) -> Any:
        handed.append(kwargs["env"])
        irefs = 5_000_000 + 1000 * int(command[-1])
        return inspect.subprocess.CompletedProcess(command, 0, "", f"I refs: {irefs:,}")

    monkeypatch.setattr(inspect.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(inspect, "worker", lambda name: [name])
    monkeypatch.setattr(inspect.subprocess, "run", run)
    plain = inspect.perf_gate.workload_environment(inspect.os.environ)
    assert inspect.count(["base", "head"], ["deep"], tolerance=None) == 0
    assert handed == [plain] * 4
    handed.clear()
    assert inspect.count(["base", "head"], ["deep"], tolerance=None, pad=600) == 0
    assert len(handed) == 4
    for env in handed:
        added = [env[name] for name in env.keys() - plain.keys()]
        assert added == ["x" * 600]
        assert {name: env[name] for name in plain} == plain


def test_the_shapes_the_orchestrator_names_are_the_table() -> None:
    # The orchestrator reads the names without importing valgebra, which the
    # environment it runs in need not hold; the two readings must agree.
    assert inspect.shape_names() == list(shapes.SHAPES)


def test_every_shape_runs_on_this_build() -> None:
    for name, build in shapes.SHAPES.items():
        check, value = build()
        check(value)
        assert callable(check), name


def _imported(name: str) -> set[str]:
    tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


@pytest.mark.interpreter
def test_the_worker_imports_the_standard_library_and_valgebra_alone() -> None:
    # Anything more is a second thing two environments differ by, and an
    # environment the script makes holds nothing more. `_lib`, which every
    # script shares, is held to the standard library too.
    standard = set(sys.stdlib_module_names) | {"__future__"}
    assert _imported("perf_shapes.py") <= standard | {"valgebra", "_lib"}
    assert _imported("_lib.py") <= standard


def test_the_performance_page_shows_each_reading() -> None:
    page = (ROOT / "docs" / "11-performance.md").read_text(encoding="utf-8")
    for step in ("env", "count", "cycles", "symbols"):
        assert f"scripts/perf_inspect.py {step} " in page, step
