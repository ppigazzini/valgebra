"""Metamorphic gate: two builds agree about values, and only grow more decisive.

A refactor that is not meant to change what valgebra means has no test of its
own. The suites check the tree against its specification, so a change that is
wrong in the *same* way on both sides of the refactor passes them both, and a
change that is right but unintended passes them too. What is missing is a
comparison against the tree as it was.

That comparison is metamorphic in the sense of Chen et al.: it relates the
outputs of two runs rather than checking one output against an expected value,
so it needs no oracle for what the answer should be. Two relations hold between
a reference build and the build under test:

* **Membership does not move.** For every schema and value in the corpus, the
  two builds accept and reject exactly the same pairs. Membership is the
  denotation, so a single flip is a semantic change -- which may be intended,
  but is never a refactor.
* **Decisions only widen.** A relation the reference decided must still be
  decided, and decided the same way. A decision is one of three answers --
  `subset`, `not_subset`, `undecided` -- for every ordered pair, and for every
  schema against `nothing`, which is its emptiness. A proof or a refutation
  that becomes `undecided` is an answer lost; one that becomes the other is a
  contradiction, since the procedure is sound and at most one of the two
  builds can be right. Gaining an answer is allowed, because the procedure is
  deliberately incomplete and a change to it may make it decide more.

The second relation has an escape the first does not, and this gate closes it: a
build that answered `subset` to everything would satisfy "only widens"
perfectly. So every new `subset` is checked against the corpus values. A claim
`A <= B` with a corpus value in `A` and not in `B` is a false proof, and so is an
emptiness proof for a schema a corpus value is a member of; the gate fails on
either rather than recording the widening. A relation that raises where the
reference answered is a failure too.

**The reference is the last release.** It is recorded from the published wheel
at every cut (`--record` run in an environment holding that wheel), so the
answers a push is held to are the ones a user of the release has. A change
meant to move one is carried by recording the reference from the tree, which a
reviewer reads in the reference's diff.

**Time and memory are held against the release in one job.** A time is a
property of the machine it was read on, so the reference keeps none. Where a
job holds both builds -- the published wheel in an environment of its own and
this tree's optimized wheel -- the first records the corpus with its times and
its peak memory (`--record-to`), and this build is held to that recording
(`--against`): a decision ten times slower than the release's and past 50 ms is
a slowdown, and so is a run whose peak memory is twice the release's and 64 MB
past it.

**The corpus holds the shapes that regressed.** Beside every ordered pair of the
schemas below, it asks a few named pairs as written, each a shape a change since
the release made slower or lost an answer on: a tuple and a record against the
corners that cover them, two discriminated unions, a list against a union of
length bounds, a guarded record against the complement of a record. A pair's two
schemas are asked of every value too, so a proof it gains is checked as any
other.

Usage:
    python scripts/metamorphic_gate.py            # compare against the reference
    python scripts/metamorphic_gate.py --record   # write the reference from this build
    python scripts/metamorphic_gate.py --record-to release.json  # with times, anywhere
    python scripts/metamorphic_gate.py --against release.json    # time and memory too

Three outcomes, three exit codes: **0** membership held, no decision was lost
and nothing was slower, **1** a value moved, a decision was lost, contradicted or
raised, a new proof has a counterexample, or a decision or the run was slower
than the release, **2** the gate **could not run** -- no built extension, no
readable reference, or a corpus the reference does not describe. A gate that
could not run has proven nothing and must not read as one that passed.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from _lib import EXIT_CANNOT_RUN, EXIT_FAIL, EXIT_OK, ROOT, arguments, git_output

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

REFERENCE_FILE = ROOT / "scripts" / "metamorphic_reference.json"


class Recording(TypedDict):
    """One build's answers to the whole corpus.

    `builds` records which spellings the frontend accepted at all, so a
    refusal that arrives or departs is visible without reading it out of the
    membership rows it silently removed.
    """

    commit: str
    version: str
    builds: dict[str, bool]
    membership: dict[str, str]
    decisions: dict[str, str]
    # Microseconds a decision took, and the run's peak memory in megabytes:
    # kept by `--record-to` alone, since a time belongs to its machine.
    times: NotRequired[dict[str, int]]
    peak_mb: NotRequired[int]


# A decision as the recording spells it: the three answers of `relation_to`,
# and `x` for a relation that raised.
ANSWERS = {"subset": "s", "not_subset": "n", "undecided": "u"}
RAISED = "x"
DECIDED = frozenset("sn")

# A decision is slower than the release's where it takes ten times as long and
# past 50 ms: the ratio is what a change moves, and the floor keeps a decision
# the runner's noise dominates out of it.
SLOWER_RATIO = 10
SLOWER_FLOOR_US = 50_000

# A run's memory is past the release's where its peak is twice the release's
# and 64 MB more: the corpus runs in tens of megabytes, so a ratio alone would
# read the allocator's rounding.
MEMORY_RATIO = 2
MEMORY_FLOOR_MB = 64


# How many failing pairs to print before stopping. A refactor that moves
# membership usually moves it for a whole class of values at once, and a
# thousand lines of the same flip says no more than a dozen.
REPORTED = 12


def _schemas() -> tuple[dict[str, object], dict[str, tuple[object, object]]]:
    """Build the schema corpus and the named pairs, spelled as a caller spells them.

    Built from the frontend's own syntax rather than from the IR, so the gate
    compares what a user can reach. A schema the frontend refuses is part of the
    corpus too -- the refusal is a verdict, and a refactor that starts or stops
    refusing has moved the boundary.
    """
    from types import GenericAlias  # noqa: PLC0415
    from typing import Annotated, Literal, TypedDict  # noqa: PLC0415

    from valgebra import (  # noqa: PLC0415
        Regex,
        Validator,
        anything,
        complement,
        intersection,
        nothing,
        recursive,
        union,
    )

    class Point(TypedDict):
        x: int
        y: int

    # `list[ref]` with a runtime schema inside it is a subscription a type
    # checker reads as a type expression and refuses. It is a value here, built
    # the way the interpreter builds `list[int]`, so it is spelled as one.
    def listed(element: object) -> object:
        return GenericAlias(list, (element,))

    def keyed(value: object) -> object:
        return GenericAlias(dict, (str, value))

    json_value = recursive(
        lambda ref: union(None, bool, int, float, str, listed(ref), keyed(ref))
    )
    json_object = keyed(json_value)

    def guarded(field: object) -> object:
        """Guard a record to objects, as a JSON Schema `properties` is written.

        Any value that is not an object passes the guard; an object must carry a
        `tag` of `field` and a `tag` of any JSON value.
        """
        return intersection(
            json_object,
            union(complement(json_object), Validator({"tag": field}).open()),
            union(complement(json_object), Validator({"tag": json_value}).open()),
        )

    guarded_4 = guarded(guarded(guarded(guarded(int))))
    integer = intersection(
        union(int, Annotated[float, MultipleOf(1)]), complement(bool)
    )
    integer_tree = recursive(lambda ref: union({"a": integer}, str, keyed(ref)))
    nested_closed = Validator(
        {
            "a": Validator(
                {"a": Validator({"tag": Validator({"a": json_value}).open()}).open()}
            ).close()
        }
    ).close()

    scalars: dict[str, object] = {
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "bytes": bytes,
        "none": None,
        "anything": anything,
        "nothing": nothing,
    }
    lattice: dict[str, object] = {
        "int|str": union(int, str),
        "int|bool": union(int, bool),
        "int&str": intersection(int, str),
        "~int": complement(int),
        "~~int": complement(complement(int)),
        "~int&~str": intersection(complement(int), complement(str)),
        "(int|str)&~str": intersection(union(int, str), complement(str)),
        "int|~int": union(int, complement(int)),
        "int&~int": intersection(int, complement(int)),
    }
    literals: dict[str, object] = {
        "lit-1": Literal[1],
        "lit-true": Literal[True],
        "lit-1-2-3": Literal[1, 2, 3],
        "lit-a": Literal["a"],
        "lit-none": Literal[None],  # noqa: PYI061 -- the spelling is the subject
        "lit-1|lit-2": union(Literal[1], Literal[2]),
    }
    containers: dict[str, object] = {
        "list[int]": list[int],
        "list[list[int]]": list[list[int]],
        "list[int|str]": listed(union(int, str)),
        "list-bare": list,
        "set[int]": set[int],
        "frozenset[str]": frozenset[str],
        "tuple[int,str]": tuple[int, str],
        "tuple[int,...]": tuple[int, ...],
        "tuple-empty": tuple[()],
        "dict[str,int]": dict[str, int],
        "dict[str,list[int]]": dict[str, list[int]],
    }
    records: dict[str, object] = {
        "typeddict-point": Point,
        "record-xy": {"x": int, "y": int},
        "record-x": {"x": int},
        "record-x-str": {"x": str},
    }
    refinements: dict[str, object] = {
        "int-ge-0": Annotated[int, Ge(0)],
        "int-ge-0-le-9": Annotated[int, Ge(0), Le(9)],
        "int-ge-5-le-1": Annotated[int, Ge(5), Le(1)],
        "str-regex-a": Annotated[str, Regex("a+")],
        "list-minlen-2": Annotated[list[int], MinLen(2)],
    }
    recursive_schemas: dict[str, object] = {
        "json": json_value,
        "tree": recursive(lambda ref: union(int, listed(ref))),
    }
    # The shapes a translation from JSON Schema writes, which the small schemas
    # above never compose: records guarded to objects and nested four deep, a
    # recursive record met with a nested closed one, a meet of two unions, a
    # literal set met with a complement, two open records met. A relation
    # between such shapes is answered by the rules that read a meet, and a
    # change to one of them moves answers here that nothing above can reach.
    composites: dict[str, object] = {
        "guarded-1": guarded(int),
        "guarded-4": guarded_4,
        "guarded-4|str": union(guarded_4, str),
        "integer-tree": integer_tree,
        "nested-closed": nested_closed,
        "integer-tree&nested-closed": intersection(integer_tree, nested_closed),
        "(int|str)&(str|none)": intersection(union(int, str), union(str, None)),
        "lit-a-1&~str": intersection(Literal["a", 1], complement(str)),
        "open-a&open-b": intersection(
            Validator({"a": int}).open(), Validator({"b": str}).open()
        ),
    }
    schemas = {
        **scalars,
        **lattice,
        **literals,
        **containers,
        **records,
        **refinements,
        **recursive_schemas,
        **composites,
    }
    return schemas, _pairs(json_value, guarded)


def _pairs(
    json_value: object, guarded: Callable[[object], object]
) -> dict[str, tuple[object, object]]:
    """Build the named pairs: the shapes a change since the release regressed on.

    Each is asked as written, `left <= right`, rather than against every
    schema, since the shapes are wide and the question is the one that moved.
    """
    from types import GenericAlias  # noqa: PLC0415
    from typing import Annotated, Literal  # noqa: PLC0415

    from valgebra import Validator, complement, intersection, union  # noqa: PLC0415

    # A tuple of `int | str` at each position and the corners that cover it,
    # which the rules split a position at a time: six decide, eight decline.
    def tuple_corners(width: int) -> tuple[object, object]:
        corners = union(
            *(
                GenericAlias(
                    tuple, tuple(int if mask >> i & 1 else str for i in range(width))
                )
                for mask in range(1 << width)
            )
        )
        return GenericAlias(tuple, tuple([int | str] * width)), corners

    # The same over the fields of a record, split a field at a time.
    def record_corners(width: int) -> tuple[object, object]:
        names = [f"f{i}" for i in range(width)]
        corners = union(
            *(
                {name: (int, str)[mask >> i & 1] for i, name in enumerate(names)}
                for mask in range(1 << width)
            )
        )
        return dict.fromkeys(names, int | str), corners

    # Records told apart by a tag, against the same records with a wider field:
    # read a record at a time, below one way and refuted the other. Each tag is
    # a literal built from a value, so it is built as the interpreter builds one.
    def tagged(width: int, field: object = int) -> object:
        tags = (Literal.__getitem__(f"t{i}") for i in range(width))
        return union(*({"tag": tag, "v": field} for tag in tags))

    def listed(*constraints: object) -> object:
        return Annotated[list, *constraints]

    # A list of at least 255 items below six bounds on its length, which the
    # rules refute below each alone and which hold it between them.
    length_bounds = union(
        complement(listed(MinLen(1))),
        listed(MaxLen(2048)),
        listed(MinLen(511)),
        intersection(listed(MinLen(64)), listed(MinLen(255)), listed(MaxLen(1000))),
        intersection(
            union(listed(MinLen(2)), listed(MaxLen(255))), listed(MinLen(257))
        ),
        listed(MaxLen(254)),
    )
    # A record whose field is guarded three deep, below the complement of a
    # record its field never meets: decided once its narrowed meets are read
    # once a query, and lost while each goal read them again.
    number = intersection(union(int, float), complement(bool))
    count = intersection(number, union(Annotated[int, Ge(0)], Annotated[float, Ge(0)]))
    guarded_record = Validator({"a": guarded(guarded(guarded(count)))}).close()
    other_record = Validator(
        {
            "a": intersection(json_value, Literal[True, "a"]),
            "b?": Validator({"a": json_value}).open(),
            str: Validator({"a": GenericAlias(tuple, (json_value, ...))}).close(),
        }
    )
    return {
        "tuple-corners-6": tuple_corners(6),
        "tuple-corners-8": tuple_corners(8),
        "record-corners-6": record_corners(6),
        "record-corners-8": record_corners(8),
        "tagged-512": (tagged(512), tagged(512, union(int, str))),
        "tagged-512-back": (tagged(512, union(int, str)), tagged(512)),
        "tagged-2048": (tagged(2048), tagged(2048, union(int, str))),
        "list-length-bounds": (listed(MinLen(255)), length_bounds),
        "guarded-record": (guarded_record, complement(other_record)),
    }


def _values() -> dict[str, object]:
    """Build the value corpus: one value per membership question worth asking.

    Every scalar region is represented, and so is each way a container can miss:
    the wrong element type, the wrong length, the wrong key type. The values are
    written out rather than generated, because a generated corpus that differs
    between two runs compares nothing.
    """
    return {
        "true": True,
        "false": False,
        "0": 0,
        "1": 1,
        "5": 5,
        "-1": -1,
        "0.0": 0.0,
        "1.5": 1.5,
        "nan": float("nan"),
        "inf": float("inf"),
        "empty-str": "",
        "a": "a",
        "aaa": "aaa",
        "b": "b",
        "bytes": b"a",
        "none": None,
        "empty-list": [],
        "list-1": [1],
        "list-1-2": [1, 2],
        "list-a": ["a"],
        "list-mixed": [1, "a"],
        "list-nested": [[1], [2]],
        "empty-set": set(),
        "set-1": {1},
        "frozenset-a": frozenset({"a"}),
        "empty-tuple": (),
        "tuple-1-a": (1, "a"),
        "tuple-1-2": (1, 2),
        "empty-dict": {},
        "dict-xy": {"x": 1, "y": 2},
        "dict-x": {"x": 1},
        "dict-x-str": {"x": "a"},
        "dict-xyz": {"x": 1, "y": 2, "z": 3},
        "dict-int-key": {1: 1},
        "json-object": {"a": [1, None, {"b": "c"}]},
        "dict-tag": {"tag": 1},
        "dict-tag-4": {"tag": {"tag": {"tag": {"tag": 1}}}},
        "dict-a-1": {"a": 1},
        "dict-a-nested": {"a": {"a": {"tag": {"a": 1}}}},
        "dict-a-b": {"a": 1, "b": "a"},
    }


class Ge:
    """A lower bound, spelled the way `annotated_types` spells it.

    Written here rather than imported so the corpus depends on nothing but
    valgebra: a gate that cannot run without an optional dependency reports
    "could not run" on the lanes that lack it, which is the outcome that proves
    the least. Each marker carries that package's module, because a constraint
    is read off its vocabulary and no other.
    """

    __module__ = "annotated_types"

    def __init__(self, ge: int) -> None:
        self.ge = ge


class Le:
    __module__ = "annotated_types"

    def __init__(self, le: int) -> None:
        self.le = le


class MinLen:
    __module__ = "annotated_types"

    def __init__(self, min_length: int) -> None:
        self.min_length = min_length


class MaxLen:
    __module__ = "annotated_types"

    def __init__(self, max_length: int) -> None:
        self.max_length = max_length


class MultipleOf:
    __module__ = "annotated_types"

    def __init__(self, multiple_of: int) -> None:
        self.multiple_of = multiple_of


def _membership(schemas: dict[str, Any], values: dict[str, object]) -> dict[str, str]:
    """Every schema against every value: `y`, `n`, or the error the walk raised.

    A raised error is a verdict like any other. `recursion_limit` and
    `mutated_during_validation` are answers this library gives on purpose, and a
    refactor that turns one into an accept has moved membership as surely as one
    that flips a `False`.
    """
    verdicts = {}
    for schema_name, validator in schemas.items():
        if validator is None:
            continue
        for value_name, value in values.items():
            try:
                answer = "y" if validator.is_valid(value) else "n"
            except Exception as err:  # noqa: BLE001 -- the class is the verdict
                answer = f"!{type(err).__name__}"
            verdicts[f"{schema_name} @ {value_name}"] = answer
    return verdicts


def _relation(left: Any, right: Any, key: str, times: dict[str, int]) -> str:
    """Ask `left <= right`, keeping how long it took under `key`.

    A relation that raises is a verdict, `x`: the procedure answers every pair,
    so a raise is an answer the release did not give.
    """
    start = time.perf_counter_ns()
    try:
        answer = ANSWERS[left.relation_to(right)]
    except Exception:  # noqa: BLE001 -- any raise is the same verdict here
        answer = RAISED
    times[key] = (time.perf_counter_ns() - start) // 1000
    return answer


def _decisions(schemas: dict[str, Any], times: dict[str, int]) -> dict[str, str]:
    """Every ordered pair under subtyping, and every schema under emptiness.

    Each in the three answers of `relation_to`, and emptiness as the schema
    against `nothing`: a boolean reads a refutation and a decline as one `False`,
    so a refutation lost to a decline would not move it.
    """
    from valgebra import nothing  # noqa: PLC0415

    verdicts = {}
    for name, validator in schemas.items():
        if validator is None:
            continue
        key = f"empty {name}"
        verdicts[key] = _relation(validator, nothing, key, times)
        for other_name, other in schemas.items():
            if other is None:
                continue
            key = f"{name} <= {other_name}"
            verdicts[key] = _relation(validator, other, key, times)
    return verdicts


def _paired(sides: dict[str, Any], times: dict[str, int]) -> dict[str, str]:
    """Each named pair as written, `name.left <= name.right`."""
    verdicts = {}
    for name in sorted({side.rpartition(".")[0] for side in sides}):
        left, right = sides[f"{name}.left"], sides[f"{name}.right"]
        if left is None or right is None:
            continue
        key = f"{name}.left <= {name}.right"
        verdicts[key] = _relation(left, right, key, times)
    return verdicts


def _build(specs: dict[str, object]) -> dict[str, Any]:
    """Build every corpus schema, keeping a refusal as the absence of one.

    A `None` here means the frontend declined the spelling. The membership and
    decision passes skip it, and the recorded build verdicts carry the refusal,
    so a spelling that starts or stops building is caught by the comparison
    rather than by a crash inside it.
    """
    from valgebra import Validator  # noqa: PLC0415

    def build(spec: object) -> Any:
        try:
            return Validator(spec)
        except Exception:  # noqa: BLE001 -- any refusal is the same verdict here
            return None

    return {name: build(spec) for name, spec in specs.items()}


def record(*, timed: bool) -> Recording | None:
    """Run the whole corpus through this build, or report why it could not.

    `timed` keeps each decision's time and the run's peak memory, which only a
    recording compared in the job that made it can read.
    """
    try:
        specs, pairs = _schemas()
    except ImportError as err:
        print(f"metamorphic_gate: no built extension to measure: {err}")
        return None
    built = _build(specs)
    sides = _build(
        {
            f"{name}.{side}": spec
            for name, (left, right) in pairs.items()
            for side, spec in (("left", left), ("right", right))
        }
    )
    times: dict[str, int] = {}
    decisions = _decisions(built, times) | _paired(sides, times)
    version = importlib.metadata.version("valgebra")
    recording = Recording(
        commit=_commit(version),
        version=version,
        builds={name: schema is not None for name, schema in (built | sides).items()},
        membership=_membership(built | sides, _values()),
        decisions=decisions,
    )
    if timed:
        recording["times"] = times
        peak = _peak_mb()
        if peak is not None:
            recording["peak_mb"] = peak
    return recording


def _commit(version: str) -> str:
    """Name the commit this recording describes, or `unknown` where none is found.

    A build of this tree is the checkout's `HEAD`; an installed release is the
    commit its version's tag names, which a clone without tags cannot say.
    """
    import valgebra  # noqa: PLC0415

    if ROOT in Path(valgebra.__file__).resolve().parents:
        return git_output("rev-parse", "HEAD", cwd=ROOT) or "unknown"
    return git_output("rev-parse", f"v{version}^{{commit}}", cwd=ROOT) or "unknown"


def _peak_mb() -> int | None:
    """Read the most memory this process has held, in megabytes, where it can."""
    try:
        import resource  # noqa: PLC0415 -- not on every platform
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes and macOS bytes.
    return peak // (1 << 20) if sys.platform == "darwin" else peak // (1 << 10)


def _keys_agree(name: str, reference: dict[str, str], measured: dict[str, str]) -> bool:
    """Refuse a comparison whose two sides ask different questions.

    A corpus that grew or shrank makes every "held" below meaningless: the
    relations are over the pairs both recordings carry, and a pair only one
    carries was never compared. This is the "could not run" case, not a failure
    -- the tree may be fine and the reference merely stale.
    """
    only_reference = sorted(set(reference) - set(measured))
    only_measured = sorted(set(measured) - set(reference))
    if not only_reference and not only_measured:
        return True
    print(f"metamorphic_gate: the {name} corpus does not match the reference")
    for key in only_reference[:REPORTED]:
        print(f"  reference only: {key}")
    for key in only_measured[:REPORTED]:
        print(f"  this build only: {key}")
    print("  re-record the reference (--record) if the corpus change is intended")
    return False


def moved_membership(reference: dict[str, str], measured: dict[str, str]) -> list[str]:
    """List the pairs whose verdict changed.

    Public because a gate that cannot be driven to fail from a test is not
    evidence, and this is the relation the gate exists to hold.
    """
    return [
        f"{key}: {reference[key]} -> {measured[key]}"
        for key in sorted(reference)
        if reference[key] != measured[key]
    ]


def lost_decisions(reference: dict[str, str], measured: dict[str, str]) -> list[str]:
    """List the decisions the reference made and this build declines.

    Soundness makes this one-directional. A proof or a refutation that became
    `undecided` lost an answer the reference had, which no refactor should do;
    an `undecided` that became an answer is the procedure deciding more, which a
    change to the procedure does on purpose.
    """
    return [
        f"{key}: decided by the reference, not by this build"
        for key in sorted(reference)
        if reference[key] in DECIDED and measured[key] == "u"
    ]


def contradicted_decisions(
    reference: dict[str, str], measured: dict[str, str]
) -> list[str]:
    """List the decisions the two builds make the opposite way.

    Both builds claim to be sound, so a `subset` against a `not_subset` means
    one of them is wrong, whichever direction the change went.
    """
    return [
        f"{key}: {reference[key]} on the reference, {measured[key]} on this build"
        for key in sorted(reference)
        if reference[key] in DECIDED
        and measured[key] in DECIDED
        and reference[key] != measured[key]
    ]


def raised_decisions(reference: dict[str, str], measured: dict[str, str]) -> list[str]:
    """List the relations this build raises on where the reference answered."""
    return [
        f"{key}: {reference[key]} on the reference, raised on this build"
        for key in sorted(reference)
        if measured[key] == RAISED and reference[key] != RAISED
    ]


def slower_decisions(
    reference_times: dict[str, int], measured_times: dict[str, int]
) -> list[str]:
    """List the decisions ten times slower than the release's and past 50 ms."""
    return [
        f"{key}: {measured_times[key]} us, against {reference_times[key]} us"
        for key in sorted(reference_times)
        if key in measured_times
        and measured_times[key] > SLOWER_RATIO * reference_times[key]
        and measured_times[key] > SLOWER_FLOOR_US
    ]


def heavier_run(reference_mb: int | None, measured_mb: int | None) -> list[str]:
    """Name a run whose peak memory is twice the release's and 64 MB past it."""
    if reference_mb is None or measured_mb is None:
        return []
    if (
        measured_mb > MEMORY_RATIO * reference_mb
        and measured_mb - reference_mb > MEMORY_FLOOR_MB
    ):
        return [f"peak memory {measured_mb} MB, against {reference_mb} MB"]
    return []


def _subtype_claim(key: str) -> tuple[str, str] | None:
    """Read `A <= B` back out of a decision key, or `None` for an emptiness row."""
    left, separator, right = key.partition(" <= ")
    return (left, right) if separator else None


def _emptiness_claim(key: str) -> str | None:
    """Read the schema an emptiness row is about, or `None` for a subtyping row."""
    prefix, separator, name = key.partition("empty ")
    return name if separator and not prefix else None


def unwitnessed_widenings(
    reference: dict[str, str],
    measured: dict[str, str],
    membership: dict[str, str],
    values: dict[str, object],
) -> list[str]:
    """List the new `True`s a corpus value contradicts.

    "Only widens" is satisfied perfectly by a build that proves everything, so
    the widenings are checked rather than counted. `A <= B` is refuted by a
    value in `A` that is not in `B`, and the membership pass has already asked
    every one of those questions -- so the search is a lookup, and it uses this
    build's own answers rather than a second opinion about them.
    """
    refuted = []
    for key in sorted(reference):
        if reference[key] == "s" or measured[key] != "s":
            continue
        empty = _emptiness_claim(key)
        if empty is not None:
            member = next(
                (v for v in values if membership.get(f"{empty} @ {v}") == "y"), None
            )
            if member is not None:
                refuted.append(f"{key}: {member} is in {empty}")
            continue
        claim = _subtype_claim(key)
        if claim is None:
            continue
        left, right = claim
        for value_name in values:
            inside = membership.get(f"{left} @ {value_name}")
            outside = membership.get(f"{right} @ {value_name}")
            if inside == "y" and outside == "n":
                refuted.append(f"{key}: {value_name} is in {left} and not in {right}")
                break
    return refuted


def _report(title: str, failures: list[str]) -> None:
    print(f"{title}: {len(failures)}")
    for line in failures[:REPORTED]:
        print(f"  {line}")
    if len(failures) > REPORTED:
        print(f"  ... and {len(failures) - REPORTED} more")


def _arguments(argv: list[str]) -> argparse.Namespace:
    """Read the command line before building anything; an unknown flag exits 2."""
    parser = arguments(__doc__)
    parser.add_argument(
        "--record",
        action="store_true",
        help="write this build's answers as the reference, from the release's wheel",
    )
    parser.add_argument(
        "--record-to",
        type=Path,
        metavar="PATH",
        help="write this build's answers, times and peak memory to PATH",
    )
    parser.add_argument(
        "--against",
        type=Path,
        metavar="PATH",
        help="hold this build's times and peak memory to the recording at PATH",
    )
    return parser.parse_args(argv)


def _write(args: argparse.Namespace, measured: Recording) -> bool:
    """Write the recording a flag asks for; whether one did."""
    if args.record_to is not None:
        args.record_to.write_text(
            json.dumps(measured, sort_keys=True), encoding="utf-8"
        )
        print(f"metamorphic_gate: valgebra {measured['version']} recorded with times")
        return True
    if args.record:
        REFERENCE_FILE.write_text(
            json.dumps(measured, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"metamorphic_gate: recorded at {measured['commit']}")
        return True
    return False


def _answers(reference: dict[str, Any], measured: Recording) -> list[str] | None:
    """Hold the answers to the reference's; the failures, or `None` if it cannot."""
    reference_membership = reference["membership"]
    reference_decisions = reference["decisions"]
    membership = measured["membership"]
    decisions = measured["decisions"]
    agreed = _keys_agree("membership", reference_membership, membership)
    agreed = _keys_agree("decision", reference_decisions, decisions) and agreed
    if not agreed:
        return None

    moved = moved_membership(reference_membership, membership)
    lost = lost_decisions(reference_decisions, decisions)
    contradicted = contradicted_decisions(reference_decisions, decisions)
    raised = raised_decisions(reference_decisions, decisions)
    refuted = unwitnessed_widenings(
        reference_decisions, decisions, membership, _values()
    )
    widened = sum(
        1
        for key in reference_decisions
        if reference_decisions[key] == "u" and decisions[key] in DECIDED
    )

    print(f"reference: {reference['commit']} (valgebra {reference.get('version')})")
    print(f"corpus:    {len(membership)} value pairs, {len(decisions)} decisions")
    _report("membership moved", moved)
    _report("decisions lost", lost)
    _report("decisions contradicted", contradicted)
    _report("decisions raised", raised)
    _report("widenings refuted by a value", refuted)
    print(f"widenings: {widened}")
    return moved + lost + contradicted + raised + refuted


def _against_the_release(path: Path, measured: Recording) -> list[str] | None:
    """Hold the times and peak memory to the release's; `None` if unreadable."""
    try:
        release = json.loads(path.read_text(encoding="utf-8"))
        release_times = release["times"]
    except (OSError, ValueError, KeyError) as err:
        print(f"metamorphic_gate: cannot read the release's times: {err}")
        return None
    slower = slower_decisions(release_times, measured.get("times", {}))
    slower += heavier_run(release.get("peak_mb"), measured.get("peak_mb"))
    print(f"release:   valgebra {release.get('version')}, {release.get('peak_mb')} MB")
    print(f"this:      {measured.get('peak_mb')} MB at its peak")
    _report("slower than the release", slower)
    return slower


def _compare(args: argparse.Namespace, measured: Recording) -> list[str] | None:
    """Every failure against the reference and the release; `None` if it cannot."""
    try:
        reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
        failures = _answers(reference, measured)
    except (OSError, ValueError, KeyError) as err:
        print(f"metamorphic_gate: cannot read the reference: {err}")
        return None
    if failures is None or args.against is None:
        return failures
    slower = _against_the_release(args.against, measured)
    return None if slower is None else failures + slower


def main(argv: list[str]) -> int:
    args = _arguments(argv)
    measured = record(timed=args.record_to is not None or args.against is not None)
    if measured is None:
        return EXIT_CANNOT_RUN
    if _write(args, measured):
        return EXIT_OK
    failures = _compare(args, measured)
    if failures is None:
        return EXIT_CANNOT_RUN
    if failures:
        return EXIT_FAIL
    print("OK: membership held, no decision was lost and nothing was slower.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
