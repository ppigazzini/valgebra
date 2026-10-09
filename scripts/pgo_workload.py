"""Profile-guided-optimization training workload for the release PGO lane.

Run against a profile-instrumented build of the extension, this exercises the
validation hot paths over a broad, production-like spread of schema shapes so the
recorded profile generalizes rather than overfitting a single micro-benchmark:
scalars, closed and open records of several widths, homogeneous and
heterogeneous sequences, nested documents, literal and structural unions,
mappings, and the JSON path. Both passing and failing values are fed, since the
fast accept path and the rejecting (and aggregating explain) paths take
different branches.

**Both halves of the surface.** The membership walk is one; the relations
between two schemas are the other, and they share no code with it. A profile
that never asks one is a profile with no counts for it, so the comparison a
caller writes is laid out by guesswork in the shipped wheel.

It depends only on ``valgebra`` and the standard library (no test, comparison,
or annotation-metadata packages), so it runs in the minimal environment maturin
sets up for ``--pgo`` -- which is why the refinements it trains are spelled
with the native ``Regex`` and with a bound class of its own, carrying
``annotated_types``' module and read by its ``ge`` as ``annotated_types.Ge`` is.
Keep it quick: a few seconds is enough to accumulate representative branch
counts.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Annotated, Literal, NamedTuple, TypedDict

from valgebra import (
    Regex,
    ValidationError,
    Validator,
    complement,
    intersection,
    recursive,
    union,
)

# A bound validator method; `...` admits is_valid/validate/is_valid_json alike.
Check = Callable[..., object]

#: Every shape the comparison gate times and the instruction gate counts, and
#: the functions below whose calls take the same reading in the profiled build:
#: the entry point the shape calls, a schema of its form, a value of its kind.
#: The comparison gate's shapes are named as `scripts/perf_compare.json` names
#: them, the instruction gate's as `MODES` in `scripts/perf_gate.py` does.
#: `tests/test_pgo_training.py` holds the two tables to both gates and to the
#: functions `main` runs.
TRAINED: dict[str, tuple[str, ...]] = {
    "scalar": ("_scalars",),
    "large_array": ("_sequences",),
    "wide_record": ("_records",),
    "deep_nesting": ("_containers",),
    "build": ("_records",),
    "error_report": ("_records",),
    "decision": ("_relations",),
    "decision-refute": ("_relations",),
    "decision-repeat": ("_relations",),
    "decision-matrix": ("_relations",),
    "binding": ("_sequences",),
    "binding-boundary": ("_scalars",),
    "binding-record": ("_records",),
    "binding-build": ("_records",),
    "binding-explain": ("_records",),
    "binding-explain-accept": ("_records",),
    "binding-explain-list": ("_sequences",),
    "binding-open": ("_containers",),
    "binding-annotated": ("_containers", "_lists_with_readers"),
    "binding-object": ("_lists_with_readers", "_relations"),
    "binding-relation": ("_relations",),
    "binding-keys": ("_documents",),
    "binding-pattern": ("_lists_with_readers",),
    "binding-deep": ("_containers",),
    "binding-refined": ("_lists_with_readers",),
    "binding-mapping": ("_mappings",),
    "binding-nullable": ("_lists_with_readers",),
}

#: The JSON shapes' common reason: training any of them re-weights the profile
#: every timed shape is laid out by.
_JSON_UNTRAINED = (
    "A parsed JSON array of records is read element by element through the "
    "general loop of `json_array_matches` in "
    "`crates/valgebra-py/src/check/walk/sequence.rs`, and the calls here read "
    "JSON records and JSON arrays of one scalar kind, which a loop of their own "
    "takes. The wheel lays the general loop out by the inliner's guess, as the "
    "list scan of a nested list was before it was trained; training it "
    "re-weights the profile every timed shape is laid out by, so it is read on "
    "the comparison gate before and after rather than added here."
)

#: The shapes no call here takes the reading of, each with what it reads. The
#: shipped wheel lays each one's path out without counts.
UNTRAINED: dict[str, str] = {
    "json_document": _JSON_UNTRAINED,
    "binding-json": _JSON_UNTRAINED,
    "binding-json-reject": _JSON_UNTRAINED,
    "binding-json-union": (
        "A union of two record kinds over parsed JSON objects, each element "
        "asked of the second branch after failing the first: no call here "
        "validates a JSON document against a union. " + _JSON_UNTRAINED
    ),
    "binding-json-open": (
        "JSON records read through a key-type clause: no call here validates a "
        "JSON document against a record with a catch-all. " + _JSON_UNTRAINED
    ),
    "binding-json-deep": (
        "A recursive schema over a parsed document: no call here validates JSON "
        "against a reference. " + _JSON_UNTRAINED
    ),
    "binding-recursive": (
        "A value walked against a recursive schema, entering the reference and "
        "its trail per level: `_relations` builds a recursive schema and relates "
        "it, and no call here walks a value against one."
    ),
    "binding-set": (
        "A `set[str]` read through its own iterator: no call here validates a set."
    ),
    "binding-subclass": (
        "A named tuple read against `tuple[int, ...]`, through the length "
        "accessor a subclass inherits: the calls here read named tuples against "
        "their own class and plain tuples against a tuple schema."
    ),
    "binding-protocol": (
        "Compiling a protocol, whose members `typing` lists and the frontend "
        "classifies name by name: no call here builds a validator from a "
        "protocol."
    ),
    "core": (
        "The simplifier, the composition remap and the record transform, run "
        "directly in Rust: the calls here reach the simplifier through `union`, "
        "`intersection` and `complement`, and no call opens or closes a record."
    ),
}


class _Point(NamedTuple):
    x: int
    y: str


@dataclass
class _Pair:
    x: int
    y: str


class _AtLeast:
    """An order bound, carried as `annotated_types.Ge` carries one.

    It carries that package's module too, because a constraint is read off its
    vocabulary and no other: without it the bound is metadata the frontend
    ignores, and the profile trains no bound at all.
    """

    __module__ = "annotated_types"

    def __init__(self, ge: int) -> None:
        self.ge = ge


def _run(check: Check, samples: Sequence[object], rounds: int) -> None:
    for _ in range(rounds):
        for value in samples:
            check(value)


def _explain(validate: Check, samples: Sequence[object], rounds: int) -> None:
    # Drive the aggregating validate/explain path, which differs from is_valid.
    for _ in range(rounds):
        for value in samples:
            with suppress(ValidationError):
                validate(value)


def _lists_with_readers() -> None:
    # The lists whose element a reader of its own settles: a union of literals
    # by its table, a class by each element's type, a scalar or `None` by its
    # kind's loop -- an integer's and a string's, the two kinds most often made
    # optional -- any other union of scalars by its branches' tests, a tuple of
    # scalars by its positions', a named tuple by its type and positions', a
    # dataclass by its type and the record of its attributes, a refinement by
    # its own check. The walk around those readers is the one every other list
    # takes, and a reader the profile never enters moves how that walk is laid
    # out.
    statuses = Validator(list[Literal["new", "open", "done"]])
    _run(statuses.is_valid, [["new", "open", "done"] * 8, ["new", "gone"]], 2000)
    days = Validator(list[datetime.date])
    _run(days.is_valid, [[datetime.date(2020, 1, 1)] * 24, [1]], 2000)
    optional = Validator(list[int | None])
    _run(optional.is_valid, [[1, None] * 12, [1, "x"]], 2000)
    _explain(optional.validate, [[1, None] * 12], 500)
    names = Validator(list[str | None])
    _run(names.is_valid, [["a", None] * 12, ["a", 1]], 2000)
    keys = Validator(list[int | str])
    _run(keys.is_valid, [[1, "a"] * 12, [1, None]], 2000)
    coordinates = Validator(list[tuple[int, str]])
    _run(coordinates.is_valid, [[(1, "a")] * 24, [(1, 2)]], 2000)
    points = Validator(list[_Point])
    _run(points.is_valid, [[_Point(1, "a")] * 24, [(1, "a")]], 2000)
    pairs = Validator(list[_Pair])
    _run(pairs.is_valid, [[_Pair(1, "a")] * 24, [(1, "a")]], 2000)
    # A field of the wrong type, which the constructor does not check.
    wrong = _Pair(1, "a")
    vars(wrong)["y"] = 2
    _explain(pairs.validate, [[_Pair(1, "a")] * 24, [wrong]], 500)
    bounded = Validator(list[Annotated[int, _AtLeast(0)]])
    _run(bounded.is_valid, [list(range(24)), [0, -1]], 2000)
    _explain(bounded.validate, [list(range(24)), [0, -1]], 500)
    worded = Validator(list[Annotated[str, Regex("[a-z]+")]])
    _run(worded.is_valid, [["ab", "cd"] * 12, ["ab", "1"]], 2000)


def _records() -> None:
    # Closed records of a few widths, with optional keys, valid and invalid.
    for width in (4, 16, 50):
        spec: dict[str, object] = {f"f{i}": int for i in range(width)}
        spec["note?"] = str
        rec = Validator(spec)
        good = {f"f{i}": i for i in range(width)}
        bad_missing = {f"f{i}": i for i in range(width - 1)}
        bad_type = {**good, "f0": "x"}
        extra = {**good, "unexpected": 1}
        samples = [good, {**good, "note": "ok"}, bad_missing, bad_type, extra]
        _run(rec.is_valid, samples, 2000)
        _explain(rec.validate, [good, bad_type, extra], 500)
        text = "{" + ", ".join(f'"f{i}": {i}' for i in range(width)) + "}"
        _run(rec.is_valid_json, [text, text.replace(": 0", ': "x"', 1)], 1500)


def _sequences() -> None:
    # Homogeneous and heterogeneous sequences of varied length, decided and
    # explained: `validate` reads a list of one kind through a loop of its own,
    # the elements before the first that fails through its test, and the rest
    # in place -- most of a list refused at its second element, which is what
    # that scan reads most, and only the failure of one refused at its last.
    for length in (8, 64, 1000):
        ints = Validator(list[int])
        data: list[object] = list(range(length))
        _run(ints.is_valid, [data, [*data[:-1], "x"]], max(50, 20000 // length))
        refused = [[data[0], "x", *data[2:]], [*data[:-1], "x"]]
        _explain(ints.validate, [data, *refused], max(20, 5000 // length))
        json_text = "[" + ", ".join(str(n) for n in range(length)) + "]"
        _run(ints.is_valid_json, [json_text], max(50, 10000 // length))
    pair = Validator(tuple[int, str])
    _run(pair.is_valid, [(1, "a"), (1, 2), ("a", "b")], 5000)


def _containers() -> None:
    # Lists whose elements are containers, which the homogeneous shapes above
    # never reach: those take a loop of their own, and every other list is read
    # through the general one. A list of records, and a list nested as deep as
    # a document nests.
    class Person(TypedDict):
        name: str
        age: int

    people = Validator(list[Person])
    crowd = [{"name": "Ada", "age": n} for n in range(20)]
    _run(people.is_valid, [crowd, [*crowd, {"name": 5, "age": 1}]], 1000)
    deep_schema: object = int
    deep_value: object = 0
    for _ in range(25):
        deep_schema = list[deep_schema]  # type: ignore[valid-type]
        deep_value = [deep_value]
    _run(Validator(deep_schema).is_valid, [deep_value, [[1]]], 4000)


def _documents() -> None:
    # Nested documents (records of lists of records), valid and invalid.
    nested = Validator({"user": {"name": str, "age?": int}, "tags": list[str]})
    _run(
        nested.is_valid,
        [
            {"user": {"name": "Ada", "age": 36}, "tags": ["a", "b"]},
            {"user": {"name": "Ada"}, "tags": []},
            {"user": {"name": 5}, "tags": ["a"]},
            {"user": {"name": "Ada"}, "tags": [1]},
        ],
        4000,
    )


def _unions() -> None:
    # Literal unions (string enum and integer codes) and a structural union.
    status = Validator(Literal["pending", "active", "paused", "finished", "failed"])
    _run(status.is_valid, ["active", "failed", "unknown", 1], 8000)
    codes = union(*range(32))
    _run(codes.is_valid, [0, 31, 32, "x"], 8000)
    scalar_or_none = Validator(int | str | None)
    _run(scalar_or_none.is_valid, [1, "a", None, 1.5], 8000)


def _mappings() -> None:
    mapping = Validator({str: int})
    big_map = {f"k{i}": i for i in range(50)}
    _run(mapping.is_valid, [big_map, {**big_map, "bad": "x"}], 1500)


def _scalars() -> None:
    # Scalars across the type lattice.
    scalars: list[tuple[object, object, object]] = [
        (int, 7, "x"),
        (str, "s", 7),
        (float, 1.5, 1),
        (bytes, b"x", "x"),
    ]
    for schema, ok, bad in scalars:
        _run(Validator(schema).is_valid, [ok, bad], 12000)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.parse_args(argv)
    _records()
    _sequences()
    _containers()
    _lists_with_readers()
    _documents()
    _unions()
    _mappings()
    _scalars()
    _relations()


def _relations() -> None:
    """Ask the relations, which are the half of the surface the walk is not.

    `relation_to`, `is_subtype_of`, `is_empty` and `is_equivalent` run the
    decision procedures, which share nothing with the membership walk above --
    different functions, different branches. A profile that never enters them
    gives the shipped wheel no counts for the whole decision path, so the
    branches a caller's comparison takes are the ones the optimizer guesses at.

    The pairs are chosen for the two levels a relation has: the shapes a *rule*
    decides, which is the common case and the one worth laying out well, and the
    shapes that reach the set representation, which is where the build happens.
    Both the proving and the refuting direction, since they take different arms.
    """

    @dataclass
    class Point:
        x: int
        y: int

    @dataclass
    class Other:
        x: int
        y: int

    class Plain:
        pass

    tree = recursive(lambda node: {"value": int, "left?": node, "right?": node})
    wide = {f"f{i}": int for i in range(8)}
    narrow = {f"f{i}": int for i in range(7)}
    worded = Regex("[a-z]+")
    pairs: list[tuple[object, object]] = [
        # Decided by a rule, in both directions.
        (bool, int),
        (int, str),
        (list[int], list[int | str]),
        (list[int], tuple[int, int]),
        (Literal["a", "b"], str),
        (wide, narrow),
        (narrow, wide),
        (Point, Other),
        (Point, list[int]),
        (Plain, int),
        (worded, str),
        (worded, int),
        (complement(int), list[int]),
        (tree, int | str),
        (tuple[int, str], tuple[int, str]),
        # Reaching the set representation.
        (dict[str, list[int]], dict[str, int]),
        (intersection(list[int], complement(list[str])), list[int]),
        (int | str, complement(bytes)),
    ]
    for left, right in pairs:
        subject = Validator(left)
        for _ in range(400):
            subject.relation_to(right)
            subject.is_subtype_of(right)
    for schema in (wide, tree, list[int], int | str, worded):
        checked = Validator(schema)
        for _ in range(400):
            checked.is_empty()
            checked.is_equivalent(schema)


if __name__ == "__main__":
    main(sys.argv[1:])
