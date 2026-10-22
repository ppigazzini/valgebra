"""Profile-guided-optimization training workload for every `--pgo` build.

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

import datetime as dt
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Annotated, Literal, NamedTuple, TypedDict

from _lib import arguments

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
    "decision-rules": ("_relations",),
    "decision-sets": ("_relations",),
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
    "binding-record-list": ("_containers",),
    "binding-nullable": ("_lists_with_readers",),
    "json_document": ("_json_documents",),
    "binding-json": ("_json_documents",),
    "binding-json-reject": ("_json_documents",),
    "binding-json-deep": ("_json_documents",),
    "binding-set": ("_sets",),
    "binding-subclass": ("_subclasses",),
}

#: Why a JSON shape the training has not been measured on is left out of it:
#: the records are trained, and the reading the shape adds to them is not.
_JSON_UNTRAINED = (
    "A parsed JSON array of records is trained (`_json_documents`), and the "
    "reading this shape adds to it is not. Training a reading re-weights the "
    "profile every timed shape is laid out by, so each is read on the "
    "comparison gate before and after rather than added here."
)

#: The shapes no call here takes the reading of, each with what it reads. The
#: shipped wheel lays each one's path out without counts.
UNTRAINED: dict[str, str] = {
    "binding-json-union": (
        "A union of two record kinds over parsed JSON objects, each element "
        "asked of the second branch after failing the first. The records' "
        "training reaches it, 12% faster on CPython 3.14; trained beside them, "
        "it read no faster, and took the list nested twenty-five deep from 5% "
        "faster than untrained to 3%."
    ),
    "binding-json-open": (
        "JSON records read through a key-type clause. The records' training "
        "reaches it, 3% fewer instructions on CPython 3.14; trained beside them "
        "and the union, it read 3% fewer again, 2% slower on one build and 10% "
        "faster on another, and took the list nested twenty-five deep from 3% "
        "faster than untrained to 3% slower."
    ),
    "binding-json-bounded": (
        "JSON records under a length bound, whose length is the count of the "
        "keys a parsed object holds: no call here validates a JSON document "
        "against a bounded mapping. " + _JSON_UNTRAINED
    ),
    "binding-recursive": (
        "A value walked against a recursive schema, entering the reference and "
        "its trail per level. The JSON reference's training reaches the "
        "reference and the trail the two walks share, which took this shape "
        "about a fifth faster on CPython 3.14; trained on its own as well, it "
        "read 0.5% more instructions here and in the list nested twenty-five "
        "deep."
    ),
    "decision-lengths": (
        "A list's length bound lowered to an automaton with a state per element "
        "and met with the element's shape: `_relations` reaches the sequence "
        "automaton's product through a meet of lists, and no call here relates "
        "a length-bounded list."
    ),
    "binding-mapping-lists": (
        "A mapping whose value is a list, read by the scan that asks the "
        "clause of each key and walks each value: no call here validates one, "
        "and `_mappings` reads a `dict[str, int]`, which a reader of its own "
        "takes."
    ),
    "binding-protocol": (
        "Compiling a protocol, whose members `typing` lists and the frontend "
        "classifies name by name: no call here builds a validator from a "
        "protocol."
    ),
    "core": (
        "The composition remap and the record transform, run directly in Rust: "
        "the calls here reach the remap through `union`, `intersection` and "
        "`complement`, and no call opens or closes a record."
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
    days = Validator(list[dt.date])
    _run(days.is_valid, [[dt.date(2020, 1, 1)] * 24, [1]], 2000)
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
    # The refined lists are read at the weight of the mapping read beside
    # them: trained at a quarter of these rounds beside a hundred-entry
    # mapping, a list of refined strings reads 0.7% more instructions on 3.12
    # and 3.14, three an element, in its base's type test and the call to it.
    bounded = Validator(list[Annotated[int, _AtLeast(0)]])
    _run(bounded.is_valid, [list(range(24)), [0, -1]], 4000)
    _explain(bounded.validate, [list(range(24)), [0, -1]], 500)
    worded = Validator(list[Annotated[str, Regex("[a-z]+")]])
    _run(worded.is_valid, [["ab", "cd"] * 12, ["ab", "1"]], 8000)


def _records() -> None:
    # Closed records of a few widths, with optional keys, valid and invalid.
    for width in (4, 16, 50):
        spec: dict[str, object] = {f"f{i}": int for i in range(width)}
        spec["note?"] = str
        rec = Validator(spec)
        # Built as often as it is checked below, since a validator is built
        # once a process and timed as a build; a frontend read three times is
        # laid out by whichever other reading moves it.
        for _ in range(200):
            Validator(spec)
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
    # never reach: a list of records, read by the records' reader, and a list
    # nested as deep as a document nests, read through the general loop.
    class Person(TypedDict):
        name: str
        age: int

    # These two carry the weight a document's records do, and the second has a
    # field that is a list, as a payload's records have: trained a quarter as
    # often and with scalar fields alone, a list of records reads 1.3% more by
    # count, its record laid out around the paths beside it.
    class Member(TypedDict):
        id: int
        name: str
        tags: list[str]

    people = Validator(list[Person])
    crowd = [{"name": "Ada", "age": n} for n in range(20)]
    _run(people.is_valid, [crowd, [*crowd, {"name": 5, "age": 1}]], 4000)
    members = Validator(list[Member])
    roster = [{"id": n, "name": "Ada", "tags": ["a", "b"]} for n in range(20)]
    refused = {"id": "x", "name": "Ada", "tags": []}
    _run(members.is_valid, [roster, [*roster, refused]], 1000)
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


def _json_documents() -> None:
    # A parsed JSON array of records, read element by element by the array's
    # reader for records, which a single JSON record never reaches: two hundred
    # records accepted, and the same refused at the hundredth, whose `id` is a
    # string.
    class JsonRecord(TypedDict):
        id: int
        name: str
        email: str
        tags: list[str]
        meta: dict[str, str]

    records = [
        {
            "id": i,
            "name": "Ada",
            "email": "a@b.c",
            "tags": ["x", "y"],
            "meta": {"k": "v"},
        }
        for i in range(200)
    ]
    refused = [*records[:100], {**records[100], "id": "x"}, *records[101:]]
    document = Validator(list[JsonRecord])
    _run(document.is_valid_json, [json.dumps(records), json.dumps(refused)], 400)
    # Not `load`, whose builder no other call here takes: trained on these
    # records, the profile lays the general list scan out for long lists, and
    # the list nested twenty-five deep pays for it at every level, 3.5 to 7%
    # more instructions on the profiled wheel, past what its gate allows.

    # A reference on the JSON path: twenty trees twelve deep, each a pair of a
    # subtree and an integer, against the schema that names itself -- whose
    # values the list of them is one of.
    def grown(depth: int) -> object:
        return 1 if depth == 0 else [grown(depth - 1), depth]

    tree = recursive(lambda node: int | list[node])  # ty: ignore[invalid-type-form]
    forest = json.dumps([grown(12) for _ in range(20)])
    _run(Validator(tree).is_valid_json, [forest], 400)


def _sets() -> None:
    # A set read through its own iterator, which no list or tuple reading
    # takes: sixty-four strings accepted, and one integer among them refused.
    words = Validator(set[str])
    names = {f"k{i}" for i in range(64)}
    _run(words.is_valid, [names, {*names, 1}], 2000)


def _subclasses() -> None:
    # A named tuple read against `tuple[int, ...]`, through the length accessor
    # a subclass of `tuple` inherits rather than the plain tuple's own.
    class Point(NamedTuple):
        x: int
        y: int
        z: int

    ints = Validator(tuple[int, ...])
    _run(ints.is_valid, [Point(1, 2, 3), _Point(1, "a")], 8000)


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


def _membership(validator: Validator, samples: Sequence[object], rounds: int) -> int:
    # The container door, `value in validator`, reaches the check through a
    # slot of its own, and the build inlines a copy of the check there: a copy
    # the profile never enters is laid out by chance, as every other reading is.
    found = 0
    for _ in range(rounds):
        for value in samples:
            found += value in validator
    return found


def _scalars() -> None:
    # Scalars across the type lattice, through the method and the operator.
    scalars: list[tuple[object, object, object]] = [
        (int, 7, "x"),
        (str, "s", 7),
        (float, 1.5, 1),
        (bytes, b"x", "x"),
    ]
    for schema, ok, bad in scalars:
        validator = Validator(schema)
        _run(validator.is_valid, [ok, bad], 12000)
        _membership(validator, [ok, bad], 12000)


def main(argv: list[str]) -> None:
    parser = arguments(__doc__)
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
    _json_documents()
    _sets()
    _subclasses()


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
