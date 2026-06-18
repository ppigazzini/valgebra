"""Profile-guided-optimization training workload for the release PGO lane.

Run against a profile-instrumented build of the extension, this exercises the
validation hot paths over a broad, production-like spread of schema shapes so the
recorded profile generalizes rather than overfitting a single micro-benchmark:
scalars, closed and open records of several widths, homogeneous and
heterogeneous sequences, nested documents, literal and structural unions,
mappings, and the JSON path. Both passing and failing values are fed, since the
fast accept path and the rejecting (and aggregating explain) paths take
different branches.

It depends only on ``valgebra`` and the standard library (no test, comparison,
or annotation-metadata packages), so it runs in the minimal environment maturin
sets up for ``--pgo``. Keep it quick: a few seconds is enough to accumulate
representative branch counts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Literal

from valgebra import ValidationError, Validator, union

# A bound validator method; `...` admits is_valid/validate/is_valid_json alike.
Check = Callable[..., object]


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


def main() -> None:
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

    # Homogeneous and heterogeneous sequences of varied length.
    for length in (8, 64, 1000):
        ints = Validator(list[int])
        data: list[object] = list(range(length))
        _run(ints.is_valid, [data, [*data[:-1], "x"]], max(50, 20000 // length))
        json_text = "[" + ", ".join(str(n) for n in range(length)) + "]"
        _run(ints.is_valid_json, [json_text], max(50, 10000 // length))
    pair = Validator(tuple[int, str])
    _run(pair.is_valid, [(1, "a"), (1, 2), ("a", "b")], 5000)

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

    # Literal unions (string enum and integer codes) and a structural union.
    status = Validator(Literal["pending", "active", "paused", "finished", "failed"])
    _run(status.is_valid, ["active", "failed", "unknown", 1], 8000)
    codes = union(*range(32))
    _run(codes.is_valid, [0, 31, 32, "x"], 8000)
    scalar_or_none = Validator(int | str | None)
    _run(scalar_or_none.is_valid, [1, "a", None, 1.5], 8000)

    # Mappings.
    mapping = Validator({str: int})
    big_map = {f"k{i}": i for i in range(50)}
    _run(mapping.is_valid, [big_map, {**big_map, "bad": "x"}], 1500)

    # Scalars across the type lattice.
    scalars: list[tuple[object, object, object]] = [
        (int, 7, "x"),
        (str, "s", 7),
        (float, 1.5, 1),
        (bytes, b"x", "x"),
    ]
    for schema, ok, bad in scalars:
        _run(Validator(schema).is_valid, [ok, bad], 12000)


if __name__ == "__main__":
    main()
