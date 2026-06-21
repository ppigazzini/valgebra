"""Snapshot coverage of the structured error model over a representative corpus.

Each case is a (schema, value) failure; the captured `errors` are snapshotted so
any change to the machine-readable output is reviewed in the snapshot diff, never
auto-accepted. Values are chosen with stable reprs so the snapshot is identical
across Python versions and platforms.

This is a change-detector for the message and value-summary shape, not the
correctness oracle: the `code` and `path` each node kind must emit are pinned
against hand-written expectations in ``tests/test_error_codes.py``, independent of
the implementation's own output. The snapshot is recorded from the code, so read
it alongside that suite rather than as a standalone correctness check.
"""

from valgebra import ValidationError, Validator, union

# (label, schema, value). The schema is either a raw spec or a compiled
# validator (the named combinators return one).
CASES: list[tuple[str, object, object]] = [
    ("scalar", int, "x"),
    ("nested_record", {"name": str, "age": int}, {"name": "Ada", "age": "old"}),
    ("aggregated_fields", {"a": int, "b": str, "c": int}, {"a": "x", "b": 1, "c": "y"}),
    ("nested_list", {"items": [int]}, {"items": [1, "x", "y"]}),
    ("missing_required_key", {"a": int}, {}),
    ("extra_forbidden", {"a": int}, {"a": 1, "b": 2}),
    ("closest_branch", union(int, {"a": int}), {"a": "x"}),
    ("generic_union", int | str, 1.5),
    ("literal_typed_singleton", 1, True),
]


def _compiled(schema: object) -> Validator:
    return schema if isinstance(schema, Validator) else Validator(schema)


def _capture(schema: object, value: object) -> list[dict[str, object]]:
    try:
        _compiled(schema).validate(value)
    except ValidationError as err:
        return [dict(item) for item in err.errors]
    return []  # pragma: no cover - every case is meant to fail


def test_structured_error_corpus(snapshot):
    captured = {label: _capture(schema, value) for label, schema, value in CASES}
    assert captured == snapshot
