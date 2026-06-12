# Error model

When a value does not satisfy a schema, `validate` raises `ValidationError`. The
exception is not just a message: it carries a stable, machine-readable model
meant to be read by tools and agents, not only humans.

## The shape

A `ValidationError` exposes:

- `errors` — a tuple of structured items, one per failure. Each item is a plain
  dict with these keys:
  - `code` — a stable, machine-readable code (e.g. `int_type`, `missing_key`,
    `literal_value`).
  - `path` — the location of the offending value from the root, a tuple of
    string keys and integer indices (empty at the root).
  - `message` — the rendered one-line human message.
  - `expected` — a short label of the expected set (e.g. `int`).
  - `value` — a repr-style summary of the offending value.
- `message`, `code`, `path`, `expected`, `value` — scalar convenience
  attributes mirroring the first item. `str(exc)` is a summary of every failure.

## Aggregation and fail-fast

By default the walk does not stop at the first failure: it collects every
independent failure — each record field, each sequence or tuple element, each
mapping entry — into `errors`, so one call reports all the problems with a value.

```python
from valgebra import ValidationError, validator

try:
    validator({"a": int, "b": str, "c": int}).validate({"a": "x", "b": 1, "c": "y"})
except ValidationError as err:
    assert [e["path"] for e in err.errors] == [("a",), ("b",), ("c",)]
```

Pass `fail_fast=True` to stop at the first failure instead:

```python
from valgebra import ValidationError, validator

try:
    validator({"a": int, "b": str}).validate({"a": "x", "b": 1}, fail_fast=True)
except ValidationError as err:
    assert len(err.errors) == 1
```

A node-level type mismatch (a value that is not a dict where a record is
expected) is terminal for that subtree: there is nothing to descend into.

## Unions report the closest branch

When a value matches no branch of a union, valgebra does not dump every branch's
failure. It reports the **closest** branch — the one that descended furthest into
the value before failing — and that branch's own (aggregated) errors:

```python
from valgebra import ValidationError, union

try:
    union(int, {"a": int}).validate({"a": "x"})
except ValidationError as err:
    # The value is a dict, so the record branch is closer than `int`.
    assert err.errors[0]["path"] == ("a",)
    assert err.errors[0]["code"] == "int_type"
```

When no branch makes any progress past the union's own location — for example
`int | str` against a float, where every branch is a flat type mismatch — there
is no closer branch, so a single `union_error` is the honest report. A
`complement` likewise reports one failure at its location.

## JSON output

Every item is JSON-serializable (the `path` is a tuple of strings and ints), so
the JSON output mode is the standard library:

```python
import json

from valgebra import ValidationError, validator

schema = validator({"name": str, "age": int})
try:
    schema.validate({"name": "Ada", "age": "old"})
except ValidationError as err:
    payload = json.dumps(err.errors)
    restored = json.loads(payload)
    assert restored[0]["code"] == "int_type"
    assert restored[0]["path"] == ["age"]
    assert restored[0]["expected"] == "int"
```

## Determinism

For a given schema and value the error model is deterministic: the same codes,
paths, and order across runs and platforms. Tools can diff it. The exact output
is locked by snapshot tests (the message format on the Rust side, the structured
`errors` on the Python side), so any change to it is reviewed, never silent.

## Message style guide

Messages and codes follow a fixed style so they stay predictable:

- One line, present tense, of the form `expected <X>, got <Y> [<code>]`; a
  located failure prefixes `at <path>: `.
- The `code` is stable and machine-readable; it is the field to branch on, not
  the prose. Codes do not change meaning across releases.
- `expected` names the set, `value` is a short repr of what was found, truncated
  so a large value cannot flood the message.
- Set-membership failures (`union`, `complement`) report at the location of the
  combinator, not inside a discarded branch.
