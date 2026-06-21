# Error model

When a value does not satisfy a schema, `validate` raises `ValidationError`. The
exception is not just a message: it carries a stable, machine-readable model
meant to be read by tools and agents, not only humans.

## The shape

A `ValidationError` exposes:

- `errors` — a tuple of structured items, one per failure. Each item is a plain
  dict with these keys:
  - `code` — a stable, machine-readable code (e.g. `int_type`, `missing_key`,
    `literal_error`).
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
from valgebra import ValidationError, Validator

try:
    Validator({"a": int, "b": str, "c": int}).validate({"a": "x", "b": 1, "c": "y"})
except ValidationError as err:
    assert [e["path"] for e in err.errors] == [("a",), ("b",), ("c",)]
```

Pass `fail_fast=True` to stop at the first failure instead:

```python
from valgebra import ValidationError, Validator

try:
    Validator({"a": int, "b": str}).validate({"a": "x", "b": 1}, fail_fast=True)
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

The closest-branch search is a bounded, best-effort heuristic: it runs only when
a value has already failed the union, and it inspects at most the first 64
branches. A union wider than that still reports correctly — the membership
decision always considers every branch — but its error may fall back to the
`union_error` summary rather than pinpointing a branch past the cap. This keeps
building an error for a pathologically wide union (a large `Literal[...]`, say)
bounded; the successful path is unaffected.

## JSON output

Every item is JSON-serializable (the `path` is a tuple of strings and ints), so
the JSON output mode is the standard library:

```python
import json

from valgebra import ValidationError, Validator

schema = Validator({"name": str, "age": int})
try:
    schema.validate({"name": "Ada", "age": "old"})
except ValidationError as err:
    payload = json.dumps(err.errors)
    restored = json.loads(payload)
    assert restored[0]["code"] == "int_type"
    assert restored[0]["path"] == ["age"]
    assert restored[0]["expected"] == "int"
```

## When a comparison raises

Checking membership reads a value through Python operations that can raise: an
`__eq__` for a literal, a rich comparison for a numeric bound, `isinstance` for a
class, `getattr` for an attribute, `__len__` for a length, `__mod__` for a
multiple-of. A value whose comparison, instance check, or attribute access
**raises an ordinary exception is treated as a non-member** — a value that cannot
answer "are you in this set?" is not in it, the same pragmatic stance
pydantic-core takes. The one ordinary-exception case carved out is a user
predicate (`Annotated[..., some_callable]`): a predicate that raises an ordinary
exception is reported as a distinct `predicate_error`, not folded into an ordinary
failed match, so a buggy predicate stays visible.

A **fatal interpreter signal is never folded** — at every site, the predicate and
attribute access included. A base exception that is not an ordinary exception
(`KeyboardInterrupt`, `SystemExit`, `GeneratorExit`), or a `MemoryError` or
`RecursionError`, means the interpreter is unwinding, not that the value is a
non-member, so it propagates out of `validate`/`is_valid` rather than being
reported as "not a member" or a `predicate_error`.

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
