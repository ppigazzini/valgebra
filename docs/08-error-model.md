---
description: Exception type, error codes, and the violation path format.
---

# Error model

When a value does not satisfy a schema, `validate` raises `ValidationError`. The
exception is not just a message: it carries a stable, machine-readable model
meant to be read by tools and agents, not only humans.

## The shape

A `ValidationError` exposes:

- `errors` — a tuple of structured items, one per failure. Each item is a plain
  dict with these keys:
  - `code` — a stable, machine-readable code (e.g. `int_type`, `missing_key`,
    `too_short`).
  - `path` — the location of the offending value from the root, a tuple of
    string keys and integer indices (empty at the root). A dict key that is not
    a string has no spelling here, so it appears as its `repr`: the segment names
    the key rather than being one a caller can index back with. A string key is
    itself, in full.
  - `message` — the rendered one-line human message.
  - `expected` — a short label of the expected set (e.g. `int`).
  - `value` — a repr-style summary of the offending value.
- `message`, `code`, `path`, `expected`, `value` — scalar convenience
  attributes mirroring the first item. `str(exc)` is a summary of every failure.

## Crossing a process boundary

The exception pickles, and the structured model travels with it. A worker that
validates and fails delivers the failure itself — code, path and every item of
`errors` — rather than a message it has flattened by hand:

```python
import pickle

from valgebra import ValidationError, Validator

try:
    Validator({"a": int}).validate({"a": "x"})
except ValidationError as err:
    restored = pickle.loads(pickle.dumps(err))
    assert restored.code == "int_type"
    assert restored.path == ("a",)
    assert restored.errors == err.errors
```

That covers a `multiprocessing` worker, a process pool, a task queue, and a test
runner that forwards failures from a subprocess.

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

## When a value changes while it is checked

Membership reads a container entry by entry and runs Python at almost every one,
so the container can move underneath the reading: a predicate that writes to the
dict it is checking, and — on a free-threaded interpreter — another thread
writing to a shared value. A reading interrupted that way decides nothing about
the contents, so it is reported rather than guessed: the value is a non-member
and `validate` names it `mutated_during_validation`.

```python
from typing import Annotated

import annotated_types as at

from valgebra import ValidationError, Validator

grown = {"a": 1, "b": 2}
schema = Validator(
    {
        "a": Annotated[int, at.Predicate(lambda _: grown.setdefault("c", 3) or True)],
        "b": int,
        "c?": int,
    }
)

assert schema.is_valid(grown) is False
try:
    schema.validate(grown)
except ValidationError as error:
    assert error.code == "mutated_during_validation"
```

Only a change in the container's **size** costs the reading; a value rewritten in
place leaves the entries where they are and the check answers normally. The
same code also reports the rarer case of a value that answers two readings
differently — a predicate or an `__eq__` that is not a function of the value —
because it is the same failure: the check has no stable value to decide about.

## The set of codes

There is no hand-maintained list of every code on this page, because a list that
drifts from the walk is worse than none. The codes are pinned by
`tests/test_error_codes.py`, which asserts the code and the path for each node
kind — read it as the enumeration, and add a case there when you need one that is
not covered.

What is guaranteed here is the property a caller depends on: a code is stable and
does not change meaning across releases, so branching on one written down today
keeps working. New codes may appear for node kinds that gain a distinct failure.

## Determinism

For a given schema and value the error model is deterministic: the same codes,
paths, and order across runs and platforms. Tools can diff it. A set has no
positions, so its failing elements are reported in the order of what they say
rather than the order the interpreter hands them over, which moves with the hash
seed; `fail_fast` keeps the first of that order. The exact output
is locked by snapshot tests (the message format on the Rust side, the structured
`errors` on the Python side), so any change to it is reviewed, never silent.

## Message style guide

Messages and codes follow a fixed style so they stay predictable:

- One line, present tense, of the form `expected <X>, got <Y> [<code>]`; a
  located failure prefixes `at <path>: `.
- The `code` is stable and machine-readable; it is the field to branch on, not
  the prose. Codes do not change meaning across releases.
- `expected` names the set, `value` is a short repr of what was found, truncated
  so a large value cannot flood the message. A union names each of its branches,
  bounded, as each would name itself alone.
- Set-membership failures (`union`, `complement`) report at the location of the
  combinator, not inside a discarded branch.

## What a union's `expected` says

A union that no branch admits reports one failure at the union's own location,
and its `expected` names each branch the way that branch names itself when it is
the only thing that failed:

```python
import enum
from typing import Literal

from valgebra import ValidationError, Validator, union


class Backend(enum.Enum):
    TORCH = "torch"


for spec, value in [
    (Literal["torch", "jax"], "tensorflow"),
    (union(Backend, Literal["cpu"]), "arcfase"),
]:
    try:
        Validator(spec).validate(value)
    except ValidationError as err:
        print(err.errors[0]["expected"])

# one of: the literal 'torch', the literal 'jax'
# one of: Backend, the literal 'cpu'
```

A `Literal[...]` builds a union of its constants, so its branches are the
constants and the message lists them. An `Enum` branch names the class, as it
does alone.

The list is bounded at 64 labels and ends in `...` beyond that, so a union wide
enough to be a generated table reports a readable prefix. A branch that is itself
a union contributes its own members, so the label count and the branch count are
not the same number. See [resource limits](10-limits.md).

**`expected` describes the schema as written, not a canonical name for the set.**
`int | str`, `str | int` and `~(~int & ~str)` denote one set and read
differently, in the same way `repr` renders the annotation that produced a
schema. Read `expected` to see what was asked for; use
[`is_equivalent`](04-algebra.md) to ask whether two schemas mean the same thing.

## Which spelling produces which code

Two spellings of the same singleton denote the same set and are `is_equivalent`,
but they build different schema shapes and so report differently:

```python
from typing import Literal

from valgebra import ValidationError, Validator

for spec in ("active", Literal["active"]):
    try:
        Validator(spec).validate("x")
    except ValidationError as err:
        print(err.errors[0]["code"], "|", err.errors[0]["expected"])

# literal_error | the literal 'active'
# union_error   | one of: literal
```

`Literal[...]` builds a union of its constants — of one branch, when there is one
constant — and a union reports `union_error`. The bare constant builds a literal
leaf and reports `literal_error`. `simplify()` reduces the one-branch union to
the leaf, so `Validator(Literal["active"]).simplify()` reports `literal_error`
and compares equal to `Validator("active")` under `==`.

Branch on the code you actually observe for the spelling you actually write.
