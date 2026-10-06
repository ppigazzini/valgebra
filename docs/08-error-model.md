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
    string keys, integer keys and integer indices (empty at the root). A string
    key is itself, in full, and an **integer key is itself as an integer** —
    whatever its size, and `True`/`False` with them, since `d[True]` and `d[1]`
    are one entry — so walking the path back down reaches the value: `d[2]` and
    `d["2"]` are different entries and the path says which. A key that is
    neither, and a string holding a lone surrogate, has no spelling here and
    appears as its `repr` — naming the key rather than being one a caller can
    index back with.
  - `message` — the rendered one-line human message, `at <location>: expected
    <expected>, got <value> [<code>]`. The location writes a key as itself and
    an index or an integer key in brackets, `items[2].id`; a key that is not a
    bare name — empty, or holding a `.`, a bracket, whitespace or a control
    character — is written as a subscript of its Python literal, `['a.b']`, so
    no two string or integer keys read alike and the message stays on one
    line. A key of any other type is written as its `repr`, and reads as the
    string key with that text.
  - `expected` — a short label of the expected set (e.g. `int`).
  - `value` — a repr-style summary of the offending value.
- `message`, `code`, `path`, `expected`, `value` — scalar convenience
  attributes mirroring the first item. `str(exc)` is a summary of every failure.

Every one of the six is present on every `ValidationError`, however it was made.
The model describes *failures*, so an error you construct yourself reports none
and reads as empty — empty strings and empty tuples — rather than raising
`AttributeError` for an attribute the type declares:

```python
from valgebra import ValidationError

error = ValidationError("something went wrong")
assert str(error) == "something went wrong"
assert error.code == ""
assert error.errors == ()
```

## Crossing a process boundary

The exception pickles, and the structured model travels with it. A worker that
validates and fails delivers the failure itself — code, path and every item of
`errors` — rather than a message it has flattened by hand:

```python
import pickle

from valgebra import ValidationError, Validator

raised = False
try:
    Validator({"a": int}).validate({"a": "x"})
except ValidationError as err:
    raised = True
    restored = pickle.loads(pickle.dumps(err))
    assert restored.code == "int_type"
    assert restored.path == ("a",)
    assert restored.errors == err.errors
assert raised
```

That covers a `multiprocessing` worker, a process pool, a task queue, and a test
runner that forwards failures from a subprocess.

## Aggregation and fail-fast

By default the walk does not stop at the first failure: it collects every
independent failure — each record field, each sequence or tuple element, each
mapping entry — into `errors`, so one call reports all the problems with a value.

```python
from valgebra import ValidationError, Validator

raised = False
try:
    Validator({"a": int, "b": str, "c": int}).validate({"a": "x", "b": 1, "c": "y"})
except ValidationError as err:
    raised = True
    assert [e["path"] for e in err.errors] == [("a",), ("b",), ("c",)]
assert raised
```

**There is no cap on how many failures come back.** A list of twenty thousand
values of the wrong type reports twenty thousand entries, and `str()` of that
exception is about a megabyte. That is deliberate: a cap would make `errors` a
sample, and a caller counting entries or looking for a particular path would be
reading a truncated list with nothing saying it was truncated. A failure costs
about 230 bytes held by the exception and about 790 once `errors` is read, so a
document of a million wrong elements reaches three quarters of a gigabyte.
Bounding the cost is the caller's, and there are two ways to do it — pass
`fail_fast=True`, or validate the value in pieces. Each individual `value` *is*
bounded, at eighty characters of its repr followed by `...` (`SUMMARY_CHARS`
in `crates/valgebra-py/src/errors.rs`).

Pass `fail_fast=True` to stop at the first failure instead:

```python
from valgebra import ValidationError, Validator

raised = False
try:
    Validator({"a": int, "b": str}).validate({"a": "x", "b": 1}, fail_fast=True)
except ValidationError as err:
    raised = True
    assert len(err.errors) == 1
assert raised
```

A node-level type mismatch (a value that is not a dict where a record is
expected) is terminal for that subtree: there is nothing to descend into. So is
a wrong length — a fixed shape's arity (`list_length`, `tuple_length`) or a
container's length bound (`too_short`, `too_long`) — which is read after the
kind and before any element, so the elements of a list too long for its bound
are not reported.

An `intersection` follows the same rule across its members. Each member is
checked and each failure collected, until one rejects the value *itself* rather
than something inside it — then the rest are not run, because they would
describe a value already known to be the wrong kind of thing. A class with
declared attributes is built as `isinstance` met with an attribute record, so a
value of the wrong class reports `instance_type` alone, not that same value's
missing attributes as well. A member that fails *inside* the value — an element,
a field, an attribute — leaves the others meaningful, and they are still
collected.

```python
from dataclasses import dataclass

from valgebra import ValidationError, Validator


@dataclass
class Point:
    x: int


raised = False
try:
    Validator(Point).validate(object())
except ValidationError as err:
    raised = True
    assert [(e["code"], e["path"]) for e in err.errors] == [("instance_type", ())]
assert raised
```

## Unions report the closest branch

When a value matches no branch of a union, valgebra does not dump every branch's
failure. It reports the **closest** branch — the one that descended furthest into
the value before failing, which is the branch whose first failure lies deepest —
and that branch's own (aggregated) errors:

```python
from valgebra import ValidationError, union

raised = False
try:
    union(int, {"a": int}).validate({"a": "x"})
except ValidationError as err:
    # The value is a dict, so the record branch is closer than `int`.
    raised = True
    assert err.errors[0]["path"] == ("a",)
    assert err.errors[0]["code"] == "int_type"
assert raised
```

When no branch makes any progress past the union's own location — for example
`int | str` against a float, where every branch is a flat type mismatch — there
is no closer branch, so a single `union_error` is the honest report. A
`complement` likewise reports one failure at its location.

Of several **record** branches, one is explained: the one that admitted the most
of the value before refusing it — the most fields, read with the fields decided
by their own value first, or the most entries where a record is read by its
entries — then one refused inside a field's nested value before one refused at
a field, the earliest on a tie. So a value tagged for one record of a union is
reported against that record:

```python
from typing import Literal

from valgebra import ValidationError, union

shapes = union(
    {"kind": Literal["circle"], "r": float},
    {"kind": Literal["square"], "side": float},
)
raised = False
try:
    shapes.validate({"kind": "square", "side": "two"})
except ValidationError as err:
    raised = True
    assert err.errors[0]["path"] == ("side",)
    assert err.errors[0]["code"] == "float_type"
assert raised
```

A value the union admits builds no report at all: the union is decided before
any branch is explained, so explaining a branch never runs the `__repr__` of a
value another branch admits.

Under `fail_fast` each branch is walked to its first failure and no further,
since that is all the choice reads, so refusing a union costs a fail-fast walk
of each branch rather than the size of the value. Without it every branch is
walked whole, the same branch is chosen, and all of its failures are reported;
`fail_fast` reports the one they lead with.

A branch the walk could not answer for keeps its own report rather than the
summary. `recursion_limit`, `recursion_loop`, `mutated_during_validation` and
`predicate_error` each say the *walk* stopped rather than that the value is
outside a set, and each of them fails at the union's own location — so the
progress rule would count it as no progress and the summary would drop the one
sentence saying what to do about it.

A `complement` reads the four the same way. A walk of its inner schema that
stopped has not said the value is outside that schema, so the complement does
not admit the value: it fails at its own location with the code the inner walk
stopped with, where a decided match fails with `unexpected_match`. A union
beside it that admits the value admits it, and a complement above it fails
with the same code.

```python
from valgebra import ValidationError, complement, recursive, union

lists = recursive(lambda t: union(int, [t]))
loop: list[object] = []
loop.append(loop)
assert not complement(lists).is_valid(loop)
try:
    complement(lists).validate(loop)
except ValidationError as error:
    assert error.code == "recursion_loop"
else:
    raise AssertionError("a value that contains itself is refused")
```

The closest-branch search is a bounded, best-effort heuristic: it runs only when
a value has already failed the union, and it inspects at most the first 64
branches (`CLOSEST_BRANCH_PROBE_LIMIT` in
`crates/valgebra-py/src/check/walk.rs`). A union wider than that still reports
correctly — the membership decision always considers every branch — but its
error may fall back to the `union_error` summary rather than pinpointing a
branch past the cap. This keeps
building an error for a pathologically wide union (a large `Literal[...]`, say)
bounded; the successful path is unaffected.

## JSON output

Every item is JSON-serializable (the `path` is a tuple of strings and ints), so
the JSON output mode is the standard library:

```python
import json

from valgebra import ValidationError, Validator

schema = Validator({"name": str, "age": int})
raised = False
try:
    schema.validate({"name": "Ada", "age": "old"})
except ValidationError as err:
    raised = True
    payload = json.dumps(err.errors)
    restored = json.loads(payload)
    assert restored[0]["code"] == "int_type"
    assert restored[0]["path"] == ["age"]
    assert restored[0]["expected"] == "int"
assert raised
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
reported as "not a member" or a `predicate_error`. Building a message is one of
those sites: a `__repr__` of the value, of a constant or bound the message
names, or of a key in the path, and a class's `__name__`, raise the signal out
of `validate` rather than reading as `<unrepresentable>`. Building a validator
and asking a relation are two more: a marker's attributes, a bound's conversion
and length, the repr a refusal names, and the predicate, `__eq__` or comparison
a relation probes a literal or a bound with, raise the signal rather than
answering `undecided`, a verdict or a refusal. Comparing, hashing and printing
a validator are the last: a constant whose `__eq__`, `__hash__` or `__repr__`
raises an ordinary exception reads as unequal, as adding nothing to the hash,
and as `<unrepresentable>`, and one raising a fatal signal raises it out of
`==`, `hash` or `repr`.

## The model is built when it is asked for

A raised `ValidationError` carries its failures, and the six attributes above
are built by the first access that wants one. A caller that logs `str(error)`
and moves on never pays for the rows; one that reads `errors` pays once, and the
value is kept on the exception so a second read is an ordinary attribute lookup.

The difference is large enough to state: over ten thousand failing rows, a
caller that reads none of them pays about a third of what building every row
would cost. Measure it against your own shape with

```bash
uv run --no-sync --group bench pytest benches/bench_validate.py -k error
```

Nothing about the model depends on it -- the same attributes, the same values,
the same `str()` -- and pickling carries the plain data, because crossing a
process boundary builds the model first.

A failure raised while another exception is being handled carries that
exception as its `__context__`, the way a `raise` inside the `except` block
would, and claims no `__cause__`.

## When a value changes while it is checked

Membership reads a container entry by entry and runs Python at almost every one,
so the container can move underneath the reading: a predicate that writes to the
dict it is checking, and — on a free-threaded interpreter — another thread
writing to a shared value. A reading interrupted that way decides nothing about
the contents, so it is reported rather than guessed: the value is a non-member.
`validate` names a list or a set that moved `mutated_during_validation`, and
reads a dict a second time and reports what it finds there -- the violation the
dict carries by then, or `mutated_during_validation` where it carries none.

```python
from typing import Annotated

import annotated_types as at

from valgebra import ValidationError, Validator

grown = {"a": 1, "b": 2}
schema = Validator(
    {
        "a": Annotated[int, at.Predicate(lambda _: bool(grown.setdefault("c", 3)))],
        "b": int,
        "c?": int,
    }
)

assert schema.is_valid(grown) is False
del grown["c"]  # the predicate added it during the check above
raised = False
try:
    schema.validate(grown)
except ValidationError as error:
    raised = True
    assert error.code == "mutated_during_validation"
assert raised
```

A **dict, a set and a list** are all read this way — each against a count taken
once, while it is read — so each reports rather than guesses; a tuple cannot be
resized and needs no guard. Only a change in the container's **size** costs the
reading, and only while that container is being read. A change that keeps the
size — a value rewritten in place, one key swapped for another, a `pop` and an
`append` — is not seen, and the check answers about what it read; neither is a
change to a container whose own reading has already finished, such as the first
element of a list while the second is walked. The
same code also reports the rarer case of a value that answers two readings
differently — a predicate or an `__eq__` that is not a function of the value —
because it is the same failure: the check has no stable value to decide about.

## The set of codes

There is no hand-maintained list of every code on this page, because a list that
drifts from the walk is worse than none. The codes are declared in two tables —
`Schema::error_code` in `crates/valgebra-core/src/ir.rs` for a leaf's own code,
and `crates/valgebra-py/src/codes.rs` for the rest — and
`tests/test_error_matrix.py` drives every code a report can carry, with the path
it reports. Read those as the enumeration.

What is guaranteed here is the property a caller depends on: a code is stable and
does not change meaning across releases, so branching on one written down today
keeps working. New codes may appear for node kinds that gain a distinct failure.

**Every code is reported the same way on both entry paths.** `validate_json` and
`load` parse the document and then walk the *Python value* the parser built, so
a document reaches the same codes a value does, with the same `path`. Three are
the exception, and each because the parser cannot build the value that reaches
them:

| Code | Why no document reaches it |
|---|---|
| `tuple_length` | a document's array is a list, never a tuple, so a tuple schema refuses it by kind first and the code is `tuple_type` |
| `recursion_loop` | a parsed document is a tree, so no value contains itself |
| `mutated_during_validation` | nothing runs against a parsed value while it is read, so it cannot move under the walk |

`tests/test_error_matrix.py` drives every code through both modes and both
paths. For `tuple_length` it carries that reason beside the code a document
gets *instead*, and it names the test that drives each of the other two, so the
claim is one a test can refute. `missing_attribute` is reached through a
`Protocol`, whose attribute record stands with no class beside it: a parsed
object is a `dict`, which carries no attribute, so a protocol refuses a
document member by member with that code.

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
  located failure prefixes `at <path>: `. The value's `repr` and a class's name
  are written as they are, so one whose text holds a newline spans lines.
- The `code` is stable and machine-readable; it is the field to branch on, not
  the prose. Codes do not change meaning across releases.
- `expected` names the set, `value` is a short repr of what was found, truncated
  so a large value cannot flood the message. A union names each of its branches,
  bounded, as each would name itself alone.
- A `complement`, and a `union` on which no branch makes progress, report at the
  location of the combinator. A union whose closest branch descended into the
  value reports that branch's failures, never a discarded branch's.

## What a union's `expected` says

A union that no branch admits reports one failure at the union's own location,
and its `expected` names each branch the way that branch names itself when it is
the only thing that failed:

```python
import enum
from typing import Literal

from valgebra import ValidationError, Validator, complement, intersection, union


class Backend(enum.Enum):
    TORCH = "torch"


def expected_of(spec: object, value: object) -> str:
    try:
        Validator(spec).validate(value)
    except ValidationError as err:
        return err.expected
    raise AssertionError("expected a failure")


assert (
    expected_of(Literal["torch", "jax"], "tensorflow")
    == "one of: the literal 'torch', the literal 'jax'"
)
assert expected_of(union(Backend, Literal["cpu"]), "arcfase") == (
    "one of: the literal 'cpu', Backend"
)
assert expected_of(union(str, intersection(int, complement(bool))), 1.5) == (
    "one of: str, int and not bool"
)
```

A `Literal[...]` builds a union of its constants, so its branches are the
constants and the message lists them. An `Enum` branch names the class, as it
does alone. A meet names what each of its members admits, joined with `and`,
and a class with declared attributes, which is a meet too, names the class.

The list is bounded at 64 labels (`UNION_LABEL_LIMIT` in
`crates/valgebra-py/src/check/walk.rs`) and ends in `...` beyond that, so a
union wide enough to be a generated table reports a readable prefix. A branch
that is itself a union contributes its own members, so the label count and the
branch count are not the same number. See [resource limits](10-limits.md).

**`expected` describes the schema as built, not a canonical name for the set.**
`int | str`, `str | int` and `~(~int & ~str)` denote one set and read
differently, in the same way `repr` renders the schema that was built. Read
`expected` to see what was asked for; use [`is_equivalent`](04-algebra.md) to ask
whether two schemas mean the same thing.

*As built* rather than *as written*, because the constructors fold: `complement`
cancels a complement, `union` collapses a join carrying a schema beside its own
complement, and `intersection` collapses the meet of that pair.
`complement(complement(int))` is the schema `int`, so it reports `int_type` and
names `int` — there is no second schema left to report. `intersection(int,
complement(int))` is `nothing`, so it reports `no_match` and names `nothing`,
for the same reason.

## Which spelling produces which code

Two spellings of the same singleton denote the same set and are `is_equivalent`,
but they build different schema shapes and so report differently:

```python
from typing import Literal

from valgebra import ValidationError, Validator


def report(spec: object) -> tuple[str, str]:
    try:
        Validator(spec).validate("x")
    except ValidationError as err:
        return err.code, err.expected
    raise AssertionError("expected a failure")


assert report("active") == ("literal_error", "the literal 'active'")
# `Literal[x]` with one member *is* that literal: a join of one member is the
# member, settled where the schema is built.
assert report(Literal["active"]) == ("literal_error", "the literal 'active'")
```

`Literal[...]` builds a union of its constants, and a union of several reports
`union_error`. A union of **one** is that one, folded where the schema is built,
so `Literal["active"]` is the literal leaf: it reports `literal_error` and
compares equal to `Validator("active")` under `==`.

Branch on the code you actually observe for the spelling you actually write.
