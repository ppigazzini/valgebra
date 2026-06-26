---
description: Guided learning path from a scalar schema to an inspected failure.
---

# Tutorial

This is a guided first session with valgebra. You will start from a single
scalar check and finish having validated a structured record — with a
constraint, an optional field, a real failure you inspect, and a JSON document —
building each step on the last. Every snippet runs as written; copy them in
order.

It assumes valgebra is [installed](installation.md). For the meaning of each
form you meet here, the [schema language](schema-language.md) is the reference;
this page is the path, not the catalogue.

## 1. Your first validator

A schema denotes a *set of Python values*, and validating asks whether a value
is in that set. Compile one with `Validator`, then ask with `is_valid`:

```python
from valgebra import Validator

is_int = Validator(int)
assert is_int.is_valid(42)
assert not is_int.is_valid("42")  # a str is not in the set of ints
```

`Validator(int)` compiles once; call `is_valid` as often as you like.

## 2. Failing loudly

`is_valid` returns a bool. When you want an exception instead, use `validate` —
it raises `ValidationError` on a value outside the set:

```python
from valgebra import ValidationError, Validator

is_int = Validator(int)
raised = False
try:
    is_int.validate("42")
except ValidationError:
    raised = True
assert raised
```

You will read what a `ValidationError` carries in step 6.

## 3. Validating a collection

Schemas nest. A `list[int]` is the set of lists whose every element is an `int`,
and valgebra checks each element:

```python
from valgebra import Validator

numbers = Validator(list[int])
assert numbers.is_valid([1, 2, 3])
assert not numbers.is_valid([1, "two", 3])  # one bad element fails the list
```

## 4. Describing a record

Real data is usually structured. Write a record as a `TypedDict` — the standard
typing form — and valgebra validates the shape and every field:

```python
from typing import TypedDict

from valgebra import Validator


class User(TypedDict):
    name: str
    age: int


users = Validator(User)
assert users.is_valid({"name": "Ada", "age": 36})
assert not users.is_valid({"name": "Ada", "age": "old"})  # age must be an int
```

## 5. Adding a constraint

Plain types say *what kind*; a refinement says *which values*. Narrow a field
with `Annotated` and an [annotated-types](refinements.md) marker — here, an age
that cannot be negative:

```python
from typing import Annotated, TypedDict

import annotated_types as at

from valgebra import Validator


class User(TypedDict):
    name: str
    age: Annotated[int, at.Ge(0)]


users = Validator(User)
assert users.is_valid({"name": "Ada", "age": 36})
assert not users.is_valid({"name": "Ada", "age": -1})  # the bound holds
```

## 6. Optional fields, and reading a failure

A dict literal is a compact record: a trailing `?` on a key marks it optional,
and the record is *closed* — an undeclared key is rejected.

```python
from valgebra import Validator

profile = Validator({"name": str, "age?": int})
assert profile.is_valid({"name": "Ada"})               # age omitted: fine
assert profile.is_valid({"name": "Ada", "age": 36})
assert not profile.is_valid({"name": "Ada", "extra": 1})  # closed: no extra keys
```

When a check fails, `validate` raises a `ValidationError` that tells you exactly
*what* failed and *where* — a machine-readable `code` and the `path` to the
offending value, even deep in a nested structure:

```python
from valgebra import ValidationError, Validator

schema = Validator({"user": {"name": str}})
try:
    schema.validate({"user": {"name": 5}})
except ValidationError as err:
    assert err.code == "string_type"
    assert err.path == ("user", "name")
```

The [error model](error-model.md) covers aggregation and union reporting.

## 7. Validating JSON

You do not have to parse first. `validate_json` and `is_valid_json` read JSON on
the Rust path and run the very same checks, so a document is judged exactly as
`json.loads` of it would be:

```python
from valgebra import Validator

users = Validator({"name": str, "age?": int})
users.validate_json('{"name": "Ada", "age": 36}')        # passes, raises nothing
assert Validator(list[int]).is_valid_json("[1, 2, 3]")   # str or bytes
```

## Where to go next

You can now check scalars, collections, records, constraints, failures, and
JSON. From here:

- The [schema language](schema-language.md) lists every form and the set it
  denotes.
- [Refinements](refinements.md) covers the full constraint vocabulary.
- The [Boolean algebra](algebra.md) composes schemas with `union`, `intersection`,
  and `complement`, and the [foundations](foundations.md) explain the theory.
- [Recursive schemas](recursion.md) handle trees and linked structures with
  `recursive`.
