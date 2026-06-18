# Quickstart

This is the condensed tour — a one-line scalar check through a refined
`TypedDict` to a composed contract — for a reader who wants valgebra at a glance.
For a guided, step-by-step path, start with the [tutorial](tutorial.md) instead.
Every snippet runs as written.

## Compile once, check many

`Validator(schema)` compiles a schema into an immutable validator. Reuse it:

```python
from valgebra import Validator

is_int = Validator(int)
assert is_int.is_valid(42)
assert not is_int.is_valid("42")
```

A validator has three entry points:

- `is_valid(obj)` returns a `bool` (the fast path).
- `validate(obj)` returns `None` or raises `ValidationError`.
- `ensure(obj)` validates and returns the object unchanged (the explicit,
  separate value-returning mode — validation is a membership check, so there is
  nothing to convert).

## Schemas are standard annotations

The primary notation is the typing you already write:

```python
from typing import Literal

from valgebra import Validator

assert Validator(list[int]).is_valid([1, 2, 3])
assert Validator(dict[str, int]).is_valid({"a": 1})
assert Validator(tuple[int, ...]).is_valid((1, 2, 3))
assert Validator(int | None).is_valid(None)
assert Validator(Literal["red", "green"]).is_valid("red")
```

## Refinements with `Annotated`

Constraints attach to a type with `Annotated` and the
[annotated-types](https://pypi.org/project/annotated-types/) markers:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

adult = Validator(Annotated[int, at.Ge(18), at.Le(150)])
assert adult.is_valid(21)
assert not adult.is_valid(5)
```

## Classes compile too

`TypedDict`, dataclasses, `NamedTuple`, enums, and runtime-checkable protocols
all compile, and refinements on their fields are enforced:

```python
from typing import Annotated, TypedDict

import annotated_types as at

from valgebra import Validator


class User(TypedDict):
    name: str
    age: Annotated[int, at.Ge(0)]


users = Validator(User)
assert users.is_valid({"name": "Ada", "age": 36})
assert not users.is_valid({"name": "Ada", "age": -1})
```

## Compose with the algebra

Any schema combines with `union`, `intersection`, and `complement`:

```python
from valgebra import complement, intersection, Validator

# an int that is not a bool
strict_int = intersection(int, complement(bool))
assert strict_int.is_valid(5)
assert not strict_int.is_valid(True)
```

## Handle failures

A failure raises `ValidationError` carrying a machine-readable `code`, the
`path` to the offending value, and a summary:

```python
from valgebra import ValidationError, Validator

try:
    Validator({"user": {"name": str}}).validate({"user": {"name": 5}})
except ValidationError as err:
    assert err.code == "string_type"
    assert err.path == ("user", "name")
```

## Validate JSON directly

`validate_json` parses and checks JSON on the Rust path, reaching the same
decision as validating the parsed object:

```python
from valgebra import Validator

users = Validator({"name": str, "age?": int})
users.validate_json('{"name": "Ada", "age": 36}')
assert Validator(list[int]).is_valid_json(b"[1, 2, 3]")
```
