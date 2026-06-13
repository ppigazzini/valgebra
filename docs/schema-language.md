# Schema language

A schema denotes a **set of Python values**. This page lists every form valgebra
reads and the set it denotes. The primary notation is standard typing; compact
native forms and the combinators are alternatives for the same sets.

## Scalars

| Schema | Denotes |
| --- | --- |
| `int` | every `int` instance |
| `float` | every `float` instance |
| `str` | every `str` instance |
| `bytes` | every `bytes` instance |
| `bool` | `{True, False}` |
| `None` | `{None}` |

The set relationships follow Python's own, exactly:

```python
from valgebra import validator

# bool is a subclass of int, so True and False are ints
assert validator(int).is_valid(True)
# int does not subclass float, so an int is not a float
assert not validator(float).is_valid(1)
assert validator(float).is_valid(1.0)
```

## `Any` versus `object`

`object` is the **top** of the lattice (`anything`): every value. `Any` is the
gradual dynamic type — at runtime it also admits every value, but it is a
distinct atom that the [simplifier](algebra.md) never rewrites, preserving
"deliberately unchecked" as different from "checked: all admitted".

```python
from typing import Any

from valgebra import validator

assert validator(object).is_valid(["anything", 1, None])
assert validator(Any).is_valid(object())
```

## Collections

| Schema | Denotes |
| --- | --- |
| `list[T]` | lists whose every element is in `T` |
| `set[T]` | sets whose every element is in `T` |
| `frozenset[T]` | frozensets whose every element is in `T` |
| `dict[K, V]` | dicts whose keys are in `K` and values in `V` |
| `tuple[A, B]` | length-2 tuples with `A` then `B` |
| `tuple[T, ...]` | tuples of any length, every element in `T` |

```python
from valgebra import validator

assert validator(list[int]).is_valid([1, 2, 3])
assert validator(dict[str, int]).is_valid({"a": 1})
assert validator(tuple[int, str]).is_valid((1, "a"))
assert validator(tuple[int, ...]).is_valid((1, 2, 3))
```

## Native forms

Compact literals build the same nodes without importing typing:

| Native form | Equivalent |
| --- | --- |
| `[T]` | `list[T]` |
| `[T, ...]` | `list[T]` |
| `{T}` | `set[T]` |
| `{KeyType: ValueType}` | `dict[KeyType, ValueType]` |
| `{"key": T, "key2?": T}` | a **record** (see below) |
| any constant `c` | `Literal[c]` |

```python
from valgebra import validator

assert validator([int]).is_valid([1, 2])           # list[int]
assert validator({int}).is_valid({1, 2})           # set[int]
assert validator({str: int}).is_valid({"a": 1})    # dict[str, int]
assert validator("active").is_valid("active")      # the literal "active"
```

A **fixed-length list** matched positionally is built with `fixed_sequence`
(see the [API reference](api.md)):

```python
from valgebra import fixed_sequence

pair = fixed_sequence(int, str)
assert pair.is_valid([1, "a"])
assert not pair.is_valid([1, 2])
assert not pair.is_valid([1])  # wrong length
```

## Literals

`Literal[...]` denotes a typed singleton: a value is a member iff it has the
**same type** as the literal and is equal to it. The same-type rule keeps
`Literal[1]`, `Literal[True]`, and `Literal[1.0]` distinct, even though Python's
`==` conflates them:

```python
from typing import Literal

from valgebra import validator

assert validator(Literal[1]).is_valid(1)
assert not validator(Literal[1]).is_valid(True)
assert not validator(Literal[1]).is_valid(1.0)
```

## Unions and `Optional`

`X | Y` and `Optional[X]` denote the union of the member sets:

```python
from typing import Optional

from valgebra import validator

assert validator(int | str).is_valid("x")
assert validator(Optional[int]).is_valid(None)
```

## Records

A dict literal with all-string keys is a **record**: named fields, closed by
default. A required field's key must be present with a matching value; a trailing
`?` on the key name marks it optional. A closed record admits no key outside the
declared names.

```python
from valgebra import validator

user = validator({"name": str, "age?": int})
assert user.is_valid({"name": "Ada"})              # optional key absent
assert user.is_valid({"name": "Ada", "age": 36})
assert not user.is_valid({"name": "Ada", "x": 1})  # closed: no extra keys
```

Open the record with `lax` (undeclared keys admitted) or re-close it with
`strict`:

```python
from valgebra import lax, validator

closed = validator({"name": str})
assert not closed.is_valid({"name": "Ada", "extra": 1})
assert lax(closed).is_valid({"name": "Ada", "extra": 1})
```

## Classes

| Form | How it validates |
| --- | --- |
| `TypedDict` | a record; required keys from the class, `Required`/`NotRequired` honored |
| dataclass | `isinstance` plus a deep check of each field |
| `NamedTuple` | `isinstance` plus a deep check of each field |
| `Enum` | an instance of the enumeration (any member) |
| runtime-checkable `Protocol` | `isinstance` against the protocol |
| `NewType` | validates the supertype it wraps |
| PEP 695 `type` alias | validates the aliased type |

```python
import enum
from dataclasses import dataclass

from valgebra import validator


class Color(enum.Enum):
    RED = 1
    GREEN = 2


@dataclass
class Point:
    x: int
    y: int


assert validator(Color).is_valid(Color.RED)
assert validator(Point).is_valid(Point(1, 2))
assert not validator(Point).is_valid(Point(1, "y"))
```

!!! note "Recursive classes"
    A class whose own type appears in a field (a tree node, a linked list) is
    recursive and cannot compile directly — express it with
    [`lazy`](recursion.md), which ties the fixpoint explicitly.

## Refinements

`Annotated[T, ...markers]` narrows `T` with constraints — bounds, lengths,
multiples, and predicates. See the [refinements guide](refinements.md).

## Stable repr

A compiled validator prints back as the annotation that produces it, which makes
schemas inspectable:

```python
from valgebra import validator

assert repr(validator(list[dict[str, int]])) == "list[dict[str, int]]"
assert repr(validator({"name": str, "age?": int})) == "{'name': str, 'age?': int}"
```
