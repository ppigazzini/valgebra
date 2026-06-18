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
from valgebra import Validator

# bool is a subclass of int, so True and False are ints
assert Validator(int).is_valid(True)
# int does not subclass float, so an int is not a float
assert not Validator(float).is_valid(1)
assert Validator(float).is_valid(1.0)
```

## `Any` versus `object`

`object` is the **top** of the lattice (`anything`): every value. `Any` is the
gradual dynamic type — at runtime it also admits every value, but it is a
distinct atom that the [simplifier](algebra.md) never rewrites, preserving
"deliberately unchecked" as different from "checked: all admitted".

```python
from typing import Any

from valgebra import Validator

assert Validator(object).is_valid(["anything", 1, None])
assert Validator(Any).is_valid(object())
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
| `tuple[A, B, ...]` | a fixed prefix `A`, then zero or more `B` (see below) |

```python
from valgebra import Validator

assert Validator(list[int]).is_valid([1, 2, 3])
assert Validator(dict[str, int]).is_valid({"a": 1})
assert Validator(tuple[int, str]).is_valid((1, "a"))
assert Validator(tuple[int, ...]).is_valid((1, 2, 3))
assert Validator(tuple[str, int, ...]).is_valid(("x", 1, 2))
```

## Native forms

A native form exists only where standard typing **cannot** spell the set: the
list literal carries the sequence shapes typing has no syntax for. Everything a
typing annotation already expresses is written that way — `set[T]`, not `{T}`;
`tuple[A, B]`, not `(A, B)`; both literals are rejected with a message pointing
to the typing spelling.

| Native form | Denotes |
| --- | --- |
| `[T]` | `list[T]` — a homogeneous list (the single-element idiom) |
| `[T, ...]` | `list[T]` — homogeneous, written with the tail marker |
| `[A, B]` | a **fixed-length list**, matched positionally (`list[A, B]` is illegal typing) |
| `[A, B, ...]` | a fixed prefix, then a repeated tail (see below) |
| `{K: V}` | `dict[K, V]` |
| `{"key": T, "key2?": T}` | a **record** (see below) |
| any constant `c` | `Literal[c]` |

```python
from valgebra import Validator

assert Validator([int]).is_valid([1, 2])           # homogeneous list[int]
assert Validator([int, str]).is_valid([1, "a"])    # fixed-length list
assert not Validator([int, str]).is_valid([1])     # wrong length
assert Validator({str: int}).is_valid({"a": 1})    # dict[str, int]
assert Validator("active").is_valid("active")      # the literal "active"
```

A **fixed-length list** is matched positionally: element `i` must satisfy the
`i`th schema and the length must match. typing cannot spell it (`list[A, B]` is
illegal), which is the reason the list literal carries the shape; a fixed-length
*tuple* is the typing `tuple[A, B]`, and the container is part of the type, so a
list is never a member of the tuple form and vice versa.

### Prefix and repeated tail

A sequence schema is, in general, a **regular expression over element types**: a
fixed positional prefix followed by an optional repeated tail. A trailing `...`
repeats the element just before it, so `[T, ...]` (any number of `T`) is the
prefix-free case. The same shape is available for tuples with `tuple[A, B, ...]`;
the container is part of the type, so a tuple is never a member of the list form
and vice versa.

| Form | Denotes |
| --- | --- |
| `[A, B, ...]` | a list: an `A`, then zero or more `B` |
| `[T, T, ...]` | a non-empty list of `T` (at least one) |
| `tuple[A, B, ...]` | a tuple: an `A`, then zero or more `B` |

```python
from valgebra import Validator

prefixed = Validator([str, int, ...])  # a str, then zero or more ints
assert prefixed.is_valid(["x"])
assert prefixed.is_valid(["x", 1, 2])
assert not prefixed.is_valid([1])  # the prefix must be a str

non_empty = Validator([int, int, ...])  # at least one int
assert non_empty.is_valid([1])
assert not non_empty.is_valid([])

tup = Validator(tuple[str, int, ...])  # the same shape, as a tuple
assert tup.is_valid(("x", 1, 2))
assert not tup.is_valid(["x", 1, 2])  # a list is not a member of the tuple form
```

## Literals

`Literal[...]` denotes a typed singleton: a value is a member iff it has the
**same type** as the literal and is equal to it. The same-type rule keeps
`Literal[1]`, `Literal[True]`, and `Literal[1.0]` distinct, even though Python's
`==` conflates them:

```python
from typing import Literal

from valgebra import Validator

assert Validator(Literal[1]).is_valid(1)
assert not Validator(Literal[1]).is_valid(True)
assert not Validator(Literal[1]).is_valid(1.0)
```

## Unions and `Optional`

`X | Y` and `Optional[X]` denote the union of the member sets:

```python
from typing import Optional

from valgebra import Validator

assert Validator(int | str).is_valid("x")
assert Validator(Optional[int]).is_valid(None)
```

## Records

A dict literal with all-string keys is a **record**: named fields, closed by
default. A required field's key must be present with a matching value; a trailing
`?` on the key name marks it optional. A closed record admits no key outside the
declared names.

```python
from valgebra import Validator

user = Validator({"name": str, "age?": int})
assert user.is_valid({"name": "Ada"})              # optional key absent
assert user.is_valid({"name": "Ada", "age": 36})
assert not user.is_valid({"name": "Ada", "x": 1})  # closed: no extra keys
```

Open the record with `open` (undeclared keys admitted) or re-close it with
`close`:

```python
from valgebra import Validator

closed = Validator({"name": str})
assert not closed.is_valid({"name": "Ada", "extra": 1})
assert closed.open().is_valid({"name": "Ada", "extra": 1})
```

### Heterogeneous maps and catch-alls

A dict schema's string keys are named fields; any *other* key is a schema that
keys a default clause for the rest. One form therefore expresses records,
mappings, and their combination: several schema keys give a **heterogeneous map**
whose value type depends on which key schema matches, and named fields plus a
schema key give a record with a **typed catch-all**. Named fields take
precedence over the catch-all.

```python
from valgebra import Validator

# str keys map to ints, int keys map to strs
hetero = Validator({str: int, int: str})
assert hetero.is_valid({"a": 1, 2: "b"})
assert not hetero.is_valid({"a": "x"})  # a str key needs an int value

# a record whose every other key must be an int
extensible = Validator({"name": str, str: int})
assert extensible.is_valid({"name": "Ada", "age": 36})
assert not extensible.is_valid({"name": "Ada", "age": "old"})
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

from valgebra import Validator


class Color(enum.Enum):
    RED = 1
    GREEN = 2


@dataclass
class Point:
    x: int
    y: int


assert Validator(Color).is_valid(Color.RED)
assert Validator(Point).is_valid(Point(1, 2))
assert not Validator(Point).is_valid(Point(1, "y"))
```

!!! note "Recursive classes"
    A class whose own type appears in a field (a tree node, a linked list) is
    recursive and cannot compile directly — express it with
    [`recursive`](recursion.md), which ties the fixpoint explicitly.

!!! note "Bare classes, callables, and the runtime boundary"
    A bare class is an `isinstance` check: `Validator(complex)` admits any
    `complex`, and any user class admits its instances. `Callable` (and
    `Callable[...]`) checks only that the value is callable — the argument and
    return types cannot be inspected at runtime, so they are not enforced. `Any`
    is admitted unchecked. Everything else is decided structurally: a `list[int]`
    schema does check each element.

## Refinements

`Annotated[T, ...markers]` narrows `T` with constraints — bounds, lengths,
multiples, and predicates. See the [refinements guide](refinements.md).

## Stable repr

A compiled validator prints back as the annotation that produces it, which makes
schemas inspectable:

```python
from valgebra import Validator

assert repr(Validator(list[dict[str, int]])) == "list[dict[str, int]]"
assert repr(Validator({"name": str, "age?": int})) == "{'name': str, 'age?': int}"
```
