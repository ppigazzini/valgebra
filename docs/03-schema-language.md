---
description: Every schema form with its denotation as a set of Python values.
---

# Schema language

A schema denotes a **set of Python values**. This page lists every form valgebra
reads and the set it denotes. The primary notation is standard typing; compact
native forms and the combinators are alternatives for the same sets.

## A form this page does not list

The list is exhaustive, so a typing form absent from it is one valgebra does not
read. It raises `NotImplementedError` when the validator is **built**, not when a
value arrives — a schema that cannot be read never becomes one that quietly
admits everything.

```python
from collections.abc import Mapping

from valgebra import Validator

try:
    Validator(Mapping[str, int])
except NotImplementedError as error:
    assert "unsupported typing form" in str(error)
```

The abstract-collection generics are the ones a reader is most likely to
reach for: `Mapping[K, V]`, `Sequence[T]`, `Set[T]`, `Iterable[T]`, and the
subscripted concrete classes such as `deque[T]` and `OrderedDict[K, V]`. These
are **not built**. Whether they should be is open; nothing here rules them out.

`type[T]` is refused as well, and is a different question: it constrains a value
that is itself a class, where every form above constrains a container's
contents.

Where the value really is a `dict` or a `list`, the builtin form denotes the set
you want: `dict[K, V]` for `Mapping[K, V]`, `list[T]` for `Sequence[T]`. Where it
is not — a `UserDict`, a `deque` — no form here denotes it, and a
[predicate refinement](05-refinements.md) over the class is the available
expression.

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
distinct atom that the [simplifier](04-algebra.md) never rewrites, preserving
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
| `tuple[A, *tuple[B, ...]]` | the same, spelled by unpacking (3.11+) |

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

assert Validator([int]).is_valid([1, 2])  # homogeneous list[int]
assert Validator([int, str]).is_valid([1, "a"])  # fixed-length list
assert not Validator([int, str]).is_valid([1])  # wrong length
assert Validator({str: int}).is_valid({"a": 1})  # dict[str, int]
assert Validator("active").is_valid("active")  # the literal "active"
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
| `tuple[A, *tuple[B, ...]]` | the same tuple, spelled by unpacking (3.11+) |

An **unpacked** variadic tuple says the prefix-and-tail shape the way PEP 646
spells it, and `Unpack[tuple[B, ...]]` is the same thing written out. An unpacked
*fixed* tuple splices its elements in, so `tuple[A, *tuple[B, C]]` is
`tuple[A, B, C]`. What a sequence cannot carry is an element **after** the
repeating tail — `tuple[*tuple[int, ...], str]` names a set this algebra does not
spell — so that form is rejected rather than read as something else.

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

unpacked = Validator(tuple[str, *tuple[int, ...]])  # the PEP 646 spelling
assert unpacked.is_valid(("x", 1, 2))
assert repr(unpacked) == "tuple[str, int, ...]"
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

### A string inside a generic is a forward reference, and is refused

The fallback reads a bare value as a literal, and that reading stops at the
argument of a typing form. `list["Account"]` is a **forward reference** to a type
named `Account`, which the typing spec resolves against the namespace the
annotation was written in — a namespace valgebra does not have, because it is
handed the runtime object rather than the source. Reading the string as a literal
instead would build a list of the *word* `"Account"`, a schema that refuses what
the annotation admits, so the position is refused:

```python
from valgebra import Validator

try:
    Validator(list["Account"])
except NotImplementedError as error:
    assert "forward reference" in str(error)

# Say the type, or say the value.
assert Validator(list[int]).is_valid([1])
assert Validator(["active"]).is_valid(["active"])  # a list of that literal
```

A class's own annotations are a different matter: `Validator(SomeClass)` resolves
their strings for you (see [classes](#classes)).

### Anything unrecognized is a literal

The literal form is also the **fallback**: an object the frontend does not read
as one of the forms above becomes `Literal[that object]`, so `Validator(x)`
denotes `{x}` for any `x` valgebra has no other reading for. That is what makes
`Validator("active")` mean the string rather than an error, and it applies to a
function, a module or an instance just the same:

```python
from valgebra import Validator


def positive(value):
    return value > 0


schema = Validator(positive)
assert repr(schema).startswith("Literal[")
assert schema.is_valid(positive)  # the function object itself
assert not schema.is_valid(1)  # not a predicate: 1 is not that function
```

A callable is the case worth naming, because the same callable **is** a
predicate one position inward, as `Annotated` metadata — see
[refinements](05-refinements.md#a-bare-callable-is-metadata-only). At the top
level there is no base for it to narrow, so the fallback applies and the schema
denotes the single function object.

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
assert user.is_valid({"name": "Ada"})  # optional key absent
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

Both apply to every record that **declares at least one field**, at any depth,
including inside a recursive definition. They are projections rather than
inverses: `open` on such a record widens any typed catch-all it carries to admit
every key, and `close` drops the catch-all, so applying either twice changes
nothing the second time.

A dict schema that declares no named field is a **mapping**, not a record, and
neither transform rewrites it — only the schemas inside its clauses are visited.
That covers `dict[K, V]` and the empty schema `{}`:

```python
from valgebra import Validator

assert repr(Validator({str: int}).open()) == "dict[str, int]"  # unchanged
assert repr(Validator({}).open()) == "{}"  # unchanged

# The same clause is widened once a named field sits beside it.
assert repr(Validator({"name": str, str: int}).open()) == "{'name': str, ...}"
```

So whether a typed catch-all is freed depends on whether the schema also
declares a field. Write the key set you want rather than reaching for `open` on a
mapping.

#### Under a `complement`, the direction reverses

A record reached through a `complement` is rewritten like any other. Opening it
makes the complemented set larger, which makes the complement — and so the whole
schema — **smaller**:

```python
from valgebra import Validator, anything, complement

not_a_k_record = complement(Validator({"k": anything}))
schema = Validator({"x": not_a_k_record})
value = {"x": {"k": 1, "z": 2}}

# `{"k": 1, "z": 2}` is not a closed `{k: …}` record, so it is in the complement.
assert schema.is_valid(value)

# Opening admits it to the inner record, so it leaves the complement.
assert not schema.open().is_valid(value)
assert repr(schema.open()) == "{'x': complement({'k': anything, ...}), ...}"
```

`close` reverses under a complement for the same reason: it narrows the inner
record, which widens the schema.

So "open admits more" holds of the record the transform rewrites, and of the
whole schema only where no complement stands between them. Under a negation the
two swap. The practical consequence is that adding or removing an `.open()`
**inside** a complement changes the schema in the direction opposite to the one
the name suggests, and the change is invisible from outside unless you check —
`repr` shows it, as above.

### Heterogeneous maps and catch-alls

A dict schema's string keys are named fields; any *other* key is a schema that
keys a default clause for the rest. One form therefore expresses records,
mappings, and their combination: several schema keys give a **heterogeneous map**
whose value type depends on which key schema matches, and named fields plus a
schema key give a record with a **typed catch-all**. Named fields take
precedence over the catch-all.

Clauses are a disjunction, not a precedence list: a key that is not a named field
is admitted when **some** clause matches both it and its value. So overlapping
key schemas widen what the map admits rather than the earlier one winning, and
writing them in a different order does not change the schema's meaning.

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

### Constraining some keys and freeing the rest

Because the clauses are a disjunction, a clause that matches every key subsumes
every narrower one. That is what [`open`](#records) does, and it is why opening
a map with a typed catch-all frees the keys that catch-all was constraining.

To leave *only* the unclaimed keys free, give the permissive clause the
**complement** of the keys the others claim. Disjoint clauses cannot widen each
other:

```python
from valgebra import Validator, anything, complement

# str keys must be ints; any key that is not a str is unconstrained.
partly_open = Validator({"name": str, str: int, complement(Validator(str)): anything})
assert partly_open.is_valid({"name": "Ada", "age": 36})
assert partly_open.is_valid({"name": "Ada", 7: object()})  # no clause claims it
assert not partly_open.is_valid({"name": "Ada", "age": "old"})

# `open` is the other thing: every key becomes free, the typed clause included.
fully_open = Validator({"name": str, str: int}).open()
assert fully_open.is_valid({"name": "Ada", "age": "old"})
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

### Pass the class, not its annotations

Give `Validator` the class itself. It reads the annotations, resolves the string
forms, and keeps the `Annotated` metadata — which is where every refinement
lives.

Extracting the hints yourself and passing the mapping is the path that goes
wrong, because `typing.get_type_hints` **drops `Annotated` metadata by default**.
The schema still builds and still validates; it has quietly lost its
constraints, and nothing raises:

```python
from typing import Annotated, TypedDict, get_type_hints

import annotated_types as at

from valgebra import Validator


class Account(TypedDict):
    balance: Annotated[int, at.Ge(0)]


assert not Validator(Account).is_valid({"balance": -5})  # the class: constrained

stripped = get_type_hints(Account)  # {'balance': <class 'int'>}
assert Validator(stripped).is_valid({"balance": -5})  # the bound is gone

kept = get_type_hints(Account, include_extras=True)
assert not Validator(kept).is_valid({"balance": -5})  # include_extras keeps it
```

If you must derive the hints — building a schema for a class chosen at runtime,
say — pass `include_extras=True`. Passing the class is the supported path and has
no such footgun.

!!! note "Recursive classes"
    A class whose own type appears in a field (a tree node, a linked list) is
    recursive and cannot compile directly — express it with
    [`recursive`](06-recursion.md), which ties the fixpoint explicitly.

!!! note "Bare classes, callables, and the runtime boundary"
    A bare class is an `isinstance` check: `Validator(complex)` admits any
    `complex`, and any user class admits its instances. `Callable` (and
    `Callable[...]`) checks only that the value is callable — the argument and
    return types cannot be inspected at runtime, so they are not enforced. `Any`
    is admitted unchecked. Everything else is decided structurally: a `list[int]`
    schema does check each element.

## Refinements

`Annotated[T, ...markers]` narrows `T` with constraints — bounds, lengths,
multiples, and predicates. See the [refinements guide](05-refinements.md).

## Stable repr

A compiled validator prints back as the annotation that produces it, which makes
schemas inspectable:

```python
from valgebra import Validator

assert repr(Validator(list[dict[str, int]])) == "list[dict[str, int]]"
assert repr(Validator({"name": str, "age?": int})) == "{'name': str, 'age?': int}"
```
