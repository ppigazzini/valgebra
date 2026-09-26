---
description: What ty, mypy and pyright read from a schema, the spellings they refuse, and the ruff settings a project that validates its own classes needs.
---

# Static checkers

What a static type checker reads from a validator, which schema spellings it
refuses and what to write instead, and the ruff settings a project that
validates its own classes needs. valgebra checks values at runtime and a
checker never runs it: the package's type stub is what the two share, and ty,
mypy and pyright read it alike except where a table below says otherwise.

## A validator's type is its set's

`Validator` is generic: `Validator[T]` is a validator whose members a checker
reads as `T`s. The constructor reads a class as the type it names, and a `True`
from `is_valid` narrows the value it was asked about:

```python
from dataclasses import dataclass
from typing import assert_type

from valgebra import Validator


@dataclass
class Point:
    x: int
    y: int


points = Validator(Point)
assert_type(points, Validator[Point])


def as_point(value: object) -> Point | None:
    if points.is_valid(value):
        return value  # a checker reads `value` as a `Point` here
    return None


assert as_point(Point(0, 0)) == Point(0, 0)
assert as_point((0, 0)) is None
```

What each spelling reads as:

| Schema | A checker reads the validator as |
| --- | --- |
| a class: `int`, a dataclass, `list[int]`, `tuple[int, str]`, a `TypedDict` | `Validator` of that type |
| a compiled validator | its own type |
| `None` | `Validator[None]` |
| a runtime-checkable `Protocol` | the protocol under ty and pyright; `object` under mypy |
| a `NewType` | the new type under mypy; `object` under ty and pyright |
| `int \| None`, `Optional[int]`, `Literal[...]`, `Annotated[...]` | `Validator[object]` |
| a dict or list literal, a constant, a value typed `object` | `Validator[object]` |
| `Any` | gradual: `Any` under mypy, `Unknown` under ty, `object` under pyright |
| `a \| b` for two typed validators | the union of their types |
| `union(a, b)` for validators of one type | that type; for two types ty reads their union, mypy and pyright `object` |
| `intersection`, `complement`, `recursive` | `Validator[object]` |

The rows that name a checker are held by a test that runs all three over one
fixture per row and compares what each reveals, so a checker release that
changes a reading changes this table rather than a caller's build.

**Where the set has no static type, the validator reads as `object`.** A meet,
a complement and a fixpoint have no static spelling. `int | None`, a `Literal`
and an `Annotated` form have one, but they are type expressions rather than
classes, and a stub reads them only through PEP 747's `TypeForm`. The stub does
not use it yet: ty reports an error at a `TypeForm` overload for any argument
that is not a type expression -- a schema held in a variable typed `object`
among them -- where the typing spec's overload rules say to try the next one.
On a `Validator[object]`, `is_valid` answers a plain `bool` and `ensure`
returns its argument as the argument's own type.

## A `True` narrows, and a `False` does not

`is_valid` is a `TypeGuard`, so it narrows only when it answers `True`. A
refusal is no evidence that a value lies outside the static type, because the
static type is often wider than the set:

```python
from valgebra import Validator

assert not Validator(float).is_valid(1)  # a checker's `float` admits `1`
assert not Validator(complex).is_valid(1.0)  # and its `complex` admits `1.0`
```

A `TypeIs` would let a checker conclude from each `False` above that the value
is not a `float`, or not a `complex`, and read the code that runs next as
unreachable. The same holds wherever the check reads more than the class: a
dataclass instance whose field holds a value outside the field's type is
refused, and is still an instance. On a typed validator `ensure` and `load`
return the type:
`Validator(Point).ensure(value)` is a `Point`, and a `TypedDict`'s validator
loads a document as that `TypedDict`.

## Write a parameter bare

A parameter that takes any validator is written `Validator`, which is
`Validator[Any]`:

```python
from valgebra import Validator


def admits(schema: Validator, value: object) -> bool:
    return schema.is_valid(value)


assert admits(Validator(int), 1)
assert not admits(Validator({"name": str}), {})
```

`Validator[object]` takes only a validator whose set reads as `object`. The
parameter is invariant, because that is what lets the stub answer a `bool` on an
untyped validator and a narrowing on a typed one, so a `Validator[int]` is not
a `Validator[object]`. At runtime `Validator[int]` is a `types.GenericAlias`, so
an annotation that is evaluated finds it; it is not a schema, and
`Validator(Validator[int])` is refused.

## Spellings a checker refuses, and what to write instead

Each pair builds one schema, and the second spelling is the one ty, mypy and
pyright accept.

| Written | Why a checker refuses it | Write instead |
| --- | --- | --- |
| `tuple[str, int, ...]` | `...` is legal only as the second of two arguments | `tuple[str, *tuple[int, ...]]` (Python 3.11+) |
| `Literal[1.5]`, `Literal[(1, 2)]`, `Literal[obj]` | `Literal` takes an `int`, `str`, `bytes` or `bool` value, an enum member or `None` | the constant itself: `Validator(1.5)`, `union(1.5, "a")` |
| `list[v]`, `dict[str, v]` for a validator `v` | a variable is not a type expression | the native forms `[v]` and `{str: v}` |
| `{"name": str}` where a type is expected | a dict literal is not a type expression | a `TypedDict` class |
| `tuple[int, v]` for a validator `v` | a variable is not a type expression | none; the line takes the checker's own ignore comment |

```python
from valgebra import Validator, union

v = Validator(int)
assert Validator(tuple[str, *tuple[int, ...]]).is_valid(("x", 1, 2))
assert union(1.5, "a").is_valid(1.5)
assert Validator([v]).is_valid([1, 2])
assert Validator({str: v}).is_valid({"a": 1})
```

## The ruff settings a validated class needs

```toml
[tool.ruff.lint.flake8-type-checking]
runtime-evaluated-decorators = ["dataclasses.dataclass"]
runtime-evaluated-base-classes = [
    "typing.TypedDict",
    "typing.NamedTuple",
    "typing_extensions.TypedDict",
    "typing_extensions.NamedTuple",
]

[tool.ruff.lint.flake8-bugbear]
extend-immutable-calls = [
    "valgebra.Validator",
    "valgebra.union",
    "valgebra.intersection",
    "valgebra.complement",
    "valgebra.recursive",
]
```

**The first table keeps a field's class importable when the validator is
built.** A dataclass, `TypedDict` or `NamedTuple` is read through
`typing.get_type_hints` at `Validator(...)`, so every class its fields name has
to be importable then. ruff's `TC001` to `TC003` move an import that only an
annotation uses under `if TYPE_CHECKING:`, and in a module with
`from __future__ import annotations` the `Validator(...)` call then raises
`NameError`. Naming the decorator and the base classes whose annotations are
read at runtime keeps those imports where they are; `typing_extensions`'
spellings are separate names and need their own entries. On Python 3.15 ruff
makes such an import lazy instead, and the frontend resolves a lazily imported
class.

**The second table accepts a validator as a shared default.** `B008` and
`RUF009` flag a call written as a function's default argument or a dataclass
field's default, because the value is built once and shared. A validator is
immutable and hashable, so one built there is the intended shared value.

## What this does not cover

- **A checker's type is an upper bound, not the set.** `Validator(Point)` reads
  as `Point` and admits only the points whose fields are members too; nothing
  here makes a checker run the validator.
- **Each checker's own configuration**, and the rules it leaves off by default,
  are the checker's documentation.
