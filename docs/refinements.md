---
description: Constraints and predicate refinements over a base schema.
---

# Refinements

A refinement narrows a base type to the subset satisfying one or more
constraints. Write it with `Annotated[T, ...markers]`; the base `T` is checked
first, then each constraint. valgebra reads the
[annotated-types](https://pypi.org/project/annotated-types/) markers
structurally, so it has no runtime dependency on that library.

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

adult = Validator(Annotated[int, at.Ge(18), at.Le(150)])
assert adult.is_valid(21)
assert not adult.is_valid(5)
```

Refinements built from bound and length markers also take part in the
[decision procedure](decidability.md): a refinement is a subtype of its base and
of a looser refinement, and a bound conjunction that cannot be satisfied is
detected as empty.

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

assert Validator(Annotated[int, at.Ge(0)]).is_subtype_of(int)  # refinement <= base
assert Validator(Annotated[int, at.Ge(0), at.Le(10)]).is_subtype_of(
    Annotated[int, at.Ge(0)]  # a tighter bound is a subtype of a looser one
)
assert Validator(Annotated[int, at.Ge(10), at.Le(0)]).is_empty()  # no such int
```

A predicate marker is checked at validation time but stays opaque to subtyping
and emptiness — its satisfiability is undecidable in general.

## Supported markers

| Marker | Constraint | Failure code |
| --- | --- | --- |
| `Ge(n)` | `value >= n` | `greater_than_equal` |
| `Gt(n)` | `value > n` | `greater_than` |
| `Le(n)` | `value <= n` | `less_than_equal` |
| `Lt(n)` | `value < n` | `less_than` |
| `MinLen(n)` | `len(value) >= n` | `too_short` |
| `MaxLen(n)` | `len(value) <= n` | `too_long` |
| `MultipleOf(n)` | `value % n == 0` | `multiple_of` |
| `Regex(p)` | the string fully matches the regex `p` | `string_pattern_mismatch` |
| `Predicate(f)` | `f(value)` is truthy | `predicate_failed` |

`Regex` is valgebra's own marker (`from valgebra import Regex`), since
`annotated-types` defines none for strings. The match is **anchored** — the whole
string must match, like `re.fullmatch` — and runs natively in Rust with a
linear-time engine (no catastrophic backtracking), so unlike a `Predicate` it
stays on the fast path and never crosses into Python per value. An invalid
pattern is rejected when the validator is built, not at first use. A compiled
`re.Pattern` works as metadata too:

```python
import re
from typing import Annotated

from valgebra import Regex, Validator

oid = Validator(Annotated[str, Regex(r"[0-9a-f]{24}")])
assert oid.is_valid("0123456789abcdef01234567")
assert not oid.is_valid("0123456789abcdef0123456X")  # not hex
assert not oid.is_valid("0123")  # not the full 24 characters

assert Validator(Annotated[str, re.compile(r"\d+")]).is_valid("123")
```

`MultipleOf(n)` requires a nonzero divisor: no value is a multiple of zero, so
`MultipleOf(0)` is an unsatisfiable constraint and is rejected with a `ValueError`
when the validator is built, rather than rejecting every value at check time.

The compound markers `Interval` and `Len` expand to the bounds they carry, so
`Interval(ge=0, le=10)` contributes `Ge(0)` and `Le(10)`, and `Len(2, 4)`
contributes `MinLen(2)` and `MaxLen(4)`:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

assert Validator(Annotated[int, at.Interval(ge=0, le=10)]).is_valid(5)
assert not Validator(Annotated[int, at.Interval(ge=0, le=10)]).is_valid(11)
assert Validator(Annotated[str, at.Len(2, 4)]).is_valid("abc")
assert not Validator(Annotated[str, at.Len(2, 4)]).is_valid("a")

assert Validator(Annotated[int, at.MultipleOf(3)]).is_valid(9)
assert not Validator(Annotated[int, at.MultipleOf(3)]).is_valid(5)
```

## Predicates: the slow path

A `Predicate` runs an arbitrary Python callable. It is the one *refinement*
constraint that leaves Rust for a caller's own code — literals, instance and
attribute checks, and comparison bounds also compare against Python objects, but
against fixed operators, not arbitrary callables — so it is a **documented slow
path**, never a silent fallback. Use it for checks the markers cannot express:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

even = Validator(Annotated[int, at.Predicate(lambda x: x % 2 == 0)])
assert even.is_valid(4)
assert not even.is_valid(3)
```

A predicate that *raises* is reported distinctly, as `predicate_error` rather
than an ordinary failure, so a buggy predicate is not mistaken for a rejected
value.

## On classes

Refinements declared on a `TypedDict`, dataclass, or `NamedTuple` field are
enforced — the constraint travels with the field:

```python
from typing import Annotated, TypedDict

import annotated_types as at

from valgebra import Validator


class Account(TypedDict):
    balance: Annotated[int, at.Ge(0)]


assert Validator(Account).is_valid({"balance": 100})
assert not Validator(Account).is_valid({"balance": -1})
```

## Unrecognized markers

Per the typing spec, metadata valgebra does not recognize as a constraint is
ignored — so non-constraint `Annotated` metadata (documentation strings, unit
markers) is harmless and carries no membership meaning.

```python
from typing import Annotated

from valgebra import Validator

assert repr(Validator(Annotated[int, "a documentation note"])) == "int"
```
