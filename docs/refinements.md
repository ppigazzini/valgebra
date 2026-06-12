# Refinements

A refinement narrows a base type to the subset satisfying one or more
constraints. Write it with `Annotated[T, ...markers]`; the base `T` is checked
first, then each constraint. valgebra reads the
[annotated-types](https://pypi.org/project/annotated-types/) markers
structurally, so it has no runtime dependency on that library.

```python
from typing import Annotated

import annotated_types as at

from valgebra import validator

adult = validator(Annotated[int, at.Ge(18), at.Le(150)])
assert adult.is_valid(21)
assert not adult.is_valid(5)
```

## Supported markers

| Marker | Constraint | Failure code |
| --- | --- | --- |
| `Ge(n)` | `value >= n` | `greater_than_equal` |
| `Gt(n)` | `value > n` | `greater_than` |
| `Le(n)` | `value <= n` | `less_than_equal` |
| `Lt(n)` | `value < n` | `less_than` |
| `MinLen(n)` | `len(value) >= n` | `too_short` |
| `MaxLen(n)` | `len(value) <= n` | `too_long` |
| `MultipleOf(n)` | `value % n == 0` | `not_multiple_of` |
| `Predicate(f)` | `f(value)` is truthy | `predicate_failed` |

The compound markers `Interval` and `Len` expand to the bounds they carry, so
`Interval(ge=0, le=10)` contributes `Ge(0)` and `Le(10)`, and `Len(2, 4)`
contributes `MinLen(2)` and `MaxLen(4)`:

```python
from typing import Annotated

import annotated_types as at

from valgebra import validator

assert validator(Annotated[int, at.Interval(ge=0, le=10)]).is_valid(5)
assert not validator(Annotated[int, at.Interval(ge=0, le=10)]).is_valid(11)
assert validator(Annotated[str, at.Len(2, 4)]).is_valid("abc")
assert not validator(Annotated[str, at.Len(2, 4)]).is_valid("a")

assert validator(Annotated[int, at.MultipleOf(3)]).is_valid(9)
assert not validator(Annotated[int, at.MultipleOf(3)]).is_valid(5)
```

## Predicates: the slow path

A `Predicate` runs an arbitrary Python callable. It is the one place validation
leaves Rust, so it is a **documented slow path**, never a silent fallback. Use it
for checks the markers cannot express:

```python
from typing import Annotated

import annotated_types as at

from valgebra import validator

even = validator(Annotated[int, at.Predicate(lambda x: x % 2 == 0)])
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

from valgebra import validator


class Account(TypedDict):
    balance: Annotated[int, at.Ge(0)]


assert validator(Account).is_valid({"balance": 100})
assert not validator(Account).is_valid({"balance": -1})
```

## Unrecognized markers

Per the typing spec, metadata valgebra does not recognize as a constraint is
ignored — so non-constraint `Annotated` metadata (documentation strings, unit
markers) is harmless and carries no membership meaning.

```python
from typing import Annotated

from valgebra import validator

assert repr(validator(Annotated[int, "a documentation note"])) == "int"
```
