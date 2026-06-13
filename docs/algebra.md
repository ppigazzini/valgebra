# The Boolean algebra

Union, intersection, and complement compose any schema — annotations, native
forms, or other compiled validators — into a complete, lawful Boolean lattice.
`anything` is the top (every value) and `nothing` is the bottom (no value).

```python
from valgebra import anything, complement, intersect, nothing, union

assert union(int, str).is_valid("x")                  # a value in either set
assert intersect(int, complement(bool)).is_valid(5)   # ints that are not bools
assert not intersect(int, complement(bool)).is_valid(True)
assert complement(nothing).is_valid(5)                # the complement of bottom is top
assert not nothing.is_valid(5)                        # bottom admits nothing
assert anything.is_valid(object())                    # top admits everything
```

The combinators accept any schema spec, so they nest and mix freely:

```python
from valgebra import complement, union, validator

color = union("red", "green", "blue")    # union of three literals
assert color.is_valid("red")
assert not color.is_valid("teal")

not_empty_text = complement(union("", b""))   # not the empty str or bytes
assert not_empty_text.is_valid("x")
assert not not_empty_text.is_valid("")
```

## The laws hold

Because membership is Boolean and the combinators are exactly *or*, *and*, and
*not*, every Boolean-algebra law holds — commutativity, associativity,
idempotence, absorption, identities, distributivity, De Morgan, and double
negation. These are property-tested in both Rust and Python against the
membership relation, not asserted.

## Conditional combinators

`ifthen` and `cond` are derived purely from the algebra — they add no new
semantics.

`ifthen(condition, then)` requires `then` of any value that matches `condition`,
and admits everything else (so it reads as "condition implies then"):

```python
from typing import Annotated

import annotated_types as at

from valgebra import ifthen

non_negative_if_int = ifthen(int, Annotated[int, at.Ge(0)])
assert non_negative_if_int.is_valid(5)
assert not non_negative_if_int.is_valid(-1)
assert non_negative_if_int.is_valid("not an int")  # not an int: admitted
```

`cond` selects the `then` of the first matching `(condition, then)` case:

```python
from typing import Annotated

import annotated_types as at

from valgebra import cond, nothing

shape = cond(
    (str, Annotated[str, at.MinLen(1)]),
    (int, Annotated[int, at.Ge(0)]),
    default=nothing,
)
assert shape.is_valid("ok")
assert shape.is_valid(5)
assert not shape.is_valid("")
assert not shape.is_valid(1.5)  # matches no case, falls to the default
```

## The simplifier

`simplify` reduces a schema by the lattice laws while admitting **exactly the
same values**. It flattens nested unions and
intersections, drops duplicates and identities, and pushes complements to
negation-normal form:

```python
from valgebra import complement, simplify, union

assert repr(simplify(complement(complement(int)))) == "int"
assert repr(simplify(union(int, int))) == "int"
```

It also decides the **complement laws** and provable **disjointness**: a schema
met with its complement, or with a provably disjoint type, is empty; a schema
joined with its complement, or with the complement of a disjoint type, is
everything.

```python
from valgebra import complement, intersect, simplify, union

assert repr(simplify(intersect(int, complement(int)))) == "nothing"
assert repr(simplify(union(int, complement(int)))) == "anything"
assert repr(simplify(intersect(int, str))) == "nothing"  # disjoint types
```

Disjointness is decided soundly for the concrete fragment — the builtin scalars
(with `bool` a subtype of `int`), the container kinds, and a refinement through
its base. Elsewhere — `Literal` and class `isinstance` checks (a class may
subclass a builtin), and recursive references — the simplifier stays
conservative: it never claims an equivalence it cannot justify, so a result is
always sound, never reporting a non-empty schema as empty. It never treats `Any`
as the top, so a deliberately-unchecked schema is preserved.

## `Any` versus `anything`

Both admit every value at runtime, but they are different in the algebra:

- `anything` is the lattice **top**. It obeys the laws —
  `complement(anything)` is `nothing`, `intersect(anything, s)` is `s`.
- `Any` is the gradual dynamic type, an **atom** the simplifier never rewrites.

```python
from typing import Any

from valgebra import anything, complement, simplify, validator

assert repr(simplify(complement(anything))) == "nothing"
assert repr(simplify(validator(Any))) == "Any"  # left untouched
```

This keeps "checked: every value is admitted" (`anything`) distinct from
"deliberately not checked" (`Any`).
