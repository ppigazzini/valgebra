---
description: The Boolean lattice (union, intersection, complement) and the law-justified simplifier.
---

# The Boolean algebra

Union, intersection, and complement compose any schema — annotations, native
forms, or other compiled validators — into a closed, lawful Boolean lattice.
`anything` is the top (every value) and `nothing` is the bottom (no value). The
typing-native spellings work too: `object` is the top and `Never` (or `NoReturn`)
is the bottom, so `Validator(object)` equals `anything` and `Validator(Never)`
equals `nothing`.

```python
from valgebra import anything, complement, intersection, nothing, union

assert union(int, str).is_valid("x")                  # a value in either set
assert intersection(int, complement(bool)).is_valid(5)   # ints that are not bools
assert not intersection(int, complement(bool)).is_valid(True)
assert complement(nothing).is_valid(5)                # the complement of bottom is top
assert not nothing.is_valid(5)                        # bottom admits nothing
assert anything.is_valid(object())                    # top admits everything
```

The combinators accept any schema spec, so they nest and mix freely:

```python
from valgebra import complement, union, Validator

color = union("red", "green", "blue")    # union of three literals
assert color.is_valid("red")
assert not color.is_valid("teal")

not_empty_text = complement(union("", b""))   # not the empty str or bytes
assert not_empty_text.is_valid("x")
assert not not_empty_text.is_valid("")
```

Union has an operator form, `|` — the same spelling typing uses for unions — so a
compiled validator joins with another schema directly. Intersection and
complement stay spelled out (typing has no operator for them, and valgebra
invents none):

```python
from valgebra import Validator, union

assert (Validator(int) | str | None).is_equivalent(union(int, str, None))
# `|` works in either order: a validator on the right is the reflected operand.
assert (int | Validator(str)).is_equivalent(union(int, str))
```

## The laws hold

Because membership is Boolean and the combinators are exactly *or*, *and*, and
*not*, every Boolean-algebra law holds — commutativity, associativity,
idempotence, absorption, identities, distributivity, De Morgan, and double
negation. These are property-tested in both Rust and Python against the
membership relation, not asserted.

The model — schemas as value-sets, subtyping as set inclusion, full union,
intersection, and complement — is *semantic subtyping*. The
[foundations](foundations.md) page records the theory and its references, and
states where the simplifier decides relationships versus where it stays
conservative.

## Composition recipes

valgebra ships only the irreducible algebra; common patterns that reduce to it
are recipes you compose, not combinators it bundles. The algebra expressing them
is the point — a named wrapper for a one-line composition would be a standard
library, not a schema algebra.

### Conditional fields

"Condition implies consequent" is a `union` of two intersections: a value either
matches the condition and must then satisfy the consequent, or fails the
condition and must satisfy the alternative (`anything` by default):

```python
from typing import Annotated

import annotated_types as at

from valgebra import anything, complement, intersection, union


def implies(condition, then, otherwise=anything):
    return union(
        intersection(condition, then),
        intersection(complement(condition), otherwise),
    )


non_negative_if_int = implies(int, Annotated[int, at.Ge(0)])
assert non_negative_if_int.is_valid(5)
assert not non_negative_if_int.is_valid(-1)
assert non_negative_if_int.is_valid("not an int")  # not an int: admitted
```

First-matching-case dispatch nests `implies` from the last case inward, so the
earliest matching condition selects its consequent:

```python
from typing import Annotated

import annotated_types as at

from valgebra import anything, complement, intersection, nothing, union, Validator


# the same implies helper as above, repeated so this example runs on its own
def implies(condition, then, otherwise=anything):
    return union(
        intersection(condition, then),
        intersection(complement(condition), otherwise),
    )


def first_match(*cases, default=anything):
    result = Validator(default)
    for condition, then in reversed(cases):
        result = implies(condition, then, result)
    return result


shape = first_match(
    (str, Annotated[str, at.MinLen(1)]),
    (int, Annotated[int, at.Ge(0)]),
    default=nothing,
)
assert shape.is_valid("ok")
assert shape.is_valid(5)
assert not shape.is_valid("")
assert not shape.is_valid(1.5)  # matches no case, falls to the default
```

### Key cardinality

"At least one of these keys is present", and its siblings, are also algebra. A
record that merely asserts a key is present is an open record requiring it —
`Validator({key: anything}).open()` — and the cardinality follows from `union`,
`intersection`, and `complement`:

```python
from valgebra import anything, complement, intersection, union, Validator


def has(key):
    return Validator({key: anything}).open()


at_least_one = union(has("a"), has("b"))
assert at_least_one.is_valid({"a": 1})
assert at_least_one.is_valid({"b": 2, "x": 0})
assert not at_least_one.is_valid({"x": 0})

at_most_one = complement(intersection(has("a"), has("b")))  # not both
assert at_most_one.is_valid({"a": 1})
assert at_most_one.is_valid({})
assert not at_most_one.is_valid({"a": 1, "b": 2})

exactly_one = union(
    intersection(has("a"), complement(has("b"))),
    intersection(has("b"), complement(has("a"))),
)
assert exactly_one.is_valid({"a": 1})
assert not exactly_one.is_valid({"a": 1, "b": 2})
assert not exactly_one.is_valid({})
```

### Fixed-length and length-bounded lists

The native list literal spells the sequence shapes typing cannot: `[A, B]` is a
fixed-length list (positional) and `[]` the empty list.

```python
from valgebra import Validator

pair = Validator([int, str])  # exactly two elements: an int then a str
assert pair.is_valid([1, "a"])
assert not pair.is_valid([1])           # wrong length
assert not pair.is_valid((1, "a"))      # a tuple is not a member of the list form
```

The single-element `[x]` is deliberately the **homogeneous** "list of x" (any
length), following Python's `list[T]` — so a fixed-length-*one* list, and any
length bound, is a refinement of the homogeneous list, not a separate form. This
is the refinement type `{ x ∈ list[T] | len bound }`, written with `Annotated`:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

# a list of exactly one int (what `[int]`, being homogeneous, does not mean)
one_int = Validator(Annotated[list[int], at.Len(1, 1)])
assert one_int.is_valid([1])
assert not one_int.is_valid([])
assert not one_int.is_valid([1, 2])

# a non-empty list, and an at-most-three list: the same length-refinement family
non_empty = Validator(Annotated[list[int], at.MinLen(1)])
assert non_empty.is_valid([1, 2]) and not non_empty.is_valid([])
small = Validator(Annotated[list[int], at.MaxLen(3)])
assert small.is_valid([1, 2, 3]) and not small.is_valid([1, 2, 3, 4])
```

## The simplifier

`simplify` is a method that reduces a schema by the lattice laws while admitting
**exactly the same values**. It flattens nested unions and
intersections, drops duplicates and identities, and pushes complements to
negation-normal form:

```python
from valgebra import complement, union

assert repr(complement(complement(int)).simplify()) == "int"
assert repr(union(int, int).simplify()) == "int"
```

A refinement is reduced to a single normal form too: nested refinements flatten
onto their shared base, and the constraint list is sorted and deduplicated, so a
repeated or reordered constraint does not change the result:

```python
from typing import Annotated

from annotated_types import Ge

from valgebra import Validator

assert repr(Validator(Annotated[int, Ge(0), Ge(0)]).simplify()) == "Annotated[int, Ge(0)]"
```

It also decides the **complement laws** and provable **disjointness**: a schema
met with its complement, or with a provably disjoint type, is empty; a schema
joined with its complement, or with the complement of a disjoint type, is
everything.

```python
from valgebra import complement, intersection, union

assert repr(intersection(int, complement(int)).simplify()) == "nothing"
assert repr(union(int, complement(int)).simplify()) == "anything"
assert repr(intersection(int, str).simplify()) == "nothing"  # disjoint types
```

The simplifier folds the scalar Boolean fragment — the builtin scalars (with
`bool` a subtype of `int`) and the complement laws — so
`intersection(int, str).simplify()` is `nothing`. It never treats `Any` as the
top, so a deliberately-unchecked
schema is preserved. The comparison operators below decide a wider fragment than
the simplifier folds; the [decidability boundary](decidability.md) maps exactly
what is decided.

`simplify` applies the lattice laws only; it does not run the emptiness
decision. An intersection that is empty by a deeper argument — contradictory
refinement bounds, for instance — is left as written, even though `is_empty`
reports it empty. So a simplified schema is a lattice normal form, not a fully
reduced one: use `is_empty`, `is_subtype_of`, and `is_equivalent` to decide
membership relations rather than reading them off the simplified structure.

```python
from typing import Annotated

from annotated_types import Ge, Le

from valgebra import Validator, intersection

contradiction = intersection(Annotated[int, Ge(10)], Annotated[int, Le(0)])
assert contradiction.is_empty()  # decided empty
assert repr(contradiction.simplify()) != "nothing"  # but simplify leaves it
```

## Subtyping, equivalence, and emptiness

A compiled validator can be compared with another schema as *sets*. `is_subtype_of`
is set inclusion, `is_equivalent` is mutual inclusion, and `is_empty` reports an
unsatisfiable schema:

```python
from valgebra import complement, intersection, union, Validator

# subtyping is set inclusion; bool is a subtype of int
assert Validator(bool).is_subtype_of(int)
assert not Validator(int).is_subtype_of(bool)
assert Validator(list[bool]).is_subtype_of(list[int])

# equivalence is mutual inclusion, whatever the syntax
assert union(bool, int).is_equivalent(int)

# emptiness detects a schema no value can satisfy
assert intersection(int, complement(int)).is_empty()
assert not Validator(int).is_empty()
```

`is_equivalent` is **semantic**: it compares the value sets, however the two
schemas are spelled. Keep it distinct from `==` on validators, which is
**syntactic** — `==` compares schema *shape*, so two schemas that denote the same
set but are written differently are equal under `is_equivalent` yet not under
`==`. Ask `is_equivalent` "do these mean the same set?" and `==` "are these the
same shape?".

```python
from valgebra import Validator, union

assert union(bool, int).is_equivalent(int)  # same set: bool is a subtype of int
assert union(bool, int) != Validator(int)  # different shape
```

The other side of a comparison is any schema spec or compiled validator. These
decisions are **sound**: a `True` is always correct, and a `False` (or a
"not empty") is either a genuine non-relation or a relation valgebra does not yet
prove — never a wrong answer. They decide completely over a wide fragment — the
scalar Boolean algebra, class and literal inclusion, refinements with bound and
length constraints, prefix-and-tail sequences, sets and frozensets, records and
mappings (including multi-clause mixed maps, and mixed maps where the supertype's
extra field is optional and the subtype's catch-all covers it), and recursion —
and stay conservative on the rest. The
[decidability boundary](decidability.md) lists exactly what is decided, what is
conservative, and what is undecidable at runtime; see the
[foundations](foundations.md) for the theory.

## `Any` versus `anything`

Both admit every value at runtime, but they are different in the algebra:

- `anything` is the lattice **top**. It obeys the laws —
  `complement(anything)` is `nothing`, `intersection(anything, s)` is `s`.
- `Any` is the gradual dynamic type, an **atom** the simplifier never rewrites.

```python
from typing import Any

from valgebra import Validator, anything, complement

assert repr(complement(anything).simplify()) == "nothing"
assert repr(Validator(Any).simplify()) == "Any"  # left untouched
```

This keeps "checked: every value is admitted" (`anything`) distinct from
"deliberately not checked" (`Any`).
