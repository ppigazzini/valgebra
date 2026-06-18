# The Boolean algebra

Union, intersection, and complement compose any schema — annotations, native
forms, or other compiled validators — into a complete, lawful Boolean lattice.
`anything` is the top (every value) and `nothing` is the bottom (no value).

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


def implies(condition, then, otherwise):
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
from valgebra import complement, intersection, simplify, union

assert repr(simplify(intersection(int, complement(int)))) == "nothing"
assert repr(simplify(union(int, complement(int)))) == "anything"
assert repr(simplify(intersection(int, str))) == "nothing"  # disjoint types
```

The simplifier folds the scalar Boolean fragment — the builtin scalars (with
`bool` a subtype of `int`) and the complement laws — so `simplify(intersection(int,
str))` is `nothing`. It never treats `Any` as the top, so a deliberately-unchecked
schema is preserved. The comparison operators below decide a wider fragment than
the simplifier folds; the [decidability boundary](decidability.md) maps exactly
what is decided.

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

The other side of a comparison is any schema spec or compiled validator. These
decisions are **sound**: a `True` is always correct, and a `False` (or a
"not empty") is either a genuine non-relation or a relation valgebra does not yet
prove — never a wrong answer. They decide completely over a wide fragment — the
scalar Boolean algebra, class and literal inclusion, refinements with bound and
length constraints, prefix-and-tail sequences, sets and frozensets, records and
mappings (including multi-clause and matching-field-name mixed maps), and
recursion — and stay conservative on the rest. The
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

from valgebra import anything, complement, simplify, Validator

assert repr(simplify(complement(anything))) == "nothing"
assert repr(simplify(Validator(Any))) == "Any"  # left untouched
```

This keeps "checked: every value is admitted" (`anything`) distinct from
"deliberately not checked" (`Any`).
