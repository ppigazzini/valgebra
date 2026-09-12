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

assert union(int, str).is_valid("x")  # a value in either set
assert intersection(int, complement(bool)).is_valid(5)  # ints that are not bools
assert not intersection(int, complement(bool)).is_valid(True)
assert complement(nothing).is_valid(5)  # the complement of bottom is top
assert not nothing.is_valid(5)  # bottom admits nothing
assert anything.is_valid(object())  # top admits everything
```

The combinators accept any schema spec, so they nest and mix freely:

```python
from valgebra import complement, union, Validator

color = union("red", "green", "blue")  # union of three literals
assert color.is_valid("red")
assert not color.is_valid("teal")

not_empty_text = complement(union("", b""))  # not the empty str or bytes
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
*not*, every Boolean-algebra law on schemas — commutativity, associativity,
idempotence, absorption, identities, distributivity, De Morgan, and double
negation — reduces to the same law on the per-value membership verdicts, where it
holds universally. That reduction is property-tested in both Rust and Python over
generated schemas and values, exercising each law against the membership relation
rather than asserting it.

The model — schemas as value-sets, subtyping as set inclusion, full union,
intersection, and complement — is *semantic subtyping*. The
[foundations](13-foundations.md) page records the theory and its references, and
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
assert not pair.is_valid([1])  # wrong length
assert not pair.is_valid((1, "a"))  # a tuple is not a member of the list form
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

## What `==` compares

`==` asks whether two validators are the **same schema**, and two spellings of
one schema are one schema: a record's fields, a map's clauses, a refinement's
markers and a union's members are sets, so the order they were written in is not
part of what they name.

```python
from typing import Annotated, Literal

import annotated_types as at

from valgebra import Validator, union

assert Validator({"a": int, "b": str}) == Validator({"b": str, "a": int})
assert Validator(Literal[1, 2]) == Validator(Literal[2, 1])
assert Validator(Annotated[int, at.Ge(0), at.Le(9)]) == Validator(
    Annotated[int, at.Le(9), at.Ge(0)]
)

# So a union of two spellings folds to one member, and a validator is a usable
# dictionary key however the schema behind it was written.
assert union({"a": int, "b": str}, {"b": str, "a": int}) == Validator(
    {"a": int, "b": str}
)
assert len({Validator(Literal[1, 2]), Validator(Literal[2, 1])}) == 1

# Two schemas that differ only in a constant are two keys, not one bucket:
# `hash` reads the constant a slot names, as `==` does.
assert len({Validator(Literal[n]) for n in range(100)}) == 100

# And one schema prints one way: a union's literals are ordered by what they
# are, not by the slot construction happened to give them.
assert repr(Validator(Literal[2, 1])) == repr(Validator(Literal[1, 2]))
```

**`==` is not `is_equivalent`.** Equality is what the constructors settle:
flattening, identities, absorption, the complement laws, and the orders above.
Equivalence is what the decision procedures *prove*, and it decides a great deal
more — `bool` and `Literal[True, False]` are one set and two terms.

```python
from typing import Literal

from valgebra import Validator, intersection, nothing

assert Validator(bool).is_equivalent(Literal[True, False])
assert Validator(bool) != Validator(Literal[True, False])
assert intersection(int, str).is_equivalent(nothing)
assert intersection(int, str) != Validator(nothing)
```

## The simplifier is going

`simplify` is **deprecated** and is removed in the next minor version. Calling it
raises a `DeprecationWarning`.

A schema is built in the lattice normal form, so the reduction `simplify`
promises is the schema you already hold: `repr` shows it and `==` compares it.

```python
from valgebra import Validator, complement, intersection, union

assert repr(complement(complement(int))) == "int"
assert repr(union(int, int)) == "int"
assert repr(intersection(int, complement(int))) == "nothing"
```

What `simplify` still does beyond that is not a law but a **decision**: a meet of
two provably disjoint kinds is the bottom, a join whose members cover every
region is the top. The three relations decide those and a great deal more,
without rewriting a term:

```python
from valgebra import Validator, complement, intersection, union

assert intersection(int, str).is_empty()  # disjoint kinds
assert union(int, complement(bool), Validator(1)).is_equivalent(object)
```

Ask the relation the question. A schema is a set, and `is_empty`,
`is_subtype_of` and `is_equivalent` are how you ask about one; a smaller term
that denotes the same set answers nothing a relation does not answer better.

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

A `False` from `is_subtype_of` folds together two answers a caller often wants
apart: a value of the subject that the other schema rejects, and a question the
procedure declines. `relation_to` reports them separately, and answers the same
question `is_subtype_of` does -- `"subset"` is exactly its `True`:

```python
from typing import NamedTuple

from valgebra import Validator


class Point(NamedTuple):
    x: int
    y: int


assert Validator(bool).relation_to(int) == "subset"
# A refutation: some string is not an integer.
assert Validator(str).relation_to(int) == "not_subset"
# A named tuple lays out a tuple, and the schema says so.
assert Validator(Point).relation_to(tuple[int, int]) == "subset"
```

Which relations answer `"undecided"` is the conservative boundary
[the decidability page](15-decidability.md) describes. A `"not_subset"` is a
statement about a value: some member of the subject is outside the other schema.
Where the subject is a class, that rests on the class holding an instance, which
is the one assumption that page records.

`is_equivalent` is **semantic**: it compares the value sets, however the two
schemas are spelled. Keep it distinct from `==` on validators, which compares the
schema's **normal form** — the shape construction builds. The lattice laws are
settled there, so a difference of order, of a repeat, or of an identity is not a
difference:

```python
from valgebra import Validator, anything, complement, intersection, nothing, union

assert union(int, str) == union(str, int)  # commutativity
assert union(int, int) == Validator(int)  # idempotence
assert union(int, Validator(nothing)) == Validator(int)  # the identity
assert intersection(int, Validator(anything)) == Validator(int)  # dually
assert intersection(int, complement(int)) == Validator(nothing)  # the complement law
```

What `==` does not see is a relation that needs a *containment*: `bool` is below
`int`, so `union(bool, int)` denotes the set `int` does, and no law rewrites one
into the other. Ask `is_equivalent` "do these mean the same set?" and `==` "are
these the same schema?".

```python
from valgebra import Validator, union

assert union(bool, int) != Validator(int)
assert union(bool, int).is_equivalent(int)
```

Shape includes the constants a schema pins, read the way `Literal` reads one:
same type and equal. `Validator(Literal[1]) != Validator(Literal[True])`, because
the two denote disjoint sets even though `1 == True`.

### Relations between your own schemas

A codebase with more than one schema for the same data has relations between them
that nothing checks: the schema an endpoint accepts against the one a record is
stored as, a narrowed variant against the schema it narrows, an intersection
against the arms it was built from. Those are set questions, so they are
assertions rather than review comments, and they belong in the test suite beside
the schemas:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator

stored = {"name": str, "runs": int}
submitted = {"name": str, "runs": int, "note?": str}

# Every persisted record is a valid submission, or a round trip through storage
# produces something the endpoint refuses.
assert Validator(stored).is_subtype_of(submitted)

# A tightened field stays within the field it tightens.
assert Validator(Annotated[int, at.Ge(0), at.Le(100)]).is_subtype_of(
    Annotated[int, at.Ge(0)]
)
```

Written as *shapes*, which are closed. The same records spelled as `TypedDict`s
relate the other way round: a `TypedDict` is [open](03-schema-language.md#a-typeddict-is-open-a-dict-literal-is-closed),
so the one that names no `note` admits a dict whose `note` is an `int` — which
the one that names `note: str` does not.

Assert the **positive** direction only. A `True` from `is_subtype_of`,
`is_equivalent` or `is_empty` is a proof; a `False` is "no, or not yet proven"
(the [decidability boundary](15-decidability.md#the-contract) says which is
which). So `assert not a.is_subtype_of(b)` passes both when the relation is false
and when it merely is not decided, and `assert not schema.is_empty()` is not a
check that the schema admits anything — a schema that admits nothing can answer
`False` there. To assert a schema is inhabited, name a value it must admit:

```python
from typing import Annotated

import annotated_types as at

from valgebra import intersection

bounded = intersection(Annotated[int, at.Ge(0)], Annotated[int, at.Le(10)])
assert bounded.is_valid(5)  # a witness, not `not bounded.is_empty()`
```

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
mappings (including multi-clause mixed maps, a closed record against a catch-all
mapping, and mixed maps where the supertype's extra field is optional and the
subtype's catch-all covers it), inclusion in a complement where the two schemas
share no value, and recursion. Where a rule declines, the relation is asked again
of the *sets* the two schemas denote, which decides what no rule about shapes
reaches — a container meet, a double complement, one regular language inside
another, a kind against its own literals, one step dividing another — under a
bound on what building those sets may cost. The two readings, and the guard that
decides when a negative answer is a refutation rather than a decline, are named
in [the foundations](13-foundations.md). What is past that bound, and what no
finite set representation holds, is where they stay conservative. The
[decidability boundary](15-decidability.md) lists exactly what is decided, what is
conservative, and what is undecidable at runtime; see the
[foundations](13-foundations.md) for the theory.

## `Any` versus `anything`

They denote the same set — every value — and they are the same schema. What
differs is what `repr` gives back.

```python
from typing import Any

from valgebra import Validator, anything, complement, intersection

assert Validator(Any) == Validator(anything)
assert repr(Validator(Any)) == "Any"
assert repr(Validator(anything)) == "anything"
assert repr(complement(Any)) == "nothing"
assert intersection(Any, complement(Any)).is_empty()
```

The lattice laws therefore reach `Any` like any other set: `complement(Any)`
denotes `nothing`, and a meet of `Any` with its complement is decided empty.

A static type checker holds `Any` apart from the top, and for a reason that does
not apply here: it asks a second question, *consistency*, at every site where a
value crosses between typed and untyped code. A validator asks one question —
does this value belong — and to it `Any` answers yes for every value, which is
what the walk has always done. Writing `Any` still says something to a reader,
and the schema keeps it: it is a spelling, not a set, so two schemas that differ
only in it are equal and nothing decides anything by it
([the boundary](15-decidability.md)).
