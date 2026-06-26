---
description: What subtyping, equivalence, and emptiness decide exactly versus conservatively.
---

# The decidability boundary

valgebra compares schemas as sets: `is_subtype_of` is set inclusion, `is_equivalent` is
mutual inclusion, and `is_empty` reports an unsatisfiable schema. The relation is
`s <= t` exactly when `s` and `not t` share no value, so every comparison reduces
to an emptiness test (see [foundations](foundations.md)).

Every answer is **sound**. A `True` from `is_subtype_of`/`is_equivalent`, or a `True`
from `is_empty`, is always correct. Where valgebra cannot yet prove a relation it
answers conservatively — `False`, or "not empty" — never a wrong `True`. So a
positive answer is a guarantee, and a negative answer is "no, or not proven".

This page states which queries valgebra decides completely, which stay
conservative, and which are undecidable at runtime and so are rejected or treated
opaquely by necessity.

## Decided completely

Over this fragment, valgebra returns the exact set-theoretic answer.

- **The scalar Boolean algebra.** Every union, intersection, and complement of the
  scalar atoms (`None`, `bool`, `int`, `float`, `str`, `bytes`), with `bool` a
  subtype of `int`. The complement laws hold: `int & ~int` is empty, `int | ~int`
  is the universe.
- **Class and literal inclusion.** A class is a subtype of its base classes
  (`issubclass`), and a literal is a subtype of any schema it is a member of.
- **Refinements.** A refinement is a subtype of its base and of a refinement with
  looser bounds — a tighter numeric or length bound entails a looser one, not only
  a verbatim-contained constraint set; a bound conjunction that cannot be satisfied
  — a lower bound above an upper bound, or a minimum length above a maximum — is
  empty. On an integer base the bounds count integers, so an interval that skips
  every integer — `Annotated[int, Gt(0), Lt(1)]` — is empty even though its
  endpoints are ordered; a `float` base stays dense, so the same bounds are not
  empty.
- **Sequences.** Homogeneous, fixed-length, and prefix-plus-tail lists and tuples,
  with the container as part of the type (a list is never a tuple). Every sequence
  schema valgebra builds takes this linear shape, so sequence inclusion is decided
  completely.
- **Sets and frozensets.** By element inclusion.
- **Records and mappings.** Closed-record width, depth, and required-ness; pure
  mappings with several key-pattern clauses (each subtype clause subsumed by a
  supertype clause); and a record mixed with a catch-all when the subtype carries
  at least the supertype's fields, or when a field the subtype lacks is optional
  in the supertype and the subtype's catch-all covers its value type (each extra
  or optional field covered by a catch-all over all string keys).
- **Recursion.** Equirecursive schemas compare at their greatest fixpoint; the
  rule is sound and is witnessed by an independent reference denotation.

```python
from typing import Annotated

import annotated_types as at

from valgebra import complement, intersection, recursive, union, Validator

assert Validator(bool).is_subtype_of(int)  # bool is a subtype of int
assert Validator(1).is_subtype_of(int)  # a literal is a member of int
assert Validator(Annotated[int, at.Ge(0)]).is_subtype_of(int)  # refinement <= base
assert Validator(Annotated[int, at.Ge(10), at.Le(0)]).is_empty()  # no such int
assert Validator(Annotated[int, at.Gt(0), at.Lt(1)]).is_empty()  # no int strictly between
assert not Validator(Annotated[float, at.Gt(0), at.Lt(1)]).is_empty()  # floats are dense
assert Validator({str: int}).is_subtype_of({str: int, int: bool})  # mapping clauses
assert Validator({str: int}).is_subtype_of({"b?": int, str: int})  # optional field, catch-all covers it
assert union(bool, int).is_equivalent(int)  # bool | int is just int
assert intersection(int, complement(int)).is_empty()  # the complement law

json_value = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
assert json_value.is_valid({"a": [1, "x", {"b": None}]})
```

## Sound but conservative

Here valgebra is correct but not complete: it may answer `False` or "not empty"
for a relation that does in fact hold. These are decidable in principle and are
tracked as future work.

- **Mixed maps where the supertype declares a _required_ field the subtype
  lacks.** When the missing field is optional, the subtype's catch-all covers it
  and the case is decided; a required field is not, because a catch-all over the
  key space does not prove that field is present. Deciding it needs the full
  quasi-constant-function comparison.
General regular-expression-types inclusion of sequences (a union of sequence
languages that splits across branches, or a repeated heterogeneous group) is not
implemented, but no schema valgebra builds takes that shape — every sequence is
the linear prefix-and-tail form, which is decided completely — so it is not a
reachable gap.

## Undecidable at runtime

These have no decidable runtime membership, so valgebra rejects them with a clear
message or treats them opaquely — it never guesses.

- **Erased generics and type variables.** A `TypeVar`, `Generic[T]`, `ParamSpec`,
  or `TypeVarTuple` is rejected; a runtime value carries no binding for a free type
  variable.
- **Abstract-collection generics.** `Sequence[int]`, `Mapping[str, int]`, and
  `Iterable[T]` are rejected; checking `Iterable` elements would consume the
  iterable, and `str`/`bytes` are themselves sequences. Use a concrete container —
  `list[int]`, `tuple[int, ...]`, `dict[str, int]` — or the bare abstract type for
  an `isinstance` check.
- **Callable signatures.** `Callable[[int], str]` checks only that the value is
  callable; a function does not expose its argument and return types at runtime.
- **Predicates.** An `Annotated[T, predicate]` runs the predicate at validation
  time; its satisfiability cannot be reasoned about (Rice's theorem), so it is
  opaque to subtyping and emptiness.
- **Typing qualifiers.** `Final` and `ClassVar` are rejected; they qualify a
  declaration and carry no value-membership meaning.

```python
from collections.abc import Sequence
from typing import TypeVar

from valgebra import Validator

T = TypeVar("T")

for undecidable in (Sequence[int], T):
    try:
        Validator(undecidable)
        raise AssertionError("expected a rejection")
    except NotImplementedError:
        pass  # rejected with a clear message, never a silent wrong validator
```

## The contract

A positive answer (`is_subtype_of`/`is_equivalent`/`is_empty` returning `True`) is a
proof. A negative answer is "no, or not yet proven". valgebra never reports a
relation it cannot justify, so widening the decided fragment can only turn a
conservative `False` into a `True` — it can never change a previously-correct
answer.
