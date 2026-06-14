# The decidability boundary

valgebra compares schemas as sets: `is_subtype` is set inclusion, `equivalent` is
mutual inclusion, and `is_empty` reports an unsatisfiable schema. The relation is
`s <= t` exactly when `s` and `not t` share no value, so every comparison reduces
to an emptiness test (see [foundations](foundations.md)).

Every answer is **sound**. A `True` from `is_subtype`/`equivalent`, or a `True`
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
- **Refinements.** A refinement is a subtype of its base and of a looser
  refinement; a bound conjunction that cannot be satisfied — a lower bound above an
  upper bound, or a minimum length above a maximum — is empty.
- **Sequences.** Homogeneous, fixed-length, and prefix-plus-tail lists and tuples,
  with the container as part of the type (a list is never a tuple).
- **Sets and frozensets.** By element inclusion.
- **Records and mappings.** Closed-record width, depth, and required-ness; pure
  mappings with several key-pattern clauses (each subtype clause subsumed by a
  supertype clause); and a record mixed with a catch-all when both sides carry the
  same field names.
- **Recursion.** Equirecursive schemas compare at their greatest fixpoint; the
  rule is sound and is witnessed by an independent reference denotation.

```python
from typing import Annotated

import annotated_types as at

from valgebra import complement, intersect, lazy, union, validator

assert validator(bool).is_subtype(int)  # bool is a subtype of int
assert validator(1).is_subtype(int)  # a literal is a member of int
assert validator(Annotated[int, at.Ge(0)]).is_subtype(int)  # refinement <= base
assert validator(Annotated[int, at.Ge(10), at.Le(0)]).is_empty()  # no such int
assert validator({str: int}).is_subtype({str: int, int: bool})  # mapping clauses
assert union(bool, int).equivalent(int)  # bool | int is just int
assert intersect(int, complement(int)).is_empty()  # the complement law

json_value = lazy(lambda j: union(None, bool, int, float, str, [j], {str: j}))
assert json_value.is_valid({"a": [1, "x", {"b": None}]})
```

## Sound but conservative

Here valgebra is correct but not complete: it may answer `False` or "not empty"
for a relation that does in fact hold. These are decidable in principle and are
tracked as future work.

- **Mixed maps with differing field names.** A record-plus-catch-all is decided
  only when both sides share field names; differing names need the full
  quasi-constant-function comparison.
- **Sequence regular-expression inclusion** beyond the prefix-and-tail form (for
  example a union of sequence languages that splits across branches). The frontend
  does not build such a shape; it arises only inside the decision procedure.
- **Integer-only emptiness** of an open interval, such as a value strictly between
  two consecutive integers.

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

from valgebra import validator

T = TypeVar("T")

for undecidable in (Sequence[int], T):
    try:
        validator(undecidable)
        raise AssertionError("expected a rejection")
    except NotImplementedError:
        pass  # rejected with a clear message, never a silent wrong validator
```

## The contract

A positive answer (`is_subtype`/`equivalent`/`is_empty` returning `True`) is a
proof. A negative answer is "no, or not yet proven". valgebra never reports a
relation it cannot justify, so widening the decided fragment can only turn a
conservative `False` into a `True` — it can never change a previously-correct
answer.
