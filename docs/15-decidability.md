---
description: What subtyping, equivalence, and emptiness decide exactly versus conservatively.
---

# The decidability boundary

valgebra compares schemas as sets: `is_subtype_of` is set inclusion, `is_equivalent` is
mutual inclusion, and `is_empty` reports an unsatisfiable schema. The relation is
`s <= t` exactly when `s` and `not t` share no value, so every comparison reduces
to an emptiness test (see [foundations](13-foundations.md)).

Every answer is **sound**. A `True` from `is_subtype_of`/`is_equivalent`, or a `True`
from `is_empty`, is always correct. Where valgebra cannot yet prove a relation it
answers conservatively — `False`, or "not empty" — never a wrong `True`. So a
positive answer is a guarantee, and a negative answer is "no, or not proven".

This page states which queries valgebra decides completely, which stay
conservative, and which are undecidable at runtime and so are rejected or treated
opaquely by necessity.

## Decided exactly

Over this fragment, valgebra returns the exact set-theoretic answer: on every
case below it agrees with set inclusion in both directions, not only the sound
one. This exactness is verified case by case against a completeness ledger — a
curated set of relations the procedure is asserted to decide — and re-checked by
a fuzzer that confirms the sound direction over a finite value universe; it is a
gated, exercised guarantee, not a proved theorem over the whole fragment. Outside
this fragment the procedure stays sound (see [Sound but
conservative](#sound-but-conservative)).

- **The scalar Boolean algebra.** Every union, intersection, and complement of the
  scalar atoms (`None`, `bool`, `int`, `float`, `str`, `bytes`), with `bool` a
  subtype of `int`. The complement laws hold: `int & ~int` is empty, `int | ~int`
  is the universe.
- **Complement and disjointness across kinds.** An intersection that carries a
  schema together with its complement (`A & ~A`), or two members of provably
  disjoint kinds (a list and a set, an `int` and a `str`), is empty — for the
  structural kinds, not only the scalars. The gradual `Any` is exempt from the
  complement law: `Any & ~Any` is *not* empty, because `Any` is the dynamic type,
  not a set whose complement cancels it.
- **Class and literal inclusion.** A class is a subtype of its base classes
  (`issubclass`), and a literal is a subtype of any schema it is a member of. A
  dataclass or named tuple relates the same way: its schema is below one over a
  base class it carries every attribute of, each with a narrower schema, and below
  the bare class it is an instance of.
- **Literals against other kinds.** A literal pins `type(x)` exactly, so it
  carries the kind of its constant and is decided against another kind:
  `Literal["a"]` is below `~int`, and `Literal["a"] & Literal["b"]` is empty.
  `Literal[1]` and `Literal[True]` are disjoint although `1 == True`, because the
  two pin different types. The rule reads the constant's type and applies only to
  the builtin scalars, whose equality is Python's own; an `Enum` member carries
  user-defined equality, so a meet of two of them stays conservative.
- **Refinements.** A refinement is a subtype of its base and of a refinement with
  looser bounds — a tighter numeric or length bound entails a looser one, not only
  a verbatim-contained constraint set; a bound conjunction that cannot be satisfied
  — a lower bound above an upper bound, or a minimum length above a maximum — is
  empty. Where the values are bounded to the integers the bounds count them, so an
  interval that skips every integer — `Annotated[int, Gt(0), Lt(1)]` — is empty
  even though its endpoints are ordered. That holds however the meet is spelled:
  on one refinement, or across an intersection whose members bound it, since an
  intersection is a subset of every member. A `bool` base counts too, because it
  subclasses `int`; a `float` base stays dense, so the same bounds are not empty.
- **Sequences.** Homogeneous, fixed-length, and prefix-plus-tail lists and tuples,
  with the container as part of the type (a list is never a tuple). Every sequence
  schema valgebra builds takes this linear shape, so sequence inclusion is decided
  completely. A **fixed-length** sequence is also decided against a union of
  fixed-length ones it splits across, where no single branch contains it:
  `tuple[int | str, int]` is below `tuple[int, int] | tuple[str, int]`. The rule
  needs a fixed component count, so a homogeneous or variadic sequence — a star,
  matching every length — is not decomposed.
- **Sets and frozensets.** By element inclusion.
- **Records and mappings.** Closed-record width, depth, and required-ness; pure
  mappings with several key-pattern clauses (each subtype clause subsumed by a
  supertype clause); and a record mixed with a catch-all when the subtype carries
  at least the supertype's fields, or when a field the subtype lacks is optional
  in the supertype and the subtype's catch-all covers its value type (each extra
  or optional field covered by a catch-all over all string keys). A closed record
  is compared against a catch-all mapping by the same rule, so `{"x": int}` is
  decided below `dict[str, int]`.
- **Inclusion in a complement.** `A` is below `~B` exactly when `A` and `B` share
  no value, so the relation is decided wherever emptiness decides disjointness:
  `list[int]` is below `~int`, and `dict[str, int]` below `~str`. This is the
  semantic-subtyping reduction applied where no structural rule can help — a
  complement has no shape on the right to recurse into.
- **Recursion.** Equirecursive schemas compare at their greatest fixpoint; the
  rule is sound and is witnessed by an independent reference denotation.

```python
from typing import Annotated, Any

import annotated_types as at

from valgebra import complement, intersection, recursive, union, Validator

assert Validator(bool).is_subtype_of(int)  # bool is a subtype of int
assert Validator(1).is_subtype_of(int)  # a literal is a member of int
assert Validator(Annotated[int, at.Ge(0)]).is_subtype_of(int)  # refinement <= base
assert Validator(Annotated[int, at.Ge(10), at.Le(0)]).is_empty()  # no such int
assert Validator(
    Annotated[int, at.Gt(0), at.Lt(1)]
).is_empty()  # no int strictly between
assert not Validator(
    Annotated[float, at.Gt(0), at.Lt(1)]
).is_empty()  # floats are dense
assert Validator({str: int}).is_subtype_of({str: int, int: bool})  # mapping clauses
assert Validator({str: int}).is_subtype_of(
    {"b?": int, str: int}
)  # optional field, catch-all covers it
assert Validator({"x": int}).is_subtype_of(
    {str: int}
)  # a closed record below a catch-all mapping
assert Validator(list[int]).is_subtype_of(
    complement(int)
)  # inside a complement: a list shares no value with an int
assert Validator(tuple[int | str, int]).is_subtype_of(
    union(tuple[int, int], tuple[str, int])
)  # a product splits across branches
assert union(bool, int).is_equivalent(int)  # bool | int is just int
assert intersection(int, complement(int)).is_empty()  # the complement law
assert intersection(
    list[int], complement(list[int])
).is_empty()  # complement law, structurally
assert intersection(
    list[int], set[int]
).is_empty()  # disjoint kinds: a list is never a set
assert not intersection(
    Any, complement(Any)
).is_empty()  # Any is exempt from the complement law

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
  key space does not prove that field is present. Deciding it in general needs
  the full quasi-constant-function comparison.

  The reachable half of this is decided: a subtype that denotes the **empty set**
  is below every schema, including one declaring a field it lacks, because the
  empty set is a subset of every set. That case does not need the comparison
  above, and both lattice bounds are decided by emptiness rather than by the
  shape of the atom — a schema that denotes nothing without being spelled
  `nothing`, and one that covers the universe without being spelled `anything`,
  are both recognised.

- **A catch-all keyed by literals covering a field name.** Whether a supertype's
  catch-all clause governs a field the subtype declares is asked by matching the
  clause's key against `str` and `object`, rather than by asking whether the
  field's name belongs to the key's set. So `{"a": int}` is decided below
  `dict[str, int]` but not below `dict[Literal["a"], int]`, which names exactly
  that key. Deciding it needs the core to compare a field name against a pooled
  constant, which is a question only the binding can answer today.

- **A constraint with no value entailment.** A bound (`Ge`, `Gt`, `Le`, `Lt`,
  `MinLen`, `MaxLen`) entails a looser one through the ordering oracle, so a
  tighter bound is decided below a looser one. The other three kinds — `Regex`,
  `MultipleOf` and a predicate — are opaque: nothing is read out of them, so a
  refinement carrying one relates to another schema only when that schema carries
  the same constraint verbatim. A regex is never turned into the language it
  denotes, so neither `Regex("a")` below `Regex("ab?")` nor `Regex("a")` below
  the singleton `Literal["a"]` is decided, and `MultipleOf(4)` is not decided
  below `MultipleOf(2)`. A predicate has no route to being decided (its
  satisfiability is undecidable, below). The other two do — regular-language
  inclusion is decidable, and so is divisibility of one integer by another — and
  each needs the core to reason about a pooled constant rather than test it for
  equality.

- **A schema that is not itself a refinement, against one that is.** Only a
  literal reaches the value oracle, which answers by running the membership.
  `Literal[5]` is decided below `Annotated[int, Ge(0)]`; `bool` is not, though
  every `bool` is an `int` at or above zero. Deciding it needs the bound compared
  against the subtype's own value range, which the core reads from a scalar
  region rather than from an enumerated set.

```python
from typing import Annotated, Literal

import annotated_types as at

from valgebra import Regex, Validator, complement, intersection

pattern = Validator(Annotated[str, Regex("a")])
assert not pattern.is_subtype_of(Annotated[str, Regex("ab?")])  # L(a) <= L(ab?)
assert not pattern.is_subtype_of(Literal["a"])  # L(a) is exactly {"a"}
assert not Validator(Annotated[int, at.MultipleOf(4)]).is_subtype_of(
    Annotated[int, at.MultipleOf(2)]
)

# A literal reaches the value oracle; a class does not.
assert Validator(Literal[5]).is_subtype_of(Annotated[int, at.Ge(0)])
assert not Validator(bool).is_subtype_of(Annotated[int, at.Ge(0)])

# A literal carries its constant's kind.
assert Validator(Literal["a"]).is_subtype_of(complement(int))
assert intersection(Literal["a"], Literal["b"]).is_empty()
assert intersection(Literal[1], Literal[True]).is_empty()  # 1 == True, types differ

# Involution decides both ways where the oracle reaches, one way elsewhere.
record = Validator({"a": int})
assert record.is_subtype_of(complement(complement(record)))
assert not complement(complement(record)).is_subtype_of(record)  # a record
assert not complement(complement(Validator(Literal["a"]))).is_subtype_of(
    Literal["a"]
)  # a pooled constant, though a leaf
assert complement(complement(Validator(int))).is_subtype_of(int)  # a scalar
```

- **Involution, in one direction.** `~~A` and `A` denote the same set, and
  `A <= ~~A` is decided everywhere: the rule for a complement on the right asks
  whether `A` shares a value with `~A`, which it never does. The converse has no
  rule — a complement on the *left* against anything but another complement
  matches no structural pair — so it reaches the value oracle, and whether it
  decides is whether the oracle can answer.

  It **does** for a scalar, for `anything` and `nothing`, for a union, and for a
  complement: those are the regions the oracle reasons about. It does **not** for
  a pooled constant or a class, so `Literal["a"]`, an `Enum`, a dataclass and an
  attribute schema are all undecided — and neither for any structural
  constructor: a list, a tuple, a set, a mapping or a record. The dividing line
  is the oracle's reach, not the constructor/leaf distinction: a literal is a
  leaf and is undecided all the same, for the same reason
  `Literal["a"] <= ~int` above is.

  This one has a short route: reduce to negation-normal form before comparing.
  `simplify` already cancels double negation, so the rewrite exists and is simply
  not on the path the decision takes.

The two catch-all entries were found by `tests/test_completeness_probe.py`, which
searches for relations answered `False` that no value refutes, and fails when one
appears that is not written down with a reason. Both are on its ledger, so
neither can quietly become permanent, and closing either fails the ledger until
the entry is removed. The literal-meet entry is held the same way by
`tests/test_completeness_ledger.py`, with the two counterexamples that rule out
each rule that would close it, and the involution entry the same way — as a
strict `xfail`, so closing it fails the ledger and forces the entry out.

That probe searches a fixed universe of schemas, so it reaches a gap only where
some atom in that universe reaches it. Its universe carries refinements of both
constraint families — an order bound, which entails a looser one and so reports
nothing, and a regex, which is opaque and reports — so the two refinement entries
above are held by the same gate as the rest.

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
  time; its satisfiability cannot be reasoned about (Rice's theorem), so nothing
  is inferred from it and two refinements relate through a predicate only when
  they carry the same one.

    A decision query may nonetheless **call** it. Deciding whether a literal is a
    subtype of a refinement is deciding whether that literal's value belongs to
    it, and belonging runs the predicate — so `is_subtype_of` and `is_equivalent`
    execute user code, as `is_empty` executes a rich comparison when it orders two
    refinement bounds. A predicate with side effects, or one that is slow, is one
    a type query pays for.
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

Every decision also runs under a fixed work budget, and exhausting it returns the
conservative answer (`False`, "not proven") rather than running unbounded. This
preserves soundness: a bail-out is never a wrong `True`.

The budget binds where the work is a **product** rather than a sum. Subtyping
distributes over both sides of a union, so relating two unions costs the product
of their member counts, and a union of about a thousand literals — an error-code
table, a currency list — reaches the ceiling. Below that the relation decides.
Depth reaches it sooner: a Boolean combination nested past a handful of levels
demands work exponential in its depth. Everything else stays far inside.

So a `False` on a wide literal union or a deep Boolean tower may mean "not proven
within the bound" rather than "not a subtype"; on anything else it means the
relation is outside the decided fragment above.
