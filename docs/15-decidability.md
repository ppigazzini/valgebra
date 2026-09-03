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
  rule: `intersection(Any, complement(Any))` is not *decided* empty, because
  `Any` is held as an atom rather than as the set it admits. At runtime it admits
  nothing, so the set is empty and the procedure declines to say so.
- **Class and literal inclusion.** A class is a subtype of its base *classes*,
  by `issubclass`, and a literal is a subtype of any schema it is a member of. A
  dataclass or named tuple relates the same way: its schema is below one over a
  base class it carries every attribute of, each with a narrower schema, and below
  the bare class it is an instance of. The relation is between two class atoms:
  a scalar or a container is a node of its own rather than a class, so
  `Validator(MyInt)` is not decided below `Validator(int)` for an `int`
  subclass, though every value of the first is a value of the second.
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
  schema valgebra builds takes this linear shape, so inclusion *between two
  sequence schemas* is decided completely — a bare `list` is a class atom rather
  than a sequence, and relates as a class does, not as a sequence. A
  **fixed-length** sequence is also decided against a union of
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
  decided below `dict[str, int]`. A **meet** of two of them is empty when some key
  one side requires cannot hold: because the types the two give it share no value,
  or because the other side is closed and does not declare it. Only a required key
  can do this — a meet of two mappings, or of two optional fields, always contains
  the empty dict.
- **Inclusion in a complement.** `A` is below `~B` exactly when `A` and `B` share
  no value, so the relation is decided wherever emptiness decides disjointness:
  `list[int]` is below `~int`, and `dict[str, int]` below `~str`. This is the
  semantic-subtyping reduction applied where no structural rule can help — a
  complement has no shape on the right to recurse into.
- **Recursion.** Equirecursive schemas compare at their greatest fixpoint; the
  rule is sound and is witnessed by an independent reference denotation. The
  *sets* are inductive — a guarded fixpoint contains the values built by finitely
  many unfoldings — while the *comparison* assumes its goal and is coinductive;
  the two agree because a value is finite. A fixpoint is decided below its own
  unfolding, so a `recursive` schema and the body written out around it relate
  in both directions.
- **The complement laws, where the constructors reach them.** `complement`
  cancels a complement and `union` folds a join carrying a schema beside its own
  complement, both where the schema is built. So `complement(complement(int))`
  **is** `int` — one schema, which `repr` and `==` report and which a comparison
  is never asked about — and a union covering the universe **is** `anything`.
  The decision procedure has no rule for either shape and never meets one built
  this way. A shape the fold does not reach is a different matter and is
  conservative (below). `Any` is exempt: it is an atom, not a set whose
  complement completes it.

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
assert intersection({"a": int}, {"a": str}).is_empty()  # 'a' cannot hold both
assert not intersection(
    {"a?": int}, {"a?": str}
).is_empty()  # the empty dict is in both
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

- **A relation against a respelled operand.** The rules that settle the
  complement laws read structural equality, so `A | ~A` is recognised as the
  universe and `A | ~B` is not — for a `B` that `is_equivalent` proves equal to
  `A`, such as `A | nothing`. The same holds of every rule the decision applies:
  two of them compose only where the operands are spelled alike. Deciding it in
  general wants the operands compared as sets wherever a rule reads equality.

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

- **A shape the complement fold does not reach.** The fold reads structural
  equality inside one validator's constants, so it settles a join written
  `union(A, complement(A))` and not one written any other way: a respelling such
  as `union(A, complement(union(A, nothing)))`, a recursive schema whose two
  occurrences compile to two definitions, or two constants that are equal without
  being the same object. The procedure has no rule for those, so they are not
  decided.

- **An emptiness that needs an inclusion.** `is_empty` decides the regions, the
  complement law and the bounds directly, and never asks whether one member of a
  meet is below another. So `list[bool] & ~list[int]` is not decided empty, even
  though `list[bool] <= list[int]` is decided — and a container meet is not met
  componentwise, so `list[int] & list[str]` is not decided to be the empty list.

- **A scalar kind against its own values.** A kind is a region bit rather than a
  set of the values in it, so `bool` is not decided below `Literal[True, False]`,
  `Literal[1]` is not decided below `int & ~bool`, and a negated literal is not
  subtracted from the kind that holds it.

- **A refinement against a base that is not one.** A bound is compared against
  another bound, and a base that is not itself a refinement reaches one only
  through the value oracle, which answers for a literal and not for a class or a
  kind. A length bound is opaque to the shape it bounds, so a two-tuple is not
  decided empty under `MinLen(3)`.

- **A map's domain as written.** The field list is the domain, so a field whose
  type is empty is not absorbed and `{"a?": nothing}` is not decided equal to
  `{}`; and a key type is matched against the string and top atoms rather than
  asked whether it admits a name, so `{"a": int}` is not decided below
  `dict[Literal["a"], int]`.

Every relation named here is a strict expected failure in
`tests/test_completeness_ledger.py`, so the day a rule decides one the mark fails
and the entry leaves both the ledger and this list.

```python
from typing import Annotated, Literal

import annotated_types as at

from valgebra import (
    Regex,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    union,
)

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

# The complement fold is a construction, so `~~A` is `A` -- one schema, not two
# the procedure relates.
record = Validator({"a": int})
assert complement(complement(record)) == record

# Written another way, the same set is a shape the fold does not reach and the
# procedure does not decide.
respelled = union(record, complement(union(record, nothing)))
assert respelled != Validator(anything)
assert not Validator(anything).is_subtype_of(respelled)
```

Two instruments hold this list to the tree. `tests/test_completeness_ledger.py`
carries each relation above as a strict expected failure, written the way a
caller writes it rather than built from the other operand — a distinction that
matters, because the shortcuts the procedure takes are keyed on two schemas
sharing their constants. `tests/test_completeness_probe.py` searches a fixed
universe for relations answered `False` that no value refutes and fails when one
appears without a written reason, so a gap nobody thought of cannot arrive
unnoticed. It reaches a gap only where some atom in its universe reaches it,
which is why that universe carries both constraint families, a fixpoint beside
its own unfolding, and a record beside a literal-keyed map.

General regular-expression-types inclusion of sequences (a union of sequence
languages that splits across branches, or a repeated heterogeneous group) is not
implemented, and no schema valgebra builds takes that shape: the sequence node
carries the linear prefix-and-tail form and has no syntax for the rest.

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
- **Typing qualifiers.** `Final` and `ClassVar` are rejected as schemas; they
  qualify a declaration and carry no value-membership meaning. On a class they
  are read as what they are: a `ClassVar` annotates the class rather than an
  instance, so a dataclass field carrying one is not an attribute the schema
  asks for, and neither is an `InitVar`, which names a constructor parameter the
  instance does not keep.

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

The Python answer is `True` or `False`, so a `False` from an exhausted budget
reads the same as a `False` the procedure decided. Inside the core the two are
distinct — emptiness answers *empty*, *inhabited*, or *neither* — which is what
lets a test say that a bail-out never claims a proof it does not have. The
distinction is not surfaced here because the contract does not change with it: a
`False` is "not proven" either way.

The budget binds where the work is a **product** rather than a sum. Subtyping
distributes over both sides of a union, so relating two unions can cost the
product of their member counts, and a Boolean combination nested past a handful
of levels demands work exponential in its depth.

One shortcut avoids the product, and it is worth knowing exactly what reaches it:
a union whose branches are *contained* in the supertype's is settled by
containment, and containment is structural equality over one validator's
constants. Two schemas share those when one is built from the other — a table
widened by a member, `union(codes, Validator("extra"))` — and not when both are
written out. So a table widened in place is decided at any size, while two tables
written separately, as a codebase with the same schema in two modules has them,
distribute against each other and reach the ceiling above roughly a thousand
members each. Both sizes are on the ledger.

So a `False` on a wide literal union or a deep Boolean tower may mean "not proven
within the bound" rather than "not a subtype"; on anything else it means the
relation is outside the decided fragment above. The bound is the price of a
procedure with no memo, and removing it is the interning work the theory names.
