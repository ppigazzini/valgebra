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

`relation_to` reports which of the two a `False` is: `"not_subset"` where a
value of the subject is outside the other schema, and `"undecided"` where no
rule and no set reading answers. Everything in the conservative list below
answers `"undecided"`.

- **The scalar Boolean algebra.** Every union, intersection, and complement of the
  scalar atoms (`None`, `bool`, `int`, `float`, `str`, `bytes`), with `bool` a
  subtype of `int`. The complement laws hold: `int & ~int` is empty, `int | ~int`
  is the universe.
- **Complement and disjointness across kinds.** An intersection that carries a
  schema together with its complement (`A & ~A`), or two members of provably
  disjoint kinds (a list and a set, an `int` and a `str`), is empty — for the
  structural kinds, not only the scalars. `Any` is the top, spelled, so the rule
  reaches it like any other set: `intersection(Any, complement(Any))` is decided
  empty.
- **A bare container class and its parameterised form.** `list` and
  `list[object]` are one schema: an unparameterised generic names its kind's
  whole set, which is what the typing spec assigns it and what the check has
  always performed. `tuple`, `set`, `frozenset` and `dict` read the same way, as
  `str` and `int` always did.
- **Class and literal inclusion.** A class is a subtype of its base *classes*,
  by `issubclass`, and a literal is a subtype of any schema it is a member of. A
  dataclass or named tuple relates the same way: its schema is below one over a
  base class it carries every attribute of, each with a narrower schema, and below
  the bare class it is an instance of. A **named tuple** relates to the tuple its
  fields lay out as well, in both directions where both hold: its positions are
  its fields, the schema says so, and the ordinary sequence rules decide from
  there. A class built on a builtin relates to that
  builtin too: `Validator(MyInt)` is below `Validator(int)` and
  `Validator(MyStr)` below `Validator(str)`, because every instance of such a
  class is a value of that kind and the class narrows the kind rather than
  standing beside it. A class built on no builtin narrows nothing — an instance
  of a subclass of it may be a string — so it relates to a kind in neither
  direction.
- **Literals against other kinds.** A literal pins `type(x)` exactly, so it
  carries the kind of its constant and is decided against another kind:
  `Literal["a"]` is below `~int`, and `Literal["a"] & Literal["b"]` is empty.
  `Literal[1]` and `Literal[True]` are disjoint although `1 == True`, because the
  two pin different types. The rule reads the constant's type and applies only to
  the builtin scalars, whose equality is Python's own, and to any type that
  compares by *identity* -- which an enumeration does unless it says otherwise,
  so two of its members are two values and a meet of them is empty. An `IntEnum`
  says otherwise: its members equal the integers they carry, and a meet of two of
  them stays conservative.
- **An enumeration against the union of its members**, when every instance of
  the class really is one of them: an `Enum` that is not a `Flag`, carrying at
  least one member, whose members compare by identity. Then `Colour` and
  `Literal[Colour.RED, Colour.GREEN]` are one set, decided in both directions;
  the class is still what `repr` prints and what a failure names. The three
  exclusions are each a value that would stand against the union:

  | kind | the value it admits that `list(cls)` never yields |
  |---|---|
  | `Flag`, `IntFlag` | `P.A \| P.B` -- `\|` builds instances the class never listed |
  | an `Enum` with no members | a member of a subclass, since a memberless enum can still be subclassed |
  | `IntEnum`, `StrEnum` | nothing new, but its members equal the values behind them, so two of them are not two values |

  Each stays the `isinstance` atom it was, which is sound for every enumeration
  and merely less complete.
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
  A bound over a **float** base is a set of floats and is decided as one: which
  side a bound lands on is chosen by the base rather than by the operand's type,
  so `Annotated[float, Gt(0)]` carries the integer zero and still orders the
  floats. `nan` is outside every interval, which is the comparison Python makes.
  A base that is neither the whole numbers nor the floats alone stays undecided,
  because narrowing it to one component would give a smaller set than the schema
  denotes.
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
  decided below `dict[str, int]`. A key the supertype **requires** and the
  subtype does not declare refutes the inclusion however open the subtype is: a
  clause governs the keys a value carries and requires none, so the subtype
  holds a value without that key, and `dict[str, int]` is decided *not* below a
  record that requires one. A **meet** of two of them is empty when some key
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
  cancels a complement, `union` folds a join carrying a schema beside its own
  complement, and `intersection` folds the meet of that pair, all where the
  schema is built. So `complement(complement(int))` **is** `int` — one schema,
  which `repr` and `==` report and which a comparison is never asked about — a
  union covering the universe **is** `anything`, and a meet cancelling to
  nothing **is** `nothing`. A predicate and a hooked class are exempt: the law
  holds of sets, and neither is one.
  The decision procedure has no rule for either shape and never meets one built
  this way. A shape the fold does not reach is a different matter and is
  conservative (below).

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
assert intersection(
    Any, complement(Any)
).is_empty()  # Any is the top, spelled, so the law reaches it

json_value = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
assert json_value.is_valid({"a": [1, "x", {"b": None}]})
```

## Sound but conservative

Here valgebra is correct but not complete: it may answer `False` or "not empty"
for a relation that does in fact hold.

**A negative answer is one of two different things**, and the core tells them
apart even though the boundary does not. A relation can be *refuted* -- there is
a value of the one schema outside the other, so the answer is `False` and will
stay `False` however much the procedure improves -- or *declined*, which is the
list below: nothing was found either way, and the same query decides once the
representation reaches it. The three relations answer with all three values
inside the core and map both negatives to `False` at the boundary, because
`False` is what the contract promises and a third value at the surface would
make every caller handle a case the guarantee does not need. What the split buys
is that "not proven" is countable: a rule that starts refuting a relation it
used to decline is a change the tests see rather than one that hides behind an
unchanged `False`.

The list is short, and it is short for one reason. Two representations answer
these questions. The **rules** recurse over the schema tree, and where they
decline the **descriptor** is asked: it holds each kind as a set, so `a ≤ b` is
`a ∧ ¬b = ∅` and the answer comes out of the sets rather than out of a rule about
the shape. What is left below is what the descriptor cannot hold.

- **Recursion, past one unfolding.** A reference is a cycle and a finite set
  representation has no room for one, so a recursive schema is lowered by
  unfolding its body **once** and putting a bound where the reference was — the
  top where the schema is used positively, the bottom under a complement, which
  is what keeps a difference sound. That decides everything about the kinds a
  fixpoint admits: `bytes` shares no value with a JSON value, and `bytes` is
  below its complement. What one unfolding does not reach is a relation that
  needs the body *twice* — a fixpoint below a differently-written fixpoint whose
  bodies only agree after two steps — and there the coinductive rule is the
  whole of the answer.

- **A length bound over a set or a dict.** A length is not a word's alone, and
  two of the kinds that have one now state it: a word's length is a pattern over
  its alphabet, and a *sequence's* is "any element, that many times", which the
  automaton holds like any other shape. So `Annotated[tuple[int, int],
  MinLen(3)]` is decided empty and `Annotated[list[int], MaxLen(0)]` is the
  empty list. A set and a dict have a length their components do not count, and
  a bound over one of those refuses rather than being lowered as if it did.

- **A schema too large to build.** The descriptor is bounded three ways: the
  nodes it will read, the nesting it will descend, and the work a build may
  spend. Past any of them it refuses, and the caller keeps the rules' answer.
  None of these is a statement about the schema -- the same schema decides under
  a larger bound -- and each exists because building a descriptor beside a
  verdict the rules already reached is work whose result is discarded. Reading
  `dev/01-schema-ir.md` gives the measurements the bounds are set from.

- **A predicate.** Its satisfiability is undecidable (below), so neither
  representation reasons about one.

Every relation named here is a strict expected failure in
`tests/test_completeness_ledger.py`, so the day one is decided the mark fails and
the entry leaves both the ledger and this list.

```python
from typing import Annotated, Literal, NamedTuple

import annotated_types as at

from valgebra import (
    Regex,
    Validator,
    anything,
    complement,
    intersection,
    nothing,
    recursive,
    union,
)


class Pair(NamedTuple):
    x: int
    y: int


# A length bound over a set or a dict is opaque: their representations do not
# count one. Over a word or a sequence it is decided.
assert not Validator(Annotated[set[int], at.MinLen(3)]).is_empty()
assert Validator(Annotated[tuple[int, int], at.MinLen(3)]).is_empty()
# A named tuple's positions are its fields, and the schema says so, so the
# relation is structural.
assert Validator(Pair).is_subtype_of(tuple[int, int])
# A recursive schema: the laws reach it, and one unfolding decides the kinds its
# body admits. What one unfolding does not reach is a relation needing the body
# twice -- here, that every value of the integer tree is a value of the list
# tree, which holds and is not proved.
mu = lambda: Validator(recursive(lambda t: union(int, list[t])))  # noqa: E731
assert intersection(mu(), complement(mu())).is_empty()
assert intersection(mu(), str).is_empty()
assert not mu().is_subtype_of(recursive(lambda t: union(int, list[list[t]])))

# Everything else here decides, on the sets rather than by a rule.
pattern = Validator(Annotated[str, Regex("a")])
assert pattern.is_subtype_of(Annotated[str, Regex("ab?")])  # L(a) <= L(ab?)
assert pattern.is_subtype_of(Literal["a"])  # L(a) is exactly {"a"}
assert Validator(Annotated[int, at.MultipleOf(4)]).is_subtype_of(
    Annotated[int, at.MultipleOf(2)]
)
assert Validator(bool).is_subtype_of(Annotated[int, at.Ge(0)])
assert Validator(bool).is_subtype_of(Literal[True, False])
assert Validator({"a": int}).is_subtype_of(dict[Literal["a"], int])

# A respelling denotes the same set, and the sets are what the relation reads --
# even though the laws construction settles do not reach this one. `A | (A & B)`
# is `A` by absorption, which needs a containment to see, and containment is the
# decision rather than a law.
record = Validator({"a": int})
respelled = union(record, intersection(record, Validator(str)))
assert respelled != record
assert record.is_subtype_of(respelled)
assert respelled.is_subtype_of(record)
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

One shape avoids the product entirely, and it is the one a contract writes most:
a union of nothing but **literals**. A literal denotes a singleton, so such a
union denotes a *finite set of values*, and inclusion between two finite sets is
membership of every value of one in the other. That is decided by lookup rather
than by distribution, and it is exact in both directions — every value found is
a proof, and one value found nowhere is a refutation, since it is in the subject
and outside the other schema. Two tables of ten thousand codes each are decided
in a few milliseconds, in either direction, whether they were written out
separately or one was built from the other:

```python
from typing import Literal

from valgebra import Validator

codes = Validator(Literal[tuple(range(10_000))])
wider = Validator(Literal[tuple(range(10_001))])
shifted = Validator(Literal[tuple(range(1, 10_001))])
backwards = Validator(Literal[tuple(reversed(range(10_000)))])

assert codes.relation_to(wider) == "subset"
assert codes.relation_to(shifted) == "not_subset"  # 0 is in one and not the other
assert codes.relation_to(backwards) == "subset"  # the same set, written the other way
```

"However they were written" is a claim about the pools: each validator numbers
its constants in the order it met them, relating two validators renumbers one
pool into the other, and a table written backwards is renumbered backwards. A
member list is read as a set only in the canonical order its constructor leaves
it in, so the transform that renumbers one sorts it again
(`crates/valgebra-core/src/ir/transform.rs`, `mapped_member_set`). Before it
did, two tables that agreed on nothing about where each constant sat were
distributed against each other as if the rule were not there.

The refutation is the bindings' to give: two constants at two pool positions are
two *values* only where their type's equality can be trusted, and a constant
that does not equal itself — `float("nan")` — denotes no value at all, so
`Literal[float("nan")]` is the empty set and is below everything. Where the
equality cannot be trusted, the relation stays undecided rather than guessing.

What is left under the budget is the Boolean tower: a deeply nested combination
of unions, meets and complements, where subtyping distributes over both sides
and the work is a product of the branches. A `False` there may mean "not proven
within the bound" rather than "not a subtype"; on anything else it means the
relation is outside the decided fragment above. The bound is the price of a
procedure with no memo over its goals, and writing one is the work the theory
names.
