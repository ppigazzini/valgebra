---
description: What subtyping, equivalence, and emptiness decide exactly versus conservatively.
---

# The decidability boundary

valgebra compares schemas as sets: `is_subtype_of` is set inclusion, `is_equivalent` is
mutual inclusion, and `is_empty` reports an unsatisfiable schema. The relation is
`s <= t` exactly when `s` and `not t` share no value, so every comparison reduces
to an emptiness test (see [foundations](13-foundations.md)).

Every answer is **sound** relative to the [trust base](14-soundness.md#what-this-argument-assumes).
A `True` from `is_subtype_of`/`is_equivalent`, or a `True` from `is_empty`, is
correct for every value inside it: one that answers Python's questions as a
function of itself, whose `__class__` is its type, and, for an enumeration's
instance, that the enumeration made, among the rest the trust base lists. Where
valgebra cannot yet prove a relation it answers conservatively — `False`, or
"not empty" — never a wrong `True`. So a positive answer is a guarantee, and a
negative answer is "no, or not proven".

This page states which queries valgebra decides completely, which stay
conservative, and which are undecidable at runtime and so are rejected or treated
opaquely by necessity.

`is_empty` answers in two, and the question behind it answers in three. A
`False` from `is_empty` covers both "a value is in this schema" and "no reading
proves it holds none", which are different claims — one names a value, the
other names a limit. Emptiness is `s <= nothing`, so `relation_to(nothing)`
is where the third answer is:

```python
from typing import Annotated

import annotated_types as at

from valgebra import Validator, nothing

opaque = Validator(Annotated[int, at.Predicate(lambda value: value > 0)])
assert opaque.is_empty() is False  # not proven empty
assert opaque.relation_to(nothing) == "undecided"  # and not proven inhabited
assert Validator(int).relation_to(nothing) == "not_subset"  # a value says so
```

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
  structural kinds, not only the scalars. A member is read by its kinds through
  a union and a meet: `int` met with a union of
  `intersection(str, union(complement(str), Annotated[str, Regex(p)]))` for
  two patterns -- each applied to strings alone, as JSON Schema applies a
  `pattern` -- is empty whatever the patterns are. `Any` is the top, spelled, so the rule
  reaches it like any other set: `intersection(Any, complement(Any))` is decided
  empty.
- **A bare container class and its parameterized form.** `list` and
  `list[object]` are one schema: an unparameterized generic names its kind's
  whole set, which is what the typing spec assigns it. `tuple`, `set`,
  `frozenset` and `dict` read the same way, as `str` and `int` do.
- **Class and literal inclusion.** A class is a subtype of its base *classes*,
  by `issubclass`, and a literal is a subtype of any schema its constant is a
  member of, where the constant is the literal's one value or the schema tells
  no two equal values apart: `Literal[0.0]` also holds `-0.0`, so a predicate
  that refuses `-0.0` leaves it undecided ([Predicates](#undecidable-at-runtime)). A
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
  direction. A class whose own `__slots__` add a slot lays down a layout as a
  builtin does, and Python refuses a class deriving from two layouts unless one
  extends the other: two such classes neither of which derives from the other
  share no instance, so their meet is empty and each is below the other's
  complement, and one laid down over no builtin is below the complement of
  every builtin kind. An empty `__slots__`, or one naming only `__dict__` and
  `__weakref__`, lays down nothing. A type defined in C lays down a layout
  where its instance is larger than its base's, as Python reads it: `complex`
  meets no JSON kind and is below the complement of each, and `ValueError`
  meets `TypeError`, both carrying the layout `BaseException` lays down. PyPy
  names no size, so there a type defined in C is read by its slots alone and
  such a pair is `undecided`. The `@disjoint_base` decorator of PEP 800
  is a promise to a checker that the runtime does not enforce, and it is not
  read.
- **A meet holding a literal.** A meet holding a literal, or a union of them,
  is read one constant at a time by the same walk: it is inhabited where a
  constant is a member of every other member, and empty where each constant is
  refused by one of them, under the condition on equal values above. So a
  literal met with the complement of its meet with a recursive universe is
  empty, at a depth the set representation does not reach. A value the walk
  cannot read, one at which a predicate raises, is refused by a set and by its
  complement alike, and decides nothing.

    ```python
    from typing import Literal

    from valgebra import complement, intersection, nothing, recursive, union

    json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    paid = Literal["paid"]
    assert intersection(paid, complement(intersection(paid, json))).is_empty()
    # `paid & list[int]` is empty, so its complement holds "paid".
    inhabited = intersection(paid, complement(intersection(paid, list[int])))
    assert inhabited.relation_to(nothing) == "not_subset"
    ```

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
  the class really is one of them -- made through the class, which
  `object.__new__` steps past
  ([the trust base](14-soundness.md#what-this-argument-assumes)): an `Enum`
  that is not a `Flag`, carrying at least one member, whose members compare by
  identity. Then `Color` and
  `Literal[Color.RED, Color.GREEN]` are one set, decided in both directions;
  the class is still what `repr` prints and what a failure names. The three
  exclusions are each a value that would stand against the union:

  | kind | the value it admits that `list(cls)` never yields |
  |---|---|
  | `Flag`, `IntFlag` | `P.A \| P.B` -- `\|` builds instances the class never listed |
  | an `Enum` with no members | a member of a subclass, since a memberless enum can still be subclassed |
  | `IntEnum`, `StrEnum` | nothing new, but its members equal the values behind them, so two of them are not two values |

  Each stays an `isinstance` atom, which is sound for every enumeration and
  merely less complete.
- **Divisibility between two moduli**, where it holds. Every multiple of `a` is
  a multiple of `b` when `b` divides `a`, so the two steps settle the inclusion
  between them and the size of either is beside the point: `MultipleOf(5000)`
  is decided below `MultipleOf(2500)`. The question is Python's `%`, asked of
  the two steps, and it carries to the values where `%` reads a value and both
  steps as one kind of number: two `int` steps over `int`, two `float` steps
  over `int` or `float`, and over `float` any two steps a float holds exactly.
  The *other* direction is a refutation, and the subject's own multiples name
  its value where its constraints are its step and integer bounds: of two
  consecutive multiples of `a`, a step that does not divide `a` divides at most
  one. A subject narrowed by anything else leaves the value to the residues, the
  conservative entry below:

    ```python
    from typing import Annotated

    from annotated_types import MultipleOf

    from valgebra import Validator

    fives = Validator(Annotated[int, MultipleOf(5000)])
    assert fives.relation_to(Annotated[int, MultipleOf(2500)]) == "subset"
    assert fives.relation_to(Annotated[int, MultipleOf(3)]) == "not_subset"  # 5000
    ```
- **Refinements.** A refinement is a subtype of its base and of a refinement with
  looser bounds — a tighter numeric or length bound entails a looser one, not only
  a verbatim-contained constraint set; a bound conjunction that cannot be satisfied
  — a lower bound above an upper bound, a minimum length above a maximum, or a
  length a sequence's shape cannot take — is empty. Where the values are bounded
  to the integers the bounds count them, so an interval that skips every integer
  — `Annotated[int, Gt(0), Lt(1)]` — is empty even though its endpoints are
  ordered. That holds however the meet is spelled: on one refinement, or across
  an intersection whose members bound it, since an intersection is a subset of
  every member. A `bool` base counts too, because it subclasses `int`; a `float`
  base holds `0.5` between the same bounds, so they are not empty.
  A bound over a **float** base is a set of floats and is decided as one: which
  side a bound lands on is chosen by the base rather than by the operand's type,
  so `Annotated[float, Gt(0)]` carries the integer zero and still orders the
  floats. `nan` is outside every interval, which is the comparison Python makes.
  No float lies between a float and the next one, so `Gt(1.0)` is decided equal
  to `Ge(math.nextafter(1.0, math.inf))`, and a pair of bounds with no float
  between them is empty.
  A base that is neither the whole numbers nor the floats alone stays undecided,
  because narrowing it to one component would give a smaller set than the schema
  denotes.
- **Sequences.** Homogeneous, fixed-length, and prefix-plus-tail lists and tuples,
  with the container as part of the type (a list is never a tuple). Every sequence
  schema valgebra builds takes this linear shape, so inclusion *between two
  sequence schemas* is decided completely — a bare `list` among them, since it is
  the sequence node its kind's top spells (above). A repeated element outside
  the supertype's repeated element refutes whatever the prefix decides, where
  the subtype is proved to hold a value: `[int, int, ...]` is decided not below
  `[Y, str, ...]` on `[1, 2]`, for a `Y` no rule can read. A
  **fixed-length** sequence is also decided against a union of
  fixed-length ones it splits across, where no single branch contains it:
  `tuple[int | str, int]` is below `tuple[int, int] | tuple[str, int]`. The rule
  needs a fixed component count, so a homogeneous or variadic sequence — a star,
  matching every length — is not decomposed. The split narrows the product by
  one branch a level, so it runs under the work budget and the relation's depth
  bound both, and its reach grows with components and branches together: six
  components each `int | str` are decided below their sixty-four corners, seven
  decline against their 128, and a two-component tuple declines against a few
  dozen branches that each fix its first position.
- **Sets and frozensets.** By element inclusion.
- **Records and mappings.** Closed-record width, depth, and required-ness; pure
  mappings with several key-pattern clauses, each subtype clause subsumed by a
  supertype clause or covered by several together over whole kinds, which the
  set representation decides: `dict[int | str, int]` is below `{int: int, str:
  int}`; and a record mixed with a catch-all when the subtype carries at least
  the supertype's fields, or when a field the subtype lacks is optional in the
  supertype and the subtype's catch-all covers its value type (each extra or
  optional field covered by a catch-all over all string keys). A closed record
  is compared against a catch-all mapping by the same rule, so `{"x": int}` is
  decided below `dict[str, int]`. A key the supertype **requires** and the
  subtype does not declare refutes the inclusion however open the subtype is: a
  clause governs the keys a value carries and requires none, so the subtype
  holds a value without that key, and `dict[str, int]` is decided *not* below a
  record that requires one. A key the subtype declares optional refutes the same
  way, whatever the field's types decide of each other: `{"a?": int}` is decided
  not below `{"a": Y}` on `{}`. A **meet** of two of them is empty when some key
  one side requires cannot hold: because the types the two give it share no value,
  or because the other side is closed and does not declare it. Only a required key
  can do this — a meet of two mappings, or of two optional fields, always contains
  the empty dict.

    A record is decided against a **union of records it splits across** as
    well, by the fixed sequence's rule one kind over: Castagna's
    backtrack-free split for records, which narrows the record by one member
    a level, a label at a time. `{"a": int | str, "b": int | str}` is below
    the four records that fix both keys; a record whose tag takes three
    literals is below the three records that each fix one; and `{"a": int,
    "c?": bool}` is below a record requiring `c` beside one leaving it out. A
    member asking for a key the subject leaves free refutes: `{"kind": "a" |
    "b" | "c", "order": int}` is decided not below three records fixing the
    tag where one of them also requires an `amount`. Its reach is the
    sequence's -- six fields against their sixty-four corners, seven declining
    against their 128 ([where that bites](#sound-but-conservative)). A member
    whose clauses the rule cannot place in the key partition -- a literal key,
    a key holding part of a kind, two clauses reading one kind -- leaves the
    pair to the set representation. So does a member the subject's clause for
    `None` or `bool` keys escapes, as far as a refutation goes: those parts
    hold one key and two, too few for the fresh key the rule's refutation
    adds to its witness.

    Read as emptiness, a record minus the records that cover it is empty, one
    complement at a time, and a record one of them leaves uncovered holds a
    value: an open record whose key takes sixty-four constants, minus the
    record of each, is proved empty.

    Records met in a meet are one record, their field-by-field meet: a field
    holds what every record gives its name, through the field or through the
    clause covering it, and may be missing only where every record lets it
    be. So the meet of open records is related as one record, below a record
    and minus records alike, and a member of another shape beside them, a
    predicate say, leaves a proof standing. The meet holds a value where every
    field it requires does -- the dict of those fields is in each record -- so
    a refutation of it stands on that value.

    ```python
    from typing import Literal

    from valgebra import complement, intersection, recursive, union

    json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    user = {"id": str, "tier": Literal["free", "pro"], str: json}
    required = {"tier": Literal["pro", "enterprise"], str: json}
    pro = {"tier": Literal["pro"], str: json}
    assert intersection(user, required).relation_to(pro) == "subset"
    assert intersection(user, required, complement(pro)).is_empty()
    free = {"tier": Literal["free"], str: json}
    assert intersection(user, required).relation_to(free) == "not_subset"
    ```

    A clause keyed by **literals** is read one constant at a time, each
    constant a key the other map reads through its field of that name or
    through the clause holding it. So `dict[Literal["a"], int]` is below
    `{"a?": int}` and refuted against `{"a?": str}` on `{"a": 1}`, and two
    mappings keyed by one literal are refuted under it at any depth, as two
    `dict[str, ...]` chains are. A record is refuted against a mapping keyed by
    literals on a `str` subclass's key equal to a field's name: the field reads
    it, since the walk looks the name up, and a literal, which pins the type,
    does not.

    ```python
    from typing import Literal

    from valgebra import Validator

    assert Validator(dict[Literal["a"], int]).is_subtype_of({"a?": int})
    assert Validator({"a": int}).relation_to(dict[Literal["a"], int]) == "not_subset"
    assert Validator(dict[str, int]).relation_to(dict[Literal["a"], int]) == "not_subset"
    ```

- **A difference written as one complemented union.** `a ∧ ¬(b ∨ c)` and
  `a ∧ ¬b ∧ ¬c` are one set, and both are decided where either is: no spelling
  of a difference is the harder one by construction. That is worth stating
  because the set representation is bounded — a union of dicts, objects or sets
  holds at most a fixed number of atoms — and complementing a union of *n* parts
  multiplies *n* complements together, which is a width the answer rarely has.
  A meet against such a union removes one part at a time, so the width the bound
  sees is the width of the answer rather than of the widest intermediate.

    The bound stands. A difference wide enough to reach it reaches it, and
    which of two spellings gets there first depends on the order the parts
    multiply, so a relation the sets decline under one spelling may be decided
    under another. No spelling is *systematically* the one that declines.
- **A complemented union or meet, as the subject.** `¬(a ∨ b)` is `¬a ∧ ¬b`,
  and `¬(a ∧ b)` is `¬a ∨ ¬b`. The subject is read with its complement pushed
  one level in, and decided where that reading is the supertype or has the
  supertype among its members, so a complement of a union is below the meet
  of the complements and below what a complemented member's complement is:

    ```python
    from valgebra import Validator, complement, intersection, union

    nested = Validator(list[list[int]])
    assert complement(union(str, complement(nested))).is_subtype_of(nested)
    assert complement(union(int, str)).is_subtype_of(
        intersection(complement(int), complement(str))
    )
    ```

    Only that reading is taken: a complement pushed in and then compared by
    every rule would be read twice wherever the answer is a decline.
- **A length bound over a base that takes any length.** A string and a bytes
  take any, so a bound their lengths admit is met by a value: `Annotated[str,
  MinLen(1)]` is the non-empty string and is decided to have one. A list or a
  tuple takes any length by repeating one element, so its element answers
  instead: a bound of zero is met by the empty container whatever the element
  admits, and a longer one by as many copies of an element as it asks for. A
  set holds each member once, so a bound over one asks how many values its
  element has: `set[bool]` under `MinLen(3)` is empty, `set[int]` under it has
  a value, and an element the rules cannot count declines. A dict's bound is
  not counted, and declines. A bound over a list is an automaton with a
  state per element, and the sets compare two such bounds up to about 1,340
  elements over a list of integers before the work a relation may spend runs
  out; past that they decline. That decides a
  fixpoint every unfolding of which needs one more element -- each element is a
  value of the fixpoint, and no finite value satisfies it. A bound over a
  *fixed* position, or over a base whose values have no length at all, is not
  this shape and stays conservative.
- **Two schemas that share no value.** The inclusion is *refuted*, whatever
  either side holds: every value of the subject is outside the supertype. The
  disjointness read here is the one the concrete types settle -- two distinct
  builtin scalars, two distinct container kinds -- so `list[int]` is decided not
  below `tuple[int, int]`, and a mapping not below a list. A subject with no
  value is the exception the reading catches: it is below both. A **class**
  joins the comparison through the bindings, which read the builtin it derives
  from: a class built on `dict` holds mappings and nothing else, since a
  subclass inherits that layout and cannot lay down a second, so a list is
  decided not below it. A class built on no builtin is *not* read this way, and
  the reason is a value: a class deriving from that one and from `str` is a
  string and an instance of it. A meet is read by a member of its own kind, or
  by a member that is a union of such: `(int | float) & ~bool` holds numbers
  whatever bounds its other members carry, so a list of anything meets it in
  nothing.
- **A kind the supertype holds none of.** Every kind is inhabited, so a subject
  that holds every value of a kind the supertype holds none of is outside it at
  a value of that kind. The universe holds every value of every kind, and a
  complement every value of a kind its inner schema holds none of. A union
  holds none of a kind where each branch holds none, and a reference none
  where its definition holds none:

    ```python
    from valgebra import Validator, anything, complement, recursive, union

    json_value = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    assert Validator(anything).relation_to(json_value) == "not_subset"  # b""
    assert complement(json_value).relation_to(json_value) == "not_subset"
    assert Validator(anything).relation_to(complement(json_value)) == "not_subset"  # None
    ```

- **A pair asked another way.** Where every rule declines a pair as it is
  written, it is asked as the algebra respells it. A difference of the two that
  holds every value of some kind is a value of the subject outside the
  supertype, whatever shape the subject has, and against a complement `~c` the
  pair is also asked as the meet with `c` below the empty set and as `c` below
  the complement of the subject:

    ```python
    from valgebra import Validator, anything, complement, recursive, union

    json_value = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    # the empty list is a JSON array
    assert Validator(anything).relation_to(complement([json_value])) == "not_subset"
    ```

- **A subject outside a base.** A refinement is a subset of its base, so a
  subject the base refutes is refuted against every refinement of it -- the
  same value settles both. The proof does not carry, since being inside the
  base says nothing about the constraints, which is why `list[int]` below
  `Annotated[list[int], MinLen(2)]` is left to the set representation while
  `list[str]` below it is refuted by a rule.
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
- **A meet with a recursive reference.** A reference denotes its definition,
  so a meet with one is the definition met with the rest, and a definition that
  is a union is the union of its branches each met with the rest. The meet of
  two records then asks the meet of each key's types, which unfolds the
  reference again where a key names it, and a meet reached again while it is
  being decided is read as empty: a value is finite, so one that needed a value
  of the same meet below it would need a smaller one. The meet is its own goal,
  never the reference's -- the key of `t & {"a": None | {"a": None}}`, for `t =
  None | {"a": t}`, asks `t & (None | {"a": None})`, which `{"a": None}` is in,
  and reading that `t` as the reference already open would read the meet
  empty. Emptiness is decided wherever the meet reads, and a value is named
  where its records meet in one record whose required fields hold values, open
  or closed: `intersection(chain, Validator({"next": None}).open())`, which
  `{"next": None}` is in, is `not_subset` of `nothing`. A meet with a class
  answers `"undecided"` (below). A reference written
  inside a key's union, the way a record that names itself is spelled, decides
  the same way:

    ```python
    from valgebra import intersection, nothing, recursive, union

    chain = recursive(lambda t: union(None, {"next": t}))
    assert intersection(chain, {"next": int}).is_empty()
    one_value = intersection(chain, {"next": {"next": None}})
    assert one_value.relation_to(nothing) == "not_subset"
    link = recursive(lambda t: {"next": union(t, None)})
    assert intersection(link, {"next": {"next": int}}).is_empty()
    ```
- **A meet of a few unions.** A meet holding a union is the union of the
  meets each branch makes with the rest, so a meet no rule reads whole is read
  one such clause at a time where its unions span four clauses at most -- the
  product of their widths, no branch of them a meet or a union. A translation
  from JSON Schema applies a keyword to its kind alone, so an object schema
  with two keywords is `OBJ & (~OBJ | R) & (~OBJ | Q)`: two unions of two, and
  each `~OBJ` meets `OBJ` in nothing. Two tags fixed two ways meet in nothing,
  and a tag beside a field it requires holds the dict with that field. A meet
  of more unions, or of a union with a meet for a branch, is left to the set
  representation, and is decided within the depth it descends.

    ```python
    from typing import Literal

    from valgebra import complement, intersection, nothing, recursive, union

    json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    obj = {str: json}
    t0 = union(complement(obj), {"tag": Literal["t0"], str: json})
    t1 = union(complement(obj), {"tag": Literal["t1"], str: json})
    required = union(complement(obj), {"tag": json, str: json})
    assert intersection(obj, t0, t1).is_empty()
    tagged = intersection(obj, t0, required)
    assert tagged.is_valid({"tag": "t0"})
    assert tagged.relation_to(nothing) == "not_subset"
    ```
- **The complement laws, where the constructors reach them.** `complement`
  cancels a complement, `union` folds a join carrying a schema beside its own
  complement, and `intersection` folds the meet of that pair, all where the
  schema is built. So `complement(complement(int))` **is** `int` — one schema,
  which `repr` and `==` report and which a comparison is never asked about —
  `int | ~int` **is** `anything`, and `int & ~int` **is** `nothing`.
  `complement(complement(X))` folds whatever `X` is. The two folds over a pair
  read two occurrences of one schema as one set, so they decline wherever the
  constructor cannot see that it is one: a schema holding a predicate or any
  class, `list[C] | ~list[C]` included. The constructor holds no class object
  to ask whether a metaclass hook answers
  `isinstance` — `Schema::union_within` and `Schema::meet_within` in
  `crates/valgebra-core/src/ir.rs` pass `NoLeafRelations`, and
  `denotes_a_set_within` in `crates/valgebra-core/src/oracle.rs` reads a class
  as a set only on the oracle's word. A plain class denotes a set all the same,
  and the law holds of it: `intersection(C, complement(C))` keeps its spelling
  and `is_empty()` decides it `True`. A union holding a union beside that
  union's complement folds too, since construction reads the flattened
  members: `int | str | ~(int | str)` **is** `anything`.
  The decision procedure has no rule for a folded shape and never meets one
  built this way. A shape the folds do not reach is the relations' to decide.

```python
from typing import Annotated, Any

import annotated_types as at

from valgebra import Validator, complement, intersection, recursive, union

assert Validator(bool).is_subtype_of(int)  # bool is a subtype of int
assert Validator(1).is_subtype_of(int)  # a literal is a member of int
assert Validator(Annotated[int, at.Ge(0)]).is_subtype_of(int)  # refinement <= base
assert Validator(Annotated[int, at.Ge(10), at.Le(0)]).is_empty()  # no such int
assert Validator(
    Annotated[int, at.Gt(0), at.Lt(1)]
).is_empty()  # no int strictly between
assert not Validator(Annotated[float, at.Gt(0), at.Lt(1)]).is_empty()  # 0.5 is between
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
    tuple[int, int] | tuple[str, int]
)  # a product splits across branches
assert Validator(
    recursive(lambda t: Annotated[list[t], at.MinLen(1)])
).is_empty()  # every unfolding needs one more element
assert intersection({"a": int}, {"a": str}).is_empty()  # 'a' cannot hold both
assert not intersection(
    {"a?": int}, {"a?": str}
).is_empty()  # the empty dict is in both
assert Validator(bool | int).is_equivalent(int)  # bool | int is just int
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

## The one assumption: a class the bindings can read has an instance

Every other answer here rests on the value model alone. This one rests on an
assumption, and it is the only place a `"not_subset"` can be wrong. No `True`
can: a proof is a proof, and the assumption is only ever read to believe a
*refutation*.

A class is opaque: what it holds is `isinstance`, and the library reads the
class hierarchy rather than running it. A class whose metaclass leaves
`isinstance` and `issubclass` alone is taken to **hold at least one object**,
which is what lets `A` be reported not below `B` for two classes neither
deriving from the other -- the difference `A ∧ ¬B` is read as holding a value.
A class no value can instantiate (a `__new__` that always raises, an abstract
class with no concrete subclass) is empty, and is below everything; the library
reports it not below, because the assumption says otherwise.

**It licenses one object, not one per kind.** A class that lays down no builtin
layout confines its instances to no kind, so it constrains values of every
kind: a subclass of it may derive from `str` as easily as from nothing. That
placement is right for inclusion -- `Plain` is not below the complement of
`int` -- and it is not a value. An *integer* that is an instance of `Plain`
exists only if some class derives from both, and which classes exist is not
something a snapshot of the order can say. So `Plain` against the complement of
a kind is **undecided**, and the same in reverse:

```python
from valgebra import Validator, complement


class Plain:
    pass


class Laid(str):
    __slots__ = ()


assert Validator(Plain).relation_to(complement(int)) == "undecided"
assert Validator(int).relation_to(complement(Plain)) == "undecided"

# A class laid out as the kind is the case this is not conservative about:
# every instance of it is a string, so it has a value on that line.
assert Validator(Laid).relation_to(complement(str)) == "not_subset"
assert Validator(Laid).relation_to(complement(int)) == "subset"
```

The assumption is the set representation's open world, which both readings
share. Neither decides it; the bindings could not answer it without running a
constructor at decision time, which is the same code the pure-metaclass test
exists to refuse. What it buys is every relation between two classes decided by
a rule, without building a set representation.

**A declared attribute does not widen this.** A dataclass compiles to the
class beside a record of its fields, and the record narrows the class's
instances rather than reaching outside them. A `NamedTuple` compiles to the
class met with the tuple its fields lay out, with no record beside it: it is
laid out as `tuple` and decides as a kind does, and a protocol over its fields
is not proved above it. So a relation that declines beside one declines for the
class: `int ≤ ~C` is "not proven" for a dataclass and for a class with no
fields alike, and a class laid out as a kind decides either way. The record is
not a second source of conservatism, and in the one place it tells two schemas
apart it makes a refutation reachable that the class alone leaves open. A
`Protocol` is the record with no class beside it, and relates as records do: a
dataclass declaring its members, each at least as narrowly, is below it, and a
protocol with more members is below one with fewer. Against a builtin kind it
declines in the direction that would stand on a value of the kind: an exact
`int` carries no `__dict__`, and whether it has an attribute is `int`'s own
namespace -- it has a `real`, it has no `x` -- which the relations do not
hold, so `int` against a protocol of `x` is undecided though `1` fails it.

**The object carries only what its class leaves room for.** The one object the
assumption licenses is a *direct* instance of the class, and a refutation about
a class met with a record stands on that instance holding a value in each
field. The class decides whether it can, read off the namespaces of its
`__mro__` once per relation: a name defined as a property or another data
descriptor holds what its getter returns, a name defined any other way cannot
be missing, an instance laid out without a `__dict__` holds nothing its slots
do not name, a `__getattribute__` other than the generic lookup answers every
lookup, and a `__getattr__` answers every lookup that finds nothing -- so beside
one an unset slot or a name the `__dict__` lacks is never missing, and a name
neither can hold is the hook's answer alone. Where the class leaves no room,
the pair is **undecided**: a subclass may carry the record, and may not exist.
A dataclass whose field is a descriptor, or which defines `__getattribute__`,
is where an annotation reaches it:

```python
from dataclasses import dataclass

from valgebra import Validator


@dataclass
class Plain:
    x: int


@dataclass
class Traced:
    x: int

    def __getattribute__(self, name: str) -> object:
        return super().__getattribute__(name)


assert Validator(Plain).relation_to(Validator(str)) == "not_subset"
assert Validator(Traced).relation_to(Validator(str)) == "undecided"
```

## Sound but conservative

Here valgebra is correct but not complete: it may answer `False` or "not empty"
for a relation that does in fact hold.

**A negative answer is one of two different things**, and `relation_to` tells
them apart where the boolean methods do not. A relation can be *refuted* --
there is a value of the one schema outside the other, so the answer is `False`
and will stay `False` however much the procedure improves -- or *declined*,
which is the list below: nothing was found either way, and the same query
decides once the representation reaches it. The three relations answer with all
three values inside the core. `is_subtype_of`, `is_equivalent` and `is_empty`
map both negatives to `False`, because `False` is what their contract promises;
`relation_to` reports the two apart, `"not_subset"` for a refutation and
`"undecided"` for a decline. What the split buys is that "not proven" is
countable: a rule that begins refuting a relation the procedure declines is a
change the tests see rather than one that hides behind an unchanged `False`.

The list is short, and it is short for one reason. Two representations answer
these questions. The **rules** recurse over the schema tree, and where they
decline the **descriptor** is asked: it holds each kind as a set, so `a ≤ b` is
`a ∧ ¬b = ∅` and the answer comes out of the sets rather than out of a rule about
the shape. What is left below is what the descriptor cannot hold.

- **A clause keyed by part of a kind.** The set representation partitions a
  dict's keys **by kind** and gives each part its own default, so a clause whose
  key is a kind, a literal, or a union or complement of kinds lands in parts:
  `complement(str)` opens every other kind and the part for a key of no listed
  kind, a class instance. A key holding *part* of a kind has no part to land in
  -- a tuple or a frozenset of narrower elements; a class; the complement of
  `Literal["a"]`, which holds a `str` subclass's key equal to `"a"` and not the
  exact one, where a label holds both -- and the lowering declines rather than
  approximating, so a relation about such a map is left to the rules. They
  refute one key kind at a time: a kind the subject's clause holds whole, read
  by one of the supertype's clauses or by none, names a key whose entry the two
  maps judge apart, so `{str: int}.open()` is outside `{str: int}` on `{7:
  "x"}`. What comes back "not proven" is a pair whose supertype reads a key
  kind in part, or with two clauses at once.

    The partition is the source's: it treats a record as a quasi-`K`-step
    function, where `K` is a *predefined finite partition* of the key domain
    and the catch-all is split across its parts, and it forbids overlapping
    domains in one map as "both a design and an implementation choice". A union
    of parts is a domain it admits, and a complement of whole kinds is the union
    of the other parts. A key holding part of a kind is not, and it may overlap
    another clause: the source says a theory of maps with overlapping domains is
    possible, and that comparing two records under it needs the machinery for
    comparing intersections of arrow types. valgebra admits such a key where the
    source has no part for it: the walk reads it exactly, and the comparison it
    needs is not built. That is a gap rather than a refusal, and what it costs
    is the proof a pair needs where a key kind is read in part -- which comes
    back "not proven" even between two spellings of every dict:

    ```python
    from typing import Literal

    from valgebra import Validator, anything, complement

    not_str = complement(Validator(str))
    free_the_rest = Validator({str: int, not_str: anything})
    # The walk is exact: membership never goes through the set representation.
    assert free_the_rest.is_valid({"a": 1, 7: "anything at all"})
    assert not free_the_rest.is_valid({"a": "not an int"})

    # A complement of a whole kind is the other kinds, and the sets decide it.
    assert Validator({str: anything, not_str: anything}).is_equivalent({object: object})
    # A key kind the supertype reads one way refutes: `{7: "x"}`.
    assert Validator({str: int}).open().relation_to({str: int}) == "not_subset"

    # A complement of a literal holds part of `str`. These two admit every dict.
    not_a = complement(Validator(Literal["a"]))
    long_way = Validator({Literal["a"]: anything, not_a: anything})
    every_dict = Validator(dict)
    assert long_way.is_valid({"a": 1, 7: "x"}) and every_dict.is_valid({"a": 1, 7: "x"})
    assert every_dict.relation_to(long_way) == "undecided"
    ```

    This is the shape `open` writes on a **mapping**: freeing the key-types a
    clause leaves over means a clause over the complement of the ones it claims,
    which is whole kinds for `dict[str, int]` and part of one for
    `dict[Literal["a"], int]`. Opening a *record* stays decided, because the
    regions it frees and the ones its clauses claim cover every key with one
    value between them, which is one catch-all clause rather than two.

- **A clause keyed by a float literal.** The partition names a literal key by
  its value, and it holds no float value: the typing spec admits none in a
  `Literal`. A map keyed by `Literal[1.0]` is left to the rules, which read a
  key by whole kinds: they refute where the literal holds none of a kind the
  other map's key holds, and a pair they do not decide comes back "not proven"
  although `{0.5: 0}` refutes it. A key literal of any other kind lands in its
  part.

    ```python
    from typing import Literal

    from valgebra import Validator

    keyed_by_a_float = Validator(dict[Literal[1.0], int])
    assert Validator(dict[int, int]).relation_to(keyed_by_a_float) == "not_subset"
    assert Validator(dict[float, int]).relation_to(keyed_by_a_float) == "undecided"
    assert Validator(dict[Literal[1], int]).relation_to(dict[int, int]) == "subset"
    ```

- **Recursion, past one unfolding.** A fixpoint as the **supertype** costs a
  *refutation* rather than an inclusion, and that is the sharper half. `a ≤ b`
  is `a ∧ ¬b = ∅`, so the supertype is where a schema stands under a
  complement, and a reference there is lowered to the bottom. The difference
  then contains the value that refutes the inclusion without being proved to
  contain it, so the answer is "not proven" where a value says plainly that it
  is false:

    ```python
    from valgebra import Validator, complement, intersection, recursive, union

    chain = recursive(lambda t: union(None, {"next": t}))
    off_chain = Validator({"next": intersection(dict, complement(chain))})
    assert not Validator(chain).is_valid({"next": {}})  # the walk is exact
    assert off_chain.is_valid({"next": {}})
    # and the relation declines rather than refuting
    assert off_chain.relation_to(chain) == "undecided"
    ```

    One unfolding does reach the kinds a fixpoint admits, and a subject that
    holds every value of a kind the fixpoint holds none of is refuted on them:
    `complement(int)` holds every string and the chain holds none, so it is
    `not_subset` of the chain at `"a"`, and so is a meet of complements holding
    every string. The pair above holds no such kind: its values are dicts, a
    kind the chain holds some of, and which of them are in the chain is a
    question two levels down.

    What the **rules** reach is narrower than that and is reached: a branch of a
    union the subject shares no value with is dropped before the rest is asked,
    so a chain of records beside a `None` branch is refuted by the record branch
    alone. Where *every* branch goes that way the subject is outside the whole
    union, which is a refutation rather than a decline as long as the subject
    holds a value at all. And a fixpoint's own unfolding is below it however
    its body is spelled: a member that is a meet is not placed by any member of
    the meet, so the reference is unfolded for it and the meet is read as the
    branch of the definition it is.

    ```python
    from valgebra import Validator, intersection, recursive, union

    meet = intersection(int, union(float, bool))
    tree = recursive(lambda t: union(meet, list[t]))
    assert Validator(union(meet, list[tree])).relation_to(tree) == "subset"
    assert tree.relation_to(union(meet, list[tree])) == "subset"
    ```

    The record rule refutes a map whatever the supertype's recursion. It reads
    one key kind at a time, and where the subject's clause holds a kind whole,
    a value the supertype's clause for that kind refuses -- or any value, where
    no clause of the supertype reads the kind -- makes an entry the two maps
    judge apart. So `{str: bytes}` is outside a JSON schema on `{"a": b""}`, and
    `dict` on `{7: None}`, whose key no clause of it reads.

    ```python
    from valgebra import Validator, recursive, union

    json_value = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    assert Validator({str: bytes}).relation_to(json_value) == "not_subset"
    assert Validator(dict).relation_to(json_value) == "not_subset"
    ```

    Branches are dropped where one of them **refutes**, which is the answer the
    narrowing carries back from the branch it leaves. A union no branch of
    which refutes narrows to a smaller union of declines, so the readings that
    drop branches -- a meet per branch, on the path already headed for the set
    representation -- are not spent on it, and the pair goes to that
    representation instead. What that gives up is the union whose branches the
    subject shares no value with and which none of them refutes: it is outside
    the union and is answered `"undecided"`. Reaching for it on every union
    adds a reading per branch to every union the decision path meets, which
    `scripts/perf_gate.py --decision-matrix` measures.

    The other half is the lowering. A reference is a cycle and a finite set
    representation has no room for one, so a recursive schema is lowered by
    unfolding its body **once** and putting a bound where the reference was —
    the top where the schema is used positively, the bottom under a complement,
    which is what keeps a difference sound. That decides everything about the
    kinds a fixpoint admits: `bytes` shares no value with a JSON value, and
    `bytes` is below its complement. What one unfolding does not reach is a
    relation that needs the body *twice* — a fixpoint below a
    differently-written fixpoint whose bodies only agree after two steps — and
    there the coinductive rule is the whole of the answer.


    **An unfolding spends depth.** The fixpoint's body sits one level below
    every member that names it, so a member met with a recursive universe under
    a complement nests past the depth a build descends. Beside the members a
    proof needs, such a member is left out of the reading and the proof stands,
    as a [predicate](#undecidable-at-runtime) is; where it is the member the
    proof needs, the relation declines, so `str & ~(str & json)` is undecided
    though `str & ~json` is proved:

    ```python
    from valgebra import complement, intersection, nothing, recursive, union

    json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
    assert intersection(str, complement(json)).relation_to(nothing) == "subset"
    nested = intersection(str, complement(intersection(str, json)))
    assert nested.relation_to(nothing) == "undecided"
    ```
- **An inhabited meet with a recursive reference, beside a class.** The
  meet's emptiness is decided ([above](#decided-exactly)), and a value is named
  where its records meet in one record whose required fields hold values, open
  records among them, so a meet with a class that holds a value is not proved
  to:

    ```python
    from valgebra import Validator, intersection, nothing, recursive, union


    class Node(dict):
        pass


    chain = recursive(lambda t: union(None, {"next": t}))
    opened = intersection(chain, Validator({"next": None}).open())
    assert opened.relation_to(nothing) == "not_subset"
    assert intersection(chain, Validator({"next": int}).open()).is_empty()
    met = intersection(chain, Node)
    assert met.is_valid(Node(next=None))
    assert met.relation_to(nothing) == "undecided"
    ```

- **A length bound over a set or a dict, in the sets.** A length is not a word's alone, and
  two of the kinds that have one state it: a word's length is a pattern over
  its alphabet, and a *sequence's* is "any element, that many times", which the
  automaton holds like any other shape. So `Annotated[tuple[int, int],
  MinLen(3)]` is decided empty and `Annotated[list[int], MaxLen(0)]` is the
  empty list. A set and a dict have a length their components do not count, and
  a bound over one of those refuses rather than being lowered as if it did.
  Whether a set under a bound *has a value* is a different question and the
  rules answer it, from how many values the element has, as the bullet on a
  length bound above says.

- **A set that needs two members Python's `==` makes one.** A set of `int | str`
  is not below `set[int] | set[str]`: `{1, "a"}` is in neither. That refutation
  stands on two members, and a Python set holds no two that compare equal --
  `{1, True}` is `{1}`, and so is `{1, 1.0}`. So a set refuted only by a member
  of each, where those members are one number in two kinds, is undecided rather
  than refuted: `set[Literal[1, True]]` is below `set[Literal[1]] |
  set[Literal[True]]`, since every set of it holds one member, and that pair
  answers `undecided`. Two members `==` keeps apart -- an `int` and a `str`, two
  integers, or two number kinds with other numbers to choose, as
  `frozenset[int | float]` against the frozensets of each -- are refuted.

- **An integer bound outside the 64-bit range.** The integer component carries
  its bounds as `i64`, so `Annotated[int, Ge(2**70)]` is held as the widest set
  the carrier spells rather than as the half-line it names. The schema validates
  exactly -- membership reads the Python integer -- and a *relation* between two
  such bounds declines: `Ge(2**70)` against `Ge(2**70 + 1)` is `undecided` in
  the direction a value refutes, and `subset` in the direction the carrier
  proves. Python's integers are unbounded and the carrier is not, which is a
  property of the representation rather than of the schema.

- **Two moduli `%` reads in different kinds of number.** Dividing is a fact
  about the steps, and `%` asks it of a value in the kind the value and the
  step meet in. `2**53 + 1` is a multiple of `4.0`, because `%` reads it as the
  float `2**53`, and not of `2`, which divides `4.0`. `2.0**53` is a multiple of
  `2**53 + 1`, which `%` reads as the same float, and not of `3`, which divides
  that step. `Decimal("2E+28")` is a multiple of `4` and raises against `2`,
  because its quotient by `2` outgrows the context's precision. So a pair of
  steps of two kinds, a step past `2**53` over `float`, and a base holding
  values of any other type -- a `Decimal`, a `Fraction`, `object` -- declines.

- **A meet of two moduli the representation cannot hold.** A `MultipleOf`
  becomes a residue class, and a class is materialized per residue up to a
  recorded period. Two steps *meet* at the period they share, so `MultipleOf(64)`
  with `MultipleOf(81)` is the multiples of 5,184 -- past the period although
  both steps are well inside it -- and a relation that turns on that meet
  declines. A rounded period would be wrong in one direction or the other, so it
  refuses instead.

    The **inclusion** between two moduli does not go that way and is decided
    above: the steps settle it between them, whatever their size, and so does
    the refutation where the subject's constraints are its step and integer
    bounds, `MultipleOf(2500)` being outside `MultipleOf(5000)` at 2500. What
    the period still bounds is a refutation the subject's other constraints take
    part in -- a float bound beside the step, or a meet of two steps -- since the
    value has to satisfy them too and naming it takes the residues.

- **A step over floats.** A `MultipleOf` becomes a residue class of the
  integers, and a float multiple of a step is no integer the classes hold, so a
  step over a `float` base has no set to land in, and the set representation
  refuses it rather than narrow the floats to integers. The rules decide two
  steps against each other and bounds against bounds (above), and what turns
  on a step and its bounds *together* declines: no float multiple of `1` lies
  strictly between 30 and 31, and over `float` that meet is not proved empty,
  where over `int` it is.

    ```python
    from typing import Annotated

    import annotated_types as at

    from valgebra import Validator, nothing

    floats = Annotated[float, at.MultipleOf(1), at.Gt(30), at.Lt(31)]
    integers = Annotated[int, at.MultipleOf(1), at.Gt(30), at.Lt(31)]
    assert Validator(floats).relation_to(nothing) == "undecided"
    assert Validator(integers).relation_to(nothing) == "subset"
    ```

- **A word boundary inside a pattern.** A pattern matches the whole text, so a
  `\b` or `\B` at either end of it asks about the first or the last character
  alone, and is read as that: `\bab\b` is the set `ab` is. One between two
  parts of a pattern asks about the characters on both sides of a position,
  which the set representation's automata do not read, so a relation turning
  on it declines -- `a\bb` included, although no string matches it.

    ```python
    from typing import Annotated

    from valgebra import Regex, Validator, nothing

    bounded = Validator(Annotated[str, Regex(r"\bab\b")])
    assert bounded.is_equivalent(Annotated[str, Regex("ab")])
    inside = Validator(Annotated[str, Regex(r"a\bb")])
    assert inside.relation_to(nothing) == "undecided"
    ```

- **A schema too large to build.** The descriptor is bounded three ways: the
  nodes it will read, the nesting it will descend, and the work a build may
  spend. Past any of them it refuses, and the caller keeps the rules' answer.
  None of these is a statement about the schema -- the same schema decides under
  a larger bound -- and each exists because building a descriptor beside a
  verdict the rules already reached is work whose result is discarded. Every
  bound is a row of the architecture table in the repository, with the gate that
  measures it beside it.

  **Where that bites is width.** A record whose fields each take two types is
  the union of the records that fix every field, and the record rule splits it
  across them as the product rule splits a tuple, so the two reach as far:
  six fields against their sixty-four corners are decided, and seven against
  their 128 spend the decision budget before the split finishes, with the
  difference past every bound a lowering has. The shape is the first an
  ordinary annotation reaches, so it is the number to know:

  ```python
  from valgebra import Validator, union


  def with_corners(width):
      names = [f"f{i}" for i in range(width)]
      corners = union(
          *[
              {name: (int, str)[mask >> i & 1] for i, name in enumerate(names)}
              for mask in range(1 << width)
          ]
      )
      return Validator(dict.fromkeys(names, int | str)), corners


  six, corners = with_corners(6)
  assert six.relation_to(corners) == "subset"
  seven, corners = with_corners(7)
  assert seven.relation_to(corners) == "undecided"
  ```

  The seventh field's `undecided` is the conservative answer and not a claim
  about the relation: it holds, and proving it takes a larger budget.

  **A union is asked a part at a time** where the pair is past a bound whole. A
  union subject is below a supertype where each member is and outside it where
  one member is, so each member is asked on its own; a union supertype is first
  narrowed to the branches the subject shares a value with. Records told apart
  by a tag meet one branch each, so a union of them is decided at any width:

  ```python
  from typing import Annotated, Literal

  from annotated_types import Ge

  from valgebra import Validator, union


  def tagged(width, shift=0):
      return union(
          *[
              Validator({"type": Literal[f"t{i}"], "v": Annotated[int, Ge(i + shift)]})
              for i in range(width)
          ]
      )


  # refuted at {"type": "t0", "v": 0}
  assert tagged(40).relation_to(tagged(40, shift=1)) == "not_subset"
  assert tagged(40, shift=1).relation_to(tagged(40)) == "subset"
  ```

  A union against a union is not asked a branch at a time: a member that lands
  in one branch of a supertype too wide to build is left `undecided`, since
  asking every member against every branch is the product of the two widths.

- **A predicate.** Its satisfiability is undecidable (below), so neither
  representation reasons about one -- and that is a statement about the
  *predicate*, not about every pair carrying one. A refinement is a subset of
  its base, so what surrounds the predicate still decides: a bounded integer
  against a predicate-refined list is refuted by the kinds, a list against its
  own bounded-and-predicated refinement by the empty list the bound leaves out,
  and a dataclass against a union with a predicate-refined branch by the kind
  that branch names. What the predicate costs is the other direction. A
  refutation stands on a value of the *subject*, and a subject carrying a
  predicate is never proven to have one, so the same pair reversed is undecided:
  `Annotated[int, Ge(0)] <= Annotated[list[int], Predicate(f)]` is refuted and
  its converse is not.

    A predicate beside the members of a meet hides none of their emptiness:
    the meet is below the members it holds, so where they share no value
    neither does the meet, and two patterns no string matches both of stay
    apart beside a predicate, whether written as three members or as one
    `Annotated` holding all three. That is a proof of emptiness only: the meet
    without the predicate holding a value says nothing of the meet with it.

    ```python
    from typing import Annotated

    from valgebra import Regex, Validator

    letters = Annotated[str, Regex("^a+$"), Regex("^b+$"), lambda s: len(s) > 2]
    assert Validator(letters).is_empty()
    ```

`tests/test_completeness_ledger.py` is where a relation that leaves this list
lands: it enumerates what the procedure must *decide* and fails in both
directions, so a relation that regresses to conservatism fails there and one
that becomes decided is added there. It also carries a strict expected-failure
mark for a relation that holds and is not decided, so the row fails on the day
it decides. Those rows are the bound-limited relations above — the seven-field
record and the seven-component tuple against their corners, each carrying the
width at which the budget stops it — and the two sets whose members are one
number in two kinds; the ledger's `_LEDGERED` list owns them.

```python
from typing import Annotated, Literal, NamedTuple

import annotated_types as at

from valgebra import (
    Regex,
    Validator,
    complement,
    intersection,
    recursive,
    union,
)


class Pair(NamedTuple):
    x: int
    y: int


# A length bound over a set or a dict is opaque to the relations: their
# representations do not count one. Whether a set under one has a value is
# counted from its element's values. Over a word or a sequence it is decided.
assert not Validator(Annotated[set[int], at.MinLen(3)]).is_empty()
assert Validator(Annotated[set[bool], at.MinLen(3)]).is_empty()
assert Validator(Annotated[tuple[int, int], at.MinLen(3)]).is_empty()
# A named tuple's positions are its fields, and the schema says so, so the
# relation is structural.
assert Validator(Pair).is_subtype_of(tuple[int, int])
# A recursive schema: the laws reach it, and one unfolding decides the kinds its
# body admits. A relation needing the body twice is the coinductive rule's: a
# tree of lists of lists is an integer tree, and `[1]` is an integer tree that
# is not one.
mu = lambda: Validator(recursive(lambda t: union(int, list[t])))  # noqa: E731
nu = recursive(lambda t: union(int, list[list[t]]))
assert intersection(mu(), complement(mu())).is_empty()
assert intersection(mu(), str).is_empty()
assert Validator(nu).is_subtype_of(mu())
assert mu().relation_to(nu) == "not_subset"

# Everything else here decides, on the sets or by a rule.
pattern = Validator(Annotated[str, Regex("a")])
assert pattern.is_subtype_of(Annotated[str, Regex("ab?")])  # L(a) <= L(ab?)
# L(a) is the word "a", and a `str` subclass's "a" is in the pattern and not the
# literal, which pins the exact type.
assert Validator(Literal["a"]).is_subtype_of(pattern)
assert pattern.relation_to(Literal["a"]) == "not_subset"
assert Validator(Annotated[int, at.MultipleOf(4)]).is_subtype_of(
    Annotated[int, at.MultipleOf(2)]
)
assert Validator(bool).is_subtype_of(Annotated[int, at.Ge(0)])
assert Validator(bool).is_subtype_of(Literal[True, False])
assert Validator(dict[Literal["a"], int]).is_subtype_of({"a?": int})

# A respelling denotes the same set, and the sets are what the relation reads --
# even though the laws construction settles do not reach this one. `A | (A & B)`
# is `A` by absorption, one case of `A | B` folding to `A` wherever `A` contains
# `B`; that fold needs a containment, and containment is the decision rather
# than a law.
record = Validator({"a": int})
respelled = union(record, intersection(record, Validator(str)))
assert respelled != record
assert record.is_subtype_of(respelled)
assert respelled.is_subtype_of(record)
```

Two instruments hold this list to the tree. `tests/test_completeness_ledger.py`
carries each relation above as a decided one, written the way a caller writes it
rather than built from the other operand — a distinction that matters, because
the shortcuts the procedure takes are keyed on two schemas sharing their
constants. An entry of the list above that stops deciding fails there as a
defect rather than a known gap; its strict expected-failure marks are for the
other direction, a relation that holds and is not decided, each naming the
limit that leaves it. `tests/test_completeness_probe.py`
searches a fixed universe for relations answered `False` that no value refutes
and fails when one appears without a written reason, so a gap nobody thought of
cannot arrive unnoticed. It reaches a gap only where some atom in its universe
reaches it, which is why that universe carries both constraint families, a
fixpoint beside its own unfolding, and a record beside a literal-keyed map.

A sequence below a union of sequence schemas it splits across, where no single
branch contains it, is decided by the set representation's symbolic automaton
(`crates/valgebra-core/src/descr/symbolic.rs`): `tuple[int, ...]` is below
`tuple[()] | tuple[int, *tuple[int, ...]]`. What has no syntax is the rest of
regular-expression-types inclusion — a repeated heterogeneous group, such as
`int` and `str` alternating to any length — because the sequence node carries
the linear prefix-and-tail form, so no schema valgebra builds takes that shape.

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
    a type query pays for. The constant speaks for its literal only where it is
    the literal's one value: `Literal[0.0]` also holds `-0.0`, and
    `Literal[Decimal("1.00")]` holds `Decimal("1.0")`, which a predicate can
    tell apart, so such a literal below a predicate is not proven.

    ```python
    import math
    from decimal import Decimal
    from typing import Annotated, Literal

    from valgebra import Validator

    positive = Annotated[float, lambda x: math.copysign(1.0, x) > 0]
    assert Validator(Literal[1.5]).relation_to(positive) == "subset"
    assert Validator(Literal[0.0]).relation_to(positive) == "undecided"  # -0.0
    two_places = Annotated[Decimal, lambda d: d.as_tuple().exponent == -2]
    assert Validator(Literal[Decimal("1.00")]).relation_to(two_places) == "undecided"
    ```
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
conservative `False` into a `True` — it can never change an answer that was
already correct.

Every decision also runs under a budget of a million decision steps, and
exhausting it returns the conservative answer (`False`, "not proven") rather
than running unbounded. The budget counts steps, not time: a step linear in a
leaf it reads is one step, so a relation over a large leaf takes time in
proportion to it ([limits](10-limits.md#what-the-bounds-do-not-bound)). This
preserves soundness: a bail-out is never a wrong `True`. A relation the two
schemas settle as written -- a schema against itself, a schema against a union
that holds it, a meet against one of its own members -- is answered whatever
the budget has left.

The boolean methods answer `True` or `False`, so through them a `False` from an
exhausted budget reads the same as a `False` the procedure decided. Inside the
core the two are distinct — emptiness answers *empty*, *inhabited*, or
*neither* — which is what lets a test say that a bail-out never claims a proof
it does not have, and `relation_to` reports the split: a query whose budget runs
out, and which the set representation does not settle either, answers
`"undecided"` rather than `"not_subset"`. The contract does not change with it:
a `False` is "not proven" either way.

The budget binds where the work is a **product** rather than a sum. Subtyping
distributes over both sides of a union, so relating two unions can cost the
product of their member counts, and a Boolean combination nested past a handful
of levels demands work exponential in its depth.

One shape avoids the product entirely, and it is the one a contract writes most:
a union of nothing but **literals**. A literal denotes its constant and the
values its type's `==` joins to it, so such a union denotes a *finite set of
constants*, each standing for its class, and inclusion between two finite sets
is membership of every constant of one in the other. That is decided by lookup rather
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
(`crates/valgebra-core/src/ir/transform.rs`, `mapped_member_set`). Without that
sort, two tables agreeing on nothing about where each constant sits fall out of
the canonical order and are distributed against each other instead.

A table met with other tables and with the complements of tables is the finite
set it leaves, decided by the same lookup, which is what a record's tag narrows
to when the record rule splits it:

```python
from typing import Literal

from valgebra import complement, intersection

left = intersection(Literal["a", "b", "c"], complement(Literal["a"]))
assert left.relation_to(Literal["b", "c"]) == "subset"
assert left.relation_to(Literal["b"]) == "not_subset"  # "c" is left
```

A constant is taken away by its pool position, which is exact, and kept only
where the bindings say it is a value apart from every table taken away, since
two positions may hold one value.

The refutation is the bindings' to give: two constants at two pool positions are
two *values* only where their type's equality can be trusted, and a constant
that does not equal itself — `float("nan")` — denotes no value at all, so
`Literal[float("nan")]` is the empty set and is below everything. Where the
equality cannot be trusted, a refutation stays undecided rather than guessing.
A proof does not ask: `Literal[c]` is below `Literal[d]` where `c == d`, which
holds only of an equality that is transitive, as
[the trust base](14-soundness.md#what-this-argument-assumes) says.

What is left under the budget is the Boolean tower: a deeply nested combination
of unions, meets and complements, where subtyping distributes over both sides
and the work is a product of the branches. A `False` there may mean "not proven
within the bound" rather than "not a subtype"; on anything else it means the
relation is outside the decided fragment above. The bound caps cost and does
not stand in for termination: every goal a query asks is drawn from a finite
closure of its subterms, which `docs/dev/02-decision.md` in the repository
writes out. Nor does it stand in for a goal memo on most shapes: over the
`--decision`, `--decision-refute` and `--decision-repeat` workloads of
`scripts/perf_gate.py` the goals a query *repeats* number zero, because the
trail absorbs recursion and the per-rule caches absorb the shape where one goal
is asked once per field. The relation matrix (`--decision-matrix`) is where a
table over goals would hit: a meet against a union repeats goals there, and
`decision::goal_tests` counts them.
