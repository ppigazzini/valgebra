# Changelog

All notable changes to valgebra are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- changelog-roll

Every feat/fix commit this section accounts for, oldest first; held to
`git log` by tests/test_changelog_ledger.py. `-- internal` marks a commit
a caller cannot see: a step of the set representation that changed no
answer of its own, or a repair to a change not yet released.

- fix: let a validation failure cross a process boundary
- fix: name a union's branches by what they accept
- feat: decide a literal against another kind
- feat: decide a fixed-length sequence that splits across union branches
- feat: decide an empty meet of two record schemas
- feat: cancel a double complement where the schema is built
- fix: bound the levels the membership walk holds open
- fix: report a container that changes under the walk
- fix: resolve a recursive marker wherever the build puts it
- fix: divide by the modulo operator, not the value's dunder
- fix: narrow a product component without writing at an index
- feat: read an unpacked variadic tuple as the shape it names
- fix: refuse a forward reference where a type argument belongs
- fix: check a class for the attributes it declares
- fix: compare a pooled constant by its type as well as its value
- fix: refuse a refinement marker that would otherwise be dropped
- fix: report a set's failures in an order the value fixes
- feat: ask both rules where both of them apply
- feat: the descriptor, and the three operations it is closed under -- internal
- feat: integer intervals, as the layer an integer set is built from -- internal
- feat: integers as an interval set per residue class -- internal
- feat: floats as intervals over the ordered line, and a bit for nan -- internal
- feat: strings and bytes as regular languages -- internal
- feat: sequences as an automaton guarded by value sets -- internal
- feat: lists and tuples as automata over descriptors -- internal
- feat: sets as a powerset over a descriptor -- internal
- feat: objects as open records over their attributes -- internal
- feat: classes as an order the core carries rather than asks for -- internal
- feat: one emptiness verdict, with a third answer for the open world -- internal
- feat: lower the schema fragment the descriptor can hold -- internal
- feat: decide by emptiness of the difference where the descriptor can -- internal
- feat: refuse the complement law over an atom that is not a set
- fix: withdraw the descriptor from the public relations -- internal
- feat: a sequence's length is what it holds
- feat: a meet cancelling to nothing is nothing
- feat: Any is the top, spelled
- feat: a class and an attribute record are a line of the kind -- internal
- feat: dicts as map atoms with a default per key kind -- internal
- feat: a field and a literal key are one label -- internal
- feat: refuse a map key schema narrowed by a constraint
- feat: read a TypedDict as the typing spec defines it
- fix: open and close read the labels on the semantic dom
- feat: the descriptor reads a class through the object pool -- internal
- feat: a constant is a value, not an object
- feat: a descriptor build spends a work budget -- internal
- feat: say what the compiled surface is
- feat: publish the construction limits where a caller can read them
- fix: a negated wanted key leaves the keys it excludes alone
- fix: render a schema as an expression that rebuilds it
- feat: a bare container class is its kind
- feat: a schema is built in the lattice normal form
- feat: retire the term rewrites the algebra does not need
- feat: a type alias that names itself is the fixpoint it writes
- feat: a schema compares equal in every order it can be written
- fix: a recursive schema is one schema wherever it is combined
- feat: decide a recursive schema against the kinds its body admits
- feat: a sequence's length is a property its automaton can state
- feat: an enumeration is the union of the members it can be
- feat: refuse to pickle a validator with a message that says what to send
- fix: an integer key stays an integer in an error path
- fix: the changelog ledger survives the window a release passes through -- internal
- fix: an unanchored ignore rule hid the frontend's test module -- internal
- fix: report a list that resizes under the walk, as a dict already is
- fix: a test that names a script as a subject is not a lane that runs it -- internal
- fix: an enumeration is the union of its members only when it is one
- fix: refuse a pattern before its automaton is built, not after
- fix: a validator takes part in the cycle collector
- fix: an integer key of any size stays an integer in an error path
- fix: a literal's hash is the constant's, not its slot's
- fix: one schema prints one way
- fix: a required-ness qualifier survives a string annotation
- fix: a bare legacy typing alias is the class it aliases
- fix: refuse a Literal argument the typing spec refuses
- fix: refuse a bound against nan, which orders nothing
- fix: the merge base is never the commit being measured -- internal
- fix: take the version from the crate, not from the metadata reader
- fix: decide two sets of literals as sets, not pair by pair
- fix: a ledger plant that judges nothing is not a miss -- internal
- feat: a bound over floats is a set the descriptor can hold
- fix: build the error model when it is asked for
- fix: a timing harness names the build it measured, and refuses a debug one -- internal
- fix: an instruction count says how much history it is comparing -- internal
- feat: subtyping answers in three values, and the boundary keeps two -- internal
- fix: a subject disjoint from a meet is below that meet's complement
- fix: a code fence closed with a sentence took a section into the block -- internal
- fix: a refinement carries its base's proof and not its refutation -- internal
- fix: a mismatch refutes an inclusion only where the subject has a value -- internal
- fix: a closed record is refuted by a key it does not declare -- internal
- feat: a named tuple denotes the tuple its fields lay out

-->

### Added

- A `NamedTuple` denotes the tuple its fields lay out, so a relation between one
  and that tuple is decided rather than declined:

  ```python
  from typing import NamedTuple

  from valgebra import Validator


  class Pair(NamedTuple):
      x: int
      y: int


  assert Validator(Pair).is_subtype_of(tuple[int, int])
  ```

  A named tuple's positions *are* its attributes -- the class lays both down at
  once -- and the schema says so instead of describing the attributes alone,
  which denoted a wider set than the class has. A failing field reports its
  **position** rather than its name, because the shape carries positions; a
  passing instance is read once rather than twice, and checks about a third
  faster.

### Fixed

- A schema disjoint from a meet is below that meet's complement. `A <= ~B` asks
  whether `A` and `B` share a value, and the meet it built for that question
  held `B` as a nested intersection where the rule that decides a meet empty
  compares the members of *one* intersection pairwise. Built through the meet
  constructor the members flatten, the pair meets, and the relation decides:

  ```python
  from valgebra import Validator, complement, intersection

  small = Validator(int)
  meet = intersection(Validator(str), Validator(bytes))
  assert small.is_subtype_of(complement(meet))
  ```

  The two deciders answered one question differently, which is what a
  disagreement between them looks like from outside: a relation that holds,
  reported as not proven.


### Added

- Relations over a bound on a **float** are decided. A bound was lowered into
  the descriptor only over whole numbers, so anything needing the descriptor and
  mentioning `Annotated[float, Gt(0)]` stayed undecided: `float > 0` was not
  known to be below `float`, nor disjoint from `int`, nor from `float < 0`. A
  float bound is now a set of floats, with the side chosen by the *base* rather
  than by the operand's type — `Gt(0)` carries the integer zero and orders the
  floats all the same — and `nan` sits outside every interval, as Python's own
  comparisons put it. Over the 6,000-pair random sweep the undecided
  corpus-true subtypes fall from 18 to 7 and the undecided emptinesses from 5
  to 1, with no relation refuted by a value.


- **Schemas compare as sets.** Each kind of value carries a representation
  closed under union, intersection and complement — integers as interval sets per
  residue class, floats as intervals over the ordered line with a bit for `nan`,
  strings and bytes as regular languages, sequences as automata over value sets,
  sets as powerset lines, dicts as map atoms with a default per key kind, classes
  as an order, objects as records over their attributes — so `a <= b` is asked
  as `a & ~b` admitting no value, which is what the relation means. Where the
  structural rules decline, that question is asked and answered:

  ```python
  from typing import Annotated, Literal

  import annotated_types as at

  from valgebra import Regex, Validator, complement, intersection

  assert Validator(Annotated[str, Regex("a")]).is_subtype_of(Annotated[str, Regex("ab?")])
  assert Validator(Annotated[int, at.MultipleOf(4)]).is_subtype_of(
      Annotated[int, at.MultipleOf(2)]
  )
  assert Validator(bool).is_subtype_of(Literal[True, False])
  assert Validator(bool).is_subtype_of(Annotated[int, at.Ge(0)])
  assert Validator({"a": int}).is_subtype_of(dict[Literal["a"], int])
  assert intersection(list[int], list[str]).is_subtype_of([])
  assert Validator(tuple[int]).is_subtype_of(complement(tuple[str]))
  ```

  Container meets, double complements, one regular language inside another, a
  kind against its own literals, one step dividing another, and the emptiness of
  a dict schema are decided this way. Twenty-five relations that answered `False`
  answer `True`, and the same shape written any other way is decided alike: a
  respelling such as `union(A, complement(union(A, nothing)))` is the universe.

  Building a set representation costs about two orders of magnitude more than a
  rule that already answers, so it is asked only where the rules decline, and
  only for a schema it can build within a bound on the nodes it reads, the
  nesting it descends and the work it spends. Past any of those the answer is the
  conservative one, as before. What stays conservative is recursion, a length
  bound over a shape that is not text, an attribute record beside a builtin kind,
  and a predicate.

- **An integer key stays an integer in an error path.** The `path` a failure
  reports is what a caller walks back down to the offending value, and every
  mapping key arrived as text -- so a dict keyed by numbers reported a location
  that indexed nothing, and `d[2]` and `d["2"]` were indistinguishable. A key
  that is a string or an integer is now itself; anything else has no spelling in
  a path made of those two and appears as its repr, as before.

  ```python
  from valgebra import ValidationError, Validator

  try:
      Validator(dict[int, int]).validate({1: 1, 2: "x"})
  except ValidationError as error:
      assert error.errors[0]["path"] == (2,)
  ```

- **A validator cannot be pickled, and says what to send instead.** It holds the
  classes and callables its schema names, so the schema is what travels;
  rebuilding is about as cheap as unpickling would be. `ValidationError` pickles
  as before.

- **A length bound on a list or a tuple is decided.** A length is a regular
  property of a sequence -- "any element, that many times" -- so the
  representation that holds sequences holds a bound on them, and a bound that
  used to be opaque to everything but a string is now part of the algebra.

  ```python
  from typing import Annotated

  import annotated_types as at

  from valgebra import Validator

  assert Validator(Annotated[tuple[int, int], at.MinLen(3)]).is_empty()
  assert Validator(Annotated[list[int], at.MinLen(3), at.MaxLen(2)]).is_empty()
  assert Validator(Annotated[list[int], at.MaxLen(0)]).is_equivalent([])
  assert Validator(Annotated[list[int], at.MinLen(2)]).is_equivalent([int, int, int, ...])
  ```

  A set and a dict have a length their representations do not count, so a bound
  over one of those still stands.

- **A recursive schema is decided against the kinds its body admits.** A
  reference is a cycle and a set representation has no room for one, so every
  relation over a fixpoint used to fall to the structural rules -- which read
  shapes, and cannot tell that a JSON value and a `bytes` share no value. The
  body is now unfolded once before the sets are built, with a bound standing
  where the reference was: the top where the schema is used positively, the
  bottom under a complement, which is what keeps a difference sound.

  ```python
  from valgebra import Validator, anything, complement, intersection, recursive, union

  json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
  assert intersection(bytes, json).is_empty()
  assert intersection(tuple, json).is_empty()
  assert Validator(bytes).is_subtype_of(complement(json))
  assert json.is_subtype_of(anything)
  ```

- **A recursive schema is one schema wherever it is combined.** Merging a
  compiled validator used to copy its definitions, so two occurrences of one
  fixpoint became two definitions and every law that compares terms failed on
  it: `intersection(json, complement(json))` was not empty and
  `union(json, complement(json))` was not the top, for the same schema object on
  both sides. A merge now reuses definitions it already holds, and a fold that
  leaves one unreachable drops it -- so the top built that way is the top built
  any other way.

  ```python
  from valgebra import (
      Validator,
      anything,
      complement,
      intersection,
      nothing,
      recursive,
      union,
  )

  json = recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
  assert intersection(json, complement(json)) == Validator(nothing)
  assert union(json, complement(json)) == Validator(anything)
  ```

- **Two spellings of one schema are one schema.** A record's fields, a map's
  clauses, a refinement's markers and a union's members are sets, so the order
  they were written in is no longer part of the term: `==` says so, `hash`
  agrees, a union of two spellings folds to one member, and a validator is a
  usable dictionary key whatever order its schema was written in. A repeated
  marker is dropped, since a constraint written twice narrows once.

  ```python
  from typing import Annotated, Literal

  import annotated_types as at

  from valgebra import Validator, union

  assert Validator({"a": int, "b": str}) == Validator({"b": str, "a": int})
  assert Validator(Literal[1, 2]) == Validator(Literal[2, 1])
  assert Validator(Annotated[int, at.Ge(0), at.Le(9)]) == Validator(
      Annotated[int, at.Le(9), at.Ge(0)]
  )
  assert union({"a": int}, {"a": int}) == Validator({"a": int})
  ```

  **`repr` prints the canonical order rather than the written one**, which is
  the visible half: `{"name": str, "age?": int}` renders as `{'age?': int,
  'name': str}`. It still rebuilds the schema. A test pinning the written order
  of a record's fields, a map's clauses or a refinement's markers needs
  updating; one pinning a union's or a literal's does not, since those keep the
  order they were built in.

- **A PEP 695 `type` alias that names itself builds the fixpoint it writes.**
  The alias is the binder: it is reached again while its own body is read, and
  there is no lambda to carry the fixpoint, so the alias carries it. What it
  builds is what the explicit call builds, and mutual recursion is two aliases
  naming each other. An alias naming itself outside a structural constructor
  denotes no set and is refused when the validator is built, as an unguarded
  `recursive` already was. Before, such an alias was refused as a schema too
  deep to compile.

  ```python
  from valgebra import Validator, recursive, union

  type Json = None | bool | int | float | str | list[Json] | dict[str, Json]

  assert Validator(Json).is_equivalent(
      recursive(lambda j: union(None, bool, int, float, str, [j], {str: j}))
  )
  ```

- The three construction bounds are importable: `MAX_SCHEMA_DEPTH`,
  `MAX_DEFINITIONS` and `MAX_SCHEMA_NODES` are exported from `valgebra` and
  listed in `__all__`, so code sizing a schema against a bound reads the number
  rather than repeating it.

- A class with declared attributes is the meet of its `isinstance` atom and a
  record of its attributes, and each half is a set the algebra relates on its
  own: an object schema is below its own class, an attribute record relates to
  another by width and depth whatever class it came from, and a record whose
  required attribute admits nothing is decided empty *and* one whose attributes
  are inhabited is decided inhabited — which the class half, being opaque, used
  to take away. The surface is unchanged: `repr(Validator(Point))` is `Point`, a
  union names the class in its branch list, and a value of the wrong class
  reports one `instance_type` violation.

- An `intersection` stops collecting violations once a member rejects the value
  itself rather than something inside it, the rule `Annotated[...]` already
  applied between a base and its constraints. A member that fails inside the
  value leaves the others meaningful and they are still collected.

- Widening a literal union is decided by containment rather than by the product
  of the two member counts, so a table whose members are the same constants as
  the wider one's relates at any size. Constants pool by value, so two tables
  written independently share theirs and relate the same way.

- `~~A` is `A`, a union carrying a schema together with its complement is the
  top, and a meet of that pair is the bottom, all settled where the schema is
  built. The decision procedure has no rule for these shapes and does not meet
  one built through the constructors; a shape built another way — a recursive
  definition, a respelling, constants equal but not identical — reaches the
  procedure and is not decided. `repr`, `==`, and the code a violation reports
  follow the cancelled form: `complement(complement(int))` reports `int_type`
  where it reported `unexpected_match`, and `intersection(int, complement(int))`
  reports `no_match` naming `nothing`. A predicate and a class with an
  `isinstance` hook are exempt from the two cancelling folds: the law is about
  sets, and an atom that answers by running code is not one.

- A meet of two record schemas is decided empty when a key one side requires
  cannot hold — because the types the two give it share no value, or because the
  other side is closed and does not declare it. `{"a": int} & {"a": str}` is
  empty, and so `{"a": int}` is below `~{"a": str}`. Only a required key empties
  a meet: two mappings, or two optional fields, always admit the empty dict.

- A fixed-length sequence is decided against a union of fixed-length sequences
  it splits across, where no single branch contains it: `tuple[int | str, int]`
  is below `tuple[int, int] | tuple[str, int]`. The rule needs a fixed component
  count, so a homogeneous or variadic sequence is not decomposed, and branches of
  another container or arity drop out rather than blocking it.

- A literal carries the kind of its constant, so it is decided against another
  kind: `Literal["a"]` is below `~int`, and `Literal["a"] & Literal["b"]` is
  empty. `Literal[1]` and `Literal[True]` are disjoint although `1 == True`,
  because a literal pins `type(x)` exactly. The rule applies to the builtin
  scalars, whose equality is Python's own; a meet of two `Enum` members stays
  conservative, since user-defined equality can admit one value for two
  constants.

- A recursive schema is decided below its own body written out, and a refinement
  of a union below that union: `recursive(lambda t: union(None, {"next": t}))` is
  a subtype of `union(None, {"next": <that schema>})`, and
  `Annotated[int | str, Ge(0)]` of `int | str`. Trying a union's branches one by
  one commits to a branch, and a subject that lands in the union only once a
  reference is unfolded or a refinement drops to its base got no answer from it.
  Both rules are sound alone, so where both apply both are asked.

- Every `ValidationError` carries the six attributes the error model documents,
  however it was made. The model describes failures and an error built by hand
  reports none, so it reads as empty — empty strings and empty tuples — rather
  than raising `AttributeError` for an attribute the type declares.

### Changed

- **Explaining a failure over a record costs a third less.** The accepting walk
  scans a dict once and resolves each key it finds through a map built with the
  validator; the explaining walk asked the dict for each declared key by name,
  and a Rust string handed to a dict lookup is decoded into a fresh Python
  string and hashed before the probe can start -- once per field, per call. The
  declared keys are interned with the validator, as the attribute names already
  were, so the lookup is the probe alone. A fifty-field record reporting one bad
  field moves from **2.38 to 1.53 of pydantic-core's time**, and the shape's
  ceiling comes down from 3.5 to 2.5. No answer changes.

- **The membership walk costs a third less.** Every scalar arm of the walk ended
  in a helper that passed a boolean through and recorded a violation when it was
  false, and every element of a sequence went through a second helper between
  the loop and the walk. Both were out-of-line calls doing almost nothing: on a
  list of integers they were fifty of the hundred and fifty instructions spent
  per element. The two are inlined and the violation-recording half is marked
  cold, so the accepting path is a test where the answer already is. Measured on
  the fixed binding workload, **661.6M instructions against 968.9M**, and against
  pydantic-core a 1,000-element array moves 0.88 to 0.57 of its time, a
  fifty-field record 0.58 to 0.49, and a JSON document 0.86 to 0.81. No answer
  changes.

- **Building and composing a schema costs less.** A field name was owned
  outright by the field that declared it, so every pass that rebuilds a schema
  — opening or closing its records, reindexing it onto another validator's
  pools, simplifying it — copied the name of every field it carried across. On
  the core's fixed workload those copies were 38% of every heap allocation the
  crate made. A name is now shared: carrying it is a refcount bump and no
  allocation. The workload runs in 161.0M instructions against 233.5M, and
  compiling a fifty-field record costs 13.7 µs against 14.2. No answer changes.

- **A schema is built in the lattice normal form.** The constructors folded two
  laws and left the rest standing, so `union(int, int)` rendered `int | int` and
  compared unequal to `int` while `union(int, complement(int))` rendered
  `anything` — `==` was equality of nothing in particular, and `repr` showed a
  shape no rule was written for. Members of a join or a meet are flattened,
  ordered and deduplicated; the identities and the absorbing elements apply;
  `~anything` is `nothing` and `~nothing` is `anything`; and a join of one member
  is that member. So `union(str, int) == union(int, str)`,
  `intersection(int, anything) == Validator(int)`, and `==` is equality of a
  canonical form.

  Three things a caller may see. `repr` shows the normal form, so a join renders
  its members in that order — a kind before a literal, for instance, while
  literals and classes keep the order they were written in, their pool slots
  being what orders them. A union's `expected` message lists its branches the
  same way. And `Literal["x"]`, a join of one member, is that literal: it reports
  `literal_error` where it reported `union_error`.

  Absorption is the one law left standing: `A | (A & B)` is `A` only when `A`
  contains `A & B`, and containment is the decision procedure — running it
  wherever a schema is built is the cost this design refuses everywhere else.
  `is_equivalent` decides it.

- **A bare container class is its kind.** `list` and `list[object]` admit the
  same values — every list, a subclass instance included — and were different
  sorts of thing: a sequence node in one case, an `isinstance` atom in no kind at
  all in the other, so neither spelling was decided below the other and
  `repr(Validator(list))` said `list` where the schema said otherwise. `list`,
  `tuple`, `set`, `frozenset` and `dict` name their kind's whole set, which is
  what the typing spec assigns an unparameterised generic and what the membership
  check always performed; the two spellings are one schema and compare equal.
  `str`, `bytes`, `int` and `float` always read this way. `repr` follows the
  schema, so `Validator(list)` renders `list[anything]`.

  A class built on a builtin narrows that kind rather than standing beside it, so
  it relates to it: `Validator(MyInt)` is below `Validator(int)` and
  `Validator(MyStr)` below `Validator(str)`, and a meet of the two is empty. A
  class built on no builtin narrows nothing and relates to a kind in neither
  direction — `class Both(Plain, MyStr)` builds and its instances are strings, so
  a plain class's instances are not confined to any kind.

- **Every argument the compiled surface takes is positional.** `Validator`, the
  combinators, and every method on a validator declare their parameters
  positional-only, matching the stub that already wrote them that way: a call
  naming one — `v.is_valid(obj=x)`, `Validator(schema=int)`,
  `complement(schema=int)` — raises `TypeError`. `fail_fast` is the one keyword
  the surface takes and it is keyword-only, as before. A stub that permits a
  keyword the extension refuses type-checks code that cannot run, and the two
  disagreed in nine places.

- **A `TypedDict` is open**, which is the set the typing spec assigns it: a dict
  carrying keys the class does not name is admitted, and the keys it does name
  are checked and required exactly as before. `closed=True` and `extra_items=T`
  (PEP 728) are obeyed where the runtime provides them. The dict-literal form
  `{"name": str}` stays closed — it is this library's own spelling, and a schema
  written as a shape means that shape. Code relying on `Validator(TD)` to reject
  an extra key should write the `TypedDict` `closed=True`, or use the dict
  literal.

- **A map key schema narrowed by a constraint is refused where it is written.**
  A clause's key says which keys it governs, and that must be a type — `str`,
  `int`, a union of them — or a `Literal`, which names the keys one by one.
  `{Annotated[str, MinLen(2)]: int}` and `dict[Annotated[int, Ge(0)], str]` now
  raise at construction. A narrowed key names *part* of a type, and two such
  clauses can overlap without either containing the other, which is a question
  this map model does not answer the same way twice. To constrain the keys
  themselves, check them beside the mapping rather than inside it. Keys that name
  a whole type are unaffected, `dict[tuple[int, int], V]` included.

- **`typing.Any` is the lattice top.** It denotes what it always admitted —
  every value — and it is now the same schema as `anything`, so every law and
  every relation reaches it: `Validator(Any) == Validator(anything)`,
  `complement(Any).simplify()` is `nothing`, `intersection(Any,
  complement(Any))` is decided empty, and `int` is decided below `Any`. A
  gradual type is held apart from the top for a second question, consistency at
  the boundary between typed and untyped code, which a validator has no site for
  and never asked. What is kept is the spelling: `repr(Validator(Any))` is still
  `Any`, and `repr(Validator(anything))` is still `anything`. The spelling is
  not part of the set, so two schemas differing only in it are equal and nothing
  decides anything by it.

  Code that read `intersection(Any, complement(Any)).is_empty()` as `False`, or
  `Validator(Any).is_equivalent(anything)` as `False`, now sees the opposite.
  Membership is unchanged: `Any` admitted every value before and admits every
  value now.

- A constant is a value rather than an object. Two equal builtin scalars built
  separately — two `"code_00042"` strings read from different files — pool into
  one constant, where pooling by object identity made them two and left a schema
  mentioning both with two nodes no rule could see as one. The rule is a
  literal's own: same exact type, and equal. `Literal[1]` and `Literal[True]`
  stay two constants because the types differ; `Literal[0.0]` and
  `Literal[-0.0]` are one, because the values are equal and the sign of zero is
  not part of either; a `nan` and an integer too wide for the key pool by
  identity, since `nan` equals nothing at all and a wide integer has no key. Two
  independently written thousand-member tables relate at any size as a result.

- `repr` renders an expression that rebuilds the schema. Three forms rendered
  something that either was not Python or was Python that builds a *different*
  schema: a recursive schema showed its back edge as `...`, an open record showed
  its catch-all as `...`, and both read back as `Literal[Ellipsis]`, an ordinary
  dict key; the nullary product printed `tuple[]`, which is not an expression.
  They render as `recursive(lambda X: {'v': int, 'n?': X})`,
  `{'name': str, anything: anything}` and `tuple[()]`. Two forms remain a
  rendering rather than a round trip, because neither is syntax: a class prints
  its name and a predicate prints `Predicate(...)`.

- A validator names the package it is imported from. `Validator.__module__` is
  `valgebra` rather than the private extension underneath, which the API
  reference reserves the right to rename and tells callers not to import; the
  class is `final`, which it was already at runtime, and the annotation says so
  to a type checker.

- A list or tuple counts one level of the construction depth bound, not two, so a
  schema can nest twice as deep before the bound refuses it: a chain of 128
  nested lists now builds where 64 was the limit, and a chain that pins a length
  on each list reaches 64 where 43 was the limit. The sequence node carries its
  elements directly rather than as a regular expression over them, and the two
  levels a list spent were the expression's own constructors — levels a walk
  descended and a reader had no way to see. The bound itself is unchanged at 128
  levels, and no schema that built before is refused now.

- A schema refused for depth says so by name. The frontend's own descent limit
  stood at the same 128 levels as the construction bound, and once a list cost
  one level rather than two the two limits met: a too-deep chain of lists
  reported `NotImplementedError` about a type that never reaches a leaf instead
  of `ValueError` about the depth. The frontend now descends one level further
  than the deepest schema construction accepts, so the bound that tripped is the
  bound that speaks.

- A union's `expected` names each branch the way that branch names itself when it
  fails alone, in place of the branch's node kind. A set of permitted strings —
  the commonest shape a field has — reported `one of: literal, literal` and now
  reports `one of: the literal 'torch', the literal 'jax'`; an `Enum` branch
  names its class. `Literal[...]` builds a union of its constants, so its
  constants are what the message lists. The list is bounded at the same number of
  branches the closest-branch search reads and ends in `...` beyond it, so a wide
  union reports a readable prefix.

- The completeness probe searches refinements. Its schema universe crossed every
  other kind the decision procedure treats differently and held no refinement, so
  that whole fragment was outside the reach of the gate
  `docs/15-decidability.md` cites as the reason an unlisted conservative answer
  cannot go unnoticed. Four atoms — two order bounds and two regexes — put it in
  reach, and the four suspected gaps they surface are on the ledger with the
  route to deciding each. No answer changed; nothing was found unsound.

### Deprecated

- **`Validator.simplify` is deprecated and is removed in the next minor
  version.** Calling it raises a `DeprecationWarning`. A schema is built in the
  lattice normal form, so the reduction it promises is the schema a caller
  already holds: `repr` shows it and `==` compares it. What it does beyond that
  is not a law but a decision — a meet of two provably disjoint kinds is the
  bottom, a join covering every region is the top — and `is_empty`,
  `is_subtype_of` and `is_equivalent` decide those and more without rewriting a
  term. Write `intersection(int, str).is_empty()` rather than
  `repr(intersection(int, str).simplify()) == "nothing"`.

  `open`, `close` and `ensure` stay. The first two rewrite every record a schema
  declares at any depth, inside its recursive definitions — a traversal that is
  not spellable one set at a time, which is what a whole-schema operation is.

### Fixed

- Two wide literal unions are decided as sets. The core compares a union with a
  union member by member, so `intersection(Literal[*range(20_000)],
  Literal[*range(20_000, 40_000)]).is_empty()` was 400 million calls into the
  bindings and **six seconds**; it is now 10 ms. The bindings answer the whole
  disjointness question in one pass where they can hash the constants, and
  decline — leaving the member walk — where they cannot.

- A raised `ValidationError` builds its structured model when something asks for
  it. Every failure became a dict, a path tuple and six attribute writes at raise
  time, so a report over 10,000 failing rows cost **26 ms** whether or not the
  caller read a row; it is now **9 ms** for a caller that logs `str(error)`, and
  still faster than before for one that reads `errors`. The attributes, their
  values and `str()` are unchanged, and pickling builds the model first so what
  crosses a process boundary is the same plain data.

- `import valgebra` costs **0.9 ms**, down from 32. `__version__` was read with
  `importlib.metadata.version()`, which pulls `email`, `zipfile`, `inspect` and
  the compression modules to read a file that says what `Cargo.toml` already
  said — 20 of those 32 milliseconds, and a dozen modules dragged into any
  process that imports valgebra. The extension carries the crate's version
  instead; `tests/test_version.py` holds it to the installed distribution's.

- A bound against `nan` is refused. Every comparison with `nan` is false, so
  `Annotated[float, Ge(nan)]` admitted no value at all — the empty set written as
  a bound, which no caller means and which neither decider proves empty.
  `MultipleOf(nan)` goes the same way. A bound that is empty because the *order*
  says so, such as `Gt(inf)`, is kept: emptiness is then an answer.

- A container is refused as a `Literal` argument, where it used to be read as a
  schema. The typing spec's `Literal` takes `None`, an enum member, or an `int`,
  `bool`, `str` or `bytes` value; Python does not reject the subscription, so
  `Literal[[1]]` reached the constant fallthrough and came out as
  `list[Literal[1]]`, and `Literal[{}]` as the empty record — sets the caller did
  not ask for, with no message saying so. A float is still accepted: it is not a
  spelling the spec allows either, but it is a *constant*, which this library
  pools like any other. A bare `ForwardRef` is refused too, with the message a
  forward reference in a type argument already gave.

- A bare legacy typing alias is the class it aliases. `typing.Tuple` was read as
  `tuple[()]` — the empty tuple, admitting `()` and nothing else — because a bare
  alias and a parametrization both carry no type arguments; `typing.List`,
  `typing.Dict` and `typing.Set` were refused for wanting one. Each is now its
  origin, so `Validator(typing.Tuple) == Validator(tuple)`. `tuple[()]` keeps
  meaning the empty tuple, and a parametrized alias is unaffected.

- `NotRequired` and `Required` are read under `from __future__ import
  annotations`. CPython computes a `TypedDict`'s `__required_keys__` when the
  class is created, from the annotations as written — under PEP 563 those are
  strings, so the qualifier was invisible to it and every optional key in every
  module using the future import compiled as **required**, failing correct data
  with `missing_key`. The resolved hint carries the qualifier and is now what
  answers, so a class means the same thing with the future import as without it.

- Two spellings of one schema print the same way. A union's members are ordered
  by the IR's own order and a literal sorts there by its **pool slot** -- the
  order the constants were first seen -- so `Literal[1, 2]` and `Literal[2, 1]`
  were one schema by `==` and by `hash` and two by `repr`, against
  `docs/04-algebra.md`'s "`repr` shows it and `==` compares it". The literals in
  a union are now ordered by what they print; members that are not literals keep
  the place the normal form gives them.

- Two validators that differ only in a constant no longer share a hash.
  `__hash__` skipped every pool slot, so `Literal[1]` through `Literal[1000]`
  were one hash and a dictionary keyed by validators -- the reason the method
  exists -- degenerated into a linear scan, at 98 microseconds per lookup over
  ten thousand entries. The constant behind a slot is now folded in, which is
  10,000 distinct hashes and 0.065 microseconds for the same registry. A
  constant with no hash contributes nothing, so a validator stays usable as a
  key whatever it pools.

- An integer key of any size reaches an error path as an integer, and so does a
  `bool`. `docs/08-error-model.md` promises a path a caller can walk back down to
  the value, and a key outside a machine word's range was rendered as its `repr`
  — `'1180591620717411303424'` — while `True` was excluded outright and arrived
  as `'True'`. Both index nothing. `d[True]` and `d[1]` are one entry in Python,
  so a `bool` arrives as the integer it is.

- Explaining a failure over a large value no longer costs the size of the value.
  Every violation summarised the value it was about by building that value's
  whole `repr` and keeping eighty characters of it, so a 20,000-deep list was
  rendered in full once per level of the walk: twelve seconds for a single
  error, against twenty microseconds for `is_valid` on the same value. A
  container is now rendered under a bound instead, which is 1.1 ms for the same
  case and flat in the depth. A value small enough to print is unchanged.

- A class that holds its own validator is collected. A validator keeps the
  classes, enum members and callables its schema names, so `Model.validator =
  Validator(Model)` is a reference cycle -- and the type was not tracked by the
  cycle collector, so the collector never saw the edge from the validator back
  to the class and every such class leaked. A validator now traverses the
  objects it owns, and can be weakly referenced, so a registry keyed by schema
  can be a `WeakValueDictionary` and let its entries go.

- A relation over a pattern whose determinisation is exponential refuses instead
  of exhausting memory. `Annotated[str, Regex("(a|b)*a(a|b){20}")]` against
  `Regex("(a|b)*")` spent six seconds and 668 MB, and two more repetitions
  aborted the process on a four-gigabyte allocation: the automaton bound was
  checked after the regex engine had built the whole dense table. The engine now
  carries the size limit, so the family answers in under 100 ms at any
  repetition count. Membership is unaffected — the walk runs the pattern, not
  the automaton — and a pattern that stays small is still decided.

- A `Flag`, an `IntFlag` and an `Enum` with no members are no longer read as the
  union of the members they list. A flag's `|` builds instances the class never
  listed, so `Validator(Permission)` was decided a subtype of
  `Literal[Permission.READ, Permission.WRITE]` although
  `Permission.READ | Permission.WRITE` is in the class and not in the literal; an
  enumeration with no members can still be subclassed, so `Validator(Base)` was
  decided a subtype of `nothing` although a subclass's member is an instance of
  it. Each is now the `isinstance` atom it was before the union reading existed,
  which leaves membership unchanged and the relations undecided.

- A **list** that changes size while it is being checked is reported as
  `mutated_during_validation`, as a dict, a set and a record already were. A
  sequence is walked by position against a length read once, so a list grown by
  a predicate — or, on a free-threaded interpreter, by another thread — hid its
  new items from the walk, and one that shrank left the walk answering about
  items that were gone. In both directions `is_valid` returned `True` for a
  value that is not a member, and `ensure` handed that value back as checked. A
  tuple cannot be resized and keeps the plain iterator.

- A dict schema is not decided below the complement of a record it shares values
  with. Negating "no key of this part, other than the ones this atom names, maps
  anywhere" tightened that part's default, and a default governs every key the
  atom does not name — the excluded ones included — so a record complemented
  twice came back forbidding the key it is about, and `dict[str, int]` was
  decided a subtype of `~{"a": int}` although `{"a": 1}` is in both.

- A numeric bound orders the booleans as well as the integers. `bool` is a kind
  of its own to the set representation and `int` denotes both, so a bound lowered
  as a set of integers alone denoted less than the schema does — and a smaller
  set has a larger complement, which is a subtype proof no value supports. A
  bound over a base that is not whole numbers, `Annotated[float, Gt(0), Lt(1)]`
  among them, is left to the structural rules rather than narrowed to integers.

- Two plain classes are not decided disjoint. A class built on no builtin lays
  down no instance layout of its own, and reading two such classes as laying down
  *different* layouts made them share no value — though `class Both(A, B)` builds
  and its instances are in both. A layout conflict, which Python refuses to build
  a class across, still decides the pair.

- **`open` and `close` are functions on sets.** `{"a?": nothing}` and `{}` admit
  exactly the empty dict — the field allows the key to be absent and admits no
  value for it, which is what a closed record already says of every key it does
  not name — yet `.open()` gave them different sets. The redundant field is now
  read away first, so equal records open to equal records. Two consequences a
  caller may see: `Validator({}).open()` now admits every dict, where it used to
  be left alone (having no field never made `{}` a mapping — a clause and no
  field does), and a record carrying a field its own clauses already cover loses
  that field when opened or closed.

- A set reports its failing elements in an order the value fixes rather than the
  one the interpreter hands them over in, which moves with the hash seed. A set
  has no positions, so an element failure carries no index and only what it
  reports distinguishes it; the report is ordered by that, and `fail_fast` keeps
  the first of that order. The error model promised this determinism and a set
  was where it did not hold.

- A dict key that is not a string appears in an error path as its full `repr`
  instead of a summary cut at forty characters, and a string key appears whole.
  A path is what a caller walks back down to the value, and a truncated key
  indexes nothing.

- A refinement marker that would be dropped is refused instead. Four shapes
  silently produced a schema that admits either everything the marker excludes or
  nothing at all: a compiled `re.Pattern`'s flags were discarded, so a
  case-insensitive pattern refused the strings it matches; a `bytes` pattern and
  a length bound no length can equal were skipped entirely, leaving the base
  unconstrained; and a marker from `annotated_types` that valgebra does not
  check, such as `Timezone` or `Unit`, was ignored as if it were someone else's
  metadata. `re.IGNORECASE`, `re.MULTILINE`, `re.DOTALL` and `re.VERBOSE` are
  written into the pattern; `re.ASCII`, `re.LOCALE` and `re.DEBUG` are refused by
  name. Metadata from outside that vocabulary is still ignored, as the typing
  spec asks.

- A constraint no value of the base can answer is refused at build.
  `Annotated[int, MinLen(1)]` asked an integer for its length, which raises, and
  a raise reads as a non-member — so the schema denoted nothing at all while
  looking like a narrowing. A constraint some value of the base *can* answer is
  unaffected: a union with a text branch, or a class that may define what the
  constraint asks for, still narrows.

- `==` on validators reads a pooled constant the way `Literal` does: same type
  and equal. Python's `==` runs across types, so comparing by equality alone made
  `Validator(Literal[1])` and `Validator(Literal[True])` the same validator while
  `is_equivalent` reported them disjoint — two answers about one pair. Comparing
  a validator with something else answers `NotImplemented` rather than `False`,
  so the other operand gets its turn as the data model asks; `==` still falls
  back to identity, so the answer a caller sees is unchanged.

- A class is checked for the attributes it declares rather than for every
  annotation on it. A dataclass carrying an `InitVar` denoted the empty set —
  the marker names a constructor parameter no instance keeps, so every instance
  was refused for a missing attribute — and one carrying a `ClassVar` failed to
  build at all. Both are read as what they are, a field declared `init=False` is
  still checked because the instance carries it, and a `TypedDict` field
  qualified `ReadOnly` compiles to the type it qualifies instead of being
  refused.

- A string in the argument of a typing form is refused instead of read as a
  literal. `list["Account"]` is a forward reference the typing spec resolves
  against the namespace the annotation was written in, and a runtime object
  carries no namespace; reading it as a constant built a list of the *word*,
  which refuses what the annotation admits. `Validator` still reads a bare
  constant as a literal wherever a value belongs — at the top level, in a native
  list or dict literal, and in `Literal[...]` — and still resolves a class's own
  string annotations.

- An unpacked variadic tuple compiles to the shape it names.
  `tuple[int, *tuple[str, ...]]` is a fixed prefix followed by a repeating tail —
  the same shape `tuple[int, str, ...]` spells — and it was read as a two-element
  tuple whose second element is a tuple, so it refused `(1, "a")` and admitted
  `(1, ("a",))`. `Unpack[...]` says the same thing and compiles the same way; an
  unpacked *fixed* tuple splices its elements in. An element after the repeating
  tail names a set the sequence node cannot carry and is refused, as is
  unpacking a `TypeVarTuple`, which binds no element types at runtime.

- The product-splitting rule builds its narrowed component through the schema
  constructors instead of writing at an index, so the double negation it used to
  manufacture — a complement of a branch component that is itself a complement —
  cancels where the rest of the tree says it does. The verdicts are unchanged;
  the rule stops producing a shape no other rule is written for.

- A `MultipleOf` divisor of a different type than the value divides. The check
  reads the `%` operator rather than the value's `__mod__` alone, and half of
  what `%` means lives on the divisor: a value that does not know it answers
  `NotImplemented`, the divisor's `__rmod__` is asked next, and `NotImplemented`
  is truthy — so it read as a non-zero remainder. `Annotated[int,
  MultipleOf(0.5)]` refused every integer, and a `Fraction` or `Decimal` divisor
  refused every value.

- A `recursive` definition nested inside another resolves its self-reference
  wherever the build put it. An inner fixpoint whose body names the *outer*
  variable compiles to a definition of its own, and the outer marker lands in
  that definition rather than in the outer body — so resolving the body alone
  left the marker dangling, and a dangling marker matches no value: the schema
  silently rejected members. Contractivity is checked over the whole system of
  definitions for the same reason, which refuses `X = ~X` and `X = X | list[Y]`
  written across a nesting; both built before and denoted no fixpoint.

- A placeholder kept past the `recursive` builder it was handed to is refused at
  construction. The placeholder is an ordinary validator, so nothing stops a
  caller storing it, and what it stands for stops existing when the builder
  returns; using one afterwards built a validator that admitted no value and said
  nothing about why.

- A container that changes size while it is being checked is reported as
  `mutated_during_validation` instead of aborting the interpreter. Membership
  runs Python at almost every entry of a dict or a set — a predicate, an
  `__eq__`, an `isinstance` hook — and a free-threaded interpreter lets another
  thread write to a shared value meanwhile; the iterators underneath both
  containers answer that with a panic, which crosses the boundary as a
  `BaseException` no caller catches as a validation failure. The walk reads both
  containers in a way that survives the change and reports a non-member, because
  a reading cut short decides nothing about the contents. Only a change in size
  costs the reading: a value rewritten in place is unaffected. The same code
  reports a value whose two readings disagree, which is the same failure of the
  check to have a stable value to decide about.

- The membership walk counts the levels it holds open and refuses past 512 of
  them with `recursion_limit`, so a value inside every published construction
  bound cannot exhaust the native stack. Counting recursive *unfoldings* alone
  does not bound the frames: an unfolding descends the whole definition body, so
  a body at the schema-depth bound turns the 128 permitted unfoldings into
  thousands of frames. A recursive schema over a deep body meets the level bound
  and reports it; a linked list at the unfolding bound is unaffected, because the
  level ceiling sits above what that shape asks for.

- `ValidationError` can be pickled, so a validation failure crosses a process
  boundary with its structured model intact. The exception's `__module__` was a
  bare `_valgebra`, which names no importable module, and `pickle` locates a
  class by that string together with the qualified name — so a worker in a
  process pool or a task queue delivered a `PicklingError` naming an internal
  module instead of the validation result. The module is now `valgebra`, the
  package the name is exported from: it is baked into every serialized error, so
  it has to be the path that keeps resolving rather than the private extension
  underneath, whose name this reference reserves the right to change. A traceback
  and a `repr` read `valgebra.ValidationError` for the same reason.

## [0.0.9] - 2026-08-26

Two `Annotated` markers were read as something other than what they mean, and
both produced a schema that denoted nothing or its complement. No other decision
changes.

### Fixed

- A marker that is itself **callable** is asked, and only one that is not is
  taken apart by its `.func`. `annotated_types.Predicate` carries its callable
  there and is not callable, which is why the attribute is read at all — but
  `Not` and `functools.partial` carry one too.

  `Not(f)` denotes the values where `f` is false, and it defines `__call__`
  because calling is what applies the negation. Read by its `.func` it
  constrained by `f` instead, so every value under it got the opposite verdict:
  `Annotated[int, Not(is_even)]` admitted the even numbers.

  `partial(eq, 1)` lost its bound argument the same way and became `eq`, which
  raises when called with a single value, so the schema admitted nothing at all.

  Callability is the discriminator `annotated_types` itself encodes, so this is
  its rule rather than a heuristic.

- A **class** in `Annotated` metadata is ignored, as the typing spec asks of
  metadata a consumer does not recognise. A marker carries its values on an
  instance and a class carries the descriptors that read them: `Ge(0)` holds
  `ge = 0`, while `Ge` holds the slot descriptor, and reading that as a bound
  built a comparison no value is ordered against. A class is also callable, and
  calling one constructs rather than asks — a unit marker written as a class
  answered no question and refused every value. Both traps ended in a schema
  denoting nothing.

  Every other callable is still a predicate: a function, a lambda, a bound
  method, an object with `__call__`.

### Changed

- The documentation pages are numbered, so a page's URL carries its number:
  `/03-schema-language/` where it was `/schema-language/`. The site's landing
  page is unchanged.

- The published benchmark figures are re-measured, and the performance page now
  records the **interpreter build** its baseline was measured on rather than
  only the version: a free-threaded CPython runs this work about twice as slow
  as a GIL build of the same version. The ratios are unchanged — 7.6x on deep
  nesting, 2.0x on the wide record, 1.8x on the large array — and no decision or
  code path moved.

## [0.0.8] - 2026-08-25

Membership costs less on a refined schema, and no decision or message changes.

### Changed

- A violation's message is built when a violation is recorded rather than on
  every check. Naming a bound takes the bound's `repr`, and a value that belongs
  produces no violation to name it in, so accepting a value used to cost the
  size of the schema's *operand* rather than the size of the value.

  Per-check cost over the bare type, release build, median of eleven runs of
  fifty thousand: a single comparison bound about 16 ns where it was about 123,
  and two about 29 where they were about 258. A comparison bound therefore costs
  less than a call into a Python predicate, which is the ordering
  [the refinements page](https://ppigazzini.github.io/valgebra/05-refinements/)
  describes. The absolute figures are one machine's and move by around a tenth
  between runs; the ratio and the ordering are what travel.

  The operand's size no longer enters into it. A passing check against
  `Annotated[str, Ge(s)]` measures about 51 ns for a one-character `s` and about
  52 ns for a two-hundred-thousand-character one, against 143 ns and 303 us.

  Every violation carries the message it carried, and every decision is
  unchanged.

- The pages carry four things a reader could otherwise only find by experiment.
  The `Regex` dialect is the Rust engine's, not `re`'s, and a pattern **both**
  engines accept can match different strings — POSIX bracket expressions,
  Unicode case folding of the Turkish dotless i, and `\p{...}` property escapes,
  so compiling successfully is not a test of which language a pattern is in.
  The depth budget accounts for refinements: a refinement is a node, so it costs
  a level on top of whatever it narrows, and the marker it carries makes no
  difference. A map can constrain some keys and leave the rest free by giving
  the permissive clause the **complement** of the claimed keys, where `open`
  admits a clause matching every key and so subsumes a narrower one. And the
  decidability boundary records that a meet of two distinct literals is not
  decided empty, with no sound rule to close it: a literal's equality is the
  value's own, so two literals can share a member while neither contains the
  other.

## [0.0.7] - 2026-08-25

### Fixed

- `open`, `close`, and `simplify` rewrite a validator's recursive definitions as
  well as its root. A recursive validator's root is a single back edge and every
  record, union, and refinement it declares lives in the definitions table, so
  all three were no-ops on exactly the schemas that carry the most structure:
  `recursive(lambda n: {"a": int, "next": n}).open()` admitted no undeclared key,
  and `simplify` left a recursive body unreduced.

- The frontend descends as far as the construction bounds publish, so a schema
  those bounds accept is no longer rejected while being compiled. Sets, dicts,
  and records nest to the documented limit rather than to 100 levels, and the
  message a rejection carries reports how far the frontend descended rather than
  naming a cause it cannot know.

### Added

- `is_empty` decides an interval that skips every integer however the meet is
  spelled. `intersection(Annotated[int, Gt(0)], Annotated[int, Lt(1)])` is empty,
  as `Annotated[int, Gt(0), Lt(1)]` already was: an intersection is a subset of
  each of its members, so a member bounded to the integers bounds the whole meet.
  A `bool` base counts integers for the same reason, since it subclasses `int`.

- `is_subtype_of` decides attribute schemas across a class hierarchy. A dataclass
  or named tuple is below one over a base class whose every attribute it carries
  with a narrower schema, and below the bare class it is an instance of.

  Both are conservative-to-decided moves: every relation that held in 0.0.6 still
  holds, and each carries the counter-direction that keeps it from over-firing.

### Changed

- `open` and `close` may raise `ValueError`. Opening a record adds a catch-all
  clause, so it grows the schema and the construction bounds apply to what it
  produces; a validator near the node limit can cross it.

- A union of no members reprs as `nothing` and a meet of none as `anything`,
  because each constructor owns the identity of its own arity. `repr(union())`
  was the empty string, which is not an expression that rebuilds the validator.

- A widening between two eight-member literal unions costs 619 ns where it cost
  1.66 us, and a union of opaque members decides up to about eight hundred
  members where it decided up to four hundred. The lattice bound asking whether a
  supertype covers the universe reads that supertype's region set instead of
  building the complement of a deep clone of it, and the emptiness folds stop
  once no later member can change the verdict.

- Membership costs what it cost. The competitive baseline in
  [the performance page](https://ppigazzini.github.io/valgebra/11-performance/) is re-measured with its spread and
  the method that produced it: against pydantic in strict mode, 7.6x on a schema
  nested twenty-five deep, 2.1x on a fifty-field closed record, and 1.8x on a
  flat array of ten thousand integers.

  On upgrade, handle `ValueError` from `open` and `close` where a schema is built
  in a loop, and expect a recursive validator to answer as the recursion says it
  should rather than as its root alone did. Nothing else changes.

## [0.0.6] - 2026-08-08

### Changed

- Internal only, with no change to what any schema admits or answers: the schema
  IR's purely structural walks share one declaration of each node's child
  schemas, and the two index remappings applied when validators compose —
  appending a constants pool, or interning one into another — share one walk over
  one set of payload sites. Every public method returns what it returned in
  0.0.5, and validation, `is_empty`, the error model, `repr`, and every compiled
  form are unchanged.

  Nothing to do on upgrade. The version exists so the reorganisation ships under
  a release of its own rather than inside one whose entries describe something
  else.

## [0.0.5] - 2026-08-08

### Added

- `tests/test_completeness_probe.py`, which searches for relations answered
  `False` that no value in a wide universe refutes, and fails when one appears
  that is not written down with a reason. Every other instrument could only
  notice a completeness gap someone had already thought of; this one searches.

### Fixed

- `is_subtype_of` decides a **closed record against a catch-all mapping**:
  `{"x": int}` is recognised below `dict[str, int]`. The closed record had a
  dispatch branch of its own that read a field the supertype covers through a
  catch-all as undecided, though the general keyed-map rule beside it already
  decided exactly that. One rule serves every keyed-map shape.

- `is_subtype_of` decides **inclusion in a complement**: `A` is below `~B` when
  the two share no value, so `list[int]` is recognised below `~int` and
  `dict[str, int]` below `~str`. There was no rule for a complement on the right
  at all, so the relation was decided only when the left side was itself a
  complement.

  Both change a public method's answer from `False` to `True`. No relation that
  answered `True` can answer `False`, and validation, `is_empty`, compilation and
  every rendered form are unchanged.

- `is_subtype_of` and `is_equivalent` decide the lattice bounds by **emptiness**
  rather than by the shape of the atom. A schema that denotes the empty set
  without being spelled `nothing` — a record with an uninhabited required field,
  a cancelling intersection — is recognised as a subtype of every schema, and one
  that covers the universe without being spelled `anything` is recognised as a
  supertype of every schema. Both previously answered `False`, which was sound
  but incomplete.

  This changes the answer of a public method from `False` to `True` for those
  schemas. Nothing else moves: `is_empty`, validation, compilation and every
  rendered form are unchanged, and no relation that answered `True` can answer
  `False`. The gap was masked whenever the other side was scalar, because the
  region check decides that case correctly, so it was only visible against a
  container, a record, an instance or `typing.Any`.

## [0.0.4] - 2026-07-13

### Added

- The schema construction bounds are published as module constants —
  `MAX_SCHEMA_DEPTH`, `MAX_DEFINITIONS`, and `MAX_SCHEMA_NODES` — so a caller can
  size a schema against them.

### Changed

- Schema construction is bounded on every growth path, not only the combinator
  operators: the `Validator` constructor, the `|` operator, `union`,
  `intersection`, `complement`, `recursive`, and `simplify` all reject a schema
  past a fixed nesting depth, recursive-definition count, or total node count
  with a `ValueError`. `simplify` can therefore raise when negation-normal form
  expands a schema past the size bound (see `docs/10-limits.md`).
- The project describes its schema algebra as *closed* under its operations
  rather than *complete*.

### Fixed

- No sequence of public calls can overflow the native stack or exhaust memory
  while building a validator. A schema grown too deep through the `Validator`
  constructor or `recursive`, or too large by combining a validator with itself
  in a loop, is rejected at construction instead of crashing the interpreter on a
  later clone, drop, decision, or render. This extends the 0.0.3 composition
  bound, which rejected only the combinator operators and left the constructor
  and `recursive` paths unbounded.
- `Annotated[int, MultipleOf(0)]` is rejected when the validator is compiled: no
  value is a multiple of zero, so the unsatisfiable schema raises a `ValueError`
  at construction instead of rejecting every value through a swallowed
  `ZeroDivisionError` at validation time.

## [0.0.3] - 2026-07-07

### Added

- Schema composition bounds nesting depth: combining validators with `|`,
  `union`, `intersection`, or `complement` past a fixed depth is rejected at
  construction with a `ValueError`, so a schema grown in an unbounded loop cannot
  overflow the native stack on its next check. A `recursive` back edge counts as
  a leaf, so a recursive schema's depth stays finite (see `docs/10-limits.md`).
- The type stub declares the `__copy__` and `__deepcopy__` methods a compiled
  validator exposes.

### Fixed

- Release builds report a well-formed validation error instead of panicking
  across the boundary when the error builder is handed no failures, and the
  per-element sequence walk folds an impossible missing-schema case to a
  non-member result rather than a panic.

## [0.0.2] - 2026-06-30

### Added

- `valgebra.__version__` exposes the installed distribution version, read from
  the package metadata that maturin derives from the Cargo workspace manifest.

## [0.0.1] - 2026-06-29

The first published release. valgebra ships to PyPI as prebuilt wheels across
the support matrix.

### Added

- Compile-once / validate-fast engine: `Validator(schema)` builds an immutable
  validator with `validate` (raises), `is_valid` (bool fast path), and `ensure`.
- Typing-annotation frontend: scalars, `None`, `Any`, `list`/`set`/`frozenset`/
  `dict`, fixed, variadic, and prefix-plus-tail tuples (`tuple[A, B, ...]`),
  unions and `Optional`, `Literal`, `TypedDict`, dataclasses, `NamedTuple`,
  enums, runtime-checkable protocols, `NewType`, PEP 695 aliases, and
  `Annotated` refinements (with bounds, length, and predicate constraints).
- Native forms: a list literal as a sequence — `[T]`, the fixed `[A, B]`, and the
  prefix-plus-tail `[A, B, ...]` (a fixed prefix then a repeated tail); a dict
  literal as a closed record (`"key?"` optional); a single `{KeyType: ValueType}`
  entry as a mapping; and any constant as a typed literal.
- A closed Boolean algebra: `union`, `intersection`, `complement`, `anything`,
  `nothing`, and a law-justified `simplify`, with the lattice laws
  property-tested. Conditional fields and key cardinality are composed from these
  (documented recipes), not shipped as combinators.
- Set-relation queries on a compiled validator: `is_subtype_of` (set inclusion),
  `is_equivalent` (mutual inclusion), and `is_empty` (an unsatisfiable schema,
  including a recursive schema with no base case). Decided soundly across
  scalars, containers, records and mappings, sequence forms, class subtyping
  (`issubclass`), and literal values (by membership), and conservative on the
  cases it cannot prove.
- Recursive schemas via the `recursive` fixpoint, with cycle and depth guards.
- A structured, machine-readable error model: aggregated failures, opt-in
  fail-fast, and closest-branch reporting for unions.
- JSON input on the Rust path: `validate_json`, `is_valid_json`, and `load`
  (validate and return the parsed value), consistent with the object path and
  faster than parse-then-validate.
- A stable `repr` that renders a schema back to its annotation form.
- Thread-safe, immutable validators.
- A performance program: criterion and pytest-benchmark suites, a recorded
  baseline against pydantic-core and jsonschema, and a deterministic
  instruction-count CI regression gate.

[Unreleased]: https://github.com/ppigazzini/valgebra/compare/v0.0.9...HEAD
[0.0.9]: https://github.com/ppigazzini/valgebra/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/ppigazzini/valgebra/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/ppigazzini/valgebra/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/ppigazzini/valgebra/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/ppigazzini/valgebra/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/ppigazzini/valgebra/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/ppigazzini/valgebra/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/ppigazzini/valgebra/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ppigazzini/valgebra/releases/tag/v0.0.1
