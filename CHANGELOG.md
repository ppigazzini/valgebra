# Changelog

All notable changes to valgebra are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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

### Fixed

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
  [the refinements page](https://ppigazzini.github.io/valgebra/refinements/)
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

- Involution and the excluded middle are decided for every schema: `~~A` is `A`
  and a union carrying a schema together with its complement is the top. Both
  are settled where the schema is built, so the decision procedure never meets
  either shape and pays nothing per comparison. `repr` and `==` follow the
  cancelled form: `complement(complement(int))` is `int`.

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

- `is_subtype_of`, `is_equivalent` and `is_empty` normalise their arguments by
  the lattice laws before comparing, so involution decides in both directions
  for every schema and a union covering the universe is recognised as the top.
- A literal carries the kind of its constant, so it is decided against another
  kind: `Literal["a"]` is below `~int`, and `Literal["a"] & Literal["b"]` is
  empty. `Literal[1]` and `Literal[True]` are disjoint although `1 == True`,
  because a literal pins `type(x)` exactly. The rule applies to the builtin
  scalars, whose equality is Python's own; a meet of two `Enum` members stays
  conservative, since user-defined equality can admit one value for two
  constants.

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
  [the performance page](https://ppigazzini.github.io/valgebra/performance/) is re-measured with its spread and
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
