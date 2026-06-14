# Changelog

All notable changes to valgebra are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No versions are published yet. The following is in place on the main branch and
will form the first release.

### Added

- Compile-once / validate-fast engine: `validator(schema)` builds an immutable
  validator with `validate` (raises), `is_valid` (bool fast path), and `cast`.
- Typing-annotation frontend: scalars, `None`, `Any`, `list`/`set`/`frozenset`/
  `dict`, fixed, variadic, and prefix-plus-tail tuples (`tuple[A, B, ...]`),
  unions and `Optional`, `Literal`, `TypedDict`, dataclasses, `NamedTuple`,
  enums, runtime-checkable protocols, `NewType`, PEP 695 aliases, and
  `Annotated` refinements (with bounds, length, and predicate constraints).
- Native forms: a list literal as a sequence — `[T]`, the fixed `[A, B]`, and the
  prefix-plus-tail `[A, B, ...]` (a fixed prefix then a repeated tail); a dict
  literal as a closed record (`"key?"` optional); a single `{KeyType: ValueType}`
  entry as a mapping; and any constant as a typed literal.
- A complete Boolean algebra: `union`, `intersect`, `complement`, `anything`,
  `nothing`, the derived `ifthen`/`cond`, and a law-justified `simplify`, with
  the lattice laws property-tested.
- Set-relation queries on a compiled validator: `is_subtype` (set inclusion),
  `equivalent` (mutual inclusion), and `is_empty` (an unsatisfiable schema, including
  a recursive schema with no base case). Decided soundly across scalars,
  containers, records and mappings, sequence forms, class subtyping
  (`issubclass`), and literal values (by membership), and conservative on the
  cases it cannot prove.
- Recursive schemas via the `lazy` fixpoint, with cycle and depth guards.
- A structured, machine-readable error model: aggregated failures, opt-in
  fail-fast, and closest-branch reporting for unions.
- JSON input on the Rust path: `validate_json` and `is_valid_json`, consistent
  with the object path and faster than parse-then-validate.
- A stable `repr` that renders a schema back to its annotation form.
- Thread-safe, immutable validators, usable on free-threaded CPython.
- A performance program: criterion and pytest-benchmark suites, a recorded
  baseline against pydantic-core and jsonschema, and a deterministic
  instruction-count CI regression gate.

[Unreleased]: https://github.com/ppigazzini/valgebra/commits/main
