# The error model

`crates/valgebra-core/src/violation.rs` owns the structured failure.
`crates/valgebra-py/src/errors.rs` turns it into the Python exception.
`crates/valgebra-py/src/render.rs` turns a schema back into an annotation string.

## A violation is a value, not a message

```rust
pub struct Violation {
    pub code: &'static str,       // stable, machine-readable
    pub path: Vec<PathSegment>,   // location from the root; empty at the root
    pub expected: String,         // short label of the expected set
    pub value_summary: String,    // repr-style summary of the offending value
}
```

The rendered sentence is derived from those four, not stored. A caller that wants
to branch on a failure reads `code`; a caller that wants to point at the input
reads `path`. Formatting is the last step and never the record.

**The code, the exception type and the path format are part of the documented
API.** They are pinned by snapshot tests, so a wording change is a reviewed diff
rather than a silent break in someone's error handling. `docs/08-error-model.md`
is the user-facing statement of the same thing.

## The path is segments, not a string

`PathSegment` is a key or an index, and `Violation::location` renders the pair as
`name[2].id`. Keeping them apart is what lets a consumer walk to the offending
value rather than parse a string back into steps.

`PathSegment::Index(usize)` is a position in the **value being validated**. It is
not one of the validator's index spaces and shares no type with them
([06-type-design.md](06-type-design.md)).

## Aggregation is the caller's choice

`validate` takes `fail_fast`. False aggregates a violation for each independent
failure — every record field, sequence element and mapping entry — and true stops
at the first. The walk carries that as a mode rather than a flag
([04-walk.md](04-walk.md)), so the mode is a fact about the whole walk and cannot
change mid-traversal.

Independent means what it says: two bad fields of one record are two violations,
and a bad field of a bad element is one, because the walk stops descending where
it fails.

## Rendering a schema back to an annotation

`render` produces the annotation that would compile to the schema, and it is what
`repr(validator)` shows. Two properties it must keep:

- **It terminates on a recursive schema.** A back edge to a reference already
  being rendered shows as `...`, so the form is finite.
- **It is stable under simplification.** The node matrix asserts
  `repr(simplify()) == repr(simplify().simplify())`, so a rendered form does not
  drift under a second normalisation pass.

## The limit

**A summary is a summary.** `value_summary` truncates, and a long or exotic
`repr` is not reproduced in full. It exists to identify the value in a message,
not to reconstruct it.

**`render` is not a round-trip guarantee.** It produces *an* annotation that
compiles to the schema, not the one the user wrote: a schema built through the
combinators may have no annotation spelling, and one that does may differ in
member order after simplification.
