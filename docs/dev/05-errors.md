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

**A code is a name in a table.** Half the vocabulary is the core's: a leaf
mismatch reports the node's own code, and `Schema::error_code` in
`crates/valgebra-core/src/ir.rs` is the arm per node. The other half is the
binding's -- a bound that failed, a key that is missing, a walk that ran out of
levels -- and `crates/valgebra-py/src/codes.rs` declares each as a `Code`
constant with what it means. Nothing else builds one: a violation takes a
`Code`, so a site cannot invent a string, and `Code::of_schema` is the single
crossing from the core's table to this one.

Written out at the call sites, as they were, the set of codes could only be
recovered by scanning files for strings that looked like codes -- which misses a
code written in a file nobody thought to scan and invents a cell for any other
snake-case string in one that was. `scripts/use_case_ledger.py` reads the two
tables instead, and `tests/test_code_table.py` holds the arrangement in both
directions: a name the table declares is one some site writes, and a violation
built with a code spelled out fails.

## The path is segments, not a string

`PathSegment` in `crates/valgebra-core/src/ir.rs` is one step of a location, and
`Violation::location` renders a sequence of them as `name[2].id`. Keeping the
steps apart is what lets a consumer walk down to the offending value rather than
parse a string back into steps.

There are **four** variants, and the split between the first three is the whole
reason the path is usable on a dict:

| variant | what it addresses |
|---|---|
| `Key(Arc<str>)` | a string key. Shared rather than owned, so a record's failure path carries the name the schema already declares instead of allocating one per failing field |
| `IntKey(i64)` | an integer key. Separate from `Key` because `d[2]` and `d["2"]` are different entries of one dict, and rendering the integer as its text made them indistinguishable |
| `BigIntKey(String)` | an integer key past `i64`. Python's integers are unbounded and a dict may be keyed by any of them; the core holds no Python object, so the digits travel and the binding rebuilds the `int` |
| `Index(usize)` | a **position** in a sequence, which is not a key at all |

A consumer that flattens the path to strings loses the distinction the type
exists to keep. `docs/08-error-model.md` states the same split for a caller, in
the form the Python tuple takes.

`Index` is a position in the **value being validated**, so it is not one of the
validator's own index spaces and shares no type with them
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

- **It terminates on a recursive schema.** A back edge shows as the parameter
  of the `recursive` lambda that bound it, so the form is finite and reads back
  as the schema it came from. `...` marks one thing only, the render depth bound.
- **It is stable under a rebuild.** A rendered form reads back as a schema that
  renders the same way, so `repr` is a fixed point rather than a form that
  drifts each time it is built again.

## The limit

**A summary is a summary.** `value_summary` truncates, and a long or exotic
`repr` is not reproduced in full. It exists to identify the value in a message,
not to reconstruct it.

**`render` is not a round-trip guarantee.** It produces *an* annotation that
compiles to the schema, not the one the user wrote: a schema built through the
combinators may have no annotation spelling, and one that does may differ in
member order from the spelling it was written in, because the constructors order
a union's members when they build it.
