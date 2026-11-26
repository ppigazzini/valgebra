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
rather than a silent break in someone's error handling. `docs/guides/error-model.md`
is the user-facing statement of the same thing.

**A code is a name in a table.** Half the vocabulary is the core's: a leaf
mismatch reports the node's own code, and `Schema::error_code` in
`crates/valgebra-core/src/ir.rs` is the arm per node. The other half is the
binding's -- a bound that failed, a key that is missing, a walk that ran out of
levels -- and `crates/valgebra-py/src/codes.rs` declares each as a `Code`
constant with what it means. Every site writes one of those constants'
`as_str()`. Ten codes are in both tables, because a walk that builds a node's
failure itself, mostly holding the node's parts rather than the node, names the
node's code with a binding constant: those are declared once, in
`crates/valgebra-core/src/codes.rs`, which the core's arms and the binding's
constants both read. `Code::of_schema` is the crossing for a leaf mismatch,
where the node is in hand. The compiler does not hold a site to that:
`Violation::code` is a `&'static str`, so a string spelled out at a site would
compile, and what refuses it is `tests/test_code_table.py`, which also refuses
a binding constant that spells one of the core's codes again.

Declared in two tables, the set of codes is read rather than recovered. A scan
for strings that look like codes would miss a code written in a file nobody
thought to scan and invent a cell for any other snake-case string in one that
was. `scripts/use_case_ledger.py` reads the two tables, and
`tests/test_code_table.py` holds the arrangement in both directions: a name the
table declares is one some site writes, and a violation built with a code
spelled out fails.

**A failure is raised as the instance it is.** `into_pyerr` in `errors.rs` builds
the `ValidationError`, attaches the failures, and hands the instance to `PyO3` as
its type's argument, so the raise is `PyErr_SetObject` with the instance: the
exception being handled is chained as `__context__`, as a Python `raise` would
chain it. A lazy `PyErr` would have to be normalized to take the failures, and
`PyO3` normalizes by detaching from the interpreter and attaching again -- a
hand-off of the lock, once per failing call, which lets any waiting thread run
in the middle of a raise. `PyErr::from_value` avoids the hand-off and loses the
chaining, since it raises through `PyErr_SetRaisedException`;
`test_a_failure_raised_while_handling_another_chains_it` holds the chaining.

## The path is segments, not a string

`PathSegment` in `crates/valgebra-core/src/ir.rs` is one step of a location, and
`Violation::location` renders a sequence of them as `name[2].id`. Keeping the
steps apart is what lets a consumer walk down to the offending value rather than
parse a string back into steps. The rendering is for a reader, and it keeps two
paths apart too: a key that is not a bare name is written as a subscript of its
`quoted` literal, `['a.b']`. Written bare, the key `"a.b"` would read as the
path `a` then `b` and `"[0]"` as an index, and a key holding a newline would
split the one-line message. `quoted` is also how a pattern, and a key or an
attribute a record names, is spelled in an `expected` label, as `repr` spells a
literal's constant: a missing key reads `required key 'a'`, and one the path
writes as `['\x7f']` reads `required key '\x7f'`.

There are **five** variants, and the split between the first three is the whole
reason the path is usable on a dict:

| variant | what it addresses |
|---|---|
| `Key(Arc<str>)` | a string key. Shared rather than owned, so a record's failure path carries the name the schema already declares instead of allocating one per failing field |
| `IntKey(i64)` | an integer key. Separate from `Key` because `d[2]` and `d["2"]` are different entries of one dict, and rendering the integer as its text would make them indistinguishable |
| `BigIntKey(String)` | an integer key past `i64`. Python's integers are unbounded and a dict may be keyed by any of them; the core holds no Python object, so the digits travel and the binding rebuilds the `int`. Past the interpreter's digit limit the digits are hexadecimal, and the binding's tuple holds them as text: an `int` there is one `json.dumps` and `repr` refuse, and every item is JSON-serializable |
| `Index(usize)` | a **position** in a sequence, which is not a key at all |
| `Entered` | the key before it, entered by a failure **inside** it. The binding's tuple ends there, since no subscript reaches into a key, and the location writes the segments after it as the key's own, `('x',), key[0]`, apart from `('x',)[0]` in its value. It is no level of the value, so `PathSegment::steps` leaves it out where a union explains its deepest branch; and it holds nothing, since a variant holding a segment made the type recursive and its drop a call at every level an explaining walk pops (+2.07% on the record explain shape). `entered_the_key` in the record walk marks the key once its own walk has failed inside it |

A consumer that flattens the path to strings loses the distinction the type
exists to keep. `docs/guides/error-model.md` states the same split for a caller, in
the form the Python tuple takes.

`Index` is a position in the **value being validated**, so it is not one of the
validator's own index spaces and shares no type with them
([the type design page](type-design.md)).

## Aggregation is the caller's choice

`validate` takes `fail_fast` and `limit`, and `ensure`, `validate_json` and
`load` take them to mean the same. By default it aggregates a violation
for each independent failure — every record field, sequence element and mapping
entry — `fail_fast` stops at the first, and `limit=n` at the `n`th. The walk
carries the choice as a mode rather than a flag ([the walk page](walk.md)),
`ExplainLimited` for a count past one, and the count as `WalkState::limit`
(`crates/valgebra-py/src/check/ctx.rs`): it holds for the whole call, and a sub-walk that reports nothing -- a
union's branch, a complement's inner walk -- runs in the fast or settling mode
beneath it. The count sits in the state's slot for what a walk rarely holds, set
by a limited walk alone, so a call of any other mode writes no word for it.

A limited report is the full report's first `n`, which is what makes it a count
rather than a sample. Every composite that stops at its first failure under
`fail_fast` stops once its report holds `n` (`stop` and `reported_enough` in
`check/walk.rs`); each report is a prefix of the full one, since a union chooses
by its branches' first failures and a set sorts its elements' reports, a report
cut at `n` sorting as the whole one does in its first `n`; and the entry point
cuts the report to `n`, since one child can add several. The memo answers a
limited walk only where nothing is reported yet, as it answers a fail-fast one:
a report kept after some failures stopped at fewer than one begun before them
needs. `test_a_limit_reports_the_first_failures_the_full_report_lists` in
`tests/test_error_contract.py` holds the prefix over drawn schemas.

Independent means what it says: two bad fields of one record are two violations,
and a bad field of a bad element is one, because the walk stops descending where
it fails. A wrong length is such a failure: a fixed shape's arity and a
container's length bound (`lengths_before_elements` in
`crates/valgebra-py/src/check/walk/scalar.rs`) are read after the kind test and
before any element, so a list refused for its length reports that alone.
That holds inside one refinement node; a bound that is its own member of a
meet, from `intersection` or a validator in `Annotated`'s metadata, is read
in the meet's turn, so the report carries the elements' violations beside it.

## Rendering a schema back to an annotation

`render` produces the annotation that would compile to the schema, and it is what
`repr(validator)` shows. Three properties it must keep:

- **It terminates on a recursive schema.** A back edge shows as the parameter
  of the `recursive` lambda that bound it, so the form is finite and reads back
  as the schema it came from. `<...>` (`GAVE_UP` in `render.rs`) marks where the
  renderer has no expression to give, which a compiled validator reaches only
  past `MAX_RENDER_DEPTH`, through a chain of definitions. It is a syntax error
  wherever it lands, so a truncated form cannot read back as a different schema;
  `...` is only ever an annotation's own spelling, `tuple[T, ...]`.
- **It is stable under a rebuild.** A rendered form reads back as a schema that
  renders the same way, so `repr` is a fixed point rather than a form that
  drifts each time it is built again.
- **It rebuilds or refuses, never another schema.** A binder is a name in scope
  over its whole body, so one equal to a name the body prints would read that
  name back as the back edge; the binders are named after the render, each the
  first spelling of its level's letter that no word of the text takes, read
  under the interpreter's NFKC normalization (`name_binders`). A constant
  prints by its own `repr` whole, since one cut where an error message cuts
  does not read back, and a literal's constant whose `repr` is a builtin's
  spelling of a subclass's value prints in angle brackets, since it would read
  back as the builtin (`literal_repr`); so does a bound's or a step's operand
  of such a subclass writing an order or a remainder of its own, which the walk
  reads reflected ahead of the builtin's, so that the builtin's bound read back
  is another set (`operand_repr`). Every refusal is a syntax error or a
  call the frontend refuses, and `docs/reference/api.md` lists them.
  `tests/test_repr.py` holds each, and the pytest mutation sweep, which owns
  `render.rs`, reads them there.

## Refused, and the test each one fails

A proposal to change what a failure is raised as starts here. Each row says
what it fails, not that it was turned down.

**A failure raised as an exception group.** A `ValidationGroup`, both a
`ValidationError` and an `ExceptionGroup`, holding one leaf per failure, would
let a caller split a report with `except*` by a leaf's class, and give a
union's refusal one sub-report per branch as a group of groups. The interpreter
prints a group as a tree: the group's line, then each leaf boxed. So a single
failure raised as a group of one prints its line twice, uncaught or through
`logging.exception`, on CPython and on PyPy alike:
`test_an_uncaught_failure_prints_as_the_line_it_carries` in
`tests/test_error_model.py`. A raise would also build one exception object per
failure, where today the failures stay in Rust until a caller reads one
(`test_the_error_model_is_built_when_it_is_asked_for`). The one split a
caller's handling turns on, a walk that stopped against a value refused, is
in the code already: `docs/guides/error-model.md` names the five codes a stopped
walk raises.

**A report that grows fallibly, raising `MemoryError` where the next slot will
not fit.** Under a memory limit (`RLIMIT_AS`) the report of a value with ten
million failures ends the process when its vector doubles past the limit, and
reserving each slot fallibly would raise `MemoryError` there instead. It fails
on what it leaves: every site that records a failure grows the report, and each
failure allocates its path and two strings besides its slot, so a fallible
vector converts the one large allocation and leaves every small one aborting
where the limit falls on it, and a process near its limit still ends. pydantic
2.14's `validate_python` ends the same way on the same value under the same
limit: a Rust allocation fails and the process aborts. A caller that bounds the
report asks for `limit=n`, which collects `n` failures, or `fail_fast=True`,
which collects one (`test_fail_fast_stops_at_the_first_failure` in
`tests/test_error_aggregation.py`), and the limits page says a Rust allocation
that fails ends the process.

## The limit

**A summary is a summary.** `value_summary` truncates, and a long or exotic
`repr` is not reproduced in full. It exists to identify the value in a message,
not to reconstruct it.

**`render` is not a round-trip guarantee.** It produces *an* annotation that
compiles to the schema, not the one the user wrote: a schema built through the
combinators may have no annotation spelling, and one that does may differ in
member order from the spelling it was written in, because the constructors order
a union's members when they build it and the render prints each run of one kind
in a set in the order its renderings sort in (`in_reading_order` in `render.rs`),
so that two schemas `==` calls one print alike.
