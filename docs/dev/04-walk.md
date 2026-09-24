# The membership walk

`crates/valgebra-py/src/check/walk.rs` decides whether a value belongs to a
schema's set. It is where soundness is decided, and it is the file to read first
when an accept looks wrong.

## One walk, two input paths, three modes

There is **one** `member` function, parameterised by mode rather than a fast one
and an explaining one side by side. Two walks drift, and the defect that follows
is an answer that depends on which of them ran.

It runs over a `Value`, which is either a borrowed Python object or a borrowed
parsed JSON value. That is what keeps the object path and the in-place JSON path
membership-equivalent by construction rather than by a test that compares them.

**Only `is_valid_json` takes the in-place path.** `validate_json` and `load`
parse the document into Python objects and then run the *object* walk over them,
because both hand back something a caller reads — `load` the value itself, and
`validate_json` a report whose `value` summaries are reprs of Python objects. So
a `Value::Json` is never explained: the explaining arms of the JSON walk are
unreachable by construction, and a document reaches the same codes with the same
`loc` a Python value reaches, which is what
[08-error-model.md](../08-error-model.md) promises. The in-place path exists for
the question that needs no objects at all, and that is the one that answers a
`bool`.

**A streaming check is refused.** Reading the document with jiter's pull
parser instead of its tree saves at most the tree's construction and drop, and
nothing where a node reads its value again -- a union reads it per branch, a
meet per member, a complement and a refinement after their inner schema. A
stream is read once, so it cannot be a `Value`: it would be the second walk this
page refuses, and it answers differently from the tree on documents the tests
hold. `next_skip` does not check UTF-8 in a value the schema never reads
(`test_invalid_utf8_in_an_unread_value_is_not_a_document`); every jiter
iterator call starts a fresh nesting budget where the tree counts from the root
(`test_the_nesting_limit_counts_from_the_root`); and a predicate would run on
values of a document that turns out malformed, or on a key the document repeats
(`test_a_predicate_sees_only_what_a_parsed_document_holds`). The three are in
`tests/test_json.py`, and each passes because the check reads a finished parse.

`WalkMode` names what the walk is for:

| Mode | Reports |
|---|---|
| `Fast` | membership only: nothing allocated, no path built, every composite short-circuits |
| `Explain` | a violation for each independent failure — every field, element and entry |
| `ExplainFailFast` | the first violation only |

Three modes, and the type says three. The pair of independent booleans this
replaced admitted a fourth combination that no caller produced. The
discriminants are ordered so both predicates the walk asks per node are a single
comparison, and a test pins both over every variant — that order is load-bearing
and measured, not cosmetic ([06-type-design.md](06-type-design.md)).

## The comparison-raises policy

Membership reads a value through Python operations that can raise: `__eq__` for a
literal, a rich comparison for a bound, `isinstance` for a class, `getattr` for
an attribute, `__mod__` for a multiple-of, `__len__` for a length.

**One rule across every such site: a value whose comparison, instance check or
attribute access raises an ordinary exception is a non-member.** A value that
cannot answer "are you in this set?" is not in it. This matches pydantic-core.

The one carve-out is a **user predicate**, whose raised error surfaces as a
distinct `predicate_error` rather than folding, so a buggy predicate is visible
instead of silently rejecting everything.

## The one error never folded

A **fatal interpreter signal** is not an answer to "are you in this set?" — the
interpreter is unwinding. `is_fatal` classifies it as two disjoint cases:

- a base exception that is not an ordinary exception: `KeyboardInterrupt`,
  `SystemExit`, `GeneratorExit`;
- `MemoryError` and `RecursionError`, which **are** ordinary exceptions, so the
  base-exception test alone misses them, and which mean the interpreter cannot
  continue.

The first such signal is recorded; the walk then short-circuits — every later
`member` call returns at once — and the entry point re-raises it. A `Cell` mirror
of "has a signal been seen" is read per node with a plain load, so the fast path
does not take a `RefCell` borrow on every step.

Each disjunct needs its own test case: a mutation collapsing the classifier to
one of them is invisible to a corpus that only raises `KeyboardInterrupt`.

## The union reports the closest branch

When no branch of a union matches, dumping every branch's errors buries the one
the value was closest to. The walk instead reports the branch that **descended
furthest** into the value before failing, measured as the greatest path depth
past the union's own location, with a tie keeping the earliest branch so the
choice is deterministic. When no branch makes progress — every branch a flat type
mismatch — it falls back to one union error.

The probe aggregates **regardless of fail-fast**, so the whole of the closest
branch is reported even to a caller that asked to stop at the first violation.
That is the point: the caller asked for less noise, not for less of the one
branch that matters.

The probe asks a predicate again wherever it re-walks one, and keeps no memo
over the answers. A predicate is user code, so each occurrence the walk
reaches is a call, and a cache keyed on the value's identity would decide for
an impure predicate which of its answers counts. The refinements page states
the count a caller sees; a predicate that must run once per value is memoised
on the caller's side, where the key is the caller's to choose.

## What is precomputed, and what correctness may not depend on

Three per-validator indexes are built once on first use and keyed by the address
of a node's own buffer:

- declared-field lookups per record;
- value sets for unions whose members are all literals;
- compiled patterns per regex source.

A thread that finds another building them waits **detached** from the
interpreter. The build interns strings, and on a free-threaded interpreter that
can wait on a lock and let a stop-the-world pause begin, which a waiter still
attached would never reach. Once built, the read is the same load either way.

**Correctness never depends on one being present.** A node absent from an index
falls back to building the map, scanning the branches, or recompiling the
pattern. The literal-union plan is consulted only on the membership path and only
for an exact int or str — an explain walk, a non-literal union, another value
type and a JSON value all fall through to the linear scan, which stays the one
source of truth for behaviour.

## Where the walk lives

`crates/valgebra-py/src/check/walk.rs` holds the dispatcher `member` and the
arms that read a value as a union, a meet, a complement, a class or a
reference. What the dispatcher descends into, and what it stops at, have a
module each, because what each shares with the rest is the dispatcher and
little else.

Two files beside them are read by every arm. `crates/valgebra-py/src/input.rs`
is the `Value` above — the two input paths and the decoders that produce them.
`crates/valgebra-py/src/check/index.rs` is the precompute the next section is
about: the record-field lookups, the literal-union tables and the compiled
patterns, built once with the validator and read by the walk that uses them.

`walk/scalar.rs` answers what a value is **without descending into it**: a
scalar kind, a literal, and the constraints that narrow one. A container's
length is answered there too, for the same reason -- `MinLen` counts what a
value holds without reading any of it. A constraint's operand is borrowed out
of the pool for as long as the walk runs, so a check that passes takes no
reference to it: on a free-threaded interpreter a reference is an atomic write
to a counter every thread sharing the validator touches.

`walk/record.rs` reads a value as a **keyed map or an attribute record**: the
shape whose membership is a question per key rather than per position -- which
keys the value carries, which of them the schema declares, and what a key the
schema does not declare is covered by. A keyed map is read one of three ways.
**By its keys** (`keyed_map_asks_for_its_keys`): one probe per declared field,
through the `RecordPlan` built with the validator, and a count of the entries
found against the entries the value holds -- a value holding exactly its
declared keys has no undeclared key for any clause to govern. **By a scan**
(`keyed_map_scan`): every entry of the value, each key resolved by name and
each undeclared key read against the clause that covers it. **Explaining**
(`keyed_map_explain`): the scan that reports every violation rather than the
first. Which of the first two a record takes is `Undeclared::of` reading the
record's clauses: an undeclared key is *refused* (no clause), *admitted* (the
top clause), *admitted when it is a string* (`str: anything`, which is what a
`TypedDict` builds), or *read* (any other clause). The invariant that makes
the by-keys path sound is that it never probes an undeclared key, so it may
take a record only where the clause's verdict on such a key is stated without
the key's value: the three readings above say it -- refused is `False`,
admitted is `True`, a string clause is `isinstance(key, str)` over every key
alike -- and a clause that reads a key together with its value keeps the scan.
The JSON reading takes the plan for an open record too, since a document's
keys are strings by the grammar. `walk/sequence.rs` reads one as a run of
**elements** -- a list, a tuple, a parsed array, a set, a frozenset -- which
differ in how an element is reached and agree on what each must be, and which
share an arity, a count taken once and compared again, and the snapshot a list
of one scalar kind is read through.

All three read the same `Frame`: where the walk is in the value, what it has found
there, and the context it may look things up in. A walk needing a different one
-- a union probing a branch into a buffer of its own, a clause pair deciding on
the fast path -- builds it from the parts it keeps.

## A container is read against a count taken once

Membership runs arbitrary Python at almost every entry — a predicate, an
`__eq__`, an `isinstance` hook — and a free-threaded interpreter lets another
thread write to the container meanwhile. Every container the walk does not own
is therefore read against a count taken at entry and re-read before each step
and after the last: `scan_dict` over entries, `scan_set` over the iterator, and
`scan_list` over positions. A count that moved makes the reading cover no state
the value was ever in, so the scan answers `Scan::Unreadable` and the caller
reports `mutated_during_validation` rather than answering from the part it saw.

The three differ only in what they count, and the sequence one is the case that
argues for all of them. A list is walked *by position* against a length read
once, so a list that grows hides its new items from the walk and one that
shrinks leaves the walk answering about items that are gone — and in both
directions `is_valid` returned `True` for a value that is not a member. A
**tuple** cannot be resized, so its arm keeps the plain iterator; a JSON array
is owned by the parser and cannot move at all.

**A tuple's length comes from the storage, and which reading gives it depends
on the type.** `PyTuple_Size` reads the storage on CPython and goes through the
object's own `__len__` on PyPy's `cpyext`, so a subclass that *overrides*
`__len__` answers the C accessor with whatever it likes — and a walk that
indexed against that read past the end of the allocation and took the process
down. Such a value is copied through the base type's own slot and the copy is
walked.

A subclass that **inherits** `tuple.__len__` is not copied, because there is
nothing to distrust: the overridden answer and the base's answer are the same
function, so the accessor reads the storage on every interpreter. That is every
`NamedTuple`, which is the tuple subclass a program is most likely to hold. The
walk tells the two apart by asking the type whether its `__len__` *is* the
base's, which costs one type-attribute lookup per validation — about 10 ns on a
three-field `NamedTuple`, where copying cost 45. Telling them apart by "is this
exactly a tuple" instead copied every `NamedTuple`: on CPython 3.14 that was
100 ns against the 57 a plain tuple takes, which is what the repair above cost
before this one was found.

**Every container the walk reads answers this way, and for one reason.** A
schema over a container denotes what the value *holds*, so the reading of it
cannot be a method the value chooses: a `str`, `bytes`, `list`, `tuple`, `set`,
`frozenset` or `dict` subclass that overrides `__len__` is counted through its
base type's slot, and a `set` or `frozenset` subclass that overrides `__iter__`
is walked through its base type's iterator. Believing an override admits a value
whose storage the schema excludes — `set[int]` held a subclass whose storage
carried a `str`, and `MinLen(3)` held one character — which is an accept no value
supports. The exactness test comes first at every one of them, so an exact
container and an inheriting subclass keep the reading their storage already
gives.

The two slots are asked of the base type rather than through a C accessor for
the reason the tuple paragraph gives, and the iterator the base returns is the
builtin one, so a set that changes size during the scan still raises where the
scan expects it to.

The cost is one length read per element, which is a pointer dereference: the
`large_array` shape of the comparative gate moved 0.881 to 0.886 against
pydantic-core when the sequence guard landed.

## A list of one scalar kind is read through a snapshot of it

Reading an element out of a list hands back an *owned* handle: a reference count
written when the handle is made and again when it drops, on an object the walk
only type-tests. Copying the list into a tuple pays the same two counts inside
the interpreter, in two loops carrying no dependent work between them, and the
tuple is frozen -- so its elements are read borrowed and the walk pays neither.
A ten-thousand element `list[int]` costs 1.64 ns per element against 4.38 on
CPython 3.12, and 2.45 against 8.31 on the free-threaded build, where the copy
also takes the container's lock once rather than once per element.

Which reading is cheaper is a property of the **interpreter**, so the walk asks
one. CPython 3.14 makes the count pair cheap enough that the copy is pure cost
and the walk reads in place there. Asking requires the interpreter's own flags,
which reach the crate that emits them and no other, so `crates/valgebra-py/build.rs`
re-emits them; without it such a question reads "an older interpreter" against
every interpreter, silently, and the fast path is taken everywhere.

Two widths bound the copy, `SNAPSHOT_MIN_ELEMENTS` and `SNAPSHOT_MAX_ELEMENTS`
in `crates/valgebra-py/src/check/walk/sequence.rs`: below the first it cannot pay for its
own allocation, above the second walking it costs more cache than the counts it
avoids, and the transient stops at two mebibytes. Both are in the bounds table of
[00-architecture.md](00-architecture.md), and neither changes an answer.

The contract of the section above is kept: the copy answers about the list as it
was when the copy was taken, so the count is compared again afterwards and a
value that moved reports the move. The **instruction** count moves the other way
-- the copy is instructions and the stall it removes is not, so a sixty-four
element walk executes 47% more of them -- which is why `scripts/perf_budget.json`
carries that reading as a recorded step against the base it steps from.

## The explaining walk resumes where the deciding one stopped

A keyed map that fails is walked twice: once to decide, once to say which field.
Both walks resolve the declared keys in the same order, and a probe is the dear
half — through the interpreter it is about 143 instructions, against a handful
for the check that follows. A fifty-field record refused at the thirty-second
position repeated thirty-one probes and thirty-one checks for nothing, because a
field the deciding walk **passed** has no violation to report.

So the deciding walk hands over where it stopped, and the explaining walk starts
there. What it hands over is two integers and no allocation: the position, and
how many declared keys had been found by then. The count travels because the
count is load-bearing — a record holding exactly the keys it declares skips the
scan for undeclared ones, and a walk that resumed without it would fall into a
scan that finds nothing.

Three things bound it.

The **entry count** guards the resumption. A field's schema can run Python — a
predicate, an `__eq__`, an `__instancecheck__` — and that can change the dict it
is being read out of, so a value whose size moved between the two walks is read
from the start. A value that changed without changing size is what the mutation
report is for: the explaining walk then finds nothing and says the value moved,
which is a truer answer than a violation about a value that has.

The **count must be exact**, not merely different. What the early return compares
is a sum, so a count too high by `n` skips the scan on a record carrying exactly
`n` undeclared keys and behaves on every other. The rows ask one extra key and
two for that reason.

And the resumption is **only ever a saving**. Starting over answers the same
thing more slowly, which is why a mutant that always starts over is excused in
`.cargo/mutants.toml` with the instruction count beside it rather than chased
with a test. The direction that can be wrong — resuming where the walk did not
stop, or with a count it did not have — is killable and is killed.

## A narrow object is covered where it lies

A parsed JSON object's keys that no field declares are covered by the default
clauses, and `json.loads` semantics say a repeated key means its **last** value.
Answering that for a whole object by collapsing it to a table of last values is
linear with a hash per key, and it allocates: the free-form section of a record
is written `dict[str, V]` and usually carries a handful of keys, so the table was
being built for objects of one and two entries.

Up to `SMALL_OBJECT` entries the same question is asked in place. An entry is the
one the document means exactly when no entry *after* it repeats the key, which is
a look forward over entries already in hand. The two readings answer alike by
construction, so the boundary between them may show in what a walk costs and
never in what it answers -- which is why the rows that hold it run either side of
the bound and across it, and why flipping the comparison is an equivalent mutant
the sweep excuses with that argument.

The key half of the question is settled before it is asked where a *lone* clause
is keyed by `str` or by anything: a parsed object's keys are strings by
construction, so such a clause admits every one of them and only its value schema
is walked. That is a smaller saving than the table -- one key-schema walk per
undeclared key rather than one allocation per object -- and the two compose.
Both are on the comparison gate's JSON document, which reads 0.78 of
pydantic-core with them and 0.87 with only the first.

The Python-dict path is untouched by both: a dict key is any object, and the
reason this reading holds is that a parsed key is not.

## Recursion is guarded by value identity

`check_ref` records `(object id, definition index)` on the path. A value that
contains itself fails with `recursion_loop` rather than looping, and a chain
deeper than the unfolding bound fails with `recursion_limit` rather than
descending.

Counting unfoldings is not counting frames, and the walk needs both. Every level
takes one native frame, and an unfolding descends the *whole definition body*, so
a body at the construction depth bound turns 128 unfoldings into thousands of
frames — more stack than a thread has. `Ctx::descend` therefore counts the levels
themselves and refuses past `MAX_WALK_DEPTH`, with the same `recursion_limit` the
unfolding bound gives, because it is the same fact about the value. The count is
a depth rather than a total because the level is released when the frame that
took it returns, so a wide value pays for its widest child and not for all of
them.

## The limit

**JSON has no tuple.** `json.loads` produces a list, so the JSON path has no
tuple arm and a tuple schema rejects an array whatever its elements are. It is
the one place the two input paths deliberately decide differently, and it is
pinned by a case.

**The walk is where an accept can be wrong, and a line floor does not see that.**
Its adequacy is measured by mutation, over a value corpus in the file itself
under the embedded-interpreter feature; [08-testing.md](08-testing.md) owns what
that measures and what it skips.
