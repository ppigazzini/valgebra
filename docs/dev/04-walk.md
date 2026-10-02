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
| `Fast` | membership only: no violation or path built, every composite short-circuits; storage a reading needs for itself is allocated where the reading says why |
| `Explain` | a violation for each independent failure — every field, element and entry |
| `ExplainFailFast` | the first violation only |

Three modes, and the type says three. A pair of independent booleans would admit
a fourth combination -- fail-fast without explaining -- that means nothing, and
the type leaves it unnameable. The
discriminants are ordered so both predicates the walk asks per node are a single
comparison, and a test pins both over every variant — that order is load-bearing
and measured, not cosmetic ([06-type-design.md](06-type-design.md)).

## The comparison-raises policy

Membership reads a value through Python operations that can raise: `__eq__` for a
literal, a rich comparison for a bound, `isinstance` for a class, `getattr` for
an attribute, `%` for a multiple-of (the operand's `__rmod__` where the value's
type does not know it), `__len__` for a length.

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

**Every site that catches a Python error asks the classifier before it answers.**
A membership probe folds through `fold`; a reading whose failure has a safe side
-- a type's slot, a key's field name, a copy of a value -- takes that side and
records a fatal signal through `record_if_fatal`. A key's `__eq__` or a
metaclass runs caller code exactly as a predicate does, and a signal raised
there is re-raised the same way.

**A message is such a reading.** Explaining a failure renders the value, the
constant a literal names, the bound or step a value missed, a key the path
names by its repr, and a class's name, and each of those runs caller code.
`summarize_in` and `class_label_in` in `check/violation.rs` are the walk's two
ways to render one: an ordinary exception reads as `<unrepresentable>`, and a
fatal signal is recorded for the entry point to raise. The `summarize` and
`class_label` in `errors.rs` are the same reading for the build and `repr`,
where no walk runs: they return the signal rather than record it.

Each disjunct needs its own test case: a mutation collapsing the classifier to
one of them is invisible to a corpus that only raises `KeyboardInterrupt`.

## The union reports the closest branch

When no branch of a union matches, dumping every branch's errors buries the one
the value was closest to. The walk instead reports the branch that **descended
furthest** into the value before failing, measured as the path depth, past the
union's own location, of the branch's **first** failure -- the first violation
its walk records -- with a tie keeping the earliest branch so the choice is
deterministic. When no branch makes progress — every branch a flat type mismatch
— it falls back to one union error.

**A branch the walk could not answer for keeps its own report.** A branch whose
first failure says the *walk* stopped -- `recursion_limit`, `recursion_loop`,
`mutated_during_validation` or `predicate_error`, the codes `walk_declined`
names -- fails at the union's own location too, so the progress rule would fold
it into the union error and drop the one sentence that says what to do about
it. Where no branch makes progress, the first such branch is reported in place
of the summary. The search reads at most `CLOSEST_BRANCH_PROBE_LIMIT` branches;
past it a branch is only decided, so a union that wide reports the closest of
the branches it read, or the summary.

**A branch whose failure no report reads is not explained.** A scalar kind, a
literal, and any branch whose kind refuses the value before its constraints or
contents are read fail at the union's own location with a mismatch, and such a
failure reaches a report only as the union's label. `decided_quietly` settles
those branches without explaining them, so a value the union admits builds no
violation and its `__repr__` never runs: explaining the `int` branch of
`int | Foo` would summarize a `Foo` the next branch admits, and a repr that
raised a fatal signal would make `validate` raise for a member. It holds each
level the branch's own walk would enter, so at the depth bound the branch is
explained and reports the bound, which a report does read.

A class is such a branch: its walk is one `isinstance`, which `decided_quietly`
asks in its place. So is a class met with the attributes it declares -- a
dataclass -- whose class refuses the value, because the meet stops at a member
that refuses the value itself; but only where the class's metaclass is `type`.
The meet's walk asks the class again where it admits, and `isinstance` against
a class `type` made reads nothing of a value it admits, where another
metaclass's `__instancecheck__` is code that would then run twice.

**A record branch is explained only once the union is refused.** A record's
walk is two passes already, one that decides and one that explains, resuming
where the first stopped, and its failure lies inside the value, where a report
does read it. `explain_union` asks each record branch's deciding pass in branch
order beside the other branches, and the explaining passes, in branch order,
only once no branch has admitted the value. A value the union admits builds no
report for a record branch before the one that matched, and a value it refuses
gets the report each branch explained in turn would give: the choice reads the
same branch reports in the same order. A fatal signal a later branch raises
stops every explaining pass that waits for it, as it stops every later walk.
`a_refused_union_reports_what_its_chosen_branch_reports_alone` in
`check/walk/interpreter.rs` holds the report to the chosen branch's own.

**Each branch is walked in the caller's mode.** The choice reads the first
failure only, which a walk stopped there has measured, so under fail-fast a
branch is walked to its first failure and no further: the report costs a
fail-fast walk of each branch, not the size of the value it refuses. The full
report walks every branch whole and chooses the same branch, so the one
violation fail-fast reports is the one the full report leads with.
`explain_union` in `check/walk.rs` owns both. Choosing by the deepest failure
anywhere in a branch would need every branch walked whole in every mode, and a
fail-fast report would then cost the size of the value.

A predicate is asked again wherever the walk reads a value twice, and the walk
keeps no memo over the answers. An explaining walk reads each branch of a union
once, since the walk that chooses the branch is the walk that reports it; what
it reads twice is a keyed map that fails, whose explaining pass re-reads the
field its deciding pass stopped at and everything beneath it, and walks a clause
once more against an undeclared key it does not cover, to report it. A keyed map
that is a union's branch takes the second read only where the union refuses the
value, since its explaining pass waits for that. A predicate
is user code, so each occurrence the walk reaches is a call, and a cache keyed
on the value's identity would decide for an impure predicate which of its
answers counts. The refinements page states the count a caller sees; a predicate
that must run once per value is memoised on the caller's side, where the key is
the caller's to choose.

## What is precomputed, and what correctness may not depend on

Four per-validator indexes -- `ValidatorIndex` in
`crates/valgebra-py/src/check/index.rs` -- are built once on first use and keyed
by the address of a node's own buffer:

- declared-field lookups and interned keys per record;
- interned attribute names per attribute record;
- value sets for unions whose members are all literals, with the addresses of
  the pooled constants behind them;
- compiled patterns per pattern constraint.

A thread that finds another building them waits **detached** from the
interpreter. The build interns strings, and on a free-threaded interpreter that
can wait on a lock and let a stop-the-world pause begin, which a waiter still
attached would never reach. Once built, the read is the same load either way.

**Correctness never depends on one being present.** A node absent from an index
falls back to building the map, scanning the branches, or recompiling the
pattern. The literal-union plan answers only for an exact int or str. The
deciding walk takes its answer either way; the explaining walk takes only its
*yes*, for the elements of a list of the union (`literal_list_matches`), and
walks every other element. A non-literal union, another value type and a JSON
value all fall through to the linear scan, which stays the one source of truth
for behaviour.

**The constant itself is found by its address.** A literal a program spells in
its own source is usually the object it validates: a string constant is
interned and a small integer cached. The pool holds each constant for the life
of the validator, so no other object has its address meanwhile, and an exact
`str`, `int`, `bool` or `bytes`, or `None`, is equal to itself. So the plan
answers an address it holds without decoding or hashing the text, and a single
literal answers its own constant without the type reads and the comparison
(`is_the_constant`): a thousand-element `list[Literal["a", "b", "c", "d"]]`
costs 37% fewer instructions. A float is not among the types -- the same `nan`
is not equal to itself -- and neither is a class with an `__eq__` of its own,
whose answer is its own and whose running is a call.

## Where the walk lives

`crates/valgebra-py/src/check/walk.rs` holds the dispatcher `member` and the
arms that read a value as a union, a meet, a complement, a class or a
reference. What the dispatcher descends into, and what it stops at, have a
module each, because what each shares with the rest is the dispatcher and
little else.

Two files beside them are read by every arm. `crates/valgebra-py/src/input.rs`
is the `Value` above — the two input paths and the decoders that produce them.
`crates/valgebra-py/src/check/index.rs` is the precompute the section above is
about: the record-field lookups, the attribute names, the literal-union tables
and the compiled patterns, built on the validator's first use (`Validator::index`
in `crates/valgebra-py/src/validator.rs`) and read by the walk that uses them.

`walk/scalar.rs` answers what a value is **without descending into it**: a
scalar kind, a literal, and the constraints that narrow one. A container's
length is answered there too, for the same reason -- `MinLen` counts what a
value holds without reading any of it -- and it is asked where a fixed shape
asks its arity: after the kind test the container's walk makes first, and before
any element. `Annotated[list[int], MinLen(1)]` and `[int, int, ...]` are one
set, so the two are refused at the same step, and a bound that fails ends the
walk of the value as a wrong length does. A constraint's operand is borrowed out
of the pool for as long as the walk runs, so a check that passes takes no
reference to it: on a free-threaded interpreter a reference is an atomic write
to a counter every thread sharing the validator touches.

`walk/record.rs` reads a value as a **keyed map or an attribute record**: the
shape whose membership is a question per key rather than per position -- which
keys the value carries, which of them the schema declares, and what a key the
schema does not declare is covered by. A keyed map is read one of three ways.
**By its keys** (`keyed_map_asks_for_its_keys`): one probe per declared field,
through the record's `RecordPlan` in the index, and a count of the entries
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
is read through where a test settles every element -- one scalar kind, a union
of them, one class, a union of literals.

All three read the same `Frame`: where the walk is in the value, what it has found
there, and the context it may look things up in. A walk needing a different one
-- a union explaining a branch into a buffer of its own, a clause pair deciding
on the fast path inside an explaining walk -- builds it from the parts it keeps.
A fast walk writes neither of a frame's buffers, so where the walk is fast
already it asks in the frame it holds: `check_union` walks each branch in the
union's frame, and a mapping's scan keeps one fast frame for every entry
`covered` asks of its clauses. Each buffer is a value with a destructor, and a
pair built and dropped per branch and per key is a tenth of what a recursive
schema's walk and a `dict[str, int]` cost (`--binding-recursive`,
`--binding-mapping`).

## A container is read against a count taken once

Membership runs arbitrary Python at almost every entry — a predicate, an
`__eq__`, an `isinstance` hook — and a free-threaded interpreter lets another
thread write to the container meanwhile. Every container the walk does not own
is therefore read against a count taken at entry. `scan_dict` over entries and
`scan_list` over positions take it themselves and re-read it before each step
and after the last; `scan_set` leaves it to the builtin set iterator, which
raises when the set changes size, and reads that raise as the move. A reading
that moved covers no state the value was ever in, so the scan answers
`Scan::Unreadable` and the caller reports `mutated_during_validation` rather
than answering from the part it saw.

The sequence case is the one that argues for all of them. A list is walked *by
position* against a length read once, so a list that grows hides its new items
from the walk and one that shrinks leaves the walk answering about items that
are gone — and in both directions a walk answering from that reading calls a
value a member that is not one. A **tuple** cannot be resized, so its arm keeps
the plain iterator; a JSON array is owned by the parser and cannot move at all.

**A tuple's length comes from the storage, and which reading gives it depends
on the interpreter.** CPython's `PyTuple_GET_SIZE` and `PyTuple_GET_ITEM` read
the storage whatever the type overrides, so there every tuple, subclass or not,
is read where it lies and its type is asked nothing. PyPy's `cpyext` goes
through the object's own `__len__`, so a subclass that *overrides* `__len__`
answers the C accessor with whatever it likes — and a walk indexing against
that answer reads past the end of the allocation and takes the process down.
There such a value is copied through the base type's own slot and the copy is
walked.

On PyPy a subclass that **inherits** `tuple.__len__` is not copied, because
there is nothing to distrust: the overridden answer and the base's answer are
the same function. That is every `NamedTuple`, which is the tuple subclass a
program is most likely to hold, and the walk tells the two apart by asking the
type whether its `__len__` *is* the base's -- not whether it is exactly a
tuple, which would copy every `NamedTuple`. CPython's tuple walk does not ask,
since the answer changes nothing it reads: the lookup is a quarter of walking a
list of `NamedTuple`s, 798 instructions an element against 581 without it. A
length bound asks it on every interpreter, because `stored_len` counts what the
value holds rather than what an override answers.

**On PyPy a subclass's contents come through its own methods too.** `cpyext`
fills a tuple subclass's C-level items from its own `__iter__`, and answers a
dict subclass's length through its own `__len__` and each value `PyDict_Next`
yields through its own `__getitem__`. So there a tuple subclass is read where it
lies only if it inherits `tuple.__iter__` as well as `tuple.__len__`, and a dict
subclass only if it inherits `dict.__len__` and `dict.__getitem__`. Any other is
copied through `tuple.__iter__` or `dict.copy`, and the copy is walked: read in
place, a tuple subclass whose `__iter__` yields one item over two takes
`validate` down, and a dict subclass overriding `__len__` is refused by a closed
record it belongs to. One whose `__iter__` yields *more* than it stores never
reaches the walk: `cpyext` refuses it at the call. CPython reads the storage in
every case, and asks the type nothing. `scan_dict` reads its entries from a copy
on PyPy for a reason of the same kind: a key swapped for another while the scan
runs Python makes `cpyext`'s `PyDict_Next` fail fatally rather than report it,
and the copy is a dict nothing else can reach.

**Every container the walk reads answers this way, and for one reason.** A
schema over a container denotes what the value *holds*, so the reading of it
cannot be a method the value chooses: a `str`, `bytes`, `list`, `tuple`, `set`,
`frozenset` or `dict` subclass that overrides `__len__` is counted through its
base type's slot, and a `set` or `frozenset` subclass that overrides `__iter__`
is walked through its base type's iterator. Believing an override admits a value
whose storage the schema excludes — `set[int]` would hold a subclass whose
storage carries a `str`, and `MinLen(3)` one character — which is an accept no
value supports. The exactness test comes first at every one of them, so an exact
container and an inheriting subclass keep the reading their storage already
gives.

The two slots are asked of the base type rather than through a C accessor for
the reason the tuple paragraph gives, and the iterator the base returns is the
builtin one, so a set that changes size during the scan still raises where the
scan expects it to.

The cost is one length read per element, which is a pointer dereference.

## A list a test settles is read through a snapshot of it

Reading an element out of a list hands back an *owned* handle: a reference count
written when the handle is made and again when it drops, on an object the walk
only type-tests. Copying the list into a tuple pays the same two counts inside
the interpreter, in two loops carrying no dependent work between them, and the
tuple is frozen -- so its elements are read borrowed and the walk pays neither.
On the free-threaded build the copy also takes the container's lock once rather
than once per element. The readers of a list whose every element one test
settles take that copy where it pays: one scalar kind and a union of them here,
one class and a union of literals below. What each reading costs per element,
in place and through the copy on each interpreter, is measured in the doc
comment on `snapshot_pays` in `crates/valgebra-py/src/check/walk/sequence.rs`,
the one place those figures live.

Which reading is cheaper is a property of the **interpreter**, so the walk asks
one. CPython 3.14 with its global lock makes the count pair cheap enough that
the copy is pure cost, and the walk reads in place there; the free-threaded
build pays a lock per element on top of the pair, and takes the copy on every
release. Asking requires the interpreter's own flags, which reach the crate that
emits them and no other, so `crates/valgebra-py/build.rs` re-emits them; without
it such a question reads "an older interpreter" against every interpreter,
silently, and the fast path is taken everywhere.

Two widths bound the copy, `SNAPSHOT_MIN_ELEMENTS` and `SNAPSHOT_MAX_ELEMENTS`
in `crates/valgebra-py/src/check/walk/sequence.rs`: below the first it cannot
pay for its own allocation under a global lock, above the second walking it
costs more cache than the counts it avoids, and the transient stops at two
mebibytes. The free-threaded build takes the copy at every width up to the
second, since the lock it saves per element outweighs the allocation. Both are
in the bounds table of [00-architecture.md](00-architecture.md), and neither
changes an answer.

The band is a rule about length, measured on distinct integers, and the walk
asks nothing about what a list holds before choosing. A list of objects the
interpreter keeps immortal -- `None`, a small integer -- pays the copy without
the counts it saves: `validate` on a thousand elements two thirds `None` reads
10% slower through the snapshot on 3.12 than in place. The explaining walk
reads a list that belongs the way the deciding walk does, band and all, so
`validate` and `is_valid` cost alike on the same list.

The contract of the section above is kept: the copy answers about the list as it
was when the copy was taken, so the count is compared again afterwards and a
value that moved reports the move. The **instruction** count moves the other way
-- the copy is instructions and the stall it removes is not, so a sixty-four
element walk executes 47% more of them. The walk's budget in
`scripts/perf_budget.json` is recorded on CPython 3.12 over the snapshot
reading, so the count it holds is the higher one.

## A scalar is its type test wherever the walk asks one

The walk reaches a scalar through `member`, which takes a level, reads the
fatal-signal flag and dispatches, all around the one type test the schema is.
These positions ask that question often enough for the frame around it to be
most of what they cost, and each asks the test directly:

- **A sequence of one scalar kind.** `list[int]` and `tuple[str, ...]` read the
  kind once for the sequence and test each element against it
  (`homogeneous_scalar`), the list through the snapshot above where one pays
  (`scalar_list_matches`). A list's loop is one per kind, its test a constant
  inside it, so no element pays the dispatch on the kind: with the dispatch in
  one shared loop, the PGO wheel laid it out an instruction an element dearer,
  which the instruction gate, building without a profile, does not see. The
  explaining walk's reader takes the same test per kind: with the kind matched
  at every element, `validate` on a `list[int]` read in place, as 3.14 and
  3.15 read one, costs a fifth more than `is_valid`.
- **A union's branch.** `int | str | None` tries its branches in order, and a
  scalar branch is its type test (`scalar_member` in `walk/scalar.rs`). A
  string checked against that union costs 18% fewer instructions, and the
  recursive walk of the binding gate 8.7% fewer.
- **A sequence whose element is a union of scalars.** `list[int | None]` is read
  the way `list[int]` is, above, with a test per branch
  (`homogeneous_scalar_union`): a thousand elements cost 87% fewer
  instructions, and a tuple of them 90%.
- **A set of one scalar kind.** `set[str]` and `frozenset[int]` read the kind
  once for the set, and each element is the kind's test, a scan per kind with
  the test a constant inside it, in the deciding walk (`elements_admitted` in
  `walk/sequence.rs`) and the explaining one (`explain_elements`), reading
  once for the scan whether the level every element sits at is free. The
  explaining scan's sort and report are one function beside the scans
  (`elements_reported`), so a kind adds a scan and no second copy of the sort.
  A set of a thousand strings costs 10% fewer instructions to accept that way,
  and 28% fewer to `validate` (`--binding-set`).
- **A parsed array.** A document's array of one scalar kind, or of a union of
  them, is one test an element (`json_array_matches`); a document is never
  explained, so the loop records nothing.
- **A mapping's clause.** `dict[str, int]` reads both halves of each entry as
  type tests in `covered`, and reads no key as a field name, since a mapping
  declares none: 37% fewer instructions on a thousand entries.
- **A tuple whose every position is a scalar.** `tuple[int, str, float]`, and a
  `NamedTuple` of builtin fields, is a type test a position
  (`scalar_positions_tuple_matches`): a thousand such tuples cost 37% fewer
  instructions.
- **A refinement's base.** `Annotated[int, Ge(0)]` asks its base before its
  constraints, and a scalar base that admits the value is its type test
  (`check_refine` in `walk/scalar.rs`); a base that refuses is walked, which
  records the mismatch an explaining walk reports. The constraints read the
  value the walk holds rather than a handle of their own, so a passing check
  writes no reference count on it: refined integers cost 8% fewer instructions
  on the binding gate (`--binding-refined`), and strings matched against a
  pattern 5% (`--binding-pattern`).
- **A record's field.** A field that is a scalar -- most of a `TypedDict` -- is
  its type test in the deciding walk (`field_holds`), read by keys, by the scan
  and in both JSON readings: the binding gate's fifty-field record costs 7%
  fewer instructions, 10% with interned keys. A dataclass's attribute is walked
  in the caller's mode, so it takes the explaining walk's quiet admit, below, in
  either mode: a list of dataclasses costs 8% fewer, and 19% under `validate`.

The two readings agree because the fast walk answers a scalar as three
questions: whether a level is free under it, whether a fatal signal has been
recorded, and the type test. The direct reading asks the same three, so it
refuses at the bound and after a fatal signal exactly where `member` does, and
`a_scalar_is_answered_as_the_walk_answers_it` holds that for every scalar schema
against every kind of value.

**What an explaining walk admits, it records nothing of.** `validate` explains
as it decides, in one walk, so it is the mode a value that belongs is read in.
A scalar records only a mismatch, and a union returns at the first branch that
matches, keeping nothing of the branches before it, so an element that passes
its test leaves nothing behind, and the sequence readings serve the explaining
walk too: each element is its test, and one that fails is walked at its own
location, which records what the walk records of it (`list_explained`,
`tuple_explained`). No element can raise before the one that fails -- a test
runs no Python, and only a failing element's summary can, which fails the
sequence first. A list that belongs is read through the deciding walk's
snapshot where one pays, since a snapshot every element passes is the whole
answer; a list holding an element that fails is read in place over the general
walk's count, so one that moves reports the move. Read in place, the list that
belongs costs `validate` twice what `is_valid` takes on 3.12, for fewer
instructions: an owned handle per element is a reference count written on a
different object each time. A parsed array is never explained, and its
readings refuse the mode.

`validate` on a thousand-element `list[int]` that belongs costs 72% fewer
instructions than it does with every element taken through the explaining
walk's dispatch and location push, on a `list[int | None]` 76%, and on a list of
`tuple[int, str, float]` 22%. Where no reading applies, an element or a set
member that `admitted_quietly` answers -- a scalar whose test passes, a union
of scalars one of whose tests does, at free levels and with no fatal signal
recorded -- is passed without the push and the dispatch: `validate` on a
`set[str]` costs 26% fewer instructions that way.

The level is read and not held. An element sits one level below its sequence and
a branch one below its union, so a sequence of a union of scalars needs two free
levels, and `homogeneous_scalar_union` holds the union's while it asks for the
branch's -- what the walk does with each element -- and declines to the general
path wherever either is missing, which then refuses at the bound.

**A class is its type pointer for its own instances.** `isinstance(obj, C)`
with `type(obj) is C` answers yes before it asks `C`'s `__instancecheck__` --
CPython's `PyObject_IsInstance` makes that test first, from 3.10 to the current
branch -- so `check_instance` reads it off the type pointer (`is_exactly_a`),
and asks `isinstance` only of a subclass instance or another value. A list of
`date`, or of one enumeration, costs 29% fewer instructions. PyPy implements
`isinstance` otherwise, and there the call answers.

**A list of one class, or of a union of literals, has a reader of its own.**
Each element is the pointer test, or the union's table found once for the list
(`instance_list_matches`, `literal_list_matches`), and an element the test does
not settle -- a subclass instance, a value the table does not decide -- is
walked, which may run Python; so a list is read in place, unless a snapshot the
test admits entirely settles it. The general loop pays a call, a dispatch and,
for the union, a lookup of its table at every element, so the readers cost a
thousand `date`s 76% fewer instructions and a thousand literals 62%. The list
arm hands both kinds of tail to one reader (`element_list_matches`) behind its
one tag test, and the readers cost the PGO wheel's walk of a nested list 3% more
instructions with the training workload reading lists of each kind, 6% without
it.

**Where a question is asked is part of what it costs.** The walk is one
recursive function under fat LTO, with the arms of `member` inlined into it, so
a test added to an arm moves the register allocation of every shape that crosses
the arm. Asked inline in `check_seq`, the union question costs a list nested
twenty-five deep 4%; a one-comparison shortcut for `dict[str, int]` in the
record walk costs the closed record 5% and the nested list 7%, though neither
reads a mapping. So the sequence readings of a union are out of line behind a
test of the tail's tag, which the nested list pays at 1.4%, and a mapping has no
reading beyond `covered`'s. The branch test is paid by every branch that is not
a scalar too, eight instructions each, which is 1% on a union of twenty record
kinds.

## The explaining walk resumes where the deciding one stopped

A keyed map that fails is walked twice: once to decide, once to say which field.
Both walks resolve the declared keys in the same order, and a probe is the dear
half — through the interpreter it is about 143 instructions, against a handful
for the check that follows. Starting over, a fifty-field record refused at the
thirty-second position repeats thirty-one probes and thirty-one checks for
nothing, because a field the deciding walk **passed** has no violation to report.

So the deciding walk hands over where it stopped, and the explaining walk starts
there. What it hands over is three integers and no allocation (`Decided` in
`walk/record.rs`): the entry count the dict held, the position, and
how many declared keys had been found by then. The found count travels because
it is load-bearing — a record holding exactly the keys it declares skips the
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
is written `dict[str, V]` and usually carries a handful of keys, so the table
would be built for objects of one and two entries.

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

**A stack probe is refused.** Levels are not bytes, so a thread started with
less stack than its platform's default can overflow before the bound does, and
[the limits page](../10-limits.md) says so. Reading the stack itself --
`stacker::remaining_stack()` beside `descend` -- would turn that overflow into
`recursion_limit`, and it is not in: it adds a crate that assembles for every
wheel target, a call on every level of every walk, and a reading of each
platform's thread-stack bounds that no lane measures, musl, Windows arm64 and
PyPy among them. A proposal starts from the test it must pass: a value at the
depth bound on a 128 KiB thread raises `recursion_limit` on every wheel
platform, and the walk shapes stay inside the perf gate's band.

## The limit

**JSON has no tuple.** `json.loads` produces a list, so the JSON path has no
tuple arm and a tuple schema rejects an array whatever its elements are. It is
the one place the two input paths deliberately decide differently, and it is
pinned by a case.

**The walk is where an accept can be wrong, and a line floor does not see
that.** Its adequacy is measured by mutation, over the value corpus in
`crates/valgebra-py/src/check/walk/interpreter.rs`, compiled under the
`interpreter-tests` feature that embeds the interpreter;
[08-testing.md](08-testing.md) owns what that measures and what it skips.
