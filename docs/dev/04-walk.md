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

**A branch whose failure no report reads is not explained.** A scalar kind, a
literal, and any branch whose kind refuses the value before its constraints or
contents are read fail at the union's own location with a mismatch, and such a
failure reaches a report only as the union's label. `decided_quietly` settles
those branches without explaining them, so a value the union admits builds no
violation and its `__repr__` never runs. Explained, `int | Foo` summarized a
`Foo` for the `int` branch before the `Foo` branch matched, and a repr that
raised made `validate` raise for a member. It holds each level the branch's own
walk would enter, so at the depth bound the branch is explained and reports the
bound, as it did.

**Each branch is walked in the caller's mode.** The choice reads the first
failure only, which a walk stopped there has measured, so under fail-fast a
branch is walked to its first failure and no further: the report costs a
fail-fast walk of each branch, not the size of the value it refuses. The full
report walks every branch whole and chooses the same branch, so the one
violation fail-fast reports is the one the full report leads with.
`explain_union` in `check/walk.rs` owns both. Choosing by the deepest failure
anywhere in a branch would need every branch walked whole in every mode, and a
fail-fast report would then cost the size of the value.

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
- value sets for unions whose members are all literals, with the addresses of
  the pooled constants behind them;
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
`crates/valgebra-py/src/check/index.rs` is the precompute the next section is
about: the record-field lookups, the literal-union tables and the compiled
patterns, built once with the validator and read by the walk that uses them.

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
of one scalar kind, or of a union of them, is read through.

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
on the interpreter.** CPython's `PyTuple_GET_SIZE` and `PyTuple_GET_ITEM` read
the storage whatever the type overrides, so there every tuple, subclass or not,
is read where it lies and its type is asked nothing. PyPy's `cpyext` goes
through the object's own `__len__`, so a subclass that *overrides* `__len__`
answers the C accessor with whatever it likes — and a walk that indexed
against that read past the end of the allocation and took the process down.
There such a value is copied through the base type's own slot and the copy is
walked.

On PyPy a subclass that **inherits** `tuple.__len__` is not copied, because
there is nothing to distrust: the overridden answer and the base's answer are
the same function. That is every `NamedTuple`, which is the tuple subclass a
program is most likely to hold, and the walk tells the two apart by asking the
type whether its `__len__` *is* the base's. Telling them apart by "is this
exactly a tuple" instead copied every `NamedTuple`. CPython asked the same
question, whose answer changes nothing it reads: the lookup was a quarter of
walking a list of `NamedTuple`s, 798 instructions an element against 581
without it.

**On PyPy a subclass's contents come through its own methods too.** `cpyext`
fills a tuple subclass's C-level items from its own `__iter__`, and answers a
dict subclass's length through its own `__len__` and each value `PyDict_Next`
yields through its own `__getitem__`. So there a tuple subclass is read where it
lies only if it inherits `tuple.__iter__` as well as `tuple.__len__`, and a dict
subclass only if it inherits `dict.__len__` and `dict.__getitem__`. Any other is
copied through `tuple.__iter__` or `dict.copy`, and the copy is walked. A tuple
subclass whose `__iter__` yielded one item over two took `validate` down, and a
dict subclass overriding `__len__` was refused by a closed record it belongs
to. One whose `__iter__` yields *more* than it stores never reaches the walk:
`cpyext` refuses it at the call. CPython reads the storage in every case, and
asks the type nothing. `scan_dict` reads its entries from a copy on PyPy
for a reason of the same kind: a key swapped for another while the scan runs
Python makes `cpyext`'s `PyDict_Next` fail fatally rather than report it, and
the copy is a dict nothing else can reach.

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

## A scalar is its type test wherever the walk asks one

The walk reaches a scalar through `member`, which takes a level, reads the
fatal-signal flag and dispatches, all around the one type test the schema is.
Four positions ask that question often enough for the frame around it to be
most of what they cost, and each asks the test directly:

- **A union's branch.** `int | str | None` tries its branches in order, and a
  scalar branch is its type test (`scalar_member` in `walk/scalar.rs`). A
  string checked against that union costs 18% fewer instructions, and the
  recursive walk of the binding gate 8.7% fewer.
- **A sequence whose element is a union of scalars.** `list[int | None]` is read
  the way `list[int]` is, above, with a test per branch
  (`homogeneous_scalar_union`): a thousand elements cost 87% fewer
  instructions, and a tuple of them 90%.
- **A mapping's clause.** `dict[str, int]` reads both halves of each entry as
  type tests in `covered`, and reads no key as a field name, since a mapping
  declares none: 37% fewer instructions on a thousand entries.
- **A tuple whose every position is a scalar.** `tuple[int, str, float]`, and a
  `NamedTuple` of builtin fields, is a type test a position
  (`scalar_positions_tuple_matches`): a thousand such tuples cost 37% fewer
  instructions.

The two readings agree because the fast walk answers a scalar as three
questions: whether a level is free under it, whether a fatal signal has been
recorded, and the type test. The direct reading asks the same three, so it
refuses at the bound and after a fatal signal exactly where `member` does, and
`a_scalar_is_answered_as_the_walk_answers_it` holds that for every scalar schema
against every kind of value. An explaining walk records the position of what it
refuses, so none of the three is taken there.

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

**Where a question is asked is part of what it costs.** The walk is one
recursive function under fat LTO, with the arms of `member` inlined into it, so
a test added to an arm moves the register allocation of every shape that
crosses the arm. Asked inline in `check_seq`, the union question cost a list
nested twenty-five deep 4%; a one-comparison shortcut for `dict[str, int]` in
the record walk cost the closed record 5% and the nested list 7%, though neither
reads a mapping. So the sequence readings of a union are out of line behind a
test of the tail's tag, which the nested list pays at 1.4%, and a mapping has no
reading beyond `covered`'s. The branch test is paid by every branch that is not
a scalar too, eight instructions each, which is 1% on a union of twenty record
kinds.

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

**The walk is where an accept can be wrong, and a line floor does not see that.**
Its adequacy is measured by mutation, over a value corpus in the file itself
under the embedded-interpreter feature; [08-testing.md](08-testing.md) owns what
that measures and what it skips.
