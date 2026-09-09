# The membership walk

`crates/valgebra-py/src/check/walk.rs` decides whether a value belongs to a
schema's set. It is where soundness is decided, and it is the file to read first
when an accept looks wrong.

## One walk, two input paths, three modes

There is **one** `member` function. It was two — a fast one and an explaining one
— and the two drifted; fusing them removed the class of defect where the answer
depended on which walk ran.

It runs over a `Value`, which is either a borrowed Python object or a borrowed
parsed JSON value. That is what keeps the object path and the in-place JSON path
membership-equivalent by construction rather than by a test that compares them.

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

## What is precomputed, and what correctness may not depend on

Three per-validator indexes are built once on first use and keyed by the address
of a node's own buffer:

- declared-field lookups per record;
- value sets for unions whose members are all literals;
- compiled patterns per regex source.

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

`walk/scalar.rs` answers what a value is **without descending into it**: a
scalar kind, a literal, and the constraints that narrow one. A container's
length is answered there too, for the same reason -- `MinLen` counts what a
value holds without reading any of it.

`walk/record.rs` reads a value as a **keyed map or an attribute record**: the
shape whose membership is a question per key rather than per position -- which
keys the value carries, which of them the schema declares, and what a key the
schema does not declare is covered by. `walk/sequence.rs` reads one as a run of
**elements** -- a list, a tuple, a parsed array, a set, a frozenset -- which
differ in how an element is reached and agree on what each must be, and which
share an arity, a count taken once and compared again, and the snapshot a list
of one scalar kind is read through.

The three entry points `walk.rs` calls into `sequence.rs` are marked `#[inline]`,
and the reason is measured rather than assumed: without it the *record* walk --
which reaches no sequence at all -- executes 3.1% more instructions, because
what the dispatcher can inline changes what fits around it. With it, that shape
reads 1.8% fewer than before the split.

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
**tuple** cannot be resized, so its arm keeps the plain iterator and pays
nothing; a JSON array is owned by the parser and cannot move at all.

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
in `crates/valgebra-py/src/check/walk.rs`: below the first it cannot pay for its
own allocation, above the second walking it costs more cache than the counts it
avoids, and the transient stops at two mebibytes. Both are in the bounds table of
[00-architecture.md](00-architecture.md), and neither changes an answer.

The contract of the section above is kept: the copy answers about the list as it
was when the copy was taken, so the count is compared again afterwards and a
value that moved reports the move. The **instruction** count moves the other way
-- the copy is instructions and the stall it removes is not, so a sixty-four
element walk executes 47% more of them -- which is why `scripts/perf_budget.json`
carries that reading as a recorded step against the base it steps from.

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
