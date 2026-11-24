//! The read-only context a membership walk carries, and the mode it runs in.
//!
//! A leaf module under `check`: both types are defined here rather than in the
//! parent, because a type defined in an aggregator and imported back by its
//! members is what puts the two in a cycle. `check.rs` re-exports them, so no
//! caller spells a new path.

use std::cell::{Cell, RefCell};
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;

use pyo3::prelude::*;
#[cfg(PyPy)]
use pyo3::sync::PyOnceLock;
use valgebra_core::Schema;

use super::index::ValidatorIndex;
use super::memo::{Kept, Memo};
use crate::codes::Code;

/// The read-only context threaded through a validation walk: the constants pool,
/// the recursion definitions, the validator's precomputed index, the call's own
/// [`WalkState`], and the walk mode.
///
/// Seven words: the pool and the definitions, and one reference each to the
/// index's five tables and to the state's four cells. A table or a cell lies at
/// a fixed offset of what is referred to, so reaching one costs the address and
/// a load, which is what a reference of its own would cost, and a level that
/// hands the context on by value copies seven words rather than a reference per
/// table and per cell.
///
/// One reference to a structure holding all of it would be two words, carried
/// in two registers and never copied, and is refused: every read through a
/// frame takes a load more that way, and the call boundary reads 8.7%
/// dearer and refined integers 4.1% (`scripts/perf_gate.py --binding-boundary`,
/// `--binding-refined`).
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub(crate) pool: &'a [Py<PyAny>],
    pub(crate) defs: &'a [Schema],
    /// The plans built once per validator ([`ValidatorIndex`]). A node absent
    /// from them falls back to its general reading, so an answer never depends
    /// on the index being complete.
    pub(crate) index: &'a ValidatorIndex,
    /// What this membership test has found so far: the recursion guard, the
    /// open levels, the first fatal signal, and the stops ([`WalkState`]).
    pub(crate) state: &'a WalkState,
    /// What the walk is for. Constant for a whole walk, so the fast path pays a
    /// test of it per node, which goes the same way every time, and none of
    /// the explain bookkeeping.
    pub(crate) mode: WalkMode,
}

/// The most levels of recursive descent allowed before a value is rejected.
///
/// Only a value nested this deep under a recursive schema reaches it; the bound
/// exists so a pathologically deep value fails with `recursion_limit` instead of
/// overflowing the native stack.
/// It is the height [`Trail`] refuses to grow past, which is where a walk turns
/// a deep value into an answer rather than a crash.
pub(crate) const MAX_RECURSION_DEPTH: usize = 128;

/// The `(object id, definition index)` pairs a walk is inside, innermost last.
///
/// A level enters its pair before walking the definition and leaves it after,
/// so the trail is a stack rather than a set with holes: the pair a level
/// removes is the pair that level added, and the height is the recursion depth.
/// Entering a pair the trail already holds is a value that contains itself, and
/// that is the whole of the cycle test.
///
/// A `Vec` rather than an array of [`MAX_RECURSION_DEPTH`]: the bound is what a
/// pathological value reaches, not what a walk holds, and an array of it would
/// be written on every membership test including the ones that enter no
/// reference at all. An empty `Vec` allocates for a recursive schema alone, and
/// scanning a trail no deeper than the bound costs less than hashing a pair.
#[derive(Default)]
pub(crate) struct Trail(Vec<(usize, usize)>);

/// How many levels the trail holds before its first growth.
///
/// A `Vec` grown from empty by pushes allocates at four pairs and reallocates
/// at five and nine, so a value nested nine deep paid three trips to the
/// allocator on every membership test -- a sixth of the recursive shape's
/// count. Sixteen pairs is one 256-byte request, inside the size the
/// allocator serves from its per-thread cache, and a schema with no reference
/// never makes it.
pub(crate) const FIRST_TRAIL: usize = 16;

/// What entering a reference at a value found.
pub(crate) enum Entered {
    /// The level is open, and [`Trail::leave`] closes it.
    Open,
    /// The pair is already on the trail, so the value contains itself.
    Cycle,
    /// The trail stands at [`MAX_RECURSION_DEPTH`], so no level was opened.
    Full,
}

impl Trail {
    /// Open a level for this pair, or say why none was opened.
    ///
    /// Nothing is pushed unless the answer is [`Entered::Open`], so a caller
    /// leaves exactly the levels it entered and the two refusals need no
    /// unwinding of their own.
    pub(crate) fn enter(&mut self, key: (usize, usize)) -> Entered {
        if self.0.contains(&key) {
            return Entered::Cycle;
        }
        if self.0.len() >= MAX_RECURSION_DEPTH {
            return Entered::Full;
        }
        if self.0.capacity() == 0 {
            self.0.reserve_exact(FIRST_TRAIL);
        }
        self.0.push(key);
        Entered::Open
    }

    /// How many levels are open: the definitions unfolded on the current path.
    pub(crate) fn height(&self) -> usize {
        self.0.len()
    }

    /// Close the level [`enter`](Self::enter) opened.
    pub(crate) fn leave(&mut self) {
        let left = self.0.pop();
        debug_assert!(left.is_some(), "a level closes a pair that was entered");
    }
}

/// What a walk has met that is not an answer, read by every arm through one
/// reference.
#[derive(Default)]
pub(crate) struct Signals {
    /// A `Cell` mirror of whether the state holds a fatal signal yet, set
    /// where [`WalkState::keep_fatal`] keeps one. The per-node short-circuit
    /// reads this with a plain load instead of taking a `RefCell` borrow on
    /// every membership step.
    pub(crate) fatal_seen: Cell<bool>,
    /// The code of the first walk that stopped rather than decided: one past
    /// the depth bound, at a value inside itself, at a value that moved while
    /// it was read, or at a predicate that raised. Recorded in every mode,
    /// since a complement decides its inner schema on the fast path, and read
    /// there: a walk that stopped has not said the value is outside a set, so
    /// the set's complement cannot say it is inside.
    pub(crate) stopped: Cell<Option<Code>>,
}

/// The mutable state one membership test carries: the recursion guard, the
/// first fatal signal, the signals, and the count of open walk levels.
///
/// One owner rather than a local per cell at each entry point. They share a
/// lifetime — one call — and the walk reads them through one reference in
/// [`Ctx`], so a caller that assembles three of them and forgets the fourth is
/// a caller the type system should not be able to spell.
pub(crate) struct WalkState {
    /// `(object id, definition index)` pairs open on the current path, so a value
    /// that contains itself fails with `recursion_loop` instead of looping.
    pub(crate) guard: RefCell<Trail>,
    /// What a walk rarely holds ([`Rare`]): the first fatal signal, and the
    /// memo's tables. Boxed, so a walk that holds neither writes one word and
    /// tests one at its end.
    rare: RareSlot,
    /// What the walk has met that is not an answer: whether a fatal signal is
    /// recorded, and the first stop.
    pub(crate) signals: Signals,
    /// How many walk levels are open below the entry point. Each level is a
    /// native stack frame, and [`MAX_WALK_DEPTH`] is the ceiling
    /// [`descend`](Ctx::descend) holds it under.
    pub(crate) depth: Cell<usize>,
    /// Where the walk's memo stands: the definitions entered, and whether a
    /// node has armed it ([`Memo`]). Its tables are in `rare`.
    pub(crate) memo: Memo,
    /// The stack address below which the walk asks `PyPy` again
    /// ([`the_interpreter_has_room`]).
    #[cfg(PyPy)]
    room_at: Cell<usize>,
    /// The lowest frame `PyPy` has answered from with room, `usize::MAX` before
    /// the first.
    #[cfg(PyPy)]
    answered: Cell<usize>,
    /// The thread's stack limit, the low end of the reserve a predicate is
    /// called with, read from the thread's at the first predicate this walk
    /// calls, and `usize::MAX` before ([`WalkState::room_for_python`]).
    #[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
    call_out_limit: Cell<usize>,
}

/// The pointer [`WalkState`] holds what a walk rarely holds behind, released
/// by hand.
///
/// The compiler's drop of a box this size is a call out of line, and inlined
/// into the state's own drop it would make that a call too: fifteen
/// instructions a call, where the state's drop is otherwise a few written where
/// it is dropped.
/// Held in `ManuallyDrop`, the box has no drop code, and the `Drop` below tests
/// the pointer and releases a box out of line only where there is one, as
/// `Unwritten` holds its buffers in `walk.rs` for the same reason.
#[derive(Default)]
struct RareSlot(ManuallyDrop<RefCell<Option<Box<Rare>>>>);

impl Drop for RareSlot {
    #[inline]
    fn drop(&mut self) {
        if self.0.get_mut().is_some() {
            self.release();
        }
    }
}

impl RareSlot {
    /// Drop the box, out of line: only a walk that met a fatal signal or kept
    /// a definition has one.
    #[cold]
    #[inline(never)]
    fn release(&mut self) {
        drop(self.0.get_mut().take());
    }
}

/// What a walk rarely holds, behind one pointer in [`WalkState`].
///
/// Each is on a path a walk seldom takes, and each is a resource the state
/// must release. Held inline, the memo's tables are eleven words to write and
/// three drops to run on every call, which the instruction gate's
/// call-boundary shape, about a hundred instructions a call, reads as 43%
/// more; behind the one pointer they share with the signal's slot, a call that
/// holds neither writes and tests that one word.
#[derive(Default)]
struct Rare {
    /// A fatal interpreter signal raised mid-walk — a base exception that is not
    /// an ordinary exception (`KeyboardInterrupt`, `SystemExit`, `GeneratorExit`),
    /// or a `MemoryError`/`RecursionError`. The first such error is recorded here;
    /// the walk then short-circuits and the entry point re-raises it instead of
    /// silently reporting a non-member. An ordinary exception during a membership
    /// probe stays folded to non-membership and never lands here.
    fatal: Option<PyErr>,
    /// What the walk keeps of a definition it walked at a value.
    kept: Kept,
    /// How many times a container moved under the walk. A move ends the scan
    /// that met it as a refusal does, and is no answer, so a complement whose
    /// inner walk saw one refuses the value whatever the rest decided: it
    /// reads the count before and after. Here rather than among the signals,
    /// because a move is as rare as a fatal signal and a walk that meets none
    /// should not write it.
    moves: u32,
    /// How many failures a limited walk reports ([`WalkState::limit`]), and
    /// `None` in every other walk. Here rather than in the state, because a
    /// word the state holds is a word every call writes: held there, the count
    /// cost the walk of a list of 64 integers an instruction a call, and read
    /// through the walk's context it held that context live across a list's
    /// scan, a spilled register and a larger frame, three more.
    limit: Option<NonZeroUsize>,
}

impl WalkState {
    pub(crate) fn new() -> Self {
        Self {
            guard: RefCell::new(Trail::default()),
            rare: RareSlot::default(),
            signals: Signals::default(),
            depth: Cell::new(0),
            memo: Memo::default(),
            #[cfg(PyPy)]
            room_at: Cell::new(stack_address().saturating_sub(ROOM_STEP)),
            #[cfg(PyPy)]
            answered: Cell::new(usize::MAX),
            #[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
            call_out_limit: Cell::new(usize::MAX),
        }
    }

    /// Whether Python called from the walk has room on the native stack, or
    /// the `RecursionError` that ends the chain where it has not
    /// ([`CALL_OUT_RESERVE`]).
    ///
    /// The limit is the walk's after its first predicate, so a predicate after
    /// it reads a local's address against a field: the address is a `lea`,
    /// and the reserve one subtraction and one comparison
    /// ([`short_of_the_reserve`]). Asked of the thread every time --
    /// `stacker::remaining_stack()`, a lazily built thread-local behind a call
    /// to an assembled routine -- it costs more than the field, as does the
    /// limit in a thread-local of its own. A
    /// shortcut for an address above the reserve is not in: answering as the
    /// reading after it does, it is a branch no test can tell from its
    /// absence.
    #[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
    #[inline]
    pub(crate) fn room_for_python(&self) -> PyResult<()> {
        let here = stack_address();
        let mut limit = self.call_out_limit.get();
        if limit == usize::MAX {
            limit = thread_limit(here);
            self.call_out_limit.set(limit);
        }
        match short_of_the_reserve(here, limit) {
            None => Ok(()),
            Some(left) => Err(stack_overflow(left)),
        }
    }

    /// Record `err` as the walk's fatal signal, unless one is recorded already:
    /// the first is the one raised. Mirrored into the cheap flag the per-node
    /// short-circuit reads.
    pub(crate) fn keep_fatal(&self, err: PyErr) {
        {
            let mut rare = self.rare.0.borrow_mut();
            let rare = rare.get_or_insert_with(Box::default);
            if rare.fatal.is_none() {
                rare.fatal = Some(err);
            }
        }
        self.signals.fatal_seen.set(true);
    }

    /// Take the fatal signal the walk recorded, leaving none: what the entry
    /// point re-raises.
    ///
    /// By reference, at an entry point as inside a walk: the walk's context
    /// holds the state's address, so a method taking the state by value is a
    /// move the compiler cannot elide, every word of it copied at the end of
    /// every membership test. A walk that met no signal pays the flag tested
    /// first and nothing else.
    pub(crate) fn take_fatal(&self) -> Option<PyErr> {
        if !self.signals.fatal_seen.get() {
            return None;
        }
        self.rare
            .0
            .borrow_mut()
            .as_mut()
            .and_then(|rare| rare.fatal.take())
    }

    /// What `read` finds in the memo's tables, where a walk has kept anything.
    pub(crate) fn recall<R>(&self, read: impl FnOnce(&Kept) -> Option<R>) -> Option<R> {
        self.rare
            .0
            .borrow()
            .as_ref()
            .and_then(|rare| read(&rare.kept))
    }

    /// Write to the memo's tables, building them on the first write.
    pub(crate) fn keep(&self, write: impl FnOnce(&mut Kept)) {
        let mut rare = self.rare.0.borrow_mut();
        write(&mut rare.get_or_insert_with(Box::default).kept);
    }

    /// Count a container that moved under the walk ([`Rare`]'s `moves`).
    pub(crate) fn note_move(&self) {
        let mut rare = self.rare.0.borrow_mut();
        let rare = rare.get_or_insert_with(Box::default);
        rare.moves = rare.moves.saturating_add(1);
    }

    /// How many containers have moved under the walk so far.
    pub(crate) fn moves(&self) -> u32 {
        self.rare.0.borrow().as_ref().map_or(0, |rare| rare.moves)
    }

    /// Report at most `count` failures: the count a limited walk
    /// ([`WalkMode::ExplainLimited`]) stops at. Only such a walk sets it, once,
    /// before it starts, so the slot it allocates is one a call to it pays.
    pub(crate) fn limit_to(&mut self, count: NonZeroUsize) {
        self.rare.0.get_mut().get_or_insert_with(Box::default).limit = Some(count);
    }

    /// How many failures the walk reports, `usize::MAX` for every one. A
    /// composite stops once its report holds this many, each report a prefix
    /// of the full one, and the entry point keeps that many.
    pub(crate) fn limit(&self) -> usize {
        self.rare
            .0
            .borrow()
            .as_ref()
            .and_then(|rare| rare.limit)
            .map_or(usize::MAX, NonZeroUsize::get)
    }
}

impl Default for WalkState {
    fn default() -> Self {
        Self::new()
    }
}

/// The most walk levels one membership test holds open at once.
///
/// Every level of the walk is a native stack frame, and the frames a value can
/// demand are not bounded by either published limit on its own: a recursive
/// definition unfolds once per level of the *value*, and every unfolding
/// descends the whole body, so the frames are the product of the unfolding bound
/// and the definition's depth. This bounds that product directly, which is what
/// makes "a value never overflows the native stack" a statement about the walk
/// rather than about the values a caller happens to pass.
///
/// The figure is three levels for every unfolding the trail allows. The
/// smallest recursive body -- a reference, the union beside its base case, the
/// container around the back edge, `int | list[T]` -- opens three levels an
/// unfolding, so it is walked to the unfolding bound: every value
/// [`MAX_RECURSION_DEPTH`] admits is walked, and the next level is refused by
/// one bound or the other. A deeper body meets this bound first, which is what
/// it is for.
///
/// What it costs is measured on the wheel that ships. The profile-guided build
/// inlines the walk's arms into frames of 2.5 to 3 KiB a level on the dict and
/// union shapes, against 0.6 KiB on a plain release build, and the dearest
/// walk is the one explaining a refused value: at 384 levels, with the walks
/// of `test_the_deepest_walk_fits_the_documented_stack`, the `CPython` 3.12
/// wheel fits in 844 KiB and overflows 828, under the 1 MiB that is the least
/// stack any thread `CPython` creates has. 512 levels needed 1.5 MiB, and 255
/// would have refused `int | list[T]` from 85 unfoldings. An unoptimized build
/// spends more again and fits no such figure, which is a build nothing ships.
/// A thread given less than 1 MiB can overflow first; `docs/guides/limits.md` says
/// so.
pub(crate) const MAX_WALK_DEPTH: usize = 3 * MAX_RECURSION_DEPTH;

/// One open level of walk descent, given out by [`Ctx::descend`].
///
/// The level is closed when this is dropped, which is what makes the counter a
/// *depth* rather than a total: a walk over a wide value takes and returns one
/// level per child, and only nesting accumulates. Every early return in the walk
/// closes the level for the same reason it releases any other guard.
pub(crate) struct Descent<'a>(&'a Cell<usize>);

impl Drop for Descent<'_> {
    fn drop(&mut self) {
        let level = self.0.get();
        debug_assert!(level > 0, "a descent closes a level that was opened");
        self.0.set(level - 1);
    }
}

impl<'a> Ctx<'a> {
    /// Whether one more level is available, without taking it.
    ///
    /// The question a *leaf* loop asks. A scalar element cannot descend, so the
    /// level it would sit at need not be held while the loop runs -- only the
    /// refusal has to match, and [`descend`](Self::descend) refuses exactly
    /// where this answers `false`. Holding the level instead costs the loop's
    /// container a counter pair and a guard drop, which on a shape whose whole
    /// check is a type test per element is a measurable share of it.
    pub(crate) fn room_to_descend(self) -> bool {
        self.state.depth.get() < MAX_WALK_DEPTH && the_interpreter_has_room(self.state)
    }

    /// Open one level of descent, or refuse when the walk already holds
    /// [`MAX_WALK_DEPTH`] of them.
    ///
    /// The caller turns a refusal into a non-member with a `recursion_limit`
    /// violation — the same answer an over-deep value gets from the unfolding
    /// bound, because it is the same fact about the value.
    pub(crate) fn descend(self) -> Option<Descent<'a>> {
        let depth = &self.state.depth;
        let level = depth.get() + 1;
        if level > MAX_WALK_DEPTH || !the_interpreter_has_room(self.state) {
            return None;
        }
        depth.set(level);
        Some(Descent(depth))
    }
}

/// Whether `PyPy` leaves the walk the native stack one more level needs.
///
/// [`MAX_WALK_DEPTH`] is measured against the stack a thread has, which on
/// `CPython` no setting moves. `PyPy` stops its C-API calls short of that: it
/// sizes the stack `cpyext` may use by the recursion limit, and a call past
/// that fails -- fetching or normalizing the error it raised among them, which
/// `cpyext` takes for a fault of its own and ends the process on. Under
/// `sys.setrecursionlimit(60)` a list nested fifty deep did so. So the walk
/// asks `PyPy`'s own check, `__pypy__.stack_almost_full`, and goes on where
/// [`ROOM_RESERVE`] bytes below it are known to be free, for the Python a level
/// runs -- a predicate, an `isinstance` hook, the explaining walk's rendering
/// -- and for raising what that Python raises. A refusal is the walk's own, a
/// non-member at `recursion_limit`, the answer the bound gives, since it is the
/// same fact about the value. A check that cannot be asked is read as no room.
///
/// **A yes is worth a known number of bytes.** The check says yes while less
/// than fifteen sixteenths of the stack is in use, and a recursion limit of `N`
/// reserves `N / 1000` of 768 KB (`PyPy`'s docstrings of the two), so a yes
/// leaves a sixteenth of that below the frame it was asked from: 48 KB at the
/// default limit, 1.8 KB at the limits page's floor of 37. The walk spends what
/// the yes left before it asks again, and the frames of one level, which no
/// constant bounds, come out of the reserve. Where a sixteenth is less than
/// the reserve, the reserve is proven by questions asked below the walk, each
/// half a sixteenth past the last answered frame ([`asked_at`]). No question is
/// asked further than that from an answered frame: asked 12 KiB below a frame
/// nothing had answered for, or a step and a level's frames past the last, the
/// question's own call ran out of stack and ended the process.
#[cfg(PyPy)]
#[inline]
fn the_interpreter_has_room(state: &WalkState) -> bool {
    let here = stack_address();
    here > state.room_at.get() || asked_for_room(state, here)
}

/// Whether [`ROOM_RESERVE`] below `here` is free, asking `PyPy` as far down as
/// that takes, and where the walk asks next. Out of line: a walk asks once
/// what the last yes left is spent.
#[cfg(PyPy)]
#[cold]
#[inline(never)]
fn asked_for_room(state: &WalkState, here: usize) -> bool {
    let Some(left) = left_by_a_yes() else {
        return false;
    };
    let mut answered = state.answered.get();
    loop {
        // A quarter of what a yes left is kept for the next question's call.
        let spent_at = answered
            .saturating_sub(left - left / 4)
            .saturating_add(ROOM_RESERVE);
        if answered != usize::MAX && spent_at < here {
            state.answered.set(answered);
            state.room_at.set(spent_at);
            return true;
        }
        let point = answered.saturating_sub(left / 2).min(here);
        if !asked_at(point) {
            return false;
        }
        answered = point;
    }
}

/// How many bytes a yes leaves below the frame it was asked from: a sixteenth
/// of the stack the recursion limit reserves, or `None` where the limit cannot
/// be read. Read at each question, since a caller may move the limit.
#[cfg(PyPy)]
fn left_by_a_yes() -> Option<usize> {
    Python::attach(|py| {
        limit_of(py)
            .and_then(PyAnyMethods::call0)
            .and_then(|limit| limit.extract::<usize>())
            .ok()
            .map(|limit| limit.saturating_mul(LEFT_PER_LIMIT))
    })
}

/// `PyPy`'s answer asked from the first frame at or below `target`: each
/// frame spends [`ROOM_GRAIN`] bytes, never written, and asks from the next
/// one down until it stands there, a grain past it at most.
#[cfg(PyPy)]
#[inline(never)]
fn asked_at(target: usize) -> bool {
    let grain = [const { std::mem::MaybeUninit::<u8>::uninit() }; ROOM_GRAIN];
    std::hint::black_box(&grain);
    let answer = if stack_address() > target {
        asked_at(target)
    } else {
        the_stack_has_room()
    };
    // Read after the call, so the frame is not reused for it.
    std::hint::black_box(&grain);
    answer
}

/// Whether `PyPy`'s check says its stack is not almost full.
#[cfg(PyPy)]
#[inline(never)]
fn the_stack_has_room() -> bool {
    Python::attach(|py| {
        almost_full(py)
            .and_then(PyAnyMethods::call0)
            .and_then(|full| full.is_truthy())
            .is_ok_and(|full| !full)
    })
}

/// `PyPy`'s stack check, imported once.
#[cfg(PyPy)]
fn almost_full(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static ALMOST_FULL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    ALMOST_FULL.import(py, "__pypy__", "stack_almost_full")
}

/// The recursion limit's reader, imported once.
#[cfg(PyPy)]
fn limit_of(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static LIMIT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    LIMIT.import(py, "sys", "getrecursionlimit")
}

/// Import what a question calls when the module loads: an import is the
/// deepest call a question makes, and a walk's first is made wherever the walk
/// stands.
#[cfg(PyPy)]
pub(crate) fn import_the_questions(py: Python<'_>) -> PyResult<()> {
    almost_full(py)?;
    limit_of(py)?;
    Ok(())
}

/// How far below its entry a walk goes before its first question, so a walk of
/// a level or two asks nothing.
#[cfg(PyPy)]
const ROOM_STEP: usize = 512;

/// The bytes of a sixteenth of the stack for each unit of the recursion limit:
/// `PyPy` reserves a thousandth of 768 KB for each.
#[cfg(PyPy)]
const LEFT_PER_LIMIT: usize = 49;

/// The stack each frame [`asked_at`] descends through spends: the most a
/// question lands past its point.
#[cfg(PyPy)]
const ROOM_GRAIN: usize = 128;

/// The stack the walk keeps free below itself on `PyPy`: room for the frames
/// of a level, for the Python a level runs and for raising what it raises.
#[cfg(PyPy)]
const ROOM_RESERVE: usize = 12_288;

/// This thread's stack limit, the low end of the [`CALL_OUT_RESERVE`] a
/// predicate is called with, or [`NO_LIMIT`] where the limit cannot be read,
/// which leaves every address on a stack room.
///
/// On `CPython` before 3.14 a predicate that calls a validator re-enters the
/// walk with no frame between the two that the interpreter checks against the
/// stack: 3.12 counts each such call against a C-recursion limit of thousands
/// and spends about 3 KiB of native stack on it, so, unchecked, a predicate
/// calling its own validator ends 3.12.16 in a segmentation fault on every
/// thread of 1 MiB or less under the default recursion limit, and on the main
/// thread under a limit of 10,000. So the walk reads the thread's own bounds before it calls a
/// predicate, and with less than [`CALL_OUT_RESERVE`] left raises what 3.14 and
/// later raise at the Python frame between two re-entries, where they read the
/// same bounds and need no question here. The bounds are read once a thread
/// through `stacker` -- `pthread_getattr_np`, `pthread_get_stackaddr_np`,
/// `GetCurrentThreadStackLimits` -- since reading them needs `unsafe`, and not
/// on musl, whose `pthread_getattr_np` gives the main thread the stack mapped
/// so far rather than the stack it may grow to.
///
/// At the predicate rather than at every entry: asked where a walk is entered,
/// the question reads the call boundary 31.73% dearer on 3.12
/// (`scripts/perf_gate.py --binding-boundary`), and a predicate is a call into
/// Python many times its price. `PyPy`
/// stops such a chain in a check of its own, which ends the process
/// (`docs/explanation/boundaries.md`).
#[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
fn thread_limit(here: usize) -> usize {
    thread_local! {
        static LIMIT: Cell<usize> = const { Cell::new(usize::MAX) };
    }
    let mut limit = LIMIT.get();
    if limit == usize::MAX {
        limit = stacker::remaining_stack()
            .and_then(|left| here.checked_sub(left))
            .unwrap_or(NO_LIMIT);
        LIMIT.set(limit);
    }
    limit
}

/// The limit a thread whose limit is not read is given: the reserve above it
/// is the top [`CALL_OUT_RESERVE`] bytes of the address space, where no
/// platform maps a thread's stack -- the kernel's half on 64-bit platforms,
/// and on 32-bit ones the kernel's or a range Windows keeps unmapped.
#[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
const NO_LIMIT: usize = usize::MAX - CALL_OUT_RESERVE;

/// The bytes left above the thread's `limit` at `here`, where they are no
/// more than [`CALL_OUT_RESERVE`]; `None` where a predicate called from
/// `here` has room. One wrapping subtraction reads both ends: an address
/// below the limit wraps past the reserve, and is on a stack other than the
/// thread's own, which a coroutine library switches to, and has room, as has
/// every address above the reserve and every address on a thread whose limit
/// is not read ([`NO_LIMIT`]).
#[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
#[inline]
fn short_of_the_reserve(here: usize, limit: usize) -> Option<usize> {
    let left = here.wrapping_sub(limit);
    (left <= CALL_OUT_RESERVE).then_some(left)
}

/// The `RecursionError` the walk raises with `left` bytes of the thread's
/// stack left ([`overflow_message`]).
#[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
#[cold]
#[inline(never)]
fn stack_overflow(left: usize) -> PyErr {
    pyo3::exceptions::PyRecursionError::new_err(overflow_message(left))
}

/// What the error says with `left` bytes of the thread's stack left, worded
/// as the one `CPython` 3.14 raises where the same bounds run out.
#[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
fn overflow_message(left: usize) -> String {
    format!(
        "Stack overflow ({} kB of the thread's stack left)",
        left / 1024
    )
}

/// The native stack a predicate is called with on `CPython` before 3.14:
/// room for the Python frame it runs, a walk of its own validator at the root,
/// and raising the error that ends the chain. 3.15 raises at about 35 KiB from
/// the end of a thread in the same chain.
#[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
const CALL_OUT_RESERVE: usize = 32_768;

/// An address on the current native stack: a local's, in the frame it is
/// inlined into. The address is taken and the local never read, so it is one
/// `lea` off the stack pointer, with nothing stored.
#[cfg(all(not(PyPy), not(Py_3_14), not(target_env = "musl")))]
#[inline]
fn stack_address() -> usize {
    let marker = 0_u8;
    (&raw const marker).addr()
}

/// An address on the current native stack: a local's, in the frame it is
/// inlined into, kept in memory, the reading [`ROOM_STEP`] is sized against.
#[cfg(PyPy)]
#[inline]
fn stack_address() -> usize {
    let marker = 0_u8;
    std::hint::black_box(&raw const marker).addr()
}

/// On `CPython` the stack a thread has is the whole question, and
/// [`MAX_WALK_DEPTH`] answers it.
#[cfg(not(PyPy))]
const fn the_interpreter_has_room(_state: &WalkState) -> bool {
    true
}

/// What a membership walk is being run for.
///
/// Four modes, and the type says four: the pair of independent booleans the
/// first three replace admitted a combination — fail-fast without explaining —
/// that no caller produced and the walk read as plain [`Fast`](WalkMode::Fast).
/// A state with no meaning is better unnameable than merely unused.
/// The discriminants are ordered so the predicates below are one comparison
/// rather than a two-way test: explaining is "at most `ExplainFailFast`",
/// stopping at the first failure is "at least `ExplainFailFast`", and stopping
/// before the last failure at all is "past `Explain`". The order is load-bearing
/// for that reason and not alphabetical or by importance; the predicates are
/// asserted over every variant in the tests.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub(crate) enum WalkMode {
    /// Membership plus a [`Violation`](valgebra_core::Violation) for each
    /// independent failure: every record field, sequence element, and mapping
    /// entry that fails is reported.
    Explain = 0,
    /// [`Explain`](WalkMode::Explain) up to [`WalkState::limit`] violations:
    /// a composite stops once its report holds that many. A mode of its own,
    /// so the full report asks the count nowhere.
    ExplainLimited = 1,
    /// Membership plus the first violation only.
    ExplainFailFast = 2,
    /// Membership only. No violation is built, `out` is never touched, no path is
    /// built, and every composite short-circuits as soon as the answer is fixed.
    Fast = 3,
    /// [`Fast`](WalkMode::Fast) for a complement's inner walk, which reads its
    /// answer in Kleene's strong logic: a stop is the third value, so a
    /// conjunction a later part refuses is refused and a union a later branch
    /// admits is admitted, whatever order the parts are read in.
    ///
    /// A stop answers `true` in this mode, with its code in the stop cell, so
    /// the scans read past it to a part that decides under the short circuit
    /// they take on a refusal, and only the readers that tell the three values
    /// apart -- a union, a complement, the memo -- ask the cell. Last, so both
    /// predicates read it as the fast mode: it explains nothing and stops at a
    /// refusal.
    Settle = 4,
}

impl WalkMode {
    /// Whether this mode builds violations. The explain-side bookkeeping — the
    /// path, the value summaries — is gated on this, once per node on the hot
    /// path, so it is one comparison.
    #[inline]
    pub(crate) fn explains(self) -> bool {
        self <= WalkMode::ExplainFailFast
    }

    /// Whether a composite stops at its first failing child rather than walking
    /// the rest. The fast path stops for a different reason than fail-fast does —
    /// it has nothing to aggregate — and both answer true here.
    #[inline]
    pub(crate) fn stops_at_first(self) -> bool {
        self >= WalkMode::ExplainFailFast
    }

    /// Whether a composite may stop before its last failing child: in every
    /// mode but the full report, which walks each child whatever it found.
    #[inline]
    pub(crate) fn may_stop(self) -> bool {
        self > WalkMode::Explain
    }

    /// The mode a caller asking to explain wants, given how many failures it
    /// asks for: one, every one (`usize::MAX`), or a count between.
    pub(crate) fn explaining(reported: usize) -> Self {
        match reported {
            1 => WalkMode::ExplainFailFast,
            usize::MAX => WalkMode::Explain,
            _ => WalkMode::ExplainLimited,
        }
    }
}

#[cfg(test)]
mod descent_tests;

#[cfg(test)]
mod mode_tests;
