//! The read-only context a membership walk carries, and the mode it runs in.
//!
//! A leaf module under `check`: both types are defined here rather than in the
//! parent, because a type defined in an aggregator and imported back by its
//! members is what puts the two in a cycle. `check.rs` re-exports them, so no
//! caller spells a new path.

use std::cell::{Cell, RefCell};

use pyo3::prelude::*;
use valgebra_core::Schema;

use super::index::{AttrsIndex, RecordIndex, RegexIndex, UnionIndex};

/// The read-only context threaded through a validation walk: the constants pool,
/// the recursion definitions, the precomputed record index, the active recursion
/// guard, and the walk mode. The guard records `(object id, definition
/// index)` pairs currently on the path so a value that contains itself fails with
/// `recursion_loop` instead of looping.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub(crate) pool: &'a [Py<PyAny>],
    pub(crate) defs: &'a [Schema],
    /// Per-record declared-field lookups, built once per validator and keyed by
    /// the address of each record's `fields` buffer. The keyed-map fast path
    /// reads it instead of rebuilding the name map on every call; a node absent
    /// from it falls back to building the map, so correctness never depends on
    /// it being complete.
    pub(crate) records: &'a RecordIndex,
    /// Per-attribute-schema interned names, built once per validator. The
    /// attribute walk hands these to `getattr` instead of building a fresh
    /// `PyString` per attribute per value.
    pub(crate) attrs: &'a AttrsIndex,
    /// Per-union value sets for unions whose members are all literals, keyed by
    /// the address of the union's members buffer. The membership fast path
    /// dispatches an exact int or str value through it instead of scanning every
    /// branch; any other case (an explain walk, a non-literal union, a value of
    /// another type, a JSON value) falls back to the linear scan.
    pub(crate) unions: &'a UnionIndex,
    /// Compiled string patterns, keyed by source pattern; the refinement walk
    /// reads it for a `Regex(...)` constraint instead of recompiling.
    pub(crate) regexes: &'a RegexIndex,
    pub(crate) guard: &'a RefCell<Trail>,
    /// How many walk levels are open below the entry point. Each level is a
    /// native stack frame, and [`MAX_WALK_DEPTH`] is the ceiling
    /// [`descend`](Ctx::descend) holds it under.
    pub(crate) depth: &'a Cell<usize>,
    /// A fatal interpreter signal raised mid-walk — a base exception that is not
    /// an ordinary exception (`KeyboardInterrupt`, `SystemExit`, `GeneratorExit`),
    /// or a `MemoryError`/`RecursionError`. The first such error is recorded here;
    /// the walk then short-circuits and the entry point re-raises it instead of
    /// silently reporting a non-member. An ordinary exception during a membership
    /// probe stays folded to non-membership and never lands here.
    pub(crate) fatal: &'a RefCell<Option<PyErr>>,
    /// A `Cell` mirror of whether [`fatal`](Self::fatal) holds a signal yet, set
    /// alongside it in `record_fatal`. The per-node short-circuit reads this with a
    /// plain load instead of taking a `RefCell` borrow on every membership step.
    pub(crate) fatal_seen: &'a Cell<bool>,
    /// What the walk is for. Constant for a whole walk, so the fast path pays
    /// nothing for the explain bookkeeping.
    pub(crate) mode: WalkMode,
}

/// The most levels of recursive descent allowed before a value is rejected.
///
/// A finite value never reaches this; the bound exists so a pathologically deep
/// value fails with `recursion_limit` instead of overflowing the native stack.
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

    /// Close the level [`enter`](Self::enter) opened.
    pub(crate) fn leave(&mut self) {
        let left = self.0.pop();
        debug_assert!(left.is_some(), "a level closes a pair that was entered");
    }
}

/// The mutable state one membership test carries: the recursion guard, the
/// first fatal signal and the flag mirroring it, and the count of open walk
/// levels.
///
/// One owner rather than a local per cell at each entry point. They share a
/// lifetime — one call — and they are read together as `Ctx`, so a caller that
/// assembles three of them and forgets the fourth is a caller the type system
/// should not be able to spell.
pub(crate) struct WalkState {
    /// `(object id, definition index)` pairs open on the current path, so a value
    /// that contains itself fails with `recursion_loop` instead of looping.
    pub(crate) guard: RefCell<Trail>,
    pub(crate) fatal: RefCell<Option<PyErr>>,
    pub(crate) fatal_seen: Cell<bool>,
    pub(crate) depth: Cell<usize>,
}

impl WalkState {
    pub(crate) fn new() -> Self {
        Self {
            guard: RefCell::new(Trail::default()),
            fatal: RefCell::new(None),
            fatal_seen: Cell::new(false),
            depth: Cell::new(0),
        }
    }

    /// The fatal interpreter signal the walk recorded, taken by the entry point
    /// that re-raises it.
    pub(crate) fn into_fatal(self) -> Option<PyErr> {
        self.fatal.into_inner()
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
/// The figure is the stack a walk needs. A level costs under a kilobyte of
/// native stack in an unoptimized build, so 512 of them sit inside the smallest
/// stack a platform gives a thread (512 KiB) and far inside the megabytes a main
/// thread gets. A schema at the construction depth bound reaches 128 of them
/// against a flat value, so the ceiling is four times the depth any
/// non-recursive schema can ask for.
pub(crate) const MAX_WALK_DEPTH: usize = 512;

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
    /// Open one level of descent, or refuse when the walk already holds
    /// [`MAX_WALK_DEPTH`] of them.
    ///
    /// The caller turns a refusal into a non-member with a `recursion_limit`
    /// violation — the same answer an over-deep value gets from the unfolding
    /// bound, because it is the same fact about the value.
    /// Whether one more level is available, without taking it.
    ///
    /// The question a *leaf* loop asks. A scalar element cannot descend, so the
    /// level it would sit at need not be held while the loop runs -- only the
    /// refusal has to match, and [`descend`](Self::descend) refuses exactly
    /// where this answers `false`. Holding the level instead costs the loop's
    /// container a counter pair and a guard drop, which on a shape whose whole
    /// check is a type test per element is a measurable share of it.
    pub(crate) fn room_to_descend(self) -> bool {
        self.depth.get() < MAX_WALK_DEPTH
    }

    pub(crate) fn descend(self) -> Option<Descent<'a>> {
        let level = self.depth.get() + 1;
        if level > MAX_WALK_DEPTH {
            return None;
        }
        self.depth.set(level);
        Some(Descent(self.depth))
    }
}

/// What a membership walk is being run for.
///
/// Three modes, and the type says three: the pair of independent booleans this
/// replaces admitted a fourth combination — fail-fast without explaining — that
/// no caller produced and the walk read as plain [`Fast`](WalkMode::Fast). A
/// state with no meaning is better unnameable than merely unused.
/// The discriminants are ordered so both predicates below are one comparison
/// rather than a two-way test: explaining is "at most `ExplainFailFast`",
/// stopping at the first failure is "at least `ExplainFailFast`". The order is
/// load-bearing for that reason and not alphabetical or by importance; both
/// predicates are asserted over every variant in the tests.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub(crate) enum WalkMode {
    /// Membership plus a [`Violation`](valgebra_core::Violation) for each
    /// independent failure: every record field, sequence element, and mapping
    /// entry that fails is reported.
    Explain = 0,
    /// Membership plus the first violation only.
    ExplainFailFast = 1,
    /// Membership only. Nothing is allocated, `out` is never touched, no path is
    /// built, and every composite short-circuits as soon as the answer is fixed.
    Fast = 2,
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

    /// The mode a caller asking to explain wants, given its fail-fast request.
    pub(crate) fn explaining(fail_fast: bool) -> Self {
        if fail_fast {
            WalkMode::ExplainFailFast
        } else {
            WalkMode::Explain
        }
    }
}

#[cfg(test)]
mod descent_tests;

#[cfg(test)]
mod mode_tests;
