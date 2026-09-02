//! The read-only context a membership walk carries, and the mode it runs in.
//!
//! A leaf module under `check`: both types are defined here rather than in the
//! parent, because a type defined in an aggregator and imported back by its
//! members is what puts the two in a cycle. `check.rs` re-exports them, so no
//! caller spells a new path.

use std::cell::{Cell, RefCell};

use pyo3::prelude::*;
use rustc_hash::FxHashSet;
use valgebra_core::Schema;

use super::index::{RecordIndex, RegexIndex, UnionIndex};

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
    /// Per-union value sets for unions whose members are all literals, keyed by
    /// the address of the union's members buffer. The membership fast path
    /// dispatches an exact int or str value through it instead of scanning every
    /// branch; any other case (an explain walk, a non-literal union, a value of
    /// another type, a JSON value) falls back to the linear scan.
    pub(crate) unions: &'a UnionIndex,
    /// Compiled string patterns, keyed by source pattern; the refinement walk
    /// reads it for a `Regex(...)` constraint instead of recompiling.
    pub(crate) regexes: &'a RegexIndex,
    pub(crate) guard: &'a RefCell<FxHashSet<(usize, usize)>>,
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
    pub(crate) guard: RefCell<FxHashSet<(usize, usize)>>,
    pub(crate) fatal: RefCell<Option<PyErr>>,
    pub(crate) fatal_seen: Cell<bool>,
    pub(crate) depth: Cell<usize>,
}

impl WalkState {
    pub(crate) fn new() -> Self {
        Self {
            guard: RefCell::new(FxHashSet::default()),
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
mod descent_tests {
    use super::{Ctx, MAX_WALK_DEPTH, WalkMode, WalkState};
    use pyo3::exceptions::PyKeyboardInterrupt;
    use rustc_hash::FxHashMap;

    /// Build a context over an empty validator. Only the depth counter is read
    /// here, and it is the one piece of the context that needs no interpreter:
    /// a `Cell<usize>` and the guard that returns a level to it.
    fn with_ctx(state: &WalkState, run: impl FnOnce(Ctx<'_>)) {
        let records = FxHashMap::default();
        let unions = FxHashMap::default();
        let regexes = FxHashMap::default();
        run(Ctx {
            pool: &[],
            defs: &[],
            records: &records,
            unions: &unions,
            regexes: &regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Fast,
        });
    }

    /// The counter is a *depth*, not a total: a level is returned when its guard
    /// drops, so a wide value takes and returns one level per child and only
    /// nesting accumulates.
    ///
    /// This is the whole reason the bound can be a fixed number. A counter that
    /// only ever rose would refuse a flat list of 512 integers, which is not a
    /// value that risks the stack.
    #[test]
    fn a_level_is_returned_when_its_descent_ends() {
        let state = WalkState::new();
        with_ctx(&state, |ctx| {
            assert_eq!(state.depth.get(), 0);
            {
                let _outer = ctx.descend().expect("the first level is open");
                assert_eq!(state.depth.get(), 1);
                {
                    let _inner = ctx.descend().expect("a second level nests");
                    assert_eq!(state.depth.get(), 2);
                }
                assert_eq!(state.depth.get(), 1, "the inner level was returned");
            }
            assert_eq!(state.depth.get(), 0, "the outer level was returned");

            // Width, which is the case the bound must not refuse: a thousand
            // siblings, each taking the level the last one gave back.
            for _ in 0..1_000 {
                let _sibling = ctx.descend().expect("a sibling reuses the level");
                assert_eq!(state.depth.get(), 1);
            }
            assert_eq!(state.depth.get(), 0);
        });
    }

    /// The ceiling admits exactly `MAX_WALK_DEPTH` open levels and refuses the
    /// next, and refusing costs nothing: the walk that gets `None` has taken no
    /// level, so the counter is where it was and the levels above it still close.
    #[test]
    fn the_ceiling_admits_its_own_number_of_levels_and_no_more() {
        let state = WalkState::new();
        with_ctx(&state, |ctx| {
            let open: Vec<_> = (0..MAX_WALK_DEPTH)
                .map(|level| {
                    ctx.descend()
                        .unwrap_or_else(|| panic!("level {level} is inside the bound"))
                })
                .collect();
            assert_eq!(state.depth.get(), MAX_WALK_DEPTH);
            assert!(
                ctx.descend().is_none(),
                "the level past the bound must be refused"
            );
            assert_eq!(
                state.depth.get(),
                MAX_WALK_DEPTH,
                "a refused descent takes no level"
            );
            drop(open);
            assert_eq!(state.depth.get(), 0);
            assert!(ctx.descend().is_some(), "the bound is not a one-way latch");
        });
    }

    /// The recorded signal reaches the entry point that re-raises it. Without
    /// that hand-off a fatal interpreter signal -- a `KeyboardInterrupt` raised
    /// inside a predicate -- is swallowed and the value reads as a non-member.
    #[test]
    fn the_recorded_signal_leaves_with_the_state() {
        assert!(
            WalkState::new().into_fatal().is_none(),
            "a walk that saw no signal carries none out"
        );
        let state = WalkState::new();
        *state.fatal.borrow_mut() = Some(PyKeyboardInterrupt::new_err("stop"));
        state.fatal_seen.set(true);
        assert!(
            state.into_fatal().is_some(),
            "the signal must reach the entry point that re-raises it"
        );
    }
}

#[cfg(test)]
mod mode_tests {
    use super::WalkMode;

    /// Both predicates are single comparisons over an ordered discriminant, so
    /// they are pinned over every variant: a reordering that changes what a mode
    /// means fails here rather than silently switching the walk's behaviour.
    #[test]
    fn every_mode_answers_both_predicates() {
        assert!(WalkMode::Explain.explains());
        assert!(WalkMode::ExplainFailFast.explains());
        assert!(!WalkMode::Fast.explains());

        assert!(!WalkMode::Explain.stops_at_first());
        assert!(WalkMode::ExplainFailFast.stops_at_first());
        assert!(WalkMode::Fast.stops_at_first());

        assert_eq!(WalkMode::explaining(true), WalkMode::ExplainFailFast);
        assert_eq!(WalkMode::explaining(false), WalkMode::Explain);
    }
}
