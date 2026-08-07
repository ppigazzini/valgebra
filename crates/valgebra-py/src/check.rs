//! The validation walk: one membership test of a value against the IR.
//!
//! [`member`] is the single walk. It returns whether the value belongs to the
//! schema's set, and in an *explain* mode (`ctx.mode`) it also aggregates a
//! [`Violation`] for each independent failure into `out` (each record field,
//! each sequence element, each mapping entry), unless the fail-fast mode stops it
//! at the first. In *fast* mode it allocates nothing and short-circuits as soon as
//! membership is decided — the path it took before this module fused the two
//! walks into one. There is no second walk to keep in sync.
//!
//! The walk runs over a [`Value`], so the object path and the in-place JSON path
//! share one traversal. The explain side only ever sees a Python value (the JSON
//! entry points materialize before explaining), so building a violation always
//! has a Python object in hand. The per-child path bookkeeping is gated on
//! `ctx.mode`, constant for a whole walk, so the fast path pays nothing for it.

use std::cell::{Cell, RefCell};

use pyo3::prelude::*;
use rustc_hash::FxHashSet;
use valgebra_core::Schema;

mod index;
mod violation;
mod walk;

use index::{RecordIndex, RegexIndex, UnionIndex};

pub(crate) use index::{ValidatorIndex, build_index, compile_pattern};
pub(crate) use walk::member;

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
