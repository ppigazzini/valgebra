//! The validation walk: one membership test of a value against the IR.
//!
//! [`member`] is the single walk. It returns whether the value belongs to the
//! schema's set, and in *explain* mode (`ctx.explain`) it also aggregates a
//! [`Violation`] for each independent failure into `out` (each record field,
//! each sequence element, each mapping entry), unless `ctx.fail_fast` stops it at
//! the first. In *fast* mode it allocates nothing and short-circuits as soon as
//! membership is decided — the path it took before this module fused the two
//! walks into one. There is no second walk to keep in sync.
//!
//! The walk runs over a [`Value`], so the object path and the in-place JSON path
//! share one traversal. The explain side only ever sees a Python value (the JSON
//! entry points materialize before explaining), so building a violation always
//! has a Python object in hand. The per-child path bookkeeping is gated on
//! `ctx.explain`, a flag constant for a whole walk, so the fast path pays nothing
//! for it.

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
/// guard, and the two mode flags. The guard records `(object id, definition
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
    /// Build violations into `out`. When false the walk is the membership fast
    /// path: it never touches `out`, never builds a path, and short-circuits.
    pub(crate) explain: bool,
    /// In explain mode, stop at the first failure instead of aggregating siblings.
    pub(crate) fail_fast: bool,
}
