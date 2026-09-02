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

pub(crate) mod ctx;
mod index;
mod violation;
mod walk;

pub(crate) use ctx::{Ctx, WalkMode, WalkState};
pub(crate) use index::{ValidatorIndex, build_index, compile_pattern};
pub(crate) use walk::member;
