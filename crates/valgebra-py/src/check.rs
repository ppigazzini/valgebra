//! The validation walk: one membership test of a value against the IR.
//!
//! [`member`] is the single walk. It returns whether the value belongs to the
//! schema's set, and in an *explain* mode (`ctx.mode`) it also aggregates a
//! [`Violation`](valgebra_core::Violation) for each independent failure into
//! `out` (each record field, each sequence element, each mapping entry), unless
//! the fail-fast mode stops it at the first. In *fast* mode it builds no
//! violation, path or value summary and short-circuits as soon as membership is
//! decided; storage a reading needs for itself is allocated at the site that
//! says why. One dispatcher serves every mode; where a reading has a reader per
//! mode, the `decide` harness and `tests/test_equivalence.py` hold the two to
//! one verdict (`docs/dev/04-walk.md`).
//!
//! The walk runs over a [`Value`](crate::input::Value), so the object path and
//! the in-place JSON path share one traversal. The explain side only ever sees
//! a Python value (the JSON entry points materialize before explaining), so
//! building a violation always has a Python object in hand. The per-child path
//! bookkeeping is gated on `ctx.mode`, constant for a whole walk, so the fast
//! path pays a test of the mode per child, which goes the same way for the
//! whole walk, and none of the bookkeeping.

pub(crate) mod ctx;
mod index;
mod memo;
mod violation;
mod walk;

pub(crate) use ctx::{Ctx, WalkMode, WalkState};
#[cfg(all(test, feature = "interpreter-tests"))]
pub(crate) use index::build_index;
pub(crate) use index::{RegexIndex, ValidatorIndex, build_index_from, compile_patterns};
pub(crate) use walk::{Frame, Unwritten, member};
