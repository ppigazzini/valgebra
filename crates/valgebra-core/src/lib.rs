//! valgebra schema intermediate representation.
//!
//! A schema denotes a set of Python values; validation is membership. This
//! crate is pure Rust: it defines the IR, the denotation of every node, and the
//! structured [`Violation`] produced when membership fails. Inspecting a Python
//! object requires `PyO3`, so the validator walk itself lives in the bindings
//! crate; this crate is the stable, language-agnostic core.
//!
//! The crate forbids `unsafe`: the security policy's no-unsafe guarantee is
//! compiler-enforced here, not merely asserted, so a future `unsafe` block fails
//! the build instead of silently voiding it.
#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};

mod decision;
pub mod descr;
mod ir;
mod simplify;
mod violation;

pub use decision::{Kind, LeafRelations, NoLeafRelations, Verdict};
pub use ir::{
    ClassIx, Clauses, CollKind, ConstIx, Constraint, Constraints, DefIx, DefShift, Field, Fields,
    Guarded, MapClause, Members, Openness, OperandIx, PathSegment, PoolShift, PredIx, Schema,
    SeqKind, SeqShape, Spelling, pruned,
};
pub use violation::Violation;

/// Fresh tokens for the transient [`Schema::SelfRef`] marker, so no two
/// `recursive` definitions ever resolve each other's self-references.
///
/// **Process-unique, deliberately.** The placeholder carrying a token is an
/// ordinary Python object, so a caller can keep one past the builder call that
/// gave it meaning and pass it into another -- on this thread or on any other.
/// The binding refuses that, by asking whether the token names a definition
/// currently being built; and that question is only sound while a token means
/// one definition across the whole process. A per-thread counter would hand two
/// threads the same first token, and one thread's escaped placeholder would then
/// answer to the other's open definition: not a refusal, but a silently
/// different schema. The counter is shared so that the tokens cannot collide.
static NEXT_SELF_TOKEN: AtomicU64 = AtomicU64::new(0);

/// Allocate a fresh self-reference token for a `recursive` definition.
///
/// `Relaxed` is the whole ordering this needs. The token is compared for
/// equality and never orders anything, so the one guarantee wanted from the
/// counter is that no two calls return the same value -- which `fetch_add`
/// gives under any ordering.
#[must_use]
pub fn fresh_self_token() -> u64 {
    NEXT_SELF_TOKEN.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod schema_tests;

#[cfg(test)]
mod laws;

/// The laws the two index-space remappings obey.
///
/// `Schema::shifted` and `Schema::reindexed` move every pool and definitions
/// index a schema holds. The unit tests above pin single sites by example, which
/// is the direction that catches a site moved *wrongly*; a law holds the whole
/// operation at once, which is the direction that catches a site moved **not at
/// all**. A payload the shift walks past is invisible to an example written for
/// the payloads somebody thought of, and it is the failure the typed index
/// spaces cannot see: the types stop a shift being applied to the wrong space,
/// and say nothing about whether it was applied.
///
/// The generator's job here is site coverage rather than algebraic variety --
/// every node that carries an index, and every structural node that must carry a
/// remap into its children. A variant it omits is a site these laws do not hold.
#[cfg(test)]
mod index_laws;
