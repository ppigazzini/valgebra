//! valgebra's schema algebra: the IR, the two deciders, and the frame they share.
//!
//! A schema denotes a set of Python values; validation is membership, inclusion
//! is `a ∧ ¬b = ∅`, and this crate is where both are defined. It holds four
//! things, in the order a reader meets them:
//!
//! * [`Schema`] and its transforms -- the node set, with each variant's doc
//!   comment stating which set of values it admits, which is where a claim
//!   about *meaning* is checked;
//! * [`kind`](Kind) and the two three-valued answers, [`Verdict`] and
//!   [`Relation`] -- the frame both deciders read, below either of them;
//! * `descr` -- the **set representation**: one representation per kind, each
//!   closed under union, intersection and complement, deciding a relation
//!   between two schemas it can build by the emptiness of one combination;
//! * `decision` -- the **structural rules** over the schema tree, which reach
//!   the recursive schemas, predicates and hooked classes the set
//!   representation cannot build, answer first, and hand it a pair they
//!   decline.
//!
//! Inspecting a Python object requires `PyO3`, so the membership walk lives in
//! the bindings crate and reaches back through [`LeafRelations`] for the
//! questions only an interpreter answers. This crate is the stable,
//! language-agnostic core, and the [`Violation`] it produces is the structured
//! report a failure carries.
//!
//! The crate forbids `unsafe`: the security policy's no-unsafe guarantee is
//! compiler-enforced here, not merely asserted, so a future `unsafe` block fails
//! the build instead of silently voiding it.
#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};

mod carries;
pub mod codes;
mod decision;
pub mod descr;
mod ir;
mod kind;
mod oracle;
mod verdict;
mod violation;

pub use carries::{
    Carries, OrderGroup, StepGroup, carries_division, carries_length, carries_order,
    carries_pattern, carries_through,
};
pub use decision::{LeafRelations, NoLeafRelations};
pub use ir::{
    Attribute, ClassIx, CollKind, ConstIx, Constraint, Constraints, DefIx, DefShift, Field, Fields,
    Guarded, LONGEST, MapClause, Measure, Members, Openness, OperandIx, PathSegment, PredIx,
    Schema, SeqKind, SeqShape, Spelling, pruned,
};
pub use kind::Kind;
pub use verdict::{Relation, Verdict};
pub use violation::{Violation, quoted};

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

/// Every unordered pair of distinct items, each once.
///
/// Two scans read it: the pairwise-disjointness law over a meet's members, and
/// a map atom's emptiness over the keys it requires, where `True` and `1` are
/// one dict key. The tail is read through `get`, so neither needs a panicking
/// index.
pub(crate) fn unordered_pairs<T>(items: &[T]) -> impl Iterator<Item = (&T, &T)> {
    items.iter().enumerate().flat_map(|(i, a)| {
        items
            .get(i + 1..)
            .unwrap_or_default()
            .iter()
            .map(move |b| (a, b))
    })
}

/// What `read` gives of each of `items`, joined two at a time, each round halving
/// the list, or `None` where a read or a join refuses; `identity` where there are
/// none.
///
/// A fold joining the whole so far with each item in turn reads the whole again
/// at every item, which costs the square of the width where a join reads what it
/// holds once, and the cube where it reads each of those against the others.
/// Joined in rounds, an item is read once a round, and there are as many rounds as
/// the width has binary digits. Every item is read before the first join, so a read
/// that refuses spends no join.
///
/// **Up to three items, these are the fold's joins, and they hold no list.** Most
/// unions are that narrow, and a list of descriptors is a request past the
/// allocator's bins, whose consolidating costs more than the union it would hold.
pub(crate) fn joined_in_rounds<'a, I, T>(
    items: &'a [I],
    mut read: impl FnMut(&'a I) -> Option<T>,
    identity: impl FnOnce() -> T,
    join: impl Fn(T, T) -> Option<T>,
) -> Option<T> {
    if let [_, _, _, _, ..] = items {
        let mut held = items.iter().map(read).collect::<Option<Vec<_>>>()?;
        // Read as the shape of the list, two members or more, so the loop's end
        // is no comparison a slip could keep open.
        while let [_, _, ..] = held.as_slice() {
            let mut pairs = held.into_iter();
            let mut next = Vec::with_capacity(pairs.len().div_ceil(2));
            while let Some(first) = pairs.next() {
                next.push(match pairs.next() {
                    Some(second) => join(first, second)?,
                    None => first,
                });
            }
            held = next;
        }
        return held.pop();
    }
    let mut items = items.iter();
    let Some(first) = items.next() else {
        return Some(identity());
    };
    items.try_fold(read(first)?, |whole, item| join(whole, read(item)?))
}

#[cfg(test)]
mod tests;

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
