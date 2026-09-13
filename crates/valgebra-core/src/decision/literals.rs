//! Tables of constants, decided as the sets they denote.
//!
//! A union of literals is a *finite set*, and inclusion between two finite sets
//! is membership rather than a rule about shapes. Read as the lattice rules
//! would read it, the pair costs a decision step per pair of members -- the
//! product the work budget binds; read as two tables, it costs a walk of each,
//! which is a sum.
//!
//! What makes the reading sound is that a literal pins `type(x)` exactly, so
//! two constants are one value only where the oracle says so. The core cannot
//! compare two Python objects, which is why every question here ends at
//! [`LeafRelations`](super::LeafRelations).

use std::borrow::Cow;
use std::slice;

use crate::ir::{ConstIx, Schema};
use crate::verdict::Relation;

use super::LeafRelations;

/// The constants of a schema that is a literal, or a union of nothing but
/// literals; `None` for anything else.
///
/// What makes the set question askable: a union carrying one non-literal member
/// has no set of constants standing for it, and the member walk is then the
/// only reading.
///
/// Borrowed from the node where the schema is one literal: a list of one built
/// on the heap to be read once and dropped is an allocation for nothing, and
/// the reading that calls this runs on every declined pair. A union keeps its
/// constants inside its member nodes, so there is no slice of indices to borrow
/// and that one is collected.
pub(super) fn literal_constants(schema: &Schema) -> Option<Cow<'_, [ConstIx]>> {
    match schema {
        Schema::Literal(index) => Some(Cow::Borrowed(slice::from_ref(index))),
        Schema::Union(members) if !members.is_empty() => members
            .iter()
            .map(|member| match member {
                Schema::Literal(index) => Some(*index),
                _ => None,
            })
            .collect::<Option<Vec<ConstIx>>>()
            .map(Cow::Owned),
        _ => None,
    }
}

/// The members of a schema that denotes a **finite set of constants**: a
/// literal, or a union of nothing but literals in the canonical order its
/// constructor leaves them in.
///
/// `None` for everything else, and for a union whose literals are not strictly
/// increasing. The variant is public and a caller may build one by hand, and
/// the rule below reads the list as a *set* by searching it, which an unordered
/// list would answer wrongly; a union the constructors did not order keeps the
/// member walk it had. The order is the derived one, so it is the order of the
/// pool indices, and the constructors sort and deduplicate.
pub(super) fn finite_set(schema: &Schema) -> Option<&[Schema]> {
    match schema {
        Schema::Literal(_) => Some(std::slice::from_ref(schema)),
        Schema::Union(members) => {
            let mut previous: Option<ConstIx> = None;
            for member in members.iter() {
                let Schema::Literal(index) = member else {
                    return None;
                };
                if previous.is_some_and(|held| held >= *index) {
                    return None;
                }
                previous = Some(*index);
            }
            // An empty union is the bottom rather than a set of constants, and
            // the lattice bound below decides it.
            previous.is_some().then(|| members.as_ref())
        }
        _ => None,
    }
}

/// `A ⊆ B` between two finite sets of constants, decided by membership.
///
/// A literal denotes a singleton, so a union of literals denotes the set of its
/// constants, and inclusion between two such sets is membership of every
/// constant of one in the other. That is **exact in both directions**, which is
/// what a set gives that a shape does not: every member found is a proof, and
/// one member found nowhere is a refutation naming the value that stands
/// against the inclusion.
///
/// Both readings are one walk. The lists are canonical, so a member is found by
/// binary search rather than by a scan, and the pair costs `n log m` where the
/// distribution it stands in front of costs a decision step per pair of
/// members. A table of a thousand codes against another is a million steps
/// there and ten thousand comparisons here, which is the difference between the
/// product the budget binds and a sum it does not
/// (`docs/15-decidability.md`).
///
/// The refutation is the oracle's to give. Two constants at two indices are two
/// *values* only where the bindings can compare them, and a constant that does
/// not equal itself denotes no value at all -- so a member found nowhere asks
/// whether it is disjoint from the whole supertype set, in the one call that
/// question has, and declines to the rules below where the oracle does. A
/// refutation from here is a mismatch like any other: the subject's own
/// emptiness is read against it by `witnessed`, at the level this pair is
/// decided.
pub(super) fn finite_set_below(
    subject: &[Schema],
    supertype: &[Schema],
    other: &Schema,
    oracle: &dyn LeafRelations,
) -> Relation {
    // Searched by the constant's index rather than by the node. Both slices
    // come from `finite_set`, which admits nothing but literals in strictly
    // increasing index order -- so the two orderings agree, and the derived one
    // runs a comparison over the whole variant table where an index is a single
    // integer compare. It was thirty-eight percent of the proving workload.
    //
    // A member that is not a literal cannot occur and is read as missing, which
    // is the conservative direction: the pair loses a proof rather than gaining
    // one.
    let index_of = |schema: &Schema| match schema {
        Schema::Literal(index) => Some(*index),
        _ => None,
    };
    let Some(missing) = subject.iter().find(|member| {
        index_of(member).is_none_or(|wanted| {
            supertype
                .binary_search_by(|held| match index_of(held) {
                    Some(index) => index.cmp(&wanted),
                    None => core::cmp::Ordering::Less,
                })
                .is_err()
        })
    }) else {
        return Relation::Holds;
    };
    // Both sides are literals by construction, so both readings are `Some`; a
    // `None` here would be this rule and `literal_constants` disagreeing about
    // what a set of constants is, and the pair keeps the rules it had.
    let (Some(alone), Some(whole)) = (literal_constants(missing), literal_constants(other)) else {
        return Relation::Unknown;
    };
    match oracle.literal_sets_disjoint(&alone, &whole) {
        Some(true) => Relation::Fails,
        _ => Relation::Unknown,
    }
}
