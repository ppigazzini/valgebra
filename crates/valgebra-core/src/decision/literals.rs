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
//! [`LeafRelations`].

use std::borrow::Cow;
use std::slice;

use crate::ir::{ConstIx, Schema};
use crate::verdict::{Relation, Verdict};

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

/// A meet of finite sets of constants and complements of finite sets, read as
/// the finite set it denotes: the constants every kept table holds and no
/// taken table does.
///
/// It is the term a record's tag narrows to when a record is split across the
/// records that fix it, one complement per record, and the term a table minus
/// each of its members is.
pub(super) struct FiniteMeet<'s> {
    kept: Vec<&'s [Schema]>,
    taken: Vec<&'s [Schema]>,
}

impl<'s> FiniteMeet<'s> {
    /// The meet of `members`, where each is a finite set of constants or the
    /// complement of one and at least one is a set kept; `None` otherwise.
    pub(super) fn of(members: &'s [Schema]) -> Option<Self> {
        // Read once without collecting, so a meet that is no such thing -- the
        // common one -- allocates nothing to find that out.
        let table = |member: &'s Schema| match member {
            Schema::Complement(inner) => finite_set(inner),
            other => finite_set(other),
        };
        if !members.iter().all(|member| table(member).is_some()) {
            return None;
        }
        let mut kept = Vec::new();
        let mut taken = Vec::new();
        for member in members {
            match member {
                Schema::Complement(inner) => taken.push(finite_set(inner)?),
                other => kept.push(finite_set(other)?),
            }
        }
        (!kept.is_empty()).then_some(Self { kept, taken })
    }

    /// Whether the meet holds a value: empty where every constant of its first
    /// kept table is out, inhabited where one is in.
    pub(super) fn verdict(&self, oracle: &dyn LeafRelations) -> Verdict {
        let Some(first) = self.kept.first() else {
            return Verdict::Unknown;
        };
        let mut verdict = Verdict::Empty;
        for member in *first {
            match self.placed(member, oracle) {
                Placed::Out => {}
                Placed::In => return Verdict::Inhabited,
                Placed::Unread => verdict = Verdict::Unknown,
            }
        }
        verdict
    }

    /// `self ⊆ supertype` for a finite set `supertype`: every constant of the
    /// meet found in it, and refuted by one that is in the meet and apart from
    /// it.
    ///
    /// A constant not proved out of the meet is asked of the supertype whatever
    /// else is known of it, since one found there by index is in it either way;
    /// the proof is conservative where the meet reads wider than it is, and
    /// never wrong.
    pub(super) fn below(&self, supertype: &[Schema], oracle: &dyn LeafRelations) -> Relation {
        let Some(first) = self.kept.first() else {
            return Relation::Unknown;
        };
        let mut answer = Relation::Holds;
        for member in *first {
            let place = self.placed(member, oracle);
            if place == Placed::Out {
                continue;
            }
            let Schema::Literal(index) = member else {
                answer = Relation::Unknown;
                continue;
            };
            if holds_index(supertype, *index) {
                continue;
            }
            if place == Placed::In && apart_from(*index, supertype, oracle) {
                return Relation::Fails;
            }
            answer = Relation::Unknown;
        }
        answer
    }

    /// Where `member` stands, read by index first and by the oracle after.
    ///
    /// One index is one constant, so a table holding the index holds the
    /// constant, and a constant whose index a taken table holds is out exactly.
    /// The converse is the oracle's: two indices may name equal constants, so a
    /// constant no kept table is missing and no taken table holds by index is
    /// in the set only where the oracle says it is a value apart from every
    /// taken table.
    fn placed(&self, member: &Schema, oracle: &dyn LeafRelations) -> Placed {
        let Schema::Literal(index) = member else {
            return Placed::Unread;
        };
        if self.taken.iter().any(|table| holds_index(table, *index)) {
            return Placed::Out;
        }
        let mut in_every = true;
        for table in &self.kept {
            if holds_index(table, *index) {
                continue;
            }
            if apart_from(*index, table, oracle) {
                return Placed::Out;
            }
            in_every = false;
        }
        let is_value = || oracle.literals_disjoint(*index, *index) == Some(false);
        let apart_from_taken = || {
            self.taken
                .iter()
                .all(|table| apart_from(*index, table, oracle))
        };
        if in_every && is_value() && apart_from_taken() {
            Placed::In
        } else {
            Placed::Unread
        }
    }
}

/// `meet ⊆ other` for a meet of `members` that is a [`FiniteMeet`] and a table
/// `other`, and `Unknown` for any other pair.
///
/// Out of line, since the meet arm it is asked from is inlined into the goal
/// recursion and nearly every pair passes it by.
#[inline(never)]
pub(super) fn finite_meet_below(
    members: &[Schema],
    other: &Schema,
    oracle: &dyn LeafRelations,
) -> Relation {
    let Some(supertype) = finite_set(other) else {
        return Relation::Unknown;
    };
    FiniteMeet::of(members).map_or(Relation::Unknown, |meet| meet.below(supertype, oracle))
}

/// Where a constant of a kept table stands in the set a [`FiniteMeet`]
/// denotes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placed {
    /// Not in it: a taken table holds its index, or a kept table is disjoint
    /// from it.
    Out,
    /// In it: every kept table holds its index, and it is a value disjoint
    /// from every taken table.
    In,
    /// Neither is proved.
    Unread,
}

/// Whether the canonical `table` holds the constant at `index`.
fn holds_index(table: &[Schema], index: ConstIx) -> bool {
    table
        .binary_search_by(|held| match held {
            Schema::Literal(at) => at.cmp(&index),
            _ => core::cmp::Ordering::Less,
        })
        .is_ok()
}

/// Whether the oracle says the constant at `index` is no constant of `table`.
fn apart_from(index: ConstIx, table: &[Schema], oracle: &dyn LeafRelations) -> bool {
    let indices: Vec<ConstIx> = table
        .iter()
        .filter_map(|held| match held {
            Schema::Literal(at) => Some(*at),
            _ => None,
        })
        .collect();
    oracle.literal_sets_disjoint(&[index], &indices) == Some(true)
}

/// `A ⊆ B` between two finite sets of constants, decided by membership.
///
/// A literal denotes a singleton [FCB08, §4.2], or the empty set for a constant
/// unequal to itself, so a union of literals denotes the set of its constants,
/// and inclusion between two such sets is membership of every constant of one
/// in the other. That is **exact in both directions**, which is what a set
/// gives that a shape does not: every member found is a proof, and one member
/// found nowhere is a refutation naming the value that stands against the
/// inclusion.
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
    // integer compare. Searched by the node, it would be thirty-eight percent
    // of the proving workload.
    //
    // A member that is not a literal cannot occur and is read as missing, which
    // is the conservative direction: the pair loses a proof rather than gaining
    // one.
    let index_of = |schema: &Schema| match schema {
        Schema::Literal(index) => Some(*index),
        _ => None,
    };
    let absent = |member: &Schema| {
        index_of(member).is_none_or(|wanted| {
            supertype
                .binary_search_by(|held| match index_of(held) {
                    Some(index) => index.cmp(&wanted),
                    None => core::cmp::Ordering::Less,
                })
                .is_err()
        })
    };
    // A member the supertype does not hold refutes only where it **is a value**.
    // A constant that does not equal itself -- `float("nan")` -- denotes the
    // empty set, which is a subset of every set, so a union carrying one beside
    // members the supertype does hold is still below it. Asking the oracle
    // whether a constant is disjoint from itself is asking whether it denotes
    // anything at all.
    let mut missing = None;
    let mut every_absent_member_settled = true;
    for member in subject {
        if !absent(member) {
            continue;
        }
        match index_of(member).and_then(|index| oracle.literals_disjoint(index, index)) {
            Some(true) => {}
            Some(false) => {
                missing = Some(member);
                break;
            }
            None => every_absent_member_settled = false,
        }
    }
    let Some(missing) = missing else {
        return if every_absent_member_settled {
            Relation::Holds
        } else {
            Relation::Unknown
        };
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
