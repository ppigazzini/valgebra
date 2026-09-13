//! The algebra of a refinement's bounds: which pairs cannot hold together, and
//! which one entails another.
//!
//! A refinement is a base set narrowed by predicates, and these are the
//! questions a decision asks about the predicates alone -- without reading the
//! base, and without an SMT solver to reason about them. A bound is compared
//! through the oracle, because the values it orders are Python objects the core
//! cannot see, and a user predicate is opaque to all of it.
//!
//! Two of the answers are worth naming apart. **Unsatisfiable** is a proof about
//! the constraints: a lower bound above an upper one admits nothing, whatever
//! the base. **Entailed** is a proof about a pair: a tighter bound is inside a
//! looser one, which is what makes a refinement a subtype of a weaker
//! refinement of the same base.

use crate::ir::{Constraint, OperandIx};

use super::LeafRelations;

/// One end of an order bound: the constant, and whether the end is strict.
type Bound = Option<(OperandIx, bool)>;

/// The shortest length a conjunction of bounds admits, which is the largest
/// `MinLen` among them and zero where none is written.
pub(super) fn shortest<'a>(constraints: impl Iterator<Item = &'a Constraint>) -> usize {
    constraints
        .filter_map(|c| match c {
            Constraint::MinLen(n) => Some(*n),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// The tightest lower and upper bound a conjunction names, and whether every
/// comparison it took was answered.
///
/// One fold, read by two questions. Emptiness ignores the flag: where the
/// oracle cannot order two bounds on one side, keeping either is sound, because
/// an interval it proves empty under the looser one is empty under the tighter.
/// Inhabitance cannot -- naming a value under the looser bound would name one
/// the tighter excludes -- so it reads the flag and declines.
pub(super) fn tightest_bounds<'a>(
    constraints: impl Iterator<Item = &'a Constraint>,
    oracle: &dyn LeafRelations,
) -> (Bound, Bound, bool) {
    let mut lower: Bound = None;
    let mut upper: Bound = None;
    let mut ordered = true;
    for constraint in constraints {
        let (bound, is_lower) = match constraint {
            Constraint::Ge(i) => ((*i, false), true),
            Constraint::Gt(i) => ((*i, true), true),
            Constraint::Le(i) => ((*i, false), false),
            Constraint::Lt(i) => ((*i, true), false),
            _ => continue,
        };
        let slot = if is_lower { &mut lower } else { &mut upper };
        ordered &= slot.is_none() || oracle.compare(bound.0, slot.unwrap_or(bound).0).is_some();
        *slot = Some(tighter_bound(*slot, bound, oracle, is_lower));
    }
    (lower, upper, ordered)
}

pub(super) fn bounds_unsatisfiable<'a>(
    constraints: impl Iterator<Item = &'a Constraint> + Clone,
    oracle: &dyn LeafRelations,
    int_discrete: bool,
) -> bool {
    use core::cmp::Ordering;
    let min_len = constraints
        .clone()
        .filter_map(|c| match c {
            Constraint::MinLen(n) => Some(*n),
            _ => None,
        })
        .max();
    let max_len = constraints
        .clone()
        .filter_map(|c| match c {
            Constraint::MaxLen(n) => Some(*n),
            _ => None,
        })
        .min();
    if let (Some(lo), Some(hi)) = (min_len, max_len)
        && lo > hi
    {
        return true;
    }
    let (lower, upper, _) = tightest_bounds(constraints, oracle);
    if let (Some((lo, lo_strict)), Some((hi, hi_strict))) = (lower, upper) {
        match oracle.compare(lo, hi) {
            Some(Ordering::Greater) => return true,
            Some(Ordering::Equal) => return lo_strict || hi_strict,
            _ => {}
        }
        // An integer-discrete base bounds the integers in the interval, so the
        // refinement is empty when no integer lies between the bounds even though
        // the endpoints themselves are ordered `lo < hi` — `Annotated[int, Gt(0),
        // Lt(1)]` admits no value. The oracle answers only for a real numeric
        // pair and stays `None` otherwise, so floats and incomparable bounds keep
        // the interval conservatively non-empty.
        if int_discrete && oracle.no_int_between(lo, lo_strict, hi, hi_strict) == Some(true) {
            return true;
        }
    }
    false
}

/// Whether a single supertype refinement constraint is *entailed* by the subtype's
/// constraint set: every value satisfying all of `narrow` also satisfies `wide`.
/// Order and length bounds entail by value (a tighter lower bound entails a looser
/// one, dually for upper and length), decided through the ordering `oracle`; the
/// remaining kinds (`MultipleOf`, `Predicate`, `Regex`) have no sound value
/// entailment and require the constraint to appear verbatim, handled by the
/// caller's syntactic-containment check. A bound the oracle cannot compare is not
/// entailed (conservative).
pub(super) fn constraint_entailed(
    wide: &Constraint,
    narrow: &[Constraint],
    oracle: &dyn LeafRelations,
) -> bool {
    use core::cmp::Ordering;
    let ge = |o: Option<Ordering>| matches!(o, Some(Ordering::Greater | Ordering::Equal));
    let gt = |o: Option<Ordering>| matches!(o, Some(Ordering::Greater));
    let le = |o: Option<Ordering>| matches!(o, Some(Ordering::Less | Ordering::Equal));
    let lt = |o: Option<Ordering>| matches!(o, Some(Ordering::Less));
    match wide {
        // x >= w holds if the subtype forces a lower bound at value >= w.
        Constraint::Ge(w) => narrow.iter().any(|c| match c {
            Constraint::Ge(n) | Constraint::Gt(n) => ge(oracle.compare(*n, *w)),
            _ => false,
        }),
        // x > w holds from Gt(n>=w), or Ge(n>w).
        Constraint::Gt(w) => narrow.iter().any(|c| match c {
            Constraint::Gt(n) => ge(oracle.compare(*n, *w)),
            Constraint::Ge(n) => gt(oracle.compare(*n, *w)),
            _ => false,
        }),
        // x <= w holds if the subtype forces an upper bound at value <= w.
        Constraint::Le(w) => narrow.iter().any(|c| match c {
            Constraint::Le(n) | Constraint::Lt(n) => le(oracle.compare(*n, *w)),
            _ => false,
        }),
        // x < w holds from Lt(n<=w), or Le(n<w).
        Constraint::Lt(w) => narrow.iter().any(|c| match c {
            Constraint::Lt(n) => le(oracle.compare(*n, *w)),
            Constraint::Le(n) => lt(oracle.compare(*n, *w)),
            _ => false,
        }),
        // Length bounds compare by their raw counts.
        Constraint::MinLen(w) => narrow
            .iter()
            .any(|c| matches!(c, Constraint::MinLen(n) if n >= w)),
        Constraint::MaxLen(w) => narrow
            .iter()
            .any(|c| matches!(c, Constraint::MaxLen(n) if n <= w)),
        // No sound value entailment without an exact match (handled by the caller).
        Constraint::MultipleOf(_) | Constraint::Predicate(_) | Constraint::Regex(_) => false,
    }
}

/// Keep the tighter of two one-sided bounds: the greater value for a lower bound,
/// the lesser for an upper bound; on equal values the strict end wins, and on an
/// incomparable pair the current bound is kept (conservative).
pub(super) fn tighter_bound(
    current: Option<(OperandIx, bool)>,
    candidate: (OperandIx, bool),
    oracle: &dyn LeafRelations,
    is_lower: bool,
) -> (OperandIx, bool) {
    use core::cmp::Ordering;
    let Some(current) = current else {
        return candidate;
    };
    match oracle.compare(candidate.0, current.0) {
        Some(Ordering::Equal) => (current.0, current.1 || candidate.1),
        Some(Ordering::Greater) => {
            if is_lower {
                candidate
            } else {
                current
            }
        }
        Some(Ordering::Less) => {
            if is_lower {
                current
            } else {
                candidate
            }
        }
        None => current,
    }
}
