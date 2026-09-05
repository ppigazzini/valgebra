//! A set of values, with a name for the top.
//!
//! A [`Guard`] is a Boolean algebra of value sets, and it has a bottom of its
//! own -- [`Guard::none`] -- but no top. It cannot have one: the letter a
//! sequence automaton reads is a descriptor, and a descriptor that named its own
//! universe would have to build itself. So the top is carried *beside* the
//! guard, as a variant rather than a value, which is the same thing the sequence
//! automaton does with its else edge.

use super::symbolic::Guard;
use crate::decision::Verdict;

/// A set of values: every one, or the ones a guard holds.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Values<G> {
    /// Every value.
    Every,
    /// The values one guard holds.
    Only(G),
}

impl<G: Guard> Values<G> {
    /// No value at all.
    #[must_use]
    pub fn none() -> Values<G> {
        Values::Only(G::none())
    }

    /// The values in both, or `None` where a guard refuses.
    #[must_use]
    pub fn meet(&self, other: &Values<G>) -> Option<Values<G>> {
        match (self, other) {
            (Values::Every, kept) | (kept, Values::Every) => Some(kept.clone()),
            (Values::Only(a), Values::Only(b)) => Some(Values::Only(a.meet(b)?)),
        }
    }

    /// The values in either, or `None` where a guard refuses.
    #[must_use]
    pub fn join(&self, other: &Values<G>) -> Option<Values<G>> {
        match (self, other) {
            (Values::Every, _) | (_, Values::Every) => Some(Values::Every),
            (Values::Only(a), Values::Only(b)) => Some(Values::Only(a.join(b)?)),
        }
    }

    /// The values in neither, which a guard always answers.
    #[must_use]
    pub fn complement(&self) -> Values<G> {
        match self {
            Values::Every => Values::none(),
            Values::Only(guard) => Values::Only(guard.complement()),
        }
    }

    /// Whether no value is in here.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.emptiness() == Verdict::Empty
    }

    /// What is known about a value being in here. The universe holds one
    /// whatever the guards say.
    #[must_use]
    pub fn emptiness(&self) -> Verdict {
        match self {
            Values::Every => Verdict::Inhabited,
            Values::Only(guard) => guard.emptiness(),
        }
    }

    /// Whether every value in `inner` is in this one, or `None` where a guard
    /// refuses.
    ///
    /// Asked as `inner ∧ ¬self = ∅`, which is the one shape a Boolean algebra
    /// answers without an order of its own.
    #[must_use]
    pub fn covers(&self, inner: &Values<G>) -> Option<bool> {
        match self {
            Values::Every => Some(true),
            Values::Only(_) => Some(inner.meet(&self.complement())?.is_empty()),
        }
    }

    /// Whether `value` is one of these.
    #[must_use]
    pub fn holds(&self, value: &G::Value) -> bool {
        match self {
            Values::Every => true,
            Values::Only(guard) => guard.holds(value),
        }
    }
}

/// What one key or attribute holds, as a subset of `T⊥`.
///
/// `absent` is the `⊥`: whether it is allowed to be missing. An optional field
/// carries it, a required one does not, and one that must *not* exist carries it
/// with an empty type.
///
/// One type for a record's attributes and a map's labels both: the two ask the
/// same question of a name -- what it holds, and whether it has to be there --
/// and the paper's `T⊥` is that question with a name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Field<G> {
    /// The type the value must be in.
    pub ty: Values<G>,
    /// Whether the key or attribute may be missing altogether.
    pub absent: bool,
}

impl<G: Guard> Field<G> {
    /// Any value, or none at all -- what an attribute no atom names holds.
    #[must_use]
    pub fn top() -> Field<G> {
        Field {
            ty: Values::Every,
            absent: true,
        }
    }

    /// The values in both, or `None` where a guard refuses.
    #[must_use]
    pub fn meet(&self, other: &Field<G>) -> Option<Field<G>> {
        Some(Field {
            ty: self.ty.meet(&other.ty)?,
            absent: self.absent && other.absent,
        })
    }

    /// The values in either, or `None` where a guard refuses. Missing on either
    /// side is missing in the union.
    #[must_use]
    pub fn join(&self, other: &Field<G>) -> Option<Field<G>> {
        Some(Field {
            ty: self.ty.join(&other.ty)?,
            absent: self.absent || other.absent,
        })
    }

    /// The rest of `T⊥`, which flips the extra element along with the type.
    #[must_use]
    pub fn complement(&self) -> Field<G> {
        Field {
            ty: self.ty.complement(),
            absent: !self.absent,
        }
    }

    /// What is known about something satisfying this field. Being allowed to be
    /// missing settles it whatever the type says.
    #[must_use]
    pub fn emptiness(&self) -> Verdict {
        if self.absent {
            Verdict::Inhabited
        } else {
            self.ty.emptiness()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Field, Values};
    use crate::descr::integers::IntSet;

    fn only(set: IntSet) -> Values<IntSet> {
        Values::Only(set)
    }

    /// The universe absorbs a join and is the unit of a meet, which is what
    /// carrying it beside the guard is for.
    #[test]
    fn the_universe_absorbs_a_join_and_units_a_meet() {
        let some = only(IntSet::just(1));
        assert_eq!(Values::Every.join(&some), Some(Values::Every));
        assert_eq!(some.join(&Values::Every), Some(Values::Every));
        assert_eq!(Values::Every.meet(&some), Some(some.clone()));
        assert_eq!(some.meet(&Values::Every), Some(some.clone()));
        assert_eq!(
            some.join(&only(IntSet::just(2))),
            Some(only(
                IntSet::just(1)
                    .union(&IntSet::just(2))
                    .expect("two points share a period of one"),
            ))
        );
    }

    /// A join of two fields is missing where *either* is, and a meet only where
    /// both are.
    ///
    /// `absent` is the `⊥` of `T⊥`, so it joins and meets as the extra element
    /// it is: a key one side allows to be missing is a key the union allows to
    /// be missing, and a key the meet allows to be missing is one both did.
    #[test]
    fn the_extra_element_joins_and_meets_as_itself() {
        let required = Field {
            ty: only(IntSet::just(1)),
            absent: false,
        };
        let optional = Field {
            ty: only(IntSet::just(2)),
            absent: true,
        };
        let joined = required.join(&optional).expect("two fields join");
        assert!(joined.absent, "either side missing makes the union missing");
        let met = required.meet(&optional).expect("two fields meet");
        assert!(!met.absent, "and the meet only where both allowed it");
        assert!(
            optional
                .join(&optional)
                .expect("a field joins itself")
                .absent
        );
        assert!(
            !required
                .join(&required)
                .expect("a field joins itself")
                .absent
        );
    }
}
