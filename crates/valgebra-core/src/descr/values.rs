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
