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
