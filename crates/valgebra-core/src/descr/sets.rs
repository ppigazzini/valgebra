//! Sets of sets, as a union of powerset lines.
//!
//! `set[T]` denotes the *powerset* of `T`: every set whose members all lie in
//! `T`. That one observation decides the kind, because a powerset is not closed
//! under two of the three operations. Meets are: `P(A) ∧ P(B) = P(A ∧ B)`, since
//! a set whose members are in both is a set of the meet. Joins and complements
//! are not -- `P(A) ∨ P(B)` holds the sets drawn wholly from `A` and those drawn
//! wholly from `B`, and no single powerset is that -- so the representation is a
//! union of **lines**, each one powerset minus finitely many others.
//!
//! Emptiness of a line is where the kind pays for itself. A set in
//! `P(T) ∧ ⋀ⱼ ¬P(Sⱼ)` is a subset of `T` that escapes every `Sⱼ`, and escaping
//! `Sⱼ` means holding a member outside it. Those members can be chosen
//! independently, one per `j`, and collected into a single set -- so the line is
//! inhabited exactly when every `T ∧ ¬Sⱼ` is. Contrapositively:
//!
//! > a line is empty exactly when some `Sⱼ` covers `T`.
//!
//! The empty set is the reason the rule reads that way and not the other. `∅` is
//! a member of every powerset, so `P(T)` is never empty however empty `T` is:
//! `set[nothing]` holds exactly one value, and it is not `nothing`. A line with
//! no subtracted powerset is therefore always inhabited, which the rule gives
//! for free -- there is no `j` to find.

use super::symbolic::Guard;

/// The most lines a union may hold.
///
/// A complement multiplies the lines, for the reason a product multiplies
/// states: the complement of a union is an intersection of complements, and each
/// one is itself a union. The bound is a limit of the representation rather than
/// an approximation -- past it there is no sound union to substitute, so the
/// operation refuses.
pub const MAX_LINES: usize = 256;

/// A set of values, with a name for the top.
///
/// A [`Guard`] has none of its own -- a descriptor that named its own universe
/// would have to build itself -- so it is carried beside the guard here, the way
/// the sequence automaton carries its else edge.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Values<G> {
    /// Every value.
    Every,
    /// The values one guard holds.
    Only(G),
}

impl<G: Guard> Values<G> {
    /// The values in both, or `None` where a guard refuses.
    fn meet(&self, other: &Values<G>) -> Option<Values<G>> {
        match (self, other) {
            (Values::Every, kept) | (kept, Values::Every) => Some(kept.clone()),
            (Values::Only(a), Values::Only(b)) => Some(Values::Only(a.meet(b)?)),
        }
    }

    /// Whether every value in `inner` is in this one, or `None` where a guard
    /// refuses.
    ///
    /// Asked as `inner ∧ ¬self = ∅`, which is the one shape a Boolean algebra
    /// answers without an order of its own.
    fn covers(&self, inner: &Values<G>) -> Option<bool> {
        match (self, inner) {
            (Values::Every, _) => Some(true),
            (Values::Only(outer), Values::Every) => Some(outer.complement().is_empty()),
            (Values::Only(outer), Values::Only(inner)) => {
                Some(inner.meet(&outer.complement())?.is_empty())
            }
        }
    }

    /// Whether `value` is one of these.
    fn holds(&self, value: &G::Value) -> bool {
        match self {
            Values::Every => true,
            Values::Only(guard) => guard.holds(value),
        }
    }
}

/// One powerset minus finitely many others.
///
/// `elements` is the set every member lies in and `minus` the powersets this
/// line excludes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Line<G> {
    elements: Values<G>,
    minus: Vec<Values<G>>,
}

impl<G: Guard> Line<G> {
    /// Whether no set satisfies this line: some subtracted powerset covers it.
    ///
    /// A refusal reads as *not* empty, which is the safe direction: a line kept
    /// is a value set that may be inhabited, and one dropped is a claim that it
    /// is not.
    fn is_empty(&self) -> bool {
        self.minus
            .iter()
            .any(|excluded| excluded.covers(&self.elements) == Some(true))
    }

    /// Whether the set whose members are `members` satisfies this line.
    fn holds(&self, members: &[G::Value]) -> bool {
        members.iter().all(|value| self.elements.holds(value))
            && self
                .minus
                .iter()
                .all(|excluded| members.iter().any(|value| !excluded.holds(value)))
    }

    /// The same line with its subtractions in canonical shape.
    ///
    /// Two steps, each removing a way to write one line twice. A subtracted
    /// powerset only matters where it meets `elements`, so it is cut down to
    /// that; and one that another covers adds nothing, since excluding the
    /// larger already excludes the smaller. What is left is an antichain, kept
    /// in order so two ways of writing it compare equal.
    fn tidy(mut self) -> Option<Line<G>> {
        for excluded in &mut self.minus {
            *excluded = excluded.meet(&self.elements)?;
        }
        let mut kept: Vec<Values<G>> = Vec::with_capacity(self.minus.len());
        for excluded in self.minus {
            // Already said by something kept -- including by an equal entry,
            // which is how a pair that covers each other keeps one rather than
            // losing both.
            if kept
                .iter()
                .any(|other| other.covers(&excluded) == Some(true))
            {
                continue;
            }
            kept.retain(|other| excluded.covers(other) != Some(true));
            kept.push(excluded);
        }
        kept.sort();
        self.minus = kept;
        Some(self)
    }
}

/// The lines a union of powerset lines complements into, or `None` past
/// [`MAX_LINES`].
///
/// The complement of a union is the intersection of the complements, and one
/// line's complement is itself a union: a set fails `P(T) ∧ ⋀ⱼ ¬P(Sⱼ)` by
/// escaping `T`, or by falling inside one of the `Sⱼ` after all.
fn complement_lines<G: Guard>(lines: &[Line<G>]) -> Option<Vec<Line<G>>> {
    let mut whole = SetLattice::all_lines();
    for line in lines {
        let mut alternatives = vec![Line {
            elements: Values::Every,
            minus: vec![line.elements.clone()],
        }];
        alternatives.extend(line.minus.iter().map(|excluded| Line {
            elements: excluded.clone(),
            minus: Vec::new(),
        }));
        whole = product(&whole, &alternatives)?;
    }
    Some(whole)
}

/// The lines of a meet, which is a meet of every pair: a set in both is drawn
/// wholly from both, so the elements meet and the subtractions collect.
fn product<G: Guard>(left: &[Line<G>], right: &[Line<G>]) -> Option<Vec<Line<G>>> {
    let mut lines = Vec::new();
    for mine in left {
        for theirs in right {
            if lines.len() >= MAX_LINES {
                return None;
            }
            let mut minus = mine.minus.clone();
            minus.extend(theirs.minus.iter().cloned());
            lines.push(Line {
                elements: mine.elements.meet(&theirs.elements)?,
                minus,
            });
        }
    }
    tidy(lines)
}

/// Drop the lines that hold nothing, put the rest in order, and refuse a union
/// past the bound.
fn tidy<G: Guard>(lines: Vec<Line<G>>) -> Option<Vec<Line<G>>> {
    let mut kept: Vec<Line<G>> = Vec::with_capacity(lines.len());
    for line in lines {
        let line = line.tidy()?;
        if !line.is_empty() && !kept.contains(&line) {
            kept.push(line);
        }
    }
    if kept.len() > MAX_LINES {
        return None;
    }
    kept.sort();
    Some(kept)
}

/// A set of sets, held as a union of powerset lines and a polarity.
///
/// The polarity is what keeps `complement` total, which the [`Guard`] the
/// sequence automaton reads its letters through requires of it. Complementing a
/// union of lines is a *product* -- an intersection of complements, each itself
/// a union -- so doing it eagerly could pass the bound and have nowhere sound to
/// go. Flipping a flag cannot, and the product is paid for later by the
/// operation that needs the lines, where a refusal is already allowed. The byte
/// automaton keeps complement total the same way, by flipping its accepting
/// states rather than rebuilding.
///
/// **Not canonical, unlike the other components.** Two unions can hold the same
/// sets and stay unequal: `P(A ∪ B)` is also the union of `P(A)`, `P(B)` and the
/// line that subtracts both, and recognising that costs a search for coverings
/// this does not run. So the laws here are checked against the *sets* rather
/// than by equality of the forms -- the same weakening the sequence automaton
/// takes, and for a reason of the same kind.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SetLattice<G: Guard> {
    lines: Vec<Line<G>>,
    /// Whether the lines are the sets held or the sets *not* held.
    negated: bool,
}

impl<G: Guard> SetLattice<G> {
    /// The one line every set satisfies.
    fn all_lines() -> Vec<Line<G>> {
        vec![Line {
            elements: Values::Every,
            minus: Vec::new(),
        }]
    }

    /// No set at all -- not even the empty one.
    #[must_use]
    pub fn empty() -> SetLattice<G> {
        SetLattice {
            lines: Vec::new(),
            negated: false,
        }
    }

    /// Every set: the one line that subtracts nothing and bounds nothing.
    #[must_use]
    pub fn all() -> SetLattice<G> {
        SetLattice {
            lines: SetLattice::all_lines(),
            negated: false,
        }
    }

    /// The sets whose members all lie in `elements`.
    #[must_use]
    pub fn of(elements: G) -> SetLattice<G> {
        SetLattice {
            lines: vec![Line {
                elements: Values::Only(elements),
                minus: Vec::new(),
            }],
            negated: false,
        }
    }

    /// The lines of the sets this holds, complementing a negated form.
    fn positive(&self) -> Option<Vec<Line<G>>> {
        if self.negated {
            complement_lines(&self.lines)
        } else {
            Some(self.lines.clone())
        }
    }

    /// Whether this holds no set.
    ///
    /// The empty set inhabits every powerset, so a line with nothing subtracted
    /// is never empty however empty its elements. A negated form has to be
    /// expanded first, and a refusal there reads as *not* empty -- the safe
    /// direction, since claiming emptiness is the claim that can be wrong.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positive()
            .is_some_and(|lines| lines.iter().all(Line::is_empty))
    }

    /// Whether the set whose members are `members` is held.
    #[must_use]
    pub fn holds(&self, members: &[G::Value]) -> bool {
        self.lines.iter().any(|line| line.holds(members)) != self.negated
    }

    /// The sets in either, or `None` past [`MAX_LINES`].
    #[must_use]
    pub fn union(&self, other: &SetLattice<G>) -> Option<SetLattice<G>> {
        let mut lines = self.positive()?;
        lines.extend(other.positive()?);
        Some(SetLattice {
            lines: tidy(lines)?,
            negated: false,
        })
    }

    /// The sets in both, or `None` past [`MAX_LINES`] or where a guard refuses.
    #[must_use]
    pub fn intersect(&self, other: &SetLattice<G>) -> Option<SetLattice<G>> {
        Some(SetLattice {
            lines: product(&self.positive()?, &other.positive()?)?,
            negated: false,
        })
    }

    /// The sets this does not hold.
    ///
    /// Total, which is what the [`Guard`] contract asks and what keeps a
    /// descriptor complementable. The lines are rebuilt where the product fits,
    /// so the common forms stay comparable -- complementing `every set` gives
    /// back exactly `no set` rather than a second spelling of it -- and the
    /// polarity carries the rest, where there is no bounded union to rebuild
    /// into.
    #[must_use]
    pub fn complement(&self) -> SetLattice<G> {
        let flipped = SetLattice {
            lines: self.lines.clone(),
            negated: !self.negated,
        };
        match flipped.positive() {
            Some(lines) => SetLattice {
                lines,
                negated: false,
            },
            None => flipped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_LINES, SetLattice};
    use crate::descr::integers::IntSet;
    use proptest::prelude::*;

    /// The member sets a law is checked over.
    ///
    /// Small sets of small integers, plus the empty one -- which carries more
    /// weight here than anywhere else, being the value every powerset holds and
    /// the reason `set[nothing]` is not `nothing`.
    const SETS: [&[i64]; 8] = [&[], &[0], &[1], &[2], &[0, 1], &[1, 2], &[0, 1, 2], &[7]];

    fn same(a: &SetLattice<IntSet>, b: &SetLattice<IntSet>) -> bool {
        SETS.iter()
            .all(|members| a.holds(members) == b.holds(members))
    }

    /// Lattices over the integer sets whose own laws are already held.
    fn lattice() -> impl Strategy<Value = SetLattice<IntSet>> {
        let leaf = prop_oneof![
            Just(SetLattice::empty()),
            Just(SetLattice::all()),
            Just(SetLattice::of(IntSet::empty())),
            (-2i64..=2).prop_map(|n| SetLattice::of(IntSet::just(n))),
            (-2i64..=2).prop_map(|lo| SetLattice::of(IntSet::between(Some(lo), None))),
            (1i64..=3).prop_map(|step| {
                SetLattice::of(IntSet::multiple_of(step).expect("a small step"))
            }),
        ];
        leaf.prop_recursive(3, 12, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.union(&b).unwrap_or_else(SetLattice::all)),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(SetLattice::empty)),
                inner.prop_map(|a| a.complement()),
            ]
        })
    }

    proptest! {
        // A bounded shrink, so a broken invariant cannot turn a caught mutation
        // into a run that outlasts a sweep.
        #![proptest_config(ProptestConfig {
            max_shrink_time: 2_000,
            ..ProptestConfig::default()
        })]

        /// The Boolean algebra, checked against the sets rather than by equality
        /// of the forms, which a union of lines does not make canonical.
        #[test]
        fn the_lattice_laws_hold_of_the_sets(a in lattice(), b in lattice(), c in lattice()) {
            let (join, meet) = (
                |x: &SetLattice<IntSet>, y: &SetLattice<IntSet>| x.union(y),
                |x: &SetLattice<IntSet>, y: &SetLattice<IntSet>| x.intersect(y),
            );
            if let (Some(ab), Some(ba)) = (join(&a, &b), join(&b, &a)) {
                prop_assert!(same(&ab, &ba), "join commutes");
            }
            if let (Some(ab), Some(ba)) = (meet(&a, &b), meet(&b, &a)) {
                prop_assert!(same(&ab, &ba), "meet commutes");
            }
            if let (Some(bc), Some(ab)) = (join(&b, &c), join(&a, &b))
                && let (Some(left), Some(right)) = (join(&a, &bc), join(&ab, &c))
            {
                prop_assert!(same(&left, &right), "join associates");
            }
            if let (Some(bc), Some(ac)) = (join(&b, &c), meet(&a, &c))
                && let (Some(ab), Some(left)) = (meet(&a, &b), meet(&a, &bc))
                && let Some(right) = ab.union(&ac)
            {
                prop_assert!(same(&left, &right), "meet distributes over join");
            }
        }

        /// The complement laws, and De Morgan both ways.
        #[test]
        fn the_complement_laws_hold_of_the_sets(a in lattice(), b in lattice()) {
            let not_a = a.complement();
            {
                let not_a = &not_a;
                if let Some(met) = a.intersect(not_a) {
                    prop_assert!(met.is_empty(), "a set is in one of the two");
                }
                if let Some(joined) = a.union(not_a) {
                    prop_assert!(same(&joined, &SetLattice::all()), "and in one of them");
                }
                prop_assert!(same(&not_a.complement(), &a), "twice is nothing");
            }
            {
                let (not_a, not_b) = (&not_a, &b.complement());
                if let (Some(joined), Some(met)) = (a.union(&b), not_a.intersect(not_b)) {
                    prop_assert!(same(&joined.complement(), &met), "de Morgan one way");
                }
                if let (Some(met), Some(joined)) = (a.intersect(&b), not_a.union(not_b)) {
                    prop_assert!(same(&met.complement(), &joined), "and the other");
                }
            }
        }

        /// Emptiness is a decision about the sets, not about the form.
        #[test]
        fn emptiness_agrees_with_the_sets(a in lattice()) {
            if a.is_empty() {
                prop_assert!(
                    SETS.iter().all(|members| !a.holds(members)),
                    "an empty lattice holds no set"
                );
            }
        }

        /// A meet of two powersets is the powerset of the meet, which is the one
        /// operation the kind is closed under.
        #[test]
        fn a_meet_of_powersets_is_the_powerset_of_the_meet(a in -2i64..=2, b in -2i64..=2) {
            let (x, y) = (IntSet::between(Some(a), None), IntSet::between(None, Some(b)));
            let met = SetLattice::of(x.clone())
                .intersect(&SetLattice::of(y.clone()))
                .expect("two small powersets");
            prop_assert!(same(&met, &SetLattice::of(x.intersect(&y))));
        }
    }

    /// Two powersets over disjoint elements meet in the empty set alone.
    ///
    /// The distinction the kind exists to make: `set[int] & set[str]` is not
    /// empty and is not `set[int]` either -- it is `set[nothing]`, whose one
    /// value is the empty set.
    #[test]
    fn disjoint_powersets_meet_in_the_empty_set_alone() {
        let evens = IntSet::multiple_of(2).expect("a small step");
        let odds = evens.complement();
        let met = SetLattice::of(evens)
            .intersect(&SetLattice::of(odds))
            .expect("two small powersets");

        assert!(!met.is_empty(), "the empty set is a member of both");
        assert!(met.holds(&[]));
        assert!(!met.holds(&[2]) && !met.holds(&[3]));
        assert!(same(&met, &SetLattice::of(IntSet::empty())));
    }

    /// The powerset of nothing is not nothing: it holds the empty set.
    #[test]
    fn the_powerset_of_nothing_holds_the_empty_set() {
        let none = SetLattice::of(IntSet::empty());

        assert!(!none.is_empty());
        assert!(none.holds(&[]));
        assert!(!none.holds(&[0]));
        assert!(SetLattice::<IntSet>::empty().is_empty());
        assert!(!SetLattice::<IntSet>::empty().holds(&[]));
    }

    /// A line is empty exactly when a subtracted powerset covers it, which is
    /// the rule the whole kind rests on.
    #[test]
    fn a_line_is_empty_when_a_subtraction_covers_it() {
        let small = SetLattice::of(IntSet::just(1));
        let wide = SetLattice::of(IntSet::between(Some(0), Some(9)));

        // `P({1}) ∧ ¬P(0..=9)` is empty, because every subset of `{1}` is a
        // subset of `0..=9`.
        let covered = small
            .intersect(&wide.complement())
            .expect("two small lattices");
        assert!(covered.is_empty());

        // The other way round it is not: `{0}` is a subset of `0..=9` and not
        // of `{1}`.
        let escaping = wide
            .intersect(&small.complement())
            .expect("two small lattices");
        assert!(!escaping.is_empty());
        assert!(escaping.holds(&[0]));
        assert!(
            !escaping.holds(&[]),
            "the empty set is a subset of every set"
        );
    }

    /// A union past the bound refuses rather than holding a form it cannot.
    #[test]
    fn a_union_past_the_bound_refuses() {
        let mut wide = SetLattice::of(IntSet::just(0));
        for n in 1..i64::try_from(MAX_LINES).unwrap_or(i64::MAX) {
            wide = wide
                .union(&SetLattice::of(IntSet::just(n)))
                .expect("inside the bound");
        }
        assert!(wide.union(&SetLattice::of(IntSet::just(-1))).is_none());
    }
}
