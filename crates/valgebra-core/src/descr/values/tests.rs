use super::{Field, Values};
use crate::descr::integers::IntSet;
use crate::descr::symbolic::Guard;
use crate::verdict::Verdict;
use proptest::prelude::*;

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

/// A guard that cannot always answer, so the third verdict has a source.
///
/// The component carries the top beside the guard, and a guard is a Boolean
/// algebra whose emptiness is a *proof*: a letter that cannot decide answers
/// `false` to `is_empty` and `Unknown` to `emptiness`. Held here rather than
/// borrowed from the sequence tests because a second `impl Guard` for one type
/// would not compile, and a guard that always decides cannot reach the arms
/// below.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Undecidable {
    Exact(IntSet),
    Undecided,
}

impl Guard for Undecidable {
    type Value = i64;

    fn none() -> Self {
        Undecidable::Exact(IntSet::empty())
    }
    fn meet(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Undecidable::Exact(a), Undecidable::Exact(b)) => {
                a.intersect(b).map(Undecidable::Exact)
            }
            _ => Some(Undecidable::Undecided),
        }
    }
    fn join(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Undecidable::Exact(a), Undecidable::Exact(b)) => a.union(b).map(Undecidable::Exact),
            _ => Some(Undecidable::Undecided),
        }
    }
    fn complement(&self) -> Self {
        match self {
            Undecidable::Exact(set) => Undecidable::Exact(IntSet::complement(set)),
            Undecidable::Undecided => Undecidable::Undecided,
        }
    }
    fn is_empty(&self) -> bool {
        matches!(self, Undecidable::Exact(set) if IntSet::is_empty(set))
    }
    fn emptiness(&self) -> Verdict {
        match self {
            Undecidable::Exact(set) if IntSet::is_empty(set) => Verdict::Empty,
            Undecidable::Exact(_) => Verdict::Inhabited,
            Undecidable::Undecided => Verdict::Unknown,
        }
    }
    fn holds(&self, value: &i64) -> bool {
        matches!(self, Undecidable::Exact(set) if IntSet::holds(set, *value))
    }
}

/// The integers a law is checked over: the points the guards below separate.
const POINTS: [i64; 5] = [-2, -1, 0, 1, 2];

fn same(a: &Values<IntSet>, b: &Values<IntSet>) -> bool {
    POINTS.iter().all(|n| a.holds(n) == b.holds(n))
}

/// Value sets whose own laws the integer component already holds.
fn values() -> impl Strategy<Value = Values<IntSet>> {
    let leaf = prop_oneof![
        Just(Values::Every),
        Just(Values::Only(IntSet::empty())),
        (-2i64..=2).prop_map(|n| Values::Only(IntSet::just(n))),
        (-2i64..=2).prop_map(|lo| Values::Only(IntSet::between(Some(lo), None))),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| a.join(&b).unwrap_or(Values::Every)),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.meet(&b).unwrap_or_else(Values::none)),
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

    /// The Boolean algebra, checked against the values rather than by equality
    /// of the forms -- `Every` and a guard naming every integer are one set and
    /// two forms, which is the whole reason the top is carried beside.
    #[test]
    fn the_lattice_laws_hold_of_the_values(
        a in values(), b in values(), c in values()
    ) {
        if let (Some(ab), Some(ba)) = (a.join(&b), b.join(&a)) {
            prop_assert!(same(&ab, &ba), "join commutes");
        }
        if let (Some(ab), Some(ba)) = (a.meet(&b), b.meet(&a)) {
            prop_assert!(same(&ab, &ba), "meet commutes");
        }
        if let (Some(bc), Some(ab)) = (b.join(&c), a.join(&b))
            && let (Some(left), Some(right)) = (a.join(&bc), ab.join(&c))
        {
            prop_assert!(same(&left, &right), "join associates");
        }
        if let (Some(bc), Some(ac)) = (b.join(&c), a.meet(&c))
            && let (Some(ab), Some(left)) = (a.meet(&b), a.meet(&bc))
            && let Some(right) = ab.join(&ac)
        {
            prop_assert!(same(&left, &right), "meet distributes over join");
        }
    }

    /// The complement laws, and De Morgan both ways.
    #[test]
    fn the_complement_laws_hold_of_the_values(a in values(), b in values()) {
        let not_a = a.complement();
        if let Some(met) = a.meet(&not_a) {
            prop_assert!(
                POINTS.iter().all(|n| !met.holds(n)),
                "a value is in one of the two"
            );
        }
        if let Some(joined) = a.join(&not_a) {
            prop_assert!(
                POINTS.iter().all(|n| joined.holds(n)),
                "and in one of them"
            );
        }
        prop_assert!(same(&not_a.complement(), &a), "twice is nothing");

        if let (Some(met), Some(joined)) = (a.meet(&b), a.complement().join(&b.complement())) {
            prop_assert!(same(&met.complement(), &joined), "de Morgan, one way");
        }
        if let (Some(joined), Some(met)) = (a.join(&b), a.complement().meet(&b.complement())) {
            prop_assert!(same(&joined.complement(), &met), "and the other");
        }
    }
}

/// A guard that cannot decide leaves both answers unknown, and neither is read
/// as a verdict.
///
/// The third verdict is what lets a bounded representation stay sound: a set
/// whose emptiness is unproved is not empty and is not inhabited, and a
/// coverage question over one settles neither direction. Reading either as an
/// answer keeps a line on no evidence, which is a set reported inhabited.
#[test]
fn a_guard_that_cannot_decide_leaves_the_answer_unknown() {
    let opaque: Values<Undecidable> = Values::Only(Undecidable::Undecided);
    let known: Values<Undecidable> = Values::Only(Undecidable::Exact(IntSet::just(1)));

    assert_eq!(opaque.emptiness(), Verdict::Unknown);
    assert_eq!(known.emptiness(), Verdict::Inhabited);
    assert_eq!(
        Values::<Undecidable>::none().emptiness(),
        Verdict::Empty,
        "the bottom is proved empty, which is what makes the third verdict a gap"
    );
    // The universe holds a value whatever the guards say, which is the whole
    // reason it is a variant rather than a guard.
    assert_eq!(Values::<Undecidable>::Every.emptiness(), Verdict::Inhabited);

    // And the coverage question: unknown in, neither out.
    assert_eq!(known.covers(&opaque), None);
    assert_eq!(
        Values::<Undecidable>::Every.covers(&opaque),
        Some(true),
        "the universe covers every set without asking the guard"
    );
}
