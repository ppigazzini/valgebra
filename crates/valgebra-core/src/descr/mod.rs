//! The set-theoretic descriptor: a set of values held one component per kind.
//!
//! The structural decision procedure reads a schema's *syntax* and applies
//! inclusion rules to it. That is why it is sound but incomplete: a relation it
//! has no rule for is declined, and the shape a caller happened to write decides
//! which rules fire. The descriptor is the
//! other approach, the one Frisch, Castagna and Benzaken take (JACM 55(4), 2008)
//! and Castagna and Duboc implement (§7): give the *set* a representation closed
//! under union, intersection and complement, and decide every relation by
//! emptiness of one combination.
//!
//! The value universe is partitioned by [`Kind`], so a set of values is a set
//! per kind and nothing more. Union, intersection and complement are then
//! componentwise, which is the whole reason to partition first: no rule relates
//! a list to an int, because they live in components that never meet.
//!
//! **This is built beside the structural procedure, not in place of it.** It
//! decides nothing a caller can reach yet. Each component starts *coarse* --
//! every value of the kind, or none -- and each later commit replaces one kind's
//! component with a representation that distinguishes its values. The type says
//! which is which, so what the descriptor can and cannot see is read off it
//! rather than inferred.

pub mod floats;
pub mod integers;
pub mod interval;

use crate::decision::Kind;
use floats::FloatSet;
use integers::IntSet;

/// The two booleans, as a subset.
///
/// `bool` is a kind with exactly two values, so a *finite set* over it is exact:
/// `Literal[True]` is `{True}`, and `Literal[True] | Literal[False]` is the
/// whole kind rather than a union the procedure must recognise. A two-bit set is
/// the smallest thing closed under the three operations here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolSet(u8);

impl BoolSet {
    /// Neither boolean.
    pub const EMPTY: BoolSet = BoolSet(0);
    /// Both booleans: the whole `bool` kind.
    pub const BOTH: BoolSet = BoolSet(0b11);

    /// The singleton holding just this boolean.
    ///
    /// The bit *is* the boolean: `false` takes bit zero and `true` bit one, so
    /// the two are related in one expression rather than by a named constant
    /// each. Two constants would be two more things to keep in step, and their
    /// disjointness -- which is what makes the union of the singletons the whole
    /// kind -- would be a fact about two literals rather than about the shift.
    #[must_use]
    pub fn just(value: bool) -> BoolSet {
        BoolSet(1 << u8::from(value))
    }

    /// Whether this set holds `value`.
    #[must_use]
    pub fn holds(self, value: bool) -> bool {
        self.0 & BoolSet::just(value).0 != 0
    }

    const fn union(self, other: BoolSet) -> BoolSet {
        BoolSet(self.0 | other.0)
    }

    const fn intersect(self, other: BoolSet) -> BoolSet {
        BoolSet(self.0 & other.0)
    }

    const fn complement(self) -> BoolSet {
        BoolSet(BoolSet::BOTH.0 & !self.0)
    }

    const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The values of one kind that a [`Descr`] admits.
///
/// One variant per representation, not per kind: a kind whose values are not yet
/// distinguished carries [`Coarse`](Component::Coarse), and moving a kind to an
/// exact representation is adding a variant and the arms that go with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Component {
    /// Every value of the kind, or none.
    ///
    /// Exact for `None`, which has one value, and coarse for every other kind
    /// that still carries it: `list[int]` and `list[str]` are the same component,
    /// so the descriptor cannot yet tell them apart. Coarse is *sound* rather
    /// than wrong -- it is the honest representation of a distinction not yet
    /// made, and emptiness over it decides the kind partition and nothing finer.
    Coarse(bool),
    /// The booleans this descriptor admits.
    Booleans(BoolSet),
    /// The integers this descriptor admits.
    ///
    /// `bool` is a separate kind, so this is the integers that are not booleans
    /// -- which is what makes the two components independent. A schema that
    /// admits both spells that as a descriptor holding a component in each.
    Integers(IntSet),
    /// The floats this descriptor admits, `nan` included.
    Floats(FloatSet),
}

impl Component {
    /// Every value of the kind.
    fn top(kind: Kind) -> Component {
        match kind {
            Kind::Bool => Component::Booleans(BoolSet::BOTH),
            Kind::Int => Component::Integers(IntSet::all()),
            Kind::Float => Component::Floats(FloatSet::all()),
            _ => Component::Coarse(true),
        }
    }

    /// No value of the kind.
    fn bottom(kind: Kind) -> Component {
        match kind {
            Kind::Bool => Component::Booleans(BoolSet::EMPTY),
            Kind::Int => Component::Integers(IntSet::empty()),
            Kind::Float => Component::Floats(FloatSet::empty()),
            _ => Component::Coarse(false),
        }
    }

    /// Whether this component admits no value at all.
    fn is_empty(&self) -> bool {
        match self {
            Component::Coarse(present) => !present,
            Component::Booleans(set) => set.is_empty(),
            Component::Integers(set) => set.is_empty(),
            Component::Floats(set) => set.is_empty(),
        }
    }

    /// The three operations, each on two components of the *same* kind.
    ///
    /// Mixing representations is a bug in the caller, not a case to handle: a
    /// descriptor holds one component per kind and combines them positionally,
    /// so both sides of every call are that kind's representation. The mismatch
    /// arm keeps the crate free of a panic across the boundary and is asserted
    /// unreachable in debug.
    fn combine(&self, other: &Component, op: Op) -> Component {
        match (self, other) {
            (Component::Coarse(a), Component::Coarse(b)) => Component::Coarse(match op {
                Op::Union => *a || *b,
                Op::Intersect => *a && *b,
            }),
            (Component::Booleans(a), Component::Booleans(b)) => Component::Booleans(match op {
                Op::Union => a.union(*b),
                Op::Intersect => a.intersect(*b),
            }),
            (Component::Integers(a), Component::Integers(b)) => Component::Integers(match op {
                Op::Union => a.union(b),
                Op::Intersect => a.intersect(b),
            }),
            (Component::Floats(a), Component::Floats(b)) => Component::Floats(match op {
                Op::Union => a.union(b),
                Op::Intersect => a.intersect(b),
            }),
            (mine, theirs) => {
                debug_assert!(false, "combining {mine:?} with {theirs:?} of another kind");
                mine.clone()
            }
        }
    }

    /// Every value of the kind this component does not admit.
    fn complement(&self) -> Component {
        match self {
            Component::Coarse(present) => Component::Coarse(!present),
            Component::Booleans(set) => Component::Booleans(set.complement()),
            Component::Integers(set) => Component::Integers(set.complement()),
            Component::Floats(set) => Component::Floats(set.complement()),
        }
    }
}

/// Which way two components combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Union,
    Intersect,
}

/// A set of values, held as one component per [`Kind`] plus everything else.
///
/// **Canonical by construction.** Each component is canonical for its
/// representation, and there is exactly one component per kind, so two
/// descriptors admit the same values exactly when they are equal. Nothing needs
/// normalising afterwards, which is what makes the three operations total and
/// their laws structural rather than up-to-equivalence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descr {
    /// One component per kind, indexed by that kind's position in [`Kind::ALL`].
    kinds: [Component; Kind::ALL.len()],
    /// The values of no listed kind -- a class instance, a callable, a generator.
    ///
    /// Coarse and staying that way for now: what separates two such values is
    /// their class, and a class is something only the bindings can compare. It
    /// exists so a complement means what it says: the complement of `int` holds
    /// every non-int value, not merely the ones this partition names.
    other: Component,
}

impl Descr {
    /// The empty set.
    #[must_use]
    pub fn nothing() -> Descr {
        Descr {
            kinds: Kind::ALL.map(Component::bottom),
            other: Component::Coarse(false),
        }
    }

    /// Every value.
    #[must_use]
    pub fn anything() -> Descr {
        Descr {
            kinds: Kind::ALL.map(Component::top),
            other: Component::Coarse(true),
        }
    }

    /// Every value of `kind` and nothing else.
    #[must_use]
    pub fn of_kind(kind: Kind) -> Descr {
        let mut descr = Descr::nothing();
        descr.set(kind, Component::top(kind));
        descr
    }

    /// The singleton holding one boolean.
    #[must_use]
    pub fn boolean(value: bool) -> Descr {
        let mut descr = Descr::nothing();
        descr.set(Kind::Bool, Component::Booleans(BoolSet::just(value)));
        descr
    }

    /// The singleton holding one integer.
    #[must_use]
    pub fn integer(value: i64) -> Descr {
        let mut descr = Descr::nothing();
        descr.set(Kind::Int, Component::Integers(IntSet::just(value)));
        descr
    }

    /// The singleton holding one float, which is empty for `nan`.
    #[must_use]
    pub fn float(value: f64) -> Descr {
        let mut descr = Descr::nothing();
        descr.set(Kind::Float, Component::Floats(FloatSet::just(value)));
        descr
    }

    /// The integers that are multiples of `step`, or `None` where the integer
    /// set cannot hold that step.
    ///
    /// The refusal is carried up rather than absorbed: a descriptor that
    /// silently widened here would be complemented into one that is wrong the
    /// other way, and the caller is the one that can decide to keep the step
    /// opaque instead.
    #[must_use]
    pub fn multiple_of(step: i64) -> Option<Descr> {
        let mut descr = Descr::nothing();
        descr.set(Kind::Int, Component::Integers(IntSet::multiple_of(step)?));
        Some(descr)
    }

    fn position(kind: Kind) -> usize {
        Kind::ALL
            .iter()
            .position(|listed| *listed == kind)
            .unwrap_or(0)
    }

    fn component(&self, kind: Kind) -> &Component {
        self.kinds
            .get(Descr::position(kind))
            .unwrap_or(&Component::Coarse(false))
    }

    fn set(&mut self, kind: Kind, component: Component) {
        if let Some(slot) = self.kinds.get_mut(Descr::position(kind)) {
            *slot = component;
        }
    }

    /// Every value in either set.
    #[must_use]
    pub fn union(&self, other: &Descr) -> Descr {
        self.zip(other, Op::Union)
    }

    /// Every value in both sets.
    #[must_use]
    pub fn intersect(&self, other: &Descr) -> Descr {
        self.zip(other, Op::Intersect)
    }

    /// Every value this set does not hold.
    ///
    /// Componentwise, which is what makes it *saturate*: complementing a
    /// descriptor that holds only ints gives one holding every value of every
    /// other kind, because each of those components was empty and is now full.
    /// A representation that carried only the kinds it mentions would have to
    /// name the rest here, and would name the wrong set the day a kind is added.
    #[must_use]
    pub fn complement(&self) -> Descr {
        Descr {
            kinds: self.kinds.each_ref().map(Component::complement),
            other: self.other.complement(),
        }
    }

    fn zip(&self, other: &Descr, op: Op) -> Descr {
        let mut kinds = self.kinds.clone();
        for (slot, theirs) in kinds.iter_mut().zip(&other.kinds) {
            *slot = slot.combine(theirs, op);
        }
        Descr {
            kinds,
            other: self.other.combine(&other.other, op),
        }
    }

    /// Whether this set admits no value.
    ///
    /// Every component empty, which is the whole emptiness decision at this
    /// resolution: the kinds partition the universe, so a value the descriptor
    /// admits is a value some component admits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kinds.iter().all(Component::is_empty) && self.other.is_empty()
    }

    /// Whether this set admits `value`.
    #[must_use]
    pub fn admits(&self, value: Value) -> bool {
        let component = match value.kind {
            Some(kind) => self.component(kind),
            None => &self.other,
        };
        match (component, value.boolean, value.integer, value.float) {
            (Component::Coarse(present), _, _, _) => *present,
            (Component::Booleans(set), Some(boolean), _, _) => set.holds(boolean),
            (Component::Integers(set), _, Some(integer), _) => set.holds(integer),
            (Component::Floats(set), _, _, Some(float)) => set.holds(float),
            // A component asked about a value that carries no payload for it:
            // the caller built a value whose kind and payload disagree.
            (Component::Booleans(_) | Component::Integers(_) | Component::Floats(_), ..) => {
                debug_assert!(false, "a {component:?} component has no payload to read");
                false
            }
        }
    }
}

/// A value, at the resolution the descriptor distinguishes.
///
/// Not a Python object: the core cannot see one. It is the *questions* a
/// descriptor can currently answer about a value -- which kind it belongs to,
/// and, where the kind's component distinguishes its values, which value it is.
/// Each commit that makes a component exact widens this alongside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Value {
    /// The kind, or `None` for a value of no listed kind.
    pub kind: Option<Kind>,
    /// Which boolean, where the kind is [`Kind::Bool`].
    pub boolean: Option<bool>,
    /// Which integer, where the kind is [`Kind::Int`].
    pub integer: Option<i64>,
    /// Which float, where the kind is [`Kind::Float`].
    pub float: Option<f64>,
}

impl Value {
    /// A representative value of `kind`, for a kind whose component is coarse.
    #[must_use]
    pub const fn of_kind(kind: Kind) -> Value {
        Value {
            kind: Some(kind),
            boolean: None,
            integer: None,
            float: None,
        }
    }

    /// One of the two booleans.
    #[must_use]
    pub const fn boolean(value: bool) -> Value {
        Value {
            kind: Some(Kind::Bool),
            boolean: Some(value),
            integer: None,
            float: None,
        }
    }

    /// One integer. `bool` is a kind of its own, so this is never a boolean.
    #[must_use]
    pub const fn integer(value: i64) -> Value {
        Value {
            kind: Some(Kind::Int),
            boolean: None,
            integer: Some(value),
            float: None,
        }
    }

    /// One float, `nan` included.
    #[must_use]
    pub const fn float(value: f64) -> Value {
        Value {
            kind: Some(Kind::Float),
            boolean: None,
            integer: None,
            float: Some(value),
        }
    }

    /// A value of no listed kind: a class instance, a callable.
    #[must_use]
    pub const fn other() -> Value {
        Value {
            kind: None,
            boolean: None,
            integer: None,
            float: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BoolSet, Component, Descr, Value};
    use crate::decision::Kind;
    use proptest::prelude::*;

    /// Every value the descriptor can currently tell apart.
    ///
    /// One per coarse kind, both booleans, and one of no listed kind. A law is
    /// checked by asking every descriptor about every one of these, which is
    /// what makes "these two sets are equal" a statement about *values* rather
    /// than about the two representations agreeing with each other.
    fn universe() -> Vec<Value> {
        let mut values: Vec<Value> = Kind::ALL
            .iter()
            .filter(|kind| !matches!(kind, Kind::Bool | Kind::Int | Kind::Float))
            .map(|kind| Value::of_kind(*kind))
            .collect();
        values.push(Value::boolean(true));
        values.push(Value::boolean(false));
        // Enough integers to separate every step and bound the generator uses:
        // a window narrower than the periods would agree by accident.
        values.extend((-14i64..=14).map(Value::integer));
        // The three floats that make the kind its own case, and a point strictly
        // inside every gap the generator's endpoints leave -- the unbounded ones
        // included, where a set like `(-inf, -1.0)` is inhabited by values no
        // endpoint names.
        values.extend(
            [
                f64::NEG_INFINITY,
                -2.0,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.0,
                f64::INFINITY,
                f64::NAN,
            ]
            .map(Value::float),
        );
        values.push(Value::other());
        values
    }

    /// Whether two descriptors admit exactly the same values.
    fn same_values(a: &Descr, b: &Descr) -> bool {
        universe().into_iter().all(|v| a.admits(v) == b.admits(v))
    }

    /// A generator over the descriptors the constructors can build, combined by
    /// the three operations to a small depth.
    fn descr() -> impl Strategy<Value = Descr> {
        let leaf = prop_oneof![
            Just(Descr::nothing()),
            Just(Descr::anything()),
            (0..Kind::ALL.len())
                .prop_map(|i| Descr::of_kind(Kind::ALL.get(i).copied().unwrap_or(Kind::Int))),
            proptest::bool::ANY.prop_map(Descr::boolean),
        ];
        leaf.prop_recursive(4, 24, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a.union(&b)),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a.intersect(&b)),
                inner.prop_map(|a| a.complement()),
            ]
        })
    }

    proptest! {
        /// The Boolean algebra, checked against membership.
        ///
        /// Each law says two ways of writing a set admit the same values, so
        /// that is what is asked. Comparing the two representations instead
        /// would be a property of the canonicalisation, which is a different
        /// claim -- and the weaker one, since it holds of any representation
        /// that normalises consistently.
        #[test]
        fn the_lattice_laws_hold_of_the_values(a in descr(), b in descr(), c in descr()) {
            prop_assert!(same_values(&a.union(&b), &b.union(&a)));
            prop_assert!(same_values(&a.intersect(&b), &b.intersect(&a)));
            prop_assert!(same_values(&a.union(&b).union(&c), &a.union(&b.union(&c))));
            prop_assert!(same_values(
                &a.intersect(&b).intersect(&c),
                &a.intersect(&b.intersect(&c))
            ));
            prop_assert!(same_values(&a.union(&a), &a));
            prop_assert!(same_values(&a.intersect(&a), &a));
            // Absorption and distributivity, which the structural simplifier
            // cannot state because it does not apply them.
            prop_assert!(same_values(&a.union(&a.intersect(&b)), &a));
            prop_assert!(same_values(&a.intersect(&a.union(&b)), &a));
            prop_assert!(same_values(
                &a.intersect(&b.union(&c)),
                &a.intersect(&b).union(&a.intersect(&c))
            ));
            prop_assert!(same_values(
                &a.union(&b.intersect(&c)),
                &a.union(&b).intersect(&a.union(&c))
            ));
        }

        /// The complement laws, and the two the structural procedure declines.
        #[test]
        fn the_complement_laws_hold_of_the_values(a in descr(), b in descr()) {
            prop_assert!(a.intersect(&a.complement()).is_empty());
            prop_assert!(same_values(&a.union(&a.complement()), &Descr::anything()));
            prop_assert!(same_values(&a.complement().complement(), &a));
            // De Morgan, both ways.
            prop_assert!(same_values(
                &a.union(&b).complement(),
                &a.complement().intersect(&b.complement())
            ));
            prop_assert!(same_values(
                &a.intersect(&b).complement(),
                &a.complement().union(&b.complement())
            ));
        }

        /// Equality *is* semantic equality, which is what canonical means.
        ///
        /// One component per kind, each canonical for its representation, so two
        /// descriptors admitting the same values have nowhere left to differ.
        /// This is the property the structural IR does not have -- there, two
        /// spellings of one set are two trees -- and it is what lets a later
        /// decision be an emptiness check rather than a search for a rule.
        #[test]
        fn admitting_the_same_values_is_being_equal(a in descr(), b in descr()) {
            prop_assert_eq!(same_values(&a, &b), a == b);
        }

        /// Emptiness is a decision, not a conservative answer: a descriptor is
        /// empty exactly when no value in the universe is in it.
        #[test]
        fn emptiness_agrees_with_the_values(a in descr()) {
            let admits_none = universe().into_iter().all(|v| !a.admits(v));
            prop_assert_eq!(a.is_empty(), admits_none);
        }

        /// A complement saturates the kinds it does not mention.
        ///
        /// The property a representation that carried only its own kinds would
        /// fail: everything not in `a` is in `¬a`, including every value of
        /// every kind `a` says nothing about.
        #[test]
        fn a_complement_holds_every_value_the_set_does_not(a in descr()) {
            for value in universe() {
                prop_assert_eq!(a.admits(value), !a.complement().admits(value));
            }
        }
    }

    /// The kind list is the partition, so it must hold every kind exactly once.
    ///
    /// A `match` over `Kind` is exhaustive and this array is not, so the list is
    /// counted against the variants rather than trusted: a kind added without a
    /// component here would be a kind the descriptor silently cannot represent.
    #[test]
    fn every_kind_has_exactly_one_component() {
        let mut seen = Kind::ALL.to_vec();
        seen.sort_by_key(|kind| format!("{kind:?}"));
        seen.dedup();
        assert_eq!(seen.len(), Kind::ALL.len(), "a kind is listed twice");

        // Exhaustive by construction: the match forces a new variant to be
        // added here, and the count then forces it into the list.
        let counted = Kind::ALL
            .iter()
            .filter(|kind| {
                matches!(
                    kind,
                    Kind::NoneType
                        | Kind::Bool
                        | Kind::Int
                        | Kind::Float
                        | Kind::Str
                        | Kind::Bytes
                        | Kind::List
                        | Kind::Tuple
                        | Kind::Set
                        | Kind::FrozenSet
                        | Kind::Dict
                )
            })
            .count();
        assert_eq!(counted, Kind::ALL.len());
    }

    /// `bool` is its two values, which is what makes the union of the two
    /// singletons the whole kind rather than a shape a rule must recognise.
    #[test]
    fn the_two_booleans_are_the_bool_kind() {
        let both = Descr::boolean(true).union(&Descr::boolean(false));
        assert_eq!(both, Descr::of_kind(Kind::Bool));
        assert!(Descr::boolean(true).admits(Value::boolean(true)));
        assert!(!Descr::boolean(true).admits(Value::boolean(false)));
        // And the complement of one singleton, inside the kind, is the other.
        let not_true = Descr::boolean(true).complement();
        assert!(not_true.admits(Value::boolean(false)));
        assert!(!not_true.admits(Value::boolean(true)));
        // ... while still holding every value of every other kind.
        assert!(not_true.admits(Value::integer(0)));
        assert!(not_true.admits(Value::of_kind(Kind::Str)));
        assert!(not_true.admits(Value::other()));
    }

    /// A coarse component is all-or-nothing, and the tests must not read that as
    /// a distinction the descriptor makes.
    #[test]
    fn a_coarse_kind_admits_all_of_its_values_or_none() {
        let strs = Descr::of_kind(Kind::Str);
        assert!(strs.admits(Value::of_kind(Kind::Str)));
        assert!(!strs.admits(Value::of_kind(Kind::Bytes)));
        assert!(!strs.admits(Value::other()));
        assert!(!strs.is_empty());
        assert!(strs.intersect(&Descr::of_kind(Kind::Bytes)).is_empty());
        assert!(!strs.union(&Descr::of_kind(Kind::Bytes)).is_empty());
    }

    /// The integers are exact, so a bound conjunction that cannot hold is
    /// decided rather than declined -- and a step is a set the coarse
    /// representation had no way to express at all.
    #[test]
    fn the_integers_are_a_set_rather_than_a_kind() {
        let ints = Descr::of_kind(Kind::Int);
        assert!(ints.admits(Value::integer(7)));
        assert!(!ints.admits(Value::of_kind(Kind::Str)));
        // A boolean is its own kind, so `int` does not admit one.
        assert!(!ints.admits(Value::boolean(true)));

        let evens = Descr::multiple_of(2).expect("two is inside the bound");
        assert!(evens.admits(Value::integer(4)));
        assert!(!evens.admits(Value::integer(3)));
        assert_eq!(
            evens.union(&evens.intersect(&ints).complement().intersect(&ints)),
            ints
        );

        // Two singletons meet in nothing, and each is inside the kind.
        assert!(Descr::integer(1).intersect(&Descr::integer(2)).is_empty());
        assert!(!Descr::integer(1).intersect(&ints).is_empty());
    }

    /// The set operations on the two booleans, driven directly: the descriptor
    /// laws above exercise them through a component, and these pin the set.
    #[test]
    fn the_boolean_set_is_a_two_element_boolean_algebra() {
        assert!(BoolSet::EMPTY.is_empty());
        assert!(!BoolSet::BOTH.is_empty());
        assert!(BoolSet::BOTH.holds(true) && BoolSet::BOTH.holds(false));
        assert!(BoolSet::just(true).holds(true) && !BoolSet::just(true).holds(false));
        assert_eq!(BoolSet::just(true).complement(), BoolSet::just(false));
        assert_eq!(
            BoolSet::just(true).union(BoolSet::just(false)),
            BoolSet::BOTH
        );
        assert_eq!(
            BoolSet::just(true).intersect(BoolSet::just(false)),
            BoolSet::EMPTY
        );
        assert_eq!(BoolSet::BOTH.complement(), BoolSet::EMPTY);
    }

    /// The bottom and top of one kind, which every constructor is written from.
    #[test]
    fn a_component_is_bottom_or_top_of_its_own_kind() {
        for kind in Kind::ALL {
            assert!(Component::bottom(kind).is_empty(), "{kind:?} bottom");
            assert!(!Component::top(kind).is_empty(), "{kind:?} top");
            assert_eq!(Component::bottom(kind).complement(), Component::top(kind));
            assert_eq!(Component::top(kind).complement(), Component::bottom(kind));
        }
    }
}
