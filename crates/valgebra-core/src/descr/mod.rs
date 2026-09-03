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
pub mod regular;

use crate::decision::Kind;
use floats::FloatSet;
use integers::IntSet;
use regular::{Alphabet, RegularSet};

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
    /// The words this descriptor admits, as a regular language.
    ///
    /// Serves `str` and `bytes` both: a word is a byte string either way, and
    /// which alphabet a *pattern* is read over is settled where the language is
    /// built rather than carried here.
    Words(RegularSet),
}

impl Component {
    /// Every value of the kind.
    fn top(kind: Kind) -> Component {
        match kind {
            Kind::Bool => Component::Booleans(BoolSet::BOTH),
            Kind::Int => Component::Integers(IntSet::all()),
            Kind::Float => Component::Floats(FloatSet::all()),
            Kind::Str | Kind::Bytes => Component::Words(RegularSet::all()),
            _ => Component::Coarse(true),
        }
    }

    /// No value of the kind.
    fn bottom(kind: Kind) -> Component {
        match kind {
            Kind::Bool => Component::Booleans(BoolSet::EMPTY),
            Kind::Int => Component::Integers(IntSet::empty()),
            Kind::Float => Component::Floats(FloatSet::empty()),
            Kind::Str | Kind::Bytes => Component::Words(RegularSet::empty()),
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
            Component::Words(set) => set.is_empty(),
        }
    }

    /// The three operations, each on two components of the *same* kind.
    ///
    /// Mixing representations is a bug in the caller, not a case to handle: a
    /// descriptor holds one component per kind and combines them positionally,
    /// so both sides of every call are that kind's representation. The mismatch
    /// arm keeps the crate free of a panic across the boundary and is asserted
    /// unreachable in debug.
    fn combine(&self, other: &Component, op: Op) -> Option<Component> {
        Some(match (self, other) {
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
            (Component::Words(a), Component::Words(b)) => {
                // A language operation can pass the automaton bound, and there
                // is no sound set to substitute -- one too wide is complemented
                // into one too narrow. The whole descriptor becomes unbuildable
                // rather than quietly wrong, which is what the `Option` on the
                // three operations carries up.
                let combined = match op {
                    Op::Union => a.union(b),
                    Op::Intersect => a.intersect(b),
                };
                Component::Words(combined?)
            }
            (mine, theirs) => {
                debug_assert!(false, "combining {mine:?} with {theirs:?} of another kind");
                mine.clone()
            }
        })
    }

    /// Every value of the kind this component does not admit.
    fn complement(&self) -> Component {
        match self {
            Component::Coarse(present) => Component::Coarse(!present),
            Component::Booleans(set) => Component::Booleans(set.complement()),
            Component::Integers(set) => Component::Integers(set.complement()),
            Component::Floats(set) => Component::Floats(set.complement()),
            Component::Words(set) => Component::Words(set.complement()),
        }
    }
}

/// Which alphabet a word kind's patterns are read over, or `None` for a kind
/// that has no words.
///
/// The one place the two word kinds differ: `str` patterns are Unicode, so `.`
/// is a code point and a length bound counts code points; `bytes` patterns are
/// not, so both count bytes.
fn alphabet_of(kind: Kind) -> Option<Alphabet> {
    match kind {
        Kind::Str => Some(Alphabet::Text),
        Kind::Bytes => Some(Alphabet::Bytes),
        _ => None,
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

    /// The words one pattern matches whole, for a word kind.
    ///
    /// `None` where the pattern does not build, where its automaton is past the
    /// bound, or where the kind is not a word kind: a pattern over a kind that
    /// has no words is a caller error rather than a set.
    #[must_use]
    pub fn pattern(pattern: &str, kind: Kind) -> Option<Descr> {
        let language = RegularSet::pattern(pattern, alphabet_of(kind)?)?;
        let mut descr = Descr::nothing();
        descr.set(kind, Component::Words(language));
        Some(descr)
    }

    /// The one-word set, for a word kind. A `str` is its UTF-8 bytes.
    #[must_use]
    pub fn word(word: &[u8], kind: Kind) -> Option<Descr> {
        alphabet_of(kind)?;
        let mut descr = Descr::nothing();
        descr.set(kind, Component::Words(RegularSet::word(word)));
        Some(descr)
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

    /// Every value in either set, or `None` where a component cannot hold the
    /// result.
    ///
    /// Fallible because one component is: a regular language has an automaton
    /// bound, and past it there is no sound set to substitute -- one too wide is
    /// complemented into one too narrow. The refusal reaches the caller rather
    /// than being absorbed into a descriptor that is quietly wrong.
    #[must_use]
    pub fn union(&self, other: &Descr) -> Option<Descr> {
        self.zip(other, Op::Union)
    }

    /// Every value in both sets, or `None` where a component cannot hold the
    /// result.
    #[must_use]
    pub fn intersect(&self, other: &Descr) -> Option<Descr> {
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

    fn zip(&self, other: &Descr, op: Op) -> Option<Descr> {
        let mut kinds = self.kinds.clone();
        for (slot, theirs) in kinds.iter_mut().zip(&other.kinds) {
            *slot = slot.combine(theirs, op)?;
        }
        Some(Descr {
            kinds,
            other: self.other.combine(&other.other, op)?,
        })
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
        match component {
            Component::Coarse(present) => *present,
            Component::Booleans(set) => value.boolean.is_some_and(|b| set.holds(b)),
            Component::Integers(set) => value.integer.is_some_and(|i| set.holds(i)),
            Component::Floats(set) => value.float.is_some_and(|f| set.holds(f)),
            Component::Words(set) => value.word.is_some_and(|w| set.holds(w)),
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
    /// Which word, where the kind is [`Kind::Str`] or [`Kind::Bytes`]. A `str`
    /// is its UTF-8 bytes, which is the same alphabet the language is over.
    pub word: Option<&'static [u8]>,
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
            word: None,
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
            word: None,
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
            word: None,
        }
    }

    /// One word, for a word kind. A `str` is its UTF-8 bytes, which is the
    /// alphabet the language is over.
    #[must_use]
    pub const fn word(word: &'static [u8], kind: Kind) -> Value {
        Value {
            kind: Some(kind),
            boolean: None,
            integer: None,
            float: None,
            word: Some(word),
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
            word: None,
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
            word: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BoolSet, Component, Descr, Value};
    use crate::decision::Kind;
    use proptest::prelude::*;
    use std::sync::LazyLock;

    /// Every value the descriptor can currently tell apart.
    ///
    /// One per coarse kind, both booleans, and one of no listed kind. A law is
    /// checked by asking every descriptor about every one of these, which is
    /// what makes "these two sets are equal" a statement about *values* rather
    /// than about the two representations agreeing with each other.
    fn universe() -> Vec<Value> {
        let mut values: Vec<Value> = Kind::ALL
            .iter()
            .filter(|kind| {
                !matches!(
                    kind,
                    Kind::Bool | Kind::Int | Kind::Float | Kind::Str | Kind::Bytes
                )
            })
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
        // Words over the alphabet the generated patterns are written in, plus
        // one non-ASCII, for both word kinds.
        for kind in [Kind::Str, Kind::Bytes] {
            for word in [
                b"".as_slice(),
                b"a",
                b"b",
                b"c",
                b"ab",
                b"ba",
                b"aba",
                "\u{e9}".as_bytes(),
            ] {
                values.push(Value::word(word, kind));
            }
        }
        values.push(Value::other());
        values
    }

    /// Whether two descriptors agree about every value in the universe.
    ///
    /// The direction enumeration can support. Equality is stronger: two regular
    /// languages agree exactly when they agree on every word shorter than the
    /// product of their state counts, which no universe can list -- so the full
    /// canonicity claim is held in each component's own module, where the word
    /// component checks it against the emptiness decision instead.
    fn agree_on_values(a: &Descr, b: &Descr) -> bool {
        universe().into_iter().all(|v| a.admits(v) == b.admits(v))
    }

    /// The word descriptors the generator draws from, built once.
    ///
    /// A pattern's automaton is determinised and minimised, which is far more
    /// work than drawing a number: built per draw it dominates the suite, and
    /// proptest's shrinking -- thousands of draws over one failure -- turns a
    /// caught mutation into a run that does not finish.
    static WORD_SETS: LazyLock<Vec<Descr>> = LazyLock::new(|| {
        ["a", "b", "ab?", "[ab]+"]
            .iter()
            .filter_map(|pattern| Descr::pattern(pattern, Kind::Str))
            .collect()
    });

    /// A generator over the descriptors the constructors can build, combined by
    /// the three operations to a small depth.
    fn descr() -> impl Strategy<Value = Descr> {
        let leaf = prop_oneof![
            Just(Descr::nothing()),
            Just(Descr::anything()),
            (0..Kind::ALL.len())
                .prop_map(|i| Descr::of_kind(Kind::ALL.get(i).copied().unwrap_or(Kind::Int))),
            proptest::bool::ANY.prop_map(Descr::boolean),
            (-9i64..=9).prop_map(Descr::integer),
            (1i64..=5).prop_map(|step| Descr::multiple_of(step).expect("a small step")),
            prop_oneof![Just(-1.0f64), Just(0.0), Just(1.0), Just(f64::NAN)].prop_map(Descr::float),
            (0..WORD_SETS.len())
                .prop_map(|i| { WORD_SETS.get(i).cloned().unwrap_or_else(Descr::nothing) }),
        ];
        leaf.prop_recursive(3, 16, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.union(&b).unwrap_or_else(Descr::anything)),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(Descr::nothing)),
                inner.prop_map(|a| a.complement()),
            ]
        })
    }

    proptest! {
        // Fewer cases than the default here, because a word component's
        // operations are automaton products: the default count spends most of
        // the suite's time on them, and a failure's shrinking -- thousands of
        // draws over one counterexample -- takes longer than a mutation sweep
        // waits, which turns a caught mutation into a run that does not finish.
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The Boolean algebra, checked by equality of the canonical forms.
        ///
        /// One component per kind, each canonical for its representation, so
        /// equality is equality of the sets and a law is checked at full
        /// strength rather than over whatever values a universe can list.
        #[test]
        fn the_lattice_laws_hold_of_the_descriptors(a in descr(), b in descr(), c in descr()) {
            prop_assert_eq!(a.union(&b), b.union(&a));
            prop_assert_eq!(a.intersect(&b), b.intersect(&a));
            prop_assert_eq!(
                a.union(&b).and_then(|ab| ab.union(&c)),
                b.union(&c).and_then(|bc| a.union(&bc))
            );
            prop_assert_eq!(
                a.intersect(&b).and_then(|ab| ab.intersect(&c)),
                b.intersect(&c).and_then(|bc| a.intersect(&bc))
            );
            prop_assert_eq!(a.union(&a), Some(a.clone()));
            prop_assert_eq!(a.intersect(&a), Some(a.clone()));
            // Absorption and distributivity, which the structural simplifier
            // cannot state because it does not apply them.
            if let Some(met) = a.intersect(&b) {
                prop_assert_eq!(a.union(&met), Some(a.clone()));
            }
            if let Some(joined) = a.union(&b) {
                prop_assert_eq!(a.intersect(&joined), Some(a.clone()));
            }
            if let (Some(left), Some(right)) = (
                b.union(&c).and_then(|bc| a.intersect(&bc)),
                a.intersect(&b).and_then(|ab| {
                    a.intersect(&c).and_then(|ac| ab.union(&ac))
                }),
            ) {
                prop_assert_eq!(left, right);
            }
        }

        /// The complement laws, and the two the structural procedure declines.
        #[test]
        fn the_complement_laws_hold_of_the_descriptors(a in descr(), b in descr()) {
            prop_assert!(
                a.intersect(&a.complement())
                    .is_some_and(|met| met.is_empty())
            );
            prop_assert_eq!(a.union(&a.complement()), Some(Descr::anything()));
            prop_assert_eq!(&a.complement().complement(), &a);
            // De Morgan, both ways.
            prop_assert_eq!(
                a.union(&b).map(|u| u.complement()),
                a.complement().intersect(&b.complement())
            );
            prop_assert_eq!(
                a.intersect(&b).map(|m| m.complement()),
                a.complement().union(&b.complement())
            );
        }

        /// Two equal descriptors agree about every value.
        #[test]
        fn equal_descriptors_agree_about_every_value(a in descr(), b in descr()) {
            if a == b {
                prop_assert!(agree_on_values(&a, &b));
            }
        }

        /// An empty descriptor admits no value, and one that admits a value of
        /// the universe is not empty.
        #[test]
        fn emptiness_agrees_with_the_values(a in descr()) {
            if a.is_empty() {
                prop_assert!(universe().into_iter().all(|v| !a.admits(v)));
            } else if universe().into_iter().any(|v| a.admits(v)) {
                prop_assert!(!a.is_empty());
            }
        }

        /// A complement saturates the kinds it does not mention.
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
        assert_eq!(both, Some(Descr::of_kind(Kind::Bool)));
        assert!(Descr::boolean(true).admits(Value::boolean(true)));
        assert!(!Descr::boolean(true).admits(Value::boolean(false)));
        // And the complement of one singleton, inside the kind, is the other.
        let not_true = Descr::boolean(true).complement();
        assert!(not_true.admits(Value::boolean(false)));
        assert!(!not_true.admits(Value::boolean(true)));
        // ... while still holding every value of every other kind.
        assert!(not_true.admits(Value::integer(0)));
        assert!(not_true.admits(Value::word(b"a", Kind::Str)));
        assert!(not_true.admits(Value::other()));
    }

    /// A coarse component is all-or-nothing, and the tests must not read that as
    /// a distinction the descriptor makes.
    #[test]
    fn a_coarse_kind_admits_all_of_its_values_or_none() {
        let sets = Descr::of_kind(Kind::Set);
        assert!(sets.admits(Value::of_kind(Kind::Set)));
        assert!(!sets.admits(Value::of_kind(Kind::FrozenSet)));
        assert!(!sets.admits(Value::other()));
        assert!(!sets.is_empty());
        assert!(
            sets.intersect(&Descr::of_kind(Kind::FrozenSet))
                .is_some_and(|met| met.is_empty())
        );
        assert!(
            sets.union(&Descr::of_kind(Kind::FrozenSet))
                .is_some_and(|joined| !joined.is_empty())
        );
    }

    /// The two word kinds are exact and separate: a pattern over one says
    /// nothing about the other, which is what keeps `str` and `bytes` disjoint
    /// while sharing a representation.
    #[test]
    fn the_word_kinds_are_languages_and_stay_apart() {
        let text = Descr::pattern("ab?", Kind::Str).expect("a small pattern");
        assert!(text.admits(Value::word(b"a", Kind::Str)));
        assert!(text.admits(Value::word(b"ab", Kind::Str)));
        assert!(!text.admits(Value::word(b"b", Kind::Str)));
        // The same word as `bytes` is a different value, in a component this
        // descriptor leaves empty.
        assert!(!text.admits(Value::word(b"a", Kind::Bytes)));
        assert!(
            text.intersect(&Descr::of_kind(Kind::Bytes))
                .is_some_and(|met| met.is_empty())
        );

        // One pattern inside another, which the structural procedure declines:
        // it relates two patterns only when they are written identically.
        let narrow = Descr::pattern("a", Kind::Str).expect("a small pattern");
        assert!(
            narrow
                .intersect(&text.complement())
                .is_some_and(|met| met.is_empty())
        );
        // And a pattern whose language is one word is that word.
        assert_eq!(narrow, Descr::word(b"a", Kind::Str).expect("a word kind"));
        // The bytes kind is a word kind too, and its patterns land in its own
        // component: the same pattern over the two kinds is two disjoint sets.
        let raw = Descr::pattern("a", Kind::Bytes).expect("a small pattern");
        assert!(raw.admits(Value::word(b"a", Kind::Bytes)));
        assert!(!raw.admits(Value::word(b"a", Kind::Str)));
        assert!(raw.intersect(&narrow).is_some_and(|met| met.is_empty()));
        // And the two kinds read a pattern over different alphabets: a byte
        // class outside UTF-8 is a language for `bytes` and no language at all
        // for `str`.
        assert!(Descr::pattern(r"(?-u:\xFF)", Kind::Bytes).is_some());
        assert!(Descr::pattern(r"(?-u:\xFF)", Kind::Str).is_none());

        // A pattern over a kind with no words is a caller error, not a set.
        assert!(Descr::pattern("a", Kind::Int).is_none());
        assert!(Descr::word(b"a", Kind::List).is_none());
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
        // The evens and the odds together are the kind, which no union of
        // intervals could say.
        let odds = evens.complement().intersect(&ints).expect("a small meet");
        assert_eq!(evens.union(&odds), Some(ints.clone()));

        // Two singletons meet in nothing, and each is inside the kind.
        assert!(
            Descr::integer(1)
                .intersect(&Descr::integer(2))
                .is_some_and(|met| met.is_empty())
        );
        assert!(
            Descr::integer(1)
                .intersect(&ints)
                .is_some_and(|met| !met.is_empty())
        );
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
