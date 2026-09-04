//! Objects described by their attributes, as a union of open record atoms.
//!
//! An **atom** names finitely many attributes and says what each one holds; every
//! attribute it does not name is unconstrained. That is Castagna's record type
//! `⟨(τ_ℓ)_{ℓ∈L}; ⊤⟩` with the default fixed at `⊤` -- the always-open reading,
//! which is the only sound one for a Python object, since an object may carry
//! attributes no schema mentions.
//!
//! A field's type lives in `T⊥`, the values extended with one more element for
//! *undefined*. That is the paper's device and it earns its place immediately:
//! optionality stops being a flag with rules of its own and becomes membership.
//! A required `int` field is `int`; an optional one is `int ∪ ⊥`; a field that
//! must be missing is `⊥` alone; and an unnamed attribute's `⊤` is `anything ∪
//! ⊥`. Meet, complement and emptiness are then the ordinary operations on `T⊥`,
//! with the extra element carried as a bit beside the guard.
//!
//! **The default being `⊤` is what makes a negative set unnecessary.** Formula
//! (13) says a record fails an atom by holding a *named* attribute outside its
//! type -- an unnamed one cannot fail, being unconstrained -- so the complement
//! of one atom is a finite union of atoms, one per label:
//!
//! ```text
//! ¬⟨(τ_ℓ)_{ℓ∈L}; ⊤⟩  =  ⋁_{ℓ∈L} ⟨ℓ: ¬τ_ℓ; ⊤⟩
//! ```
//!
//! A union of atoms is therefore closed under all three operations, and the
//! `S` the paper carries for maps is not wanted here. It is wanted there because
//! a map's keys are a *region* with defaults per kind, where a difference cannot
//! be pushed onto finitely many labels; an attribute namespace has no such
//! regions.

use super::classes::Class;
use super::symbolic::Guard;
use super::values::Values;
use crate::decision::Verdict;
use std::collections::{BTreeMap, BTreeSet};

/// The most atoms a union may hold.
///
/// A complement multiplies them: it is an intersection over the atoms, and each
/// one contributes a union over its labels. The bound is a limit of the
/// representation rather than an approximation -- past it there is no sound
/// union to substitute, so the operation refuses.
pub const MAX_ATOMS: usize = 256;

/// What one attribute holds, as a subset of `T⊥`.
///
/// `absent` is the `⊥`: whether the attribute is allowed to be missing. An
/// optional field carries it, a required one does not, and a field that must not
/// exist carries it with an empty type.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Field<G> {
    ty: Values<G>,
    absent: bool,
}

impl<G: Guard> Field<G> {
    /// Any value, or none at all -- what an attribute no atom names holds.
    fn top() -> Field<G> {
        Field {
            ty: Values::Every,
            absent: true,
        }
    }

    /// The values in both, or `None` where a guard refuses.
    fn meet(&self, other: &Field<G>) -> Option<Field<G>> {
        Some(Field {
            ty: self.ty.meet(&other.ty)?,
            absent: self.absent && other.absent,
        })
    }

    /// The rest of `T⊥`, which flips the extra element along with the type.
    fn complement(&self) -> Field<G> {
        Field {
            ty: self.ty.complement(),
            absent: !self.absent,
        }
    }

    /// What is known about something satisfying this field. Being allowed to be
    /// missing settles it whatever the type says.
    fn emptiness(&self) -> Verdict {
        if self.absent {
            Verdict::Inhabited
        } else {
            self.ty.emptiness()
        }
    }
}

/// One open record: finitely many attributes constrained, the rest free, and
/// finitely many classes the value must or must not be an instance of.
///
/// The classes sit in the same atom as the attributes rather than beside them,
/// and that is what keeps the complement finite. A value fails the atom by
/// holding a named attribute outside its type, by not being an instance of one
/// of `is_a`, or by being an instance of one of `not_a` -- finitely many ways,
/// each of them an atom again.
///
/// Both maps are ordered, so two ways of writing one atom compare equal.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Atom<G> {
    fields: BTreeMap<String, Field<G>>,
    /// Classes every value here is an instance of.
    is_a: BTreeSet<Class>,
    /// Classes no value here is an instance of.
    not_a: BTreeSet<Class>,
}

impl<G: Guard> Atom<G> {
    /// Every object: nothing constrained at all.
    fn top() -> Atom<G> {
        Atom {
            fields: BTreeMap::new(),
            is_a: BTreeSet::new(),
            not_a: BTreeSet::new(),
        }
    }

    /// The objects in both: every label of either, met where they share one,
    /// and every class constraint of either, which simply collect.
    fn meet(&self, other: &Atom<G>) -> Option<Atom<G>> {
        let mut fields = self.fields.clone();
        for (label, theirs) in &other.fields {
            let met = match fields.get(label) {
                Some(mine) => mine.meet(theirs)?,
                None => theirs.clone(),
            };
            fields.insert(label.clone(), met);
        }
        Some(Atom {
            fields,
            is_a: self.is_a.union(&other.is_a).cloned().collect(),
            not_a: self.not_a.union(&other.not_a).cloned().collect(),
        })
    }

    /// Whether no object satisfies this atom, proved.
    fn is_empty(&self) -> bool {
        self.emptiness() == Verdict::Empty
    }

    /// What is known about an object satisfying this atom.
    ///
    /// Three ways to be empty, and each is a pair that cannot both hold. Some
    /// attribute holds nothing and may not be missing either; some class it must
    /// be an instance of derives from one it must not; or two it must be an
    /// instance of are laid out apart.
    ///
    /// **The open world is what makes the third answer necessary.** Two classes
    /// it must both be an instance of, neither deriving from the other and
    /// neither laid out apart, are satisfied only by a class deriving from both
    /// -- and whether one exists is not something a snapshot of the order can
    /// say. Reading that as inhabited would be a claim; reading it as empty
    /// would be a worse one.
    fn emptiness(&self) -> Verdict {
        if self
            .is_a
            .iter()
            .any(|mine| self.not_a.iter().any(|barred| mine.derives_from(barred)))
            || self
                .is_a
                .iter()
                .any(|mine| self.is_a.iter().any(|other| mine.disjoint_from(other)))
        {
            return Verdict::Empty;
        }
        let unrelated = self.is_a.iter().any(|mine| {
            self.is_a.iter().any(|other| {
                other != mine && !mine.derives_from(other) && !other.derives_from(mine)
            })
        });
        let fields = Verdict::every(self.fields.values().map(Field::emptiness));
        if unrelated && fields != Verdict::Empty {
            return Verdict::Unknown;
        }
        fields
    }

    /// The objects failing this atom, one atom per label.
    ///
    /// Formula (13) at its simplest, which the open default earns: an object
    /// fails by holding some *named* attribute outside its type, and each label
    /// is one way to fail.
    fn complement(&self) -> Vec<Atom<G>> {
        let attributes = self.fields.iter().map(|(label, field)| Atom {
            fields: BTreeMap::from([(label.clone(), field.complement())]),
            ..Atom::top()
        });
        let barred = self.is_a.iter().map(|class| Atom {
            not_a: BTreeSet::from([class.clone()]),
            ..Atom::top()
        });
        let required = self.not_a.iter().map(|class| Atom {
            is_a: BTreeSet::from([class.clone()]),
            ..Atom::top()
        });
        attributes.chain(barred).chain(required).collect()
    }

    /// Whether the object carrying `attributes` satisfies this atom.
    ///
    /// An attribute the object does not carry satisfies the field exactly when
    /// the field admits being missing, which is the `⊥` again.
    fn holds(&self, class: Option<&Class>, attributes: &[(&str, G::Value)]) -> bool {
        let instance_of = |wanted: &Class| class.is_some_and(|held| held.derives_from(wanted));
        self.fields.iter().all(|(label, field)| {
            match attributes.iter().find(|(name, _)| name == label) {
                Some((_, value)) => field.ty.holds(value),
                None => field.absent,
            }
        }) && self.is_a.iter().all(instance_of)
            && !self.not_a.iter().any(instance_of)
    }

    /// Drop what constrains nothing, so two ways of writing one atom compare
    /// equal.
    ///
    /// A label whose field is the top says nothing. So does a class implied by
    /// another already there: being an instance of a class implies being one of
    /// its ancestors, so `is_a` keeps only what derives from nothing else in it;
    /// and *not* being an instance of a class implies not being one of its
    /// descendants, so `not_a` keeps only the ancestors.
    fn tidy(mut self) -> Atom<G> {
        self.fields.retain(|_, field| field != &Field::top());
        let is_a = self.is_a.clone();
        self.is_a.retain(|mine| {
            !is_a
                .iter()
                .any(|other| other != mine && other.derives_from(mine))
        });
        let not_a = self.not_a.clone();
        self.not_a.retain(|mine| {
            !not_a
                .iter()
                .any(|other| other != mine && mine.derives_from(other))
        });
        self
    }
}

/// A set of objects, held as a union of open record atoms and a polarity.
///
/// The polarity is what keeps `complement` total, which the [`Guard`] a
/// descriptor must be requires of it. Complementing a union of atoms is a
/// product -- an intersection over the atoms, each contributing a union over its
/// labels -- so doing it eagerly could pass the bound and have nowhere sound to
/// go. Flipping a flag cannot, and the product is paid by the operation that
/// needs the atoms, where a refusal is already allowed. The powerset component
/// keeps complement total the same way, and for the same reason.
///
/// **Not canonical**, for the reason a union of powerset lines is not: two
/// unions can hold the same objects and stay unequal, and recognising that costs
/// a search this does not run.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecordLattice<G: Guard> {
    atoms: Vec<Atom<G>>,
    /// Whether the atoms are the objects held or the objects *not* held.
    negated: bool,
}

impl<G: Guard> RecordLattice<G> {
    /// No object at all.
    #[must_use]
    pub fn empty() -> RecordLattice<G> {
        RecordLattice {
            atoms: Vec::new(),
            negated: false,
        }
    }

    /// Every object: the one atom that constrains no attribute.
    #[must_use]
    pub fn all() -> RecordLattice<G> {
        RecordLattice {
            atoms: vec![Atom::top()],
            negated: false,
        }
    }

    /// The objects that are instances of `class`.
    #[must_use]
    pub fn instance_of(class: Class) -> RecordLattice<G> {
        RecordLattice {
            atoms: vec![Atom {
                is_a: BTreeSet::from([class]),
                ..Atom::top()
            }],
            negated: false,
        }
    }

    /// The objects carrying `label`, whose value is in `ty`.
    ///
    /// `optional` admits the objects that do not carry it at all, which is the
    /// `⊥` in the field's type rather than a rule beside it.
    #[must_use]
    pub fn attribute(label: &str, ty: G, optional: bool) -> RecordLattice<G> {
        RecordLattice {
            atoms: vec![Atom {
                fields: BTreeMap::from([(
                    label.to_owned(),
                    Field {
                        ty: Values::Only(ty),
                        absent: optional,
                    },
                )]),
                ..Atom::top()
            }],
            negated: false,
        }
    }

    /// The objects that do *not* carry `label` at all.
    #[must_use]
    pub fn without(label: &str) -> RecordLattice<G> {
        RecordLattice {
            atoms: vec![Atom {
                fields: BTreeMap::from([(
                    label.to_owned(),
                    Field {
                        ty: Values::none(),
                        absent: true,
                    },
                )]),
                ..Atom::top()
            }],
            negated: false,
        }
    }

    /// The atoms of the objects this holds, complementing a negated form.
    fn positive(&self) -> Option<Vec<Atom<G>>> {
        if self.negated {
            complement_atoms(&self.atoms)
        } else {
            Some(self.atoms.clone())
        }
    }

    /// Whether this holds no object.
    ///
    /// A negated form has to be expanded first, and a refusal there reads as
    /// *not* empty -- the safe direction, since claiming emptiness is the claim
    /// that can be wrong.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.emptiness() == Verdict::Empty
    }

    /// What is known about this holding an object.
    ///
    /// A negated form has to be expanded first, and a refusal there is
    /// *unknown* rather than inhabited: past the bound there is no union to
    /// read, so nothing has been proved either way.
    #[must_use]
    pub fn emptiness(&self) -> Verdict {
        match self.positive() {
            Some(atoms) => Verdict::any(atoms.iter().map(Atom::emptiness)),
            None => Verdict::Unknown,
        }
    }

    /// Whether the object carrying `attributes` is held.
    #[must_use]
    pub fn holds(&self, class: Option<&Class>, attributes: &[(&str, G::Value)]) -> bool {
        self.atoms.iter().any(|atom| atom.holds(class, attributes)) != self.negated
    }

    /// The objects in either, or `None` past [`MAX_ATOMS`].
    #[must_use]
    pub fn union(&self, other: &RecordLattice<G>) -> Option<RecordLattice<G>> {
        let mut atoms = self.positive()?;
        atoms.extend(other.positive()?);
        Some(RecordLattice {
            atoms: tidy(atoms)?,
            negated: false,
        })
    }

    /// The objects in both, or `None` past [`MAX_ATOMS`] or where a guard
    /// refuses.
    #[must_use]
    pub fn intersect(&self, other: &RecordLattice<G>) -> Option<RecordLattice<G>> {
        Some(RecordLattice {
            atoms: product(&self.positive()?, &other.positive()?)?,
            negated: false,
        })
    }

    /// The objects this does not hold.
    ///
    /// Total, which is what the [`Guard`] contract asks. The atoms are rebuilt
    /// where the product fits, so the common forms stay comparable, and the
    /// polarity carries the rest.
    #[must_use]
    pub fn complement(&self) -> RecordLattice<G> {
        let flipped = RecordLattice {
            atoms: self.atoms.clone(),
            negated: !self.negated,
        };
        match flipped.positive() {
            Some(atoms) => RecordLattice {
                atoms,
                negated: false,
            },
            None => flipped,
        }
    }
}

/// The atoms a union of atoms complements into, or `None` past [`MAX_ATOMS`].
fn complement_atoms<G: Guard>(atoms: &[Atom<G>]) -> Option<Vec<Atom<G>>> {
    let mut whole = vec![Atom::top()];
    for atom in atoms {
        whole = product(&whole, &atom.complement())?;
    }
    Some(whole)
}

/// The atoms of a meet, which is a meet of every pair.
fn product<G: Guard>(left: &[Atom<G>], right: &[Atom<G>]) -> Option<Vec<Atom<G>>> {
    let mut atoms = Vec::new();
    for mine in left {
        for theirs in right {
            if atoms.len() >= MAX_ATOMS {
                return None;
            }
            atoms.push(mine.meet(theirs)?);
        }
    }
    tidy(atoms)
}

/// Drop the atoms that hold nothing, put the rest in order, and refuse a union
/// past the bound.
fn tidy<G: Guard>(atoms: Vec<Atom<G>>) -> Option<Vec<Atom<G>>> {
    let mut kept: Vec<Atom<G>> = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let atom = atom.tidy();
        if !atom.is_empty() && !kept.contains(&atom) {
            kept.push(atom);
        }
    }
    if kept.len() > MAX_ATOMS {
        return None;
    }
    kept.sort();
    Some(kept)
}

#[cfg(test)]
mod tests {
    use super::{MAX_ATOMS, RecordLattice};
    use crate::descr::classes::Class;
    use crate::descr::integers::IntSet;
    use proptest::prelude::*;

    /// The objects a law is checked over: the attributes the generator names,
    /// carried or not, holding one of a few integers.
    fn objects() -> Vec<Vec<(&'static str, i64)>> {
        let mut objects = vec![Vec::new()];
        for label in ["x", "y"] {
            for value in [-1i64, 0, 1, 4] {
                objects.push(vec![(label, value)]);
            }
        }
        objects.push(vec![("x", 0), ("y", 1)]);
        objects.push(vec![("x", 1), ("y", 0)]);
        objects.push(vec![("x", 4), ("y", 4)]);
        // One attribute no atom names, which the open reading must ignore.
        objects.push(vec![("z", 0)]);
        objects.push(vec![("x", 0), ("z", 0)]);
        objects
    }

    fn holds(lattice: &RecordLattice<IntSet>, object: &[(&'static str, i64)]) -> bool {
        let attributes: Vec<(&str, i64)> = object.to_vec();
        lattice.holds(None, &attributes)
    }

    fn same(a: &RecordLattice<IntSet>, b: &RecordLattice<IntSet>) -> bool {
        objects()
            .iter()
            .all(|object| holds(a, object) == holds(b, object))
    }

    fn lattice() -> impl Strategy<Value = RecordLattice<IntSet>> {
        let leaf = prop_oneof![
            Just(RecordLattice::empty()),
            Just(RecordLattice::all()),
            (prop_oneof![Just("x"), Just("y")]).prop_map(RecordLattice::without),
            (
                prop_oneof![Just("x"), Just("y")],
                -1i64..=1,
                proptest::bool::ANY,
            )
                .prop_map(|(label, n, optional)| {
                    RecordLattice::attribute(label, IntSet::just(n), optional)
                }),
            (prop_oneof![Just("x"), Just("y")], proptest::bool::ANY).prop_map(
                |(label, optional)| {
                    let evens = IntSet::multiple_of(2).expect("a small step");
                    RecordLattice::attribute(label, evens, optional)
                }
            ),
        ];
        leaf.prop_recursive(3, 12, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.union(&b).unwrap_or_else(RecordLattice::all)),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(RecordLattice::empty)),
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

        /// The Boolean algebra, checked against the objects rather than by
        /// equality of the forms, which a union of atoms does not make
        /// canonical.
        #[test]
        fn the_lattice_laws_hold_of_the_objects(
            a in lattice(),
            b in lattice(),
            c in lattice(),
        ) {
            if let (Some(ab), Some(ba)) = (a.union(&b), b.union(&a)) {
                prop_assert!(same(&ab, &ba), "join commutes");
            }
            if let (Some(ab), Some(ba)) = (a.intersect(&b), b.intersect(&a)) {
                prop_assert!(same(&ab, &ba), "meet commutes");
            }
            if let (Some(bc), Some(ab)) = (b.union(&c), a.union(&b))
                && let (Some(left), Some(right)) = (a.union(&bc), ab.union(&c))
            {
                prop_assert!(same(&left, &right), "join associates");
            }
            if let (Some(bc), Some(ac)) = (b.union(&c), a.intersect(&c))
                && let (Some(ab), Some(left)) = (a.intersect(&b), a.intersect(&bc))
                && let Some(right) = ab.union(&ac)
            {
                prop_assert!(same(&left, &right), "meet distributes over join");
            }
        }

        /// The complement laws, and De Morgan both ways.
        #[test]
        fn the_complement_laws_hold_of_the_objects(a in lattice(), b in lattice()) {
            let not_a = a.complement();
            if let Some(met) = a.intersect(&not_a) {
                prop_assert!(met.is_empty(), "an object is in one of the two");
            }
            if let Some(joined) = a.union(&not_a) {
                prop_assert!(same(&joined, &RecordLattice::all()), "and in one of them");
            }
            prop_assert!(same(&not_a.complement(), &a), "twice is nothing");
            let not_b = b.complement();
            if let (Some(joined), Some(met)) = (a.union(&b), not_a.intersect(&not_b)) {
                prop_assert!(same(&joined.complement(), &met), "de Morgan one way");
            }
            if let (Some(met), Some(joined)) = (a.intersect(&b), not_a.union(&not_b)) {
                prop_assert!(same(&met.complement(), &joined), "and the other");
            }
        }

        /// Emptiness is a decision about the objects, not about the form.
        #[test]
        fn emptiness_agrees_with_the_objects(a in lattice()) {
            if a.is_empty() {
                prop_assert!(
                    objects().iter().all(|object| !holds(&a, object)),
                    "an empty lattice holds no object"
                );
            }
        }
    }

    /// A class constrains a value the way an attribute does, and its complement
    /// is a constraint again rather than an absence.
    #[test]
    fn a_class_and_its_complement_are_both_sets() {
        let animal = Class::root(1);
        let dog = Class::new(2, 1, std::slice::from_ref(&animal));
        let dogs = RecordLattice::<IntSet>::instance_of(dog.clone());

        assert!(dogs.holds(Some(&dog), &[]));
        assert!(!dogs.holds(Some(&animal), &[]), "a base is not an instance");
        assert!(!dogs.holds(None, &[]), "a value with no class is not one");

        let others = dogs.complement();
        assert!(!others.holds(Some(&dog), &[]));
        assert!(others.holds(Some(&animal), &[]) && others.holds(None, &[]));
    }

    /// Being an instance of a class is being one of its bases, so a meet with a
    /// base says nothing new -- and a meet with what a base excludes is empty.
    #[test]
    fn deriving_decides_the_meet_and_the_emptiness() {
        let animal = Class::root(1);
        let dog = Class::new(2, 1, std::slice::from_ref(&animal));
        let dogs = RecordLattice::<IntSet>::instance_of(dog.clone());
        let animals = RecordLattice::instance_of(animal.clone());

        let both = dogs.intersect(&animals).expect("two small atoms");
        assert!(same(&both, &dogs), "a dog is already an animal");

        let neither = dogs
            .intersect(&animals.complement())
            .expect("two small atoms");
        assert!(neither.is_empty(), "no dog is not an animal");
    }

    /// The world is open: a class nothing derives from leaves the atom
    /// inhabited, because a class outside the list may yet describe a value.
    #[test]
    fn excluding_an_unrelated_class_leaves_the_atom_inhabited() {
        let animal = Class::root(1);
        let mineral = Class::root(2);
        let animals = RecordLattice::<IntSet>::instance_of(animal.clone());

        let not_mineral = animals
            .intersect(&RecordLattice::instance_of(mineral).complement())
            .expect("two small atoms");
        assert!(!not_mineral.is_empty());
        assert!(not_mineral.holds(Some(&animal), &[]));
    }

    /// Two classes that cannot both describe a value make the atom empty, which
    /// the derivation order alone does not show.
    #[test]
    fn two_classes_of_conflicting_layouts_meet_in_nothing() {
        let ints = Class::root(1);
        let words = Class::root(2);
        let unrelated = Class::new(3, 1, &[]);

        let met = RecordLattice::<IntSet>::instance_of(ints.clone())
            .intersect(&RecordLattice::instance_of(words))
            .expect("two small atoms");
        assert!(met.is_empty(), "no value is laid out both ways");

        // Same layout and no derivation between them: a class deriving from both
        // may exist, so this is *not* empty.
        let open = RecordLattice::<IntSet>::instance_of(ints)
            .intersect(&RecordLattice::instance_of(unrelated))
            .expect("two small atoms");
        assert!(!open.is_empty());
    }

    /// A class and an attribute constrain one value together, which is what
    /// putting them in one atom is for.
    #[test]
    fn a_class_and_an_attribute_constrain_one_value() {
        let dog = Class::root(1);
        let named = RecordLattice::instance_of(dog.clone())
            .intersect(&RecordLattice::attribute("x", IntSet::just(1), false))
            .expect("two small atoms");

        assert!(named.holds(Some(&dog), &[("x", 1)]));
        assert!(!named.holds(Some(&dog), &[("x", 2)]));
        assert!(!named.holds(None, &[("x", 1)]));
    }

    /// An atom constrains the attributes it names and no others, which is what
    /// makes the record *open*.
    #[test]
    fn an_atom_ignores_the_attributes_it_does_not_name() {
        let with_x = RecordLattice::attribute("x", IntSet::all(), false);

        assert!(with_x.holds(None, &[("x", 0)]));
        assert!(with_x.holds(None, &[("x", 0), ("z", 9)]), "z is not named");
        assert!(!with_x.holds(None, &[("z", 9)]), "x is missing");
        assert!(!with_x.holds(None, &[]));
    }

    /// An optional attribute is the type with `⊥` in it: carried and of the
    /// right type, or not carried at all.
    #[test]
    fn an_optional_attribute_admits_the_object_without_it() {
        let evens = IntSet::multiple_of(2).expect("a small step");
        let optional = RecordLattice::attribute("x", evens.clone(), true);
        let required = RecordLattice::attribute("x", evens, false);

        assert!(optional.holds(None, &[]) && !required.holds(None, &[]));
        assert!(optional.holds(None, &[("x", 2)]) && required.holds(None, &[("x", 2)]));
        assert!(
            !optional.holds(None, &[("x", 1)]),
            "carried, so the type decides"
        );
    }

    /// A field that must be missing is the empty type with `⊥` in it, which is
    /// what the complement of an always-present field gives.
    #[test]
    fn an_attribute_that_must_be_missing_is_the_bottom_with_undefined() {
        let without = RecordLattice::<IntSet>::without("x");
        let with_any = RecordLattice::attribute("x", IntSet::all(), false);

        assert!(without.holds(None, &[]) && !without.holds(None, &[("x", 0)]));
        assert!(same(&with_any.complement(), &without));
    }

    /// The complement of an atom is one atom per label, which is what the open
    /// default buys: an unnamed attribute cannot make an object fail.
    #[test]
    fn a_complement_splits_over_the_labels() {
        let both = RecordLattice::attribute("x", IntSet::just(0), false)
            .intersect(&RecordLattice::attribute("y", IntSet::just(0), false))
            .expect("two small atoms");
        let failing = both.complement();

        assert!(both.holds(None, &[("x", 0), ("y", 0)]));
        assert!(!failing.holds(None, &[("x", 0), ("y", 0)]));
        // Failing at either label is enough, and so is missing either one.
        assert!(failing.holds(None, &[("x", 1), ("y", 0)]));
        assert!(failing.holds(None, &[("x", 0), ("y", 1)]));
        assert!(failing.holds(None, &[("x", 0)]));
        assert!(failing.holds(None, &[]));
    }

    /// Two atoms naming different attributes meet in the object carrying both.
    #[test]
    fn atoms_over_different_labels_meet_rather_than_conflict() {
        let met = RecordLattice::attribute("x", IntSet::just(0), false)
            .intersect(&RecordLattice::attribute("y", IntSet::just(1), false))
            .expect("two small atoms");

        assert!(!met.is_empty());
        assert!(met.holds(None, &[("x", 0), ("y", 1)]));
        assert!(!met.holds(None, &[("x", 0)]) && !met.holds(None, &[("y", 1)]));
    }

    /// A required attribute whose type is empty admits nothing, which is
    /// emptiness (11) for the always-open record.
    #[test]
    fn a_required_attribute_of_no_values_is_empty() {
        let impossible = RecordLattice::attribute("x", IntSet::empty(), false);
        assert!(impossible.is_empty());

        // Two required types that do not meet are the same thing found by (12).
        let conflict = RecordLattice::attribute("x", IntSet::just(0), false)
            .intersect(&RecordLattice::attribute("x", IntSet::just(1), false))
            .expect("two small atoms");
        assert!(conflict.is_empty());
    }

    /// A union past the bound refuses rather than holding a form it cannot.
    #[test]
    fn a_union_past_the_bound_refuses() {
        let mut wide = RecordLattice::attribute("x", IntSet::just(0), false);
        for n in 1..i64::try_from(MAX_ATOMS).unwrap_or(i64::MAX) {
            wide = wide
                .union(&RecordLattice::attribute("x", IntSet::just(n), false))
                .expect("inside the bound");
        }
        assert!(
            wide.union(&RecordLattice::attribute("x", IntSet::just(-1), false))
                .is_none()
        );
    }
}
