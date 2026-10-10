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

use super::classes::{Class, Reach};
use super::polar::{PolarUnion, Summand};
use super::symbolic::Guard;
use super::values::{Field, Values};
use crate::Kind;
use crate::verdict::Verdict;
use std::collections::{BTreeMap, BTreeSet};

/// The most atoms a union may hold.
///
/// A complement multiplies them: it is an intersection over the atoms, and each
/// one contributes a union over its labels. The bound is a limit of the
/// representation rather than an approximation -- past it there is no sound
/// union to substitute, so the operation refuses.
pub const MAX_ATOMS: usize = 256;

/// Whether an attribute that can be `reach` on a direct instance can satisfy
/// `field`, which is known to admit something -- a value, or its absence.
///
/// [`Reach::Anything`] meets any field that admits something. An attribute that
/// takes any value meets a field only through a value, and one that is never
/// there only a field that may be missing. One whose value is code's answer
/// meets none: the direct instance may hold anything there, or nothing.
fn carries<G: Guard>(reach: Reach, field: &Field<G>) -> bool {
    match reach {
        Reach::Anything => true,
        Reach::AnyValue => field.ty.emptiness() == Verdict::Inhabited,
        Reach::Missing => field.absent,
        Reach::Unread => false,
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
    /// and every class constraint of either, which collect.
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
    ///
    /// **One class is one value, and the class says what that value carries.**
    /// What makes an atom with one class inhabited is a *direct* instance of
    /// the class holding a witness in each field -- the one object the class
    /// assumption licenses. A class can leave no room for it: a name its body
    /// defines cannot be missing, a property answers with its getter, and an
    /// instance laid out without a dictionary holds nothing its slots do not
    /// name. Where the class's [`Reach`] for some field rules the direct
    /// instance out, a subclass may still carry the field and may not exist, so
    /// the answer is unknown -- never empty, which would claim no subclass
    /// could.
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
        // Two classes left in `is_a` are incomparable, because [`Atom::tidy`]
        // drops one that another derives from -- so counting them is the whole
        // question. Re-testing the order here would be asking what tidying has
        // already answered, and the mutation sweep says as much: neither guard
        // can be made to change an answer.
        let unrelated = self.is_a.len() > 1;
        let fields = Verdict::every(self.fields.values().map(Field::emptiness));
        if unrelated && fields != Verdict::Empty {
            return Verdict::Unknown;
        }
        if fields == Verdict::Inhabited
            && let Some(class) = self.is_a.first()
            && !self
                .fields
                .iter()
                .all(|(name, field)| carries(class.reach(name), field))
        {
            return Verdict::Unknown;
        }
        fields
    }

    /// The same, asked of the objects of one *kind*.
    ///
    /// A class this atom requires that confines its instances to no kind says
    /// nothing about whether an object of this kind satisfies it: that needs a
    /// class deriving from both it and the kind's builtin, which is the open
    /// world the rule above already declines two unrelated classes for. So the
    /// kind is read as one more class the value must be an instance of, and a
    /// class that is not exactly this kind's leaves the answer unknown -- never
    /// empty, since a class deriving from both may exist, and never inhabited,
    /// since it may not.
    ///
    /// `None` is the line of objects that have no builtin kind, where the class
    /// is the whole of what is asked.
    ///
    /// **An atom with no class names its fields of the kind's own values.** On
    /// a builtin kind's line the value that would satisfy it is an exact `int`,
    /// `str` or `list`, the one the kind holds without a subclass being
    /// assumed, and such a value carries no `__dict__`: a name its builtin does
    /// not define is never there, and one it defines holds what the builtin
    /// says -- `int.real`, `list.__len__` -- which the core cannot read. So an
    /// atom that names a field is unknown there, as the class with no room is
    /// ([`Reach::Unread`]); reading it as inhabited refuted `int` below a record
    /// of `real` that every integer satisfies. The line of objects with no
    /// builtin kind keeps the reading: a plain object carries what it is given.
    fn emptiness_of_kind(&self, kind: Option<Kind>) -> Verdict {
        let known = self.emptiness();
        match kind {
            Some(kind)
                if known != Verdict::Empty
                    && (self.is_a.iter().any(|class| class.kind() != Some(kind))
                        || (self.is_a.is_empty() && !self.fields.is_empty())) =>
            {
                Verdict::Unknown
            }
            _ => known,
        }
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

impl<G: Guard> Summand for Atom<G> {
    /// Nothing: an atom's universe is every object, whatever kind carries it.
    type Within = ();
    type Complement = Vec<Atom<G>>;
    const MAX: usize = MAX_ATOMS;

    fn top((): ()) -> Atom<G> {
        Atom::top()
    }

    fn meet(&self, other: &Atom<G>) -> Option<Atom<G>> {
        Atom::meet(self, other)
    }

    fn complement(&self, (): ()) -> Vec<Atom<G>> {
        Atom::complement(self)
    }

    /// Each atom in its tidied shape, without the ones holding no object, each
    /// once, and in order.
    ///
    /// Once by equality rather than by position after a sort. A guard's
    /// equality can be coarser than its order -- an integer set compares two
    /// spellings of one set equal and sorts them apart -- so two equal atoms
    /// need not be neighbours once sorted, and a sort cannot be what finds the
    /// repeat.
    fn compacted(atoms: Vec<Atom<G>>) -> Option<Vec<Atom<G>>> {
        let mut kept: Vec<Atom<G>> = Vec::with_capacity(atoms.len());
        for atom in atoms {
            let atom = atom.tidy();
            if !atom.is_empty() && !kept.contains(&atom) {
                kept.push(atom);
            }
        }
        kept.sort();
        Some(kept)
    }
}

/// A set of objects, held as a union of open record atoms and a polarity.
///
/// The polarity is what keeps `complement` total, which the [`Guard`] a
/// descriptor must be requires of it: complementing a union of atoms is a
/// product -- an intersection over the atoms, each contributing a union over
/// its labels -- so doing it eagerly could pass the bound and have nowhere
/// sound to go. The device is the one every lattice built from parts shares,
/// in `descr/polar.rs`.
///
/// **Not canonical**, for the reason a union of powerset lines is not: two
/// unions can hold the same objects and stay unequal, and recognising that costs
/// a search this does not run.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecordLattice<G: Guard>(PolarUnion<Atom<G>>);

impl<G: Guard> RecordLattice<G> {
    /// No object at all.
    #[must_use]
    pub fn empty() -> RecordLattice<G> {
        RecordLattice(PolarUnion::of(Vec::new()))
    }

    /// Every object: the one atom that constrains no attribute.
    #[must_use]
    pub fn all() -> RecordLattice<G> {
        RecordLattice::one(Atom::top())
    }

    /// The objects one atom admits.
    fn one(atom: Atom<G>) -> RecordLattice<G> {
        RecordLattice(PolarUnion::of(vec![atom]))
    }

    /// The objects that are instances of `class`.
    #[must_use]
    pub fn instance_of(class: Class) -> RecordLattice<G> {
        RecordLattice::one(Atom {
            is_a: BTreeSet::from([class]),
            ..Atom::top()
        })
    }

    /// The objects carrying `label`, whose value is in `ty`.
    ///
    /// `optional` admits the objects that do not carry it at all, which is the
    /// `⊥` in the field's type rather than a rule beside it.
    #[must_use]
    pub fn attribute(label: &str, ty: G, optional: bool) -> RecordLattice<G> {
        RecordLattice::one(Atom {
            fields: BTreeMap::from([(
                label.to_owned(),
                Field {
                    ty: Values::Only(ty),
                    absent: optional,
                },
            )]),
            ..Atom::top()
        })
    }

    /// The objects that do *not* carry `label` at all.
    #[must_use]
    pub fn without(label: &str) -> RecordLattice<G> {
        RecordLattice::one(Atom {
            fields: BTreeMap::from([(
                label.to_owned(),
                Field {
                    ty: Values::none(),
                    absent: true,
                },
            )]),
            ..Atom::top()
        })
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
    #[must_use]
    pub fn emptiness(&self) -> Verdict {
        self.emptiness_of_kind(None)
    }

    /// The same, asked of the objects of one kind.
    ///
    /// A constraint on objects is carried on every kind's line, because a class
    /// that lays down no layout of its own confines nothing and a subclass of
    /// it may be laid out as anything. That placement is right for *inclusion*
    /// and it is not a value: an object of kind `k` that is an instance of a
    /// class confining nothing exists only if some class derives from both, and
    /// which classes exist is not something a snapshot of the order can say.
    /// That is the same open world the atom's own emptiness already declines two
    /// unrelated classes for, with the kind standing as the second class.
    ///
    /// `None` is the line of objects that have no builtin kind, where the class
    /// is the whole of what is asked and the documented assumption -- a class
    /// the bindings can read has an instance -- is the answer.
    #[must_use]
    pub fn emptiness_of_kind(&self, kind: Option<Kind>) -> Verdict {
        self.0.verdict((), |atom| atom.emptiness_of_kind(kind))
    }

    /// Whether the object carrying `attributes` is held.
    #[must_use]
    pub fn holds(&self, class: Option<&Class>, attributes: &[(&str, G::Value)]) -> bool {
        self.0.holds(|atom| atom.holds(class, attributes))
    }

    /// The objects in either, or `None` past [`MAX_ATOMS`].
    #[must_use]
    pub fn union(&self, other: &RecordLattice<G>) -> Option<RecordLattice<G>> {
        self.0.union(&other.0, ()).map(RecordLattice)
    }

    /// The objects in both, or `None` past [`MAX_ATOMS`] or where a guard
    /// refuses.
    #[must_use]
    pub fn intersect(&self, other: &RecordLattice<G>) -> Option<RecordLattice<G>> {
        self.0.intersect(&other.0, ()).map(RecordLattice)
    }

    /// The objects this does not hold.
    ///
    /// Total, which is what the [`Guard`] contract asks.
    #[must_use]
    pub fn complement(&self) -> RecordLattice<G> {
        RecordLattice(self.0.complement(()))
    }

    /// The atoms as held, whichever polarity reads them.
    #[cfg(test)]
    fn atoms(&self) -> &[Atom<G>] {
        self.0.summands()
    }

    /// Whether the atoms are the objects *not* held.
    #[cfg(test)]
    const fn negated(&self) -> bool {
        self.0.is_negated()
    }
}

#[cfg(test)]
mod tests;
