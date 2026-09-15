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

use super::budget;
use super::classes::Class;
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
    fn emptiness_of_kind(&self, kind: Option<Kind>) -> Verdict {
        let known = self.emptiness();
        match kind {
            Some(kind)
                if known != Verdict::Empty
                    && self.is_a.iter().any(|class| class.kind() != Some(kind)) =>
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
        match self.positive() {
            Some(atoms) => Verdict::any(atoms.iter().map(|atom| atom.emptiness_of_kind(kind))),
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
///
/// Every pair charges the build's allowance: the count is the product of the
/// two, and a meet of a guard against a guard descends a level of nesting for
/// each pair. See [`budget`](super::budget).
fn product<G: Guard>(left: &[Atom<G>], right: &[Atom<G>]) -> Option<Vec<Atom<G>>> {
    let mut atoms = Vec::new();
    for mine in left {
        for theirs in right {
            if atoms.len() >= MAX_ATOMS || !budget::spend() {
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
mod tests;
