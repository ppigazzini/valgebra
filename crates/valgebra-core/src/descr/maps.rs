//! Maps as record atoms with per-kind defaults and a negative set.
//!
//! A dict is a *quasi-constant function* from keys to values (Castagna, ICFP
//! 2023, Definition 2.2, after Frisch 2004): total on the keys, and constant on
//! all but finitely many of them. So a set of dicts is written by naming the
//! finitely many keys it constrains and saying what every other key maps to --
//! which is Definition 4.1's atom,
//!
//! ```text
//! ⟨ (τ_ℓ)_{ℓ∈L} ; t₀ ; S ⟩
//! ```
//!
//! with `L` the labels named, `τ_ℓ` what each maps to, `t₀` the default, and `S`
//! a set of constraints each reading "and there is a key outside `L` whose value
//! is in `s`". `S` is what makes the atom closed under *difference*: without it
//! the complement of a map is not a map, and the union-of-atoms form the whole
//! representation rests on would not exist.
//!
//! Three properties do the work, and they are the paper's (11), (12) and (13):
//!
//! ```text
//! ⟨τ; t₀; S⟩ ≠ ∅   ⟺   ∀ℓ. τ_ℓ ≠ ∅   and   ∀s ∈ S. t₀ ∧ s ≠ ∅              (11)
//! ⟨τ; t₀; S⟩ ∧ ⟨τ′; t′₀; S′⟩  =  ⟨τ ∧ τ′; t₀ ∧ t′₀; S ∪ S′⟩                (12)
//! ⟨τ; t₀; S⟩ ∖ ⟨τ′; t′₀; ∅⟩  =  ⟨τ; t₀; S ∪ {¬t′₀}⟩ ∨ ⋁_{ℓ₀} ⟨τ^{ℓ₀}; t₀; S⟩ (13)
//! ```
//!
//! where `τ^{ℓ₀}_ℓ` is `τ_ℓ ∧ not(τ′_ℓ)` at `ℓ₀` and `τ_ℓ` elsewhere. Read (13)
//! as the ways to *fail* an atom: some named key's value is outside its type, or
//! some unnamed key's value is outside the default. Finitely many, each an atom
//! again, which is what keeps the representation closed.
//!
//! **The default is per key kind.** The paper's §4.4 maps are indexed by a
//! partition of the key space whose parts are pairwise disjoint -- "we propose to
//! forbid any overlapping between domains in the same map" -- because a key
//! belonging to two domains would have two types and the theory would be that of
//! function intersections instead. The kind partition is exactly such a
//! partition, and using it is what lets the atom stay a function. A key schema
//! that is neither a kind nor a constant has no part to land in, and the frontend
//! refuses it rather than approximating.

use std::collections::{BTreeMap, BTreeSet};

use super::budget;
use super::symbolic::Guard;
use super::values::{Field, Values};
use crate::decision::{Kind, Verdict};

/// The most atoms a union may carry.
///
/// A meet multiplies the two sides' counts and a difference adds one atom per
/// label and per key kind, so the union needs a bound for the reason every other
/// union here does: past it there is no sound set to substitute, since one too
/// wide is complemented into one too narrow.
pub const MAX_ATOMS: usize = 256;

/// The listed kinds a dict key may have: the hashable ones.
///
/// A `list`, a `set` and a `dict` are unhashable, so no key has those kinds and
/// the atom carries no default for them. The array is what makes the default a
/// function on a partition rather than a set of clauses that might overlap.
pub const KEY_KINDS: [Kind; 8] = [
    Kind::NoneType,
    Kind::Bool,
    Kind::Int,
    Kind::Float,
    Kind::Str,
    Kind::Bytes,
    Kind::Tuple,
    Kind::FrozenSet,
];

/// The parts the key space is divided into: one per key kind, and one more for
/// a key of no listed kind -- an instance of a class that defines `__hash__`.
///
/// The last part is what keeps the default a *total* function on the keys. A key
/// the partition did not cover would be governed by nothing, and an atom would
/// then admit a dict it says nothing about.
const PARTS: usize = KEY_KINDS.len() + 1;

/// The part a key of `kind` falls in, or `None` for a kind no key can have.
///
/// A `list`, a `set` and a `dict` are unhashable, so a dict carrying one as a key
/// is not a dict at all, and every atom declines it.
fn key_slot(kind: Option<Kind>) -> Option<usize> {
    match kind {
        Some(kind) => KEY_KINDS.iter().position(|listed| *listed == kind),
        None => Some(KEY_KINDS.len()),
    }
}

/// One key a map atom names: a constant, and the part it falls in.
///
/// A field of a record and a literal key of a mapping are the same thing --
/// `{"a": int}` and `dict[Literal["a"], int]` name one key and say what it maps
/// to -- so one type serves both, and an atom cannot tell which spelling it came
/// from. Ordered, so two ways of writing an atom compare equal.
///
/// A `float` is not here: the typing spec disallows one as a `Literal`, on the
/// grounds that infinity and `nan` have no clean spelling. Nor is a tuple or a
/// frozenset, which are hashable but not constants a `Literal` names.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Label {
    /// `None`.
    NoneType,
    /// One of the two booleans.
    Bool(bool),
    /// An integer, which is never a boolean: `bool` is its own kind here.
    Int(i64),
    /// A word, under the kind that reads it -- a `str` as its UTF-8 bytes.
    Word(Vec<u8>, Kind),
}

impl Label {
    /// A `str` key with this text, which is what a record's field is.
    #[must_use]
    pub fn str(text: &str) -> Label {
        Label::Word(text.as_bytes().to_vec(), Kind::Str)
    }

    /// The kind of the key, which is the part of the partition it falls in.
    #[must_use]
    pub fn kind(&self) -> Kind {
        match self {
            Label::NoneType => Kind::NoneType,
            Label::Bool(_) => Kind::Bool,
            Label::Int(_) => Kind::Int,
            Label::Word(_, kind) => *kind,
        }
    }
}

/// One entry of a dict being asked about: its key and what it maps to.
///
/// The key is given twice over, because an atom asks two questions of it:
/// whether it is a key the atom *names*, and which part of the partition it
/// falls in. A key that is not a constant has no label and only a part.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry<V> {
    /// The key, where it is a constant an atom could name.
    pub label: Option<Label>,
    /// The key's kind, or `None` where the key is of no listed kind.
    pub kind: Option<Kind>,
    /// What the key maps to.
    pub value: V,
}

/// One constraint of `S`: some key outside `besides`, of this part, maps into
/// `ty`.
///
/// The kind travels with the type because the default does: "some unnamed key
/// maps outside `t₀`" is a different claim per part of the partition, and (13)
/// produces one such constraint per part.
///
/// **The labels travel with it too, and that is not the paper's shape.** The
/// paper reads `S` against the atom's own `L`, and can, because Theorem 4.2 fixes
/// one `L` for every atom of a normal form before any of them is combined. Here
/// each atom carries its own labels, and a meet unions them -- so an `S` read
/// against "the atom's labels" would quietly *strengthen* as the atom learned
/// more names, and `A ∧ B` would hold fewer maps than `A` and `B` share. Naming
/// the exclusion set at the point the constraint is made keeps its meaning fixed
/// under every later operation, which is the same fixing the paper does globally.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Wanted<G> {
    slot: usize,
    ty: Values<G>,
    besides: BTreeSet<Label>,
}

/// One map atom, `⟨(τ_ℓ)_{ℓ∈L} ; t₀ ; S⟩`.
///
/// Both collections are ordered, so two ways of writing one atom compare equal.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MapAtom<G> {
    /// `(τ_ℓ)_{ℓ∈L}`, keyed by the constant each names.
    labels: BTreeMap<Label, Field<G>>,
    /// `t₀`, one entry per key kind.
    defaults: [Field<G>; PARTS],
    /// `S`.
    wanted: Vec<Wanted<G>>,
}

impl<G: Guard> MapAtom<G> {
    /// Every dict: no label named, every key free to map anywhere.
    fn top() -> MapAtom<G> {
        MapAtom {
            labels: BTreeMap::new(),
            defaults: core::array::from_fn(|_| Field::top()),
            wanted: Vec::new(),
        }
    }

    /// What the atom says about a label it may or may not name.
    fn field(&self, label: &Label) -> Field<G> {
        match self.labels.get(label) {
            Some(field) => field.clone(),
            // A label the atom does not name is governed by the default for its
            // own part, which is what Definition 2.2's quasi-constant function
            // says it maps to -- so reading it here is exact rather than a
            // widening.
            None => self.default_for(Some(label.kind())),
        }
    }

    /// The same atom with every label its part's default already covers dropped.
    ///
    /// This is the paper's `dom` read **semantically**. Definition 2.2 notes that
    /// the notation is "not univocal (unless we require zᵢ ≠ z...)": naming a key
    /// and giving it exactly what the default already gives it says nothing, and
    /// two atoms that differ only in such a name are one atom. Dropping them is
    /// what makes `{"a?": nothing}` and `{}` the same closed map rather than two
    /// that merely admit the same dicts.
    fn absorbed(mut self) -> MapAtom<G> {
        let covered: Vec<Label> = self
            .labels
            .iter()
            .filter(|(label, field)| **field == self.default_for(Some(label.kind())))
            .map(|(label, _)| label.clone())
            .collect();
        for label in covered {
            self.labels.remove(&label);
        }
        self
    }

    /// The default for the part a key of `kind` falls in.
    ///
    /// A key of a kind no key can have -- an unhashable one -- is governed by
    /// nothing, which is read as the empty field: a dict cannot carry it, so an
    /// atom that is asked about one holds no such dict.
    fn default_for(&self, kind: Option<Kind>) -> Field<G> {
        key_slot(kind)
            .and_then(|slot| self.defaults.get(slot))
            .cloned()
            .unwrap_or_else(|| Field {
                ty: Values::none(),
                absent: true,
            })
    }

    /// (11): what is known about this atom holding a dict.
    ///
    /// Empty as soon as a named key's type is, or no key can satisfy a
    /// constraint of `S`.
    ///
    /// The paper's (11) reads `S` against the default alone, because its labels
    /// are fixed and `S` is always about a key outside them. Here a constraint
    /// carries its own exclusion set, so a label the set does not cover is a
    /// witness too, and both are read. What is *not* read is whether the part has
    /// a key left at all: with every label a `str`, and `str` inexhaustible, it
    /// always does. A label of a finite kind -- which is a typed literal key,
    /// not a name -- is what makes that clause able to fire, and it belongs with
    /// the commit that introduces one.
    fn emptiness(&self) -> Verdict {
        let labels = self.labels.values().map(Field::emptiness);
        let wanted = self.wanted.iter().map(|want| {
            let Some(default) = self.defaults.get(want.slot) else {
                return Verdict::Empty;
            };
            match default.ty.meet(&want.ty) {
                Some(shared) => shared.emptiness(),
                // Past a guard's own bound there is no set to read, so nothing is
                // proved either way.
                None => Verdict::Unknown,
            }
        });
        Verdict::every(labels.chain(wanted))
    }

    /// (12): the dicts in both atoms.
    ///
    /// The label sets are unioned first, each side reading a label it does not
    /// name off its own default -- which is what Definition 2.2's quasi-constant
    /// function says it maps to, so the extension is exact rather than a
    /// widening.
    fn meet(&self, other: &MapAtom<G>) -> Option<MapAtom<G>> {
        let mut labels = BTreeMap::new();
        for label in self.labels.keys().chain(other.labels.keys()) {
            if labels.contains_key(label) {
                continue;
            }
            labels.insert(label.clone(), self.field(label).meet(&other.field(label))?);
        }
        let mut defaults = Vec::with_capacity(PARTS);
        for (mine, theirs) in self.defaults.iter().zip(&other.defaults) {
            defaults.push(mine.meet(theirs)?);
        }
        let mut wanted = self.wanted.clone();
        wanted.extend(other.wanted.iter().cloned());
        wanted.sort();
        wanted.dedup();
        Some(MapAtom {
            labels,
            defaults: defaults.try_into().ok()?,
            wanted,
        })
    }

    /// (13): the ways to fail this atom, as atoms.
    ///
    /// An atom is `P ∖ N`, with `P = ⟨τ; t₀; ∅⟩` and `N` the maps that meet none
    /// of `S`. So `¬A` is `¬P ∨ N`, and there are three ways in:
    ///
    /// - a **named** key's value is outside what the atom allows there, which is
    ///   (13)'s per-label disjunct taken from the whole;
    /// - an **unnamed** key of some part maps outside that part's default, which
    ///   is (13)'s `¬t′₀` disjunct, one per part because the default is one per
    ///   part;
    /// - a constraint of `S` is simply not met: **no** unnamed key of its part
    ///   maps into it, which is the `N` above.
    ///
    /// The third is why this is a complement rather than a difference. (13) is
    /// stated for a subtrahend whose `S` is empty, and an atom that came out of a
    /// complement does not have one -- so the caller folds complements with a
    /// product instead of subtracting, and the case is handled here where the
    /// `S` is in reach.
    ///
    /// The labels of `L` are written explicitly into the last two, at their top,
    /// so the witness a constraint asks for is a key **outside** `L`.
    fn complement(&self) -> Vec<MapAtom<G>> {
        let mut atoms = Vec::new();
        for (label, field) in &self.labels {
            let mut atom = MapAtom::top();
            atom.labels.insert(label.clone(), field.complement());
            atoms.push(atom);
        }
        let widened = || {
            let mut atom = MapAtom::top();
            for label in self.labels.keys() {
                atom.labels.insert(label.clone(), Field::top());
            }
            atom
        };
        let named: BTreeSet<Label> = self.labels.keys().cloned().collect();
        for (slot, default) in self.defaults.iter().enumerate() {
            let mut atom = widened();
            atom.wanted.push(Wanted {
                slot,
                ty: default.ty.complement(),
                besides: named.clone(),
            });
            atoms.push(atom);
        }
        // "No key outside `besides` maps into `ty`" constrains the default for
        // that part *and* every label the exclusion set does not cover.
        for want in &self.wanted {
            let mut atom = widened();
            if let Some(default) = atom.defaults.get_mut(want.slot) {
                *default = Field {
                    ty: want.ty.complement(),
                    absent: true,
                };
            }
            let narrowed: Vec<Label> = self
                .labels
                .keys()
                .filter(|label| {
                    key_slot(Some(label.kind())) == Some(want.slot)
                        && !want.besides.contains(*label)
                })
                .cloned()
                .collect();
            for label in narrowed {
                atom.labels.insert(
                    label,
                    Field {
                        ty: want.ty.complement(),
                        absent: true,
                    },
                );
            }
            atoms.push(atom);
        }
        atoms
    }

    /// Whether this atom holds the dict carrying `entries`.
    fn holds(&self, entries: &[Entry<G::Value>]) -> bool {
        let named = |label: &Label| {
            entries
                .iter()
                .find(|entry| entry.label.as_ref() == Some(label))
        };
        for (label, field) in &self.labels {
            match named(label) {
                Some(entry) => {
                    if !field.ty.holds(&entry.value) {
                        return false;
                    }
                }
                // A key the dict does not carry satisfies the field only where
                // the field allows it to be missing.
                None => {
                    if !field.absent {
                        return false;
                    }
                }
            }
        }
        let unnamed = |entry: &Entry<G::Value>| {
            !entry
                .label
                .as_ref()
                .is_some_and(|label| self.labels.contains_key(label))
        };
        // Every key the labels do not name is governed by its part's default.
        for entry in entries {
            if unnamed(entry) && !self.default_for(entry.kind).ty.holds(&entry.value) {
                return false;
            }
        }
        // And every constraint of `S` wants a key outside its own exclusion set
        // to witness it.
        self.wanted.iter().all(|want| {
            entries.iter().any(|entry| {
                !entry
                    .label
                    .as_ref()
                    .is_some_and(|label| want.besides.contains(label))
                    && key_slot(entry.kind) == Some(want.slot)
                    && want.ty.holds(&entry.value)
            })
        })
    }
}

/// The dicts a descriptor admits, as a union of map atoms or their complement.
///
/// The polarity is the device the record and powerset lattices carry, and for
/// the same reason: a complement must be total, and its positive form can pass
/// [`MAX_ATOMS`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MapLattice<G: Guard> {
    atoms: Vec<MapAtom<G>>,
    /// Whether the atoms are the dicts held or the dicts *not* held.
    negated: bool,
}

impl<G: Guard> MapLattice<G> {
    /// No dict at all.
    #[must_use]
    pub fn empty() -> MapLattice<G> {
        MapLattice {
            atoms: Vec::new(),
            negated: false,
        }
    }

    /// Every dict: the one atom that names no label and lets every key map
    /// anywhere.
    #[must_use]
    pub fn all() -> MapLattice<G> {
        MapLattice {
            atoms: vec![MapAtom::top()],
            negated: false,
        }
    }

    /// The dicts whose `label` maps into `ty`, every other key free.
    ///
    /// `optional` admits the dicts without the key at all, which is the `⊥` in
    /// the field's type rather than a rule beside it.
    #[must_use]
    pub fn label(label: Label, ty: G, optional: bool) -> MapLattice<G> {
        let mut atom = MapAtom::top();
        atom.labels.insert(
            label,
            Field {
                ty: Values::Only(ty),
                absent: optional,
            },
        );
        MapLattice {
            atoms: vec![atom.absorbed()],
            negated: false,
        }
    }

    /// The dicts a keyed map spells: the keys it names, and what a key of each
    /// part may hold besides them.
    ///
    /// Built as one atom rather than composed from meets, because closing a map
    /// and naming a key are not independent: "no keys at all" met with
    /// `{"a": int}` is empty, since the closing default governs `a` too. Here the
    /// labels are written after the defaults, which is the order the atom means
    /// -- `t₀` governs the keys `L` does not name.
    ///
    /// A part `opened` does not mention is shut, which is what a record with no
    /// catch-all forbids. Two clauses over one part join, which is the IR's
    /// reading of overlapping clauses: a key belongs when *some* clause admits
    /// it.
    #[must_use]
    pub fn record(
        labels: impl IntoIterator<Item = (Label, G, bool)>,
        opened: impl IntoIterator<Item = (Option<Kind>, G)>,
    ) -> Option<MapLattice<G>> {
        let mut atom = MapAtom::top();
        for default in &mut atom.defaults {
            *default = Field {
                ty: Values::none(),
                absent: true,
            };
        }
        for (part, ty) in opened {
            if let Some(slot) = key_slot(part)
                && let Some(default) = atom.defaults.get_mut(slot)
            {
                let widened = Field {
                    ty: Values::Only(ty),
                    absent: true,
                };
                *default = default.join(&widened)?;
            }
        }
        for (label, ty, optional) in labels {
            let named = Field {
                ty: Values::Only(ty),
                absent: optional,
            };
            // Two clauses naming one key both constrain it, so the second meets
            // the first rather than replacing it.
            let field = match atom.labels.get(&label) {
                Some(held) => held.meet(&named)?,
                None => named,
            };
            atom.labels.insert(label, field);
        }
        Some(MapLattice {
            atoms: tidy(vec![atom])?,
            negated: false,
        })
    }

    /// The dicts every one of whose keys of `kind` maps into `ty`.
    #[must_use]
    pub fn keyed(kind: Kind, ty: G) -> MapLattice<G> {
        let mut atom = MapAtom::top();
        if let Some(slot) = key_slot(Some(kind))
            && let Some(default) = atom.defaults.get_mut(slot)
        {
            *default = Field {
                ty: Values::Only(ty),
                absent: true,
            };
        }
        MapLattice {
            atoms: vec![atom],
            negated: false,
        }
    }

    /// The dicts carrying no key outside `parts`.
    ///
    /// What *closes* a map. `dict[str, int]` on its own admits a dict with an
    /// integer key, because the default for the integer part is untouched;
    /// meeting it with this says the map has string keys and no others. A named
    /// key is unaffected -- the default governs the keys the labels do not name,
    /// so closing a part does not shut a label of it.
    ///
    /// `None` in `parts` is the part for a key of no listed kind, which a closed
    /// record has to shut too: an object with a `__hash__` is a key.
    #[must_use]
    pub fn keys_among(parts: &[Option<Kind>]) -> MapLattice<G> {
        let mut atom = MapAtom::top();
        for slot in 0..PARTS {
            let open = parts.iter().any(|part| key_slot(*part) == Some(slot));
            if !open && let Some(default) = atom.defaults.get_mut(slot) {
                *default = Field {
                    ty: Values::none(),
                    absent: true,
                };
            }
        }
        MapLattice {
            atoms: vec![atom],
            negated: false,
        }
    }

    /// The atoms of the dicts this holds, complementing a negated form.
    fn positive(&self) -> Option<Vec<MapAtom<G>>> {
        if self.negated {
            complement_atoms(&self.atoms)
        } else {
            Some(self.atoms.clone())
        }
    }

    /// What is known about this holding a dict.
    ///
    /// A negated form has to be expanded first, and a refusal there is *unknown*
    /// rather than inhabited: past the bound there is no union to read.
    #[must_use]
    pub fn emptiness(&self) -> Verdict {
        match self.positive() {
            Some(atoms) => Verdict::any(atoms.iter().map(MapAtom::emptiness)),
            None => Verdict::Unknown,
        }
    }

    /// Whether the dict carrying `entries` is held.
    #[must_use]
    pub fn holds(&self, entries: &[Entry<G::Value>]) -> bool {
        self.atoms.iter().any(|atom| atom.holds(entries)) != self.negated
    }

    /// The dicts in either, or `None` past [`MAX_ATOMS`].
    #[must_use]
    pub fn union(&self, other: &MapLattice<G>) -> Option<MapLattice<G>> {
        let mut atoms = self.positive()?;
        atoms.extend(other.positive()?);
        Some(MapLattice {
            atoms: tidy(atoms)?,
            negated: false,
        })
    }

    /// The dicts in both, or `None` past [`MAX_ATOMS`] or where a guard refuses.
    #[must_use]
    pub fn intersect(&self, other: &MapLattice<G>) -> Option<MapLattice<G>> {
        Some(MapLattice {
            atoms: product(&self.positive()?, &other.positive()?)?,
            negated: false,
        })
    }

    /// The dicts this does not hold.
    ///
    /// Total, which is what the [`Guard`] contract asks. The atoms are rebuilt
    /// where the difference fits, and the polarity carries the rest.
    #[must_use]
    pub fn complement(&self) -> MapLattice<G> {
        let flipped = MapLattice {
            atoms: self.atoms.clone(),
            negated: !self.negated,
        };
        match flipped.positive() {
            Some(atoms) => MapLattice {
                atoms,
                negated: false,
            },
            None => flipped,
        }
    }
}

/// The atoms a union of atoms complements into, or `None` past [`MAX_ATOMS`].
///
/// `¬⋁ᵢ Aᵢ` is `⋀ᵢ ¬Aᵢ`, and each `¬Aᵢ` is the union [`MapAtom::complement`]
/// gives, so the fold is a product rather than a subtraction.
fn complement_atoms<G: Guard>(atoms: &[MapAtom<G>]) -> Option<Vec<MapAtom<G>>> {
    let mut whole = vec![MapAtom::top()];
    for atom in atoms {
        whole = product(&whole, &atom.complement())?;
    }
    Some(whole)
}

/// The atoms of a meet, which is a meet of every pair.
///
/// Every pair charges the build's allowance, for the reason the record atoms'
/// product does. See [`budget`](super::budget).
fn product<G: Guard>(left: &[MapAtom<G>], right: &[MapAtom<G>]) -> Option<Vec<MapAtom<G>>> {
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
fn tidy<G: Guard>(atoms: Vec<MapAtom<G>>) -> Option<Vec<MapAtom<G>>> {
    let mut kept: Vec<MapAtom<G>> = atoms
        .into_iter()
        .map(MapAtom::absorbed)
        .filter(|atom| atom.emptiness() != Verdict::Empty)
        .collect();
    kept.sort();
    kept.dedup();
    (kept.len() <= MAX_ATOMS).then_some(kept)
}

#[cfg(test)]
mod tests {
    use super::{Entry, KEY_KINDS, Label, MapLattice, key_slot};
    use crate::decision::{Kind, Verdict};
    use crate::descr::integers::IntSet;

    /// One dict entry: a `str` key with this text, mapping to this integer.
    fn at(label: &str, value: i64) -> Entry<i64> {
        Entry {
            label: Some(Label::str(label)),
            kind: Some(Kind::Str),
            value,
        }
    }

    /// The union holds the dicts of both sides, and is a union rather than a
    /// refusal.
    ///
    /// Read at this level because the layer above absorbs a refusal: a kind's
    /// lines merge two structures by joining them, and where the join declines
    /// the two stay separate lines -- the same set, spelled longer. So a union
    /// that never succeeded would still denote the right dicts, and only the
    /// form would say so.
    #[test]
    fn a_union_of_maps_holds_the_dicts_of_both() {
        let a = MapLattice::label(Label::str("a"), IntSet::just(1), false);
        let b = MapLattice::label(Label::str("b"), IntSet::just(2), false);
        let joined = a.union(&b).expect("two maps join");
        assert_eq!(joined.emptiness(), Verdict::Inhabited);
        assert!(joined.holds(&[at("a", 1)]));
        assert!(joined.holds(&[at("b", 2)]));
        assert!(!joined.holds(&[at("a", 2)]));
        assert!(!joined.holds(&[]));
    }

    /// A meet is (12): the labels are shared and each side reads one it does not
    /// name off its own default.
    #[test]
    fn a_meet_of_maps_holds_only_the_dicts_of_both() {
        let a = MapLattice::label(Label::str("a"), IntSet::just(1), false);
        let b = MapLattice::label(Label::str("b"), IntSet::just(2), false);
        let met = a.intersect(&b).expect("two maps meet");
        assert!(met.holds(&[at("a", 1), at("b", 2)]));
        assert!(!met.holds(&[at("a", 1)]));
        // Two maps that disagree about one label share no dict at all.
        let other = MapLattice::label(Label::str("a"), IntSet::just(2), false);
        let disagreeing = a.intersect(&other).expect("two maps meet");
        assert_eq!(disagreeing.emptiness(), Verdict::Empty);
    }

    /// A complement is a map, and a dict is outside an atom by a missing key or
    /// a wrong one.
    #[test]
    fn a_complement_of_a_map_is_a_map() {
        let a = MapLattice::label(Label::str("a"), IntSet::just(1), false);
        let outside = a.complement();
        assert!(!outside.holds(&[at("a", 1)]));
        assert!(outside.holds(&[at("a", 2)]));
        assert!(outside.holds(&[]));
        // And complementing twice comes back to the dicts it started with.
        let back = outside.complement();
        assert!(back.holds(&[at("a", 1)]));
        assert!(!back.holds(&[at("a", 2)]));
    }

    /// A constraint about one part narrows the labels of that part alone.
    ///
    /// "No key outside `besides` of *this* part maps into `ty`" says nothing
    /// about a label of another part, and narrowing one would make the
    /// complement hold too few dicts. Every label here is a `str`, so a
    /// constraint about the integer keys must leave them alone.
    #[test]
    fn a_constraint_about_one_part_leaves_the_other_parts_labels_alone() {
        // "some integer key maps outside {1}", met with "b maps to 2".
        let int_keys = MapLattice::keyed(Kind::Int, IntSet::just(1));
        let some_other_int = int_keys.complement();
        let b_is_two = MapLattice::label(Label::str("b"), IntSet::just(2), false);
        let met = some_other_int.intersect(&b_is_two).expect("the two meet");
        // Its complement holds a dict whose `b` is 2 and whose integer keys are
        // all 1 -- the label is untouched by a constraint about another part.
        let outside = met.complement();
        assert!(outside.holds(&[at("b", 2)]));
    }

    /// A constraint about the labels' own part narrows them, and the complement
    /// is wrong without it.
    ///
    /// "No `str` key outside `besides` maps into `ty`" constrains the default for
    /// that part *and* every label the exclusion set does not cover -- a label is
    /// a `str` key too. Leaving the labels at their top would let the complement
    /// hold a dict the atom itself holds.
    #[test]
    fn a_constraint_about_the_labels_own_part_narrows_them() {
        // "some str key maps outside {1}", met with "b maps to 2".
        let str_keys = MapLattice::keyed(Kind::Str, IntSet::just(1));
        let some_other_str = str_keys.complement();
        let b_is_two = MapLattice::label(Label::str("b"), IntSet::just(2), false);
        let met = some_other_str.intersect(&b_is_two).expect("the two meet");
        // `{"b": 2}` is in the meet: `b` maps to 2, and `b` is itself the str key
        // that maps outside {1}.
        assert!(met.holds(&[at("b", 2)]));
        // So it must be outside the complement -- which it is only if the
        // constraint narrowed `b` along with the default.
        assert!(!met.complement().holds(&[at("b", 2)]));
    }

    /// The key partition covers every key, the kindless one included.
    ///
    /// A part missing from the array would leave the keys of that kind governed
    /// by nothing, and an atom would admit a dict it says nothing about.
    #[test]
    fn every_key_falls_in_exactly_one_part() {
        for kind in KEY_KINDS {
            assert!(key_slot(Some(kind)).is_some(), "{kind:?}");
        }
        // A key of no listed kind has the last part rather than none.
        assert_eq!(key_slot(None), Some(KEY_KINDS.len()));
        // An unhashable kind is no key at all.
        for kind in [Kind::List, Kind::Set, Kind::Dict] {
            assert_eq!(key_slot(Some(kind)), None, "{kind:?}");
        }
    }
}
