//! Maps as record atoms with per-kind defaults and a negative set.
//!
//! A dict is a *quasi K-step function* from keys to values [Cas23, Def. 4.6],
//! after Frisch's quasi-constant ones [Cas23, Def. 2.2], total on the keys and
//! constant on all but finitely many of the keys of each part of a partition
//! of them. So a set of dicts is written by naming the finitely many keys it
//! constrains and saying what every other key of each part maps to -- which is
//! the atom of [Cas23, Def. 4.1],
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
//! Three properties do the work, [Cas23, (11)], [Cas23, (12)] and
//! [Cas23, (13)], written out below.
//!
//! ```text
//! ⟨τ; t₀; S⟩ ≠ ∅   ⟺   ∀ℓ. τ_ℓ ≠ ∅   and   ∀s ∈ S. t₀ ∧ s ≠ ∅              (11)
//! ⟨τ; t₀; S⟩ ∧ ⟨τ′; t′₀; S′⟩  =  ⟨τ ∧ τ′; t₀ ∧ t′₀; S ∪ S′⟩                (12)
//! ⟨τ; t₀; S⟩ ∖ ⟨τ′; t′₀; ∅⟩  =  ⟨τ; t₀; S ∪ {¬t′₀}⟩ ∨ ⋁_{ℓ₀} ⟨τ^{ℓ₀}; t₀; S⟩ (13)
//! ```
//!
//! where `τ^{ℓ₀}_ℓ` is `τ_ℓ ∧ not(τ′_ℓ)` at `ℓ₀` and `τ_ℓ` elsewhere. Read the
//! third as the ways to *fail* an atom: some named key's value is outside its
//! type, or some unnamed key's value is outside the default. Finitely many,
//! each an atom again, which is what keeps the representation closed.
//!
//! **The default is per key kind.** The maps of [Cas23, §4.4] are indexed by a
//! partition of the key space whose parts are pairwise disjoint -- "we propose to
//! forbid any overlapping between domains in the same map" -- because a key
//! belonging to two domains would have two types and the theory would be that of
//! function intersections instead [Cas23, §4.5]. The kind partition is such a
//! partition but for its two finite parts, `None` and `bool`, whose keys the
//! paper would write as labels, and using it is what lets the atom stay a
//! function. A key schema
//! that is neither a kind nor a constant has no part to land in, and the frontend
//! refuses it rather than approximating.
//!
//! **A `str` part may be split further, by regular languages.** The partition of
//! [Cas23, Def. 4.6] is any finite one, and the paper fixes it in advance as a
//! design and implementation choice [Cas23, §4.5], for the reason it names: two
//! maps with overlapping domains meet in an atom whose domains overlap. Words
//! meet in that same atom and stay a partition, since a regular language has a
//! complement and two of them an intersection, which is how a checker over
//! JSON Schema's `patternProperties` keeps them apart [HSHP20, §4.1]. So an atom
//! holds its own split of the `str` keys ([`Narrowed`]), and a meet reads the two
//! splits at their common refinement. A cell holding few keys is not read as
//! having a fresh one: the hypothesis [Cas23, §4.4] states for [Cas23, Lem. 4.7],
//! that a key type is infinite, is met by a cell's holding more keys than the atom
//! can use, and its keys are pinned otherwise, as a finite part's are. A word is
//! one key to this, and a label a word of a language: true of a key whose hash and
//! `==` are its text's, and named in `docs/14-soundness.md` for one whose are not.

use std::collections::{BTreeMap, BTreeSet};

use super::budget;
use super::polar::{PolarUnion, Summand, tidy};
use super::regular::{Alphabet, RegularSet};
use super::symbolic::Guard;
use super::values::{Field, Values};
use crate::kind::Kind;
use crate::unordered_pairs;
use crate::verdict::Verdict;

/// The most atoms a union may carry.
///
/// A meet multiplies the two sides' counts and a difference adds one atom per
/// label and per key kind, so the union needs a bound for the reason every other
/// union here does: past it there is no sound set to substitute, since one too
/// wide is complemented into one too narrow.
pub(crate) const MAX_ATOMS: usize = 256;

/// The listed kinds a dict key may have: the hashable ones.
///
/// A `list`, a `set` and a `dict` are unhashable, so no key has those kinds and
/// the atom carries no default for them. The array is what makes the default a
/// function on a partition rather than a set of clauses that might overlap.
pub(crate) const KEY_KINDS: [Kind; 8] = [
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
pub(crate) const PARTS: usize = KEY_KINDS.len() + 1;

/// The part a key of `kind` falls in, or `None` for a kind no key can have.
///
/// The parts are the key types, pairwise disjoint and covering the keys
/// [Cas23, §4.4], each with a default of its own [Cas23, Def. 4.6].
///
/// A `list`, a `set` and a `dict` are unhashable, so a dict carrying one as a key
/// is not a dict at all, and every atom declines it.
pub(crate) fn key_slot(kind: Option<Kind>) -> Option<usize> {
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
pub(crate) enum Label {
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
    pub(crate) fn str(text: &str) -> Label {
        Label::Word(text.as_bytes().to_vec(), Kind::Str)
    }

    /// The kind of the key, which is the part of the partition it falls in.
    #[must_use]
    pub(crate) fn kind(&self) -> Kind {
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
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Entry<V> {
    /// The key, where it is a constant an atom could name.
    pub(crate) label: Option<Label>,
    /// The key's kind, or `None` where the key is of no listed kind.
    pub(crate) kind: Option<Kind>,
    /// What the key maps to.
    pub(crate) value: V,
}

/// One constraint of `S`: some key outside `besides`, of this part, maps into
/// `ty`.
///
/// The kind travels with the type because the default does: "some unnamed key
/// maps outside `t₀`" is a different claim per part of the partition, and
/// [Cas23, (13)] produces one such constraint per part.
///
/// **The labels travel with it too, and that is not the paper's shape.** The
/// paper reads `S` against the atom's own `L`, and can, because
/// [Cas23, Thm. 4.2] fixes one `L` for every atom of a normal form before any
/// of them is combined. Here each atom carries its own labels, and a meet
/// unions them -- so an `S` read against "the atom's labels" would quietly
/// *strengthen* as the atom learned more names, and `A ∧ B` would hold fewer
/// maps than `A` and `B` share. Naming the exclusion set at the point the
/// constraint is made keeps its meaning fixed under every later operation,
/// which is the same fixing the paper does globally.
#[derive(Debug, Clone, Eq, PartialOrd, Ord)]
struct Wanted<G> {
    slot: usize,
    ty: Values<G>,
    besides: BTreeSet<Label>,
    /// The `str` keys the constraint may be met by, where it is about part of
    /// them: the failure of one cell of a [`Narrowed`] split.
    keys: Option<RegularSet>,
}

/// Equal where every part is, the slot and the labels read before the type,
/// which is a descriptor.
impl<G: PartialEq> PartialEq for Wanted<G> {
    fn eq(&self, other: &Wanted<G>) -> bool {
        self.slot == other.slot
            && self.besides == other.besides
            && self.keys == other.keys
            && self.ty == other.ty
    }
}

/// Whether `label` is a key a constraint reading `keys` may be met by.
fn among(label: &Label, keys: Option<&RegularSet>) -> bool {
    keys.is_none_or(|keys| matches!(label, Label::Word(word, Kind::Str) if keys.holds(word)))
}

/// Whether `slot` is the part of the `str` keys, the one a split divides.
fn is_str_part(slot: usize) -> bool {
    matches!(KEY_KINDS.get(slot), Some(Kind::Str))
}

/// The keys a constraint of `S` may be pinned to, each with what it maps to.
type Listed<'a, G> = Vec<(Label, &'a Values<G>)>;

/// Some `str` keys a clause narrows, and what a key of them maps to.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Cell<G> {
    keys: RegularSet,
    field: Field<G>,
}

/// A split of the `str` part: disjoint cells read in place of its default, and
/// the keys none of them holds, which the default goes on reading. A quasi
/// `K`-step function over a partition finer than the kinds [Cas23, Def. 4.6].
///
/// `rest` is kept rather than read off the cells because a complement is total
/// and reads it, and a union of languages can refuse past the state bound.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Narrowed<G> {
    cells: Vec<Cell<G>>,
    rest: RegularSet,
}

impl<G: Guard> Narrowed<G> {
    /// The split `keys` read `field` and every other key the default.
    fn of(keys: &RegularSet, field: Field<G>) -> Narrowed<G> {
        Narrowed {
            rest: keys.complement(),
            cells: vec![Cell {
                keys: keys.clone(),
                field,
            }],
        }
    }

    /// A cell holding `keys`, joined to the one that reads the same field, or
    /// nothing where `keys` is empty.
    fn add(cells: &mut Vec<Cell<G>>, keys: RegularSet, field: Field<G>) -> Option<()> {
        if !budget::spend() {
            return None;
        }
        if keys.is_empty() {
            return Some(());
        }
        match cells.iter_mut().find(|cell| cell.field == field) {
            Some(cell) => cell.keys = cell.keys.union(&keys)?,
            None => cells.push(Cell { keys, field }),
        }
        Some(())
    }

    /// A side's cells and the keys they leave: no cells and every key, for a side
    /// with no split.
    fn read(split: Option<&Narrowed<G>>) -> (&[Cell<G>], RegularSet) {
        match split {
            Some(split) => (&split.cells, split.rest.clone()),
            None => (&[], RegularSet::all(Alphabet::Text)),
        }
    }

    /// The two splits at their common refinement, each cell's field met with what
    /// the other reads there: the rewrite that keeps overlapping patterns disjoint
    /// [HSHP20, §4.1]. An atom holding no split reads its default on every word.
    fn meet(
        mine: Option<&Narrowed<G>>,
        my_default: &Field<G>,
        theirs: Option<&Narrowed<G>>,
        their_default: &Field<G>,
    ) -> Option<Box<Narrowed<G>>> {
        let (my_cells, my_rest) = Self::read(mine);
        let (their_cells, their_rest) = Self::read(theirs);
        let mut cells = Vec::new();
        for a in my_cells {
            for b in their_cells {
                Self::add(
                    &mut cells,
                    a.keys.intersect(&b.keys)?,
                    a.field.meet(&b.field)?,
                )?;
            }
            Self::add(
                &mut cells,
                a.keys.intersect(&their_rest)?,
                a.field.meet(their_default)?,
            )?;
        }
        for b in their_cells {
            Self::add(
                &mut cells,
                b.keys.intersect(&my_rest)?,
                my_default.meet(&b.field)?,
            )?;
        }
        cells.sort();
        Some(Box::new(Narrowed {
            cells,
            rest: my_rest.intersect(&their_rest)?,
        }))
    }
}

/// Whether two labels name one key of a dict.
///
/// `True` hashes as `1` and equals it, and `False` as `0`, so `{1: a, True: b}`
/// is a dict of one entry. Every other pair of distinct labels is two keys: the
/// kinds are disjoint and, within a kind, two constants that differ are two
/// keys.
fn one_key(a: &Label, b: &Label) -> bool {
    match (a, b) {
        (Label::Int(n), Label::Bool(flag)) | (Label::Bool(flag), Label::Int(n)) => {
            *n == i64::from(*flag)
        }
        _ => false,
    }
}

/// The most keys the search for a dict meeting an atom's pinned constraints
/// tries, before it declines.
///
/// Each constraint of `S` is given one of the keys that could meet it, so the
/// search is a product of those counts. It ends without the bound: an atom pins
/// finitely many constraints and each has finitely many keys that could meet
/// it, so the product is finite, and the bound caps what that product costs. It
/// is small wherever an atom came out of the schemas a caller writes, and past
/// the bound the atom is *unknown*, which is sound: nothing is proved empty or
/// inhabited.
pub(crate) const MAX_ASSIGNMENTS: usize = 4096;

/// Every key of a part with finitely many, or `None` for a part with
/// infinitely many.
///
/// `bool`'s part has two keys and `None`'s one, and those are the parts where
/// a key no label names is not always there to meet a constraint: the paper
/// writes such a part's keys as labels [Cas23, §4.4], since its proof takes a
/// key no label names [Cas23, App. C.1].
fn finite_keys(slot: usize) -> Option<Vec<Label>> {
    match KEY_KINDS.get(slot)? {
        Kind::Bool => Some(vec![Label::Bool(false), Label::Bool(true)]),
        Kind::NoneType => Some(vec![Label::NoneType]),
        _ => None,
    }
}

/// A constraint of `S` no fresh key is known to meet, and the keys that could.
struct Pinned<'a, G> {
    /// What the key's value must be in.
    ty: &'a Values<G>,
    /// Whether a fresh key might meet it, which the default could not settle.
    fresh: bool,
    /// Each key the constraint may be met by, with the type the atom gives it.
    keys: Vec<(Label, &'a Values<G>)>,
}

/// The search for keys meeting every pinned constraint of one atom.
struct Search<'a, G> {
    /// The keys every dict of the atom carries.
    required: Vec<&'a Label>,
    /// The keys given a constraint so far, with what each must hold by now.
    held: Vec<(Label, Values<G>)>,
    /// The keys tried, against [`MAX_ASSIGNMENTS`].
    steps: usize,
}

impl<G: Guard> Search<'_, G> {
    /// What is known about some dict meeting every constraint in `pinned`,
    /// with the keys held so far: the constraints [Cas23, (11)] reads one at a
    /// time, read together where they share a key.
    fn assign(&mut self, pinned: &[Pinned<'_, G>]) -> Verdict {
        let Some((want, rest)) = pinned.split_first() else {
            return Verdict::Inhabited;
        };
        // A fresh key, where the default could not say whether one meets it.
        let mut verdict = if want.fresh {
            Verdict::every([Verdict::Unknown, self.assign(rest)].into_iter())
        } else {
            Verdict::Empty
        };
        for (key, ty) in &want.keys {
            if verdict == Verdict::Inhabited {
                break;
            }
            self.steps += 1;
            if self.steps > MAX_ASSIGNMENTS {
                return Verdict::any([verdict, Verdict::Unknown].into_iter());
            }
            let held = self.held.iter().position(|(other, _)| other == key);
            // A key new to the dict must be one it can carry beside the rest.
            if held.is_none()
                && self
                    .required
                    .iter()
                    .copied()
                    .chain(self.held.iter().map(|(other, _)| other))
                    .any(|other| one_key(other, key))
            {
                continue;
            }
            let before = held.and_then(|at| self.held.get(at)).map(|(_, now)| now);
            let Some(shared) = before.unwrap_or(ty).meet(want.ty) else {
                verdict = Verdict::any([verdict, Verdict::Unknown].into_iter());
                continue;
            };
            let here = shared.emptiness();
            if here == Verdict::Empty {
                continue;
            }
            let after = if let Some(at) = held {
                self.within(at, shared, rest)
            } else {
                self.held.push((key.clone(), shared));
                let after = self.assign(rest);
                self.held.pop();
                after
            };
            verdict =
                Verdict::any([verdict, Verdict::every([here, after].into_iter())].into_iter());
        }
        verdict
    }

    /// [`assign`](Self::assign) with the key held at `at` asked for `shared`
    /// instead of what it held, and given it back after.
    fn within(&mut self, at: usize, shared: Values<G>, rest: &[Pinned<'_, G>]) -> Verdict {
        let Some((_, now)) = self.held.get_mut(at) else {
            return Verdict::Unknown;
        };
        let before = core::mem::replace(now, shared);
        let after = self.assign(rest);
        if let Some((_, now)) = self.held.get_mut(at) {
            *now = before;
        }
        after
    }
}

/// One map atom, `⟨(τ_ℓ)_{ℓ∈L} ; t₀ ; S⟩`.
///
/// Both collections are ordered, so two ways of writing one atom compare equal.
#[derive(Debug, Clone, Eq, PartialOrd, Ord)]
struct MapAtom<G> {
    /// `(τ_ℓ)_{ℓ∈L}`, keyed by the constant each names.
    labels: BTreeMap<Label, Field<G>>,
    /// `t₀`, one entry per key kind.
    defaults: [Field<G>; PARTS],
    /// The split of the `str` part, where a clause narrows its keys. Boxed because
    /// nearly every atom has none, and an atom is moved and compared often.
    narrowed: Option<Box<Narrowed<G>>>,
    /// `S`.
    wanted: Vec<Wanted<G>>,
}

/// Equal where every part is, the parts compared exactly read before the
/// types, which are descriptors.
///
/// A union keeps an atom once by scanning the atoms it holds for an equal one
/// ([`Summand::admit`]), so equality is asked of every pair a product builds
/// against every atom kept, and almost always answers `false`. Atoms of one
/// union share most of their shape, so the derived order -- the labels with
/// their types, then the defaults, then `S` -- read a descriptor before the
/// slot or the key that tells two atoms apart. The keys, the flags and the
/// slots are compared first, and the types only between atoms they leave
/// equal; the answer is the derived one.
impl<G: PartialEq> PartialEq for MapAtom<G> {
    fn eq(&self, other: &MapAtom<G>) -> bool {
        self.labels.len() == other.labels.len()
            && self.wanted.len() == other.wanted.len()
            && self.labels.keys().eq(other.labels.keys())
            && self
                .labels
                .values()
                .zip(other.labels.values())
                .all(|(mine, theirs)| mine.absent == theirs.absent)
            && self
                .defaults
                .iter()
                .zip(&other.defaults)
                .all(|(mine, theirs)| mine.absent == theirs.absent)
            && self
                .wanted
                .iter()
                .zip(&other.wanted)
                .all(|(mine, theirs)| (mine.slot, &mine.besides) == (theirs.slot, &theirs.besides))
            && self.labels == other.labels
            && self.defaults == other.defaults
            && self.narrowed == other.narrowed
            && self.wanted == other.wanted
    }
}

impl<G: Guard> MapAtom<G> {
    /// Every dict: no label named, every key free to map anywhere.
    fn top() -> MapAtom<G> {
        MapAtom {
            labels: BTreeMap::new(),
            defaults: core::array::from_fn(|_| Field::top()),
            narrowed: None,
            wanted: Vec::new(),
        }
    }

    /// What the atom says about a label it may or may not name: its own field,
    /// or its part's default [Cas23, Def. 4.6].
    fn field(&self, label: &Label) -> Field<G> {
        match self.labels.get(label) {
            Some(field) => field.clone(),
            // A label the atom does not name is governed by the default for its
            // own part, which is what the quasi K-step function maps it to --
            // so reading it here is exact rather than a widening.
            None => self.unnamed(label),
        }
    }

    /// What a key the labels do not name maps to: the cell holding it where the
    /// `str` part is split, and its part's default otherwise.
    fn unnamed(&self, label: &Label) -> Field<G> {
        if let (Some(split), Label::Word(word, Kind::Str)) = (&self.narrowed, label)
            && let Some(cell) = split.cells.iter().find(|cell| cell.keys.holds(word))
        {
            return cell.field.clone();
        }
        self.default_for(Some(label.kind()))
    }

    /// The same atom with every label its part's default already covers dropped.
    ///
    /// This is the paper's `dom`, read up to the equality of fields, a label
    /// set against the default of its own part [Cas23, Def. 4.6]. The paper
    /// notes [Cas23, Def. 2.2] that the notation is "not univocal (unless we
    /// require zᵢ ≠ z...)": naming a key and giving it exactly what the default
    /// already gives it says nothing, and two atoms that differ only in such a
    /// name are one atom. Equality is finer than a field's set where its
    /// descriptor's is (`Descr`), so a name it misses stays: a second spelling
    /// of one atom, never a wrong one. Dropping the names is what makes `{"a?":
    /// nothing}` and `{}` the same closed map rather than two that merely admit
    /// the same dicts.
    fn absorbed(mut self) -> MapAtom<G> {
        let covered: Vec<Label> = self
            .labels
            .iter()
            .filter(|(label, field)| **field == self.unnamed(label))
            .map(|(label, _)| label.clone())
            .collect();
        for label in covered {
            self.labels.remove(&label);
        }
        self
    }

    /// The `str` keys a constraint may be met by, each with what it maps to: the
    /// split's cells and the rest, each read inside `keys`, or `None` for the part
    /// read whole.
    fn regions(&self, want: &Wanted<G>) -> Option<Vec<(Option<RegularSet>, &Field<G>)>> {
        let default = self.defaults.get(want.slot)?;
        let inside = |keys: &RegularSet| match &want.keys {
            Some(only) => keys.intersect(only),
            None => Some(keys.clone()),
        };
        let Some(split) = &self.narrowed else {
            return Some(vec![(want.keys.clone(), default)]);
        };
        let mut regions = vec![(inside(&split.rest), default)];
        regions.extend(
            split
                .cells
                .iter()
                .map(|cell| (inside(&cell.keys), &cell.field)),
        );
        Some(regions)
    }

    /// [`emptiness`](Self::emptiness) of one constraint over a split: whether a
    /// fresh key meets it, and the keys it may be pinned to otherwise.
    ///
    /// A region holding more keys than the atom names and constrains has one
    /// left for this constraint whatever the others take, which is what the
    /// infinite key type gives [Cas23, Lem. 4.7]. A region holding fewer lists
    /// them, and each is read as a label the atom does not name.
    fn over_a_split(&self, want: &Wanted<G>) -> Option<(Verdict, Listed<'_, G>)> {
        let room = self.labels.len() + want.besides.len() + self.wanted.len();
        let mut fresh = Verdict::Empty;
        let mut pinned = Vec::new();
        for (keys, field) in self.regions(want)? {
            let Some(keys) = keys else {
                fresh = Verdict::any([fresh, Verdict::Unknown].into_iter());
                continue;
            };
            if let Some(words) = keys.words(room) {
                pinned.extend(
                    words
                        .into_iter()
                        .map(|word| Label::Word(word, Kind::Str))
                        .filter(|key| !self.labels.contains_key(key) && !want.besides.contains(key))
                        .map(|key| (key, &field.ty)),
                );
            } else {
                let here = field
                    .ty
                    .meet(&want.ty)
                    .map_or(Verdict::Unknown, |shared| shared.emptiness());
                fresh = Verdict::any([fresh, here].into_iter());
            }
        }
        Some((fresh, pinned))
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

    /// What is known about this atom holding a dict, as [Cas23, (11)] reads it.
    ///
    /// Empty as soon as a named key's type is, or no dict can satisfy every
    /// constraint of `S` at once.
    ///
    /// The property reads each constraint of `S` against the default alone, and
    /// each on its own. Both rest on the hypothesis [Cas23, §4.4] states for
    /// [Cas23, Lem. 4.7], that every key type is infinite: a constraint is met
    /// by a key no label names, and a second constraint by a second such key.
    /// That holds here of a part with infinitely many keys, and a constraint
    /// its default can meet is met by a fresh key of its own. The rest are
    /// **pinned** to keys the atom can name -- a label of the part, or a key of
    /// `bool`'s part or `None`'s, which have two keys and one -- and two pinned
    /// to one key ask that key for a value in both. So those are read together:
    /// each is given a key, every key a value in what each of its constraints
    /// asks, and the keys the dict then carries must be keys one dict can hold.
    fn emptiness(&self) -> Verdict {
        let labels = Verdict::every(self.labels.values().map(Field::emptiness));
        if labels == Verdict::Empty {
            return Verdict::Empty;
        }
        // The keys every dict of the atom carries: a label it does not let go
        // missing. `True` hashes as `1` and equals it, so a dict has one entry
        // for the two and no dict carries both -- the parts are separate here,
        // because `Literal[1]` and `Literal[True]` are disjoint *sets*, but an
        // atom requiring both keys describes a dict Python cannot build.
        let required: Vec<&Label> = self
            .labels
            .iter()
            .filter(|(_, field)| !field.absent)
            .map(|(label, _)| label)
            .collect();
        if unordered_pairs(&required).any(|(a, b)| one_key(a, b)) {
            return Verdict::Empty;
        }
        let mut pinned = Vec::new();
        for want in &self.wanted {
            // A want past the key partition is no element of the model -- the
            // model gives a default on the partition and nowhere else
            // [Cas23, Def. 4.6] -- so an atom carrying one is proved nothing,
            // as the search's own reading of the same miss declines.
            let Some(default) = self.defaults.get(want.slot) else {
                return Verdict::Unknown;
            };
            let split = is_str_part(want.slot) && (self.narrowed.is_some() || want.keys.is_some());
            let finite = finite_keys(want.slot);
            let (fresh, listed) = if split {
                let Some(read) = self.over_a_split(want) else {
                    return Verdict::Unknown;
                };
                read
            } else if finite.is_some() {
                (Verdict::Empty, Vec::new())
            } else {
                // Past a guard's own bound there is no set to read, so nothing
                // is proved either way.
                let fresh = default
                    .ty
                    .meet(&want.ty)
                    .map_or(Verdict::Unknown, |shared| shared.emptiness());
                (fresh, Vec::new())
            };
            if fresh == Verdict::Inhabited {
                continue;
            }
            let named = self.labels.iter().filter(|(label, _)| {
                key_slot(Some(label.kind())) == Some(want.slot)
                    && !want.besides.contains(*label)
                    && among(label, want.keys.as_ref())
            });
            let named = named.map(|(label, field)| (label.clone(), &field.ty));
            let unnamed = finite
                .into_iter()
                .flatten()
                .filter(|key| !self.labels.contains_key(key) && !want.besides.contains(key));
            let keys = named
                .chain(listed)
                .chain(unnamed.map(|key| (key, &default.ty)))
                .collect();
            pinned.push(Pinned {
                ty: &want.ty,
                fresh: fresh == Verdict::Unknown,
                keys,
            });
        }
        let mut search = Search {
            required,
            held: Vec::new(),
            steps: 0,
        };
        Verdict::every([labels, search.assign(&pinned)].into_iter())
    }

    /// The dicts in both atoms, met as [Cas23, (12)] meets them.
    ///
    /// The label sets are unioned first, each side reading a label it does not
    /// name off its own default -- which is what the quasi K-step function maps
    /// it to [Cas23, Def. 4.6], so the extension is exact rather than a
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
        let narrowed = if self.narrowed.is_none() && other.narrowed.is_none() {
            None
        } else {
            Some(self.narrowed_with(other)?)
        };
        // `S ∪ S′`, each constraint once by equality: a type's equality can be
        // coarser than its order, so a sort need not put two equal
        // constraints side by side.
        let mut wanted = self.wanted.clone();
        for want in &other.wanted {
            if !wanted.contains(want) {
                wanted.push(want.clone());
            }
        }
        wanted.sort();
        Some(MapAtom {
            labels,
            defaults: defaults.try_into().ok()?,
            narrowed,
            wanted,
        })
    }

    /// The split a meet of this atom and `other` reads the `str` keys by. Out of
    /// line, since nearly every meet holds none.
    #[inline(never)]
    fn narrowed_with(&self, other: &MapAtom<G>) -> Option<Box<Narrowed<G>>> {
        let slot = KEY_KINDS.iter().position(|kind| *kind == Kind::Str)?;
        Narrowed::meet(
            self.narrowed.as_deref(),
            self.defaults.get(slot)?,
            other.narrowed.as_deref(),
            other.defaults.get(slot)?,
        )
    }

    /// The ways to fail this atom, as atoms, by [Cas23, (13)].
    ///
    /// An atom is `P ∖ N` [Cas23, Def. 4.1], with `P = ⟨τ; t₀; ∅⟩` and `N` the
    /// maps that meet none of `S`. So `¬A` is `¬P ∨ N`, and there are three
    /// ways in:
    ///
    /// - a **named** key's value is outside what the atom allows there, which is
    ///   the per-label disjunct of the difference, taken from the whole;
    /// - an **unnamed** key of some part maps outside that part's default, which
    ///   is its `¬t′₀` disjunct, one per part because the default is one per
    ///   part;
    /// - a constraint of `S` is not met: **no** unnamed key of its part
    ///   maps into it, which is the `N` above.
    ///
    /// The third is why this is a complement rather than a difference. The
    /// difference is stated for a subtrahend whose `S` is empty, and an atom
    /// that came out of a complement does not have one -- so the caller folds
    /// complements with a product instead of subtracting, and the case is
    /// handled here where the `S` is in reach.
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
            let split = self.narrowed.as_ref().filter(|_| is_str_part(slot));
            atom.wanted.push(Wanted {
                slot,
                ty: default.ty.complement(),
                besides: named.clone(),
                keys: split.map(|split| split.rest.clone()),
            });
            atoms.push(atom);
            // A key of a cell fails the atom by mapping outside that cell's field.
            for cell in split.iter().flat_map(|split| &split.cells) {
                let mut atom = widened();
                atom.wanted.push(Wanted {
                    slot,
                    ty: cell.field.ty.complement(),
                    besides: named.clone(),
                    keys: Some(cell.keys.clone()),
                });
                atoms.push(atom);
            }
        }
        // "No key outside `besides` maps into `ty`" constrains the default for
        // that part *and* every label the exclusion set does not cover.
        for want in &self.wanted {
            let mut atom = widened();
            let outside = Field {
                ty: want.ty.complement(),
                absent: true,
            };
            match (&want.keys, atom.defaults.get_mut(want.slot)) {
                (None, Some(default)) => *default = outside.clone(),
                // A constraint over part of the keys tightens that part alone.
                (Some(keys), Some(_)) => {
                    atom.narrowed = Some(Box::new(Narrowed::of(keys, outside.clone())));
                }
                // Past the partition the want is kept as it stands, so every
                // reading built from this atom declines on it as the atom it
                // came from does, rather than widening in silence.
                (_, None) => atom.wanted.push(want.clone()),
            }
            // A default governs the keys the atom does not name, so tightening
            // one reaches every key of the part -- the exclusion set included,
            // which is the one group the want says nothing about. Naming each of
            // them takes it back out of the default's reach. Without this, `¬¬A`
            // would come out *smaller* than `A`: the record `{a: int}`
            // complemented twice would forbid the key it is about.
            for label in &want.besides {
                if key_slot(Some(label.kind())) == Some(want.slot) {
                    atom.labels.entry(label.clone()).or_insert_with(Field::top);
                }
            }
            let narrowed: Vec<Label> = self
                .labels
                .keys()
                .filter(|label| {
                    key_slot(Some(label.kind())) == Some(want.slot)
                        && !want.besides.contains(*label)
                        && among(label, want.keys.as_ref())
                })
                .cloned()
                .collect();
            for label in narrowed {
                atom.labels.insert(label, outside.clone());
            }
            atoms.push(atom);
        }
        atoms
    }

    /// Whether this atom holds the dict carrying `entries`.
    #[cfg(test)]
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
            let read = match &entry.label {
                Some(label) => self.unnamed(label),
                None => self.default_for(entry.kind),
            };
            if unnamed(entry) && !read.ty.holds(&entry.value) {
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
                    && want.keys.as_ref().is_none_or(|keys| {
                        entry
                            .label
                            .as_ref()
                            .is_some_and(|label| among(label, Some(keys)))
                    })
                    && want.ty.holds(&entry.value)
            })
        })
    }
}

impl<G: Guard> Summand for MapAtom<G> {
    /// Nothing: an atom's universe is every dict.
    type Within = ();
    type Complement = Vec<MapAtom<G>>;
    const MAX: usize = MAX_ATOMS;

    fn top((): ()) -> MapAtom<G> {
        MapAtom::top()
    }

    fn meet(&self, other: &MapAtom<G>) -> Option<MapAtom<G>> {
        MapAtom::meet(self, other)
    }

    fn complement(&self, (): ()) -> Vec<MapAtom<G>> {
        MapAtom::complement(self)
    }

    /// The atom with the labels its defaults already say absorbed, unless it
    /// holds no dict or the list holds it already. A repeat kept would be a
    /// summand of width toward [`MAX_ATOMS`] and a factor in every complement
    /// of the union.
    fn admit(kept: &mut Vec<MapAtom<G>>, atom: MapAtom<G>) -> Option<()> {
        let atom = atom.absorbed();
        if atom.emptiness() != Verdict::Empty && !kept.contains(&atom) {
            kept.push(atom);
        }
        Some(())
    }
}

/// The dicts a descriptor admits, as a union of map atoms or their complement.
///
/// The polarity is the device every lattice built from parts carries, in
/// `descr/polar.rs`, and for the same reason: a complement must be total, and
/// its positive form can pass [`MAX_ATOMS`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MapLattice<G: Guard>(PolarUnion<MapAtom<G>>);

impl<G: Guard> MapLattice<G> {
    /// No dict at all.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn empty() -> MapLattice<G> {
        MapLattice(PolarUnion::of(Vec::new()))
    }

    /// Every dict: the one atom that names no label and lets every key map
    /// anywhere.
    #[must_use]
    pub(crate) fn all() -> MapLattice<G> {
        MapLattice(PolarUnion::of(vec![MapAtom::top()]))
    }

    /// The dicts whose `label` maps into `ty`, every other key free.
    ///
    /// `optional` admits the dicts without the key at all, which is the `⊥` in
    /// the field's type rather than a rule beside it.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn label(label: Label, ty: G, optional: bool) -> MapLattice<G> {
        let mut atom = MapAtom::top();
        atom.labels.insert(
            label,
            Field {
                ty: Values::Only(ty),
                absent: optional,
            },
        );
        MapLattice(PolarUnion::of(vec![atom.absorbed()]))
    }

    /// The dicts a keyed map spells, with no clause over part of the `str` keys.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn record(
        labels: impl IntoIterator<Item = (Label, G, bool)>,
        opened: impl IntoIterator<Item = (Option<Kind>, G)>,
    ) -> Option<MapLattice<G>> {
        Self::narrowed_record(labels, opened, [])
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
    /// catch-all forbids, a closed record giving every key it does not name
    /// `⊥` [Cas23, §3.2]. Two clauses over one part join, which is the IR's
    /// reading of overlapping clauses where the paper forbids them
    /// [Cas23, §4.4], so a key belongs when *some* clause admits it.
    ///
    /// `narrowed` holds the clauses over part of the `str` keys, and a cell of the
    /// split maps to the join of the clauses that hold it, the `str` part's default
    /// among them: two clauses overlap and the atom reads one function, a step
    /// function on the common refinement of their keys.
    #[must_use]
    pub(crate) fn narrowed_record(
        labels: impl IntoIterator<Item = (Label, G, bool)>,
        opened: impl IntoIterator<Item = (Option<Kind>, G)>,
        narrowed: impl IntoIterator<Item = (RegularSet, G)>,
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
        let narrowed: Vec<(RegularSet, G)> = narrowed.into_iter().collect();
        if !narrowed.is_empty() {
            atom.narrowed = Some(Self::split_of(narrowed, &atom)?);
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
        Some(MapLattice(PolarUnion::of(tidy(vec![atom])?)))
    }

    /// The split the narrowed clauses spell over `atom`'s default for `str` keys.
    fn split_of(narrowed: Vec<(RegularSet, G)>, atom: &MapAtom<G>) -> Option<Box<Narrowed<G>>> {
        let default = atom.defaults.get(key_slot(Some(Kind::Str))?)?;
        let mut split = Narrowed {
            cells: Vec::new(),
            rest: RegularSet::all(Alphabet::Text),
        };
        for (keys, ty) in narrowed {
            let widened = Field {
                ty: Values::Only(ty),
                absent: true,
            };
            let mut cells = Vec::new();
            for cell in split.cells {
                let both = cell.keys.intersect(&keys)?;
                Narrowed::add(&mut cells, both, cell.field.join(&widened)?)?;
                let outside = cell.keys.intersect(&keys.complement())?;
                Narrowed::add(&mut cells, outside, cell.field)?;
            }
            let fresh = split.rest.intersect(&keys)?;
            Narrowed::add(&mut cells, fresh, default.join(&widened)?)?;
            split = Narrowed {
                cells,
                rest: split.rest.intersect(&keys.complement())?,
            };
        }
        split.cells.sort();
        Some(Box::new(split))
    }

    /// The dicts every one of whose keys of `kind` maps into `ty`.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn keyed(kind: Kind, ty: G) -> MapLattice<G> {
        let mut atom = MapAtom::top();
        if let Some(slot) = key_slot(Some(kind))
            && let Some(default) = atom.defaults.get_mut(slot)
        {
            *default = Field {
                ty: Values::Only(ty),
                absent: true,
            };
        }
        MapLattice(PolarUnion::of(vec![atom]))
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
    #[cfg(test)]
    #[must_use]
    pub(crate) fn keys_among(parts: &[Option<Kind>]) -> MapLattice<G> {
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
        MapLattice(PolarUnion::of(vec![atom]))
    }

    /// What is known about this holding a dict.
    ///
    /// A negated form has to be expanded first, and a refusal there is *unknown*
    /// rather than inhabited: past the bound there is no union to read.
    #[must_use]
    pub(crate) fn emptiness(&self) -> Verdict {
        self.0.verdict((), MapAtom::emptiness)
    }

    /// Whether the dict carrying `entries` is held.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn holds(&self, entries: &[Entry<G::Value>]) -> bool {
        self.0.holds(|atom| atom.holds(entries))
    }

    /// The dicts in either, or `None` past [`MAX_ATOMS`].
    #[must_use]
    pub(crate) fn union(&self, other: &MapLattice<G>) -> Option<MapLattice<G>> {
        self.0.union(&other.0, ()).map(MapLattice)
    }

    /// The dicts in both, or `None` past [`MAX_ATOMS`] or where a guard refuses.
    ///
    /// Four two-field records against their corners are the shape that shows
    /// why a negated side is removed one atom at a time: rebuilt first, their
    /// complement passes the bound, while meeting each factor into what is
    /// already held drops the atoms holding no dict before the next factor
    /// multiplies them.
    #[must_use]
    pub(crate) fn intersect(&self, other: &MapLattice<G>) -> Option<MapLattice<G>> {
        self.0.intersect(&other.0, ()).map(MapLattice)
    }

    /// The dicts this does not hold.
    ///
    /// Total, which is what the [`Guard`] contract asks.
    #[must_use]
    pub(crate) fn complement(&self) -> MapLattice<G> {
        MapLattice(self.0.complement(()))
    }

    /// The atoms as held, whichever polarity reads them.
    #[cfg(test)]
    fn atoms(&self) -> &[MapAtom<G>] {
        self.0.summands()
    }

    /// Whether the atoms are the dicts *not* held.
    #[cfg(test)]
    const fn negated(&self) -> bool {
        self.0.is_negated()
    }
}

#[cfg(test)]
mod tests;
