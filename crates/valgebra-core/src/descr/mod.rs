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

pub mod budget;
pub mod classes;
pub mod floats;
pub mod integers;
pub mod interval;
mod lines;
pub mod lower;
pub mod maps;
pub mod records;
pub mod regular;
pub mod sets;
pub mod symbolic;
pub mod values;

use std::sync::{Arc, OnceLock};

use crate::decision::{Kind, Verdict};
use classes::Class;
use floats::FloatSet;
use integers::IntSet;
use lines::Lines;
use maps::{Entry, Label, MapLattice};
use records::RecordLattice;
use regular::{Alphabet, RegularSet};
use sets::SetLattice;
use symbolic::{Guard, SymbolicDfa};

/// The two booleans, as a subset.
///
/// `bool` is a kind with exactly two values, so a *finite set* over it is exact:
/// `Literal[True]` is `{True}`, and `Literal[True] | Literal[False]` is the
/// whole kind rather than a union the procedure must recognise. A two-bit set is
/// the smallest thing closed under the three operations here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

    pub(crate) const fn union(self, other: BoolSet) -> BoolSet {
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Component {
    /// Every value of the kind, or none.
    ///
    /// Exact for `None`, which has one value, and coarse for every other kind
    /// that still carries it: `set[int]` and `set[str]` are the same component,
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
    /// The sequences this descriptor admits, as an automaton over value sets.
    ///
    /// Serves `list` and `tuple` both, and the difference between them is the
    /// kind rather than the language: `list[int]` is a loop and `tuple[int, str]`
    /// is a chain, which one constructor spells. The letters are descriptors, so
    /// the component is recursive -- through the automaton's *states*, where the
    /// cycle is an edge and every guard stays a finite descriptor.
    Sequences(SymbolicDfa<Arc<Descr>>),
    /// The sets this descriptor admits, as a union of powerset lines.
    ///
    /// Serves `set` and `frozenset` both. A set is its *members* and nothing
    /// else -- there is no order for an automaton to walk -- so the component is
    /// the powerset of a descriptor rather than a language over it.
    Sets(SetLattice<Arc<Descr>>),
    /// The dicts this descriptor admits, as a union of map atoms.
    ///
    /// A dict is a *quasi-constant function* from keys to values, so the
    /// component names the finitely many keys it constrains and says what every
    /// other key of each kind maps to ([`maps`]).
    Maps(MapLattice<Arc<Descr>>),
}

impl Component {
    /// Every value of the kind.
    fn top(kind: Kind) -> Component {
        match kind {
            Kind::Bool => Component::Booleans(BoolSet::BOTH),
            Kind::Int => Component::Integers(IntSet::all()),
            Kind::Float => Component::Floats(FloatSet::all()),
            Kind::Str | Kind::Bytes => Component::Words(RegularSet::all()),
            Kind::List | Kind::Tuple => Component::Sequences(SymbolicDfa::all()),
            Kind::Set | Kind::FrozenSet => Component::Sets(SetLattice::all()),
            Kind::Dict => Component::Maps(MapLattice::all()),
            // `None` has one value, so all-or-nothing is exact for it.
            Kind::NoneType => Component::Coarse(true),
        }
    }

    /// What is known about this component admitting a value.
    ///
    /// Exact for every representation whose emptiness is a computation over a
    /// finite structure. The two held as a *union* can answer `Unknown`: past
    /// their bound there is no union to read, and a class constraint can rest on
    /// a subclass the core cannot enumerate.
    fn emptiness(&self) -> Verdict {
        let exact = |empty: bool| {
            if empty {
                Verdict::Empty
            } else {
                Verdict::Inhabited
            }
        };
        match self {
            Component::Coarse(present) => exact(!present),
            Component::Booleans(set) => exact(set.is_empty()),
            Component::Integers(set) => exact(set.is_empty()),
            Component::Floats(set) => exact(set.is_empty()),
            Component::Words(set) => exact(set.is_empty()),
            Component::Sequences(set) => exact(set.is_empty()),
            Component::Sets(set) => set.emptiness(),
            Component::Maps(set) => set.emptiness(),
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
            (Component::Integers(a), Component::Integers(b)) => {
                // Refused for the reason the languages below are, one
                // representation over: two steps a caller writes independently
                // meet at their least common multiple, which can be past the
                // period this holds even when each step is far inside it. A
                // rounded period describes a different set, so the descriptor
                // becomes unbuildable and the relation stays undecided.
                let combined = match op {
                    Op::Union => a.union(b),
                    Op::Intersect => a.intersect(b),
                };
                Component::Integers(combined?)
            }
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
            (Component::Sequences(a), Component::Sequences(b)) => {
                // Refused for the same reason, one alphabet up: a product of two
                // automata can pass the bound, and a language too wide is
                // complemented into one too narrow.
                let combined = match op {
                    Op::Union => a.union(b),
                    Op::Intersect => a.intersect(b),
                };
                Component::Sequences(combined?)
            }
            (Component::Maps(a), Component::Maps(b)) => {
                // A union of map atoms multiplies under a meet and grows by a
                // label and a key part under a difference, and refuses past the
                // bound for the reason the others do.
                let combined = match op {
                    Op::Union => a.union(b),
                    Op::Intersect => a.intersect(b),
                };
                Component::Maps(combined?)
            }
            (Component::Sets(a), Component::Sets(b)) => {
                // Refused for the same reason again: a union of powerset lines
                // multiplies under a meet, and past the bound there is no sound
                // union to substitute.
                let combined = match op {
                    Op::Union => a.union(b),
                    Op::Intersect => a.intersect(b),
                };
                Component::Sets(combined?)
            }
            (mine, theirs) => {
                debug_assert!(false, "combining {mine:?} with {theirs:?} of another kind");
                mine.clone()
            }
        })
    }

    /// Every value of the kind this component describes a part of.
    ///
    /// The kind read off the *representation* rather than passed alongside it: a
    /// line complementing its structure needs the whole of the kind to put
    /// beside the other half, and the variant already names which kind that is.
    fn top_like(&self) -> Component {
        match self {
            Component::Coarse(_) => Component::Coarse(true),
            Component::Booleans(_) => Component::Booleans(BoolSet::BOTH),
            Component::Integers(_) => Component::Integers(IntSet::all()),
            Component::Floats(_) => Component::Floats(FloatSet::all()),
            Component::Words(_) => Component::Words(RegularSet::all()),
            Component::Sequences(_) => Component::Sequences(SymbolicDfa::all()),
            Component::Sets(_) => Component::Sets(SetLattice::all()),
            Component::Maps(_) => Component::Maps(MapLattice::all()),
        }
    }

    /// Every value of the kind this component does not admit.
    fn complement(&self) -> Component {
        match self {
            Component::Coarse(present) => Component::Coarse(!present),
            Component::Booleans(set) => Component::Booleans(set.complement()),
            Component::Integers(set) => Component::Integers(set.complement()),
            Component::Floats(set) => Component::Floats(set.complement()),
            Component::Words(set) => Component::Words(set.complement()),
            Component::Sequences(set) => Component::Sequences(set.complement()),
            Component::Sets(set) => Component::Sets(set.complement()),
            Component::Maps(set) => Component::Maps(set.complement()),
        }
    }
}

/// The constant a value is, where a map atom could name it as a key.
///
/// `None` for a value no `Literal` names -- a float, a tuple, an object -- which
/// a map reads through the default for its part rather than by name.
fn label_of(value: &Value) -> Option<Label> {
    match value.kind? {
        Kind::NoneType => Some(Label::NoneType),
        Kind::Bool => value.boolean.map(Label::Bool),
        Kind::Int => value.integer.map(Label::Int),
        kind @ (Kind::Str | Kind::Bytes) => value.word.map(|word| Label::Word(word.to_vec(), kind)),
        _ => None,
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

/// The values a set may hold: everything but the three kinds Python cannot hash.
///
/// A `list`, a `set` and a `dict` are mutable and unhashable, so no set holds
/// one. Written as the complement of those three rather than as a list of the
/// rest, so a kind added later is hashable until someone says otherwise -- the
/// direction that leaves a set *larger*, which declines rather than admits.
fn hashable() -> Descr {
    let mut unhashable = Descr::nothing();
    for kind in [Kind::List, Kind::Set, Kind::Dict] {
        unhashable.put(kind, Component::top(kind));
    }
    unhashable.complement()
}

/// Whether a kind's values are sets of values, which is what the powerset
/// component reads.
fn is_set(kind: Kind) -> bool {
    matches!(kind, Kind::Set | Kind::FrozenSet)
}

/// Whether a kind's values are sequences of values, which is what the automaton
/// component reads.
///
/// A dict is a container too, but its elements are pairs, so it wants its own
/// rule. A set has no order for an automaton to walk and gets [`is_set`].
fn is_sequence(kind: Kind) -> bool {
    matches!(kind, Kind::List | Kind::Tuple)
}

/// Which way two components combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Union,
    Intersect,
}

/// A set of values, held as one component per [`Kind`] plus everything else.
///
/// **Canonical by construction, with one exception.** Each component is
/// canonical for its representation, and there is exactly one component per
/// kind, so two descriptors that differ in a canonical component admit
/// different values. Nothing needs normalising afterwards, which is what makes
/// the operations' laws structural rather than up-to-equivalence.
///
/// The exceptions are three. [`sets`] and [`records`] are held as a *union*, and
/// a union can hold the same values two ways -- `P(A ∪ B)` is also the union of
/// `P(A)`, `P(B)` and the line subtracting both -- which costs a search for
/// coverings neither runs. [`symbolic`] is the third once its letters are
/// descriptors: minimisation asks the guards to join and to say what they leave,
/// and a descriptor cannot always answer, because answering rebuilds the very
/// automata being minimised. Each such refusal leaves a coarser table for the
/// same language.
///
/// So equality is *finer* than agreeing on values once any of the three is
/// involved: equal descriptors still admit the same values, but two that admit
/// the same values may compare unequal. Every law over a descriptor that can
/// hold one is therefore checked against the values, and the laws checked by
/// equality are the ones over the components that are canonical.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Descr {
    /// One union of lines per kind, indexed by that kind's position in
    /// [`Kind::ALL`].
    kinds: [Lines; Kind::ALL.len()],
    /// The values of no listed kind -- a callable, a generator, an instance of a
    /// class that derives from none of the builtins.
    ///
    /// One more union of lines, over a structure this representation does not
    /// distinguish, so its lines are told apart by their classes and attributes
    /// alone. The slot exists so a complement means what it says: the complement
    /// of `int` holds every non-int value, not merely the ones this partition
    /// names.
    other: Lines,
}

/// The whole of the kindless slot, which its complements are taken against.
///
/// A value of no listed kind has no structure this representation distinguishes,
/// so the slot's every line carries the same coarse one and the classes and
/// attributes do the work.
const KINDLESS: Component = Component::Coarse(true);

impl Descr {
    /// The empty set.
    #[must_use]
    pub fn nothing() -> Descr {
        Descr {
            kinds: core::array::from_fn(|_| Lines::bottom()),
            other: Lines::bottom(),
        }
    }

    /// Every value.
    #[must_use]
    pub fn anything() -> Descr {
        Descr {
            kinds: Kind::ALL.map(|kind| Lines::everything(Component::top(kind))),
            other: Lines::everything(KINDLESS),
        }
    }

    /// Every value of `kind` and nothing else.
    #[must_use]
    pub fn of_kind(kind: Kind) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(kind, Component::top(kind));
        descr
    }

    /// Every value the objects `constraint` admits, of **any** kind.
    ///
    /// A class and an attribute record constrain a value within its kind rather
    /// than instead of it, so the constraint goes on one line of every kind --
    /// including the kindless slot. Meeting the result with a kind keeps that
    /// kind's line and empties the rest, which is how a dataclass deriving from
    /// `int` keeps both halves of what it is.
    fn objects(constraint: &RecordLattice<Arc<Descr>>) -> Descr {
        Descr {
            kinds: Kind::ALL.map(|kind| Lines::objects(&Component::top(kind), constraint.clone())),
            other: Lines::objects(&KINDLESS, constraint.clone()),
        }
    }

    /// The singleton holding one boolean.
    #[must_use]
    pub fn boolean(value: bool) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(Kind::Bool, Component::Booleans(BoolSet::just(value)));
        descr
    }

    /// The singleton holding one integer.
    #[must_use]
    pub fn integer(value: i64) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(Kind::Int, Component::Integers(IntSet::just(value)));
        descr
    }

    /// The singleton holding one float, which is empty for `nan`.
    #[must_use]
    pub fn float(value: f64) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(Kind::Float, Component::Floats(FloatSet::just(value)));
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
        descr.put(kind, Component::Words(language));
        Some(descr)
    }

    /// The one-word set, for a word kind. A `str` is its UTF-8 bytes.
    #[must_use]
    pub fn word(word: &[u8], kind: Kind) -> Option<Descr> {
        alphabet_of(kind)?;
        let mut descr = Descr::nothing();
        descr.put(kind, Component::Words(RegularSet::word(word)));
        Some(descr)
    }

    /// The sequences a shape spells, for a sequence kind.
    ///
    /// One constructor for the three spellings, because they are one shape with
    /// different parts filled in. `tuple[A, B]` is a prefix and no tail;
    /// `list[T]` is no prefix and a tail; `tuple[A, *tuple[B, ...], C]` is a
    /// prefix, a tail and a prefix, which is the same chain with a loop in it.
    ///
    /// `None` where the kind's values are not sequences, which is a caller error
    /// rather than a set.
    #[must_use]
    pub fn sequence(prefix: &[Descr], tail: Option<&Descr>, kind: Kind) -> Option<Descr> {
        if !is_sequence(kind) {
            return None;
        }
        // The shape's letters become the automaton's guards, so they are moved
        // behind a handle here -- once each, at the one place a caller's
        // descriptor becomes a letter.
        let prefix: Vec<Arc<Descr>> = prefix.iter().cloned().map(Arc::new).collect();
        let tail = tail.cloned().map(Arc::new);
        let mut descr = Descr::nothing();
        descr.put(
            kind,
            Component::Sequences(SymbolicDfa::shape(&prefix, tail.as_ref())),
        );
        Some(descr)
    }

    /// The sequences of `kind` holding at least `least` elements.
    ///
    /// A length is a *regular* property of a sequence -- "any letter, at least
    /// this many times" -- so the component that already holds sequences holds
    /// this too, and a length bound over a list stops being opaque. The letters
    /// are the top because the bound says nothing about what the elements are;
    /// the caller meets this with the shape that does.
    ///
    /// Refuses past the state bound rather than building an automaton with one
    /// state per element a caller asked for: `MinLen(100_000)` is a schema
    /// nobody writes and an automaton nobody can hold.
    #[must_use]
    pub fn sequences_at_least(least: usize, kind: Kind) -> Option<Descr> {
        Descr::sequences_at_least_within(least, symbolic::MAX_STATES, kind)
    }

    /// [`sequences_at_least`](Self::sequences_at_least) with the bound named,
    /// so the edge can be tested without building an automaton the size of
    /// the real one.
    fn sequences_at_least_within(least: usize, bound: usize, kind: Kind) -> Option<Descr> {
        if least > bound {
            return None;
        }
        let anything = Descr::anything();
        Descr::sequence(&vec![anything.clone(); least], Some(&anything), kind)
    }

    /// The sequences of `kind` holding at most `most` elements.
    ///
    /// The complement of "at least one more", taken inside the kind so the other
    /// ten kinds are not swept in with it.
    #[must_use]
    pub fn sequences_at_most(most: usize, kind: Kind) -> Option<Descr> {
        let longer = Descr::sequences_at_least(most.checked_add(1)?, kind)?;
        Descr::of_kind(kind).intersect(&longer.complement())
    }

    /// The sets whose members all lie in `elements`, for a set kind.
    ///
    /// The members are first cut down to what a set can *hold*. A set's members
    /// are hashed, and a list, a set and a dict are not hashable, so `elements`
    /// meets the hashable values before the powerset is taken. That is what makes
    /// `set[list[int]]` the same set as `set[nothing]`: neither holds a list, so
    /// both hold exactly one value, the empty set.
    ///
    /// The cut is coarser than Python's rule in one place, and coarser in the
    /// direction that declines rather than admits: a tuple is hashable only when
    /// its elements are, and this keeps every tuple. So `set[tuple[list[int]]]`
    /// reads as inhabited by more than the empty set when it is not, which
    /// leaves a question undecided rather than answering it wrongly.
    ///
    /// `None` where the kind's values are not sets, or where a component cannot
    /// hold the meet.
    #[must_use]
    pub fn set(elements: &Descr, kind: Kind) -> Option<Descr> {
        if !is_set(kind) {
            return None;
        }
        let members = elements.intersect(&hashable())?;
        let mut descr = Descr::nothing();
        descr.put(kind, Component::Sets(SetLattice::of(Arc::new(members))));
        Some(descr)
    }

    /// The dicts whose `key` maps into `ty`, every other key free.
    ///
    /// One constructor for a record's field and a mapping's literal key, because
    /// they are one thing: `{"a": int}` and `dict[Literal["a"], int]` name one
    /// key and say what it maps to.
    ///
    /// `optional` admits the dicts without the key at all, which is the `⊥` in
    /// the field's type rather than a rule beside it.
    #[must_use]
    pub fn label(key: Label, ty: &Descr, optional: bool) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(
            Kind::Dict,
            Component::Maps(MapLattice::label(key, Arc::new(ty.clone()), optional)),
        );
        descr
    }

    /// The dicts every one of whose keys of `kind` maps into `ty`.
    ///
    /// One part of the key partition at a time, which is what makes the default
    /// a function: `dict[str, int]` says what a `str` key maps to and leaves
    /// every other kind of key alone, and the two spellings meet rather than
    /// overlapping.
    #[must_use]
    pub fn mapping(kind: Kind, ty: &Descr) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(
            Kind::Dict,
            Component::Maps(MapLattice::keyed(kind, Arc::new(ty.clone()))),
        );
        descr
    }

    /// The dicts a keyed map spells: the keys it names, and what a key of each
    /// part may hold besides them.
    ///
    /// One atom rather than a meet of one-aspect maps, because closing and
    /// naming are not independent: "no keys at all" met with `{"a": int}` is
    /// empty, since the closing default governs `a` too.
    #[must_use]
    pub fn keyed_map(
        labels: impl IntoIterator<Item = (Label, Descr, bool)>,
        opened: impl IntoIterator<Item = (Option<Kind>, Descr)>,
    ) -> Option<Descr> {
        let lattice = MapLattice::record(
            labels
                .into_iter()
                .map(|(label, ty, optional)| (label, Arc::new(ty), optional)),
            opened.into_iter().map(|(part, ty)| (part, Arc::new(ty))),
        )?;
        let mut descr = Descr::nothing();
        descr.put(Kind::Dict, Component::Maps(lattice));
        Some(descr)
    }

    /// The dicts carrying no key outside `parts`, which is what closes a map.
    ///
    /// `None` in `parts` is the part for a key of no listed kind: an object with
    /// a `__hash__` is a key, and a closed record has to shut it too.
    #[must_use]
    pub fn keys_among(parts: &[Option<Kind>]) -> Descr {
        let mut descr = Descr::nothing();
        descr.put(Kind::Dict, Component::Maps(MapLattice::keys_among(parts)));
        descr
    }

    /// The values carrying `label`, whose value is in `ty`.
    ///
    /// The record is always **open**: an object may carry attributes no schema
    /// mentions, so naming one constrains that attribute and no other. That is
    /// what makes a complement finite -- an object fails by holding a *named*
    /// attribute outside its type, so the complement splits over the labels.
    ///
    /// `optional` admits the values that do not carry the attribute at all,
    /// which is membership of the undefined value rather than a rule beside the
    /// type.
    #[must_use]
    pub fn attribute(label: &str, ty: &Descr, optional: bool) -> Descr {
        Descr::objects(&RecordLattice::attribute(
            label,
            Arc::new(ty.clone()),
            optional,
        ))
    }

    /// The values that are instances of `class`.
    ///
    /// Only a **pure** class belongs here -- one whose metaclass leaves
    /// `isinstance` and `issubclass` alone. A class with a hook answers
    /// arbitrary code and is not a set this algebra holds.
    #[must_use]
    pub fn instance_of(class: Class) -> Descr {
        // A class that confines its instances to a kind belongs on that kind's
        // line alone, which is exact: every instance of a `str` subclass is a
        // string, so the class narrows the `Str` kind rather than standing
        // outside it, and that is where `MyStr <= str` is decided. A class that
        // confines nothing stays on every line, because a subclass of it may lay
        // down any layout at all.
        let Some(kind) = class.kind() else {
            return Descr::objects(&RecordLattice::instance_of(class));
        };
        let lattice = RecordLattice::instance_of(class);
        let mut descr = Descr::nothing();
        if let Some(slot) = descr.kinds.get_mut(Descr::position(kind)) {
            *slot = Lines::objects(&Component::top(kind), lattice);
        }
        descr
    }

    /// The values that do not carry `label` at all.
    #[must_use]
    pub fn without_attribute(label: &str) -> Descr {
        Descr::objects(&RecordLattice::without(label))
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
        descr.put(Kind::Int, Component::Integers(IntSet::multiple_of(step)?));
        Some(descr)
    }

    fn position(kind: Kind) -> usize {
        Kind::ALL
            .iter()
            .position(|listed| *listed == kind)
            .unwrap_or(0)
    }

    fn component(&self, kind: Kind) -> &Lines {
        static EMPTY: OnceLock<Lines> = OnceLock::new();
        self.kinds
            .get(Descr::position(kind))
            .unwrap_or_else(|| EMPTY.get_or_init(Lines::bottom))
    }

    /// Put an integer set in the `int` slot, leaving every other kind empty.
    ///
    /// The one place a caller outside this module builds a component directly:
    /// a refinement is a set of integers before it is anything else, and the
    /// lowering meets it with whatever the base admits.
    pub fn integers(&mut self, set: IntSet) {
        self.put(Kind::Int, Component::Integers(set));
    }

    /// Put a float set in the `float` slot, leaving every other kind empty.
    ///
    /// Beside [`integers`](Self::integers), for a bound that orders floats. The
    /// two cannot be folded together: a float bound orders `nan` out of every
    /// side of itself, and an integer bound says nothing about a float at all.
    pub fn floats(&mut self, set: FloatSet) {
        self.put(Kind::Float, Component::Floats(set));
    }

    /// Put a boolean set in the `bool` slot, leaving every other kind empty.
    ///
    /// Beside [`integers`](Self::integers) and for the same caller: `bool` is a
    /// kind of its own here, so a bound that orders the integers orders these
    /// two as well and has to say so in both slots.
    pub fn booleans(&mut self, set: BoolSet) {
        self.put(Kind::Bool, Component::Booleans(set));
    }

    fn put(&mut self, kind: Kind, structure: Component) {
        if let Some(slot) = self.kinds.get_mut(Descr::position(kind)) {
            *slot = Lines::everything(structure);
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
        let mut kinds = self.kinds.clone();
        for (slot, kind) in kinds.iter_mut().zip(Kind::ALL) {
            *slot = slot.complement(&Component::top(kind));
        }
        Descr {
            kinds,
            other: self.other.complement(&KINDLESS),
        }
    }

    /// Charges the build's allowance: a meet of two descriptors multiplies each
    /// kind's lines against the other's, and a union then has both to carry.
    fn zip(&self, other: &Descr, op: Op) -> Option<Descr> {
        if !budget::spend() {
            return None;
        }
        let mut kinds = self.kinds.clone();
        for ((slot, theirs), kind) in kinds.iter_mut().zip(&other.kinds).zip(Kind::ALL) {
            *slot = slot.combine(theirs, op, &Component::top(kind))?;
        }
        Some(Descr {
            kinds,
            other: self.other.combine(&other.other, op, &KINDLESS)?,
        })
    }

    /// Whether this set is **proved** to admit no value.
    ///
    /// The reduction every relation makes, named once: `Unknown` is not a proof,
    /// so it answers `false`. [`emptiness`](Descr::emptiness) is what tells a
    /// proof of inhabitation apart from a failure to prove emptiness.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.emptiness() == Verdict::Empty
    }

    /// What is known about this set admitting a value.
    ///
    /// The kinds partition the universe, so a value the descriptor admits is a
    /// value some component admits: the verdict is the union's, empty only when
    /// every component is proved empty and inhabited as soon as one is proved
    /// inhabited. `Unknown` is what is left -- no component proved either way,
    /// and one of them could not say.
    #[must_use]
    pub fn emptiness(&self) -> Verdict {
        Verdict::any(
            self.kinds
                .iter()
                .zip(Kind::ALL)
                .map(|(lines, kind)| lines.emptiness(&Component::top(kind)))
                .chain([self.other.emptiness(&KINDLESS)]),
        )
    }

    /// Whether this set admits `value`.
    #[must_use]
    pub fn admits(&self, value: Value) -> bool {
        let lines = match value.kind {
            Some(kind) => self.component(kind),
            None => &self.other,
        };
        lines.admits(
            &|structure| match structure {
                Component::Coarse(present) => *present,
                Component::Booleans(set) => value.boolean.is_some_and(|b| set.holds(b)),
                Component::Integers(set) => value.integer.is_some_and(|i| set.holds(i)),
                Component::Floats(set) => value.float.is_some_and(|f| set.holds(f)),
                Component::Words(set) => value.word.is_some_and(|w| set.holds(w)),
                Component::Sequences(set) => value.elements.is_some_and(|e| set.holds(e)),
                Component::Sets(set) => value.elements.is_some_and(|e| set.holds(e)),
                Component::Maps(set) => value.entries.is_some_and(|entries| {
                    // The component is generic over its guard and cannot read a
                    // value, so the key is decomposed here: the constant it is
                    // where an atom could name it, and the part it falls in.
                    let decomposed: Vec<Entry<Value>> = entries
                        .iter()
                        .map(|(key, mapped)| Entry {
                            label: label_of(key),
                            kind: key.kind,
                            value: *mapped,
                        })
                        .collect();
                    set.holds(&decomposed)
                }),
            },
            // A value the core was told no attributes for is read as one
            // carrying none, which is what an *open* record already means by an
            // absent label. So a line with no object constraint admits it and
            // one naming a class or an attribute declines it -- declines, rather
            // than admits, being the safe direction for a value nothing is known
            // about.
            &|objects| objects.holds(value.class, value.attributes.unwrap_or(&[])),
        )
    }
}

/// A descriptor is a letter of the sequence automaton, which is what makes the
/// component recursive -- and it is held **by handle**, which is what keeps that
/// recursion affordable.
///
/// A `Descr` is one component per kind, stored inline, so it is the same size
/// whatever it describes; an edge holding one by value is that size again, and
/// an edge's guard has its own automaton with its own edges. Nesting multiplied
/// rather than added, and cloning a descriptor deep-copied every level of it --
/// which is what a product of two automata does to their guards, once per pair
/// of states. Behind an [`Arc`] an edge holds a pointer, a clone is a reference
/// count, and two guards that came from one are one allocation. Equality gets
/// the same discount: `Arc` compares pointers before contents, so a guard
/// compared with itself is settled without a walk.
///
/// The three operations and emptiness are the ones above; the trait is the
/// interface the automaton asks a letter for, and nothing here is new work. It
/// is the *fallibility* that shows through: a guard that cannot join leaves the
/// table coarser rather than wrong, which is why the automaton's minimisation
/// asks for a join and accepts a refusal.
impl Guard for Arc<Descr> {
    type Value = Value;

    fn none() -> Arc<Descr> {
        // The empty guard is asked for once per edge that needs a sink, so it is
        // built once and shared rather than allocated per ask.
        static NONE: OnceLock<Arc<Descr>> = OnceLock::new();
        Arc::clone(NONE.get_or_init(|| Arc::new(Descr::nothing())))
    }

    /// The allowance is charged by the descriptor meet underneath, so descending a
    /// level of nesting costs a unit whether it is spent here or one deeper.
    /// The shortcut above it is why an idempotent meet costs nothing: there is
    /// no product to take.
    fn meet(&self, other: &Arc<Descr>) -> Option<Arc<Descr>> {
        if let Some(same) = idempotent(self, other) {
            return Some(same);
        }
        self.intersect(other).map(Arc::new)
    }

    fn join(&self, other: &Arc<Descr>) -> Option<Arc<Descr>> {
        if let Some(same) = idempotent(self, other) {
            return Some(same);
        }
        self.union(other).map(Arc::new)
    }

    /// Charges a unit of its own, because complementing is the one operation
    /// that recurses through every guard without taking a product.
    /// It cannot refuse -- the [`Guard`] contract has it total -- so the charge
    /// is what makes the *next* operation refuse instead.
    fn complement(&self) -> Arc<Descr> {
        budget::spend();
        Arc::new(Descr::complement(self))
    }

    fn is_empty(&self) -> bool {
        Descr::is_empty(self)
    }

    fn emptiness(&self) -> Verdict {
        Descr::emptiness(self)
    }

    fn holds(&self, value: &Value) -> bool {
        self.admits(*value)
    }
}

/// The result of meeting or joining a guard with itself, which is the guard.
///
/// Both laws are idempotent, and the automaton's product asks them of every pair
/// of states that reach on the same letter -- where the two sides are usually
/// the *same handle*, because they came from one. Reading that off the pointer
/// settles the pair without walking either side and without allocating a third
/// descriptor to hold an answer one of them already is.
///
/// Sharing rather than equality on purpose: two guards that are equal but
/// separately allocated are still two sets to compare, and comparing them is the
/// work this is avoiding.
fn idempotent(left: &Arc<Descr>, right: &Arc<Descr>) -> Option<Arc<Descr>> {
    Arc::ptr_eq(left, right).then(|| Arc::clone(left))
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
    /// The elements, where the kind is a sequence or a set kind.
    ///
    /// A value again, which is what makes the question recursive: whether a
    /// sequence is admitted is asked of its elements, one letter at a time.
    /// For a set kind these are its *members*, in no particular order -- a set
    /// is its members and the powerset rule reads them as a whole rather than
    /// as a word.
    /// The class, where the value is of no listed kind.
    ///
    /// What the value *is*, as far as the snapshot of the class order can say.
    /// A value with none is one whose class the core was never told, which every
    /// class constraint declines rather than admits.
    pub class: Option<&'static Class>,
    /// The attributes, where the value is of no listed kind.
    ///
    /// What an object *is*, at this resolution: the names it carries and what
    /// each one holds. An attribute absent from the list is one the object does
    /// not carry, which an open record reads as the undefined value.
    pub attributes: Option<&'static [(&'static str, Value)]>,
    pub elements: Option<&'static [Value]>,
    /// The entries, where the kind is [`Kind::Dict`]: each key beside what it
    /// maps to.
    ///
    /// A key is a value again, which is what lets the component read its *kind*
    /// -- the part of the key partition it falls in -- and, where it is a
    /// string, the label it may be named by.
    pub entries: Option<&'static [(Value, Value)]>,
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
            elements: None,
            attributes: None,
            class: None,
            entries: None,
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
            elements: None,
            attributes: None,
            class: None,
            entries: None,
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
            elements: None,
            attributes: None,
            class: None,
            entries: None,
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
            elements: None,
            attributes: None,
            class: None,
            entries: None,
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
            elements: None,
            attributes: None,
            class: None,
            entries: None,
        }
    }

    /// A value of no listed kind carrying no attribute anyone named.
    #[must_use]
    pub const fn other() -> Value {
        Value::object(&[])
    }

    /// The same value, also carrying `attributes`.
    ///
    /// A value has a kind *and* the attributes it carries: an object deriving
    /// from `int` is an integer and carries its fields both, and the descriptor
    /// holds the two on one line. This is what lets one be written down to ask a
    /// law about.
    #[must_use]
    pub const fn carrying(self, attributes: &'static [(&'static str, Value)]) -> Value {
        Value {
            attributes: Some(attributes),
            ..self
        }
    }

    /// The same value, also seen as an instance of `class`.
    #[must_use]
    pub const fn of_class(
        self,
        class: &'static Class,
        attributes: &'static [(&'static str, Value)],
    ) -> Value {
        Value {
            class: Some(class),
            ..self.carrying(attributes)
        }
    }

    /// An instance of `class`, described by that and the attributes it carries.
    #[must_use]
    pub const fn instance(
        class: &'static Class,
        attributes: &'static [(&'static str, Value)],
    ) -> Value {
        Value {
            class: Some(class),
            ..Value::object(attributes)
        }
    }

    /// A value of no listed kind -- a class instance, a callable -- described by
    /// the attributes it carries.
    #[must_use]
    pub const fn object(attributes: &'static [(&'static str, Value)]) -> Value {
        Value {
            kind: None,
            boolean: None,
            integer: None,
            float: None,
            word: None,
            elements: None,
            attributes: Some(attributes),
            class: None,
            entries: None,
        }
    }

    /// One dict, as its entries: each key beside what it maps to.
    #[must_use]
    pub const fn dict(entries: &'static [(Value, Value)]) -> Value {
        Value {
            kind: Some(Kind::Dict),
            boolean: None,
            integer: None,
            float: None,
            word: None,
            elements: None,
            attributes: None,
            class: None,
            entries: Some(entries),
        }
    }
    /// One sequence, for a sequence kind. The elements are values again, which
    /// is the letters the automaton reads.
    #[must_use]
    pub const fn sequence(elements: &'static [Value], kind: Kind) -> Value {
        Value {
            kind: Some(kind),
            boolean: None,
            integer: None,
            float: None,
            word: None,
            elements: Some(elements),
            attributes: None,
            class: None,
            entries: None,
        }
    }
}

#[cfg(test)]
mod tests;
