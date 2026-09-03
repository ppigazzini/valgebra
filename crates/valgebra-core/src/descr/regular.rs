//! Sets of strings and byte strings as regular languages.
//!
//! A `str` refinement can pin (`Literal["a"]`), bound its length
//! (`MinLen`/`MaxLen`) and match a pattern (`Regex`). All three denote regular
//! languages, and regular languages are closed under union, intersection and
//! complement -- which is exactly what a descriptor component has to be. The
//! structural procedure relates two patterns only when they are written
//! identically; here `Regex("a")` is below `Regex("ab?")` because the automaton
//! says so.
//!
//! **The representation is a minimal deterministic automaton, canonically
//! numbered.** A regular language has one minimal DFA up to isomorphism, so
//! minimising and then renumbering the states in a fixed order makes the
//! representation canonical: two sets hold the same words exactly when their
//! tables are equal. That is what the components either side of this one give by
//! construction, and it is what lets a decision compare representations instead
//! of running a language-equivalence check at every step.
//!
//! **The alphabet is bytes, in equivalence classes.** A word is a byte string --
//! UTF-8 for `str`, arbitrary for `bytes` -- so one machine serves both, and the
//! only difference is whether a pattern is read in Unicode mode. Transitions are
//! stored per *class* rather than per byte: a pattern like `[a-z]+` distinguishes
//! two classes out of 256, and the classes are recomputed after every operation
//! so the partition is as coarse as the language allows, which is what keeps it
//! canonical.
//!
//! The automaton comes from `regex-automata`, which the binding's `regex`
//! already depends on, so the core shares a version rather than adding a
//! dependency. What is taken from it is the *parser and determiniser* for one
//! pattern; the three set operations and the emptiness decision are here,
//! because the crate builds automata for searching and offers no complement.

use regex_automata::dfa::{Automaton, dense};
use regex_automata::util::syntax;
use regex_automata::{Anchored, Input};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// The most states an automaton may hold.
///
/// A product doubles the exponent -- `|A| * |B|` states before minimisation --
/// and a complement of a product does it again, so a bound is what keeps a
/// pathological pattern from exhausting memory rather than answering. Past it
/// the set is not representable, and the constructor says so rather than
/// returning an automaton for a different language.
pub const MAX_STATES: usize = 4096;

/// A byte's equivalence class. Two bytes share a class when no state's
/// transition tells them apart.
type Class = u16;

/// A deterministic, complete, minimal automaton over byte classes.
///
/// State zero is the start. Every state has a transition for every class, so a
/// walk never falls off the table and a complement is a flip of the accepting
/// flags rather than a construction.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Dfa {
    /// The class of each byte.
    classes: Vec<Class>,
    /// How many classes the alphabet has.
    class_count: usize,
    /// Row-major, `state * class_count + class`.
    transitions: Vec<u32>,
    accepting: Vec<bool>,
}

impl Dfa {
    fn state_count(&self) -> usize {
        self.accepting.len()
    }

    fn class_of(&self, byte: u8) -> Class {
        self.classes.get(usize::from(byte)).copied().unwrap_or(0)
    }

    fn step(&self, state: u32, class: Class) -> u32 {
        let index = (state as usize)
            .checked_mul(self.class_count)
            .and_then(|row| row.checked_add(usize::from(class)));
        index
            .and_then(|i| self.transitions.get(i))
            .copied()
            .unwrap_or(0)
    }

    fn accepts(&self, state: u32) -> bool {
        self.accepting.get(state as usize).copied().unwrap_or(false)
    }

    /// Whether this automaton accepts `word`.
    fn holds(&self, word: &[u8]) -> bool {
        let mut state = 0;
        for byte in word {
            state = self.step(state, self.class_of(*byte));
        }
        self.accepts(state)
    }

    /// Whether any reachable state accepts.
    ///
    /// Read by a walk rather than off the minimal form, so it answers before
    /// minimisation as well as after -- which is what the product construction
    /// needs when it stops early.
    fn is_empty(&self) -> bool {
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        let mut pending: VecDeque<u32> = VecDeque::from([0]);
        while let Some(state) = pending.pop_front() {
            // `insert` answers whether the state is new, so the walk visits
            // each one once and terminates because the states are finite.
            if !seen.insert(state) {
                continue;
            }
            if self.accepts(state) {
                return false;
            }
            for class in 0..self.class_count {
                pending.push_back(self.step(state, class.try_into().unwrap_or(0)));
            }
        }
        true
    }

    /// The automaton accepting nothing, over the one-class alphabet.
    fn empty() -> Dfa {
        Dfa {
            classes: vec![0; 256],
            class_count: 1,
            transitions: vec![0],
            accepting: vec![false],
        }
    }

    /// The automaton accepting every word.
    fn universal() -> Dfa {
        Dfa {
            classes: vec![0; 256],
            class_count: 1,
            transitions: vec![0],
            accepting: vec![true],
        }
    }

    /// Every word this automaton does not accept.
    ///
    /// A flip of the accepting flags, which is only sound because the table is
    /// *complete*: a partial automaton rejects by falling off the end, and
    /// flipping its flags would accept those words instead of the ones it meant.
    fn complement(&self) -> Dfa {
        Dfa {
            classes: self.classes.clone(),
            class_count: self.class_count,
            transitions: self.transitions.clone(),
            accepting: self.accepting.iter().map(|a| !a).collect(),
        }
        .minimal()
    }

    /// The product of two automata, accepting where `accept` says so.
    ///
    /// One BFS over reachable state *pairs*, so the states built are the ones a
    /// word can reach rather than the whole cross product. The alphabets are
    /// refined together first: a class of the product is a pair of classes, one
    /// from each side, and only the pairs some byte realises get an id.
    fn product(&self, other: &Dfa, accept: impl Fn(bool, bool) -> bool) -> Option<Dfa> {
        let (classes, class_count, pairs) = refine(self, other);
        let mut ids: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        // A draining queue, so the walk ends when nothing is left rather than
        // when an index catches up with a list that is still growing.
        let mut pending: VecDeque<(u32, u32)> = VecDeque::from([(0, 0)]);
        ids.insert((0, 0), 0);
        let mut transitions: Vec<u32> = Vec::new();
        let mut accepting: Vec<bool> = Vec::new();
        let mut built = 0usize;
        while let Some((mine, theirs)) = pending.pop_front() {
            built += 1;
            accepting.push(accept(self.accepts(mine), other.accepts(theirs)));
            for pair in &pairs {
                let next = (self.step(mine, pair.0), other.step(theirs, pair.1));
                let id = if let Some(id) = ids.get(&next) {
                    *id
                } else {
                    if ids.len() >= MAX_STATES {
                        return None;
                    }
                    let id = u32::try_from(ids.len()).ok()?;
                    ids.insert(next, id);
                    pending.push_back(next);
                    id
                };
                transitions.push(id);
            }
        }
        debug_assert_eq!(built, accepting.len(), "one row is built per state");
        Some(
            Dfa {
                classes,
                class_count,
                transitions,
                accepting,
            }
            .minimal(),
        )
    }

    /// The minimal automaton for the same language, canonically numbered.
    ///
    /// Three passes, and each is needed for the form to be canonical: merge the
    /// states no word can tell apart, coarsen the alphabet to the partition the
    /// merged states induce, and renumber by a walk that visits classes in
    /// order. A minimal DFA is unique up to isomorphism, so fixing the numbering
    /// fixes the table.
    fn minimal(&self) -> Dfa {
        let merged = self.merge_equivalent();
        let coarsened = merged.coarsen();
        coarsened.renumber()
    }

    /// Merge the states no word distinguishes, by refining a partition until it
    /// stops changing (Moore's algorithm).
    fn merge_equivalent(&self) -> Dfa {
        // Start by separating the accepting states from the rest: a word of
        // length zero already tells those apart.
        let mut block: Vec<u32> = self.accepting.iter().map(|a| u32::from(*a)).collect();
        // At most one round per state: each round either splits a block or is
        // the last, and a block cannot split more often than it has members.
        // Bounding it is what makes a wrong termination test leave a coarser
        // automaton -- which the canonicity property rejects -- rather than run
        // without end.
        for _ in 0..=self.state_count() {
            // A state's signature is its own block and the block each class
            // leads to. Two states stay together only while their signatures
            // agree, which is one more letter of lookahead per round.
            let mut ids: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
            let mut next: Vec<u32> = Vec::with_capacity(self.state_count());
            for state in 0..self.state_count() {
                let state = u32::try_from(state).unwrap_or(0);
                let mut signature = vec![block.get(state as usize).copied().unwrap_or(0)];
                for class in 0..self.class_count {
                    let target = self.step(state, class.try_into().unwrap_or(0));
                    signature.push(block.get(target as usize).copied().unwrap_or(0));
                }
                let count = u32::try_from(ids.len()).unwrap_or(0);
                next.push(*ids.entry(signature).or_insert(count));
            }
            if next == block {
                break;
            }
            block = next;
        }
        let count = block.iter().copied().max().map_or(1, |m| m as usize + 1);
        let mut transitions = vec![0u32; count * self.class_count];
        let mut accepting = vec![false; count];
        for state in 0..self.state_count() {
            let state = u32::try_from(state).unwrap_or(0);
            let Some(id) = block.get(state as usize).copied() else {
                continue;
            };
            if let Some(flag) = accepting.get_mut(id as usize) {
                *flag = self.accepts(state);
            }
            for class in 0..self.class_count {
                let target = self.step(state, class.try_into().unwrap_or(0));
                let target_block = block.get(target as usize).copied().unwrap_or(0);
                if let Some(slot) = transitions.get_mut(id as usize * self.class_count + class) {
                    *slot = target_block;
                }
            }
        }
        Dfa {
            // Merging states leaves the alphabet alone; `coarsen` is the pass
            // that reduces it, once the blocks are known.
            classes: self.classes.clone(),
            class_count: self.class_count,
            transitions,
            accepting,
        }
    }

    /// Coarsen the alphabet to the partition this automaton's states induce.
    ///
    /// Two bytes belong together when no state's transition separates them.
    /// Without this pass a language could carry a finer partition than it needs
    /// -- a product refines the alphabet whether or not the result uses the
    /// distinction -- and two equal languages would have unequal tables.
    fn coarsen(&self) -> Dfa {
        let mut ids: FxHashMap<Vec<u32>, Class> = FxHashMap::default();
        let mut classes: Vec<Class> = Vec::with_capacity(256);
        let mut columns: Vec<Class> = Vec::new();
        for byte in 0u16..256 {
            let old = self.class_of(u8::try_from(byte).unwrap_or(0));
            let column: Vec<u32> = (0..self.state_count())
                .map(|state| self.step(u32::try_from(state).unwrap_or(0), old))
                .collect();
            let count = Class::try_from(ids.len()).unwrap_or(0);
            let id = *ids.entry(column).or_insert_with(|| {
                columns.push(old);
                count
            });
            classes.push(id);
        }
        let class_count = columns.len().max(1);
        let mut transitions = vec![0u32; self.state_count() * class_count];
        for state in 0..self.state_count() {
            for (fresh, old) in columns.iter().enumerate() {
                let target = self.step(u32::try_from(state).unwrap_or(0), *old);
                if let Some(slot) = transitions.get_mut(state * class_count + fresh) {
                    *slot = target;
                }
            }
        }
        Dfa {
            classes,
            class_count,
            transitions,
            accepting: self.accepting.clone(),
        }
    }

    /// Renumber the states by a walk that takes classes in order, dropping the
    /// ones no word reaches.
    fn renumber(&self) -> Dfa {
        let mut ids: FxHashMap<u32, u32> = FxHashMap::default();
        let mut order: Vec<u32> = Vec::new();
        let mut pending: VecDeque<u32> = VecDeque::from([0]);
        ids.insert(0, 0);
        while let Some(state) = pending.pop_front() {
            order.push(state);
            for class in 0..self.class_count {
                let target = self.step(state, class.try_into().unwrap_or(0));
                let fresh = u32::try_from(ids.len()).unwrap_or(0);
                if let std::collections::hash_map::Entry::Vacant(slot) = ids.entry(target) {
                    slot.insert(fresh);
                    pending.push_back(target);
                }
            }
        }
        let mut transitions = Vec::with_capacity(order.len() * self.class_count);
        let mut accepting = Vec::with_capacity(order.len());
        for state in &order {
            accepting.push(self.accepts(*state));
            for class in 0..self.class_count {
                let target = self.step(*state, class.try_into().unwrap_or(0));
                transitions.push(ids.get(&target).copied().unwrap_or(0));
            }
        }
        Dfa {
            classes: self.classes.clone(),
            class_count: self.class_count,
            transitions,
            accepting,
        }
    }
}

/// The alphabet two automata share: a class per pair of classes some byte
/// realises, plus the byte-to-class map over it.
fn refine(a: &Dfa, b: &Dfa) -> (Vec<Class>, usize, Vec<(Class, Class)>) {
    let mut ids: FxHashMap<(Class, Class), Class> = FxHashMap::default();
    let mut pairs: Vec<(Class, Class)> = Vec::new();
    let mut classes: Vec<Class> = Vec::with_capacity(256);
    for byte in 0u16..256 {
        let byte = u8::try_from(byte).unwrap_or(0);
        let pair = (a.class_of(byte), b.class_of(byte));
        let count = Class::try_from(ids.len()).unwrap_or(0);
        let id = *ids.entry(pair).or_insert_with(|| {
            pairs.push(pair);
            count
        });
        classes.push(id);
    }
    let class_count = pairs.len().max(1);
    (classes, class_count, pairs)
}

/// Which words a pattern is read over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Alphabet {
    /// UTF-8 text, where `.` is one code point and a pattern may name a
    /// Unicode class. This is what a `str` refinement means.
    Text,
    /// Arbitrary bytes, where `.` is one byte. This is what a `bytes`
    /// refinement means, and it admits words no `str` can hold.
    Bytes,
}

/// A set of words: a regular language.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegularSet {
    dfa: Dfa,
}

impl RegularSet {
    /// The empty set.
    #[must_use]
    pub fn empty() -> RegularSet {
        RegularSet { dfa: Dfa::empty() }
    }

    /// Every word.
    #[must_use]
    pub fn all() -> RegularSet {
        RegularSet {
            dfa: Dfa::universal(),
        }
    }

    /// The language of one pattern, matched whole, or `None` where the pattern
    /// does not build or its automaton is past [`MAX_STATES`].
    ///
    /// Anchored at both ends, because that is what a `Regex` constraint means:
    /// the walk matches a pattern against the whole text, not a substring of it.
    #[must_use]
    pub fn pattern(pattern: &str, alphabet: Alphabet) -> Option<RegularSet> {
        let anchored = format!("(?:{pattern})");
        let syntax = syntax::Config::new().utf8(alphabet == Alphabet::Text);
        let built = dense::Builder::new()
            .syntax(syntax)
            .configure(dense::Config::new().start_kind(regex_automata::dfa::StartKind::Anchored))
            .build(&anchored)
            .ok()?;
        Dfa::from_automaton(&built).map(|dfa| RegularSet { dfa: dfa.minimal() })
    }

    /// The one-word language.
    #[must_use]
    pub fn word(word: &[u8]) -> RegularSet {
        // Built as a chain rather than through the pattern parser: a literal is
        // not a pattern, and escaping one into a pattern is a second place the
        // two spellings could disagree.
        let states = word.len() + 1;
        let mut classes: Vec<Class> = vec![0; 256];
        let mut seen: FxHashMap<u8, Class> = FxHashMap::default();
        for byte in word {
            let count = Class::try_from(seen.len()).unwrap_or(0);
            seen.entry(*byte).or_insert(count + 1);
        }
        for (byte, class) in &seen {
            if let Some(slot) = classes.get_mut(usize::from(*byte)) {
                *slot = *class;
            }
        }
        let class_count = seen.len() + 1;
        // One extra state is the sink every wrong byte leads to.
        let mut transitions = vec![u32::try_from(states).unwrap_or(0); (states + 1) * class_count];
        for (position, byte) in word.iter().enumerate() {
            let class = classes.get(usize::from(*byte)).copied().unwrap_or(0);
            if let Some(slot) = transitions.get_mut(position * class_count + usize::from(class)) {
                *slot = u32::try_from(position + 1).unwrap_or(0);
            }
        }
        let mut accepting = vec![false; states + 1];
        if let Some(flag) = accepting.get_mut(word.len()) {
            *flag = true;
        }
        RegularSet {
            dfa: Dfa {
                classes,
                class_count,
                transitions,
                accepting,
            }
            .minimal(),
        }
    }

    /// The words of at least `length` symbols -- code points for
    /// [`Alphabet::Text`], bytes for [`Alphabet::Bytes`].
    #[must_use]
    pub fn at_least(length: usize, alphabet: Alphabet) -> Option<RegularSet> {
        RegularSet::pattern(&format!("{}{{{length},}}", dot(alphabet)), alphabet)
    }

    /// The words of at most `length` symbols.
    #[must_use]
    pub fn at_most(length: usize, alphabet: Alphabet) -> Option<RegularSet> {
        RegularSet::pattern(&format!("{}{{0,{length}}}", dot(alphabet)), alphabet)
    }

    /// Whether this set holds no word.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dfa.is_empty()
    }

    /// Whether this set holds `word`.
    #[must_use]
    pub fn holds(&self, word: &[u8]) -> bool {
        self.dfa.holds(word)
    }

    /// The words in either set, or `None` past [`MAX_STATES`].
    #[must_use]
    pub fn union(&self, other: &RegularSet) -> Option<RegularSet> {
        self.dfa
            .product(&other.dfa, |a, b| a || b)
            .map(|dfa| RegularSet { dfa })
    }

    /// The words in both sets, or `None` past [`MAX_STATES`].
    #[must_use]
    pub fn intersect(&self, other: &RegularSet) -> Option<RegularSet> {
        self.dfa
            .product(&other.dfa, |a, b| a && b)
            .map(|dfa| RegularSet { dfa })
    }

    /// Every word this set does not hold.
    ///
    /// Total, unlike the other two: a complement neither adds states nor
    /// refines the alphabet, so it cannot pass the bound.
    #[must_use]
    pub fn complement(&self) -> RegularSet {
        RegularSet {
            dfa: self.dfa.complement(),
        }
    }
}

/// The pattern for one symbol of an alphabet.
fn dot(alphabet: Alphabet) -> &'static str {
    match alphabet {
        // `(?s)` so a newline counts: a length bound is about how many symbols
        // the word has, not about which of them a pattern would skip.
        Alphabet::Text => "(?s:.)",
        Alphabet::Bytes => "(?s-u:.)",
    }
}

impl Dfa {
    /// Read a built regex automaton into a table, by walking it.
    ///
    /// The crate's automaton is for searching, so it is asked the two questions
    /// a walk needs -- where a byte leads, and whether the end of input from
    /// here is a match -- rather than for its own state table. Accepting is
    /// "the end of input from this state matches", which is what makes the
    /// language the whole-word one.
    fn from_automaton(built: &dense::DFA<Vec<u32>>) -> Option<Dfa> {
        // The crate's alphabet carries a class for the end of input, which no
        // byte realises. The alphabet here is the classes some byte does, so a
        // table is never built over a column no word can take.
        let crate_classes = built.byte_classes();
        let mut used: Vec<u8> = Vec::new();
        let mut byte_class: Vec<Class> = Vec::with_capacity(256);
        for byte in 0u16..256 {
            let byte = u8::try_from(byte).unwrap_or(0);
            let theirs = crate_classes.get(byte);
            let index = if let Some(index) = used.iter().position(|c| *c == theirs) {
                index
            } else {
                used.push(theirs);
                used.len() - 1
            };
            byte_class.push(Class::try_from(index).ok()?);
        }
        let alphabet = used.len();
        let start = built
            .start_state_forward(&Input::new("").anchored(Anchored::Yes))
            .ok()?;
        let mut ids: FxHashMap<usize, u32> = FxHashMap::default();
        let mut pending: VecDeque<_> = VecDeque::from([start]);
        ids.insert(start.as_usize(), 0);
        let mut transitions: Vec<u32> = Vec::new();
        let mut accepting: Vec<bool> = Vec::new();
        while let Some(state) = pending.pop_front() {
            accepting.push(built.is_match_state(built.next_eoi_state(state)));
            for class in 0..alphabet {
                // Every byte of a class takes the same transition, so one
                // representative stands for the column.
                let byte = representative(&byte_class, class)?;
                let next = built.next_state(state, byte);
                let id = if let Some(id) = ids.get(&next.as_usize()) {
                    *id
                } else {
                    if ids.len() >= MAX_STATES {
                        return None;
                    }
                    let id = u32::try_from(ids.len()).ok()?;
                    ids.insert(next.as_usize(), id);
                    pending.push_back(next);
                    id
                };
                transitions.push(id);
            }
        }
        Some(Dfa {
            classes: byte_class,
            class_count: alphabet,
            transitions,
            accepting,
        })
    }
}

/// A byte belonging to `class`, or `None` where no byte does.
///
/// The crate's alphabet includes a class for the end of input, which no byte
/// realises; a table built over it would carry a column no word can take.
fn representative(byte_class: &[Class], class: usize) -> Option<u8> {
    let wanted = Class::try_from(class).ok()?;
    byte_class
        .iter()
        .position(|c| *c == wanted)
        .and_then(|byte| u8::try_from(byte).ok())
}

#[cfg(test)]
mod tests {
    use super::{Alphabet, RegularSet};
    use proptest::prelude::*;

    /// The words a law is checked over.
    ///
    /// Every word of up to two letters over `{a, b, c}`, plus a three-letter one
    /// and a non-ASCII one. Small, but every generated pattern below is written
    /// over that alphabet, so a language that differs from another differs on a
    /// word this short -- two regular languages agreeing on all words up to the
    /// product of their state counts are equal, and these automata have a
    /// handful of states each.
    fn universe() -> Vec<&'static [u8]> {
        vec![
            b"",
            b"a",
            b"b",
            b"c",
            b"aa",
            b"ab",
            b"ac",
            b"ba",
            b"bb",
            b"bc",
            b"ca",
            b"cb",
            b"cc",
            b"aba",
            b"abc",
            "é".as_bytes(),
        ]
    }

    /// Whether two sets agree about every word in the universe.
    ///
    /// A *weaker* question than equality, and deliberately: two regular
    /// languages agree exactly when they agree on every word shorter than the
    /// product of their state counts, which for the automata here is longer
    /// than any universe can enumerate. So this is used in the direction it can
    /// support -- two equal sets must agree about every word -- and equality
    /// itself is checked against the emptiness decision instead.
    fn agree_on_words(a: &RegularSet, b: &RegularSet) -> bool {
        universe().into_iter().all(|w| a.holds(w) == b.holds(w))
    }

    /// Whether two sets hold the same words, decided by the algebra.
    ///
    /// The semantic-subtyping reduction in both directions: `A = B` when
    /// `A ∧ ¬B` and `B ∧ ¬A` are both empty. This reaches the whole language
    /// where enumeration cannot, and it reads the answer off the reachability
    /// walk rather than off the canonical form -- so comparing it with `==`
    /// below is two independent pieces of this module agreeing, not one of them
    /// agreeing with itself.
    fn same_language(a: &RegularSet, b: &RegularSet) -> bool {
        let inside = |x: &RegularSet, y: &RegularSet| {
            x.intersect(&y.complement())
                .is_some_and(|met| met.is_empty())
        };
        inside(a, b) && inside(b, a)
    }

    /// A pattern's language, or the empty set where the bound refuses it.
    fn language(pattern: &str) -> RegularSet {
        RegularSet::pattern(pattern, Alphabet::Text).expect("a small pattern builds")
    }

    /// Languages over the three-letter alphabet the universe covers.
    fn regular_set() -> impl Strategy<Value = RegularSet> {
        let leaf = prop_oneof![
            Just(RegularSet::empty()),
            Just(RegularSet::all()),
            Just(language("a")),
            Just(language("b")),
            Just(language("ab?")),
            Just(language("[ab]+")),
            Just(language("a*")),
            Just(RegularSet::word(b"ab")),
            Just(RegularSet::at_least(1, Alphabet::Text).expect("a small bound")),
            Just(RegularSet::at_most(1, Alphabet::Text).expect("a small bound")),
        ];
        leaf.prop_recursive(3, 12, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.union(&b).unwrap_or_else(RegularSet::all)),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(RegularSet::empty)),
                inner.prop_map(|a| a.complement()),
            ]
        })
    }

    proptest! {
        // Fewer cases than the default, for the reason the descriptor module
        // gives: every operation here is an automaton product, and shrinking a
        // failure over thousands of draws outruns a mutation sweep's patience.
        #![proptest_config(ProptestConfig {
            cases: 64,
            // A bounded shrink, so a broken invariant cannot turn a caught
            // mutation into a run that outlasts a sweep.
            max_shrink_time: 2_000,
            ..ProptestConfig::default()
        })]

        /// The Boolean algebra, checked by equality of the canonical forms.
        ///
        /// Equality *is* language equality here -- that is what the minimal,
        /// canonically-numbered table earns, and the property below holds it to
        /// the emptiness decision. So a law is checked at full strength rather
        /// than over whatever words a universe can list, which for a regular
        /// language is never enough.
        #[test]
        fn the_lattice_laws_hold_of_the_languages(
            a in regular_set(),
            b in regular_set(),
            c in regular_set(),
        ) {
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

        /// The complement laws, and De Morgan both ways.
        #[test]
        fn the_complement_laws_hold_of_the_languages(a in regular_set(), b in regular_set()) {
            prop_assert!(
                a.intersect(&a.complement())
                    .is_some_and(|met| met.is_empty())
            );
            prop_assert_eq!(a.union(&a.complement()), Some(RegularSet::all()));
            prop_assert_eq!(&a.complement().complement(), &a);
            prop_assert_eq!(
                a.union(&b).map(|set| set.complement()),
                a.complement().intersect(&b.complement())
            );
            prop_assert_eq!(
                a.intersect(&b).map(|set| set.complement()),
                a.complement().union(&b.complement())
            );
        }

        /// Equality of the canonical forms is equality of the languages.
        ///
        /// The claim the whole representation rests on, held against the
        /// *emptiness* decision -- a reachability walk, which shares no code
        /// with the minimisation and renumbering that make the form canonical.
        /// Two spellings of one language have one table, and two languages that
        /// differ have two.
        #[test]
        fn being_equal_is_holding_the_same_language(a in regular_set(), b in regular_set()) {
            prop_assert_eq!(same_language(&a, &b), a == b);
        }

        /// Two equal sets agree about every word, which is the direction
        /// enumeration can support.
        #[test]
        fn equal_sets_agree_about_every_word(a in regular_set(), b in regular_set()) {
            if a == b {
                prop_assert!(agree_on_words(&a, &b));
            }
        }

        /// An empty verdict is contradicted by no word.
        #[test]
        fn an_empty_language_holds_no_word(a in regular_set()) {
            if a.is_empty() {
                prop_assert!(universe().into_iter().all(|w| !a.holds(w)));
            }
        }
    }

    /// The two relations this component exists to decide, both declined by the
    /// structural procedure because it relates patterns only when they are
    /// written identically.
    #[test]
    fn one_pattern_is_decided_inside_another() {
        let narrow = language("a");
        let wide = language("ab?");
        // `a` is below `ab?`: the meet with the complement is empty, which is
        // the semantic-subtyping reduction applied to a language.
        assert!(
            narrow
                .intersect(&wide.complement())
                .is_some_and(|set| set.is_empty())
        );
        // And not the other way: `ab` is in the wider one alone.
        assert!(
            !wide
                .intersect(&narrow.complement())
                .is_some_and(|set| set.is_empty())
        );
        assert!(wide.holds(b"ab") && !narrow.holds(b"ab"));

        // A pattern whose language is one word is that word's literal.
        assert_eq!(language("a"), RegularSet::word(b"a"));
        assert_eq!(language("abc"), RegularSet::word(b"abc"));
        // Which is what decides `Regex("a") <= Literal["a"]`, in both
        // directions: the two are one set.
        assert!(
            language("a")
                .intersect(&RegularSet::word(b"a").complement())
                .is_some_and(|set| set.is_empty())
        );
    }

    /// Two spellings of one language are one set, which is the canonicity claim
    /// stated on the cases a reader would doubt.
    #[test]
    fn two_spellings_of_one_language_are_one_set() {
        assert_eq!(language("a|a"), language("a"));
        assert_eq!(language("(a)"), language("a"));
        assert_eq!(language("a{1}"), language("a"));
        assert_eq!(language("[aa]"), language("a"));
        assert_eq!(language("a|b"), language("[ab]"));
        assert_eq!(language("a*a*"), language("a*"));
        assert_eq!(language("(a|b)*"), language("[ab]*"));
        // And two that are *not* one language stay apart.
        assert_ne!(language("a"), language("b"));
        assert_ne!(language("a*"), language("a+"));
    }

    /// A pattern matches the whole word, which is what a `Regex` constraint
    /// means: the walk matches against the text, not a substring of it.
    #[test]
    fn a_pattern_is_anchored_at_both_ends() {
        let a = language("a");
        assert!(a.holds(b"a"));
        assert!(!a.holds(b"ab"));
        assert!(!a.holds(b"ba"));
        assert!(!a.holds(b""));
        // An alternation is anchored as a whole rather than per branch, which is
        // why the pattern is wrapped before it is built.
        let either = language("a|bb");
        assert!(either.holds(b"a") && either.holds(b"bb"));
        assert!(!either.holds(b"abb") && !either.holds(b"ab"));
    }

    /// A length bound counts symbols, and which symbol depends on the alphabet:
    /// a code point for text, a byte for bytes.
    #[test]
    fn a_length_bound_counts_the_alphabet_symbols() {
        let two_text = RegularSet::at_most(2, Alphabet::Text).expect("a small bound");
        assert!(two_text.holds(b"ab"));
        assert!(!two_text.holds(b"abc"));
        // Two code points, four bytes: the text alphabet counts the first.
        assert!(two_text.holds("éé".as_bytes()));
        assert!(!two_text.holds("ééé".as_bytes()));

        let two_bytes = RegularSet::at_most(2, Alphabet::Bytes).expect("a small bound");
        assert!(two_bytes.holds(b"ab"));
        // One code point is two bytes, so the byte alphabet counts two.
        assert!(two_bytes.holds("é".as_bytes()));
        assert!(!two_bytes.holds("éé".as_bytes()));

        // A minimum, and the two bounds meeting at an exact length.
        let least = RegularSet::at_least(2, Alphabet::Text).expect("a small bound");
        assert!(!least.holds(b"a") && least.holds(b"ab") && least.holds(b"abc"));
        let exactly = least.intersect(&two_text).expect("a small meet");
        assert!(exactly.holds(b"ab") && !exactly.holds(b"a") && !exactly.holds(b"abc"));
        // A minimum above a maximum admits nothing, which is a bound
        // conjunction decided by the language rather than by comparing bounds.
        let impossible = RegularSet::at_least(3, Alphabet::Text)
            .expect("a small bound")
            .intersect(&two_text)
            .expect("a small meet");
        assert!(impossible.is_empty());
    }

    /// A pattern that does not build, and one whose automaton is too large, are
    /// refused rather than answered.
    #[test]
    fn an_unbuildable_or_oversized_pattern_is_refused() {
        assert!(RegularSet::pattern("(", Alphabet::Text).is_none());
        assert!(RegularSet::pattern("a{2,1}", Alphabet::Text).is_none());
        // A byte pattern is not a text pattern: a non-UTF-8 byte class is
        // refused in text mode and built in byte mode.
        assert!(RegularSet::pattern(r"(?-u:\xFF)", Alphabet::Text).is_none());
        assert!(RegularSet::pattern(r"(?-u:\xFF)", Alphabet::Bytes).is_some());
        // A length bound past the state bound is refused rather than truncated.
        assert!(RegularSet::at_least(super::MAX_STATES + 1, Alphabet::Text).is_none());
    }

    /// The empty and universal languages, and the empty *word*, which is a
    /// different thing from either.
    #[test]
    fn the_empty_language_and_the_empty_word_are_different_sets() {
        assert!(RegularSet::empty().is_empty());
        assert!(!RegularSet::empty().holds(b""));
        assert!(!RegularSet::all().is_empty());
        assert!(RegularSet::all().holds(b"") && RegularSet::all().holds(b"anything"));

        let just_empty_word = RegularSet::word(b"");
        assert!(!just_empty_word.is_empty());
        assert!(just_empty_word.holds(b""));
        assert!(!just_empty_word.holds(b"a"));
        assert_eq!(just_empty_word, language(""));

        assert_eq!(RegularSet::empty().complement(), RegularSet::all());
        assert_eq!(RegularSet::all().complement(), RegularSet::empty());
    }
}
