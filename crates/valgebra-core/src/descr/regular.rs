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
use std::sync::{Arc, OnceLock};

/// The most states an automaton may hold.
///
/// A product doubles the exponent -- `|A| * |B|` states before minimisation --
/// and a complement of a product does it again, so a bound is what keeps a
/// pathological pattern from exhausting memory rather than answering. Past it
/// the set is not representable, and the constructor says so rather than
/// returning an automaton for a different language.
pub const MAX_STATES: usize = 4096;

/// What `regex-automata` may spend building one pattern's table, in bytes.
///
/// [`MAX_STATES`] bounds the automaton this module keeps; this bounds the one
/// the builder makes on the way to it, which is a different quantity and the
/// larger of the two. A dense table is a row of 256 byte-classes per state, so
/// four thousand states is a few megabytes and eight is generous headroom --
/// while a pattern whose determinisation is exponential passes right through
/// `MAX_STATES` because it never reaches the check.
///
/// Both limits are set: `determinize_size_limit` bounds the scratch space the
/// subset construction uses, and `dfa_size_limit` the table it produces. A
/// pattern past either is `None` here, which is a refusal every caller of a
/// descriptor operation already handles -- the relation stays undecided and the
/// walk still matches the pattern, since the walk runs the regex rather than
/// the automaton.
pub const BUILD_SIZE_LIMIT: usize = 8 * 1024 * 1024;

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

    /// The automaton accepting exactly the valid UTF-8 words.
    ///
    /// Written out rather than compiled from a pattern, for the reason
    /// [`RegularSet::word`] is: this is the `str` kind's *universe*, so it is
    /// asked for wherever a complement is taken, and a constructor that can
    /// refuse would leave a caller choosing between a wrong set and a panic.
    /// Nine states and thirteen byte classes, which is the standard decoder
    /// read as an acceptor:
    ///
    /// * `0x00..=0x7F` stands alone;
    /// * `0xC2..=0xDF` takes one continuation, `0xE0..=0xEF` two and
    ///   `0xF0..=0xF4` three;
    /// * `0xE0`, `0xED`, `0xF0` and `0xF4` narrow the *first* of those, which
    ///   is what rules out an overlong encoding, a surrogate and a code point
    ///   above `U+10FFFF`;
    /// * `0xC0`, `0xC1` and `0xF5..=0xFF` begin nothing.
    fn utf8() -> Dfa {
        // Class 0 is the catch-all, which here is the bytes no well-formed word
        // contains at all; the states are named by what they still expect, and
        // `DEAD` absorbs.
        const ASCII: Class = 1;
        const CONT_LOW: Class = 2;
        const CONT_MID: Class = 3;
        const CONT_HIGH: Class = 4;
        const LEAD_TWO: Class = 5;
        const LEAD_E0: Class = 6;
        const LEAD_THREE: Class = 7;
        const LEAD_ED: Class = 8;
        const LEAD_THREE_HIGH: Class = 9;
        const LEAD_F0: Class = 10;
        const LEAD_FOUR: Class = 11;
        const LEAD_F4: Class = 12;
        const CLASSES: usize = 13;
        const START: u32 = 0;
        const ONE: u32 = 1;
        const TWO: u32 = 2;
        const TWO_HIGH: u32 = 3;
        const TWO_LOW: u32 = 4;
        const THREE_MID_UP: u32 = 5;
        const THREE_LOW: u32 = 6;
        const THREE: u32 = 7;
        const DEAD: u32 = 8;
        const STATES: usize = 9;

        let mut classes: Vec<Class> = vec![0; 256];
        let mut put = |range: core::ops::RangeInclusive<u8>, class: Class| {
            for byte in range {
                if let Some(slot) = classes.get_mut(usize::from(byte)) {
                    *slot = class;
                }
            }
        };
        put(0x00..=0x7F, ASCII);
        put(0x80..=0x8F, CONT_LOW);
        put(0x90..=0x9F, CONT_MID);
        put(0xA0..=0xBF, CONT_HIGH);
        put(0xC2..=0xDF, LEAD_TWO);
        put(0xE0..=0xE0, LEAD_E0);
        put(0xE1..=0xEC, LEAD_THREE);
        put(0xED..=0xED, LEAD_ED);
        put(0xEE..=0xEF, LEAD_THREE_HIGH);
        put(0xF0..=0xF0, LEAD_F0);
        put(0xF1..=0xF3, LEAD_FOUR);
        put(0xF4..=0xF4, LEAD_F4);

        let mut transitions = vec![DEAD; STATES * CLASSES];
        let mut edge = |state: u32, class: Class, target: u32| {
            let index = (state as usize) * CLASSES + usize::from(class);
            if let Some(slot) = transitions.get_mut(index) {
                *slot = target;
            }
        };
        edge(START, ASCII, START);
        edge(START, LEAD_TWO, ONE);
        edge(START, LEAD_E0, TWO_HIGH);
        edge(START, LEAD_THREE, TWO);
        edge(START, LEAD_ED, TWO_LOW);
        edge(START, LEAD_THREE_HIGH, TWO);
        edge(START, LEAD_F0, THREE_MID_UP);
        edge(START, LEAD_FOUR, THREE);
        edge(START, LEAD_F4, THREE_LOW);
        for class in [CONT_LOW, CONT_MID, CONT_HIGH] {
            edge(ONE, class, START);
            edge(TWO, class, ONE);
            edge(THREE, class, TWO);
        }
        edge(TWO_HIGH, CONT_HIGH, ONE);
        edge(TWO_LOW, CONT_LOW, ONE);
        edge(TWO_LOW, CONT_MID, ONE);
        edge(THREE_MID_UP, CONT_MID, TWO);
        edge(THREE_MID_UP, CONT_HIGH, TWO);
        edge(THREE_LOW, CONT_LOW, TWO);

        let mut accepting = vec![false; STATES];
        if let Some(flag) = accepting.get_mut(START as usize) {
            *flag = true;
        }
        Dfa {
            classes,
            class_count: CLASSES,
            transitions,
            accepting,
        }
        .minimal()
    }

    /// Every word this automaton does not accept.
    ///
    /// A flip of the accepting flags, which is only sound because the table is
    /// *complete*: a partial automaton rejects by falling off the end, and
    /// flipping its flags would accept those words instead of the ones it meant.
    ///
    /// **The flip of a minimal table is minimal**, so this re-minimises nothing.
    /// A word distinguishes two states in `L` exactly when it distinguishes them
    /// in `¬L` -- it leads one to acceptance and the other not, either way round
    /// -- so the state partition is the same; and the transitions and the byte
    /// classes are untouched, so the canonical numbering a walk over them
    /// produces is too. Every constructor here returns a minimal table, which is
    /// what makes that precondition hold, and
    /// `a_complement_is_already_minimal` holds the conclusion.
    fn complement(&self) -> Dfa {
        Dfa {
            classes: self.classes.clone(),
            class_count: self.class_count,
            transitions: self.transitions.clone(),
            accepting: self.accepting.iter().map(|a| !a).collect(),
        }
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

/// The valid UTF-8 words, which is the `str` kind's universe.
static UTF8: OnceLock<Arc<Dfa>> = OnceLock::new();

/// Every byte string, which is the `bytes` kind's universe.
static EVERY_WORD: OnceLock<Arc<Dfa>> = OnceLock::new();

/// A set of words: a regular language.
///
/// The table sits behind a handle, so a copy is a reference count. A descriptor
/// carries one of these per word kind and is cloned wherever a line is built --
/// a kind's universe alone is rebuilt at every complement of every slot -- and
/// the table is immutable once built, which is the shape a shared handle is
/// for. Equality and order read the table through it, so two spellings of one
/// language stay one value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegularSet {
    dfa: Arc<Dfa>,
}

impl RegularSet {
    /// The empty set.
    #[must_use]
    pub fn empty() -> RegularSet {
        RegularSet {
            dfa: Arc::new(Dfa::empty()),
        }
    }

    /// Every word of an alphabet.
    ///
    /// The two differ, and the difference is the `str` kind's whole complement:
    /// every byte string is a `bytes`, and only the valid UTF-8 ones are a
    /// `str`. Reading the wider set as the `str` universe puts words in that
    /// kind's complement that no value of it takes, so a difference that holds
    /// only those reads as inhabited and refutes an inclusion nothing stands
    /// against.
    #[must_use]
    pub fn all(alphabet: Alphabet) -> RegularSet {
        // Both are built once and handed out as handles. A universe is asked for
        // wherever a slot is complemented -- which is every kind of every
        // descriptor -- and neither table depends on anything, so rebuilding one
        // per ask is a cost a constant has no business carrying.
        RegularSet {
            dfa: Arc::clone(match alphabet {
                Alphabet::Text => UTF8.get_or_init(|| Arc::new(Dfa::utf8())),
                Alphabet::Bytes => EVERY_WORD.get_or_init(|| Arc::new(Dfa::universal())),
            }),
        }
    }

    /// The language of one pattern, matched whole, or `None` where the pattern
    /// does not build, exceeds [`BUILD_SIZE_LIMIT`], or its automaton is past
    /// [`MAX_STATES`].
    ///
    /// Anchored at both ends, because that is what a `Regex` constraint means:
    /// the walk matches a pattern against the whole text, not a substring of it.
    ///
    /// The size limits are the load-bearing part. `MAX_STATES` is checked in
    /// `Dfa::from_automaton`, which runs *after* `regex-automata` has built
    /// the complete dense table -- so a pattern whose determinisation is
    /// exponential allocated gigabytes and then refused, or aborted the process
    /// trying. `(a|b)*a(a|b){k}` is the family: every `k` doubles the states,
    /// and at 20 the build reached 668 MB before valgebra saw a single state.
    /// Handing the limits to the builder makes the refusal happen where the
    /// memory would be spent.
    #[must_use]
    pub fn pattern(pattern: &str, alphabet: Alphabet) -> Option<RegularSet> {
        let anchored = format!("(?:{pattern})");
        let syntax = syntax::Config::new().utf8(alphabet == Alphabet::Text);
        let config = dense::Config::new()
            .start_kind(regex_automata::dfa::StartKind::Anchored)
            .dfa_size_limit(Some(BUILD_SIZE_LIMIT))
            .determinize_size_limit(Some(BUILD_SIZE_LIMIT));
        let built = dense::Builder::new()
            .syntax(syntax)
            .configure(config)
            .build(&anchored)
            .ok()?;
        Dfa::from_automaton(&built).map(|dfa| RegularSet {
            dfa: Arc::new(dfa.minimal()),
        })
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
            dfa: Arc::new(
                Dfa {
                    classes,
                    class_count,
                    transitions,
                    accepting,
                }
                .minimal(),
            ),
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
            .map(|dfa| RegularSet { dfa: Arc::new(dfa) })
    }

    /// The words in both sets, or `None` past [`MAX_STATES`].
    #[must_use]
    pub fn intersect(&self, other: &RegularSet) -> Option<RegularSet> {
        self.dfa
            .product(&other.dfa, |a, b| a && b)
            .map(|dfa| RegularSet { dfa: Arc::new(dfa) })
    }

    /// Every word this set does not hold, **over every byte string**.
    ///
    /// Total, unlike the other two: a complement neither adds states nor
    /// refines the alphabet, so it cannot pass the bound.
    ///
    /// A `RegularSet` carries no alphabet, so this cannot cut its answer back to
    /// one. That matters for [`Alphabet::Text`] alone -- every byte string is a
    /// `bytes` and only the valid UTF-8 ones are a `str` -- and the cut belongs
    /// to the caller that knows the kind: `Line::complement` meets this with
    /// [`RegularSet::all`] for the kind it complements within.
    #[must_use]
    pub fn complement(&self) -> RegularSet {
        RegularSet {
            dfa: Arc::new(self.dfa.complement()),
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
mod tests;
