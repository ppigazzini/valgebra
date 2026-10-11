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
//! The determiniser is asked for every match rather than for the one a search
//! prefers, since a set of words has no preference among its members.

use regex_automata::dfa::{Automaton, dense};
use regex_automata::util::syntax;
use regex_automata::{Anchored, Input};
use rustc_hash::FxHashMap;
use std::sync::{Arc, OnceLock};

pub use super::automaton::MAX_STATES;
use super::automaton::{Reached, canonical_order, reaches};

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
///
/// **The lookups below answer a miss with state or class zero rather than
/// refusing.** A well-formed table has no miss -- [`well_formed`](Self::well_formed)
/// says what that means -- and the operations that read one are total: a
/// complement, a membership walk, the three passes of a minimisation. Read as
/// zero, a miss would be another language, so [`minimal`](Self::minimal),
/// which every table here passes through or is built as, asserts the table it
/// reads and each one it builds: a malformed table stops a debug build where it
/// is made.
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

    /// Whether every lookup on this table lands inside it: a class for each of
    /// the 256 bytes and each below the class count, a row of transitions per
    /// state, each to a state of the table, and a state to start from. A class
    /// below the count is what makes the count positive, so no clause of its
    /// own says so.
    fn well_formed(&self) -> bool {
        let states = self.state_count();
        states > 0
            && self.classes.len() == 256
            && self
                .classes
                .iter()
                .all(|class| usize::from(*class) < self.class_count)
            && Some(self.transitions.len()) == states.checked_mul(self.class_count)
            && self
                .transitions
                .iter()
                .all(|target| (*target as usize) < states)
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
        !reaches(
            |state| self.accepts(state),
            |state, pending| {
                for class in 0..self.class_count {
                    pending.push_back(self.step(state, class.try_into().unwrap_or(0)));
                }
            },
        )
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

    /// The automaton accepting the encodings of every Python `str`.
    ///
    /// A `str` is a sequence of code points, and a **lone surrogate is one of
    /// them**: `"\u{d800}"` is one character long and is a member of `str`. Its
    /// encoding is not valid UTF-8 -- no codec produces one -- so the language
    /// here is WTF-8, which is UTF-8 with the surrogate code points spelled the
    /// way the three-byte form spells everything else.
    ///
    /// That is the kind's *universe*, and it has to hold them. A pattern's
    /// language is built in UTF-8 and holds none, which is the walk's answer too
    /// (a `Regex` matches the text of a string and a lone surrogate has none),
    /// so a universe without them makes the difference between `str` and a
    /// catch-all pattern empty -- proving an inclusion `"\u{d800}"` refutes.
    ///
    /// Written out rather than compiled from a pattern, for the reason
    /// [`RegularSet::word`] is: it is asked for wherever a complement is taken,
    /// and a constructor that can refuse would leave a caller choosing between a
    /// wrong set and a panic. Thirteen byte classes, which is the standard
    /// decoder read as an acceptor:
    ///
    /// * `0x00..=0x7F` stands alone;
    /// * `0xC2..=0xDF` takes one continuation, `0xE0..=0xEF` two and
    ///   `0xF0..=0xF4` three;
    /// * `0xE0`, `0xF0` and `0xF4` narrow the *first* of those, which is what
    ///   rules out an overlong encoding and a code point above `U+10FFFF`;
    /// * `0xED` does **not** narrow it, which is what admits the surrogates;
    /// * `0xC0`, `0xC1` and `0xF5..=0xFF` begin nothing.
    fn wtf8() -> Dfa {
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
        // `0xED 0xA0..0xBF` is `U+D800..U+DFFF`, the surrogates. A UTF-8
        // acceptor stops here and a `str` does not, so the edge is what makes
        // this the kind's universe rather than the codecs'.
        edge(TWO_LOW, CONT_HIGH, ONE);
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
    /// One walk over reachable state *pairs* ([`Reached`]), so the states built
    /// are the ones a word can reach rather than the whole cross product. The
    /// alphabets are refined together first: a class of the product is a pair
    /// of classes, one from each side, and only the pairs some byte realises
    /// get an id.
    fn product(&self, other: &Dfa, accept: impl Fn(bool, bool) -> bool) -> Option<Dfa> {
        let (classes, class_count, pairs) = refine_alphabets(self, other);
        let mut reached = Reached::from((0, 0));
        let mut transitions: Vec<u32> = Vec::new();
        let mut accepting: Vec<bool> = Vec::new();
        let mut built = 0usize;
        while let Some((mine, theirs)) = reached.next() {
            built += 1;
            accepting.push(accept(self.accepts(mine), other.accepts(theirs)));
            for pair in &pairs {
                let next = (self.step(mine, pair.0), other.step(theirs, pair.1));
                transitions.push(reached.id(next, || true)?);
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
        debug_assert!(self.well_formed(), "a table is minimised as it was built");
        let merged = self.merge_equivalent();
        debug_assert!(merged.well_formed(), "merging the states kept the table");
        let coarsened = merged.coarsen();
        debug_assert!(coarsened.well_formed(), "coarsening the alphabet kept it");
        let minimal = coarsened.renumber();
        debug_assert!(minimal.well_formed(), "renumbering the states kept it");
        minimal
    }

    /// Merge the states no word distinguishes: one state for each block of
    /// [`equivalent_states`](Self::equivalent_states).
    fn merge_equivalent(&self) -> Dfa {
        let block = self.equivalent_states();
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

    /// The states no word tells apart: a block for each state, numbered from
    /// zero at state zero's, two states sharing one exactly when they accept
    /// the same words.
    ///
    /// Hopcroft's algorithm. The accepting states and the rest start as two
    /// blocks, and a block is cut wherever one class leads part of it into a
    /// block, the *splitter*, and the rest of it elsewhere. Each part a cut
    /// makes is a splitter in turn -- except that of the two parts only the
    /// smaller is queued: a block no class cuts against a set and against one
    /// part of it is one no class cuts against the other part either. So a
    /// state is in a queued block at most `log n` times, and the work is
    /// `k n log n` over `n` states and `k` classes.
    ///
    /// A pass that signs every state afresh each round until a round changes
    /// nothing reaches the same partition and spends a round per state on a
    /// chain. A literal is a chain, and one of four thousand bytes took 0.6 s
    /// to build that way on a release build, which every product keeping the
    /// chain spent again. `the_halves_cut_where_the_rounds_do` holds the two to
    /// one partition.
    ///
    /// Each block is queued once, when it is made, and a block is a nonempty
    /// set of states apart from every other, so the queue drains within one
    /// pop per state. Bounding the loop by that is what makes a broken cut
    /// leave a wrong table, which the properties reject, rather than run
    /// without end.
    fn equivalent_states(&self) -> Vec<u32> {
        let predecessors = Predecessors::of(self);
        let mut partition = Partition::by_acceptance(&self.accepting);
        let mut queued: Vec<u32> = (0..partition.block_count())
            .filter_map(|block| u32::try_from(block).ok())
            .collect();
        let mut splitter: Vec<u32> = Vec::new();
        for _ in 0..self.state_count() {
            let Some(block) = queued.pop() else {
                break;
            };
            // A copy of the splitter's members: a cut below can move them, or
            // cut this very block, and every class still reads the set the
            // block was when it left the queue.
            splitter.clear();
            splitter.extend_from_slice(partition.members(block));
            for class in 0..self.class_count {
                for target in &splitter {
                    for source in predecessors.leading_to(class, *target) {
                        partition.mark(*source);
                    }
                }
                partition.cut_marked(&mut queued);
            }
        }
        // The merged table starts where state zero does, so that block trades
        // its number with block zero's.
        let start = partition.block.first().copied().unwrap_or(0);
        for block in &mut partition.block {
            if *block == start {
                *block = 0;
            } else if *block == 0 {
                *block = start;
            }
        }
        partition.block
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
        for byte in u8::MIN..=u8::MAX {
            let old = self.class_of(byte);
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
    /// ones no word reaches ([`canonical_order`]).
    fn renumber(&self) -> Dfa {
        let (order, ids) = canonical_order(|state| {
            (0..self.class_count).map(move |class| self.step(state, class.try_into().unwrap_or(0)))
        });
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
fn refine_alphabets(a: &Dfa, b: &Dfa) -> (Vec<Class>, usize, Vec<(Class, Class)>) {
    let mut ids: FxHashMap<(Class, Class), Class> = FxHashMap::default();
    let mut pairs: Vec<(Class, Class)> = Vec::new();
    let mut classes: Vec<Class> = Vec::with_capacity(256);
    for byte in u8::MIN..=u8::MAX {
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

/// For each class and target state, the states the class leads there from:
/// the transitions read backwards, which is what a cut asks.
///
/// One run of `sources` for each pair of a class and a target, laid end to
/// end, so the whole reversal is two allocations however many states there
/// are.
struct Predecessors {
    /// Where each run begins in `sources`, and one entry past the last run, so
    /// a run ends where the next begins.
    starts: Vec<usize>,
    sources: Vec<u32>,
    /// How many runs one class has, which is the table's state count.
    states: usize,
}

impl Predecessors {
    fn of(dfa: &Dfa) -> Predecessors {
        let states = dfa.state_count();
        let run = |class: usize, target: u32| class * states + target as usize;
        let rows = || dfa.transitions.chunks(dfa.class_count.max(1)).enumerate();
        // Count each run's sources into the entry after it, so a running sum
        // turns the counts into where each run begins.
        let mut starts = vec![0usize; dfa.class_count * states + 1];
        for (_, row) in rows() {
            for (class, target) in row.iter().enumerate() {
                if let Some(count) = starts.get_mut(run(class, *target) + 1) {
                    *count += 1;
                }
            }
        }
        let mut sum = 0;
        for start in &mut starts {
            sum += *start;
            *start = sum;
        }
        // Then fill each run from where it begins.
        let mut next = starts.clone();
        let mut sources = vec![0u32; dfa.transitions.len()];
        for (state, row) in rows() {
            for (class, target) in row.iter().enumerate() {
                if let Some(at) = next.get_mut(run(class, *target)) {
                    if let Some(slot) = sources.get_mut(*at) {
                        *slot = u32::try_from(state).unwrap_or(0);
                    }
                    *at += 1;
                }
            }
        }
        Predecessors {
            starts,
            sources,
            states,
        }
    }

    /// The states `class` leads to `target` from.
    fn leading_to(&self, class: usize, target: u32) -> &[u32] {
        let run = class * self.states + target as usize;
        let begin = self.starts.get(run).copied().unwrap_or(0);
        let end = self.starts.get(run + 1).copied().unwrap_or(begin);
        self.sources.get(begin..end).unwrap_or(&[])
    }
}

/// A partition of a table's states into blocks, refined by marking the states
/// a splitter reaches and cutting each block between its marked and unmarked
/// members.
///
/// A block's members sit side by side in `members`, its marked ones first, so
/// a mark is a swap and a cut is a boundary moved: the work is in the states
/// marked and the states relabelled, never in the size of a block they leave.
struct Partition {
    /// The states, each block's members side by side.
    members: Vec<u32>,
    /// Where each state sits in `members`.
    position: Vec<usize>,
    /// Each state's block.
    block: Vec<u32>,
    /// Each block's run of `members`, from `start` to `end`.
    start: Vec<usize>,
    end: Vec<usize>,
    /// How many of each block's members are marked.
    marked: Vec<usize>,
    /// The blocks with a marked member, each once.
    touched: Vec<u32>,
}

impl Partition {
    /// The rejecting states and the accepting ones, each a block where it has
    /// a member: the cut a word of length zero already makes.
    fn by_acceptance(accepting: &[bool]) -> Partition {
        let mut partition = Partition {
            members: Vec::with_capacity(accepting.len()),
            position: vec![0; accepting.len()],
            block: vec![0; accepting.len()],
            start: Vec::new(),
            end: Vec::new(),
            marked: Vec::new(),
            touched: Vec::new(),
        };
        for side in [false, true] {
            let begin = partition.members.len();
            let id = u32::try_from(partition.block_count()).unwrap_or(0);
            for (state, accepts) in accepting.iter().enumerate() {
                if *accepts != side {
                    continue;
                }
                if let Some(slot) = partition.position.get_mut(state) {
                    *slot = partition.members.len();
                }
                if let Some(slot) = partition.block.get_mut(state) {
                    *slot = id;
                }
                partition.members.push(u32::try_from(state).unwrap_or(0));
            }
            if partition.members.len() > begin {
                partition.start.push(begin);
                partition.end.push(partition.members.len());
                partition.marked.push(0);
            }
        }
        partition
    }

    fn block_count(&self) -> usize {
        self.start.len()
    }

    fn members(&self, block: u32) -> &[u32] {
        let block = block as usize;
        let begin = self.start.get(block).copied().unwrap_or(0);
        let end = self.end.get(block).copied().unwrap_or(begin);
        self.members.get(begin..end).unwrap_or(&[])
    }

    /// Mark `state`: it trades places with its block's first unmarked member.
    ///
    /// The lookups answer a miss with zero, as the table's do: every state has
    /// a block and a place, and every block a run, by construction.
    fn mark(&mut self, state: u32) {
        let index = state as usize;
        let block = self.block.get(index).copied().unwrap_or(0);
        let at = self.position.get(index).copied().unwrap_or(0);
        let marked = self.marked.get(block as usize).copied().unwrap_or(0);
        let boundary = self.start.get(block as usize).copied().unwrap_or(0) + marked;
        if at < boundary {
            return;
        }
        if let Some(count) = self.marked.get_mut(block as usize) {
            *count += 1;
        }
        if marked == 0 {
            self.touched.push(block);
        }
        let displaced = self.members.get(boundary).copied().unwrap_or(state);
        if let Some(slot) = self.members.get_mut(at) {
            *slot = displaced;
        }
        if let Some(slot) = self.members.get_mut(boundary) {
            *slot = state;
        }
        if let Some(slot) = self.position.get_mut(displaced as usize) {
            *slot = at;
        }
        if let Some(slot) = self.position.get_mut(index) {
            *slot = boundary;
        }
    }

    /// Cut each touched block between its marked and unmarked members, give
    /// the smaller part a new block, and queue it.
    ///
    /// The larger part keeps the block's id, and with it the block's place in
    /// the queue where it had one, so both parts of a queued block are queued.
    /// A block whose members are all marked is not cut. The marks are cleared.
    fn cut_marked(&mut self, queued: &mut Vec<u32>) {
        for block in std::mem::take(&mut self.touched) {
            let block = block as usize;
            let begin = self.start.get(block).copied().unwrap_or(0);
            let end = self.end.get(block).copied().unwrap_or(begin);
            let count = self.marked.get_mut(block).map_or(0, std::mem::take);
            let size = end - begin;
            if count == size {
                continue;
            }
            // The marked members are the first `count`, and a tie moves them.
            let (moved, kept) = if count <= size - count {
                (begin..begin + count, begin + count..end)
            } else {
                (begin + count..end, begin..begin + count)
            };
            if let (Some(first), Some(last)) = (self.start.get_mut(block), self.end.get_mut(block))
            {
                (*first, *last) = (kept.start, kept.end);
            }
            let id = u32::try_from(self.block_count()).unwrap_or(0);
            self.start.push(moved.start);
            self.end.push(moved.end);
            self.marked.push(0);
            for state in self.members.get(moved).unwrap_or(&[]) {
                if let Some(slot) = self.block.get_mut(*state as usize) {
                    *slot = id;
                }
            }
            queued.push(id);
        }
    }
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

/// The encodings of every Python `str`, which is that kind's universe.
static STR_WORDS: OnceLock<Arc<Dfa>> = OnceLock::new();

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
    /// every byte string is a `bytes`, and a `str` is one that encodes a
    /// sequence of code points. Reading the wider set as the `str` universe puts
    /// words in that kind's complement that no value of it takes, so a
    /// difference that holds only those reads as inhabited and refutes an
    /// inclusion nothing stands against.
    #[must_use]
    pub fn all(alphabet: Alphabet) -> RegularSet {
        // Both are built once and handed out as handles. A universe is asked for
        // wherever a slot is complemented -- which is every kind of every
        // descriptor -- and neither table depends on anything, so rebuilding one
        // per ask is a cost a constant has no business carrying.
        RegularSet {
            dfa: Arc::clone(match alphabet {
                Alphabet::Text => STR_WORDS.get_or_init(|| Arc::new(Dfa::wtf8())),
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
        // A caller's pattern is read over *text*, whichever kind it constrains,
        // which is what the walk does with it: a `Regex` matches the characters
        // of a string. So it compiles in UTF-8 mode for `str`, and its language
        // holds no surrogate -- as the kind's universe does and the walk's
        // answer says.
        let utf8 = alphabet == Alphabet::Text;
        // The pattern on its own first. One that does not parse alone can parse
        // wrapped -- `a)|(b` closes the group and opens one the wrapper closes
        // -- and would then be a language nobody wrote.
        syntax::parse_with(pattern, &syntax::Config::new().utf8(utf8)).ok()?;
        RegularSet::compile(&format!("(?:{pattern})"), utf8)
    }

    /// Build a language from a pattern the caller of this module wrote.
    fn compile(anchored: &str, utf8: bool) -> Option<RegularSet> {
        let syntax = syntax::Config::new().utf8(utf8);
        // Every match, which is the classical construction. The builder's
        // default is leftmost-first, a search semantics: once the thread a
        // backtracking engine prefers has matched, the threads ranked below it
        // are dropped, so `a|ab` reaches no state past `a` and `a*?` stops at the
        // empty word. That table holds the matches a search reports, a smaller
        // set than the words the pattern matches whole -- which is what the walk
        // asks and what a `Regex` denotes -- and a relation read off it proves an
        // inclusion a value refutes.
        let config = dense::Config::new()
            .start_kind(regex_automata::dfa::StartKind::Anchored)
            .match_kind(regex_automata::MatchKind::All)
            .dfa_size_limit(Some(BUILD_SIZE_LIMIT))
            .determinize_size_limit(Some(BUILD_SIZE_LIMIT));
        let built = dense::Builder::new()
            .syntax(syntax)
            .configure(config)
            .build(anchored)
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
        RegularSet::compile(&format!("{}{{{length},}}", code_point(alphabet)), false)
    }

    /// The words of at most `length` symbols.
    #[must_use]
    pub fn at_most(length: usize, alphabet: Alphabet) -> Option<RegularSet> {
        RegularSet::compile(&format!("{}{{0,{length}}}", code_point(alphabet)), false)
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

/// The pattern for one symbol of an alphabet, written over **bytes**.
///
/// A length bound counts the symbols a value has, and for text a symbol is a
/// code point -- a lone surrogate among them, since a `str` may carry one and
/// `"\u{d800}"` is one character long. UTF-8 mode cannot spell that character,
/// so the expansion is written out and read as bytes: it is the kind's universe
/// ([`Dfa::wtf8`]) taken one character at a time.
///
/// Neither arm skips a newline. A length bound is about how many symbols the
/// word has, not about which of them a pattern would pass over.
const fn code_point(alphabet: Alphabet) -> &'static str {
    match alphabet {
        Alphabet::Text => concat!(
            r"(?s-u:[\x00-\x7F]",
            r"|[\xC2-\xDF][\x80-\xBF]",
            r"|\xE0[\xA0-\xBF][\x80-\xBF]",
            r"|[\xE1-\xEF][\x80-\xBF][\x80-\xBF]",
            r"|\xF0[\x90-\xBF][\x80-\xBF][\x80-\xBF]",
            r"|[\xF1-\xF3][\x80-\xBF][\x80-\xBF][\x80-\xBF]",
            r"|\xF4[\x80-\x8F][\x80-\xBF][\x80-\xBF])"
        ),
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
        for byte in u8::MIN..=u8::MAX {
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
        let mut reached = Reached::from(start);
        let mut transitions: Vec<u32> = Vec::new();
        let mut accepting: Vec<bool> = Vec::new();
        while let Some(state) = reached.next() {
            accepting.push(built.is_match_state(built.next_eoi_state(state)));
            for class in 0..alphabet {
                // Every byte of a class takes the same transition, so one
                // representative stands for the column.
                let byte = representative(&byte_class, class)?;
                transitions.push(reached.id(built.next_state(state, byte), || true)?);
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
