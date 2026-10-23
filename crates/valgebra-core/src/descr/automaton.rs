//! What the two automata share: the bound on their states, and the walks a
//! deterministic table is read by.
//!
//! The word automaton (`regular.rs`) steps on byte classes and the sequence
//! automaton (`symbolic.rs`) on guarded edges, and each keeps its own table
//! and its own minimization passes. What is the same in both is the reading of
//! a table as a graph from state zero: whether an accepting state is reachable,
//! the order a canonical numbering visits the states in, and the walk that
//! numbers the states a product or a builder reaches. Each is written once
//! here, over the targets the caller says a state leads to and in the order the
//! caller gives them, so the numbering a walk produces -- which is what makes a
//! minimal table canonical -- is the caller's order and nothing of this
//! module's.
//!
//! The refinement that merges the states no input tells apart is here too, and
//! the sequence automaton alone reads it. The word automaton cuts its blocks by
//! the transitions read backwards, which takes `k n log n` work over its byte
//! classes. Over guards, reading the transitions backwards needs the minterms
//! of every guard, or D'Antoni & Veanes's Figure 4, which does without them at
//! the cost of a satisfiability check per split; rounds of signatures need
//! neither. From one block they take a round per state on a chain, so the
//! sequence automaton starts them from a partition by each state's distances
//! (`seed` in `symbolic.rs`), which parts a chain before the first round.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, VecDeque, btree_map};
use std::hash::Hash;

use rustc_hash::{FxHashMap, FxHashSet};

/// The most states an automaton may hold.
///
/// A product multiplies the two state counts -- `|A| * |B|` states before
/// minimization -- and a complement of a product does it again, so a bound is
/// what keeps a pathological pattern or shape from exhausting memory rather than
/// answering. Past it the set is not representable, and the operation says so
/// rather than returning an automaton for a different language. One bound for
/// both automata, since both multiply the same way.
///
/// It holds every table as it is built, before minimization: a table reached
/// by a walk through `Reached`, and a table built as a chain -- a literal word,
/// a sequence shape -- through `chain_fits`.
pub(crate) const MAX_STATES: usize = 4096;

/// Whether a chain of `letters` fits [`MAX_STATES`] as it is built: a state for
/// each letter, the state after the last, and the sink a wrong letter leads to.
///
/// A leaf built as a chain is never walked, so [`Reached`] does not see it, and
/// past the bound its table buys no answer, since the first product it enters
/// refuses. Asked before anything is built, the refusal costs one comparison:
/// built, a word of a million bytes is a table of a million states minimized
/// before that product refuses it.
pub(super) const fn chain_fits(letters: usize) -> bool {
    letters <= MAX_STATES - 2
}

/// Whether a walk from state zero reaches a state `accepts` holds, following
/// the targets `follow` pushes for each state it visits.
///
/// Each state is visited once, so the walk ends because the states are finite.
/// The caller decides which transitions are followed: every one, for a proof
/// that no accepting state is reachable, or only the ones a value is proved to
/// take, for a proof that one is.
pub(super) fn reaches(
    accepts: impl Fn(u32) -> bool,
    mut follow: impl FnMut(u32, &mut VecDeque<u32>),
) -> bool {
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    let mut pending: VecDeque<u32> = VecDeque::from([0]);
    while let Some(state) = pending.pop_front() {
        // `insert` answers whether the state is new.
        if !seen.insert(state) {
            continue;
        }
        if accepts(state) {
            return true;
        }
        follow(state, &mut pending);
    }
    false
}

/// The states a breadth-first walk from state zero reaches, in the order it
/// first reaches them, and each one's position in that order.
///
/// The canonical numbering: a minimal table is unique up to renaming its
/// states, and over guards up to their equivalence (D'Antoni & Veanes, Theorem
/// 1); numbering the states by a walk that takes each state's transitions in a
/// fixed order removes the renaming. That order is `targets`', so it is the
/// caller's -- the byte classes in order, or the edges in guard order -- and a
/// state no walk reaches gets no number.
pub(super) fn canonical_order<I: IntoIterator<Item = u32>>(
    targets: impl Fn(u32) -> I,
) -> (Vec<u32>, FxHashMap<u32, u32>) {
    let mut ids: FxHashMap<u32, u32> = FxHashMap::default();
    let mut order: Vec<u32> = Vec::new();
    let mut pending: VecDeque<u32> = VecDeque::from([0]);
    ids.insert(0, 0);
    while let Some(state) = pending.pop_front() {
        order.push(state);
        for target in targets(state) {
            let fresh = u32::try_from(ids.len()).unwrap_or(0);
            if let Entry::Vacant(slot) = ids.entry(target) {
                slot.insert(fresh);
                pending.push_back(target);
            }
        }
    }
    (order, ids)
}

/// The coarsest partition of `count` states that `signature` cannot split,
/// refined from `block`: Moore's algorithm, by signatures.
///
/// Each round splits every block by its members' signatures under the current
/// partition -- one more input of lookahead. The part holding a block's first
/// member keeps the block's number and each other part takes the next free
/// one, so the numbers stay dense and state zero keeps the number the start
/// gave it. A block of one state cannot split, so a round signs only the
/// members of larger blocks, and a start that parts most of the states costs
/// each round that much less. The rounds stop when one splits nothing. At most
/// one round per state: each either splits a block or is the last, and a block
/// cannot split more often than it has members. Bounding it is what makes a
/// wrong termination test leave a coarser table -- which the canonicity
/// properties reject -- rather than run without end.
///
/// The signatures are numbered in an ordered map rather than a hash map,
/// because a sequence automaton's signature holds guards and a guard is
/// ordered rather than hashed. A scan for an equal signature would give the
/// same ids and read every kept signature per state, which is a round
/// quadratic in the state count on a table a length bound sizes.
pub(super) fn refine<K: Ord>(
    count: usize,
    mut block: Vec<u32>,
    signature: impl Fn(usize, &[u32]) -> K,
) -> Vec<u32> {
    // Each block's members, whether a part of it has kept its number, and the
    // next partition: kept across rounds, so a round allocates only its
    // signatures. The two tallies are apart so each is filled as one run.
    let (mut members, mut kept, mut next) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..=count {
        let blocks = block
            .iter()
            .copied()
            .max()
            .map_or(0, |last| last as usize + 1);
        members.clear();
        members.resize(blocks, 0u32);
        for id in &block {
            if let Some(size) = members.get_mut(*id as usize) {
                *size += 1;
            }
        }
        kept.clear();
        kept.resize(blocks, false);
        let mut fresh = u32::try_from(blocks).unwrap_or(0);
        let mut ids: BTreeMap<(u32, K), u32> = BTreeMap::new();
        next.clone_from(&block);
        for (state, slot) in next.iter_mut().enumerate() {
            let own = *slot;
            if members.get(own as usize).is_none_or(|size| *size < 2) {
                continue;
            }
            *slot = match ids.entry((own, signature(state, &block))) {
                btree_map::Entry::Occupied(seen) => *seen.get(),
                btree_map::Entry::Vacant(part) => {
                    let first = kept
                        .get_mut(own as usize)
                        .is_some_and(|taken| !std::mem::replace(taken, true));
                    let id = if first {
                        own
                    } else {
                        fresh += 1;
                        fresh - 1
                    };
                    *part.insert(id)
                }
            };
        }
        if fresh as usize == blocks {
            break;
        }
        std::mem::swap(&mut block, &mut next);
    }
    block
}

/// The states a walk reaches from one start, each numbered the first time the
/// walk reaches it: a product's reachable pairs of states, or the states of an
/// automaton built elsewhere as a walk reads it in.
///
/// Only the states a walk reaches are built, so a product holds the pairs an
/// input can reach rather than the whole cross product. A draining queue, so
/// the walk ends when nothing is left rather than when an index catches up with
/// a list that is still growing.
pub(super) struct Reached<K> {
    ids: FxHashMap<K, u32>,
    pending: VecDeque<K>,
}

impl<K: Hash + Eq + Copy> Reached<K> {
    /// The walk at its start: `start`, numbered zero.
    pub(super) fn from(start: K) -> Reached<K> {
        let mut ids = FxHashMap::default();
        ids.insert(start, 0);
        Reached {
            ids,
            pending: VecDeque::from([start]),
        }
    }

    /// The next state to build a row for, in the order the walk reached them.
    pub(super) fn next(&mut self) -> Option<K> {
        self.pending.pop_front()
    }

    /// The id of `state`, numbering it and queueing it where it is new, or
    /// `None` where a new state would pass [`MAX_STATES`] or `admit` refuses
    /// it. `admit` is asked only of a new state, after the bound.
    pub(super) fn id(&mut self, state: K, admit: impl FnOnce() -> bool) -> Option<u32> {
        if let Some(id) = self.ids.get(&state) {
            return Some(*id);
        }
        if self.ids.len() >= MAX_STATES || !admit() {
            return None;
        }
        let id = u32::try_from(self.ids.len()).ok()?;
        self.ids.insert(state, id);
        self.pending.push_back(state);
        Some(id)
    }
}
