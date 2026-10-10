//! What the two automata share: the bound on their states, and the walks a
//! deterministic table is read by.
//!
//! The word automaton (`regular.rs`) steps on byte classes and the sequence
//! automaton (`symbolic.rs`) on guarded edges, and each keeps its own table,
//! its own signature of a state and its own minimisation passes. What is the
//! same in both is the reading of a table as a graph from state zero: whether
//! an accepting state is reachable, the order a canonical numbering visits the
//! states in, the bounded refinement that merges the states no input tells
//! apart, and the walk that numbers the states a product or a builder reaches.
//! Each is written once here, over the targets the caller says a state leads to
//! and in the order the caller gives them, so the numbering a walk produces --
//! which is what makes a minimal table canonical -- is the caller's order and
//! nothing of this module's.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, VecDeque};
use std::hash::Hash;

use rustc_hash::{FxHashMap, FxHashSet};

/// The most states an automaton may hold.
///
/// A product multiplies the two state counts -- `|A| * |B|` states before
/// minimisation -- and a complement of a product does it again, so a bound is
/// what keeps a pathological pattern or shape from exhausting memory rather than
/// answering. Past it the set is not representable, and the operation says so
/// rather than returning an automaton for a different language. One bound for
/// both automata, since both multiply the same way.
pub const MAX_STATES: usize = 4096;

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
/// states, and numbering them by a walk that takes each state's transitions in
/// a fixed order removes the renaming. That order is `targets`', so it is the
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

/// The numbering a refinement round gives its signatures: each the next id the
/// first time it is seen.
///
/// A hash map where a signature can be hashed and an ordered map where it can
/// only be compared -- a sequence automaton's signature holds guards, which are
/// ordered rather than hashed. The ids depend only on the order the states are
/// read in, not on the map, so the two give one numbering.
pub(super) trait Numbering<K>: Default {
    /// The id of `signature`, the next one where it is new.
    fn number(&mut self, signature: K) -> u32;
}

impl<K: Hash + Eq> Numbering<K> for FxHashMap<K, u32> {
    fn number(&mut self, signature: K) -> u32 {
        let fresh = u32::try_from(self.len()).unwrap_or(0);
        *self.entry(signature).or_insert(fresh)
    }
}

impl<K: Ord> Numbering<K> for BTreeMap<K, u32> {
    fn number(&mut self, signature: K) -> u32 {
        let fresh = u32::try_from(self.len()).unwrap_or(0);
        *self.entry(signature).or_insert(fresh)
    }
}

/// The coarsest partition of `count` states that `signature` cannot split,
/// refined from `block` (Moore's algorithm).
///
/// Each round gives every state the id of its signature under the current
/// partition -- one more input of lookahead -- numbering signatures in state
/// order, so state zero's block is zero. The rounds stop when one changes
/// nothing. At most one round per state: each either splits a block or is the
/// last, and a block cannot split more often than it has members. Bounding it
/// is what makes a wrong termination test leave a coarser table -- which the
/// canonicity properties reject -- rather than run without end.
pub(super) fn refine<K, N: Numbering<K>>(
    count: usize,
    mut block: Vec<u32>,
    signature: impl Fn(usize, &[u32]) -> K,
) -> Vec<u32> {
    for _ in 0..=count {
        let mut ids = N::default();
        let next: Vec<u32> = (0..count)
            .map(|state| ids.number(signature(state, &block)))
            .collect();
        if next == block {
            break;
        }
        block = next;
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
