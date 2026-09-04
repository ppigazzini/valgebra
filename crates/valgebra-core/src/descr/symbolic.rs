//! Sequences as an automaton whose transitions are guarded by value sets.
//!
//! A sequence type is a regular language over its *elements*, which is the
//! reason Hosoya, Vouillon and Pierce give for it being a first-class member of
//! the algebra: regular languages are closed under union, intersection and
//! complement, so the closure a descriptor component needs comes for free.
//! `str` already uses that (see [`regular`](super::regular)); this is the same
//! construction one alphabet up, where a letter is a *set of values* rather than
//! a byte.
//!
//! An alphabet of value sets cannot be enumerated, so the transitions carry
//! **guards**: each edge is labelled by a set, and the edges leaving a state
//! partition the universe -- pairwise disjoint, and covering. That is what keeps
//! the machine deterministic and complete without listing an alphabet, and it is
//! what makes a complement a flip of the accepting states rather than a
//! construction. The technique is a symbolic automaton; the only thing it asks
//! of a letter is that value sets form a Boolean algebra, which is what
//! [`Guard`] says.
//!
//! **The recursion lives in the states, not in the guards.** `list[T]` is one
//! accepting state with a self-loop guarded by `T` -- the cycle is an edge, so
//! the guard is an ordinary finite descriptor and nothing has to be interned to
//! break a cycle. `tuple[A, B]` is a chain of three, and
//! `tuple[A, *tuple[B, ...], C]` is a chain with a loop in the middle, which is
//! why the three spellings need one constructor rather than three nodes.

use crate::decision::Verdict;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// The most states an automaton may hold, for the reason
/// [`regular`](super::regular) gives: a product multiplies the state counts, so
/// a bound is what keeps a pathological combination from exhausting memory
/// rather than answering.
pub const MAX_STATES: usize = 4096;

/// The most transitions one state may have.
///
/// The other dimension a product multiplies. [`MAX_STATES`] bounds how many
/// positions a shape distinguishes; this bounds how many *alternatives* one
/// position holds, and neither implies the other -- a product's row is the
/// pairwise meets of the two sides' rows, so a table well inside the state bound
/// can still have rows too wide to hold.
///
/// The same figure as the state bound: as many alternatives at one position as
/// there are positions in a shape. Both are far past what an annotation writes
/// -- `list[int | str]` branches three ways -- and both are limits of the
/// representation rather than approximations, so a shape past either refuses.
pub const MAX_ROW: usize = MAX_STATES;

/// A Boolean algebra of value sets, which is what an automaton's guards must
/// form.
///
/// The three operations plus emptiness are all the machine asks of a letter.
/// `meet` and `join` may refuse -- a component of the descriptor is bounded, and
/// past that bound there is no sound set to return -- so a construction that
/// needs one carries the refusal up rather than substituting a set that is wrong
/// in one direction and, complemented, wrong in the other.
///
/// **There is deliberately no way to name the whole universe.** A letter here
/// may be a whole descriptor, and a descriptor's universe holds every sequence,
/// whose guard would be that universe again -- so `any()` cannot be written for
/// the letter this machine exists to serve. The edge lists carry an *else* edge
/// instead, which is what a total transition is without a total guard.
///
/// The total order is not part of the algebra; it is what makes the automaton's
/// form *canonical*. Edges leaving a state are held in guard order, so two
/// states with the same transition function have the same edge list, and the
/// minimal automaton has one table rather than one per edge permutation.
pub trait Guard: Clone + Eq + Ord + core::fmt::Debug {
    /// A value a guard is asked about.
    type Value;

    /// The set holding no value.
    fn none() -> Self;
    /// The values in both sets.
    fn meet(&self, other: &Self) -> Option<Self>;
    /// The values in either set.
    fn join(&self, other: &Self) -> Option<Self>;
    /// The values in neither.
    #[must_use]
    fn complement(&self) -> Self;
    /// Whether the set holds no value.
    ///
    /// A *proof*, not a guess: a letter that cannot decide answers `false`, and
    /// [`emptiness`](Guard::emptiness) is what tells the two apart.
    fn is_empty(&self) -> bool;

    /// What is known about the set holding a value.
    ///
    /// Exact for a letter whose emptiness is a computation over a finite
    /// structure, which is why the default answers from
    /// [`is_empty`](Guard::is_empty) alone. A letter that carries an open-world
    /// constraint -- a class the core cannot enumerate the subclasses of -- has
    /// a third answer and overrides this.
    fn emptiness(&self) -> Verdict {
        if self.is_empty() {
            Verdict::Empty
        } else {
            Verdict::Inhabited
        }
    }
    /// Whether the set holds `value`.
    fn holds(&self, value: &Self::Value) -> bool;
}

/// One outgoing edge: the values that take it, and where they go.
///
/// A guard of `None` is the *else* edge, taking every value the state's other
/// edges do not. Every state has exactly one, last in the list, which is how a
/// transition is total without a guard that names the universe -- and it is why
/// a letter needs no `any()`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Edge<G> {
    pub(crate) guard: Option<G>,
    pub(crate) target: u32,
}

impl<G: Guard> Edge<G> {
    /// Whether this edge takes `value`, which for the else edge is decided by
    /// the caller having tried the others first.
    fn takes(&self, value: &G::Value) -> bool {
        self.guard.as_ref().is_none_or(|guard| guard.holds(value))
    }

    /// Whether no value takes this edge. The else edge's own set is what the
    /// others leave, which only [`rest_of`] can say.
    fn is_dead(&self) -> bool {
        self.guard.as_ref().is_some_and(Guard::is_empty)
    }
}

/// What distinguishes one state from another during minimisation: whether it
/// accepts, and the values it sends to each block it can reach.
type Signature<G> = (bool, Vec<(u32, Option<G>)>);

/// A deterministic, complete automaton over guarded transitions.
///
/// State zero is the start. The edges leaving each state partition the value
/// universe and are held in guard order, so the table is a name for the
/// language: equal tables are equal languages, and the minimisation below is
/// what makes the converse true.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SymbolicDfa<G: Guard> {
    edges: Vec<Vec<Edge<G>>>,
    accepting: Vec<bool>,
}

impl<G: Guard> SymbolicDfa<G> {
    /// The language holding no sequence.
    #[must_use]
    pub fn empty() -> SymbolicDfa<G> {
        SymbolicDfa::single(false)
    }

    /// The language holding every sequence.
    #[must_use]
    pub fn all() -> SymbolicDfa<G> {
        SymbolicDfa::single(true)
    }

    /// One state looping to itself on every value, through its else edge.
    fn single(accepting: bool) -> SymbolicDfa<G> {
        SymbolicDfa {
            edges: vec![vec![Edge {
                guard: None,
                target: 0,
            }]],
            accepting: vec![accepting],
        }
    }

    /// The sequences whose first elements match `prefix` positionally and whose
    /// remaining elements all match `tail`.
    ///
    /// The one constructor the three spellings need. An empty prefix with a tail
    /// is `list[T]`; a prefix with no tail is `tuple[A, B]`; both together are
    /// `tuple[A, *tuple[B, ...]]`. A sequence longer than the prefix with no
    /// tail, or holding an element the guard rejects, reaches the sink and is
    /// not accepted.
    #[must_use]
    pub fn shape(prefix: &[G], tail: Option<&G>) -> SymbolicDfa<G> {
        let last = prefix.len();
        // The sink sits past the prefix states and the tail state.
        let sink = u32::try_from(last + 1).unwrap_or(0);
        let mut edges: Vec<Vec<Edge<G>>> = Vec::with_capacity(last + 2);
        for (position, guard) in prefix.iter().enumerate() {
            let next = u32::try_from(position + 1).unwrap_or(sink);
            edges.push(partition(guard, next, sink));
        }
        match tail {
            // The tail state loops on itself, which is the cycle that makes the
            // language infinite without anything recursive in the guard.
            Some(guard) => edges.push(partition(guard, u32::try_from(last).unwrap_or(0), sink)),
            None => edges.push(vec![Edge {
                guard: None,
                target: sink,
            }]),
        }
        edges.push(vec![Edge {
            guard: None,
            target: sink,
        }]);
        let mut accepting = vec![false; last + 2];
        if let Some(flag) = accepting.get_mut(last) {
            *flag = true;
        }
        SymbolicDfa { edges, accepting }.minimal()
    }

    pub(crate) fn state_count(&self) -> usize {
        self.accepting.len()
    }

    fn accepts(&self, state: u32) -> bool {
        self.accepting.get(state as usize).copied().unwrap_or(false)
    }

    pub(crate) fn outgoing(&self, state: u32) -> &[Edge<G>] {
        self.edges.get(state as usize).map_or(&[], Vec::as_slice)
    }

    /// Whether this language holds no sequence.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        let mut pending: VecDeque<u32> = VecDeque::from([0]);
        while let Some(state) = pending.pop_front() {
            if !seen.insert(state) {
                continue;
            }
            if self.accepts(state) {
                return false;
            }
            let row = self.outgoing(state);
            // The else edge's own set is what the guarded edges leave, so it is
            // asked separately -- and where a guard refuses to answer, the edge
            // is followed. That direction is the safe one: it can only report a
            // language inhabited, never empty.
            let rest_is_dead = rest_of(row).as_ref().is_some_and(Guard::is_empty);
            for edge in row {
                let dead = match &edge.guard {
                    Some(_) => edge.is_dead(),
                    None => rest_is_dead,
                };
                if !dead {
                    pending.push_back(edge.target);
                }
            }
        }
        true
    }

    /// Whether this language holds the sequence `values`.
    #[must_use]
    pub fn holds(&self, values: &[G::Value]) -> bool {
        let mut state = 0;
        for value in values {
            // The guarded edges are disjoint, so at most one takes the value;
            // the else edge takes it when none does, which is why the order of
            // this scan is the invariant rather than an accident.
            let Some(edge) = self
                .outgoing(state)
                .iter()
                .find(|edge| edge.guard.is_some() && edge.takes(value))
                .or_else(|| {
                    self.outgoing(state)
                        .iter()
                        .find(|edge| edge.guard.is_none())
                })
            else {
                // The guards cover the universe, so this cannot happen; folding
                // to a non-member keeps a broken table from being read as an
                // accept.
                debug_assert!(false, "the guards leaving state {state} do not cover");
                return false;
            };
            state = edge.target;
        }
        self.accepts(state)
    }

    /// Every sequence this language does not hold.
    ///
    /// A flip of the accepting states, which is sound only because the edges
    /// leaving each state *cover* the universe: a machine that could fall off
    /// the table rejects by falling, and flipping its flags would accept those
    /// sequences instead of the ones it meant.
    #[must_use]
    pub fn complement(&self) -> SymbolicDfa<G> {
        SymbolicDfa {
            edges: self.edges.clone(),
            accepting: self.accepting.iter().map(|a| !a).collect(),
        }
        .minimal()
    }

    /// The sequences in either language, or `None` past [`MAX_STATES`] or
    /// [`MAX_ROW`], or where a guard operation refuses.
    #[must_use]
    pub fn union(&self, other: &SymbolicDfa<G>) -> Option<SymbolicDfa<G>> {
        self.product(other, |a, b| a || b)
    }

    /// The sequences in both languages.
    #[must_use]
    pub fn intersect(&self, other: &SymbolicDfa<G>) -> Option<SymbolicDfa<G>> {
        self.product(other, |a, b| a && b)
    }

    /// The product, accepting where `accept` says so.
    ///
    /// One walk over reachable state *pairs*. The guards of a pair are the
    /// pairwise meets of the two sides' guards: each side's edges are disjoint
    /// and covering, so the meets are too, and the product needs no minterm
    /// search -- which is what a set of guards with no such invariant would.
    fn product(
        &self,
        other: &SymbolicDfa<G>,
        accept: impl Fn(bool, bool) -> bool,
    ) -> Option<SymbolicDfa<G>> {
        let mut ids: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut pending: VecDeque<(u32, u32)> = VecDeque::from([(0, 0)]);
        ids.insert((0, 0), 0);
        let mut edges: Vec<Vec<Edge<G>>> = Vec::new();
        let mut accepting: Vec<bool> = Vec::new();
        while let Some((mine, theirs)) = pending.pop_front() {
            accepting.push(accept(self.accepts(mine), other.accepts(theirs)));
            let (ours, yours) = (self.outgoing(mine), other.outgoing(theirs));
            // What each side's else edge takes, which is what its guarded edges
            // leave. Needed to meet an else edge with a guarded one, and the one
            // place the letters are asked to complement.
            let (our_rest, your_rest) = (rest_of(ours)?, rest_of(yours)?);
            // The product's else edge takes what *both* sides leave, so one side
            // leaving nothing is enough to make it dead. Asked of the two rests
            // already in hand rather than of their meet: a meet here would be a
            // guard operation inside the operation that builds guards, and it
            // does not descend.
            let rest_is_empty = our_rest.is_empty() || your_rest.is_empty();
            let mut row: Vec<Edge<G>> = Vec::new();
            for ours in ours {
                for yours in yours {
                    // A pair of edges meets on the values both take. The else
                    // edge's set is the rest, so each of the four combinations
                    // is one meet -- and the else-with-else pair is the
                    // product's own else edge, which needs no guard at all.
                    let guard = match (&ours.guard, &yours.guard) {
                        (Some(a), Some(b)) => Some(a.meet(b)?),
                        (Some(a), None) => Some(a.meet(&your_rest)?),
                        (None, Some(b)) => Some(our_rest.meet(b)?),
                        (None, None) => None,
                    };
                    if guard.as_ref().is_some_and(Guard::is_empty) {
                        continue;
                    }
                    // The product's own else edge is the pair of else edges. Where
                    // it is dead the row is spelled without it, and the last
                    // guarded edge becomes the else edge below -- the remaining
                    // guards cover everything between them.
                    if guard.is_none() && rest_is_empty {
                        continue;
                    }
                    let pair = (ours.target, yours.target);
                    let target = if let Some(id) = ids.get(&pair) {
                        *id
                    } else {
                        if ids.len() >= MAX_STATES {
                            return None;
                        }
                        let id = u32::try_from(ids.len()).ok()?;
                        ids.insert(pair, id);
                        pending.push_back(pair);
                        id
                    };
                    if row.len() >= MAX_ROW {
                        return None;
                    }
                    row.push(Edge { guard, target });
                }
            }
            if rest_is_empty {
                // Every value is taken by a guarded edge, so the largest of them
                // is exactly what the others leave: naming it the else edge
                // spells the row the one canonical way.
                row.sort_by(|a, b| a.guard.cmp(&b.guard).then(a.target.cmp(&b.target)));
                if let Some(last) = row.last_mut() {
                    last.guard = None;
                }
            }
            edges.push(row);
        }
        Some(SymbolicDfa { edges, accepting }.merge_targets().minimal())
    }

    /// Join the guards of edges that share a target, so a state's edge list has
    /// one entry per target.
    ///
    /// Two edges to one state are two ways to say one transition, and a form
    /// that kept them apart would make two equal languages unequal. A guard
    /// operation that refuses leaves them apart, which costs canonicity and
    /// nothing else -- so this returns the table either way rather than the
    /// whole construction failing.
    fn merge_targets(self) -> SymbolicDfa<G> {
        let edges = self
            .edges
            .into_iter()
            .map(|row| {
                let mut merged: Vec<Edge<G>> = Vec::with_capacity(row.len());
                for edge in row {
                    match merged.iter_mut().find(|kept| kept.target == edge.target) {
                        // Either side being the else edge makes the pair one:
                        // the values the other edge took are values the rest
                        // now takes, since removing a guarded edge is exactly
                        // what widens the else edge.
                        Some(kept) if kept.guard.is_none() || edge.guard.is_none() => {
                            kept.guard = None;
                        }
                        Some(kept) => {
                            match (kept.guard.as_ref(), edge.guard.as_ref()) {
                                (Some(a), Some(b)) => match a.join(b) {
                                    Some(joined) => kept.guard = Some(joined),
                                    // Keep both rather than lose one: the
                                    // language is the same, only the form is
                                    // coarser.
                                    None => merged.push(edge),
                                },
                                _ => merged.push(edge),
                            }
                        }
                        None => merged.push(edge),
                    }
                }
                merged
            })
            .collect();
        SymbolicDfa {
            edges,
            accepting: self.accepting,
        }
    }

    /// The minimal automaton for the same language, canonically numbered.
    ///
    /// The same three passes [`regular`](super::regular) uses, with the middle
    /// one replaced: there is no alphabet to coarsen, so the edges of a state
    /// are put in guard order instead. Merge the states no sequence tells apart,
    /// order every edge list, and renumber by a walk that takes edges in that
    /// order.
    ///
    /// Total, and deliberately. The passes ask the guards to *join*, and a
    /// guard past its own bound refuses; a refusal there leaves two edges where
    /// one would do, which is a coarser form for the same language rather than
    /// a wrong one. A complement never needs to fail, so it does not.
    #[must_use]
    fn minimal(self) -> SymbolicDfa<G> {
        self.merge_equivalent().sorted().renumber()
    }

    /// Merge the states no sequence distinguishes.
    ///
    /// A state's *signature* is its accepting flag together with, for each block
    /// it can reach, the values that reach it. Two states with one signature
    /// have one transition function, so they merge; refining until the
    /// signatures stop changing is Moore's algorithm.
    ///
    /// The signature is only a faithful reading of the transition function
    /// because the edges are normalised first: one edge per target, guards in
    /// order. Without that, two states could send the same values to the same
    /// places through differently-split guards and read as different.
    fn merge_equivalent(self) -> SymbolicDfa<G> {
        let normalised = self.merge_targets().sorted();
        let count = normalised.state_count();
        // Every state in one block to begin with, which is the coarsest
        // partition -- and the right place to start, because the accepting flag
        // is part of a signature, so the first round separates the accepting
        // states anyway. Seeding with that split instead is work the loop
        // repeats, and the sweep said so: a seed that collapses to one block
        // changed no answer.
        let mut block: Vec<u32> = vec![0; count];
        // At most one round per state: each round either splits a block or is
        // the last, and a block cannot split more often than it has members.
        for _ in 0..=count {
            // Signatures are collected in state order and each gets the id of
            // its first appearance. That numbering is what puts the start
            // state's block at zero -- its signature is the first seen -- and
            // it makes the ids depend on the signatures rather than on how many
            // splits the round made. A list rather than a map, because a
            // signature holds guards, and a guard is ordered rather than
            // hashed.
            let mut seen: Vec<Signature<G>> = Vec::with_capacity(count);
            let mut next: Vec<u32> = Vec::with_capacity(count);
            for state in 0..count {
                let signature = normalised.signature(state, &block);
                let id = if let Some(id) = seen.iter().position(|kept| *kept == signature) {
                    id
                } else {
                    seen.push(signature);
                    seen.len() - 1
                };
                next.push(u32::try_from(id).unwrap_or(0));
            }
            if next == block {
                break;
            }
            block = next;
        }
        let blocks = block.iter().copied().max().map_or(1, |m| m as usize + 1);
        let mut edges: Vec<Vec<Edge<G>>> = vec![Vec::new(); blocks];
        let mut accepting = vec![false; blocks];
        for state in 0..count {
            let Some(id) = block.get(state).copied() else {
                continue;
            };
            if let Some(flag) = accepting.get_mut(id as usize) {
                *flag = normalised.accepts(u32::try_from(state).unwrap_or(0));
            }
            if let Some(slot) = edges.get_mut(id as usize) {
                *slot = normalised
                    .outgoing(u32::try_from(state).unwrap_or(0))
                    .iter()
                    .map(|edge| Edge {
                        guard: edge.guard.clone(),
                        target: block.get(edge.target as usize).copied().unwrap_or(0),
                    })
                    .collect();
            }
        }
        SymbolicDfa { edges, accepting }.merge_targets()
    }

    /// The blocks this state reaches, with the values that reach each, plus
    /// whether it accepts.
    fn signature(&self, state: usize, block: &[u32]) -> Signature<G> {
        let state = u32::try_from(state).unwrap_or(0);
        let mut reached: Vec<(u32, Option<G>)> = Vec::new();
        for edge in self.outgoing(state) {
            let target = block.get(edge.target as usize).copied().unwrap_or(0);
            let Some((_, kept)) = reached.iter_mut().find(|(seen, _)| *seen == target) else {
                reached.push((target, edge.guard.clone()));
                continue;
            };
            // Two edges to one block are one transition, so their guards join.
            // Either being the else edge makes the pair one, since the rest
            // absorbs what the other took.
            match (kept.as_ref(), edge.guard.as_ref()) {
                (Some(a), Some(b)) => match a.join(b) {
                    Some(joined) => *kept = Some(joined),
                    // A refusal keeps the two apart rather than collapsing them
                    // into the else. Collapsing would make this state's
                    // signature *coarser*, which merges states that send
                    // different values to the block -- a wrong merge, where
                    // keeping them apart only forgoes a right one.
                    None => reached.push((target, edge.guard.clone())),
                },
                _ => *kept = None,
            }
        }
        reached.sort();
        (self.accepts(state), reached)
    }

    /// Put every edge list in guard order and drop the edges no value can take.
    ///
    /// Both halves are canonicity. A guarded edge whose guard is empty is a
    /// transition no sequence uses -- `shape` makes one for a guard that admits
    /// nothing -- and keeping it would give two equal languages two tables.
    /// Dropping it leaves the cover intact, since the else edge takes whatever
    /// the guarded ones do not. The else edge itself is never dropped and sorts
    /// last, because `None` orders before `Some` and the list is reversed into
    /// place.
    fn sorted(mut self) -> SymbolicDfa<G> {
        for row in &mut self.edges {
            row.retain(|edge| !edge.is_dead());
            row.sort_by(|a, b| match (&a.guard, &b.guard) {
                // The else edge last, so a state's list reads as "these values
                // go here, and the rest go there".
                (None, Some(_)) => core::cmp::Ordering::Greater,
                (Some(_), None) => core::cmp::Ordering::Less,
                (a_guard, b_guard) => a_guard.cmp(b_guard).then(a.target.cmp(&b.target)),
            });
        }
        self
    }

    /// Renumber by a walk that takes edges in order, dropping unreachable states.
    fn renumber(&self) -> SymbolicDfa<G> {
        let mut ids: FxHashMap<u32, u32> = FxHashMap::default();
        let mut order: Vec<u32> = Vec::new();
        let mut pending: VecDeque<u32> = VecDeque::from([0]);
        ids.insert(0, 0);
        while let Some(state) = pending.pop_front() {
            order.push(state);
            for edge in self.outgoing(state) {
                let fresh = u32::try_from(ids.len()).unwrap_or(0);
                if let std::collections::hash_map::Entry::Vacant(slot) = ids.entry(edge.target) {
                    slot.insert(fresh);
                    pending.push_back(edge.target);
                }
            }
        }
        let edges = order
            .iter()
            .map(|state| {
                self.outgoing(*state)
                    .iter()
                    .map(|edge| Edge {
                        guard: edge.guard.clone(),
                        target: ids.get(&edge.target).copied().unwrap_or(0),
                    })
                    .collect()
            })
            .collect();
        let accepting = order.iter().map(|state| self.accepts(*state)).collect();
        SymbolicDfa { edges, accepting }
    }
}

/// The two edges a single guard induces: the values it holds go one way, and the
/// else edge takes the rest. Together they cover the universe, which is the
/// invariant every state's edge list carries -- and the else edge is why the
/// guard's complement is not needed to state it.
fn partition<G: Guard>(guard: &G, taken: u32, sink: u32) -> Vec<Edge<G>> {
    // A guard that leaves nothing leaves the else edge dead, and a row with a
    // dead else edge is the same transition table as one whose only edge *is*
    // the else edge. Spelling it the second way here is what keeps a loop
    // guarded by every value equal to a state that accepts everything --
    // otherwise one language carries two tables.
    //
    // Asked here rather than in [`SymbolicDfa::sorted`] because it is a question
    // *about the guard*, and a letter may be a descriptor whose own complement
    // rebuilds automata: asking it during minimisation would run the
    // minimisation again, without end.
    if guard.complement().is_empty() {
        return vec![Edge {
            guard: None,
            target: taken,
        }];
    }
    vec![
        Edge {
            guard: Some(guard.clone()),
            target: taken,
        },
        Edge {
            guard: None,
            target: sink,
        },
    ]
}

/// The values a state's else edge takes: everything its guarded edges do not.
///
/// `None` where a guard refuses to join or complement, which is the bounded
/// case: the answer is then unknown rather than empty, and a caller that was
/// asking whether the edge is dead has to assume it is not.
fn rest_of<G: Guard>(row: &[Edge<G>]) -> Option<G> {
    let mut taken = G::none();
    for edge in row {
        if let Some(guard) = &edge.guard {
            taken = taken.join(guard)?;
        }
    }
    Some(taken.complement())
}

#[cfg(test)]
mod tests {
    use super::{Edge, Guard, MAX_ROW, SymbolicDfa};
    use crate::descr::integers::IntSet;
    use proptest::prelude::*;

    /// Integer sets as guards, so the machine is exercised over a letter whose
    /// algebra is already held to its own laws.
    ///
    /// A "sequence" is then a list of integers, which is small enough to
    /// enumerate and rich enough to separate the languages below -- and it is
    /// the same algebra a sequence component will use one level up, where the
    /// letter is a whole descriptor.
    impl Guard for IntSet {
        type Value = i64;

        fn none() -> Self {
            IntSet::empty()
        }
        fn meet(&self, other: &Self) -> Option<Self> {
            Some(self.intersect(other))
        }
        fn join(&self, other: &Self) -> Option<Self> {
            Some(self.union(other))
        }
        fn complement(&self) -> Self {
            IntSet::complement(self)
        }
        fn is_empty(&self) -> bool {
            IntSet::is_empty(self)
        }
        fn holds(&self, value: &i64) -> bool {
            IntSet::holds(self, *value)
        }
    }

    /// The sequences a law is checked over: every list of up to three integers
    /// drawn from the four the guards below distinguish.
    fn universe() -> Vec<Vec<i64>> {
        let letters = [0i64, 1, 2, 3];
        let mut words = vec![Vec::new()];
        for _ in 0..3 {
            let mut longer = Vec::new();
            for word in &words {
                for letter in letters {
                    let mut next = word.clone();
                    next.push(letter);
                    longer.push(next);
                }
            }
            words.extend(longer);
        }
        words
    }

    fn agree_on_sequences(a: &SymbolicDfa<IntSet>, b: &SymbolicDfa<IntSet>) -> bool {
        universe().iter().all(|w| a.holds(w) == b.holds(w))
    }

    /// The guards the generator draws from: two overlapping sets and two
    /// disjoint ones, so a product has meets that are empty and meets that are
    /// not.
    fn guards() -> Vec<IntSet> {
        vec![
            IntSet::just(0),
            IntSet::just(1),
            IntSet::between(Some(0), Some(1)),
            IntSet::between(Some(2), None),
        ]
    }

    fn language() -> impl Strategy<Value = SymbolicDfa<IntSet>> {
        let guard =
            (0..guards().len()).prop_map(|i| guards().get(i).cloned().unwrap_or_else(IntSet::all));
        let leaf = prop_oneof![
            Just(SymbolicDfa::empty()),
            Just(SymbolicDfa::all()),
            // `list[T]`.
            guard
                .clone()
                .prop_map(|g| SymbolicDfa::shape(&[], Some(&g))),
            // `tuple[A]` and `tuple[A, B]`.
            guard.clone().prop_map(|g| SymbolicDfa::shape(&[g], None)),
            (guard.clone(), guard.clone()).prop_map(|(a, b)| SymbolicDfa::shape(&[a, b], None)),
            // `tuple[A, *tuple[B, ...]]`.
            (guard.clone(), guard).prop_map(|(a, b)| SymbolicDfa::shape(&[a], Some(&b))),
        ];
        leaf.prop_recursive(3, 12, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.union(&b).unwrap_or_else(SymbolicDfa::all)),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(SymbolicDfa::empty)),
                inner.prop_map(|a| a.complement()),
            ]
        })
    }

    proptest! {
        // Fewer cases than the default: every operation is an automaton product
        // over guards that are themselves sets, and shrinking a failure over the
        // default count outruns what a mutation sweep waits for.
        #![proptest_config(ProptestConfig {
            cases: 64,
            // A bounded shrink, so a broken invariant cannot turn a caught
            // mutation into a run that outlasts a sweep.
            max_shrink_time: 2_000,
            ..ProptestConfig::default()
        })]

        /// The Boolean algebra, checked against the sequences.
        ///
        /// Against the sequences rather than by equality of the tables: the
        /// canonical form is earned only where the guards can answer every meet,
        /// so a law held by `==` alone would be a claim about the minimisation.
        #[test]
        fn the_lattice_laws_hold_of_the_sequences(
            a in language(),
            b in language(),
            c in language(),
        ) {
            let joined = a.union(&b);
            prop_assert!(matches(joined.as_ref(), b.union(&a).as_ref()));
            let met = a.intersect(&b);
            prop_assert!(matches(met.as_ref(), b.intersect(&a).as_ref()));
            prop_assert!(matches(
                joined.as_ref().and_then(|ab| ab.union(&c)).as_ref(),
                b.union(&c).as_ref().and_then(|bc| a.union(bc)).as_ref()
            ));
            prop_assert!(matches(
                met.as_ref().and_then(|ab| ab.intersect(&c)).as_ref(),
                b.intersect(&c).as_ref().and_then(|bc| a.intersect(bc)).as_ref()
            ));
            if let Some(inner) = &met {
                prop_assert!(matches(a.union(inner).as_ref(), Some(&a)));
            }
            if let Some(inner) = &joined {
                prop_assert!(matches(a.intersect(inner).as_ref(), Some(&a)));
            }
        }

        /// The complement laws, and De Morgan both ways.
        #[test]
        fn the_complement_laws_hold_of_the_sequences(a in language(), b in language()) {
            prop_assert!(
                a.intersect(&a.complement())
                    .is_some_and(|met| met.is_empty())
            );
            prop_assert!(matches(
                a.union(&a.complement()).as_ref(),
                Some(&SymbolicDfa::all())
            ));
            prop_assert!(agree_on_sequences(&a.complement().complement(), &a));
            prop_assert!(matches(
                a.union(&b).map(|u| u.complement()).as_ref(),
                a.complement().intersect(&b.complement()).as_ref()
            ));
            prop_assert!(matches(
                a.intersect(&b).map(|m| m.complement()).as_ref(),
                a.complement().union(&b.complement()).as_ref()
            ));
        }

        /// An empty verdict is contradicted by no sequence, and a sequence in
        /// the language contradicts one.
        #[test]
        fn emptiness_agrees_with_the_sequences(a in language()) {
            if a.is_empty() {
                prop_assert!(universe().iter().all(|w| !a.holds(w)));
            } else if universe().iter().any(|w| a.holds(w)) {
                prop_assert!(!a.is_empty());
            }
        }

        /// The edges leaving every state cover the letters, and the guarded
        /// ones are disjoint.
        ///
        /// The invariant everything else rests on: determinism, completeness,
        /// and a complement that is a flip rather than a construction. Checked
        /// over the letters, since two guards are disjoint exactly when no value
        /// takes both edges -- and the else edge is what makes the cover total
        /// without a guard naming the universe.
        #[test]
        fn the_edges_leaving_a_state_cover_the_letters(a in language()) {
            for state in 0..a.state_count() {
                let row = a.outgoing(u32::try_from(state).unwrap_or(0));
                prop_assert_eq!(
                    row.iter().filter(|edge| edge.guard.is_none()).count(),
                    1,
                    "state {} has one else edge",
                    state
                );
                for letter in [0i64, 1, 2, 3, -1, 9] {
                    let guarded = row
                        .iter()
                        .filter(|edge| {
                            edge.guard.as_ref().is_some_and(|g| Guard::holds(g, &letter))
                        })
                        .count();
                    prop_assert!(guarded <= 1, "state {} letter {}", state, letter);
                }
            }
        }
    }

    /// Whether two optional languages both exist and hold the same sequences.
    fn matches(a: Option<&SymbolicDfa<IntSet>>, b: Option<&SymbolicDfa<IntSet>>) -> bool {
        match (a, b) {
            (Some(a), Some(b)) => agree_on_sequences(a, b),
            (None, None) => true,
            _ => false,
        }
    }

    /// The three spellings one constructor covers, and the recursion living in
    /// an edge rather than in a guard.
    #[test]
    fn one_constructor_covers_the_three_sequence_spellings() {
        let zero = IntSet::just(0);
        let one = IntSet::just(1);

        // `list[0]`: any number of zeros, the empty list included.
        let homogeneous = SymbolicDfa::shape(&[], Some(&zero));
        assert!(homogeneous.holds(&[]));
        assert!(homogeneous.holds(&[0, 0, 0]));
        assert!(!homogeneous.holds(&[0, 1]));

        // `tuple[0, 1]`: exactly two elements, positionally.
        let fixed = SymbolicDfa::shape(&[zero.clone(), one.clone()], None);
        assert!(fixed.holds(&[0, 1]));
        assert!(!fixed.holds(&[0]) && !fixed.holds(&[0, 1, 1]) && !fixed.holds(&[1, 0]));

        // `tuple[0, *tuple[1, ...]]`: one element, then any number.
        let prefixed = SymbolicDfa::shape(&[zero], Some(&one));
        assert!(prefixed.holds(&[0]) && prefixed.holds(&[0, 1, 1]));
        assert!(!prefixed.holds(&[]) && !prefixed.holds(&[1]) && !prefixed.holds(&[0, 0]));

        // The infinite language is two states, because the cycle is an edge:
        // nothing in a guard refers to the language it guards.
        assert_eq!(homogeneous.state_count(), 2);
    }

    /// A union of two fixed sequences is the sequence of their union, and the
    /// letters in neither stay out.
    ///
    /// The case that puts two *guarded* edges on one state leading to one block.
    /// Reading them as the else edge instead would say every letter reaches the
    /// accepting block -- `tuple[anything]` rather than `tuple[A | B]` -- which
    /// no law above separates, because both sides of a law carry the same
    /// reading.
    #[test]
    fn a_union_of_fixed_sequences_joins_their_element_sets() {
        let zero = IntSet::just(0);
        let one = IntSet::just(1);
        let joined = SymbolicDfa::shape(std::slice::from_ref(&zero), None)
            .union(&SymbolicDfa::shape(std::slice::from_ref(&one), None))
            .expect("a small union");

        assert!(joined.holds(&[0]) && joined.holds(&[1]));
        assert!(!joined.holds(&[2]), "a letter in neither is not admitted");
        assert!(!joined.holds(&[]) && !joined.holds(&[0, 1]));
        assert!(agree_on_sequences(
            &joined,
            &SymbolicDfa::shape(&[zero.union(&one)], None)
        ));
    }

    /// A guard that leaves nothing leaves no else edge to write down.
    ///
    /// The edges of a state partition the universe, and the else edge is the
    /// part the guards do not take. Where they take everything that part is
    /// empty, so a row spelling it out is the same transition table as one
    /// without it -- and two equal languages must be one table, or minimisation
    /// has not finished.
    #[test]
    fn a_guard_that_takes_everything_leaves_no_else_edge() {
        let looped = SymbolicDfa::shape(&[], Some(&IntSet::all()));
        assert_eq!(looped, SymbolicDfa::all(), "a loop on every value");

        // The prefix form too: one letter, any value, and nothing after it.
        let one = SymbolicDfa::shape(&[IntSet::all()], None);
        assert!(one.holds(&[0]) && !one.holds(&[]) && !one.holds(&[0, 0]));
    }

    /// A product row too wide to hold refuses rather than being built.
    ///
    /// [`MAX_STATES`] bounds the states and [`MAX_ROW`] bounds the rows, which
    /// is the other dimension a product multiplies. The constructors keep rows
    /// narrow, so the two tables here are written directly: each is one state
    /// looping on many overlapping guards, and every pair of guards meets, which
    /// is the shape that makes a row quadratic.
    #[test]
    fn a_product_past_the_row_bound_refuses() {
        let wide = |count: i64| SymbolicDfa {
            edges: vec![
                (0..count)
                    .map(|n| Edge {
                        guard: Some(IntSet::between(Some(-n), None)),
                        target: 0,
                    })
                    .collect(),
            ],
            accepting: vec![true],
        };
        // A row of the product is the two rows multiplied, so a wide side and a
        // narrow one reach the bound between them.
        let long = i64::try_from(MAX_ROW).unwrap_or(i64::MAX) / 2 + 2;

        assert!(wide(long).intersect(&wide(3)).is_none());
        assert!(
            wide(2).intersect(&wide(2)).is_some(),
            "a narrow one still answers"
        );
    }

    /// Two states reaching one block through *separate* equivalent targets stay
    /// apart when the letters that reach it differ.
    ///
    /// The sharper form of the case below, and the one that needs three letters
    /// to reach: after the first letter, one state sends `0` and `1` onward and
    /// the other sends `0` and `2`, each through its own successor. Those
    /// successors are equivalent, so they land in one block -- and a reading
    /// that recorded only *that* a block is reached, rather than by which
    /// letters, would merge the two states and admit `[0, 2, 9]` and
    /// `[5, 1, 9]`, which no branch spells.
    #[test]
    fn two_states_reaching_one_block_through_separate_targets_stay_apart() {
        let triple = |a: i64, b: i64| {
            SymbolicDfa::shape(&[IntSet::just(a), IntSet::just(b), IntSet::just(9)], None)
        };
        let language = triple(0, 0)
            .union(&triple(0, 1))
            .and_then(|left| left.union(&triple(5, 0)))
            .and_then(|left| left.union(&triple(5, 2)))
            .expect("a small union");

        for word in [[0, 0, 9], [0, 1, 9], [5, 0, 9], [5, 2, 9]] {
            assert!(language.holds(&word), "{word:?} was written into the union");
        }
        for word in [[0, 2, 9], [5, 1, 9]] {
            assert!(!language.holds(&word), "{word:?} is in no branch");
        }
    }

    /// Two states that send *different* letters to one block stay apart.
    ///
    /// The case that separates joining a block's guards from collapsing them.
    /// After the first letter, one state accepts `0` or `1` and the other
    /// accepts `0` or `2`; both reach the accepting block by two guarded edges,
    /// so a reading that recorded "some values reach it" rather than *which*
    /// would merge them -- and the language would gain `[0, 2]` and `[1, 1]`,
    /// which no spelling put in it.
    #[test]
    fn two_states_reaching_one_block_by_different_letters_stay_apart() {
        let pair = |a: i64, b: i64| SymbolicDfa::shape(&[IntSet::just(a), IntSet::just(b)], None);
        let language = pair(0, 0)
            .union(&pair(0, 1))
            .and_then(|left| left.union(&pair(1, 0)))
            .and_then(|left| left.union(&pair(1, 2)))
            .expect("a small union");

        for word in [[0, 0], [0, 1], [1, 0], [1, 2]] {
            assert!(language.holds(&word), "{word:?} was written into the union");
        }
        for word in [[0, 2], [1, 1], [2, 0]] {
            assert!(!language.holds(&word), "{word:?} is in no branch");
        }
    }

    /// The three relations the structural procedure declines on sequences,
    /// decided here because the component is closed under complement.
    #[test]
    fn the_declined_sequence_relations_are_decided() {
        let ints = IntSet::between(Some(0), None);
        let strs = IntSet::between(None, Some(-1));
        let bools = IntSet::just(0);

        // A meet of two element-disjoint lists is the empty list alone, which
        // the structural procedure cannot see because it does not intersect a
        // container componentwise.
        let met = SymbolicDfa::shape(&[], Some(&ints))
            .intersect(&SymbolicDfa::shape(&[], Some(&strs)))
            .expect("a small meet");
        assert!(met.holds(&[]), "the empty list is in both");
        assert!(!met.holds(&[0]) && !met.holds(&[-1]));
        assert_eq!(
            met,
            SymbolicDfa::shape(&[], Some(&IntSet::empty())),
            "a list of nothing is the empty list alone"
        );

        // A fixed sequence is inside the complement of one that differs
        // positionally, which is `A <= ~B` -- and that is `A & B` being empty,
        // not `A & ~~B`.
        let left = SymbolicDfa::shape(&[ints.clone(), strs.clone()], None);
        let right = SymbolicDfa::shape(&[strs, ints.clone()], None);
        assert!(
            left.intersect(&right).is_some_and(|met| met.is_empty()),
            "the two orders share no sequence"
        );
        // The complement is what makes that a decision rather than a shape
        // rule: `left` really is inside `~right`, and `~right` is inhabited.
        assert!(!right.complement().is_empty());
        assert!(right.complement().holds(&[0, -1]));

        // And a meet with a complement is the componentwise difference:
        // `tuple[int] & ~tuple[bool]` is `tuple[int & ~bool]`.
        let difference = SymbolicDfa::shape(std::slice::from_ref(&ints), None)
            .intersect(&SymbolicDfa::shape(std::slice::from_ref(&bools), None).complement())
            .expect("a small meet");
        let narrowed = SymbolicDfa::shape(&[ints.intersect(&bools.complement())], None);
        assert!(agree_on_sequences(&difference, &narrowed));
    }
}
