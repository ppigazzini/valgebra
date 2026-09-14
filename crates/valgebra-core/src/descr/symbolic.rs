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

use super::budget;
use crate::verdict::Verdict;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// The most states an automaton may hold.
///
/// For the reason [`regular`](super::regular) gives: a product multiplies the
/// state counts, so a bound is what keeps a pathological combination from
/// exhausting memory rather than answering.
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

/// The most edges a product's whole table may hold.
///
/// [`MAX_STATES`] and [`MAX_ROW`] bound the two dimensions *separately*, and a
/// table can sit inside both while being far too large to hold: four thousand
/// states each carrying a four-thousand-wide row is sixteen million edges, which
/// is gigabytes. Neither bound sees that, because neither is exceeded -- so
/// `Annotated[str, Regex("(a|b)*a(a|b){20}")]` against `Regex("(a|b)*")` spent
/// six seconds and 668 MB answering, and two more repetitions aborted the
/// process on a four-gigabyte allocation. A subtype question is not allowed to
/// do that.
///
/// This bounds the product of the two, which is where the memory is. Sixty-four
/// thousand edges is a table of a few megabytes, past anything an annotation
/// writes -- a pattern in a contract distinguishes tens of positions, not
/// thousands -- and reaching it refuses, which every caller of a descriptor
/// operation already handles.
pub const MAX_EDGES: usize = 1 << 16;

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
///
/// Being *total* is what the machine needs of it, and agreeing with `Eq` is not
/// required: a letter whose representation is not canonical -- the integer set
/// is one, where a period a set does not need is a second spelling -- has to
/// choose, because an order read off a pair of spellings is not an order at
/// all. A letter that orders two equal sets apart keeps an edge a merge would
/// have folded, which is a larger table and never a different language.
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

    /// The sequences in either language, or `None` past [`MAX_STATES`],
    /// [`MAX_ROW`] or [`MAX_EDGES`], or where a guard operation refuses.
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
    ///
    /// That disjointness is a *precondition*, not something asserted here, and
    /// the reason is in this module's own tests: the two that drive the size
    /// bounds build tables by hand whose guards deliberately overlap, so every
    /// pair meets and a row is the two rows multiplied. A `debug_assert` on the
    /// row was written and removed -- it cannot tell a fixture reaching for a
    /// bound from a machine that lost the invariant, and no placement avoids
    /// them, since they call the public operations. What holds it instead is
    /// `the_complement_laws_hold_of_the_sequences`: a complement flips accepting
    /// states, so two guards sharing a value flip it twice and the law fails.
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
        let mut held = 0usize;
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
                        if ids.len() >= MAX_STATES || !budget::spend() {
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
                // spells the row one way. One way up to the guards' own order,
                // which reads a spelling where a guard has more than one -- see
                // the `Guard` doc above.
                row.sort_by(|a, b| a.guard.cmp(&b.guard).then(a.target.cmp(&b.target)));
                if let Some(last) = row.last_mut() {
                    last.guard = None;
                }
            }
            // The two dimensions are bounded above; this bounds their product,
            // which is the quantity that is actually allocated.
            held += row.len();
            if held > MAX_EDGES {
                return None;
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
mod tests;
