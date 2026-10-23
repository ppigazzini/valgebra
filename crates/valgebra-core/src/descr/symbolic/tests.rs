use super::{Edge, Guard, MAX_EDGES, MAX_ROW, MAX_STATES, SymbolicDfa};
use crate::descr::automaton::refine;
use crate::descr::integers::IntSet;
use crate::descr::lattice_tests::Algebra;
use crate::verdict::Verdict;
use proptest::prelude::*;

/// A shape inside the state bound, as every one these tests build is but the
/// one past it.
fn shape<G: Guard>(prefix: &[G], tail: Option<&G>) -> SymbolicDfa<G> {
    SymbolicDfa::shape(prefix, tail).expect("a shape inside the state bound")
}

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
    // The two that may refuse, and here they do so for the reason the
    // trait allows: two periods can meet past the one this holds.
    fn meet(&self, other: &Self) -> Option<Self> {
        self.intersect(other)
    }
    fn join(&self, other: &Self) -> Option<Self> {
        self.union(other)
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

/// The sequence languages' operations, as the shared laws ask them, compared
/// against the sequences: the canonical form is earned only where the guards
/// can answer every meet, so a law held by `==` alone would be a claim about
/// the minimization.
const SEQUENCES: Algebra<SymbolicDfa<IntSet>> = Algebra {
    join: SymbolicDfa::union,
    meet: SymbolicDfa::intersect,
    complement: SymbolicDfa::complement,
    same: agree_on_sequences,
    holds_nothing: SymbolicDfa::is_empty,
    holds_everything: |a| agree_on_sequences(a, &SymbolicDfa::all()),
};

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
        guard.clone().prop_map(|g| shape(&[], Some(&g))),
        // `tuple[A]` and `tuple[A, B]`.
        guard.clone().prop_map(|g| shape(&[g], None)),
        (guard.clone(), guard.clone()).prop_map(|(a, b)| shape(&[a, b], None)),
        // `tuple[A, *tuple[B, ...]]`.
        (guard.clone(), guard).prop_map(|(a, b)| shape(&[a], Some(&b))),
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
    // A quarter of the default: every operation is an automaton product
    // over guards that are themselves sets, and shrinking a failure over the
    // default count outruns what a mutation sweep waits for. A fraction
    // rather than a count, so a deeper run reaches here too.
    #![proptest_config(ProptestConfig {
        cases: ProptestConfig::default().cases / 4,
        // A bounded shrink, so a broken invariant cannot turn a caught
        // mutation into a run that outlasts a sweep.
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    // THEORY: a-sequence-is-a-language, each-kind-is-closed
    /// The Boolean algebra, checked against the sequences.
    ///
    /// Against the sequences rather than by equality of the tables: the
    /// canonical form is earned only where the guards can answer every meet,
    /// so a law held by `==` alone would be a claim about the minimization.
    #[test]
    fn the_lattice_laws_hold_of_the_sequences(
        a in language(),
        b in language(),
        c in language(),
    ) {
        SEQUENCES.lattice_laws(&a, &b, &c)?;
        // The machine refuses alike whichever side comes first: a product
        // explores the same pairs of states in either order.
        prop_assert_eq!(a.union(&b).is_some(), b.union(&a).is_some());
        prop_assert_eq!(a.intersect(&b).is_some(), b.intersect(&a).is_some());
    }

    // THEORY: a-sequence-is-a-language
    /// The complement laws, and De Morgan both ways.
    #[test]
    fn the_complement_laws_hold_of_the_sequences(a in language(), b in language()) {
        SEQUENCES.complement_laws(&a, &b)?;
        // And a language met or joined with its own complement builds.
        prop_assert!(a.intersect(&a.complement()).is_some());
        prop_assert!(a.union(&a.complement()).is_some());
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

    // THEORY: a-sequence-is-a-language
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

    /// A shape whose prefix letters all hold a value is put in canonical form
    /// without the merge, and that form is the one the merge gives.
    ///
    /// The letters include the empty set and the whole one, so a prefix can
    /// hold a letter that leaves its position unable to reach acceptance --
    /// the case the shortcut must leave to the merge -- and a letter that
    /// spells its position with the else edge alone.
    #[test]
    fn a_chain_of_inhabited_letters_is_minimal_as_built(
        prefix in prop::collection::vec(letter(), 0..5),
        tail in prop::option::of(letter()),
    ) {
        let (table, minimal) = SymbolicDfa::chain(&prefix, tail.as_ref());
        prop_assert_eq!(
            minimal,
            prefix.iter().all(|guard| guard.emptiness() == Verdict::Inhabited)
        );
        prop_assert_eq!(shape(&prefix, tail.as_ref()), table.minimal());
    }

    // THEORY: one-language-one-table
    /// Complementing a minimal table leaves a minimal table.
    ///
    /// The property `SymbolicDfa::complement` relies on to skip a minimization
    /// pass: the flip leaves every edge where it was, so a second pass over the
    /// flipped table has nothing to merge, sort or renumber. The drawn table is
    /// minimal as built, which is the precondition.
    #[test]
    fn a_complement_of_a_sequence_language_is_already_minimal(a in language()) {
        prop_assert_eq!(a.clone().minimal(), a.clone());
        let flipped = a.complement();
        prop_assert_eq!(flipped.clone().minimal(), flipped);
    }

    /// A side that is the whole language or the empty one is answered by the
    /// algebra's identities, and the answer is the table the product builds.
    #[test]
    fn a_whole_or_empty_side_answers_as_the_product_does(a in language()) {
        let (all, empty) = (SymbolicDfa::all(), SymbolicDfa::empty());
        let both = |p: bool, q: bool| p && q;
        let either = |p: bool, q: bool| p || q;
        prop_assert_eq!(a.intersect(&all), a.product(&all, both));
        prop_assert_eq!(all.intersect(&a), all.product(&a, both));
        prop_assert_eq!(a.intersect(&empty), a.product(&empty, both));
        prop_assert_eq!(empty.intersect(&a), empty.product(&a, both));
        prop_assert_eq!(a.union(&empty), a.product(&empty, either));
        prop_assert_eq!(empty.union(&a), empty.product(&a, either));
        prop_assert_eq!(a.union(&all), a.product(&all, either));
        prop_assert_eq!(all.union(&a), all.product(&a, either));
    }

    // THEORY: one-language-one-table
    /// The merge's rounds started from the distances end at the partition the
    /// rounds started from one block reach.
    ///
    /// Over tables drawn state by state rather than built by the operations,
    /// which minimize what they build: a drawn table has states no sequence
    /// tells apart, cycles, and states that cannot accept, so the seed parts
    /// some states, leaves others for the rounds, and is wrong if it parts two
    /// the rounds would join.
    #[test]
    fn the_rounds_from_the_distances_end_where_the_rounds_from_one_block_do(
        drawn in table(),
    ) {
        let table = drawn.merge_targets().sorted();
        let count = table.state_count();
        let rounds = |start| refine(count, start, |state, block| table.signature(state, block));
        prop_assert_eq!(
            in_order_seen(&rounds(table.seed())),
            in_order_seen(&rounds(vec![0; count]))
        );
    }
}

/// A letter for a shape: one of the drawn guards, the empty set or every
/// integer.
fn letter() -> impl Strategy<Value = IntSet> {
    let mut letters = guards();
    letters.extend([IntSet::empty(), IntSet::all()]);
    prop::sample::select(letters)
}

/// A partition numbered afresh, each block by the first state in it, so two
/// numberings of one partition read alike.
fn in_order_seen(blocks: &[u32]) -> Vec<u32> {
    let mut seen: Vec<u32> = Vec::new();
    blocks
        .iter()
        .map(|block| {
            let at = seen
                .iter()
                .position(|first| first == block)
                .unwrap_or_else(|| {
                    seen.push(*block);
                    seen.len() - 1
                });
            u32::try_from(at).expect("a drawn table is small")
        })
        .collect()
}

/// A round signs only the states that share a block, since a block of one
/// cannot split, and a block that splits keeps its number for the part
/// holding its first member while each other part takes the next free one.
///
/// Five states, `0` and `3` alone and the rest in block one, whose signature
/// parts state `4` from `1` and `2`: the first round signs the three, the
/// second the two left together, and nothing else is ever signed.
#[test]
fn a_round_signs_only_the_states_that_share_a_block() {
    let signed = std::cell::RefCell::new(Vec::new());
    let blocks = refine(5, vec![0, 1, 1, 2, 1], |state, _| {
        signed.borrow_mut().push(state);
        state == 4
    });
    assert_eq!(blocks, [0, 1, 1, 2, 3]);
    assert_eq!(*signed.borrow(), [1, 2, 4, 1, 2]);
}

/// A table of up to eight states drawn row by row: each row sends `0`, `1`
/// and the integers from two up to a drawn state or leaves them to its else
/// edge, which goes to a drawn state too, so every row is complete and its
/// edges take disjoint values.
fn table() -> impl Strategy<Value = SymbolicDfa<IntSet>> {
    (1u32..=8).prop_flat_map(|states| {
        let target = 0..states;
        let row = (
            prop::option::of(target.clone()),
            prop::option::of(target.clone()),
            prop::option::of(target.clone()),
            target,
        )
            .prop_map(|(zero, one, two_up, rest)| {
                let letters = [
                    (IntSet::just(0), zero),
                    (IntSet::just(1), one),
                    (IntSet::between(Some(2), None), two_up),
                ];
                let mut row: Vec<Edge<IntSet>> = letters
                    .into_iter()
                    .filter_map(|(letter, to)| {
                        to.map(|target| Edge {
                            guard: Some(letter),
                            target,
                        })
                    })
                    .collect();
                row.push(Edge {
                    guard: None,
                    target: rest,
                });
                row
            });
        let states = states as usize;
        (
            prop::collection::vec(row, states),
            prop::collection::vec(any::<bool>(), states),
        )
            .prop_map(|(edges, accepting)| SymbolicDfa { edges, accepting })
    })
}

/// The three distances the merge starts from, on a table that has each case
/// one of them reads: a chain to an accepting state that can go no further,
/// a dead state, a cycle through an accepting state, and a state before both
/// the chain and the cycle.
///
/// The longest walk is what parts the states of an upper bound, which are all
/// accepting and all one element from refusing; the chain here is held to it
/// by states `0` to `2`. States `4` and `6` agree on all three and are not
/// equivalent, which leaves the rounds a block to cut, and the bound's own
/// table, where every state is apart before the first round, has a block per
/// state in the seed.
#[test]
fn the_merge_starts_from_three_distances() {
    let edge = |guard: Option<IntSet>, target| Edge { guard, target };
    let table = SymbolicDfa {
        edges: vec![
            vec![edge(None, 1)],
            vec![edge(None, 2)],
            vec![edge(None, 3)],
            vec![edge(None, 3)],
            vec![edge(None, 5)],
            vec![edge(None, 4)],
            vec![edge(Some(IntSet::just(0)), 2), edge(None, 4)],
        ],
        accepting: vec![false, false, true, false, false, true, false],
    };
    let never = u32::MAX;
    let to_accepting = table.distances(true);
    assert_eq!(to_accepting, [2, 1, 0, never, 1, 0, 1]);
    assert_eq!(table.distances(false), [0, 0, 1, 0, 0, 1, 0]);
    let live: Vec<bool> = to_accepting.iter().map(|d| *d != never).collect();
    assert_eq!(
        table.longest_live_walks(&live),
        [2, 1, 0, never, never, never, never]
    );
    assert_eq!(table.seed(), [0, 1, 2, 3, 4, 5, 4]);

    for n in [1usize, 4, 9] {
        let zeros = shape(&[], Some(&IntSet::just(0)));
        let at_least = shape(&vec![IntSet::just(0); n + 1], Some(&IntSet::just(0)));
        let at_most = zeros.intersect(&at_least.complement()).expect("a chain");
        let seed = at_most.seed();
        let mut blocks = seed.clone();
        blocks.sort_unstable();
        blocks.dedup();
        assert_eq!(blocks.len(), at_most.state_count(), "at most {n}: {seed:?}");
    }
}

/// A product past the build's allowance refuses, and the same product
/// succeeds under one that covers it.
///
/// Each state of the product is a pair of states, so the machine is where a
/// build multiplies hardest -- and the allowance is charged per new pair,
/// which is what makes a determinization stop while it is still cheap. Both
/// directions, because an allowance that only ever refuses would pass half
/// of this.
#[test]
fn a_product_past_the_allowance_refuses() {
    let ints = shape(&[IntSet::just(1), IntSet::just(2)], None);
    let others = shape(&[IntSet::just(1)], Some(&IntSet::all()));

    assert!(crate::descr::budget::under(1, || ints.intersect(&others)).is_none());
    assert!(crate::descr::budget::under(4096, || ints.intersect(&others)).is_some());
}

/// The three spellings one constructor covers, and the recursion living in
/// an edge rather than in a guard.
#[test]
fn one_constructor_covers_the_three_sequence_spellings() {
    let zero = IntSet::just(0);
    let one = IntSet::just(1);

    // `list[0]`: any number of zeros, the empty list included.
    let homogeneous = shape(&[], Some(&zero));
    assert!(homogeneous.holds(&[]));
    assert!(homogeneous.holds(&[0, 0, 0]));
    assert!(!homogeneous.holds(&[0, 1]));

    // `tuple[0, 1]`: exactly two elements, positionally.
    let fixed = shape(&[zero.clone(), one.clone()], None);
    assert!(fixed.holds(&[0, 1]));
    assert!(!fixed.holds(&[0]) && !fixed.holds(&[0, 1, 1]) && !fixed.holds(&[1, 0]));

    // `tuple[0, *tuple[1, ...]]`: one element, then any number.
    let prefixed = shape(&[zero], Some(&one));
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
    let joined = shape(std::slice::from_ref(&zero), None)
        .union(&shape(std::slice::from_ref(&one), None))
        .expect("a small union");

    assert!(joined.holds(&[0]) && joined.holds(&[1]));
    assert!(!joined.holds(&[2]), "a letter in neither is not admitted");
    assert!(!joined.holds(&[]) && !joined.holds(&[0, 1]));
    assert!(agree_on_sequences(
        &joined,
        &shape(&[zero.union(&one).expect("a period of one")], None)
    ));
}

/// A state reaching one block through two equivalent targets, by two letters,
/// merges with a state reaching that block by the two letters joined.
///
/// The case the signature's join is for. `merge_targets` leaves one edge per
/// *target*, so two edges into one block survive it only where the block
/// holds two distinct targets, which the rounds find equivalent. Read as the
/// else edge instead, the pair gives the first state a signature naming the
/// block with no letter where its partners name it with both, so the rounds
/// keep it apart and the table ends with three states where the language
/// takes two: a correct table, and not the canonical one that every equality
/// of two languages compares.
///
/// Written row by row because the operations minimize what they build, and a
/// product reaches this table only on some draws of a property.
#[test]
fn two_letters_into_one_block_join_in_the_signature() {
    let edge = |guard: Option<IntSet>, target| Edge { guard, target };
    let zero_or_one = IntSet::between(Some(0), Some(1));
    // `list[0 | 1]`, with the first letter split across two equal states.
    let table = SymbolicDfa {
        edges: vec![
            vec![
                edge(Some(IntSet::just(0)), 1),
                edge(Some(IntSet::just(1)), 2),
                edge(None, 3),
            ],
            vec![edge(Some(zero_or_one.clone()), 1), edge(None, 3)],
            vec![edge(Some(zero_or_one.clone()), 2), edge(None, 3)],
            vec![edge(None, 3)],
        ],
        accepting: vec![true, true, true, false],
    };
    let minimal = table.minimal();
    assert_eq!(minimal.state_count(), 2, "{minimal:?}");
    assert_eq!(minimal, shape(&[], Some(&zero_or_one)));
}

/// A guard that leaves nothing leaves no else edge to write down.
///
/// The edges of a state partition the universe, and the else edge is the
/// part the guards do not take. Where they take everything that part is
/// empty, so a row spelling it out is the same transition table as one
/// without it -- and two equal languages must be one table, or minimization
/// has not finished.
#[test]
fn a_guard_that_takes_everything_leaves_no_else_edge() {
    let looped = shape(&[], Some(&IntSet::all()));
    assert_eq!(looped, SymbolicDfa::all(), "a loop on every value");

    // The prefix form too: one letter, any value, and nothing after it.
    let one = shape(&[IntSet::all()], None);
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

/// A product whose whole table is too large refuses, and one that fills it
/// exactly does not.
///
/// [`MAX_ROW`] and [`MAX_STATES`] bound the two dimensions separately, and a
/// table inside both can still be far past what may be allocated, so
/// [`MAX_EDGES`] bounds the running total as the rows are built. The bound
/// is asserted from both sides: the largest table it admits is built, and
/// one row more is refused. Reading the total as anything but a sum, or the
/// comparison as anything but strict, moves one of the two answers.
#[test]
fn a_product_past_the_edge_bound_refuses() {
    // Each state leaves by this many overlapping guards, so every pair of
    // guards meets and a row of the product is the two rows multiplied.
    const WIDTH: usize = 16;
    const ROW: usize = WIDTH * WIDTH;
    // A chain of `states`, the last looping on itself. Written directly:
    // the constructors keep both dimensions far below the bound, and one
    // side of a product is what carries the table's height.
    let chain = |states: usize| SymbolicDfa {
        edges: (0..states)
            .map(|state| {
                let next =
                    u32::try_from(state + usize::from(state + 1 < states)).unwrap_or(u32::MAX);
                (0..WIDTH)
                    .map(|n| Edge {
                        guard: Some(IntSet::between(Some(-i64::try_from(n).unwrap_or(0)), None)),
                        target: next,
                    })
                    .collect()
            })
            .collect(),
        accepting: vec![true; states],
    };
    // The other side is one looping state, so the product has one state per
    // link of the chain and each row holds `ROW` edges.
    let held = chain(1);

    assert!(
        chain(MAX_EDGES / ROW).intersect(&held).is_some(),
        "a table filled to the bound is held"
    );
    assert!(
        chain(MAX_EDGES / ROW + 1).intersect(&held).is_none(),
        "one row past it is refused"
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
    let triple = |a: i64, b: i64| shape(&[IntSet::just(a), IntSet::just(b), IntSet::just(9)], None);
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
    let pair = |a: i64, b: i64| shape(&[IntSet::just(a), IntSet::just(b)], None);
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
    let met = shape(&[], Some(&ints))
        .intersect(&shape(&[], Some(&strs)))
        .expect("a small meet");
    assert!(met.holds(&[]), "the empty list is in both");
    assert!(!met.holds(&[0]) && !met.holds(&[-1]));
    assert_eq!(
        met,
        shape(&[], Some(&IntSet::empty())),
        "a list of nothing is the empty list alone"
    );

    // A fixed sequence is inside the complement of one that differs
    // positionally, which is `A <= ~B` -- and that is `A & B` being empty,
    // not `A & ~~B`.
    let left = shape(&[ints.clone(), strs.clone()], None);
    let right = shape(&[strs, ints.clone()], None);
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
    let difference = shape(std::slice::from_ref(&ints), None)
        .intersect(&shape(std::slice::from_ref(&bools), None).complement())
        .expect("a small meet");
    let narrowed = shape(
        &[ints
            .intersect(&bools.complement())
            .expect("a period of one")],
        None,
    );
    assert!(agree_on_sequences(&difference, &narrowed));
}

/// A product past the state bound refuses rather than answering about
/// another language.
///
/// The row bound above catches a product that is *wide*; this one catches a
/// product that is *long*, and the two are independent. Two cycles of
/// coprime lengths advance together for their product before they agree
/// again, so a pair well inside the bound apiece is past it together: the
/// language is "a length divisible by both", and no smaller machine reads
/// it.
#[test]
fn a_product_past_the_state_bound_refuses() {
    // Every letter steps once around a cycle of `n`, accepting where the
    // count is back at nought.
    let cycle = |n: u32| SymbolicDfa::<IntSet> {
        edges: (0..n)
            .map(|at| {
                vec![Edge {
                    guard: None,
                    target: (at + 1) % n,
                }]
            })
            .collect(),
        accepting: (0..n).map(|at| at == 0).collect(),
    };
    let past = u32::try_from(MAX_STATES).unwrap_or(u32::MAX);

    assert!(
        cycle(71).intersect(&cycle(73)).is_none(),
        "71 * 73 > {past}"
    );
    // A pair whose product is inside the bound answers, so the refusal
    // above is the count rather than the shape: both machines are cycles
    // either way.
    assert!(
        cycle(31).intersect(&cycle(37)).is_some(),
        "31 * 37 is inside it"
    );
}

/// A shape is a chain, held to the state bound before it is built.
///
/// With no tail every chain reaches its sink, so the table two letters short
/// of the bound holds the bound exactly, and one more letter refuses. Built,
/// the longer chain would be a table no product could take.
#[test]
fn a_shape_past_the_state_bound_refuses() {
    let letters = vec![IntSet::all(); MAX_STATES - 2];
    let edge = SymbolicDfa::shape(&letters, None).expect("a shape at the bound");
    assert_eq!(edge.state_count(), MAX_STATES);
    let longer = vec![IntSet::all(); MAX_STATES - 1];
    assert!(SymbolicDfa::shape(&longer, None).is_none());
}

/// A letter that can decline, which is the third answer a guard may give.
///
/// The machine's own emptiness is exact only where its letters are. One level
/// up a guard is a whole descriptor, and a descriptor carrying a class the core
/// cannot enumerate the subclasses of answers neither empty nor inhabited --
/// so the language behind such an edge is unproved rather than reachable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Opaque {
    /// A letter whose emptiness is a computation.
    Exact(IntSet),
    /// A letter that holds a value, or does not, and cannot say which.
    Undecided,
}

impl Guard for Opaque {
    type Value = i64;

    fn none() -> Self {
        Opaque::Exact(IntSet::empty())
    }

    fn meet(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Opaque::Exact(a), Opaque::Exact(b)) => a.intersect(b).map(Opaque::Exact),
            _ => Some(Opaque::Undecided),
        }
    }

    fn join(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Opaque::Exact(a), Opaque::Exact(b)) => a.union(b).map(Opaque::Exact),
            _ => Some(Opaque::Undecided),
        }
    }

    fn complement(&self) -> Self {
        match self {
            Opaque::Exact(set) => Opaque::Exact(IntSet::complement(set)),
            Opaque::Undecided => Opaque::Undecided,
        }
    }

    /// A *proof*, so an undecided letter answers `false`: it is not known empty.
    fn is_empty(&self) -> bool {
        matches!(self, Opaque::Exact(set) if IntSet::is_empty(set))
    }

    fn emptiness(&self) -> Verdict {
        match self {
            Opaque::Exact(set) if IntSet::is_empty(set) => Verdict::Empty,
            Opaque::Exact(_) => Verdict::Inhabited,
            Opaque::Undecided => Verdict::Unknown,
        }
    }

    fn holds(&self, value: &i64) -> bool {
        matches!(self, Opaque::Exact(set) if IntSet::holds(set, *value))
    }
}

// THEORY: a-sequence-is-a-language
/// An accepting state reachable only through a letter that cannot say whether
/// it holds a value leaves the language unproved, not inhabited.
///
/// Two walks decide it: the generous one follows every edge not proved dead, so
/// reaching nothing proves the language empty; the certain one follows only
/// edges proved to hold a value, so reaching an accepting state names a
/// sequence. A state the first reaches and the second does not is behind
/// exactly such a letter, and the answer is neither.
#[test]
fn an_accepting_state_behind_an_undecided_letter_is_unproved() {
    let sink = 2;
    let behind = |guard: Opaque| SymbolicDfa {
        edges: vec![
            vec![
                Edge {
                    guard: Some(guard),
                    target: 1,
                },
                Edge {
                    guard: None,
                    target: sink,
                },
            ],
            vec![Edge {
                guard: None,
                target: sink,
            }],
            vec![Edge {
                guard: None,
                target: sink,
            }],
        ],
        accepting: vec![false, true, false],
    };

    assert_eq!(
        behind(Opaque::Undecided).emptiness(),
        Verdict::Unknown,
        "a sequence exists only if the letter holds a value, and it cannot say"
    );
    assert_eq!(
        behind(Opaque::Exact(IntSet::all())).emptiness(),
        Verdict::Inhabited,
        "a letter that holds a value names the sequence"
    );
    assert_eq!(
        behind(Opaque::Exact(IntSet::empty())).emptiness(),
        Verdict::Empty,
        "a letter proved empty leads nowhere"
    );
    assert!(
        !behind(Opaque::Undecided).is_empty(),
        "unproved is not a proof of emptiness"
    );
}

/// A table with an edge to a state it does not have stops a debug build where
/// it is minimized, for the reason the byte automaton's test gives.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a table is minimized as it was built")]
fn a_table_with_an_edge_off_it_stops_a_debug_build() {
    let malformed: SymbolicDfa<IntSet> = SymbolicDfa {
        edges: vec![vec![Edge {
            guard: None,
            target: 5,
        }]],
        accepting: vec![true],
    };
    let _ = malformed.minimal();
}

/// Each clause of a well-formed table refuses a table that breaks it alone: an
/// edge to the state one past the last, a state with no edges, a row too few,
/// and no state at all. The table they are made from is well formed, with an
/// edge to its last state.
#[test]
fn each_defect_of_a_table_is_one_well_formed_refuses() {
    let edge = |target| Edge {
        guard: None::<IntSet>,
        target,
    };
    let good = || SymbolicDfa {
        edges: vec![vec![edge(1)], vec![edge(1)]],
        accepting: vec![false, true],
    };
    assert!(good().well_formed());
    let mut off_the_table = good();
    off_the_table.edges[0][0].target = 2;
    let mut an_empty_row = good();
    an_empty_row.edges[1].clear();
    let mut a_row_too_few = good();
    a_row_too_few.edges.pop();
    let no_state: SymbolicDfa<IntSet> = SymbolicDfa {
        edges: Vec::new(),
        accepting: Vec::new(),
    };
    for (defect, table) in [
        ("an edge off the table", off_the_table),
        ("an empty row", an_empty_row),
        ("a row too few", a_row_too_few),
        ("no state", no_state),
    ] {
        assert!(!table.well_formed(), "{defect}");
    }
}

/// A letter that cannot say whether it holds a value leaves a shape to the
/// merge, which is what keeps an unproved position from being read as one
/// that reaches acceptance.
#[test]
fn a_chain_with_an_undecided_letter_is_left_to_the_merge() {
    let held = Opaque::Exact(IntSet::just(1));
    for prefix in [
        vec![Opaque::Undecided],
        vec![held.clone(), Opaque::Undecided],
        vec![Opaque::Undecided, held.clone()],
    ] {
        let (table, minimal) = SymbolicDfa::chain(&prefix, Some(&held));
        assert!(!minimal, "{prefix:?}");
        assert_eq!(shape(&prefix, Some(&held)), table.minimal());
    }
    assert!(SymbolicDfa::chain(&[held.clone(), held.clone()], None).1);
}
