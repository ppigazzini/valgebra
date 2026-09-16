use super::{Entry, KEY_KINDS, Label, MAX_ATOMS, MapLattice, key_slot, unordered_pairs};
use crate::descr::budget;
use crate::descr::integers::IntSet;
use crate::kind::Kind;
use crate::verdict::Verdict;
use proptest::prelude::*;

/// One dict entry: a `str` key with this text, mapping to this integer.
fn at(label: &str, value: i64) -> Entry<i64> {
    Entry {
        label: Some(Label::str(label)),
        kind: Some(Kind::Str),
        value,
    }
}

/// A wanted key is witnessed by a *labelled* key too, not by the part's
/// default alone.
///
/// `S` asks for some key of a part, outside its exclusion set, to map into
/// a type. A label of that part is such a key, and reading only the default
/// reports an atom empty that [`MapAtom::holds`] admits dicts for -- which
/// makes a difference come out empty when it is not, and a subtype proof
/// out of nothing.
///
/// Both directions, because each is a different way to get the witness
/// scan wrong: taking labels of the wrong part, and taking labels the
/// exclusion set names.
#[test]
fn a_labelled_key_witnesses_a_wanted_key() {
    let words = |labels: Vec<(Label, IntSet, bool)>| {
        MapLattice::record(labels, [(Some(Kind::Str), IntSet::just(1))]).expect("a small record")
    };
    // `{a: 1, b: 2}` with every other string key mapping to 1, minus `{a: 1}`
    // with the same default. `b` maps to 2, which the default forbids, so the
    // difference holds that dict -- and `b` is the key that witnesses it.
    let wide = words(vec![
        (Label::str("a"), IntSet::just(1), false),
        (Label::str("b"), IntSet::just(2), false),
    ]);
    let narrow = words(vec![(Label::str("a"), IntSet::just(1), false)]);
    let outside = wide
        .intersect(&narrow.complement())
        .expect("a small difference");
    assert_eq!(outside.emptiness(), Verdict::Inhabited);
    assert!(outside.holds(&[at("a", 1), at("b", 2)]));

    // The same shape with nothing outside the exclusion set: `{a: 2}` minus
    // itself. `a` is named on both sides, so it cannot be the key that maps
    // outside the default, and the difference is empty.
    let same = words(vec![(Label::str("a"), IntSet::just(2), false)]);
    assert_eq!(
        same.intersect(&same.complement())
            .expect("a small difference")
            .emptiness(),
        Verdict::Empty
    );
}

/// Complementing twice gives the set back.
///
/// `¬A` names the ways to fail `A`, and one of them is "no key of this part,
/// other than the ones `A` names, maps anywhere". Negating *that* tightens
/// the part's default -- which governs every key the atom does not name, the
/// excluded ones included, so the record `{a: int}` came back forbidding the
/// key it is about and `¬¬A` was empty.
///
/// Checked against a value as well as against the verdict, because an atom
/// set that is merely spelled differently is fine and one that holds
/// different dicts is not.
#[test]
fn complementing_twice_gives_the_dicts_back() {
    let record = MapLattice::record(
        vec![(Label::str("a"), IntSet::just(1), false)],
        core::iter::empty(),
    )
    .expect("a record");
    let twice = record.complement().complement();

    assert_eq!(twice.emptiness(), Verdict::Inhabited);
    assert!(twice.holds(&[at("a", 1)]));
    assert!(
        !twice.holds(&[at("a", 2)]),
        "and no more than the record does"
    );
    assert!(!twice.holds(&[at("a", 1), at("b", 1)]));
    // And the meet with a mapping that admits the same dict is inhabited,
    // which is the relation the unsoundness was found through: `dict[str,
    // int]` was decided a subtype of `¬{a: int}`.
    let mapping = MapLattice::keyed(Kind::Str, IntSet::just(1));
    assert_eq!(
        mapping.intersect(&twice).expect("a meet").emptiness(),
        Verdict::Inhabited
    );
}

/// A meet past the build's allowance refuses, and the same meet succeeds
/// under one that covers it.
///
/// The atom product is a loop over every pair, so it is one of the places a
/// build multiplies and one of the places the allowance is charged. Both
/// directions are asserted: an allowance that only ever refuses would pass
/// half of this, and so would one that never does.
#[test]
fn a_meet_past_the_allowance_refuses() {
    let a = MapLattice::label(Label::str("a"), IntSet::just(1), false);
    let b = MapLattice::label(Label::str("b"), IntSet::just(2), false);

    assert!(budget::under(0, || a.intersect(&b)).is_none());
    assert!(budget::under(64, || a.intersect(&b)).is_some());
}

/// The union holds the dicts of both sides, and is a union rather than a
/// refusal.
///
/// Read at this level because the layer above absorbs a refusal: a kind's
/// lines merge two structures by joining them, and where the join declines
/// the two stay separate lines -- the same set, spelled longer. So a union
/// that never succeeded would still denote the right dicts, and only the
/// form would say so.
#[test]
fn a_union_of_maps_holds_the_dicts_of_both() {
    let a = MapLattice::label(Label::str("a"), IntSet::just(1), false);
    let b = MapLattice::label(Label::str("b"), IntSet::just(2), false);
    let joined = a.union(&b).expect("two maps join");
    assert_eq!(joined.emptiness(), Verdict::Inhabited);
    assert!(joined.holds(&[at("a", 1)]));
    assert!(joined.holds(&[at("b", 2)]));
    assert!(!joined.holds(&[at("a", 2)]));
    assert!(!joined.holds(&[]));
}

/// A meet is (12): the labels are shared and each side reads one it does not
/// name off its own default.
#[test]
fn a_meet_of_maps_holds_only_the_dicts_of_both() {
    let a = MapLattice::label(Label::str("a"), IntSet::just(1), false);
    let b = MapLattice::label(Label::str("b"), IntSet::just(2), false);
    let met = a.intersect(&b).expect("two maps meet");
    assert!(met.holds(&[at("a", 1), at("b", 2)]));
    assert!(!met.holds(&[at("a", 1)]));
    // Two maps that disagree about one label share no dict at all.
    let other = MapLattice::label(Label::str("a"), IntSet::just(2), false);
    let disagreeing = a.intersect(&other).expect("two maps meet");
    assert_eq!(disagreeing.emptiness(), Verdict::Empty);
}

/// A complement is a map, and a dict is outside an atom by a missing key or
/// a wrong one.
#[test]
fn a_complement_of_a_map_is_a_map() {
    let a = MapLattice::label(Label::str("a"), IntSet::just(1), false);
    let outside = a.complement();
    assert!(!outside.holds(&[at("a", 1)]));
    assert!(outside.holds(&[at("a", 2)]));
    assert!(outside.holds(&[]));
    // And complementing twice comes back to the dicts it started with.
    let back = outside.complement();
    assert!(back.holds(&[at("a", 1)]));
    assert!(!back.holds(&[at("a", 2)]));
}

/// A constraint about one part narrows the labels of that part alone.
///
/// "No key outside `besides` of *this* part maps into `ty`" says nothing
/// about a label of another part, and narrowing one would make the
/// complement hold too few dicts. Every label here is a `str`, so a
/// constraint about the integer keys must leave them alone.
#[test]
fn a_constraint_about_one_part_leaves_the_other_parts_labels_alone() {
    // "some integer key maps outside {1}", met with "b maps to 2".
    let int_keys = MapLattice::keyed(Kind::Int, IntSet::just(1));
    let some_other_int = int_keys.complement();
    let b_is_two = MapLattice::label(Label::str("b"), IntSet::just(2), false);
    let met = some_other_int.intersect(&b_is_two).expect("the two meet");
    // Its complement holds a dict whose `b` is 2 and whose integer keys are
    // all 1 -- the label is untouched by a constraint about another part.
    let outside = met.complement();
    assert!(outside.holds(&[at("b", 2)]));
}

/// A constraint about the labels' own part narrows them, and the complement
/// is wrong without it.
///
/// "No `str` key outside `besides` maps into `ty`" constrains the default for
/// that part *and* every label the exclusion set does not cover -- a label is
/// a `str` key too. Leaving the labels at their top would let the complement
/// hold a dict the atom itself holds.
#[test]
fn a_constraint_about_the_labels_own_part_narrows_them() {
    // "some str key maps outside {1}", met with "b maps to 2".
    let str_keys = MapLattice::keyed(Kind::Str, IntSet::just(1));
    let some_other_str = str_keys.complement();
    let b_is_two = MapLattice::label(Label::str("b"), IntSet::just(2), false);
    let met = some_other_str.intersect(&b_is_two).expect("the two meet");
    // `{"b": 2}` is in the meet: `b` maps to 2, and `b` is itself the str key
    // that maps outside {1}.
    assert!(met.holds(&[at("b", 2)]));
    // So it must be outside the complement -- which it is only if the
    // constraint narrowed `b` along with the default.
    assert!(!met.complement().holds(&[at("b", 2)]));
}

/// The key partition covers every key, the kindless one included.
///
/// A part missing from the array would leave the keys of that kind governed
/// by nothing, and an atom would admit a dict it says nothing about.
#[test]
fn every_key_falls_in_exactly_one_part() {
    for kind in KEY_KINDS {
        assert!(key_slot(Some(kind)).is_some(), "{kind:?}");
    }
    // A key of no listed kind has the last part rather than none.
    assert_eq!(key_slot(None), Some(KEY_KINDS.len()));
    // An unhashable kind is no key at all.
    for kind in [Kind::List, Kind::Set, Kind::Dict] {
        assert_eq!(key_slot(Some(kind)), None, "{kind:?}");
    }
}

/// A union past the atom bound refuses rather than holding a form it cannot.
///
/// The count is what a meet multiplies and a difference adds to, so a union
/// that has already reached the bound has no sound wider form to return --
/// and returning a narrower one would decide a difference by leaving values
/// out.
#[test]
fn a_union_past_the_bound_refuses() {
    let entry = |n: i64| MapLattice::label(Label::str("x"), IntSet::just(n), false);
    let mut wide = entry(0);
    for n in 1..i64::try_from(MAX_ATOMS).unwrap_or(i64::MAX) {
        wide = wide.union(&entry(n)).expect("inside the bound");
    }
    assert!(wide.union(&entry(-1)).is_none());
}

/// A dict has one entry for `1` and for `True`, so an atom requiring both
/// keys holds no dict.
///
/// The two are separate labels and stay separate, because `Literal[1]` and
/// `Literal[True]` are disjoint *sets* -- a key is an `int` or a `bool` and
/// the walk tells them apart. What they are not is two entries: `True` hashes
/// as `1` and equals it, so `{1: "a", True: "b"}` is a dict of one key. An
/// atom that requires both describes a dict Python cannot build, and reading
/// it as inhabited is a complement wrongly wide -- which is the only way to
/// reach it, since nothing a caller writes directly requires two keys at once.
///
/// `0` and `False` are the other pair, and `1` and `False` are not one.
#[test]
fn an_atom_requiring_a_key_and_its_boolean_holds_no_dict() {
    let required = |label: Label| MapLattice::label(label, IntSet::just(1), false);
    let both = |a: Label, b: Label| {
        required(a)
            .intersect(&required(b))
            .expect("two single-label atoms meet")
            .emptiness()
    };

    assert_eq!(
        both(Label::Int(1), Label::Bool(true)),
        Verdict::Empty,
        "no dict carries a key 1 and a key True"
    );
    assert_eq!(
        both(Label::Int(0), Label::Bool(false)),
        Verdict::Empty,
        "nor a key 0 and a key False"
    );
    assert_eq!(
        both(Label::Int(1), Label::Bool(false)),
        Verdict::Inhabited,
        "1 and False are two keys, and a dict carries both"
    );
    assert_eq!(
        both(Label::Int(1), Label::Int(2)),
        Verdict::Inhabited,
        "and two integers are always two keys"
    );
}

/// An optional key is not a required one, so the pair above is only empty
/// where the atom asks for both.
#[test]
fn an_optional_boolean_key_leaves_the_integer_key_alone() {
    let one = MapLattice::label(Label::Int(1), IntSet::just(1), false);
    let maybe_true = MapLattice::label(Label::Bool(true), IntSet::just(1), true);

    assert_eq!(
        one.intersect(&maybe_true)
            .expect("two atoms meet")
            .emptiness(),
        Verdict::Inhabited,
        "a dict of the integer key alone lets the boolean one go missing"
    );
}

/// Every unordered pair of *distinct* items, each once.
///
/// The contract the name states, and the one the caller rests on: the scan for
/// two labels that are one dict key asks the question of two labels, and a pair
/// of a label with itself is not that question. The helper is small enough to
/// read and small enough to get wrong -- an off-by-one in the slice it takes
/// yields each item beside itself, which no dict distinguishes because the
/// collision test is false on a self-pair, and which the next caller would
/// inherit.
#[test]
fn unordered_pairs_yields_each_distinct_pair_once() {
    let items = ["a", "b", "c"];
    let pairs: Vec<(&&str, &&str)> = unordered_pairs(&items).collect();
    assert_eq!(pairs.len(), 3, "three items make three pairs");
    assert_eq!(
        pairs,
        vec![(&"a", &"b"), (&"a", &"c"), (&"b", &"c")],
        "each pair once, in order, and never an item with itself"
    );
    for (left, right) in &pairs {
        assert_ne!(left, right, "a pair is of two distinct items");
    }
    // The degenerate sizes, where an off-by-one shows first.
    assert_eq!(unordered_pairs(&items[..1]).count(), 0, "one item, no pair");
    assert_eq!(unordered_pairs::<&str>(&[]).count(), 0, "no items, no pair");
    assert_eq!(unordered_pairs(&items[..2]).count(), 1);
}

// --- The lattice laws, over the dicts --------------------------------------
//
// Every other component of the descriptor holds these as a property; this one
// held its operations to hand-written pairs alone, which is the shape that
// confirms the cases somebody thought of. A dict is the kind with the most
// structure -- labelled keys, a default per part of the key partition, and the
// wanted-key set (13) adds -- so it is the one where a law is most likely to
// part from the form, and the `emptiness` arm the sweep accepts as equivalent
// is inside it.
//
// The laws are read against the *dicts* rather than by comparing forms: a union
// of atoms is not canonical, so two lattices holding the same dicts need not be
// equal, and asserting equality would hold the representation to more than it
// promises.

/// The dicts a law is checked over.
///
/// Every combination of the two `str` labels the generator names, with and
/// without a key of another part, plus the empty dict -- which is the value a
/// record of optional fields turns on and the one an exclusion set is most
/// often wrong about.
fn dicts() -> Vec<Vec<Entry<i64>>> {
    let other = |value: i64| Entry {
        label: None,
        kind: Some(Kind::Int),
        value,
    };
    vec![
        vec![],
        vec![at("a", 1)],
        vec![at("a", 2)],
        vec![at("b", 1)],
        vec![at("a", 1), at("b", 1)],
        vec![at("a", 2), at("b", 1)],
        vec![other(1)],
        vec![at("a", 1), other(1)],
    ]
}

fn same(a: &MapLattice<IntSet>, b: &MapLattice<IntSet>) -> bool {
    dicts().iter().all(|d| a.holds(d) == b.holds(d))
}

/// Lattices over the integer sets whose own laws are already held.
fn lattice() -> impl Strategy<Value = MapLattice<IntSet>> {
    let leaf = prop_oneof![
        Just(MapLattice::empty()),
        Just(MapLattice::all()),
        (1i64..=2).prop_map(|n| MapLattice::label(Label::str("a"), IntSet::just(n), false)),
        (1i64..=2).prop_map(|n| MapLattice::label(Label::str("a"), IntSet::just(n), true)),
        (1i64..=2).prop_map(|n| MapLattice::label(Label::str("b"), IntSet::just(n), false)),
        Just(MapLattice::keyed(Kind::Str, IntSet::just(1))),
        Just(MapLattice::keyed(Kind::Int, IntSet::all())),
        Just(MapLattice::keys_among(&[Some(Kind::Str)])),
    ];
    leaf.prop_recursive(3, 12, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.union(&b).unwrap_or_else(MapLattice::all)),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| a.intersect(&b).unwrap_or_else(MapLattice::empty)),
            inner.prop_map(|a| a.complement()),
        ]
    })
}

proptest! {
    // A bounded shrink, so a broken invariant cannot turn a caught mutation
    // into a run that outlasts a sweep.
    #![proptest_config(ProptestConfig {
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    /// The Boolean algebra, checked against the dicts.
    #[test]
    fn the_lattice_laws_hold_of_the_dicts(
        a in lattice(),
        b in lattice(),
        c in lattice(),
    ) {
        if let (Some(ab), Some(ba)) = (a.union(&b), b.union(&a)) {
            prop_assert!(same(&ab, &ba), "join commutes");
        }
        if let (Some(ab), Some(ba)) = (a.intersect(&b), b.intersect(&a)) {
            prop_assert!(same(&ab, &ba), "meet commutes");
        }
        if let (Some(bc), Some(ab)) = (b.union(&c), a.union(&b))
            && let (Some(left), Some(right)) = (a.union(&bc), ab.union(&c))
        {
            prop_assert!(same(&left, &right), "join associates");
        }
        if let (Some(bc), Some(ab)) = (b.intersect(&c), a.intersect(&b))
            && let (Some(left), Some(right)) = (a.intersect(&bc), ab.intersect(&c))
        {
            prop_assert!(same(&left, &right), "meet associates");
        }
        if let Some(met) = a.intersect(&b)
            && let Some(absorbed) = a.union(&met)
        {
            prop_assert!(same(&absorbed, &a), "join absorbs the meet");
        }
        if let (Some(bc), Some(ac)) = (b.union(&c), a.intersect(&c))
            && let (Some(ab), Some(left)) = (a.intersect(&b), a.intersect(&bc))
            && let Some(right) = ab.union(&ac)
        {
            prop_assert!(same(&left, &right), "meet distributes over join");
        }
    }

    /// The complement laws, and De Morgan both ways.
    #[test]
    fn the_complement_laws_hold_of_the_dicts(a in lattice(), b in lattice()) {
        let not_a = a.complement();
        if let Some(met) = a.intersect(&not_a) {
            prop_assert!(
                dicts().iter().all(|d| !met.holds(d)),
                "a dict is in one of the two"
            );
        }
        if let Some(joined) = a.union(&not_a) {
            prop_assert!(same(&joined, &MapLattice::all()), "and in one of them");
        }
        prop_assert!(same(&not_a.complement(), &a), "twice is nothing");
        let not_b = b.complement();
        if let (Some(joined), Some(met)) = (a.union(&b), not_a.intersect(&not_b)) {
            prop_assert!(same(&joined.complement(), &met), "de Morgan one way");
        }
        if let (Some(met), Some(joined)) = (a.intersect(&b), not_a.union(&not_b)) {
            prop_assert!(same(&met.complement(), &joined), "and the other");
        }
    }

    /// Emptiness is a claim about the dicts: a lattice proved empty holds
    /// none, and one proved inhabited is contradicted by no dict either.
    ///
    /// The third verdict is the point. `Unknown` is neither claim, and reading
    /// it as "inhabited" is what a two-valued answer would do -- so the law
    /// asserts only what each of the two proofs says, and says nothing for the
    /// decline.
    #[test]
    fn emptiness_agrees_with_the_dicts(a in lattice()) {
        match a.emptiness() {
            Verdict::Empty => prop_assert!(
                dicts().iter().all(|d| !a.holds(d)),
                "an empty lattice holds no dict"
            ),
            Verdict::Inhabited | Verdict::Unknown => {}
        }
    }
}
