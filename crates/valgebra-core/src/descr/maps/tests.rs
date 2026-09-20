use std::collections::BTreeSet;

use super::{
    Entry, KEY_KINDS, Label, MAX_ATOMS, MapAtom, MapLattice, Wanted, key_slot, tidy,
    unordered_pairs,
};
use crate::descr::budget;
use crate::descr::integers::IntSet;
use crate::descr::values::{Field, Values};
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

// THEORY: the-key-partition-is-by-kind
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

// THEORY: records-maps-and-structs
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

// THEORY: a-clause-is-a-region, clauses-are-quasi-k-step
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

// THEORY: the-key-partition-is-by-kind
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

// THEORY: the-key-partition-is-by-kind
/// A want with one candidate left names a key the atom requires, and the
/// required set is read from the *viable* candidates.
///
/// `S` asks for some key of a part to map into a type, and the candidates
/// are the part's labels and its default. Where every candidate but one is
/// empty, every dict of the atom carries the one left, so it joins the
/// required set -- which is what the `1`/`True` reading consults. Built by
/// hand, because no public construction reaches an atom whose want is
/// witnessed by exactly one label while its boolean twin is required: the
/// generator's complements produce wants over defaults, and the lattice
/// operations tidy an empty atom away before its verdict is read.
///
/// Both directions, so the reading cannot be inverted and pass: with the
/// label the only viable witness the atom is empty, and with the default the
/// only viable witness the same atom is inhabited, because a want the
/// default satisfies names no key.
#[test]
fn a_want_with_one_viable_witness_requires_that_key() {
    let int_slot = key_slot(Some(Kind::Int)).expect("the integers are a part");
    let one = || Values::Only(IntSet::just(1));
    let atom = |label_ty: Values<IntSet>, default_ty: Values<IntSet>| {
        let mut atom: MapAtom<IntSet> = MapAtom::top();
        atom.labels.insert(
            Label::Bool(true),
            Field {
                ty: one(),
                absent: false,
            },
        );
        atom.labels.insert(
            Label::Int(1),
            Field {
                ty: label_ty,
                absent: true,
            },
        );
        atom.defaults[int_slot] = Field {
            ty: default_ty,
            absent: true,
        };
        atom.wanted.push(Wanted {
            slot: int_slot,
            ty: one(),
            besides: BTreeSet::new(),
        });
        atom
    };

    // The label is the one viable witness: the key `1` is required beside the
    // required key `True`, and one dict cannot carry both.
    let by_the_label = atom(one(), Values::none());
    assert_eq!(by_the_label.emptiness(), Verdict::Empty);
    for dict in dicts() {
        assert!(
            !by_the_label.holds(&dict),
            "{dict:?} is a dict of an empty atom"
        );
    }
    // The default is the one viable witness: any other integer key satisfies
    // the want, so no key is required by it and the atom is inhabited.
    let by_the_default = atom(Values::none(), one());
    assert_eq!(by_the_default.emptiness(), Verdict::Inhabited);
    let witness = vec![
        Entry {
            label: Some(Label::Bool(true)),
            kind: Some(Kind::Bool),
            value: 1,
        },
        Entry {
            label: Some(Label::Int(2)),
            kind: Some(Kind::Int),
            value: 1,
        },
    ];
    assert!(by_the_default.holds(&witness));
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

// THEORY: the-descriptor
/// A meet with a negated union answers what the same meet spelled out answers.
///
/// `R` names two keys and gives each two values; the four corners fix both
/// keys. Their union *is* `R`, so `R` minus them holds no dict -- and the
/// question is whether both spellings of "minus them" say so, because they are
/// one set and a descriptor decides a set.
///
/// Complementing the union expands it whole: a product of four ten-atom
/// complements, which passes [`MAX_ATOMS`] before the meet with `R` can drop a
/// single atom. Meeting one factor at a time prunes first -- eight of `¬c₁`'s
/// ten atoms want a key outside `a` and `b`, which a closed `R` has none of --
/// so the width the bound sees is the width of the answer rather than the
/// width of the widest intermediate.
#[test]
fn a_meet_with_a_negated_union_answers_as_the_spelled_out_meet() {
    let two = IntSet::between(Some(1), Some(2));
    let closed = |labels: Vec<(Label, IntSet, bool)>| {
        MapLattice::record(labels, []).expect("a two-field record")
    };
    let corner = |a: i64, b: i64| {
        closed(vec![
            (Label::str("a"), IntSet::just(a), false),
            (Label::str("b"), IntSet::just(b), false),
        ])
    };
    let whole = closed(vec![
        (Label::str("a"), two.clone(), false),
        (Label::str("b"), two, false),
    ]);
    let corners = [corner(1, 1), corner(1, 2), corner(2, 1), corner(2, 2)];

    // Spelled out: each corner removed in turn.
    let stepwise = corners
        .iter()
        .try_fold(whole.clone(), |left, corner| {
            left.intersect(&corner.complement())
        })
        .expect("a difference no wider than the record");
    assert_eq!(
        stepwise.emptiness(),
        Verdict::Empty,
        "the four corners are the record, so what is left holds no dict"
    );

    // The same set, written as one complemented union.
    let joined = corners
        .iter()
        .skip(1)
        .try_fold(corners[0].clone(), |left, corner| left.union(corner))
        .expect("a union of four records");
    let together = whole
        .intersect(&joined.complement())
        .expect("the same difference, spelled once");
    assert_eq!(
        together.emptiness(),
        Verdict::Empty,
        "and the spelling of the difference is not what decides it"
    );
}

/// A negated union too wide to expand answers *unknown*, never inhabited.
///
/// The polarity carries a complement the bound cannot rebuild into a union, and
/// every reading of such a form has to say so rather than guess. Emptiness is
/// the reading where guessing is expensive in one direction: a lattice reported
/// empty is a lattice a difference is proved against, so an unexpanded negation
/// read as "empty" would prove an inclusion out of a refusal to look.
///
/// Driven under an allowance small enough that the expansion refuses, which is
/// the same thing the bound does on a wide union and is reachable without one.
#[test]
fn a_negation_the_bound_cannot_expand_answers_unknown() {
    let closed = |name: &str, value: i64| {
        MapLattice::record(
            vec![(Label::str(name), IntSet::just(value), false)],
            core::iter::empty(),
        )
        .expect("a one-field record")
    };
    let wide = closed("a", 1)
        .union(&closed("b", 2))
        .expect("two records join");

    // Unbudgeted, the complement is a union and the verdict is a proof.
    assert_eq!(wide.complement().emptiness(), Verdict::Inhabited);

    // Under an allowance the expansion cannot pay for, the negation is carried
    // and the reading declines.
    budget::under(0, || {
        let held = wide.complement();
        assert_eq!(
            held.emptiness(),
            Verdict::Unknown,
            "a negation nothing expanded is neither empty nor inhabited"
        );
        // And it still answers about a dict, because `holds` reads the polarity
        // rather than the expansion.
        assert!(
            !held.holds(&[at("a", 1)]),
            "a dict the union holds is outside"
        );
        assert!(held.holds(&[at("a", 2)]), "and one it does not is inside");
    });
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

/// One drawn entry: a key of any part the generator's atoms name, and a
/// value in the range their integer sets separate.
///
/// The labelled keys are the two the atoms name and one they do not; the
/// unlabelled ones cover the integer part, the boolean part -- whose labels
/// fold onto the integers' -- and a part no atom constrains.
fn entry() -> impl Strategy<Value = Entry<i64>> {
    let key = prop_oneof![
        Just((Some(Label::str("a")), Some(Kind::Str))),
        Just((Some(Label::str("b")), Some(Kind::Str))),
        Just((Some(Label::str("c")), Some(Kind::Str))),
        Just((Some(Label::Int(1)), Some(Kind::Int))),
        Just((Some(Label::Int(2)), Some(Kind::Int))),
        Just((Some(Label::Bool(true)), Some(Kind::Bool))),
        Just((Some(Label::Bool(false)), Some(Kind::Bool))),
        Just((Some(Label::NoneType), Some(Kind::NoneType))),
        Just((None, Some(Kind::Int))),
        Just((None, Some(Kind::Float))),
        Just((None, None)),
    ];
    (key, 0i64..=3).prop_map(|((label, kind), value)| Entry { label, kind, value })
}

/// A drawn dict: up to four entries over distinct keys.
///
/// Distinct as a dict's keys are: `1` and `True` are one key, so an entry
/// carrying either drops the other. The fixed eight of [`dicts`] are kept
/// beside these, so a law asked over the draw is asked over them too.
fn drawn_dict() -> impl Strategy<Value = Vec<Entry<i64>>> {
    proptest::collection::vec(entry(), 0..=4).prop_map(|entries| {
        let mut kept: Vec<Entry<i64>> = Vec::new();
        for entry in entries {
            let repeats = kept.iter().any(|held| {
                held.kind == entry.kind
                    && (held.label == entry.label
                        || matches!((&held.label, &entry.label), (Some(a), Some(b)) if super::one_key(a, b)))
            });
            if !repeats {
                kept.push(entry);
            }
        }
        kept
    })
}

/// A universe of dicts: the fixed eight and the drawn ones.
fn dicts_with(drawn: &[Vec<Entry<i64>>]) -> Vec<Vec<Entry<i64>>> {
    let mut all = dicts();
    all.extend(drawn.iter().cloned());
    all
}

/// Agreement over a universe a test draws rather than the fixed eight.
fn same_over(a: &MapLattice<IntSet>, b: &MapLattice<IntSet>, dicts: &[Vec<Entry<i64>>]) -> bool {
    dicts.iter().all(|d| a.holds(d) == b.holds(d))
}

/// Lattices over the integer sets whose own laws are already held.
fn lattice() -> impl Strategy<Value = MapLattice<IntSet>> {
    let leaf = prop_oneof![
        Just(MapLattice::empty()),
        Just(MapLattice::all()),
        (1i64..=2).prop_map(|n| MapLattice::label(Label::str("a"), IntSet::just(n), false)),
        (1i64..=2).prop_map(|n| MapLattice::label(Label::str("a"), IntSet::just(n), true)),
        (1i64..=2).prop_map(|n| MapLattice::label(Label::str("b"), IntSet::just(n), false)),
        (1i64..=3).prop_map(|n| MapLattice::label(Label::str("c"), IntSet::just(n), true)),
        Just(MapLattice::keyed(Kind::Str, IntSet::just(1))),
        Just(MapLattice::keyed(Kind::Str, IntSet::just(3))),
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
    // into a run that outlasts a sweep, and an allowance per case so a
    // mutation that removes a pruning shortcut cannot either: see
    // `budget::law`.
    #![proptest_config(ProptestConfig {
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    // THEORY: records-maps-and-structs, each-kind-is-closed
    /// The Boolean algebra, checked against the dicts.
    #[test]
    fn the_lattice_laws_hold_of_the_dicts(
        a in lattice(),
        b in lattice(),
        c in lattice(),
    ) {
        let _allowance = budget::law();
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
        let _allowance = budget::law();
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

    // THEORY: the-descriptor
    /// One set written two ways is one set, and is decided once.
    ///
    /// `a ∧ ¬(b ∨ c)` and `a ∧ ¬b ∧ ¬c` are one set by De Morgan. The laws
    /// above hold each spelling against the dicts *separately*; this holds the
    /// two against each other, which is what the meet against a negated side
    /// has to preserve now that it removes one factor at a time rather than
    /// rebuilding the union first.
    ///
    /// Two claims, and the second is the one membership cannot make. The
    /// dicts must agree, because the two are one set. And where both spellings
    /// *build*, the verdict must agree too -- a decline and a proof admit the
    /// same dicts, namely none, so a law read against values alone passes on a
    /// pair where one spelling answers and the other does not.
    ///
    /// Where one spelling does not build, nothing is asserted, and that is not
    /// a gap in the law. [`MAX_ATOMS`] bounds the *result*, the two spellings
    /// reach it through different intermediates, and neither order is narrower
    /// everywhere: a difference wide enough to reach the bound reaches it under
    /// one spelling before the other, whichever way round the factors go. What
    /// the meet guarantees is that no spelling is harder *by construction*,
    /// which is what `a_meet_with_a_negated_union_answers_as_the_spelled_out_meet`
    /// pins on the shape that showed it.
    #[test]
    fn the_verdict_is_stable_under_de_morgan(
        a in lattice(),
        b in lattice(),
        c in lattice(),
    ) {
        let _allowance = budget::law();
        // Where the union itself does not fit there is no second spelling to
        // compare: the law is about the *difference* being written two ways,
        // and `b ∪ c` is a term the caller writes before either difference.
        let Some(joined) = b.union(&c) else {
            return Ok(());
        };
        let stepwise = a
            .intersect(&b.complement())
            .and_then(|left| left.intersect(&c.complement()));
        let together = a.intersect(&joined.complement());
        if let (Some(stepwise), Some(together)) = (&stepwise, &together) {
            prop_assert!(
                same(stepwise, together),
                "one set, two spellings, two sets"
            );
            prop_assert_eq!(
                stepwise.emptiness(),
                together.emptiness(),
                "one set, two spellings, two verdicts"
            );
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
        let _allowance = budget::law();
        match a.emptiness() {
            Verdict::Empty => prop_assert!(
                dicts().iter().all(|d| !a.holds(d)),
                "an empty lattice holds no dict"
            ),
            Verdict::Inhabited | Verdict::Unknown => {}
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        max_shrink_time: 2_000,
        ..ProptestConfig::default()
    })]

    // THEORY: records-maps-and-structs
    /// ICFP 2023's (12): the meet of two atoms holds exactly the dicts both
    /// hold, over dicts the test draws rather than the eight it lists.
    ///
    /// The meet is the componentwise meet -- each label's type met, each
    /// part's default met, the wanted sets united -- and that construction is
    /// a set only if it holds a dict exactly when both operands do. Drawn
    /// dicts reach the keys the fixed eight do not: a third label, a boolean
    /// key beside its integer, a key of a part no atom constrains.
    #[test]
    fn a_meet_holds_the_dicts_of_both_over_drawn_dicts(
        a in lattice(),
        b in lattice(),
        drawn in proptest::collection::vec(drawn_dict(), 1..6),
    ) {
        let _allowance = budget::law();
        let Some(met) = a.intersect(&b) else {
            return Ok(());
        };
        for dict in dicts_with(&drawn) {
            prop_assert_eq!(
                met.holds(&dict),
                a.holds(&dict) && b.holds(&dict),
                "the meet of {:?} and {:?} disagrees about {:?}", a, b, dict
            );
        }
        if let Some(joined) = a.union(&b) {
            for dict in dicts_with(&drawn) {
                prop_assert_eq!(
                    joined.holds(&dict),
                    a.holds(&dict) || b.holds(&dict),
                    "the join of {:?} and {:?} disagrees about {:?}", a, b, dict
                );
            }
        }
    }

    // THEORY: records-maps-and-structs
    /// The lattice and complement laws, over drawn dicts.
    ///
    /// The same laws the block above holds, with the universe drawn: a law
    /// that holds on eight dicts over two labels holds on the fragment those
    /// eight reach, and the atoms reach more.
    #[test]
    fn the_lattice_laws_hold_of_the_dicts_over_drawn_dicts(
        a in lattice(),
        b in lattice(),
        drawn in proptest::collection::vec(drawn_dict(), 1..6),
    ) {
        let _allowance = budget::law();
        let universe = dicts_with(&drawn);
        let not_a = a.complement();
        prop_assert!(same_over(&not_a.complement(), &a, &universe), "twice is nothing");
        for dict in &universe {
            prop_assert_ne!(a.holds(dict), not_a.holds(dict), "{:?} is in {:?} and its complement, or neither", dict, a);
        }
        if let Some(met) = a.intersect(&b)
            && let Some(absorbed) = a.union(&met)
        {
            prop_assert!(same_over(&absorbed, &a, &universe), "join absorbs the meet");
        }
        let not_b = b.complement();
        if let (Some(joined), Some(met)) = (a.union(&b), not_a.intersect(&not_b)) {
            prop_assert!(same_over(&joined.complement(), &met, &universe), "de Morgan one way");
        }
        if let (Some(met), Some(joined)) = (a.intersect(&b), not_a.union(&not_b)) {
            prop_assert!(same_over(&met.complement(), &joined, &universe), "and the other");
        }
    }

    // THEORY: no-negative-clause-component
    /// Two spellings of one atom are one atom: a label written with its
    /// region's own default says nothing, and is dropped.
    ///
    /// ICFP 2023 Definition 2.2 declines univocality of the written notation
    /// and Theorem 4.2 makes the operators independent of the finite `L`
    /// chosen over `dom`; the descriptor answers both by reading `dom`
    /// semantically. Over drawn lattices, every atom padded with a label
    /// carrying its part's default absorbs back to the atom, and the lattice
    /// rebuilt from the padded atoms compares equal to the one it was drawn
    /// as -- equality, not agreement on dicts, since the claim is about the
    /// representation.
    #[test]
    fn a_label_carrying_its_regions_default_is_absorbed(a in lattice()) {
        let _allowance = budget::law();
        // A label no drawn atom names, so the padding adds a spelling rather
        // than overwriting a constraint.
        let label = Label::str("d");
        let padded: Vec<MapAtom<IntSet>> = a
            .atoms
            .iter()
            .map(|atom| {
                let mut padded = atom.clone();
                padded.labels.insert(label.clone(), atom.default_for(Some(Kind::Str)));
                padded
            })
            .collect();
        for (atom, padded) in a.atoms.iter().zip(&padded) {
            prop_assert_eq!(&padded.clone().absorbed(), &atom.clone().absorbed());
        }
        let Some(atoms) = tidy(padded) else {
            return Ok(());
        };
        let rebuilt = MapLattice {
            atoms,
            negated: a.negated,
        };
        prop_assert_eq!(&rebuilt, &a, "the padded spelling is another lattice");
    }

    // THEORY: records-maps-and-structs
    /// ICFP 2023's (11), in the direction a proof runs: an atom decided empty
    /// holds no dict, drawn or listed.
    ///
    /// The other direction is not a law a finite universe can state -- an
    /// inhabited atom need not have a witness among a few drawn dicts -- so
    /// the proof is what is held, and it is held over dicts the eight fixed
    /// ones do not spell: a key and its boolean, which a dict carries as one
    /// entry, is where a required-key reading goes wrong.
    #[test]
    fn an_emptiness_holds_no_drawn_dict(
        a in lattice(),
        drawn in proptest::collection::vec(drawn_dict(), 1..6),
    ) {
        let _allowance = budget::law();
        if a.emptiness() == Verdict::Empty {
            for dict in dicts_with(&drawn) {
                prop_assert!(!a.holds(&dict), "{:?} is decided empty and holds {:?}", a, dict);
            }
        }
    }
}

// THEORY: the-descriptor
/// A complement is expanded where the expansion is one product, and carried
/// under the flag past that.
///
/// Three widths, three claims. No atoms complements into every dict and every
/// dict back into none, which is what keeps the cheap forms canonical: two
/// descriptors holding the same dicts compare equal rather than differing by
/// the route each took. Two atoms is a product of two complements, a width the
/// meet it is headed for would have pruned, so the atoms are carried as they
/// are and the polarity says what they mean.
#[test]
fn a_complement_is_expanded_only_where_it_is_one_product() {
    let none: MapLattice<IntSet> = MapLattice::empty();
    let every: MapLattice<IntSet> = MapLattice::all();
    assert_eq!(
        none.complement(),
        every,
        "no atoms are every dict complemented"
    );
    assert_eq!(
        every.complement(),
        none,
        "and every dict is none complemented"
    );

    let keyed = |label: &str, value: i64| {
        MapLattice::record([(Label::str(label), IntSet::just(value), false)], [])
            .expect("a one-field record")
    };
    let two = keyed("a", 0).union(&keyed("b", 1)).expect("two atoms");
    let carried = two.complement();
    assert!(
        carried.negated,
        "two atoms are carried rather than expanded"
    );
    assert_eq!(carried.atoms, two.atoms, "with the atoms as they were");
    assert!(
        same(&carried.complement(), &two),
        "and the flag complements back into the union it carries"
    );
}

/// The bound reads the width of the union, never the number of pairs.
///
/// Twenty atoms met with twenty is four hundred pairs, and the pairs are where
/// the count grows: it passes [`MAX_ATOMS`] long before the meet is done. The
/// union they collapse to is twenty atoms wide, because an atom wanting two
/// values under one key holds no dict and a union carries no such atom.
/// Refusing on the raw count would make the bound a question about the order
/// the factors were multiplied in, which is a property of how a difference was
/// written rather than of the dicts it names.
#[test]
fn a_meet_is_bounded_by_the_width_of_its_union_and_not_by_its_pairs() {
    let keyed = |value: i64| {
        MapLattice::record([(Label::str("a"), IntSet::just(value), false)], [])
            .expect("a one-field record")
    };
    let wide = (1..20i64)
        .try_fold(keyed(0), |left, n| left.union(&keyed(n)))
        .expect("twenty atoms");
    let met = wide
        .intersect(&wide)
        .expect("four hundred pairs, one union");
    assert!(
        wide.atoms.len() * wide.atoms.len() > MAX_ATOMS,
        "the row needs a pair count the bound would refuse"
    );
    assert_eq!(
        met.atoms.len(),
        wide.atoms.len(),
        "and an answer no wider than either side"
    );
    assert!(same(&met, &wide), "a meet with itself holds what it held");
}
