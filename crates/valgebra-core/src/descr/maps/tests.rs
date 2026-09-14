use super::{Entry, KEY_KINDS, Label, MAX_ATOMS, MapLattice, key_slot};
use crate::descr::budget;
use crate::descr::integers::IntSet;
use crate::kind::Kind;
use crate::verdict::Verdict;

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
