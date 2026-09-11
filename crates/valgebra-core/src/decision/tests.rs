use std::sync::Arc;

use super::*;
use crate::descr::classes::Class;
use crate::descr::lower::Operand;
use crate::ir::Openness;

/// The structural inclusion procedure alone, with no descriptor beside it.
///
/// `is_subtype_of_under` asks the descriptor where the rules decline, which
/// is what widens the public relations -- and what makes a defect in a rule
/// invisible through them, since the answer comes out right for the other
/// reason. A rule is pinned by asking it on its own.
fn structural(sub: &Schema, sup: &Schema) -> bool {
    let budget = Cell::new(DECISION_BUDGET);
    sub.is_subtype_rec(
        sup,
        SubtypeCx {
            oracle: &NoLeafRelations,
            defs: &[],
            budget: &budget,
        },
        &mut Vec::new(),
    )
    .holds()
}

/// Every set is below the universe, however the universe is spelled.
///
/// A refinement with no constraint denotes exactly its base, and until it
/// said so the two spellings were decided differently: `Anything` reached
/// `Refine { base: Anything }` through the refinement rule, and the gradual
/// `Any` reached `Anything` through the region bound but not the refinement,
/// because only a region set carries that bound. The fuzzer found it as a
/// union holding both.
#[test]
fn every_set_is_below_the_universe_however_it_is_spelled() {
    let bare = Schema::Refine {
        base: Arc::new(Schema::ANYTHING),
        constraints: Vec::new().into(),
    };
    // The complement of the universe is empty, which is the same fact read
    // through the *other* fold: `region_set` decides the inclusion above,
    // and `empty_and_region` decides this. Both carry the rule, so both are
    // asked -- a fix in one of two folds is half a fix.
    assert!(Schema::Complement(Arc::new(bare.clone())).is_empty());
    assert!(!bare.is_empty(), "and the universe itself is not");

    for universe in [
        Schema::ANYTHING,
        bare.clone(),
        Schema::Union(vec![bare].into()),
    ] {
        for sub in [
            Schema::ANY,
            Schema::ANYTHING,
            Schema::Int,
            Schema::Union(vec![Schema::ANY, Schema::ANYTHING].into()),
        ] {
            assert!(
                sub.is_subtype_of(&universe),
                "{sub:?} is below the universe {universe:?}"
            );
        }
    }
}

/// The structural inclusion rules, held to their own work.
///
/// The descriptor is asked after these rules and decides much of what they
/// do, so a defect in one is invisible through `is_subtype_of` -- the answer
/// comes out right for the other reason. `is_subtype_of_under` is the
/// structural procedure alone, which is where each rule has to be pinned.
#[test]
fn the_structural_inclusion_rules_decide_without_the_descriptor() {
    // Built raw rather than through the smart constructors, which fold a
    // meet of two kinds before it ever reaches the rule under test.
    let meet = |members: [Schema; 2]| Schema::Intersection(members.into());
    let joined = Schema::union([Schema::Int, Schema::Float]);

    // `A ⊆ (Y ∩ Z)` needs both conjuncts.
    assert!(structural(
        &Schema::Int,
        &meet([Schema::Int, joined.clone()])
    ));
    assert!(!structural(
        &Schema::Int,
        &meet([Schema::Int, Schema::Float])
    ));
    // `(A ∩ B) ⊆ C` when some conjunct already is, and when the meet lands
    // in one branch of a union supertype -- the second half of that arm.
    assert!(structural(&meet([Schema::Int, Schema::Str]), &Schema::Int));
    assert!(structural(
        &meet([Schema::Int, Schema::Bytes]),
        &Schema::union([Schema::Int, Schema::Str])
    ));
    // An inhabited meet that no branch covers: an empty one would be a
    // subtype of everything and would say nothing about the rule.
    assert!(!structural(
        &meet([Schema::Int, joined.clone()]),
        &Schema::union([Schema::Float, Schema::Str])
    ));

    // Set and frozenset inclusion reduces to element inclusion, and the two
    // kinds do not cross.
    let ints = Schema::set(Schema::Int);
    assert!(structural(&ints, &Schema::set(joined.clone())));
    assert!(!structural(&ints, &Schema::frozen_set(joined.clone())));
    assert!(structural(
        &Schema::frozen_set(Schema::Int),
        &Schema::frozen_set(joined.clone())
    ));

    // Each rule above is a *shortcut*: delete it and the general reduction
    // `A ⊆ B` to `A ∩ ¬B = ∅` decides the same thing. What the shortcut is
    // for is the case where that reduction declines -- and an atom carrying
    // a callback is exactly one, since the complement law does not hold of
    // it. So each rule is pinned over a schema the reduction cannot read.
    let opaque = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
    };
    assert!(structural(&opaque, &meet([opaque.clone(), opaque.clone()])));
    assert!(structural(&meet([opaque.clone(), Schema::Str]), &opaque));
    assert!(structural(
        &meet([opaque.clone(), Schema::Str]),
        &Schema::union([opaque.clone(), Schema::Float])
    ));

    // Complement is contravariant: the inclusion under it runs the other way.
    let wider = Schema::Complement(Arc::new(Schema::Int));
    let narrower = Schema::Complement(Arc::new(joined));
    assert!(structural(&narrower, &wider));
    assert!(!structural(&wider, &narrower));
    // And over an atom the reduction cannot read, where nothing else does.
    assert!(structural(
        &Schema::Complement(Arc::new(opaque.clone())),
        &Schema::Complement(Arc::new(meet([opaque.clone(), Schema::Str])))
    ));
}

/// A fixed-arity sequence splits across the branches of a union, decided by
/// the product rule rather than by the descriptor beside it.
#[test]
fn a_product_splits_across_a_union_without_the_descriptor() {
    let pair = |a: Schema, b: Schema| Schema::tuple(SeqShape::fixed([a, b]));
    let subject = pair(Schema::union([Schema::Int, Schema::Str]), Schema::Int);
    let split = Schema::union([
        pair(Schema::Int, Schema::Int),
        pair(Schema::Str, Schema::Int),
    ]);

    assert!(structural(&subject, &split), "the branches cover it");
    assert!(
        !structural(
            &subject,
            &Schema::union([
                pair(Schema::Int, Schema::Int),
                pair(Schema::Bytes, Schema::Int),
            ])
        ),
        "and branches that do not cover it decide no"
    );
    // A repeated tail is not a product, so there is nothing to split.
    let variadic = Schema::tuple(SeqShape::homogeneous(Schema::union([
        Schema::Int,
        Schema::Str,
    ])));
    assert!(!structural(
        &variadic,
        &Schema::union([
            Schema::tuple(SeqShape::homogeneous(Schema::Int)),
            Schema::tuple(SeqShape::homogeneous(Schema::Str)),
        ])
    ));
}

/// A callback hides behind every container, and the walk that looks for one
/// must enter each.
///
/// `A ∩ ¬A = ∅` is declined for an atom that is not a set, and a predicate
/// nested inside a container makes the whole container one -- so each way a
/// schema holds another is a way the search must descend.
#[test]
fn a_callback_is_found_through_every_container() {
    let predicate = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
    };
    let field = |schema: Schema| Field {
        name: "x".into(),
        schema,
        required: true,
    };
    let wrappers: [Schema; 11] = [
        predicate.clone(),
        Schema::set(predicate.clone()),
        Schema::frozen_set(predicate.clone()),
        Schema::Complement(Arc::new(predicate.clone())),
        Schema::union([predicate.clone(), Schema::Int]),
        Schema::list(SeqShape::fixed([predicate.clone()])),
        Schema::list(SeqShape::homogeneous(predicate.clone())),
        Schema::KeyedMap {
            fields: vec![field(predicate.clone())].into(),
            defaults: Vec::new().into(),
        },
        Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause {
                key: Schema::Str,
                value: predicate.clone(),
            }]
            .into(),
        },
        Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause {
                key: predicate.clone(),
                value: Schema::Str,
            }]
            .into(),
        },
        Schema::AttrRecord {
            fields: vec![field(predicate.clone())].into(),
        },
    ];
    for wrapper in wrappers {
        assert!(
            !denotes_a_set_within(&wrapper, &NoLeafRelations, &[]),
            "a predicate inside {wrapper:?} is still a predicate"
        );
        let meet = Schema::Intersection(
            vec![
                wrapper.clone(),
                Schema::Complement(Arc::new(wrapper.clone())),
            ]
            .into(),
        );
        assert!(
            !meet.is_empty_under(&[]),
            "so the law must decline {wrapper:?}"
        );
    }

    // Without one, the same shapes are sets and the law still decides.
    let plain = Schema::set(Schema::Int);
    assert!(denotes_a_set_within(&plain, &NoLeafRelations, &[]));
    assert!(
        Schema::Intersection(vec![plain.clone(), Schema::Complement(Arc::new(plain))].into())
            .is_empty_under(&[])
    );
}

/// An oracle that reports every atom it is asked about as a set, standing
/// for the bindings' answer about a pure class.
struct Pure;

impl Constants for Pure {}

impl LeafRelations for Pure {
    fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
        None
    }

    fn atom_denotes_a_set(&self, _atom: &Schema) -> Option<bool> {
        Some(true)
    }
}

/// A reference is refused by the fold whatever it stands for. The law asks
/// `A` twice, and a back edge is not a set until it is unfolded -- which the
/// syntactic fold does not do, and which a self-reference has not yet been
/// resolved enough to allow.
#[test]
fn a_reference_is_not_a_set_the_fold_may_cancel() {
    for reference in [Schema::Ref(DefIx::new(0)), Schema::SelfRef(0)] {
        assert!(
            !denotes_a_set_within(&reference, &NoLeafRelations, &[]),
            "{reference:?}"
        );
        // Nested, so a reference reached through a constructor refuses too.
        let nested = Schema::set(reference.clone());
        assert!(!denotes_a_set_within(&nested, &NoLeafRelations, &[]));
        // And the constructors decline the two cancelling laws for it.
        let not = |s: Schema| Schema::Complement(Arc::new(s));
        assert_ne!(
            Schema::union([reference.clone(), not(reference.clone())]),
            Schema::ANYTHING
        );
        assert_ne!(
            Schema::meet([reference.clone(), not(reference)]),
            Schema::Nothing
        );
    }
}

/// A class is referred to the oracle, and a core with none declines it.
///
/// The default answers nothing, which is what makes an unwired core
/// conservative rather than wrong: `Instance ∩ ¬Instance` is not decided
/// empty until the bindings say the class is pure.
#[test]
fn a_class_without_an_oracle_is_not_a_set() {
    let class = Schema::Instance(ClassIx::new(0));
    assert_eq!(NoLeafRelations.atom_denotes_a_set(&class), None);
    assert!(!denotes_a_set_within(&class, &NoLeafRelations, &[]));

    assert!(denotes_a_set_within(&class, &Pure, &[]));
}

/// The structural region rules decide without the descriptor, and stay
/// pinned where it would otherwise answer for them.
///
/// `is_empty` asks the descriptor after these rules, so a defect in them is
/// invisible through it -- the answer comes out right for the other reason.
/// `is_empty_under` is the structural procedure alone, which is where a rule
/// has to be held to its own work.
#[test]
fn the_scalar_regions_decide_a_disjoint_meet_without_the_descriptor() {
    for (left, right) in [
        (Schema::NoneType, Schema::Str),
        (Schema::Str, Schema::Float),
        (Schema::NoneType, Schema::Bytes),
    ] {
        let meet = Schema::meet([left.clone(), right.clone()]);
        assert!(
            meet.is_empty_under(&[]),
            "{left:?} and {right:?} share no region"
        );
    }
}

/// A sequence with an uninhabited prefix element admits nothing, decided by
/// the structural rule rather than by the descriptor beside it.
#[test]
fn an_uninhabited_prefix_empties_a_sequence_without_the_descriptor() {
    let empty_element = Schema::list(SeqShape::fixed([Schema::Nothing]));
    assert!(empty_element.is_empty_under(&[]));

    // The tail is not a prefix: a sequence may stop before it, so an
    // uninhabited tail leaves the sequence that ends at the prefix.
    let empty_tail = Schema::list(SeqShape::homogeneous(Schema::Nothing));
    assert!(!empty_tail.is_empty_under(&[]));
}

use crate::ir::{ClassIx, ConstIx, PredIx};
use proptest::prelude::*;

/// A generator mixing the scalar-decidable atoms with opaque leaves (the
/// gradual `Any`, a literal, a content-bearing set) under the Boolean
/// combinators, so both the region-carrying and the region-`None` paths and
/// their propagation through every combinator are exercised.
fn schema() -> impl Strategy<Value = Schema> {
    let leaf = prop_oneof![
        Just(Schema::ANYTHING),
        Just(Schema::Nothing),
        Just(Schema::NoneType),
        Just(Schema::Bool),
        Just(Schema::Int),
        Just(Schema::Float),
        Just(Schema::Str),
        Just(Schema::Bytes),
        Just(Schema::ANY),
        Just(Schema::Literal(ConstIx::new(0))),
        Just(Schema::set(Schema::Int)),
    ];
    leaf.prop_recursive(4, 24, 3, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 1..4).prop_map(|m| Schema::Union(m.into())),
            proptest::collection::vec(inner.clone(), 1..4)
                .prop_map(|m| Schema::Intersection(m.into())),
            inner.prop_map(|s| Schema::Complement(Arc::new(s))),
        ]
    })
}

proptest! {
    /// Asking whether a schema covers the universe by reading its region is the
    /// same question as asking whether its complement is empty. The lattice
    /// bound `A subset-of U` is decided the second way in principle and the
    /// first way in the code, because the first builds nothing; this holds the
    /// two together so the cheaper one cannot drift from the rule it stands in
    /// for.
    #[test]
    fn covering_the_universe_is_the_complement_being_empty(s in schema()) {
        let budget = Cell::new(DECISION_BUDGET);
        let via_complement = Schema::Complement(Arc::new(s.clone()))
            .is_empty_rec(&NoLeafRelations, &[], &mut Vec::new(), &budget);
        let budget = Cell::new(DECISION_BUDGET);
        let (_, regions) =
            s.empty_and_region(&NoLeafRelations, &[], &mut Vec::new(), &budget);
        prop_assert_eq!(via_complement, regions == Regions::Known(Region::ALL));
    }

    /// The bottom-up region folded by `empty_and_region` is exactly the region
    /// `region_set` recomputes from scratch, for every schema. This pins the two
    /// region code paths together so a future change to one cannot silently
    /// diverge from the other (the emptiness decision relies on their agreement).
    #[test]
    fn empty_and_region_folds_the_same_region_as_region_set(s in schema()) {
        let folded = s
            .empty_and_region(
                &NoLeafRelations,
                &[],
                &mut Vec::new(),
                &Cell::new(DECISION_BUDGET),
            )
            .1;
        prop_assert_eq!(folded, s.region_set());
    }
}

/// A union fold stops only once a member is known inhabited. The stopping rule
/// reads the *inhabited* accumulator, not the empty one: a member that is
/// empty and opaque leaves the verdict open, and breaking there would report a
/// union empty on the strength of the members walked so far.
///
/// The witness needs a member that is empty with an unknown region -- a record
/// with an uninhabited required field -- followed by an inhabited one, because
/// a member that is empty with a *known* region cannot make the accumulator
/// absorbing on its own.
/// The unordered pairs of a slice: every distinct pair once, in neither order
/// twice, and none of an element with itself. Both disjointness laws scan them
/// -- over members, and over the inners of complements -- so the scan is one
/// function and the law it serves is decided in one place.
#[test]
fn unordered_pairs_yields_each_distinct_pair_once() {
    let pairs: Vec<(i32, i32)> = unordered_pairs(&[1, 2, 3]).map(|(a, b)| (*a, *b)).collect();
    assert_eq!(pairs, [(1, 2), (1, 3), (2, 3)]);
    // The degenerate lengths a member list can reach: no pair to compare.
    assert_eq!(unordered_pairs::<i32>(&[]).count(), 0);
    assert_eq!(unordered_pairs(&[1]).count(), 0);
    // n elements give n*(n-1)/2 pairs, so nothing is visited twice.
    assert_eq!(unordered_pairs(&[1, 2, 3, 4, 5]).count(), 10);
}

#[test]
fn a_union_fold_stops_only_once_a_member_is_inhabited() {
    let uninhabited = Schema::record(
        vec![Field {
            name: "a".into(),
            schema: Schema::Nothing,
            required: true,
        }],
        crate::ir::Openness::Closed,
    );
    assert!(uninhabited.is_empty());
    assert_eq!(uninhabited.region_set(), Regions::Unknown);

    // The union is inhabited by its second member, which the fold reaches only
    // by not stopping at the first.
    let union = Schema::union([uninhabited.clone(), Schema::Int]);
    assert!(!union.is_empty());
    // Every member uninhabited is still empty, so the stop does not hide that.
    assert!(Schema::union([uninhabited.clone(), uninhabited]).is_empty());
}

#[test]
fn is_empty_decides_complement_and_disjoint_intersections() {
    let list = |e| Schema::list(SeqShape::homogeneous(e));
    let not = |s| Schema::Complement(Arc::new(s));

    // A ∩ ¬A is empty for a structural A the scalar region bitset cannot see.
    let a = list(Schema::Int);
    assert!(Schema::Intersection(vec![a.clone(), not(a)].into()).is_empty());

    // `Any` is the top, spelled, so it obeys the law the top obeys: the
    // spelling is not a set and no rule reads it.
    assert!(Schema::Intersection(vec![Schema::ANY, not(Schema::ANY)].into()).is_empty());
    assert!(Schema::Intersection(vec![Schema::ANY, not(Schema::ANYTHING)].into()).is_empty());

    // Disjoint structural kinds: a list is never a set.
    assert!(
        Schema::Intersection(vec![list(Schema::Int), Schema::set(Schema::Int)].into()).is_empty()
    );

    // A refined int is still an int, disjoint from str.
    assert!(
        Schema::Intersection(
            vec![
                Schema::Refine {
                    base: Arc::new(Schema::Int),
                    constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
                },
                Schema::Str,
            ]
            .into()
        )
        .is_empty()
    );

    // Sanity: two same-kind lists share the empty list, so not empty.
    assert!(!Schema::Intersection(vec![list(Schema::Int), list(Schema::Bool)].into()).is_empty());
}

/// The two rules that read only the left side -- a reference unfolds, a
/// refinement drops to its base -- and the union rule beside them.
///
/// A union on the right is tried branch by branch, which commits to a
/// branch. Both readings are sound and neither subsumes the other, so where
/// both apply both are asked; these are the cases that separate them.
#[test]
fn a_reference_and_a_refinement_are_read_beside_the_union_rule() {
    let defs = vec![Schema::union([Schema::Int, Schema::Str])];
    let reference = Schema::Ref(DefIx::new(0));
    let body = Schema::union([Schema::Int, Schema::Str]);
    let refined = |base: Schema| Schema::Refine {
        base: Arc::new(base),
        constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
    };

    // Against a union: only the left-side rule decides these. The reference
    // is in no branch of its own body, and the refinement is in neither
    // branch of the union it refines.
    assert!(reference.is_subtype_of_under(&body, &NoLeafRelations, &defs));
    assert!(refined(body.clone()).is_subtype_of(&body));

    // Against a non-union: the same two rules, reached through their own
    // arms rather than through the union rule.
    assert!(Schema::Ref(DefIx::new(0)).is_subtype_of_under(
        &Schema::union([Schema::Int, Schema::Str, Schema::Bytes]),
        &NoLeafRelations,
        &defs
    ));
    assert!(refined(Schema::Int).is_subtype_of(&Schema::Int));
    let int_def = vec![Schema::Int];
    assert!(Schema::Ref(DefIx::new(0)).is_subtype_of_under(
        &Schema::Int,
        &NoLeafRelations,
        &int_def
    ));

    // The union rule is not replaced by them. A branch equal to the subject
    // settles it, and the base rule would answer no: `int` is not below the
    // refinement, so a subject that IS the refinement is decided only by the
    // branch it equals.
    let narrowed = refined(Schema::Int);
    assert!(narrowed.is_subtype_of(&Schema::union([narrowed.clone(), Schema::Str])));
    // And a subject that neither rule reaches still lands in its branch.
    assert!(Schema::Int.is_subtype_of(&Schema::union([Schema::Int, Schema::Str])));

    // Sound in the other direction: unfolding a reference is not a licence.
    assert!(!Schema::Ref(DefIx::new(0)).is_subtype_of_under(
        &Schema::union([Schema::Int, Schema::Bytes]),
        &NoLeafRelations,
        &defs
    ));
    // A reference no definition resolves decides nothing.
    assert!(!Schema::Ref(DefIx::new(9)).is_subtype_of_under(&body, &NoLeafRelations, &defs));
}

/// A field, spelled once rather than at each of the many sites below.
fn field(name: &str, schema: Schema, required: bool) -> Field {
    Field {
        name: name.into(),
        schema,
        required,
    }
}

/// A record: declared fields and no catch-all clause, so it is closed.
fn closed(fields: Vec<Field>) -> Schema {
    Schema::KeyedMap {
        fields: fields.into(),
        defaults: Vec::new().into(),
    }
}

/// A record with a catch-all clause, which is what makes it open.
fn open(fields: Vec<Field>) -> Schema {
    Schema::KeyedMap {
        fields: fields.into(),
        defaults: vec![MapClause {
            key: Schema::Str,
            value: Schema::ANYTHING,
        }]
        .into(),
    }
}

fn meet_is_empty(members: &[Schema]) -> bool {
    keyed_map_meet_empty(members, &NoLeafRelations, &[], &Cell::new(DECISION_BUDGET))
}

/// The two rules ICFP formulae (11) and (12) give for a meet of record atoms,
/// and the footnote-11 guard that stops each from firing where it must not.
///
/// Driven against the rule rather than through `is_empty`, because the
/// question is which of these shapes the rule answers for: reached through
/// the decision procedure, an intersection has half a dozen other reasons to
/// be reported empty, and a test that only watches the verdict cannot tell
/// which one spoke.
#[test]
fn a_record_meet_is_empty_only_where_a_required_key_cannot_hold() {
    // Rule one: a key required somewhere whose types meet to nothing.
    assert!(meet_is_empty(&[
        closed(vec![field("a", Schema::Int, true)]),
        closed(vec![field("a", Schema::Str, true)]),
    ]));
    // The types must actually meet to nothing. `bool` is below `int`, so
    // these two agree on every bool.
    assert!(!meet_is_empty(&[
        closed(vec![field("a", Schema::Int, true)]),
        closed(vec![field("a", Schema::Bool, true)]),
    ]));

    // Rule two: a key required somewhere and absent from a closed map.
    assert!(meet_is_empty(&[
        closed(vec![field("a", Schema::Int, true)]),
        closed(vec![field("b", Schema::Int, true)]),
    ]));
    // The map that lacks the key must be closed. A clause admits keys the
    // field list does not name, so an open map is no obstacle.
    assert!(!meet_is_empty(&[
        closed(vec![field("a", Schema::Int, true)]),
        open(vec![]),
    ]));
    // And a closed map that declares the key is no obstacle either, which is
    // the same walk reading the field list the other way.
    assert!(!meet_is_empty(&[
        closed(vec![field("a", Schema::Int, true)]),
        closed(vec![field("a", Schema::ANYTHING, true)]),
    ]));

    // Footnote 11: only a REQUIRED key can empty a meet. Two optional fields
    // whose types share nothing still admit the empty dict, and so does a
    // key absent from a closed map when nothing requires it.
    assert!(!meet_is_empty(&[
        closed(vec![field("a", Schema::Int, false)]),
        closed(vec![field("a", Schema::Str, false)]),
    ]));
    assert!(!meet_is_empty(&[
        closed(vec![field("a", Schema::Int, false)]),
        closed(vec![field("b", Schema::Int, false)]),
    ]));
    // Required on ONE side is enough: the meet must satisfy both maps, so a
    // key one of them demands is a key every value in the meet carries.
    assert!(meet_is_empty(&[
        closed(vec![field("a", Schema::Int, true)]),
        closed(vec![field("a", Schema::Str, false)]),
    ]));

    // The first rule is about types from DIFFERENT maps meeting to nothing.
    // A required key whose type is uninhabited in a single map empties that
    // map on its own, and the keyed-map node's own rule decides it -- so
    // this one declines rather than answering a question already answered.
    assert!(!meet_is_empty(&[
        closed(vec![field("a", Schema::Nothing, true)]),
        open(vec![]),
    ]));
    assert!(
        Schema::Intersection(
            vec![
                closed(vec![field("a", Schema::Nothing, true)]),
                open(vec![]),
            ]
            .into()
        )
        .is_empty()
    );

    // Fewer than two maps is not a meet of maps. One map alone is decided by
    // the node's own rule, and a meet with a non-map member says nothing
    // about the keys.
    assert!(!meet_is_empty(&[closed(vec![field(
        "a",
        Schema::Nothing,
        true
    )])]));
    assert!(!meet_is_empty(&[
        closed(vec![field("a", Schema::Int, true)]),
        Schema::Str,
    ]));
    // Three maps: the pair that cannot hold together need not be the first
    // two, so the scan runs over all of them rather than stopping at a count.
    assert!(meet_is_empty(&[
        open(vec![]),
        closed(vec![field("a", Schema::Int, true)]),
        closed(vec![field("a", Schema::Str, true)]),
    ]));
}

/// The structural rules alone, without the descriptor that widens them.
///
/// A test about a *rule's* scope has to ask the rule: through
/// [`Schema::is_subtype_of`] a decline is invisible, because the descriptor
/// answers behind it and a relation the rule was never meant to reach is
/// decided anyway. Which is a better answer and a worse test.
fn by_the_rules(sub: &Schema, sup: &Schema) -> bool {
    let budget = Cell::new(DECISION_BUDGET);
    sub.is_subtype_rec(
        sup,
        SubtypeCx {
            oracle: &NoLeafRelations,
            defs: &[],
            budget: &budget,
        },
        &mut Vec::new(),
    )
    .holds()
}

/// Emptiness by the structural rules alone, for the reason
/// [`by_the_rules`] exists.
fn empty_by_the_rules(schema: &Schema) -> bool {
    empty_by_the_rules_under(schema, &NoLeafRelations)
}

/// [`empty_by_the_rules`] with an oracle, for the arms that need one to
/// order two pooled bounds.
fn empty_by_the_rules_under(schema: &Schema, oracle: &dyn LeafRelations) -> bool {
    schema.is_empty_rec(oracle, &[], &mut Vec::new(), &Cell::new(DECISION_BUDGET))
}

/// A refinement carrying no constraint denotes exactly its base, and both
/// halves of the procedure read it that way.
///
/// Without it the universe had two spellings that decided differently:
/// `anything` is below `Refine { base: anything }` through the refinement
/// arm, and the gradual `Any` is below `anything`, but was not below the
/// refinement -- because only a region set says so. A fuzzer found it.
#[test]
fn the_rules_read_a_refinement_with_no_constraint_as_its_base() {
    let bare = |base| Schema::Refine {
        base: Arc::new(base),
        constraints: Vec::new().into(),
    };
    assert!(by_the_rules(&Schema::ANY, &bare(Schema::ANYTHING)));
    assert!(by_the_rules(
        &Schema::Union(vec![Schema::ANY, Schema::ANYTHING].into()),
        &bare(Schema::ANYTHING),
    ));
    // And the same reading on the emptiness side, where it is the *regions*
    // that the base lends: a bare refinement over the universe covers every
    // region, so its complement covers none and the meet below is empty.
    // Without that the complement has no region set and nothing decides it.
    assert!(empty_by_the_rules(&Schema::Intersection(
        vec![
            Schema::ANYTHING,
            Schema::Complement(Arc::new(bare(Schema::ANYTHING))),
        ]
        .into()
    )));
    assert!(empty_by_the_rules(&bare(Schema::Nothing)));
    assert!(!empty_by_the_rules(&bare(Schema::Str)));
}

/// A meet of a schema and its own complement is empty, where the regions
/// cannot say so.
///
/// The emptiness fold has several ways to reach `Empty` and they are read in
/// order, so a member whose regions are opaque is what leaves the
/// complementary-pair rule as the only one that can answer. A constrained
/// refinement is such a member: its regions are unknown, because a narrowed
/// region set read back through a complement would report an inhabited
/// schema empty.
#[test]
fn the_complementary_pair_rule_answers_where_the_regions_cannot() {
    let bounded = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
    };
    assert_eq!(bounded.region_set(), Regions::Unknown, "no regions to read");
    assert!(empty_by_the_rules(&Schema::Intersection(
        vec![bounded.clone(), Schema::Complement(Arc::new(bounded)),].into()
    )));
}

/// The two complement arms of the subtyping rules, asked of the rules.
///
/// Both relations are also decided by the descriptor, which holds a
/// complement as a set again -- so through [`Schema::is_subtype_of`] either
/// arm could be deleted and every test would still pass. What the arms are
/// for is deciding them *without* building a descriptor, and that is what is
/// asserted here.
#[test]
fn the_rules_relate_a_complement_on_either_side() {
    // Contravariance: `¬A ≤ ¬B` is `B ≤ A`, read backwards. Over containers,
    // because that is where the arm earns its place -- `list[bool] ≤
    // list[int]` is decided by recursing on the element, while the meet
    // `¬list[int] ∧ list[bool]` is not decided empty, so the arm below
    // cannot answer this.
    let not = |schema| Schema::Complement(Arc::new(schema));
    let list = |element| Schema::list(SeqShape::homogeneous(element));
    assert!(by_the_rules(
        &not(list(Schema::Int)),
        &not(list(Schema::Bool))
    ));
    assert!(!by_the_rules(
        &not(list(Schema::Bool)),
        &not(list(Schema::Int))
    ));

    // A complement on the right alone: the question is whether the two share
    // a value, which is emptiness of the meet.
    assert!(by_the_rules(
        &Schema::list(SeqShape::homogeneous(Schema::Int)),
        &not(Schema::Int),
    ));
    assert!(!by_the_rules(&Schema::Bool, &not(Schema::Bool)));
}

/// Two schemas sharing no value, asked of the rules.
///
/// The cheap half reads two kind discriminants; the general half builds the
/// meet and asks whether it is empty. Both are needed, and neither is
/// visible through a relation the descriptor also answers.
#[test]
fn the_rules_decide_that_two_schemas_share_no_value() {
    // Different kinds, settled by the discriminants.
    assert!(Schema::list(SeqShape::homogeneous(Schema::Int)).disjoint(&Schema::Int));
    // One kind, settled by the meet: a complement on the right sends the
    // subtyping question here.
    assert!(by_the_rules(
        &Schema::Str,
        &Schema::Complement(Arc::new(Schema::Int)),
    ));
    assert!(!by_the_rules(
        &Schema::Union(vec![Schema::Str, Schema::Int].into()),
        &Schema::Complement(Arc::new(Schema::Int)),
    ));
}

/// Each way the emptiness fold reaches a verdict, asked of the rules.
///
/// The descriptor decides most of these too, so through
/// [`Schema::is_empty`] a deleted rule is invisible: the answer is still
/// right, and the test that was meant to hold the rule up holds nothing.
/// Asking the rules is what keeps each of them covered.
#[test]
fn the_rules_reach_each_way_a_meet_is_empty() {
    // A member that is itself empty empties the meet.
    assert!(empty_by_the_rules(&Schema::Intersection(
        vec![Schema::Nothing, Schema::Str,].into()
    )));
    // Two kinds that share no region.
    assert!(empty_by_the_rules(&Schema::Intersection(
        vec![Schema::Str, Schema::Float,].into()
    )));
    // A schema beside its own complement.
    assert!(empty_by_the_rules(&Schema::Intersection(
        vec![Schema::Str, Schema::Complement(Arc::new(Schema::Str)),].into()
    )));
    // Two bounds that no value satisfies, which needs an oracle to order
    // the two pooled operands.
    let bounded = |constraint| Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![constraint].into(),
    };
    assert!(empty_by_the_rules_under(
        &Schema::Intersection(
            vec![
                bounded(Constraint::Ge(OperandIx::new(1))),
                bounded(Constraint::Le(OperandIx::new(0))),
            ]
            .into()
        ),
        &ByIndex,
    ));
    // And a meet that is inhabited is not reported empty by any of them.
    assert!(!empty_by_the_rules(&Schema::Intersection(
        vec![
            Schema::Str,
            Schema::Union(vec![Schema::Str, Schema::Int].into()),
        ]
        .into()
    )));
}

/// A sequence is empty exactly when a prefix element is, and the rule says
/// so on its own.
///
/// A tail repeats zero times, so an empty *tail* empties nothing: the
/// sequence that stops at the prefix is still a member. Both halves are
/// asserted, because a rule that only ever answers one way is a rule a
/// deletion cannot be seen through.
#[test]
fn the_rules_read_a_sequence_s_emptiness_off_its_prefix() {
    assert!(empty_by_the_rules(&Schema::list(SeqShape::fixed([
        Schema::Nothing
    ]))));
    assert!(!empty_by_the_rules(&Schema::list(SeqShape::homogeneous(
        Schema::Nothing
    ))));
    assert!(!empty_by_the_rules(&Schema::list(SeqShape::fixed([
        Schema::Str
    ]))));
}

/// The descriptor proves an emptiness no rule reaches: a container meet is
/// the meet of the element sets, and the rules never take one.
#[test]
fn the_descriptor_proves_an_emptiness_the_rules_decline() {
    let bools = Schema::list(SeqShape::homogeneous(Schema::Bool));
    let not_ints = Schema::Complement(Arc::new(Schema::list(SeqShape::homogeneous(Schema::Int))));
    let meet = Schema::Intersection(vec![bools, not_ints].into());

    assert!(
        !empty_by_the_rules(&meet),
        "no rule about shapes reaches it"
    );
    assert!(meet.is_empty(), "and the sets decide it");
}

/// A fixed-arity sequence is a product, and a product is decided against a
/// union of products by the backtrack-free `Phi` -- so a value that lands in
/// no single branch is still decided, which is the whole reason the rule
/// exists.
#[test]
fn a_fixed_sequence_splits_across_the_branches_that_share_its_shape() {
    let tuple = |elements: [Schema; 2]| Schema::tuple(SeqShape::fixed(elements));
    let int_or_str = Schema::union([Schema::Int, Schema::Str]);

    // The split: neither branch contains the subject, and together they do.
    let subject = tuple([int_or_str.clone(), Schema::Int]);
    let split = Schema::union([
        tuple([Schema::Int, Schema::Int]),
        tuple([Schema::Str, Schema::Int]),
    ]);
    assert!(subject.is_subtype_of(&split));
    // Sound in the other direction: branches that do not cover it decide no.
    assert!(!subject.is_subtype_of(&Schema::union([
        tuple([Schema::Int, Schema::Int]),
        tuple([Schema::Bytes, Schema::Int]),
    ])));

    // A branch of another container kind shares no value with the subject, so
    // it drops out rather than being read as a component-wise cover.
    let list_branches = Schema::union([
        Schema::list(SeqShape::fixed([Schema::Int, Schema::Int])),
        Schema::list(SeqShape::fixed([Schema::Str, Schema::Int])),
    ]);
    assert!(!subject.is_subtype_of(&list_branches));
    // A branch of another arity drops out the same way.
    assert!(!subject.is_subtype_of(&Schema::union([
        Schema::tuple(SeqShape::fixed([Schema::Int])),
        Schema::tuple(SeqShape::fixed([Schema::Str])),
    ])));
    // With no branch of the subject's shape at all there is nothing to split
    // over, and the rule declines rather than deciding on an empty product.
    assert!(!subject.is_subtype_of(&Schema::union([Schema::Int, Schema::Str])));

    // The subject must be a product. A repeated tail admits every length, so
    // there is no tuple of components to split.
    let variadic = Schema::tuple(SeqShape::homogeneous(int_or_str.clone()));
    assert!(!variadic.is_subtype_of(&Schema::union([
        Schema::tuple(SeqShape::homogeneous(Schema::Int)),
        Schema::tuple(SeqShape::homogeneous(Schema::Str)),
    ])));
    // Nor is a branch with a tail a product, so it drops out of the branches.
    // The relation *holds* -- a two-tuple of ints is one int followed by
    // ints -- so this asks the rule, which is what has a scope to pin.
    let with_a_tail = Schema::union([
        Schema::tuple(SeqShape::prefix_tail([Schema::Int], Schema::Int)),
        tuple([Schema::Str, Schema::Int]),
    ]);
    assert!(!by_the_rules(&subject, &with_a_tail));
    assert!(
        subject.is_subtype_of(&with_a_tail),
        "and the descriptor decides it, holding the branch as a language"
    );

    // The empty prefix is the nullary product, and it is covered by itself.
    let nullary = Schema::tuple(SeqShape::fixed([]));
    assert!(nullary.is_subtype_of(&Schema::union([nullary.clone(), Schema::Int])));
}

/// An oracle that kinds a pooled constant by its index and settles a pair of
/// them, standing in for the bindings' reading of two Python objects.
///
/// Index 0 is an `int`, 1 a `str`, 2 a second `int`. Two constants are
/// disjoint when their kinds differ or their indices do -- the same rule the
/// bindings apply to a builtin scalar, whose equality is Python's own.
///
/// It reads classes on the same terms the bindings do: class 0 is laid out as
/// a `list` and holds list values and nothing else, class 1 lays down no
/// layout and is declined -- a subclass of it may derive from a builtin, so
/// its instances are not confined to any kind.
struct Kinded;
impl Constants for Kinded {}

impl LeafRelations for Kinded {
    fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
        None
    }
    fn literal_kind(&self, constant: ConstIx) -> Option<Kind> {
        match constant.get() {
            0 | 2 => Some(Kind::Int),
            1 => Some(Kind::Str),
            _ => None,
        }
    }
    fn class_admits_kind(&self, class: ClassIx, kind: Kind) -> Option<bool> {
        match class.get() {
            0 => Some(kind == Kind::List),
            _ => None,
        }
    }
    fn literals_disjoint(&self, left: ConstIx, right: ConstIx) -> Option<bool> {
        Some(left != right)
    }
}

/// What a class contributes to disjointness, and what it may not.
///
/// A class is the atom the core cannot read, so the oracle answers for it: a
/// class laid out as a builtin holds values of that kind and of no other, and
/// a subclass inherits the layout rather than laying down a second. A class
/// laying down none is a different matter -- a subclass of it may derive from
/// a builtin as well -- and the oracle declines, which must leave the pair
/// undecided rather than refuted.
#[test]
fn a_class_is_disjoint_by_the_layout_its_oracle_reads() {
    let laid_out = Schema::Instance(ClassIx::new(0));
    let no_layout = Schema::Instance(ClassIx::new(1));
    let list_of_int = Schema::list(SeqShape::homogeneous(Schema::Int));

    // Laid out as a list: every other kind is disjoint from it, and its own
    // kind is not.
    for other in [
        Schema::Int,
        Schema::Str,
        Schema::set(Schema::Int),
        Schema::tuple(SeqShape::fixed([Schema::Int])),
        Schema::mapping(MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }),
    ] {
        assert!(
            laid_out.disjoint_with(&other, &Kinded),
            "{other:?} shares no value with a class laid out as a list"
        );
        assert!(other.disjoint_with(&laid_out, &Kinded), "{other:?}");
    }
    assert!(!laid_out.disjoint_with(&list_of_int, &Kinded));
    assert!(!list_of_int.disjoint_with(&laid_out, &Kinded));

    // Laying down no layout: the oracle declines and the pair stays undecided
    // in both directions. A class whose instances a subclass may give a builtin
    // layout confines nothing.
    assert!(!no_layout.disjoint_with(&list_of_int, &Kinded));
    assert!(!list_of_int.disjoint_with(&no_layout, &Kinded));
    assert!(!no_layout.disjoint_with(&Schema::Int, &Kinded));

    // And with no oracle at all, a class says nothing either way.
    assert!(!laid_out.disjoint_with(&Schema::Int, &NoLeafRelations));

    // The relation the reading decides: a container below a class of another
    // layout is refuted, and the subject has values to refute with.
    let relation = |sub: &Schema, sup: &Schema| {
        let budget = Cell::new(DECISION_BUDGET);
        sub.subtype_relation(sup, &Kinded, &[], &budget)
    };
    assert_eq!(relation(&list_of_int, &laid_out), Relation::Unknown);
    assert_eq!(
        relation(&Schema::set(Schema::Int), &laid_out),
        Relation::Fails
    );
    assert_eq!(
        relation(&Schema::set(Schema::Int), &no_layout),
        Relation::Unknown
    );
}

/// What a literal contributes to disjointness, and where each answer comes
/// from. Three rules meet at a literal and they are asked in this order: two
/// literals go to the constants, a literal against anything else goes to the
/// kind, and an unkinded constant declines.
#[test]
fn a_literal_is_disjoint_by_its_constants_then_by_its_kind() {
    let lit = |i: usize| Schema::Literal(ConstIx::new(i));

    // Two literals: the constants settle it, and the kind rule never runs --
    // which matters because the kind rule exempts bool/int and would answer
    // differently for two int constants.
    assert!(lit(0).disjoint_with(&lit(1), &Kinded));
    assert!(lit(0).disjoint_with(&lit(2), &Kinded));
    assert!(!lit(0).disjoint_with(&lit(0), &Kinded));
    // An oracle that declines leaves the pair conservative rather than
    // falling through to a rule that would answer for it.
    assert!(!lit(0).disjoint_with(&lit(1), &NoLeafRelations));

    // A literal against a kind: the constant's kind places it in the
    // partition. Without it the literal is opaque and nothing is decided.
    assert!(lit(0).disjoint_with(&Schema::Str, &Kinded));
    assert!(!lit(0).disjoint_with(&Schema::Int, &Kinded));
    assert!(!lit(0).disjoint_with(&Schema::Str, &NoLeafRelations));
    // Read the other way round too: the arms are ordered, and only one of
    // them puts the literal on the left.
    assert!(Schema::Str.disjoint_with(&lit(0), &Kinded));

    // A union is disjoint from a schema when every member is, which is how a
    // `Literal[...]` -- built as a union of its constants -- is reached at
    // all. One overlapping member is enough to decline.
    let table = Schema::union([lit(0), lit(2)]);
    assert!(table.disjoint_with(&Schema::Str, &Kinded));
    assert!(!table.disjoint_with(&Schema::Int, &Kinded));
    assert!(Schema::Str.disjoint_with(&table, &Kinded));
    assert!(!Schema::Int.disjoint_with(&table, &Kinded));
    // An empty union denotes nothing, so it is disjoint from everything --
    // by the bottom rule above these arms, not by "every member is".
    assert!(Schema::union(Vec::new()).disjoint_with(&Schema::Int, &Kinded));
}

/// An oracle that answers the *set* question, recording what it was handed.
///
/// `literal_sets_disjoint` is the whole-question form of `literals_disjoint`:
/// an implementor that holds its constants settles two unions in one pass,
/// where the member walk asks it once per pair. Which sets it is handed is
/// therefore part of the contract and not an implementation detail, so this
/// records them for a test to read back.
#[derive(Default)]
struct Sets {
    asked: core::cell::RefCell<Vec<(Vec<usize>, Vec<usize>)>>,
}
impl Constants for Sets {}

impl LeafRelations for Sets {
    fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
        None
    }
    fn literals_disjoint(&self, left: ConstIx, right: ConstIx) -> Option<bool> {
        Some(left != right)
    }
    fn literal_sets_disjoint(&self, left: &[ConstIx], right: &[ConstIx]) -> Option<bool> {
        let indices = |set: &[ConstIx]| {
            set.iter()
                .map(|constant| constant.get())
                .collect::<Vec<_>>()
        };
        self.asked
            .borrow_mut()
            .push((indices(left), indices(right)));
        Some(left.iter().all(|constant| !right.contains(constant)))
    }
}

/// Two sets of literals are one question, and both sets reach the oracle whole.
///
/// The member walk asks once per pair, which is quadratic across the binding
/// boundary: two twenty-thousand-member unions were four hundred million calls.
/// So where both sides are nothing but literals the constants are gathered and
/// handed over together. Both halves of that are asserted, because the verdict
/// alone cannot tell the two readings apart -- they agree, which is the point
/// -- and what the optimisation *is* is which question gets asked.
#[test]
fn two_sets_of_literals_are_asked_as_sets() {
    let lit = |i: usize| Schema::Literal(ConstIx::new(i));
    let oracle = Sets::default();

    // A bare literal is a one-constant set, and the verdict is the oracle's.
    assert!(lit(0).disjoint_with(&lit(1), &oracle));
    assert!(!lit(0).disjoint_with(&lit(0), &oracle));
    assert_eq!(
        oracle.asked.take(),
        vec![(vec![0], vec![1]), (vec![0], vec![0])]
    );

    // A union of literals is its members' constants, on either side of the
    // question. `Schema::union` orders its members, so these read in order.
    let table = Schema::union([lit(0), lit(2)]);
    assert!(table.disjoint_with(&Schema::union([lit(1), lit(3)]), &oracle));
    assert!(!table.disjoint_with(&lit(2), &oracle));
    assert_eq!(
        oracle.asked.take(),
        vec![(vec![0, 2], vec![1, 3]), (vec![0, 2], vec![2])]
    );

    // A union carrying a member that is not a literal has no set of constants
    // standing for it, so the question is not asked and the walk stands.
    assert!(!Schema::union([lit(0), Schema::Str]).disjoint_with(&lit(1), &oracle));
    // Nor is it asked of a union with no members. That schema denotes nothing
    // and is disjoint from everything, which the arm below these settles; an
    // empty set of constants would be a different question, and one this oracle
    // would answer `true` for whatever stood against it.
    assert!(!Schema::Union(Vec::new().into()).disjoint_with(&lit(0), &oracle));
    assert!(oracle.asked.take().is_empty());
}

/// An oracle treating each pool index as its own value, so comparing indices
/// orders the bound values they stand for.
struct ByIndex;
impl Constants for ByIndex {}

impl LeafRelations for ByIndex {
    fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
        None
    }
    fn compare(&self, a: OperandIx, b: OperandIx) -> Option<core::cmp::Ordering> {
        Some(a.get().cmp(&b.get()))
    }
}

#[test]
fn constraint_entailment_covers_every_ordering_arm() {
    let o = &ByIndex;
    // Ge(w): a tighter-or-equal lower bound, from Ge or Gt, entails a looser one.
    assert!(constraint_entailed(
        &Constraint::Ge(OperandIx::new(3)),
        &[Constraint::Ge(OperandIx::new(5))],
        o
    ));
    assert!(constraint_entailed(
        &Constraint::Ge(OperandIx::new(3)),
        &[Constraint::Gt(OperandIx::new(5))],
        o
    ));
    assert!(!constraint_entailed(
        &Constraint::Ge(OperandIx::new(5)),
        &[Constraint::Ge(OperandIx::new(3))],
        o
    ));
    // Gt(w): Gt(n) with n >= w, or Ge(n) with n > w.
    assert!(constraint_entailed(
        &Constraint::Gt(OperandIx::new(3)),
        &[Constraint::Gt(OperandIx::new(3))],
        o
    ));
    assert!(constraint_entailed(
        &Constraint::Gt(OperandIx::new(3)),
        &[Constraint::Ge(OperandIx::new(5))],
        o
    ));
    assert!(!constraint_entailed(
        &Constraint::Gt(OperandIx::new(5)),
        &[Constraint::Ge(OperandIx::new(5))],
        o
    ));
    // Le(w): Le(n) or Lt(n) with n <= w.
    assert!(constraint_entailed(
        &Constraint::Le(OperandIx::new(5)),
        &[Constraint::Le(OperandIx::new(3))],
        o
    ));
    assert!(constraint_entailed(
        &Constraint::Le(OperandIx::new(5)),
        &[Constraint::Lt(OperandIx::new(3))],
        o
    ));
    assert!(!constraint_entailed(
        &Constraint::Le(OperandIx::new(3)),
        &[Constraint::Le(OperandIx::new(5))],
        o
    ));
    // Lt(w): Lt(n) with n <= w, or Le(n) with n < w.
    assert!(constraint_entailed(
        &Constraint::Lt(OperandIx::new(5)),
        &[Constraint::Lt(OperandIx::new(5))],
        o
    ));
    assert!(constraint_entailed(
        &Constraint::Lt(OperandIx::new(5)),
        &[Constraint::Le(OperandIx::new(3))],
        o
    ));
    assert!(!constraint_entailed(
        &Constraint::Lt(OperandIx::new(5)),
        &[Constraint::Le(OperandIx::new(5))],
        o
    ));
    // Length bounds compare by their raw counts, no oracle needed.
    assert!(constraint_entailed(
        &Constraint::MinLen(3),
        &[Constraint::MinLen(5)],
        o
    ));
    assert!(!constraint_entailed(
        &Constraint::MinLen(5),
        &[Constraint::MinLen(3)],
        o
    ));
    assert!(constraint_entailed(
        &Constraint::MaxLen(5),
        &[Constraint::MaxLen(3)],
        o
    ));
    assert!(!constraint_entailed(
        &Constraint::MaxLen(3),
        &[Constraint::MaxLen(5)],
        o
    ));
    // A multiple-of or predicate bound has no order entailment.
    assert!(!constraint_entailed(
        &Constraint::MultipleOf(OperandIx::new(0)),
        &[Constraint::MultipleOf(OperandIx::new(0))],
        o
    ));
}

/// An oracle that orders pool indices as the values they stand for and reports
/// integer adjacency by the same arithmetic the binding uses, so the
/// discreteness rule can be driven without an interpreter.
struct Adjacent;
impl Constants for Adjacent {}

impl LeafRelations for Adjacent {
    fn leaf_subtype(&self, _: &Schema, _: &Schema) -> Option<bool> {
        None
    }
    fn compare(&self, a: OperandIx, b: OperandIx) -> Option<core::cmp::Ordering> {
        Some(a.get().cmp(&b.get()))
    }
    fn no_int_between(
        &self,
        lo: OperandIx,
        lo_strict: bool,
        hi: OperandIx,
        hi_strict: bool,
    ) -> Option<bool> {
        let least = lo.get() + usize::from(lo_strict);
        let greatest = hi.get().checked_sub(usize::from(hi_strict))?;
        Some(least > greatest)
    }
}

/// One set spelled two ways gets one verdict. The bounds of a refinement and
/// the bounds gathered across an intersection are the same conjunction, so a
/// rule that fires for one fires for the other: an intersection is a subset of
/// every member, and a member bounded to the integers bounds the meet.
#[test]
fn both_meets_ask_the_same_question_of_their_base() {
    let refine = |constraints| Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints,
    };
    let (gt0, lt1) = (
        Constraint::Gt(OperandIx::new(0)),
        Constraint::Lt(OperandIx::new(1)),
    );
    let on_one = refine(vec![gt0.clone(), lt1.clone()].into());
    let across = Schema::meet([refine(vec![gt0].into()), refine(vec![lt1].into())]);
    assert!(on_one.is_empty_with(&Adjacent, &[]));
    assert_eq!(
        on_one.is_empty_with(&Adjacent, &[]),
        across.is_empty_with(&Adjacent, &[])
    );
}

/// A dataclass schema is the meet of an `isinstance` atom and an attribute
/// record, and each half is reachable through the meet: it is below its own
/// class by the conjunct rule, and below a wider record by record inclusion.
/// One node holding both halves could do neither -- the pair only ever
/// related to another pair.
#[test]
fn each_half_of_an_attribute_schema_is_reachable_through_the_meet() {
    let record = |schema| Schema::AttrRecord {
        fields: vec![Field {
            name: "a".into(),
            schema,
            required: true,
        }]
        .into(),
    };
    let object = Schema::meet([Schema::Instance(ClassIx::new(0)), record(Schema::Bool)]);
    assert!(object.is_subtype_of(&Schema::Instance(ClassIx::new(0))));
    // A different class is a nominal question, and the core's default oracle
    // decides nothing, so it stays conservative.
    assert!(!object.is_subtype_of(&Schema::Instance(ClassIx::new(1))));
    // The record half relates on its own, in both directions: `bool` is below
    // `int`, so the narrower attribute is the subtype.
    assert!(object.is_subtype_of(&record(Schema::Int)));
    assert!(!record(Schema::Int).is_subtype_of(&record(Schema::Bool)));
}

/// An attribute record carries no class, so its fields decide inhabitation
/// as well as emptiness: an object carrying one witness per attribute is a
/// value of it. The class half is the opaque one, and it is now a separate
/// conjunct that only the meet is unknown about.
#[test]
fn an_attribute_record_is_inhabited_by_its_fields() {
    let record = |schema| Schema::AttrRecord {
        fields: vec![Field {
            name: "a".into(),
            schema,
            required: true,
        }]
        .into(),
    };
    assert_eq!(record(Schema::Int).verdict(), Verdict::Inhabited);
    assert_eq!(record(Schema::Nothing).verdict(), Verdict::Empty);
    // An optional field admitting nothing does not empty it: a value that
    // does not carry the attribute is still a value of the record.
    let optional = Schema::AttrRecord {
        fields: vec![Field {
            name: "a".into(),
            schema: Schema::Nothing,
            required: false,
        }]
        .into(),
    };
    assert_eq!(optional.verdict(), Verdict::Inhabited);
    // Meeting it with a class is unknown in the other direction only: the
    // class may have no instances, and that is what the core cannot read.
    let object = Schema::meet([Schema::Instance(ClassIx::new(0)), record(Schema::Int)]);
    assert_eq!(object.verdict(), Verdict::Unknown);
    assert_eq!(
        Schema::meet([Schema::Instance(ClassIx::new(0)), record(Schema::Nothing)]).verdict(),
        Verdict::Empty
    );
}

/// A boolean base is bounded to the integers, so it counts them too. Sound but
/// not complete: the rule sees the integers in the interval, not the two
/// values `bool` actually has, so an interval holding an integer that is
/// neither 0 nor 1 stays conservatively non-empty.
#[test]
fn a_boolean_base_counts_integers() {
    let refine = |base, constraints| Schema::Refine {
        base: Arc::new(base),
        constraints,
    };
    let open_unit = vec![
        Constraint::Gt(OperandIx::new(0)),
        Constraint::Lt(OperandIx::new(1)),
    ];
    assert!(refine(Schema::Bool, open_unit.into()).is_empty_with(&Adjacent, &[]));
    // A dense base is not bounded to the integers and stays inhabited.
    let dense = vec![
        Constraint::Gt(OperandIx::new(0)),
        Constraint::Lt(OperandIx::new(1)),
    ];
    assert!(!refine(Schema::Float, dense.into()).is_empty_with(&Adjacent, &[]));
}

#[test]
fn tighter_refinement_bounds_subtype_looser_ones_through_the_oracle() {
    // The entailment feeds the refinement subtype rule: a tighter bound makes a
    // refinement a subtype of a refinement with a looser one, even when the
    // constraints are not identical (so the verbatim path does not apply).
    let refine = |constraints| Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints,
    };
    let tight = refine(vec![Constraint::Ge(OperandIx::new(5))].into());
    let loose = refine(vec![Constraint::Ge(OperandIx::new(3))].into());
    assert!(tight.is_subtype_of_under(&loose, &ByIndex, &[]));
    assert!(!loose.is_subtype_of_under(&tight, &ByIndex, &[]));
}

#[test]
fn scalar_type_tags_decide_disjointness() {
    // Distinct concrete scalars are provably disjoint; bool is a subtype of
    // int, so the two overlap. This pins each scalar's own type tag: dropping
    // one would make its disjointness with every other scalar undecidable.
    assert!(Schema::Bool.disjoint(&Schema::Str));
    assert!(Schema::Int.disjoint(&Schema::Str));
    assert!(Schema::Float.disjoint(&Schema::Int));
    assert!(Schema::Bytes.disjoint(&Schema::Str));
    assert!(!Schema::Bool.disjoint(&Schema::Int));
    assert!(!Schema::Int.disjoint(&Schema::Int));
}

#[test]
fn equal_bounds_keep_the_strict_end_when_narrowing() {
    // Narrowing two equal lower bounds keeps the strict one: Ge(5) ∩ Gt(5) is
    // Gt(5), so Ge(5) ∩ Gt(5) ∩ Le(5) is x > 5 ∧ x <= 5 — empty. If the strict
    // ends were combined the other way (both-strict rather than either-strict)
    // the lower bound would relax to Ge(5) and the range {5} would look
    // inhabited, so this pins the strictness combination.
    let refine = |constraints| Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints,
    };
    let empty = Schema::Intersection(
        vec![
            refine(vec![Constraint::Ge(OperandIx::new(5))].into()),
            refine(vec![Constraint::Gt(OperandIx::new(5))].into()),
            refine(vec![Constraint::Le(OperandIx::new(5))].into()),
        ]
        .into(),
    );
    assert!(empty.is_empty_with(&ByIndex, &[]));
    // Both bounds non-strict: the singleton {5} is inhabited.
    let inhabited = Schema::Intersection(
        vec![
            refine(vec![Constraint::Ge(OperandIx::new(5))].into()),
            refine(vec![Constraint::Le(OperandIx::new(5))].into()),
        ]
        .into(),
    );
    assert!(!inhabited.is_empty_with(&ByIndex, &[]));
}

#[test]
fn a_union_of_disjoint_complements_simplifies_to_the_top() {
    // De Morgan: ¬A ∪ ¬B = ¬(A ∩ B), which is ⊤ when A and B are disjoint. int
    // and str are disjoint, so their complements cover the universe.
    let disjoint = Schema::Union(
        vec![
            Schema::Complement(Arc::new(Schema::Int)),
            Schema::Complement(Arc::new(Schema::Str)),
        ]
        .into(),
    );
    assert_eq!(disjoint.simplify(), Schema::ANYTHING);
    // bool is a subtype of int, so int and bool overlap and their complements
    // do not cover the universe.
    let overlapping = Schema::Union(
        vec![
            Schema::Complement(Arc::new(Schema::Int)),
            Schema::Complement(Arc::new(Schema::Bool)),
        ]
        .into(),
    );
    assert_ne!(overlapping.simplify(), Schema::ANYTHING);
}

/// A union member that is proven empty but whose region is opaque does not
/// end the fold: the stop is on "not proven empty", and an empty member with
/// an unknown region is neither absorbing nor a verdict on the union.
#[test]
fn an_empty_member_with_an_opaque_region_does_not_decide_the_union() {
    // A refinement over an empty base is empty with an unknown region.
    let emptied = Schema::Refine {
        base: Arc::new(Schema::Intersection(vec![Schema::Int, Schema::Str].into())),
        constraints: vec![Constraint::MinLen(1)].into(),
    };
    assert!(emptied.is_empty());
    let union = Schema::Union(vec![emptied, Schema::Str].into());
    assert!(!union.is_empty());
}

/// A reference on the left is unfolded before the arms that read the right
/// side's shape: against a refinement, whose arm reads the subject's own
/// constraints, a reference is decided through its body rather than declined.
#[test]
fn a_reference_on_the_left_is_read_through_its_body_against_any_right_side() {
    let narrowed = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::Ge(OperandIx::new(0))].into(),
    };
    let defs = vec![narrowed.clone()];
    let reference = Schema::Ref(DefIx::new(0));
    assert!(reference.is_subtype_of_under(&narrowed, &NoLeafRelations, &defs));
    // And through nothing else: a bound the body does not carry is not entailed.
    let other = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::Gt(OperandIx::new(0))].into(),
    };
    assert!(!reference.is_subtype_of_under(&other, &NoLeafRelations, &defs));
}

/// An oracle that claims every relation against a union, standing for the one
/// question the union rule may ask it: whether an *instance* is below the union.
struct BelowAnyUnion;

impl Constants for BelowAnyUnion {}

impl LeafRelations for BelowAnyUnion {
    fn leaf_subtype(&self, _sub: &Schema, sup: &Schema) -> Option<bool> {
        matches!(sup, Schema::Union(_)).then_some(true)
    }
}

/// The union rule asks the oracle about an `Instance` subject and about
/// nothing else: a subject the rules already declined is not handed to an
/// oracle that would say yes.
#[test]
fn the_union_rule_asks_the_oracle_only_about_an_instance() {
    let union = Schema::union([Schema::Str, Schema::Bytes]);
    // A subject the region partition cannot settle, so the union rule is the
    // one that answers.
    let narrowed = Schema::Refine {
        base: Arc::new(Schema::Int),
        constraints: vec![Constraint::MinLen(1)].into(),
    };
    assert!(!narrowed.is_subtype_of_under(&union, &BelowAnyUnion, &[]));
    assert!(Schema::Instance(ClassIx::new(0)).is_subtype_of_under(&union, &BelowAnyUnion, &[]));
}

/// A catch-all covers an *optional* field of the supertype, and only an
/// optional one.
///
/// A clause guarantees what a key's value must be if the key is there; it
/// guarantees nothing about the key being there at all. So a mapping is below a
/// record that names an extra *optional* field over the same clause -- every
/// value either lacks that key or carries a value the field admits -- and is
/// not below the same record with the field required, since the mapping admits
/// the value that omits it. Both directions, because a rule that answered the
/// same for the two would be wrong on one of them.
#[test]
fn a_catch_all_covers_an_optional_field_and_not_a_required_one() {
    let clause = || MapClause {
        key: Schema::Str,
        value: Schema::Int,
    };
    let mapping = Schema::mapping(clause());
    let with_field = |required| {
        Schema::keyed_map(
            vec![Field {
                name: "a".into(),
                schema: Schema::Int,
                required,
            }],
            vec![clause()],
        )
    };
    assert!(structural(&mapping, &with_field(false)));
    assert!(!structural(&mapping, &with_field(true)));
    // And the value the second one is about: a mapping admits the empty dict,
    // which a map requiring a key does not.
    assert!(!Schema::meet([mapping, with_field(true).complement()]).is_empty());
}

/// A meet with a recursive schema is decided by unfolding it once: the body
/// names the kinds it admits, and a kind it never admits is disjoint from it.
#[test]
fn a_meet_with_a_recursive_schema_is_decided_by_one_unfolding() {
    let defs = vec![Schema::union([
        Schema::Int,
        Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(0)))),
    ])];
    let meet = Schema::Intersection(vec![Schema::Ref(DefIx::new(0)), Schema::Bytes].into());
    assert!(meet.is_empty_under(&defs));
    // And the unfolding is sound in the other direction: a kind the body does
    // admit is not proven disjoint.
    let meet = Schema::Intersection(vec![Schema::Ref(DefIx::new(0)), Schema::Int].into());
    assert!(!meet.is_empty_under(&defs));
}

/// What the rules refute, prove, and decline, enumerated.
///
/// The three-valued answer is only worth its plumbing if the three values can
/// be told apart, and the way a conservative procedure rots is by quietly
/// turning a decline into a refutation -- which no `bool` test can see, because
/// both come out `false` at the boundary. Each row below names a pair and the
/// answer the rules must give it; a rule that starts claiming a proof it does
/// not have moves a row from `Unknown` to `Fails` and fails here.
#[test]
fn a_subject_with_no_value_is_below_a_shape_it_cannot_match() {
    // A rule refutes from a mismatch, and a mismatch is a witness only when the
    // subject has a value to offer. A subject with none is below every set,
    // including one whose shape it could never take.
    let relation = |sub: &Schema, sup: &Schema| {
        let budget = Cell::new(DECISION_BUDGET);
        sub.subtype_relation(sup, &NoLeafRelations, &[], &budget)
    };
    let empty_list = Schema::list(SeqShape::fixed([]));

    // A list of at most zero elements *is* the empty list, and its base is
    // nowhere near it. Reading the refinement through its base reaches the arity
    // rule, which refutes what the constraint has already settled -- so the base
    // is read for its proof alone and the rule declines instead.
    let bounded = Schema::refine(
        Schema::list(SeqShape::homogeneous(Schema::Int)),
        vec![Constraint::MaxLen(0)],
    );
    assert_eq!(relation(&bounded, &empty_list), Relation::Unknown);
    assert!(bounded.is_subtype_of(&empty_list));
    assert!(bounded.is_equivalent(&empty_list));

    // And the mismatch that decides the other way: a fixed sequence whose first
    // position admits no value is empty, so it is below the shape it cannot
    // match. The position is a length bound over a container of nothing, which
    // the fold reads -- a list of at least one element of a set with none has
    // none -- so the arity rule's refutation is read against a subject proven
    // empty and establishes the opposite.
    let unfillable = Schema::list(SeqShape::fixed([
        Schema::refine(
            Schema::list(SeqShape::homogeneous(Schema::Nothing)),
            vec![Constraint::MinLen(1)],
        ),
        Schema::ANYTHING,
    ]));
    assert_eq!(relation(&unfillable, &empty_list), Relation::Holds);
    assert!(unfillable.is_subtype_of(&empty_list));

    // A repeated tail with no value is not a repeat. A list of that same
    // refinement is inhabited -- the empty list is in it whatever the element
    // admits -- so the subject offers a value; but the refutation "a tail
    // repeats past a fixed length" stands on the *element* having one, and this
    // element has none. The tail is dropped and the subject is its prefix,
    // which is the empty list.
    let unrepeating = Schema::list(SeqShape::homogeneous(Schema::refine(
        Schema::list(SeqShape::homogeneous(Schema::Nothing)),
        vec![Constraint::MinLen(1)],
    )));
    assert_eq!(relation(&unrepeating, &empty_list), Relation::Holds);
    assert!(unrepeating.is_subtype_of(&empty_list));
    assert!(unrepeating.is_equivalent(&empty_list));
}

/// A length bound over a container that repeats one element is read for the
/// values it has, not only for the values it cannot have.
///
/// A bound is satisfiable in the abstract and still empty over its base, which
/// is why a refinement is unknown in general. A container that repeats one
/// element is the exception the bound is written for: a value of any length is
/// as many copies of one element, so the element decides it. That is what
/// closes the fixpoint whose every unfolding needs one more element.
#[test]
fn a_length_bound_over_a_repeated_element_is_decided_by_the_element() {
    let bounded = |base: Schema, min: usize| Schema::refine(base, vec![Constraint::MinLen(min)]);
    let ints = Schema::list(SeqShape::homogeneous(Schema::Int));
    let nothings = Schema::list(SeqShape::homogeneous(Schema::Nothing));
    // An element with values: any length is built from it.
    assert_eq!(bounded(ints.clone(), 2).verdict(), Verdict::Inhabited);
    assert_eq!(
        bounded(Schema::set(Schema::Int), 2).verdict(),
        Verdict::Inhabited
    );
    // An element with none: only the empty container is left, and a bound above
    // zero rules it out.
    assert_eq!(bounded(nothings.clone(), 1).verdict(), Verdict::Empty);
    // A bound of zero is met by the empty container whatever the element says.
    assert_eq!(bounded(nothings.clone(), 0).verdict(), Verdict::Inhabited);
    assert_eq!(
        Schema::refine(nothings.clone(), vec![Constraint::MaxLen(3)]).verdict(),
        Verdict::Inhabited
    );
    // A fixed position is not a repeated element: the bound says nothing about
    // what fills it, and the reading declines rather than guessing. The second
    // row is the one that matters -- a shape with a prefix *and* a tail, whose
    // tail has values and whose one fixed position may have none, so reading
    // the tail alone would claim a value the shape may not have.
    assert_eq!(
        bounded(Schema::tuple(SeqShape::fixed([Schema::Int])), 1).verdict(),
        Verdict::Unknown
    );
    assert_eq!(
        bounded(
            Schema::list(SeqShape::prefix_tail(
                [Schema::Instance(ClassIx::new(0))],
                Schema::Int,
            )),
            1,
        )
        .verdict(),
        Verdict::Unknown
    );
    // Nor is a constraint that is not a length: a predicate may refuse every
    // value of the base.
    assert_eq!(
        Schema::refine(
            ints.clone(),
            vec![Constraint::MinLen(1), Constraint::MaxLen(9)]
        )
        .verdict(),
        Verdict::Inhabited
    );
    assert_eq!(
        Schema::refine(ints, vec![Constraint::Predicate(PredIx::new(0))]).verdict(),
        Verdict::Unknown
    );
}

/// What the shallow reading of a value may and may not say.
///
/// The reading answers from a schema's own form, so its boundary is where the
/// form stops deciding: a container that admits an empty one has a value
/// whatever its elements are, and one that must be filled does not. A reading
/// that widened to the whole variant -- every sequence, every record -- would
/// claim a value for a schema that has none, which is the claim a refutation
/// stands on. The search over the value corpus in `laws.rs` makes the same
/// claim over generated schemas; these are the rows that pin the edge.
#[test]
fn the_shallow_reading_answers_for_a_form_and_declines_the_rest() {
    let nothing = Schema::Nothing;
    let has_a_value = [
        Schema::Int,
        Schema::Str,
        Schema::ANYTHING,
        // Every container that admits an empty one: the value is the empty
        // list, tuple, set or mapping, whatever the elements say.
        Schema::list(SeqShape::homogeneous(nothing.clone())),
        Schema::tuple(SeqShape::homogeneous(nothing.clone())),
        Schema::set(nothing.clone()),
        open(vec![field("f", nothing.clone(), false)]),
        closed(vec![]),
        // A union has a value where a member does. Written as the node: the
        // constructor drops a member it can see is empty, and the member this
        // row needs is one only a descent would see is empty.
        Schema::Union(
            vec![
                Schema::list(SeqShape::fixed([nothing.clone()])),
                Schema::Int,
            ]
            .into(),
        ),
    ];
    for schema in has_a_value {
        assert!(
            schema.holds_a_value_shallowly(),
            "{schema:?} has a value by its form"
        );
        assert_ne!(schema.verdict(), Verdict::Empty, "{schema:?}");
    }
    let declines = [
        // A position that must be filled, and a key that must be there: the
        // form does not say, and the descent does.
        Schema::list(SeqShape::fixed([nothing.clone()])),
        Schema::tuple(SeqShape::fixed([Schema::Int])),
        closed(vec![field("f", nothing.clone(), true)]),
        open(vec![field("f", nothing.clone(), true)]),
        // A union of members that do not answer either -- written as the node,
        // since the constructor collapses a union of the empty set to it.
        Schema::Union(
            vec![
                Schema::list(SeqShape::fixed([nothing.clone()])),
                closed(vec![field("f", nothing.clone(), true)]),
            ]
            .into(),
        ),
        // And the shapes with no form of their own to read.
        nothing.clone(),
        Schema::Complement(Arc::new(Schema::Int)),
        Schema::refine(Schema::Int, vec![Constraint::MinLen(1)]),
    ];
    for schema in declines {
        assert!(
            !schema.holds_a_value_shallowly(),
            "{schema:?} is not answered by its form"
        );
    }
}

/// A refutation about a part is about that part's values, and a part with none
/// refutes nothing.
///
/// The subject of a comparison one level down is a part of the subject above
/// it, and the composition that carries its refutation up is only as good as
/// the values it stands on: a list of an element with no value is the empty
/// list, which is below a list of anything, however the element compares. The
/// reading that settles this is the subject's own emptiness, taken at the level
/// the refutation is made rather than once at the top -- where the list is
/// inhabited by the empty list and says nothing about its element.
#[test]
fn a_refutation_about_a_part_with_no_value_is_not_one() {
    // A part that is empty, and that the rules cannot prove empty: two
    // sequences of one position each whose positions share no value. No rule
    // reads a meet of two shapes for the values it holds, so proving it empty
    // takes the descriptor -- which is what makes this the case the guard is
    // for.
    let empty = Schema::meet([
        Schema::tuple(SeqShape::fixed([Schema::Int])),
        Schema::tuple(SeqShape::fixed([Schema::Str])),
    ]);
    let narrow = closed(vec![field("f", empty.clone(), true)]);
    let wide = closed(vec![
        field("f", empty.clone(), true),
        field("g", Schema::Int, true),
    ]);
    let cases: Vec<(&str, Schema, Schema)> = vec![
        // The part itself: the subject has no value, so the refutation its
        // fields report is about nothing and the inclusion holds vacuously.
        ("the part itself", narrow.clone(), wide.clone()),
        // The same part below each container that carries an element's
        // refutation up. Every one of these subjects has a value -- the empty
        // list, the empty set, the mapping with no keys -- and none of those
        // values is outside the supertype.
        (
            "in a list",
            Schema::list(SeqShape::homogeneous(narrow.clone())),
            Schema::list(SeqShape::homogeneous(wide.clone())),
        ),
        (
            "in a tuple's repeated tail",
            Schema::tuple(SeqShape::homogeneous(narrow.clone())),
            Schema::tuple(SeqShape::homogeneous(wide.clone())),
        ),
        (
            "in a set",
            Schema::set(narrow.clone()),
            Schema::set(wide.clone()),
        ),
        // A record whose field is optional: the record holds the mapping with
        // no keys whatever the field's schema admits, so the field's
        // refutation is about values the record need not have.
        (
            "under an optional field",
            closed(vec![field("k", narrow.clone(), false)]),
            closed(vec![field("k", wide.clone(), false)]),
        ),
    ];
    for (label, sub, sup) in cases {
        let budget = Cell::new(DECISION_BUDGET);
        assert_ne!(
            sub.subtype_relation(&sup, &NoLeafRelations, &[], &budget),
            Relation::Fails,
            "{label}: the rules refuted an inclusion that holds"
        );
        assert_eq!(
            sub.descriptor_contained_in(&sup, &NoLeafRelations, &[]),
            Relation::Holds,
            "{label}: the sets decide it"
        );
        assert!(sub.is_subtype_of(&sup), "{label}");
    }

    // One level further down, where neither decider reaches the answer: the
    // rules decline, as above, and the sets decline too -- lowering a record
    // inside a list inside a list is past what the set representation builds.
    // The pair is unproven, which the relation is allowed to be; what it may
    // not be is refuted.
    let deeper = |record: &Schema| {
        Schema::list(SeqShape::homogeneous(closed(vec![field(
            "k",
            record.clone(),
            true,
        )])))
    };
    let budget = Cell::new(DECISION_BUDGET);
    assert_eq!(
        deeper(&narrow).subtype_relation(&deeper(&wide), &NoLeafRelations, &[], &budget),
        Relation::Unknown,
    );
    assert_eq!(
        deeper(&narrow).descriptor_contained_in(&deeper(&wide), &NoLeafRelations, &[]),
        Relation::Unknown,
    );
}

/// A key the supertype requires and the subject does not declare refutes the
/// inclusion, whatever the subject's clauses say.
///
/// A clause governs the keys a value carries and never requires one, so a
/// subject admits a value without such a key however open it is: take a value
/// of the subject and drop the key, and every required field is still there
/// and every key left is one a clause already covered. That value is one the
/// supertype rejects. The reading around the rule is what keeps it honest: a
/// subject with no value at all is below every schema, this one included.
#[test]
fn a_required_key_the_subject_does_not_declare_refutes_the_inclusion() {
    let relation = |sub: &Schema, sup: &Schema| {
        let budget = Cell::new(DECISION_BUDGET);
        sub.subtype_relation(sup, &NoLeafRelations, &[], &budget)
    };
    let field = |name: &str, required: bool| Field {
        name: name.into(),
        schema: Schema::Int,
        required,
    };
    let complete = Schema::record(vec![field("f0", true), field("f1", true)], Openness::Closed);
    let closed_without = Schema::record(vec![field("f1", true)], Openness::Closed);
    let open_without = Schema::record(vec![field("f1", true)], Openness::Open);
    assert_eq!(relation(&closed_without, &complete), Relation::Fails);
    // The open subject refutes for the same reason: its clause admits a key it
    // does not declare and requires none, so the value without that key is one
    // it has and the supertype rejects.
    assert_eq!(relation(&open_without, &complete), Relation::Fails);

    // The reading that keeps the refutation honest: a subject whose own
    // required field admits nothing has no value to stand against the
    // inclusion, so the same missing key decides the other way.
    let empty_subject = Schema::record(
        vec![Field {
            name: "f1".into(),
            schema: Schema::Nothing,
            required: true,
        }],
        Openness::Closed,
    );
    assert_eq!(relation(&empty_subject, &complete), Relation::Holds);
}

/// A pair whose kinds cannot overlap is refuted, with nothing else read.
///
/// The shapes below have no structural rule between them -- a list is not
/// compared to a tuple position by position, a mapping not to either -- so the
/// pair reaches the end of the match, where disjointness answers it: every
/// value of the subject is outside the supertype. Without that reading the
/// pair went unproven and the set representation was asked to lower both
/// sides.
#[test]
fn a_pair_that_shares_no_value_is_refuted() {
    let relation = |sub: &Schema, sup: &Schema| {
        let budget = Cell::new(DECISION_BUDGET);
        sub.subtype_relation(sup, &NoLeafRelations, &[], &budget)
    };
    let list_of_int = Schema::list(SeqShape::homogeneous(Schema::Int));
    let tuple_of_two = Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Int]));
    let mapping = Schema::mapping(MapClause {
        key: Schema::Str,
        value: Schema::Int,
    });
    for (sub, sup) in [
        (&list_of_int, &tuple_of_two),
        (&list_of_int, &mapping),
        (&mapping, &list_of_int),
        (&Schema::set(Schema::Int), &list_of_int),
    ] {
        assert_eq!(relation(sub, sup), Relation::Fails, "{sub:?} <= {sup:?}");
    }
    // The reading that keeps it honest: a subject of one kind with no value is
    // below the other kind, not outside it.
    let empty_list = Schema::list(SeqShape::fixed([Schema::Nothing]));
    assert_eq!(relation(&empty_list, &mapping), Relation::Holds);
    // And a pair the rule must not reach: two lists whose elements differ have
    // a rule of their own, and it refutes on the elements rather than on the
    // kinds -- which are the same.
    assert_eq!(
        relation(
            &list_of_int,
            &Schema::list(SeqShape::homogeneous(Schema::Str))
        ),
        Relation::Fails
    );
}

#[test]
fn the_rules_refute_prove_and_decline_these() {
    let relation = |sub: &Schema, sup: &Schema| {
        let budget = Cell::new(DECISION_BUDGET);
        sub.subtype_relation(sup, &NoLeafRelations, &[], &budget)
    };
    let attr = |name: &str, schema: Schema, required: bool| Schema::AttrRecord {
        fields: vec![Field {
            name: name.into(),
            schema,
            required,
        }]
        .into(),
    };

    // Proven: the region partition, the lattice bounds, reflexivity.
    assert_eq!(relation(&Schema::Bool, &Schema::Int), Relation::Holds);
    assert_eq!(relation(&Schema::Nothing, &Schema::Str), Relation::Holds);
    assert_eq!(relation(&Schema::Str, &Schema::ANYTHING), Relation::Holds);

    // Refuted: two scalars whose regions are disjoint, an arity that cannot
    // match, an attribute the subject does not carry, and one it carries only
    // sometimes.
    assert_eq!(relation(&Schema::Str, &Schema::Int), Relation::Fails);
    assert_eq!(
        relation(
            &Schema::tuple(SeqShape::fixed([Schema::Int])),
            &Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Int])),
        ),
        Relation::Fails
    );
    assert_eq!(
        relation(&attr("a", Schema::Int, true), &attr("b", Schema::Int, true),),
        Relation::Fails
    );
    assert_eq!(
        relation(
            &attr("a", Schema::Int, false),
            &attr("a", Schema::Int, true),
        ),
        Relation::Fails
    );

    // A length that cannot match refutes whatever the elements are *provided
    // the subject has a value*: the refutation is a value of the subject with
    // the wrong number of positions, and a subject that admits none offers no
    // such value. The pair below carries an opaque class no oracle here can
    // decide inhabited, so the arity rule declines. The same pair with elements
    // the core can read refutes on the shapes alone.
    assert_eq!(
        relation(
            &Schema::tuple(SeqShape::fixed([
                Schema::Instance(ClassIx::new(0)),
                Schema::Int,
            ])),
            &Schema::tuple(SeqShape::fixed([Schema::Instance(ClassIx::new(1))])),
        ),
        Relation::Unknown
    );
    assert_eq!(
        relation(
            &Schema::tuple(SeqShape::fixed([Schema::Str, Schema::Int])),
            &Schema::tuple(SeqShape::fixed([Schema::Str])),
        ),
        Relation::Fails
    );
    // The lengths decide before the elements are asked, and the pair below is
    // what says so: a subject the core can read against a supertype whose one
    // position it cannot relate to anything. Comparing position by position
    // would meet the opaque class first and decline; the arities settle it
    // without looking, and the subject has values, so it refutes.
    assert_eq!(
        relation(
            &Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Int])),
            &Schema::tuple(SeqShape::fixed([Schema::Instance(ClassIx::new(1))])),
        ),
        Relation::Fails
    );

    // Declined: a leaf pair only an oracle can relate, and a class beside a
    // literal. `NoLeafRelations` answers neither, and the rules say so rather
    // than reporting a refutation they have not earned.
    assert_eq!(
        relation(
            &Schema::Instance(ClassIx::new(0)),
            &Schema::Instance(ClassIx::new(1)),
        ),
        Relation::Unknown
    );
    assert_eq!(
        relation(&Schema::Literal(ConstIx::new(0)), &Schema::Int),
        Relation::Unknown
    );
    // A required key the supertype declares and the subject's catch-all cannot
    // guarantee present: refuted, because a clause never requires a key. The
    // mapping holds the value with no keys at all, and the record rejects it.
    assert_eq!(
        relation(
            &Schema::mapping(MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }),
            &Schema::record(
                vec![Field {
                    name: "a".into(),
                    schema: Schema::Int,
                    required: true,
                }],
                Openness::Closed,
            ),
        ),
        Relation::Fails
    );
}

/// A subject disjoint from a meet is below that meet's complement.
///
/// `A <= ~B` and `A & B = {}` are one question asked two ways, and the rules
/// answered them differently: the meet this rule builds held `B` as a nested
/// intersection, and the emptiness rule that decides most meets compares the
/// members of *one* intersection pairwise, so a tuple and the dict inside the
/// nested meet were never compared. Built through the meet constructor, the
/// members are flattened into one list and the pair meets.
#[test]
fn a_subject_disjoint_from_a_meet_is_below_its_complement() {
    let record = Schema::record(
        vec![Field {
            name: "a".into(),
            schema: Schema::Int,
            required: true,
        }],
        Openness::Closed,
    );
    let dicts = Schema::mapping(MapClause {
        key: Schema::Str,
        value: Schema::Int,
    });
    let inner = Schema::meet([dicts, record.complement()]);
    let tuples = Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Str]));
    // The meet is empty, so the inclusion holds; both readings must say so.
    assert!(Schema::meet([tuples.clone(), inner.clone()]).is_empty());
    assert!(tuples.is_subtype_of(&inner.complement()));
}

/// A pool where the index is the value, except that two indices hold one.
///
/// The duplicate is the whole point. An oracle whose every index holds a
/// different value cannot tell a rule that reads a *set of indices* from one
/// that reads a set of **values**, so the difference the literal oracle exists
/// for is invisible to it -- and a rule refuting a membership by index alone
/// would pass every property written over such a pool.
struct Values;

impl Values {
    /// The value at an index: its own number, except that the last index of the
    /// three below repeats the first.
    fn at(index: ConstIx) -> Option<usize> {
        match index.get() {
            0..=2 => Some(index.get()),
            3 => Some(0),
            _ => None,
        }
    }
}

impl Constants for Values {}

impl LeafRelations for Values {
    fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
        None
    }

    fn literals_disjoint(&self, left: ConstIx, right: ConstIx) -> Option<bool> {
        Some(Values::at(left)? != Values::at(right)?)
    }

    fn literal_sets_disjoint(&self, left: &[ConstIx], right: &[ConstIx]) -> Option<bool> {
        let read = |set: &[ConstIx]| {
            set.iter()
                .map(|index| Values::at(*index))
                .collect::<Option<Vec<_>>>()
        };
        let (left, right) = (read(left)?, read(right)?);
        Some(!left.iter().any(|value| right.contains(value)))
    }
}

/// A table of literals against another, decided as the two sets they denote.
///
/// The relation asked here is the rules' alone: the descriptor decides these
/// too, on a table small enough for it to hold, and asking through the public
/// relation would report a right answer for the other reason.
fn table_relation(subject: &[usize], supertype: &[usize], oracle: &dyn LeafRelations) -> Relation {
    let table = |indices: &[usize]| {
        Schema::union(
            indices
                .iter()
                .map(|index| Schema::Literal(ConstIx::new(*index))),
        )
    };
    let (subject, supertype) = (table(subject), table(supertype));
    let budget = Cell::new(DECISION_BUDGET);
    subject.subtype_relation(&supertype, oracle, &[], &budget)
}

/// Every constant found is a proof, and one found nowhere is a refutation.
///
/// The refutation is what a shape rule cannot give: a union of literals is a
/// *finite set*, so a constant of the subject that is in no member of the
/// supertype is a value in one and outside the other, which is the whole of
/// what refuting an inclusion means.
#[test]
fn a_table_of_literals_is_decided_as_the_set_it_denotes() {
    assert_eq!(
        table_relation(&[0, 1], &[0, 1, 2], &Values),
        Relation::Holds
    );
    assert_eq!(table_relation(&[0, 1], &[0, 1], &Values), Relation::Holds);
    assert_eq!(table_relation(&[0, 2], &[0, 1], &Values), Relation::Fails);
    assert_eq!(table_relation(&[2], &[0, 1], &Values), Relation::Fails);
}

/// A constant the subject holds and the supertype spells at another index is
/// not missing, and the rule that reads the two lists by index does not say it
/// is. Index three holds the value of index zero.
#[test]
fn a_constant_spelled_at_another_index_is_not_a_refutation() {
    assert_ne!(table_relation(&[3], &[0, 1], &Values), Relation::Fails);
    assert_ne!(table_relation(&[0, 3], &[0, 1], &Values), Relation::Fails);
}

/// Without an oracle the core cannot read a constant, so a member found nowhere
/// is not a refutation: two indices may hold one value, and the rules say only
/// what they can prove.
#[test]
fn a_table_without_an_oracle_proves_and_does_not_refute() {
    assert_eq!(
        table_relation(&[0, 1], &[0, 1, 2], &NoLeafRelations),
        Relation::Holds
    );
    assert_ne!(
        table_relation(&[2], &[0, 1], &NoLeafRelations),
        Relation::Fails
    );
}

/// A union a caller built by hand is not in the canonical order, and reading an
/// unordered list as a set answers by where a member happens to sit. Such a
/// union keeps the member walk it had, which decides the inclusion the slow way
/// and gets it right.
#[test]
fn an_unordered_union_of_literals_is_not_read_as_a_set() {
    let raw = |indices: &[usize]| {
        Schema::Union(
            indices
                .iter()
                .map(|index| Schema::Literal(ConstIx::new(*index)))
                .collect::<Vec<_>>()
                .into(),
        )
    };
    // The list the constructor would order as [0, 1, 2], written backwards. The
    // subject is in it, so the inclusion holds however the list is read -- and
    // a binary search over it would look at the middle, find 1, and go the
    // wrong way for 2.
    let backwards = raw(&[2, 1, 0]);
    assert_eq!(
        finite_set(&backwards),
        None,
        "an unordered list is not a set"
    );
    let budget = Cell::new(DECISION_BUDGET);
    assert_eq!(
        Schema::Literal(ConstIx::new(2)).subtype_relation(&backwards, &Values, &[], &budget),
        Relation::Holds
    );
}

/// A literal denotes the values equal to its constant, which for a constant
/// that does not equal itself is none of them. The oracle answers it as the
/// question it has: a singleton disjoint from itself is the empty set.
#[test]
fn a_constant_that_does_not_equal_itself_denotes_no_value() {
    /// A pool of one constant, which is not equal to itself.
    struct NotAValue;
    impl Constants for NotAValue {}
    impl LeafRelations for NotAValue {
        fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
            None
        }
        fn literals_disjoint(&self, _left: ConstIx, _right: ConstIx) -> Option<bool> {
            Some(true)
        }
    }
    let verdict = |oracle: &dyn LeafRelations| {
        Schema::Literal(ConstIx::new(0)).verdict_rec(
            oracle,
            &[],
            &mut Vec::new(),
            &Cell::new(DECISION_BUDGET),
        )
    };
    assert_eq!(verdict(&NotAValue), Verdict::Empty);
    assert_eq!(verdict(&Values), Verdict::Inhabited);
    let literal = Schema::Literal(ConstIx::new(0));
    // The empty set is below every set, and a refutation over it is not one.
    assert!(literal.is_subtype_of_under(&Schema::Str, &NotAValue, &[]));
}

/// A schema the rules prove **inhabited** is not lowered.
///
/// The descriptor is the second reading of one question, asked where the rules
/// reach neither answer. A proof of inhabitation is an answer: a value of the
/// schema is a value of it, and a sound second reading cannot say otherwise --
/// so asking is work that cannot change a verdict, and a lowering determinises
/// automata and takes products to do it.
///
/// The instrument is the pool: a lowering reads every constant it meets, so a
/// pool nobody asked is a lowering that did not happen.
#[test]
fn a_schema_proven_inhabited_is_not_lowered() {
    /// A pool that counts what it is asked, and holds one constant that is a
    /// value -- enough for the rules to prove the record inhabited.
    struct Counted(Cell<usize>);

    impl Constants for Counted {
        fn constant(&self, _index: ConstIx) -> Option<Operand> {
            self.0.set(self.0.get() + 1);
            None
        }

        fn operand(&self, _index: OperandIx) -> Option<Operand> {
            self.0.set(self.0.get() + 1);
            None
        }

        fn class(&self, _index: ClassIx) -> Option<Class> {
            self.0.set(self.0.get() + 1);
            None
        }
    }

    impl LeafRelations for Counted {
        fn leaf_subtype(&self, _sub: &Schema, _sup: &Schema) -> Option<bool> {
            None
        }

        fn literals_disjoint(&self, _left: ConstIx, _right: ConstIx) -> Option<bool> {
            Some(false)
        }
    }

    let record = Schema::record(
        vec![Field {
            name: "leaf".into(),
            schema: Schema::Literal(ConstIx::new(0)),
            required: true,
        }],
        Openness::Closed,
    );
    let pool = Counted(Cell::new(0));
    assert!(
        !record.is_empty_with(&pool, &[]),
        "the record holds a value"
    );
    assert_eq!(
        pool.0.get(),
        0,
        "the descriptor was asked to overturn a proof of inhabitation"
    );

    // And the schema the rules cannot read is still asked about, which is what
    // makes the reading above a saving rather than a narrowing.
    let opaque = Schema::Instance(ClassIx::new(0));
    let pool = Counted(Cell::new(0));
    assert!(!opaque.is_empty_with(&pool, &[]));
    assert!(
        pool.0.get() > 0,
        "an unknown verdict left the descriptor unasked"
    );
}

/// The field cache answers the goal it was asked and not one beside it.
///
/// A record whose fields repeat a schema asks one goal per field, and the rule
/// remembers the last pair to answer it once. Both halves of that pair are the
/// goal: a subject field equal to the last one says nothing about the relation
/// when the supertype's field is a different schema, and a cache reading one
/// side would report the second field decided by the first.
#[test]
fn the_field_cache_reads_both_halves_of_the_goal() {
    let record = |fields: Vec<(&str, Schema)>| {
        Schema::record(
            fields
                .into_iter()
                .map(|(name, schema)| Field {
                    name: name.into(),
                    schema,
                    required: true,
                })
                .collect(),
            Openness::Closed,
        )
    };
    // Two fields of one schema, against two fields of two: the first pair holds
    // and the second does not, so the record does not.
    let subject = record(vec![("a", Schema::Int), ("b", Schema::Int)]);
    let mixed = record(vec![("a", Schema::Int), ("b", Schema::Str)]);
    assert!(!structural(&subject, &mixed));
    assert!(!subject.is_subtype_of(&mixed));
    // And the shape the cache exists for still decides: one goal, twice.
    let wider = record(vec![
        ("a", Schema::union([Schema::Int, Schema::Str])),
        ("b", Schema::union([Schema::Int, Schema::Str])),
    ]);
    assert!(structural(&subject, &wider));
    // The other side of the same reading: the supertype's fields repeat and the
    // subject's do not, so the second field is a goal of its own.
    let mixed_subject = record(vec![("a", Schema::Int), ("b", Schema::NoneType)]);
    assert!(!structural(&mixed_subject, &wider));
}

/// The position cache answers the goal it was asked, as the field cache does.
///
/// A tuple whose positions repeat a schema asks one goal per position, and the
/// rule remembers the last pair to answer it once. A position whose element
/// equals the last one says nothing where the other side's element differs, and
/// a cache that read the question as answered would report the second position
/// decided by the first.
#[test]
fn the_position_cache_reads_both_halves_of_the_goal() {
    let tuple = |elements: Vec<Schema>| Schema::tuple(SeqShape::fixed(elements));
    // Two positions of one schema, against two positions of two: the first
    // aligns and the second does not, so the tuple does not.
    let subject = tuple(vec![Schema::Int, Schema::Int]);
    let mixed = tuple(vec![Schema::Int, Schema::Str]);
    assert!(!structural(&subject, &mixed));
    assert!(!subject.is_subtype_of(&mixed));
    // The shape the cache exists for decides: one goal, twice.
    let wider = tuple(vec![
        Schema::union([Schema::Int, Schema::Str]),
        Schema::union([Schema::Int, Schema::Str]),
    ]);
    assert!(structural(&subject, &wider));
    // And the other side: the supertype's positions repeat and the subject's do
    // not, so the second position is a goal of its own.
    assert!(!structural(
        &tuple(vec![Schema::Int, Schema::NoneType]),
        &wider
    ));
    // A repeated tail is the same question past the prefix, so a prefix element
    // equal to the last one does not answer for the tail either.
    let by_tail = Schema::tuple(SeqShape::prefix_tail(vec![Schema::Int], Schema::Str));
    assert!(!structural(
        &tuple(vec![Schema::Int, Schema::Int]),
        &by_tail
    ));
}
