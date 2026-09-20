//! The rule, read against every node of the IR.
//!
//! The rule is a table, so the test is the table: one row per variant, with
//! what that variant's values can answer, and the answers asserted rather
//! than described. A variant added to the IR fails to compile in [`variant`]
//! until it has a row.

use super::{
    Carries, OrderGroup, carries_division, carries_length, carries_order, carries_pattern,
    carries_through,
};
use crate::ir::{ClassIx, ConstIx, Constraint, DefIx, Field, MapClause, Schema, SeqShape};
use std::sync::Arc;

/// The variant a node is, by the name the enum gives it.
///
/// Exhaustive on purpose: this is what makes the table below a claim about
/// the node set rather than about the rows somebody remembered to write.
fn variant(schema: &Schema) -> &'static str {
    match schema {
        Schema::Anything(_) => "Anything",
        Schema::Nothing => "Nothing",
        Schema::NoneType => "NoneType",
        Schema::Bool => "Bool",
        Schema::Int => "Int",
        Schema::Float => "Float",
        Schema::Str => "Str",
        Schema::Bytes => "Bytes",
        Schema::Literal(_) => "Literal",
        Schema::Seq { .. } => "Seq",
        Schema::Coll { .. } => "Coll",
        Schema::KeyedMap { .. } => "KeyedMap",
        Schema::Union(_) => "Union",
        Schema::Intersection(_) => "Intersection",
        Schema::Complement(_) => "Complement",
        Schema::Instance(_) => "Instance",
        Schema::AttrRecord { .. } => "AttrRecord",
        Schema::Refine { .. } => "Refine",
        Schema::Ref(_) => "Ref",
        Schema::SelfRef(_) => "SelfRef",
    }
}

/// A field, for the two nodes that carry one.
fn field(name: &str) -> Field {
    Field {
        name: name.into(),
        schema: Schema::Int,
        required: true,
    }
}

/// What each base's values can answer: a length, a pattern to match, a
/// divisor, and an order against a number.
///
/// The order column is the one an operand's group decides, and a number is
/// the group the pool's operands have; [`an_order_bound_asks_about_the_pair`]
/// asks the same bases against every group there is.
///
/// A node the rule has nothing to say about answers `Maybe`, and that is the
/// answer a refusal must never be built from: the frontend refuses only on
/// `No`.
fn table() -> Vec<(Schema, Carries, Carries, Carries, Carries)> {
    use Carries::{Maybe, No, Yes};
    vec![
        // The two bounds say nothing: the top holds values of every kind and
        // the bottom holds none, so neither rules a constraint out.
        (Schema::ANYTHING, Maybe, Maybe, Maybe, Maybe),
        (Schema::ANY, Maybe, Maybe, Maybe, Maybe),
        (Schema::Nothing, Maybe, Maybe, Maybe, Maybe),
        // `None` answers no constraint at all, and orders against nothing.
        (Schema::NoneType, No, No, No, No),
        // The numbers: no length, no pattern, a divisor, and an order with
        // another number. `bool` is among them because it subclasses `int`.
        (Schema::Bool, No, No, Yes, Yes),
        (Schema::Int, No, No, Yes, Yes),
        (Schema::Float, No, No, Yes, Yes),
        // Text has a length and a pattern; bytes has the length and not the
        // pattern, because a pattern here is matched against text.
        (Schema::Str, Yes, Yes, No, No),
        (Schema::Bytes, Yes, No, No, No),
        // A literal's node says only "a literal": the kind is its constant's
        // and the constant is in the pool, which this rule cannot read.
        (Schema::Literal(ConstIx::new(0)), Maybe, Maybe, Maybe, Maybe),
        // The sized kinds.
        (
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            Yes,
            No,
            No,
            No,
        ),
        (Schema::set(Schema::Int), Yes, No, No, No),
        (
            Schema::mapping(MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }),
            Yes,
            No,
            No,
            No,
        ),
        // A union answers for its members, and a member that can answer makes
        // the constraint a narrowing of the union rather than an emptying of
        // it. Two rows, because one member answering and none answering are
        // the two sides of that fold.
        (
            Schema::Union(vec![Schema::Int, Schema::Str].into()),
            Yes,
            Yes,
            Yes,
            Yes,
        ),
        (
            Schema::Union(vec![Schema::Int, Schema::Float].into()),
            No,
            No,
            Yes,
            Yes,
        ),
        // A meet and a complement narrow a set this rule does not compute, and
        // a reference names one it would have to resolve. All three stand
        // aside rather than guess.
        (
            Schema::Intersection(vec![Schema::Int, Schema::Str].into()),
            Maybe,
            Maybe,
            Maybe,
            Maybe,
        ),
        (
            Schema::Complement(Arc::new(Schema::Int)),
            Maybe,
            Maybe,
            Maybe,
            Maybe,
        ),
        (Schema::Ref(DefIx::new(0)), Maybe, Maybe, Maybe, Maybe),
        (Schema::SelfRef(0), Maybe, Maybe, Maybe, Maybe),
        // Only the bindings hold a class, so the core cannot say what its
        // values answer -- an attribute record included, since the object
        // carrying the attribute may be of any kind.
        (
            Schema::Instance(ClassIx::new(0)),
            Maybe,
            Maybe,
            Maybe,
            Maybe,
        ),
        (
            Schema::attr_record(vec![field("a")]),
            Maybe,
            Maybe,
            Maybe,
            Maybe,
        ),
        // A refinement answers with its own base: the frontend folds nested
        // markers onto the base they narrow, so the constraints already there
        // say nothing about the ones being added.
        (
            Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: vec![Constraint::MinLen(1)].into(),
            },
            No,
            No,
            Yes,
            Yes,
        ),
    ]
}

#[test]
fn every_variant_of_the_ir_has_a_row() {
    let covered: std::collections::BTreeSet<&'static str> =
        table().iter().map(|(base, ..)| variant(base)).collect();
    let every: std::collections::BTreeSet<&'static str> = [
        "Anything",
        "Nothing",
        "NoneType",
        "Bool",
        "Int",
        "Float",
        "Str",
        "Bytes",
        "Literal",
        "Seq",
        "Coll",
        "KeyedMap",
        "Union",
        "Intersection",
        "Complement",
        "Instance",
        "AttrRecord",
        "Refine",
        "Ref",
        "SelfRef",
    ]
    .into_iter()
    .collect();
    assert_eq!(covered, every, "the table and the node set have to agree");
}

#[test]
fn each_base_answers_the_constraints_its_values_can() {
    for (base, length, pattern, division, order) in table() {
        assert_eq!(carries_length(&base), length, "length of {base:?}");
        assert_eq!(carries_pattern(&base), pattern, "pattern on {base:?}");
        assert_eq!(carries_division(&base), division, "divisor on {base:?}");
        assert_eq!(
            carries_order(&base, Some(OrderGroup::Number)),
            order,
            "order of {base:?} against a number"
        );
    }
}

/// Every base with an order of its own, and the group its values compare in.
fn ordered_bases() -> Vec<(Schema, OrderGroup)> {
    vec![
        (Schema::Bool, OrderGroup::Number),
        (Schema::Int, OrderGroup::Number),
        (Schema::Float, OrderGroup::Number),
        (Schema::Str, OrderGroup::Text),
        (Schema::Bytes, OrderGroup::Bytes),
        (
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            OrderGroup::List,
        ),
        (
            Schema::tuple(SeqShape::fixed([Schema::Int])),
            OrderGroup::Tuple,
        ),
        (Schema::set(Schema::Int), OrderGroup::Set),
        (Schema::frozen_set(Schema::Int), OrderGroup::Set),
    ]
}

#[test]
fn an_order_bound_asks_about_the_pair() {
    const EVERY_GROUP: [OrderGroup; 6] = [
        OrderGroup::Number,
        OrderGroup::Text,
        OrderGroup::Bytes,
        OrderGroup::List,
        OrderGroup::Tuple,
        OrderGroup::Set,
    ];

    // A base with an order of its own says nothing on its own: it answers for
    // the group its values compare in and refuses every other, because Python
    // raises across the groups rather than ordering them. A rule reading only
    // the base would admit every mismatched bound and build the schema that
    // admits nothing.
    for (base, own) in ordered_bases() {
        for group in EVERY_GROUP {
            let expected = if group == own {
                Carries::Yes
            } else {
                Carries::No
            };
            assert_eq!(
                carries_order(&base, Some(group)),
                expected,
                "order of {base:?} against {group:?}"
            );
        }
        // An operand of no group at all orders against nothing.
        assert_eq!(
            carries_order(&base, None),
            Carries::No,
            "order of {base:?} against an operand of no group"
        );
    }

    // The two kinds that order against no group answer before the group is
    // read: `None` has no comparison, and a dict's values are unordered
    // however ordered its keys are.
    for base in [
        Schema::NoneType,
        Schema::mapping(MapClause {
            key: Schema::Str,
            value: Schema::Int,
        }),
    ] {
        for group in EVERY_GROUP {
            assert_eq!(
                carries_order(&base, Some(group)),
                Carries::No,
                "{base:?} orders against {group:?}"
            );
        }
        assert_eq!(carries_order(&base, None), Carries::No);
    }

    // The two set kinds share one order, which is inclusion, so either
    // spelling of the base answers for the group either operand belongs to.
    assert_eq!(
        carries_order(&Schema::set(Schema::Int), Some(OrderGroup::Set)),
        carries_order(&Schema::frozen_set(Schema::Int), Some(OrderGroup::Set))
    );
    // A list and a tuple are two kinds to Python's comparison as much as to
    // this one, so the sequence container is read rather than the shape.
    assert_eq!(
        carries_order(
            &Schema::list(SeqShape::homogeneous(Schema::Int)),
            Some(OrderGroup::Tuple)
        ),
        Carries::No
    );
    assert_eq!(
        carries_order(
            &Schema::tuple(SeqShape::homogeneous(Schema::Int)),
            Some(OrderGroup::List)
        ),
        Carries::No
    );
}

#[test]
fn a_union_answers_for_its_members_and_a_plain_base_for_itself() {
    // A plain base has no children to fold, so the fold stands aside and the
    // caller's own table answers. This is what makes `carries_through` a
    // pre-pass rather than the rule.
    assert_eq!(carries_through(&Schema::Str, &carries_length), None);
    assert_eq!(carries_through(&Schema::Int, &carries_division), None);

    // The fold is `or` and not `and`: one member that can answer makes the
    // constraint a narrowing.
    let mixed = Schema::Union(vec![Schema::Int, Schema::Str].into());
    assert_eq!(
        carries_through(&mixed, &carries_length),
        Some(Carries::Yes),
        "a member with a length answers for the union"
    );
    let numbers = Schema::Union(vec![Schema::Int, Schema::Float].into());
    assert_eq!(
        carries_through(&numbers, &carries_length),
        Some(Carries::No),
        "no member with a length leaves the union refusing"
    );

    // An empty union folds from the unit of the fold, which is the refusal:
    // nothing in it can answer, because there is nothing in it.
    let empty = Schema::Union(Vec::new().into());
    assert_eq!(carries_through(&empty, &carries_length), Some(Carries::No));

    // A meet, a complement and a reference answer `Maybe` from the fold
    // itself rather than deferring to the caller's table. The two routes give
    // the same answer to all four callers here, so the arm is observable only
    // from this side -- and it is the arm that keeps a caller whose own
    // catch-all refuses from reading "the base does not say" as "no value
    // can".
    for opaque in [
        Schema::Intersection(vec![Schema::Int, Schema::Str].into()),
        Schema::Complement(Arc::new(Schema::Int)),
        Schema::Ref(DefIx::new(0)),
        Schema::SelfRef(0),
    ] {
        assert_eq!(
            carries_through(&opaque, &carries_length),
            Some(Carries::Maybe),
            "{opaque:?} narrows a set this rule does not compute"
        );
    }

    // A refinement answers with its own base, whatever it already carries.
    let refined = Schema::Refine {
        base: Arc::new(Schema::Str),
        constraints: vec![Constraint::MinLen(1)].into(),
    };
    assert_eq!(
        carries_through(&refined, &carries_pattern),
        Some(Carries::Yes)
    );
}

#[test]
fn an_answer_folds_the_way_a_union_does() {
    use Carries::{Maybe, No, Yes};
    // The table of the fold, which is the join of the three answers ordered
    // No < Maybe < Yes: a member that answers carries the union, and a member
    // that does not say leaves it not saying rather than refusing.
    for (left, right, folded) in [
        (Yes, Yes, Yes),
        (Yes, No, Yes),
        (No, Yes, Yes),
        (Yes, Maybe, Yes),
        (Maybe, Yes, Yes),
        (Maybe, Maybe, Maybe),
        (Maybe, No, Maybe),
        (No, Maybe, Maybe),
        (No, No, No),
    ] {
        assert_eq!(left.or(right), folded, "{left:?} or {right:?}");
    }
}
