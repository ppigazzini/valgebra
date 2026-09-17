//! The number the work budget's argument rests on: goals a query asks twice.
//!
//! `DECISION_BUDGET` stands in for a termination argument, and its doc says a
//! goal memo would not pay the debt down -- because over the decision workloads
//! the goals a query *repeats* number zero. That sentence was prose. The
//! counter in [`super::goals`] makes it a measurement, and this asks it of the
//! shapes the claim is about.
//!
//! The shapes are the workloads' own, written here rather than imported: an
//! example is a binary and a test cannot call into one. What is measured is the
//! shape, so each is spelled the way `examples/decision_repeat_workload.rs` and
//! `examples/decision_matrix_workload.rs` spell it, and a change to either
//! belongs in both.
//!
//! A repeat counted here is a goal the procedure really re-derives: the
//! recorder sits inside `is_subtype_rec`, which is past the caches that answer
//! a repeated field or position without asking again. So zero here is "the
//! caches absorb it", not "the goal was never reached".

use std::sync::Arc;

use super::goals::{self, Counts};
use crate::{
    ClassIx, Constraint, DefIx, Field, Kind, LeafRelations, Openness, OperandIx, Relation, Schema,
    SeqShape,
};

/// Two classes that lay down no builtin layout, answered as the bindings answer
/// a plain Python class. The matrix workload's oracle, for the matrix's pairs:
/// an oracle that declines every class question leaves the class readings
/// unreachable, and a count over pairs nothing reaches is a count of nothing.
struct PlainClasses;

impl crate::descr::lower::Constants for PlainClasses {}

impl LeafRelations for PlainClasses {
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool> {
        match (sub, sup) {
            (Schema::Instance(a), Schema::Instance(b)) => Some(a == b),
            _ => None,
        }
    }

    fn class_admits_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        None
    }

    fn direct_instance_of_kind(&self, _class: ClassIx, _kind: Kind) -> Option<bool> {
        Some(false)
    }

    fn kind_derives_from(&self, _kind: Kind, _class: ClassIx) -> Option<bool> {
        Some(false)
    }

    fn atom_denotes_a_set(&self, atom: &Schema) -> Option<bool> {
        Some(matches!(atom, Schema::Instance(_)))
    }
}

/// What one query asked, with the verdict beside it.
///
/// The verdict is carried out so a test can say the query it counted was the
/// query it meant: a pair that stopped being decided would otherwise read as a
/// happy zero.
fn ask(sub: &Schema, sup: &Schema, defs: &[Schema]) -> (Relation, Counts) {
    goals::counted(|| sub.subtype_relation_under(sup, &PlainClasses, defs))
}

/// The goals one query asks twice.
fn repeated(sub: &Schema, sup: &Schema, defs: &[Schema]) -> usize {
    ask(sub, sup, defs).1.repeated
}

/// `list[T]`.
fn list_of(element: Schema) -> Schema {
    Schema::list(SeqShape::homogeneous(element))
}

/// A list nested `depth` deep over `leaf`.
fn nested_lists(depth: usize, leaf: Schema) -> Schema {
    (0..depth).fold(leaf, |inner, _| list_of(inner))
}

/// A record whose fields all carry `element`.
fn repeating_record(width: usize, element: &Schema) -> Schema {
    Schema::record(
        (0..width)
            .map(|index| Field {
                name: format!("f{index}").into(),
                schema: element.clone(),
                required: true,
            })
            .collect(),
        Openness::Closed,
    )
}

/// A tuple whose positions all carry `element`.
fn repeating_tuple(width: usize, element: &Schema) -> Schema {
    Schema::tuple(SeqShape::fixed((0..width).map(|_| element.clone())))
}

/// A refinement carrying one constraint.
fn refined(base: Schema, constraint: Constraint) -> Schema {
    Schema::Refine {
        base: Arc::new(base),
        constraints: vec![constraint].into(),
    }
}

/// A class met with the attributes its instances carry.
fn dataclass(class: ClassIx, width: usize) -> Schema {
    let fields = (0..width)
        .map(|i| Field {
            name: format!("f{i}").into(),
            schema: Schema::Int,
            required: true,
        })
        .collect::<Vec<_>>();
    Schema::meet([
        Schema::Instance(class),
        Schema::AttrRecord {
            fields: fields.into(),
        },
    ])
}

/// A recursive record, reached through the reference that names it.
fn tree_defs() -> Vec<Schema> {
    vec![Schema::KeyedMap {
        fields: vec![
            Field {
                name: "value".into(),
                schema: Schema::Int,
                required: true,
            },
            Field {
                name: "left".into(),
                schema: Schema::Ref(DefIx::new(0)),
                required: false,
            },
        ]
        .into(),
        defaults: Vec::new().into(),
    }]
}

// THEORY: repeated-goals-are-counted
/// The counter is wired to the procedure, so a zero from it is a reading.
///
/// The control every count below needs. A recorder attached to nothing reports
/// no repeats for every query, which reads exactly like a procedure that
/// repeats nothing -- so what is asserted here is that goals arrive at all.
/// Stated as `asked` rather than as `repeated` on purpose: a shape that repeats
/// today may stop, and a control that went away with it would leave every
/// number below unwitnessed.
#[test]
fn the_counter_sees_the_goals_a_query_asks() {
    let narrow = list_of(list_of(Schema::Int));
    let wide = list_of(list_of(Schema::union([Schema::Int, Schema::Str])));
    let (verdict, counts) = ask(&narrow, &wide, &[]);
    assert_eq!(verdict, Relation::Holds);
    assert!(
        counts.asked >= 3,
        "the counter saw {} goal(s)",
        counts.asked
    );
    assert_eq!(counts.repeated, 0);
}

/// The shape a memo is usually proposed for: one goal, once per field.
#[test]
fn a_record_whose_fields_share_a_schema_repeats_no_goal() {
    let element = nested_lists(2, Schema::Int);
    let wider = nested_lists(2, Schema::union([Schema::Int, Schema::Str]));
    let narrow = repeating_record(8, &element);
    let wide = repeating_record(8, &wider);

    // The proof: every field's element is inside the wider one, which is one
    // goal reached once per field, and the field cache answers it after the
    // first.
    assert_eq!(ask(&narrow, &wide, &[]).0, Relation::Holds);
    assert_eq!(repeated(&narrow, &wide, &[]), 0);
    // And the refutation, which walks the same repeated goal the other way.
    let (verdict, counts) = ask(&wide, &narrow, &[]);
    assert_eq!(counts.repeated, 0);
    assert_ne!(verdict, Relation::Holds);
}

/// The same repetition in the container whose elements are reached by position.
#[test]
fn a_tuple_whose_positions_share_a_schema_repeats_no_goal() {
    let element = nested_lists(2, Schema::Int);
    let wider = nested_lists(2, Schema::union([Schema::Int, Schema::Str]));
    let narrow = repeating_tuple(8, &element);
    let wide = repeating_tuple(8, &wider);

    assert_eq!(ask(&narrow, &wide, &[]).0, Relation::Holds);
    assert_eq!(repeated(&narrow, &wide, &[]), 0);
    let (verdict, counts) = ask(&wide, &narrow, &[]);
    assert_eq!(counts.repeated, 0);
    assert_ne!(verdict, Relation::Holds);
}

// THEORY: repeated-goals-are-counted
/// The width the page names, so the number it states is the number here.
#[test]
fn a_record_of_thirty_two_fields_sharing_one_schema_repeats_no_goal() {
    let element = nested_lists(2, Schema::Int);
    let wider = nested_lists(2, Schema::union([Schema::Int, Schema::Str]));
    let narrow = repeating_record(32, &element);
    let wide = repeating_record(32, &wider);
    assert_eq!(ask(&narrow, &wide, &[]).0, Relation::Holds);
    assert_eq!(repeated(&narrow, &wide, &[]), 0);
}

// THEORY: repeated-goals-are-counted
/// The matrix's own pairs, each counted on its own -- and two of them repeat.
///
/// The structural readings' slow set, which the three older workloads do not
/// carry: a refinement on both sides, two sequences read as stars, a class met
/// with its attributes, a complement, and a reference on the supertype's side.
/// Every pair is one query, so the number is per query and not a total over the
/// matrix -- a budget is per query too.
///
/// **A meet against a union repeats four goals.** The union distribution asks
/// the meet of each branch, and the meet rule then asks the class atom against
/// the same thing once per member with no cache between the two rules. That is
/// word for word the shape [`super::DECISION_BUDGET`]'s argument names as what
/// would reopen the question of a goal memo, and this workload -- written after
/// that argument -- is where it is in hand. The numbers are recorded rather
/// than rounded to a claim: a cache between those two rules would take them to
/// zero, and this table is what would say so.
#[test]
fn the_matrix_repeats_a_goal_only_where_a_meet_meets_a_union() {
    let long_list = refined(list_of(Schema::Int), Constraint::MinLen(2));
    let positive = refined(Schema::Int, Constraint::Ge(OperandIx::new(0)));
    let long_text = refined(Schema::Str, Constraint::MinLen(1));
    let longer_list = refined(list_of(Schema::Int), Constraint::MinLen(3));
    let bare_list = list_of(Schema::Int);
    let strings = list_of(Schema::Str);
    let nested = list_of(list_of(Schema::Int));
    let point = dataclass(ClassIx::new(0), 2);
    let kinds = Schema::union([Schema::Int, Schema::Str]);
    let optional = Schema::union([Schema::Int, Schema::NoneType]);
    let not_int = Schema::Complement(Arc::new(Schema::Int));
    let a_class = Schema::Instance(ClassIx::new(1));
    let own_class = Schema::Instance(ClassIx::new(0));
    let defs = tree_defs();
    let tree = Schema::Ref(DefIx::new(0));
    let none: &[Schema] = &[];

    let matrix: Vec<(&str, &Schema, &Schema, &[Schema], usize)> = vec![
        ("a bound against a sign", &long_list, &positive, none, 0),
        ("a sign against a bound", &positive, &long_list, none, 0),
        (
            "a bound below a looser one",
            &longer_list,
            &long_list,
            none,
            0,
        ),
        ("two element kinds apart", &long_list, &strings, none, 0),
        ("an element against a nesting", &long_list, &nested, none, 0),
        (
            "a bare sequence against a bound",
            &bare_list,
            &long_list,
            none,
            0,
        ),
        (
            "a word bound against a list one",
            &long_text,
            &long_list,
            none,
            0,
        ),
        (
            "a refinement below its own base",
            &long_list,
            &bare_list,
            none,
            0,
        ),
        // The two that repeat: a meet against a union, twice over.
        ("a class against a union of kinds", &point, &kinds, none, 4),
        ("a class against an optional", &point, &optional, none, 4),
        ("a class against one kind", &point, &Schema::Int, none, 0),
        (
            "a class against the class it is built on",
            &point,
            &own_class,
            none,
            0,
        ),
        ("a complement against a class", &not_int, &a_class, none, 0),
        ("a complement against a meet", &not_int, &point, none, 0),
        ("a class against a fixpoint", &point, &tree, &defs, 0),
        ("a sequence against a fixpoint", &bare_list, &tree, &defs, 0),
    ];

    let mut total = 0;
    for (name, sub, sup, defs, expected) in matrix {
        let (_, counts) = ask(sub, sup, defs);
        assert!(counts.asked > 0, "{name} asked no goal at all");
        assert_eq!(counts.repeated, expected, "{name}");
        total += counts.repeated;
    }
    // The matrix's whole number, so a pair that starts repeating somewhere else
    // while one of the two stops cannot cancel out row by row.
    assert_eq!(total, 8);
}
