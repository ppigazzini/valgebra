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

/// A test-side count of the goals one query asks twice.
///
/// [`DECISION_BUDGET`]'s own argument rests on a number -- over the decision
/// workloads the goals a query *repeats* are zero, so the ceiling stands in for
/// a termination argument and not for a memo -- and nothing read that number.
/// This reads it. Compiled for this crate's own tests only, so the procedure a
/// caller runs carries no counter and pays nothing for one.
///
/// A goal is the pair the recursion is asked about, and two are the same goal
/// when the pair is equal. Equality rather than the interned address: a query
/// builds goals of its own -- [`seq_splits_across_union`] makes a sequence per
/// branch -- and those are temporaries, so an address freed and handed out
/// again would read as a repeat that never happened. Equality also counts at
/// least as many repeats as identity can, which is the safe direction for a
/// claim that there are none.
pub(crate) mod goals {
    use std::cell::RefCell;

    use rustc_hash::FxHashMap;

    use crate::ir::Schema;

    thread_local! {
        /// The goals asked inside a [`counted`] call, and how often each was.
        static ASKED: RefCell<Option<FxHashMap<(Schema, Schema), u32>>> =
            const { RefCell::new(None) };
        /// The deepest the trail grew inside a [`counted`] call.
        static DEEPEST: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// What one counted query asked.
    ///
    /// Both numbers, because either alone can be read wrong: `repeated` is the
    /// claim, and `asked` is what says the recorder was wired to the procedure
    /// at all. A count of zero repeats over a query that asked nothing is the
    /// shape a silent recorder gives, and it reads exactly like a good answer.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Counts {
        /// Goals the recursion was asked, counting a repeat once per ask.
        pub(crate) asked: usize,
        /// How many of those asks were of a pair already asked.
        pub(crate) repeated: usize,
        /// The most pairs the trail held at once.
        pub(crate) deepest: usize,
    }

    /// Run `query` with its goals counted, and give back what it asked.
    ///
    /// One count per thread: two at once would each be about part of the
    /// other's query, and a number about part of a query is not the number the
    /// argument needs.
    pub(crate) fn counted<T>(query: impl FnOnce() -> T) -> (T, Counts) {
        ASKED.with(|asked| {
            let mut slot = asked.borrow_mut();
            assert!(slot.is_none(), "a count is already running on this thread");
            *slot = Some(FxHashMap::default());
        });
        DEEPEST.with(|deepest| deepest.set(0));
        let answer = query();
        let table = ASKED
            .with(|asked| asked.borrow_mut().take())
            .expect("the table installed above is still there");
        let counts = Counts {
            asked: table.values().map(|times| *times as usize).sum(),
            repeated: table.values().map(|times| (times - 1) as usize).sum(),
            deepest: DEEPEST.with(std::cell::Cell::get),
        };
        (answer, counts)
    }

    /// Record the trail's depth after a push, where a count is running.
    pub(crate) fn trail(depth: usize) {
        DEEPEST.with(|deepest| deepest.set(deepest.get().max(depth)));
    }

    /// Record one goal, where a count is running.
    pub(crate) fn record(subject: &Schema, other: &Schema) {
        ASKED.with(|asked| {
            if let Some(table) = asked.borrow_mut().as_mut() {
                *table.entry((subject.clone(), other.clone())).or_insert(0) += 1;
            }
        });
    }
}

use crate::{
    ClassIx, Constraint, DefIx, Field, Kind, LeafRelations, Openness, OperandIx, Relation, Schema,
    SeqShape,
};
use goals::Counts;

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

// THEORY: repeated-goals-are-counted, regularity-bounds-the-goals
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

/// A fixed pair of unions: the subject the product rule splits across the
/// branches of [`product_split`], where no single branch contains it.
fn product_subject() -> Schema {
    Schema::tuple(SeqShape::fixed([
        Schema::union([Schema::Int, Schema::Str, Schema::Bytes]),
        Schema::union([Schema::Int, Schema::Str]),
    ]))
}

/// The union of pairs that covers [`product_subject`] only taken together.
fn product_split() -> Schema {
    let pair = |a: Schema, b: Schema| Schema::tuple(SeqShape::fixed([a, b]));
    Schema::union([
        pair(Schema::Int, Schema::Int),
        pair(Schema::Str, Schema::Int),
        pair(Schema::Bytes, Schema::Int),
        pair(
            Schema::union([Schema::Int, Schema::Str, Schema::Bytes]),
            Schema::Str,
        ),
    ])
}

/// Every distinct subtree of `schema`, and of every definition it can reach.
///
/// The set regularity is about: a schema is a finite tree with back edges, so
/// unfolding it forever reaches only these, and a goal the procedure asks is
/// a pair of them unless a rule built a term of its own.
fn subterms(schema: &Schema, defs: &[Schema]) -> Vec<Schema> {
    fn collect(schema: &Schema, out: &mut Vec<Schema>) {
        if !out.contains(schema) {
            out.push(schema.clone());
        }
        for child in schema.children() {
            collect(child, out);
        }
    }
    let mut out = Vec::new();
    collect(schema, &mut out);
    for body in defs {
        collect(body, &mut out);
    }
    out
}

/// The distinct goals a query asks, against the pairs of subterms the two
/// sides have.
fn distinct_against_subterm_pairs(sub: &Schema, sup: &Schema, defs: &[Schema]) -> (usize, usize) {
    let (_, counts) = ask(sub, sup, defs);
    let pairs = subterms(sub, defs).len() * subterms(sup, defs).len();
    (counts.asked - counts.repeated, pairs)
}

// THEORY: regularity-bounds-the-goals
/// Regularity, spent: the distinct goals a query asks are bounded by the
/// pairs of subterms the two schemas have.
///
/// A schema is regular by construction -- a finite tree with back edges into
/// a finite table of definitions -- so the subterms reachable by unfolding are
/// finitely many, and a goal the procedure asks is a pair of them. That is
/// the finiteness JACM §6.9 uses as a termination argument, and the number
/// the work budget stands in for. Held here on the shapes the budget was
/// measured over, on the recursive pairs the trail is for, and on the one
/// rule that builds terms of its own: the product rule narrows a component
/// by a branch it does not cover and asks the narrowed tuple against the
/// rest, and the narrowed tuple is no subterm of the pair. Counted, that
/// shape asks fewer distinct goals than the pair has subterm pairs -- the
/// narrowings are few and each is asked once -- so the bound holds of it as
/// a measurement where it holds of the others as an argument, and the row
/// is what says so when a rule starts building past it.
#[test]
fn the_goals_a_query_asks_are_pairs_of_the_subterms() {
    let element = nested_lists(2, Schema::Int);
    let wider = nested_lists(2, Schema::union([Schema::Int, Schema::Str]));
    let tree = tree_defs();
    let list_tree = vec![Schema::union([
        Schema::Int,
        list_of(Schema::Ref(DefIx::new(0))),
    ])];
    let pairs: Vec<(Schema, Schema, Vec<Schema>)> = vec![
        (
            repeating_record(8, &element),
            repeating_record(8, &wider),
            vec![],
        ),
        (
            repeating_tuple(6, &element),
            repeating_tuple(6, &wider),
            vec![],
        ),
        (
            nested_lists(6, Schema::Bool),
            nested_lists(6, Schema::Int),
            vec![],
        ),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::union([Schema::Ref(DefIx::new(0)), Schema::Str]),
            tree.clone(),
        ),
        (Schema::Ref(DefIx::new(0)), Schema::Ref(DefIx::new(0)), tree),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::union([Schema::Int, list_of(Schema::ANYTHING)]),
            list_tree.clone(),
        ),
        (
            list_of(Schema::Ref(DefIx::new(0))),
            Schema::Ref(DefIx::new(0)),
            list_tree,
        ),
        (product_subject(), product_split(), vec![]),
    ];
    assert_eq!(
        ask(&product_subject(), &product_split(), &[]).0,
        Relation::Holds
    );
    for (sub, sup, defs) in &pairs {
        let (distinct, bound) = distinct_against_subterm_pairs(sub, sup, defs);
        assert!(
            distinct >= 1,
            "the counter saw no goal for {sub:?} <= {sup:?}"
        );
        assert!(
            distinct <= bound,
            "{sub:?} <= {sup:?} asked {distinct} distinct goals over {bound} subterm pairs"
        );
    }
}

// THEORY: the-trail-holds-terms
/// The longest trail any recursive shape in hand builds is three pairs.
///
/// The deviation's cost is a scan over the trail at every recursive goal, and
/// the sentence that bounds the cost was a measurement taken once over the
/// relation matrix -- two pairs -- and written into prose. It is a number the
/// recorder reads: every recursive pair the counter's shapes carry, and the
/// pairs the fixpoint laws are written over, are asked with the trail's depth
/// recorded, and the deepest is the figure the page states. The first run of
/// this row found the matrix's figure short by one: a list over a reference
/// against a union holding that reference unfolds the reference on both
/// sides and once more under the list, which is three. A shape whose trail
/// grows past it fails here by name, which is what turns the sentence into a
/// claim the tree holds.
#[test]
fn the_longest_trail_any_recursive_shape_builds_is_three_pairs() {
    let tree = tree_defs();
    let list_tree = vec![Schema::union([
        Schema::Int,
        list_of(Schema::Ref(DefIx::new(0))),
    ])];
    let word_and_not = vec![
        Schema::union([Schema::Str, list_of(Schema::Ref(DefIx::new(0)))]),
        Schema::union([
            Schema::Int,
            list_of(Schema::Complement(Arc::new(Schema::Ref(DefIx::new(1))))),
        ]),
    ];
    let pairs: Vec<(Schema, Schema, Vec<Schema>)> = vec![
        (
            Schema::Ref(DefIx::new(0)),
            Schema::Ref(DefIx::new(0)),
            tree.clone(),
        ),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::union([Schema::Ref(DefIx::new(0)), Schema::Str]),
            tree,
        ),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::union([Schema::Int, list_of(Schema::ANYTHING)]),
            list_tree.clone(),
        ),
        (
            list_of(Schema::Ref(DefIx::new(0))),
            Schema::Ref(DefIx::new(0)),
            list_tree,
        ),
        (
            Schema::Ref(DefIx::new(0)),
            Schema::Ref(DefIx::new(1)),
            word_and_not.clone(),
        ),
        (
            Schema::Ref(DefIx::new(1)),
            Schema::Ref(DefIx::new(0)),
            word_and_not.clone(),
        ),
        (
            list_of(Schema::Ref(DefIx::new(1))),
            Schema::union([Schema::Int, Schema::Ref(DefIx::new(1))]),
            word_and_not,
        ),
    ];
    let mut deepest = 0;
    for (sub, sup, defs) in &pairs {
        let (_, counts) = ask(sub, sup, defs);
        assert!(
            counts.asked >= 1,
            "the counter saw no goal for {sub:?} <= {sup:?}"
        );
        assert!(
            counts.deepest <= 3,
            "{sub:?} <= {sup:?} grew the trail to {} pairs",
            counts.deepest
        );
        deepest = deepest.max(counts.deepest);
    }
    assert_eq!(deepest, 3, "no shape reached the length the page states");
}

/// The same bound, asked of drawn pairs over drawn definitions.
mod drawn {
    use super::distinct_against_subterm_pairs;
    use crate::laws::{drawn_defs, recursive_schema};
    use proptest::prelude::*;

    proptest! {
        // A bounded shrink, so a broken bound cannot turn a caught mutation
        // into a run that outlasts a sweep: see `budget::law`.
        #![proptest_config(ProptestConfig {
            max_shrink_time: 2_000,
            ..ProptestConfig::default()
        })]

        // THEORY: regularity-bounds-the-goals
        /// The distinct goals a drawn query asks are bounded by the pairs of
        /// subterms its two sides have, over drawn definitions.
        ///
        /// The eight chosen pairs above hold the bound on the shapes the
        /// budget was measured over; this holds it on whatever the recursive
        /// fragment draws, references into drawn definitions included, so a
        /// rule that starts building goals past the subterms is caught on a
        /// shape nobody chose.
        #[test]
        fn the_goals_a_drawn_query_asks_are_pairs_of_the_subterms(
            sub in recursive_schema(),
            sup in recursive_schema(),
            defs in drawn_defs(),
        ) {
            let (distinct, pairs) = distinct_against_subterm_pairs(&sub, &sup, &defs);
            prop_assert!(distinct >= 1, "the counter saw no goal for {sub:?} <= {sup:?}");
            prop_assert!(
                distinct <= pairs,
                "{distinct} distinct goals over {pairs} subterm pairs for {sub:?} <= {sup:?}"
            );
        }
    }
}
