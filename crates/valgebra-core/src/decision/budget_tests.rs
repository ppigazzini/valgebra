use super::{Budget, DECISION_BUDGET, MAX_DECISION_DEPTH, descending, spend};

// THEORY: the-budget-declines
/// The work ceiling one decision query may spend. An exhausted budget must
/// answer `false`, because that is the whole signal a budgeted decision has
/// for stopping and reporting the conservative answer.
///
/// Driven here rather than through a decision, deliberately: a budget that
/// never exhausts makes an adversarial schema run without bound, so the
/// experiment would not finish and a timeout is a rig fault, not a
/// detection. One unit of the counter is the whole of what there is to test.
#[test]
fn an_exhausted_budget_refuses_to_spend() {
    let budget = Budget::new(2);
    assert!(spend(&budget));
    assert_eq!(budget.left(), 1);
    assert!(spend(&budget));
    assert_eq!(budget.left(), 0);
    // Exhausted: refuses, and does not wrap round to a fresh budget.
    assert!(!spend(&budget));
    assert_eq!(budget.left(), 0);
    assert!(!spend(&budget));
    assert_eq!(budget.left(), 0);

    // A budget of zero refuses on its first call.
    assert!(!spend(&Budget::new(0)));
    // And the shipped ceiling admits a real query: a budget of one spends
    // once and then refuses, so a ceiling of zero would refuse every query
    // before it started.
    let shipped = Budget::new(DECISION_BUDGET);
    assert!(spend(&shipped));
    assert_eq!(shipped.left(), DECISION_BUDGET - 1);
}

// THEORY: the-budget-declines
/// An exhausted budget declines on every subtyping path, and refutes on none.
///
/// The contract's only observable: a `Fails` is a value of the subject outside
/// the supertype, so a procedure that answered it because it ran out of steps
/// would be reporting a value nobody has. The unit above holds the counter; this
/// holds what every path does when the counter says no.
///
/// Three paths take the caller's budget and are driven here. The fourth is the
/// cell an equivalence query carries across its two directions, and it has its
/// own test below, because the query owns that cell rather than accepting one.
#[test]
fn the_budget_declines_on_every_subtyping_path() {
    use super::super::{NoLeafRelations, Relation, Schema, SeqShape};
    use std::sync::Arc;

    let pairs: Vec<(Schema, Schema)> = vec![
        // The recursion's own spend, before any rule is reached.
        (Schema::Int, Schema::Str),
        (
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            Schema::list(SeqShape::homogeneous(Schema::Str)),
        ),
        // The product rule: a fixed sequence against a union of sequences,
        // which is the one rule that spends a step of its own per branch.
        (
            Schema::list(SeqShape::fixed([Schema::Int, Schema::Str])),
            Schema::union([
                Schema::list(SeqShape::fixed([Schema::Int, Schema::Int])),
                Schema::list(SeqShape::fixed([Schema::Str, Schema::Str])),
            ]),
        ),
        // The disjointness reading, which builds the meet and asks emptiness
        // with the same budget: a refutation from it is a value, so an
        // exhausted emptiness must not become one.
        (
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            Schema::Complement(Arc::new(Schema::list(SeqShape::homogeneous(Schema::Int)))),
        ),
    ];

    for (sub, sup) in &pairs {
        // No budget at all: nothing is proven and nothing is refuted.
        let spent = Budget::new(0);
        assert_eq!(
            sub.subtype_relation(sup, &NoLeafRelations, &[], &spent),
            Relation::Unknown,
            "a query with no budget answered about {sub:?} <= {sup:?}"
        );

        // And at every budget short of the one the pair needs, the answer is
        // never a refutation -- it is the decline, or the proof it reached.
        let needed = sub.subtype_steps(sup);
        for budget in 1..needed.min(24) {
            let cell = Budget::new(budget);
            let answer = sub.subtype_relation(sup, &NoLeafRelations, &[], &cell);
            assert_ne!(
                answer,
                Relation::Fails,
                "{sub:?} <= {sup:?} refuted on a budget of {budget}, which \
                 reports a value the procedure never found"
            );
        }
    }
}

// THEORY: the-budget-declines
/// The fourth path: the cell an equivalence query carries across its two
/// directions.
///
/// The three above take the caller's budget, so the test hands each a small
/// one. Equivalence *owns* its cell -- both inclusions share it, so the query
/// cannot spend twice the ceiling and its answer does not depend on which
/// direction allocated first -- and a cell nothing can reach is a path nothing
/// can drive. The relation below is that body with the cell passed in, which is
/// the shape `subtype_relation` already has beside it.
///
/// The claim is **not** that a spent budget declines here, and writing it that
/// way is how this test first failed: the rules decline and the query then asks
/// the descriptor, which carries no budget because its cost is bounded by the
/// term rather than by the search. On a pair the descriptor can read, a zero
/// budget still answers, and answers correctly.
///
/// What must hold is that the budget never *changes* a decision: at every
/// allowance the answer is the one the ceiling gives, or it is `Unknown`. A
/// `Fails` out of exhaustion would name a value in one direction that nobody
/// found, and a `Holds` would be a proof nobody finished. A recursive pair is
/// drawn beside a descriptor-decided one, because a fixpoint is where the rules
/// answer alone -- the descriptor cannot hold a cycle -- and it is therefore the
/// only pair whose answer the budget can move at all.
#[test]
fn a_budgeted_equivalence_query_decides_the_same_or_declines() {
    use super::super::{DefIx, Field, NoLeafRelations, Relation, Schema, SeqShape};

    let field = |name: &str, schema, required| Field {
        name: name.into(),
        schema,
        required,
    };
    let list_of = |value, next| {
        Schema::Union(
            vec![
                Schema::NoneType,
                Schema::KeyedMap {
                    fields: vec![
                        field("value", value, true),
                        field("next", Schema::Ref(next), true),
                    ]
                    .into(),
                    defaults: Vec::new().into(),
                },
            ]
            .into(),
        )
    };
    let linked = [
        list_of(Schema::Int, DefIx::new(0)),
        list_of(Schema::Int, DefIx::new(1)),
    ];

    let cases: [(&str, Schema, Schema, &[Schema]); 2] = [
        // Two spellings of one recursive type: the rules decide it, so the
        // budget is the only thing that can stop them.
        (
            "a recursive pair",
            Schema::Ref(DefIx::new(0)),
            Schema::Ref(DefIx::new(1)),
            &linked,
        ),
        // And a pair the descriptor reads, which answers whatever the cell says.
        (
            "a pair the descriptor reads",
            Schema::list(SeqShape::homogeneous(Schema::Int)),
            Schema::list(SeqShape::homogeneous(Schema::Str)),
            &[],
        ),
    ];

    for (what, left, right, defs) in &cases {
        let decided =
            left.equivalence_relation(right, &NoLeafRelations, defs, &Budget::new(DECISION_BUDGET));
        // The detector: a pair the ceiling cannot decide would make every
        // assertion below hold about nothing.
        assert_ne!(
            decided,
            Relation::Unknown,
            "{what} is undecided at the ceiling"
        );

        // Where the crossing is belongs to the procedure's cost, so it is
        // searched for rather than written down; a pair with no crossing inside
        // this bound is one the sweep would step over.
        let crossing = (0..256u32).find(|budget| {
            left.equivalence_relation(right, &NoLeafRelations, defs, &Budget::new(*budget))
                != Relation::Unknown
        });
        let crossing = crossing.unwrap_or_else(|| panic!("{what} decides at no budget under 256"));

        for budget in 0..=crossing + 4 {
            let answer =
                left.equivalence_relation(right, &NoLeafRelations, defs, &Budget::new(budget));
            assert!(
                answer == decided || answer == Relation::Unknown,
                "{what} answered {answer:?} on a budget of {budget}, where the \
                 ceiling answers {decided:?}"
            );
        }
    }

    // And the two cases part on the crossing itself, so the sweep above is
    // asked of a query the budget moves and of one it does not.
    let recursive = (0..256u32).find(|budget| {
        cases[0].1.equivalence_relation(
            &cases[0].2,
            &NoLeafRelations,
            cases[0].3,
            &Budget::new(*budget),
        ) != Relation::Unknown
    });
    assert!(
        recursive.unwrap_or_default() > 0,
        "the rules decided for free"
    );
    assert_eq!(
        cases[1]
            .1
            .equivalence_relation(&cases[1].2, &NoLeafRelations, cases[1].3, &Budget::new(0)),
        Relation::Fails,
        "the descriptor needs a budget it does not take"
    );
}

/// The same obligation, asked of drawn pairs.
mod drawn {
    use super::super::{Budget, DECISION_BUDGET, NoLeafRelations, Relation};
    use crate::laws::decidable_schema;
    use proptest::prelude::*;

    proptest! {
        // A bounded shrink, so a broken obligation cannot turn a caught
        // mutation into a run that outlasts a sweep: see `budget::law`.
        #![proptest_config(ProptestConfig {
            max_shrink_time: 2_000,
            ..ProptestConfig::default()
        })]

        // THEORY: the-budget-declines
        /// The budget never changes a decision, over drawn pairs.
        ///
        /// The four chosen pairs above drive one path each. This asks every
        /// pair the decidable fragment draws: with no budget the query answers
        /// neither way, and at every allowance short of the one the pair
        /// needs the answer is the one the ceiling gives or it is the decline.
        /// A refutation is a value of the subject outside the supertype, and
        /// a budget that ran out found none.
        #[test]
        fn the_budget_declines_on_every_drawn_pair(
            sub in decidable_schema(),
            sup in decidable_schema(),
        ) {
            let none = Budget::new(0);
            prop_assert_eq!(
                sub.subtype_relation(&sup, &NoLeafRelations, &[], &none),
                Relation::Unknown
            );
            let ceiling = Budget::new(DECISION_BUDGET);
            let decided = sub.subtype_relation(&sup, &NoLeafRelations, &[], &ceiling);
            let needed = sub.subtype_steps(&sup);
            for budget in 1..needed.min(24) {
                let cell = Budget::new(budget);
                let answer = sub.subtype_relation(&sup, &NoLeafRelations, &[], &cell);
                prop_assert!(
                    answer == Relation::Unknown || answer == decided,
                    "{sub:?} <= {sup:?} answered {answer:?} on a budget of {budget} \
                     and {decided:?} at the ceiling"
                );
            }
        }
    }
}

// THEORY: the-budget-declines
/// The depth bound declines as the budget does, and gives its levels back.
///
/// Two recursive schemas whose bodies nest `p` and `q` lists around the back
/// edge meet the coinductive hypothesis only after `lcm(p, q)` levels of goals,
/// a few steps each, so the budget never stops them and the stack did. Under
/// [`MAX_DECISION_DEPTH`] the pair is proved; past it the query declines, and a
/// query after it on the same thread decides, which it could not if the
/// declining one had kept the levels it held. A chain of definitions nested
/// past the bound has its emptiness declined the same way, and proved under it.
///
/// On a thread of its own, wide enough for an unoptimized build, whose frames
/// are several times the shipped build's: the bound is what is under test, not
/// the stack it is sized for.
#[test]
fn the_depth_bound_declines_and_gives_its_levels_back() {
    use super::super::{DefIx, NoLeafRelations, Relation, Schema, SeqShape, Verdict};

    let nest = |levels: usize, leaf: Schema| {
        (0..levels).fold(leaf, |inner, _| Schema::list(SeqShape::homogeneous(inner)))
    };
    let cycles = move |p: usize, q: usize| {
        let defs = vec![
            nest(p, Schema::Ref(DefIx::new(0))),
            nest(q, Schema::Ref(DefIx::new(1))),
        ];
        Schema::Ref(DefIx::new(0)).subtype_relation(
            &Schema::Ref(DefIx::new(1)),
            &NoLeafRelations,
            &defs,
            &Budget::new(DECISION_BUDGET),
        )
    };
    // One body a hundred one-tuples deep a definition, each naming the one
    // before it, so the emptiness of the last descends a hundred levels a link.
    let chain = |links: usize| {
        let tuple = |inner| {
            Schema::tuple(SeqShape {
                prefix: vec![inner].into(),
                tail: None,
            })
        };
        let defs: Vec<Schema> = (0..links)
            .map(|link| {
                let leaf = link
                    .checked_sub(1)
                    .map_or(Schema::Int, |before| Schema::Ref(DefIx::new(before)));
                (0..100).fold(leaf, |inner, _| tuple(inner))
            })
            .collect();
        Schema::Ref(DefIx::new(links - 1)).verdict_rec(
            &NoLeafRelations,
            &defs,
            &mut Vec::new(),
            &Budget::new(DECISION_BUDGET),
        )
    };
    let shallow = (21 * 20, 3 * 100);
    let deep = (24 * 23, 8 * 100);
    assert!(shallow.0 < MAX_DECISION_DEPTH && MAX_DECISION_DEPTH < deep.0);
    assert!(shallow.1 < MAX_DECISION_DEPTH && MAX_DECISION_DEPTH < deep.1);
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || {
            assert_eq!(cycles(21, 20), Relation::Holds);
            assert_eq!(cycles(24, 23), Relation::Unknown);
            assert_eq!(cycles(21, 20), Relation::Holds, "the levels came back");
            assert_eq!(chain(3), Verdict::Inhabited);
            assert_eq!(chain(8), Verdict::Unknown);
            assert_eq!(chain(3), Verdict::Inhabited, "and from emptiness too");
        })
        .expect("the thread starts")
        .join()
        .expect("the thread answers");
}

/// The depth admits exactly [`MAX_DECISION_DEPTH`] levels and refuses the next,
/// and a refusal takes no level: the levels below it still close, the count is
/// back where the query found it once they have, and the next descent reaches
/// the same depth.
#[test]
fn the_depth_admits_its_own_number_of_levels_and_no_more() {
    fn deepest(budget: &Budget, level: u32) -> u32 {
        descending(budget, level, || deepest(budget, level + 1))
    }
    let budget = Budget::new(DECISION_BUDGET);
    assert_eq!(deepest(&budget, 0), MAX_DECISION_DEPTH);
    assert_eq!(
        deepest(&budget, 0),
        MAX_DECISION_DEPTH,
        "the levels came back"
    );
    assert_eq!(budget.left(), DECISION_BUDGET, "a level costs no step");
}
