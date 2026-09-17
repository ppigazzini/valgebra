use super::{DECISION_BUDGET, spend};
use std::cell::Cell;

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
    let budget = Cell::new(2u32);
    assert!(spend(&budget));
    assert_eq!(budget.get(), 1);
    assert!(spend(&budget));
    assert_eq!(budget.get(), 0);
    // Exhausted: refuses, and does not wrap round to a fresh budget.
    assert!(!spend(&budget));
    assert_eq!(budget.get(), 0);
    assert!(!spend(&budget));
    assert_eq!(budget.get(), 0);

    // A budget of zero refuses on its first call.
    assert!(!spend(&Cell::new(0)));
    // And the shipped ceiling admits a real query: a budget of one spends
    // once and then refuses, so a ceiling of zero would refuse every query
    // before it started.
    let shipped = Cell::new(DECISION_BUDGET);
    assert!(spend(&shipped));
    assert_eq!(shipped.get(), DECISION_BUDGET - 1);
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
        let spent = Cell::new(0u32);
        assert_eq!(
            sub.subtype_relation(sup, &NoLeafRelations, &[], &spent),
            Relation::Unknown,
            "a query with no budget answered about {sub:?} <= {sup:?}"
        );

        // And at every budget short of the one the pair needs, the answer is
        // never a refutation -- it is the decline, or the proof it reached.
        let needed = sub.subtype_steps(sup);
        for budget in 1..needed.min(24) {
            let cell = Cell::new(budget);
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
            left.equivalence_relation(right, &NoLeafRelations, defs, &Cell::new(DECISION_BUDGET));
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
            left.equivalence_relation(right, &NoLeafRelations, defs, &Cell::new(*budget))
                != Relation::Unknown
        });
        let crossing = crossing.unwrap_or_else(|| panic!("{what} decides at no budget under 256"));

        for budget in 0..=crossing + 4 {
            let answer =
                left.equivalence_relation(right, &NoLeafRelations, defs, &Cell::new(budget));
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
            &Cell::new(*budget),
        ) != Relation::Unknown
    });
    assert!(
        recursive.unwrap_or_default() > 0,
        "the rules decided for free"
    );
    assert_eq!(
        cases[1]
            .1
            .equivalence_relation(&cases[1].2, &NoLeafRelations, cases[1].3, &Cell::new(0)),
        Relation::Fails,
        "the descriptor needs a budget it does not take"
    );
}
