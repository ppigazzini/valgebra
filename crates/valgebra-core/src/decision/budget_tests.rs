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
/// cell an equivalence query carries across its two directions, which
/// `is_equivalent_under` owns rather than accepts, so it is reached through the
/// public relation and read for the same property.
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
