use super::{DECISION_BUDGET, spend};
use std::cell::Cell;

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
