use super::WalkMode;

/// The predicates are single comparisons over an ordered discriminant, so
/// they are pinned over every variant: a reordering that changes what a mode
/// means fails here rather than silently switching the walk's behavior.
#[test]
fn every_mode_answers_every_predicate() {
    assert!(WalkMode::Explain.explains());
    assert!(WalkMode::ExplainLimited.explains());
    assert!(WalkMode::ExplainFailFast.explains());
    assert!(!WalkMode::Fast.explains());
    assert!(!WalkMode::Settle.explains());

    assert!(!WalkMode::Explain.stops_at_first());
    assert!(!WalkMode::ExplainLimited.stops_at_first());
    assert!(WalkMode::ExplainFailFast.stops_at_first());
    assert!(WalkMode::Fast.stops_at_first());
    assert!(WalkMode::Settle.stops_at_first());

    assert!(!WalkMode::Explain.may_stop());
    assert!(WalkMode::ExplainLimited.may_stop());
    assert!(WalkMode::ExplainFailFast.may_stop());
    assert!(WalkMode::Fast.may_stop());
    assert!(WalkMode::Settle.may_stop());

    assert_eq!(WalkMode::explaining(1), WalkMode::ExplainFailFast);
    assert_eq!(WalkMode::explaining(2), WalkMode::ExplainLimited);
    assert_eq!(
        WalkMode::explaining(usize::MAX - 1),
        WalkMode::ExplainLimited
    );
    assert_eq!(WalkMode::explaining(usize::MAX), WalkMode::Explain);
}
