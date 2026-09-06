use super::WalkMode;

/// Both predicates are single comparisons over an ordered discriminant, so
/// they are pinned over every variant: a reordering that changes what a mode
/// means fails here rather than silently switching the walk's behaviour.
#[test]
fn every_mode_answers_both_predicates() {
    assert!(WalkMode::Explain.explains());
    assert!(WalkMode::ExplainFailFast.explains());
    assert!(!WalkMode::Fast.explains());

    assert!(!WalkMode::Explain.stops_at_first());
    assert!(WalkMode::ExplainFailFast.stops_at_first());
    assert!(WalkMode::Fast.stops_at_first());

    assert_eq!(WalkMode::explaining(true), WalkMode::ExplainFailFast);
    assert_eq!(WalkMode::explaining(false), WalkMode::Explain);
}
