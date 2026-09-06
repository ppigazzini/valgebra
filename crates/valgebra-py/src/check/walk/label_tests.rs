use super::{BranchLabels, UNION_LABEL_LIMIT};

/// The list a union's `expected` reads is bounded, and says so when it is
/// short.
///
/// A wide union -- an error-code table, a currency list -- has more branches
/// than a message can carry, so the list stops and ends in `...`. The two
/// halves are the claim: a reader must not be shown a truncated list as if
/// it were the whole set, and must not be shown `...` after a complete one.
/// Neither needs an interpreter: the labels arrive as strings.
#[test]
fn the_branch_list_stops_at_its_limit_and_marks_that_it_did() {
    let mut labels = BranchLabels::new();
    labels.push("int".to_owned());
    labels.push("str".to_owned());
    assert_eq!(labels.render(), "one of: int, str");

    // Exactly the limit is a complete list: the boundary belongs to the
    // labels, not to the ellipsis.
    let mut full = BranchLabels::new();
    for i in 0..UNION_LABEL_LIMIT {
        full.push(format!("b{i}"));
    }
    let rendered = full.render();
    assert!(!rendered.ends_with(", ..."), "{rendered} is not truncated");
    assert_eq!(rendered.matches(", ").count(), UNION_LABEL_LIMIT - 1);

    // One past it is short, and the label is dropped rather than kept.
    full.push("dropped".to_owned());
    let rendered = full.render();
    assert!(rendered.ends_with(", ..."), "{rendered} is truncated");
    assert!(!rendered.contains("dropped"));
    assert_eq!(rendered.matches(", ").count(), UNION_LABEL_LIMIT);

    // An empty union renders its prefix and nothing else, which is what a
    // renderer returning a fixed string would also do for every other case.
    assert_eq!(BranchLabels::new().render(), "one of: ");
}
