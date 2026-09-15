//! One row per reserved form, and one per form that only looks like it.

use super::*;

/// Each operator is refused, by the description `re` warns with.
#[test]
fn a_class_set_operator_is_refused() {
    for (pattern, said) in [
        (r"[\w--\d]", "set difference"),
        (r"[\w&&\d]", "set intersection"),
        (r"[\w~~\d]", "set symmetric difference"),
        (r"[a[bc]]", "nested set"),
    ] {
        let err = reject_reserved_class_syntax(pattern)
            .expect_err("a reserved form is refused rather than read");
        let message = err.to_string();
        assert!(
            message.contains(said),
            "{pattern} was refused as {message}, which does not say {said}"
        );
    }
}

/// The same characters outside a class, and the positions where `re` reads them
/// as literal, are left alone: a refusal wider than the warning it mirrors is a
/// pattern the two engines agree on, turned away.
#[test]
fn a_form_the_two_engines_agree_on_is_left_alone() {
    for pattern in [
        // Outside a class, three literals to both engines.
        "a--b",
        "a&&b",
        "a~~b",
        // At the start of a class, and after `^`, the first `-` is literal.
        "[--/]",
        "[^--/]",
        // Escaped, so neither engine reads an operator.
        r"[\w\-\-\d]",
        // A class that holds `]` first, which is literal in both.
        r"[]-]",
        // A POSIX class, which the refinements page names as a divergence and
        // shows the two readings of: loud enough to port against, unlike the
        // operators above.
        r"[[:alpha:]]",
        r"[[:alpha:][:digit:]]",
        r"[^[:space:]]+",
        // The ordinary patterns, none of which carry a doubled operator.
        r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
        r"\d{4}-\d{2}-\d{2}",
    ] {
        assert!(
            reject_reserved_class_syntax(pattern).is_ok(),
            "{pattern} was refused, and `re` warns about none of it"
        );
    }
}
