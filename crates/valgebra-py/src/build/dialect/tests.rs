//! One row per form, refused or left alone, and the reason each is where it is.
//!
//! The corpus is the instrument. This is a character scan, so what decides it
//! is *position* -- which character sits where, and what the one before it did
//! to the cursor -- and a handful of short patterns exercise almost none of
//! that. Every row below is a pattern whose answer changes if the scan's
//! bookkeeping is wrong by one, and each is checked against what `re` does
//! with it, which is recorded beside it.

use super::*;

/// Every form this refuses, with the operator the message has to name.
///
/// `re` answers these three ways, which is the argument for refusing all of
/// them: it raises on a doubled hyphen after a member, warns about the two
/// symbol operators and about a `[` in a class's first place, and says nothing
/// at all about a `[` anywhere else.
#[test]
fn a_reserved_form_is_refused_by_the_operator_it_would_have_read() {
    for (pattern, said) in [
        // The three operators, each between two classes.
        (r"[\w--\d]", "set difference"),
        (r"[\w&&\d]", "set intersection"),
        (r"[\w~~\d]", "set symmetric difference"),
        (r"[x&&y]", "set intersection"),
        (r"[ab--cd]", "set difference"),
        // A nested set, which `re` reads as four literal characters.
        (r"[a[bc]]", "nested set"),
        // A `]` in the first member's place is that member, so it takes the
        // position where a doubled character would be literal and the operator
        // after it is an operator. `re` warns at the same position.
        (r"[]&&x]", "set intersection"),
        (r"[]~~x]", "set symmetric difference"),
        (r"[]--x]", "set difference"),
        (r"[^]&&x]", "set intersection"),
        // A POSIX class is skipped whole, and an operator after it is still an
        // operator -- which is what makes the skip land exactly on its close
        // rather than near it.
        (r"[[:alpha:]&&x]", "set intersection"),
        (r"[[:alpha:]--x]", "set difference"),
        (r"[[:alpha:]~~x]", "set symmetric difference"),
        (r"[[:alpha:][:digit:]&&x]", "set intersection"),
        (r"[[:alpha:]a[bc]]", "nested set"),
        // An escape carries its character, so the `[` below is not a nested set
        // and the `&&` after it is reached with the cursor in the right place.
        (r"[\[&&x]", "set intersection"),
        (r"[\]&&x]", "set intersection"),
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

/// Every form the two engines agree on, which a refusal must not reach.
///
/// A refusal wider than the divergence it mirrors turns away a pattern that was
/// never ambiguous, and that is the failure this half exists to catch. `re`
/// warns about none of these and raises on none of them.
#[test]
fn a_form_the_two_engines_agree_on_is_left_alone() {
    for pattern in [
        // Outside a class, three literals to both engines.
        "a--b",
        "a&&b",
        "a~~b",
        // Between two classes, which is still outside one.
        r"[abc]--[def]",
        // In the first member's place, where a doubled character is literal to
        // both engines: `[--/]` is the range from `-` to `/` either way.
        "[--/]",
        "[^--/]",
        "[&&]",
        "[--]",
        // A `]` first, then a single hyphen, which is a member.
        r"[]-]",
        // Escaped, so neither engine reads an operator.
        r"[\w\-\-\d]",
        r"[\--x]",
        // A POSIX class, the divergence `docs/05-refinements.md` names with
        // both readings shown -- and the only spelling of one there is.
        r"[[:alpha:]]",
        r"[[:alpha:][:digit:]]",
        r"[^[:space:]]+",
        r"[a[:alpha:]b]",
        // The ordinary patterns, none of which carries a doubled operator.
        r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
        r"\d{4}-\d{2}-\d{2}",
        r"\p{L}+",
        "",
    ] {
        assert!(
            reject_reserved_class_syntax(pattern).is_ok(),
            "{pattern} was refused, and `re` agrees with this engine about it"
        );
    }
}

/// A pattern that does not balance is the compile's to refuse, and this scan's
/// only duty is to finish.
///
/// The scan is not a parser and must not become one: it reads position, and a
/// class or a POSIX name that never closes has no position to read past. Each
/// of these returns rather than running off the end or looping, and the
/// `Regex::new` that follows gives the caller the parse error.
#[test]
fn an_unbalanced_pattern_is_left_to_the_compile() {
    for pattern in [
        "[",
        "[^",
        "[]",
        "[abc",
        r"[[:alpha",
        r"[[:alpha:",
        r"[[:",
        r"[[",
        "]",
        "\\",
        r"[\",
    ] {
        // The answer is the compile's; what is asserted is that there is one.
        let _ = reject_reserved_class_syntax(pattern);
    }
}

/// The message carries the position, so a long pattern says where to look.
#[test]
fn a_refusal_names_the_position_and_the_pattern() {
    let pattern = r"^abc[\w&&\d]$";
    let err = reject_reserved_class_syntax(pattern).expect_err("the class carries an intersection");
    let message = err.to_string();
    // The index of the first character of the operator, which is the index
    // `re` names in its own warning about the same pattern.
    assert!(
        message.contains("position 7"),
        "{message} does not name where the operator sits"
    );
    // The pattern as a reader would paste it back, which is what makes a
    // message about a pattern built by a frontend traceable to its source.
    assert!(
        message.contains(&format!("{pattern:?}")),
        "{message} does not carry the pattern"
    );
}
