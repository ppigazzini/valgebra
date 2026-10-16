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
///
/// Reading the message needs a live interpreter, because a `PyErr` renders
/// through one, so this row runs under the `interpreter-tests` feature. The two
/// rows that only ask whether a pattern is refused carry no such need and run
/// on every build.
#[cfg(feature = "interpreter-tests")]
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
        // Verbose mode, where `re` keeps a space and a `#` in a class as
        // members and this engine ignores the one and starts a comment at the
        // other -- turned on at the start, mid-pattern, or for one group.
        ("(?x)[ a]", "whitespace inside a character class"),
        ("(?x)[a\tb]", "whitespace inside a character class"),
        ("(?x)[#a]", "`#` inside a character class"),
        ("a(?x)[ b]", "whitespace inside a character class"),
        ("(?x)(a)[ b]", "whitespace inside a character class"),
        ("(?x:[ a])", "whitespace inside a character class"),
        ("(?ix)[ a]", "whitespace inside a character class"),
        ("(?x)(?-x:a)[ b]", "whitespace inside a character class"),
        // Turned on inside a group, for the rest of it: the flags and their `)`
        // are taken whole, and that `)` closes no group.
        ("((?x)[ a])", "whitespace inside a character class"),
        // A `(` and a `)` inside a class are members, which open, close and
        // set nothing.
        ("(?x)[(?-x) a]", "whitespace inside a character class"),
        ("((?x)[)][ a])", "whitespace inside a character class"),
        // Outside verbose mode a `#` is a member and starts no comment, so the
        // operator after it is reached.
        ("[#&&x]", "set intersection"),
        // Word-boundary escapes outside a class, which `re` reads as the
        // literal characters: `\<b\>` admits `"b"` here and `"<b>"` there.
        (r"\<b\>", "the start of a word"),
        (r"a\>", "the end of a word"),
        (r"\b{start}a", "a named word boundary"),
        (r"[a]\<", "the start of a word"),
        // A `+` after a quantifier: possessive to `re` from 3.11 and an error
        // before, a second repetition here. `a?+` refuses `"aa"` there and
        // admits it here. After a lazy quantifier `re` raises, and the scan
        // refuses it in the same words rather than read it.
        ("a?+", "possessive"),
        ("a*+", "possessive"),
        ("a++", "possessive"),
        (r"\d++\d", "possessive"),
        ("(?:a)?+", "possessive"),
        ("a+?+", "possessive"),
        // A count is a quantifier too, and taken whole: the `+` is read with
        // the cursor just past its `}`. An escape's braces are taken whole
        // before it, so the count after one is still a count.
        ("a{2}+", "possessive"),
        ("a{1,3}+", "possessive"),
        ("a{2,}+", "possessive"),
        (r"\p{L}{2}+", "possessive"),
        // Only an escape that takes braces takes them: `\d{2}` is a count, and
        // `\x41` holds no brace to take, so what follows each is read.
        (r"\d{2}+", "possessive"),
        (r"\x41a?+", "possessive"),
        // Whitespace in a count's braces, which this engine skips in every
        // mode and `re` reads as text: `a{1, 3}` admits `"aa"` here and
        // `"a{1, 3}"` there. Verbose mode changes nothing about it in `re`,
        // which reads a brace that is not exactly a count as a literal and
        // then drops the whitespace after it.
        ("a{1, 3}", "whitespace inside the counted repetition"),
        ("a{1 ,3}", "whitespace inside the counted repetition"),
        ("a{ 2}", "whitespace inside the counted repetition"),
        ("a{2 }", "whitespace inside the counted repetition"),
        ("a{1,\t3}", "whitespace inside the counted repetition"),
        ("a{\u{3000}2}", "whitespace inside the counted repetition"),
        ("(?x)a{1, 3}", "whitespace inside the counted repetition"),
        // In verbose mode this engine also skips whitespace between digits,
        // after the comma, and a comment, which runs to its line's end and
        // can hold the `}` the count does not close at.
        ("(?x)a{1 2}", "whitespace inside the counted repetition"),
        ("(?x)a{2, }", "whitespace inside the counted repetition"),
        ("(?x)a{2#}\n}", "a comment inside the counted repetition"),
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
        // The escapes this scan leaves alone: inside a class an escaped `<` or
        // `>` is a member to `re` and a parse error here, which the compile
        // reports loudly; a `\b` is a boundary in both; and a doubled
        // backslash before a `<` is a literal backslash and a literal `<`.
        r"[\<\>]",
        r"\bword\b",
        r"\\<",
        r"a\{b",
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
        // Outside verbose mode a space and a `#` are members to both engines,
        // and so is an escaped one inside it.
        "[ a]",
        "[#a]",
        r"(?x)[\ a]",
        r"(?x)[\#a]",
        // Verbose mode turned off for a group, or never turned on: `(?:` and a
        // named group are groups, not flags.
        "(?x)(?-x:[ a])",
        "(?i)(?x:a)[ b]",
        "(?:x)[ a]",
        r"(?P<x>a)[ b]",
        // A comment runs to the end of the line, and a `[` in it opens nothing.
        "(?x) a # a comment [ that opens nothing\n b",
        // Lazy quantifiers, which both engines read alike, and a quantifier
        // after a character that followed one: the flag lasts one character.
        "a+?",
        "a??",
        "a*?",
        "a{2,3}?",
        "a?b+",
        "a+b+",
        // Counts both engines read alike, and whitespace in verbose mode
        // before a count rather than inside it, which both skip.
        "a{2}",
        "a{2,}",
        "a{2,3}",
        "(?x)a {2}",
        // An escape's braces are the escape's, taken whole, so the `+` after
        // them is the first quantifier: `\x{41}` is a count to nobody. `re`
        // refuses these escapes, which is the loud direction.
        r"\x{41}+",
        r"\u{41}+",
        r"\U{41}+",
        r"\P{L}+",
        // Inside a class a `+`, a `?` and a brace are members to both.
        "[a?+]",
        "[a{1, 3}]",
        // An escaped brace is a literal to both, and so is what follows it.
        r"a\{1, 3}",
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

/// A brace this engine reads as no count is the compile's to refuse.
///
/// `re` reads each of these as text and this engine refuses each at the parse,
/// which is the loud direction: calling one a count with whitespace in it
/// would say this engine reads it, which it does not. After a brace that is no
/// count, a `+` follows no quantifier.
#[test]
fn a_brace_this_engine_reads_as_no_count_is_left_to_the_compile() {
    for pattern in [
        "a{1 2}",
        "a{2, }",
        "a{, 3}",
        "a{,3}",
        "a{}",
        "a{}+",
        "x{a}",
        "a{",
        "a{1",
        // An escape's braces are taken whole whatever they hold, and a `#`
        // starts a comment in verbose mode alone, which runs to its line's end
        // past any `}` in it.
        r"\p{a?+}",
        "a{2#\n}",
        "(?x)a{2#}5}",
    ] {
        assert!(
            reject_reserved_class_syntax(pattern).is_ok(),
            "{pattern} was refused, and this engine's parse refuses it with its own error"
        );
    }
}

/// The message carries the position, so a long pattern says where to look.
///
/// Reading it needs a live interpreter, as the row above does.
#[cfg(feature = "interpreter-tests")]
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
