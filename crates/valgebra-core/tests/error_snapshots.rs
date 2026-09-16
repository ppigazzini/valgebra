//! Snapshot coverage of the violation message format.
//!
//! The one-line rendering of a [`Violation`] is part of the error-model
//! contract (the message style guide in `docs/08-error-model.md`). This locks the
//! exact format across a representative corpus so any change to it is reviewed
//! in the snapshot diff, never silently accepted.
//!
//! The rows are built here rather than taken from a walk, because the subject
//! is the *format* and a walk would tie the corpus to which failures a schema
//! can reach. The cost is that a row's code is a string this file chose: one
//! that was renamed, or never existed, renders perfectly and pins a format for
//! a failure no caller can meet. It happened -- `extra_key` sat here while the
//! walk wrote `extra_forbidden` -- so `tests/test_use_case_ledger.py` reads
//! these codes against the ones the tree emits.

use valgebra_core::{PathSegment, Violation};

fn violation(code: &'static str, path: Vec<PathSegment>, expected: &str, value: &str) -> Violation {
    Violation {
        code,
        path,
        expected: expected.to_owned(),
        value_summary: value.to_owned(),
    }
}

#[test]
fn violation_message_format() {
    let key = |name: &str| PathSegment::Key(name.into());
    let corpus = [
        violation("int_type", vec![], "int", "'x'"),
        violation(
            "string_type",
            vec![key("name"), PathSegment::Index(2)],
            "str",
            "5",
        ),
        violation(
            "missing_key",
            vec![key("age")],
            "required key \"age\"",
            "missing",
        ),
        violation(
            "extra_forbidden",
            vec![key("extra")],
            "no unexpected key",
            "'extra'",
        ),
        violation(
            "union_error",
            vec![key("status")],
            "one of: int, str",
            "1.5",
        ),
        violation("list_length", vec![], "list of length 2", "[1]"),
        violation("unexpected_match", vec![], "not int", "5"),
        violation(
            "recursion_loop",
            vec![key("next")],
            "a finite (non-cyclic) value",
            "{...}",
        ),
    ];
    let rendered = corpus
        .iter()
        .map(Violation::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    insta::assert_snapshot!(rendered);
}
