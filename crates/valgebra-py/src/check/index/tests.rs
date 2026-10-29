use super::*;

#[test]
fn compile_pattern_anchors_the_whole_string() {
    // Anchoring gives `re.fullmatch` semantics: the whole string must match,
    // and the non-capturing wrap keeps an alternation from escaping it.
    let alternation = compile_pattern("a|b").expect("valid pattern");
    assert!(alternation.is_match("a"));
    assert!(alternation.is_match("b"));
    assert!(!alternation.is_match("xa"));
    assert!(!alternation.is_match("ab"));
    let digits = compile_pattern("[0-9]+").expect("valid pattern");
    assert!(digits.is_match("123"));
    assert!(!digits.is_match("12a"));
    assert!(!digits.is_match(""));
}

/// A pattern no index holds compiles once on a thread, a refusal among them,
/// and answers every text after from what it kept, until the query forgets it.
#[test]
fn a_pattern_no_index_holds_compiles_once_until_forgotten() {
    assert!(matches_unindexed("a.*", "abc"));
    assert!(!matches_unindexed("a.*", "xa"));
    assert_eq!(
        unindexed_held(),
        1,
        "the second text reads the kept pattern"
    );
    assert!(!matches_unindexed("(", "("), "a refusal matches nothing");
    assert!(!matches_unindexed("(", ""));
    assert_eq!(unindexed_held(), 2, "a refusal is kept beside the pattern");
    assert!(matches_unindexed("b|c", "c"));
    assert!(
        !matches_unindexed("b|c", "bc"),
        "the kept pattern is anchored"
    );
    forget_unindexed();
    assert_eq!(unindexed_held(), 0);
    assert!(matches_unindexed("a.*", "a"));
}

/// A counted Unicode class written as an ordinary bound compiles: the program
/// is past the crate's default and inside [`PATTERN_SIZE_LIMIT`], and one four
/// times larger is refused.
#[test]
fn a_counted_unicode_class_compiles_within_the_pattern_size_limit() {
    let word = compile_pattern(r"\w{1,255}").expect("a counted class compiles");
    assert!(word.is_match(&"é".repeat(255)));
    assert!(!word.is_match(&"a".repeat(256)));
    assert!(!word.is_match(""));
    assert!(compile_pattern(r"\w{1,255}@\w{1,255}").is_ok());
    assert!(compile_pattern(r"\w{1,1000}").is_err());
}

/// A validator's patterns are bounded together by what each keeps: two
/// counted Unicode classes fit, and a third is refused with the pattern that
/// takes them past the bound named. A text met twice is one program, and each
/// place the tree holds a pattern finds its compiled one.
#[test]
fn a_schemas_patterns_compile_within_one_allowance_together() {
    let refined =
        |pattern: &str| Schema::refine(Schema::Str, vec![Constraint::Regex(pattern.to_owned())]);
    let words = |count: usize| {
        Schema::Union(
            (0..count)
                .map(|at| refined(&format!(r"\w{{1,255}}x{at}")))
                .collect::<Vec<_>>()
                .into(),
        )
    };
    let two = words(2);
    let compiled = compile_patterns(&two, &[]).expect("two counted classes fit");
    assert_eq!(compiled.len(), 2);
    for pattern in two.patterns() {
        assert!(
            compiled[&(pattern.as_ptr() as usize)]
                .is_match(&format!("é{}", &pattern[pattern.len() - 2..]))
        );
    }
    let refused = compile_patterns(&words(3), &[]).expect_err("a third one passes the bound");
    assert!(
        refused.contains("at most 32 MiB together") && refused.contains("{1,255}x"),
        "{refused}"
    );
    // One text held three times, once in a definition, beside a second text:
    // four places, two programs, each charged once.
    let thrice = Schema::Union(
        vec![
            refined(r"\w{1,255}x0"),
            refined(r"\w{1,255}x0"),
            refined(r"\w{1,255}x1"),
        ]
        .into(),
    );
    let compiled = compile_patterns(&thrice, &[refined(r"\w{1,255}x0")])
        .expect("two programs, each charged once");
    assert_eq!(compiled.len(), 4);
    let invalid = compile_patterns(&refined("(unclosed"), &[]).expect_err("no pattern");
    assert!(
        invalid.starts_with("invalid regular expression"),
        "{invalid}"
    );
}

#[test]
fn compile_pattern_rejects_an_invalid_pattern() {
    assert!(compile_pattern("(unclosed").is_err());
}

#[test]
fn compile_pattern_refuses_a_pattern_that_parses_only_wrapped() {
    // `a)|(b` closes the wrapper's group and opens one the wrapper closes, so
    // the anchored form parses, and its alternation escapes both anchors: it
    // matched `xxb`. The pattern alone does not parse, and neither does this.
    for pattern in ["a)|(b", ")|(", "a)(b"] {
        assert!(compile_pattern(pattern).is_err(), "{pattern} compiled");
    }
    // A verbose pattern ending in a comment parses alone and not wrapped, which
    // is the other way round, and it compiles.
    let commented = compile_pattern("(?x) a # a note").expect("a verbose pattern");
    assert!(commented.is_match("a"));
    assert!(!commented.is_match("xa"));
}

// Tests that need a live interpreter; compiled and run only under the
// `interpreter-tests` feature, which links an embedded Python.
#[cfg(feature = "interpreter-tests")]
mod interpreter {
    use super::super::*;
    use crate::check::walk::literal_matches;
    use pyo3::types::{PyBool, PyString};
    use valgebra_core::ConstIx;

    #[test]
    fn union_plan_decide_agrees_with_the_linear_scan() {
        // The literal-union fast path must never disagree with the linear scan
        // it replaces: whenever `decide` commits to an answer, that answer
        // equals membership decided by `literal_matches` over the same pooled
        // literals.
        Python::attach(|py| {
            let pool: Vec<Py<PyAny>> = vec![
                1i64.into_pyobject(py).unwrap().into_any().unbind(),
                7i64.into_pyobject(py).unwrap().into_any().unbind(),
                PyString::new(py, "ok").into_any().unbind(),
                PyString::new(py, "yes").into_any().unbind(),
            ];
            let members: Vec<Schema> = (0..pool.len())
                .map(|i| Schema::Literal(ConstIx::new(i)))
                .collect();
            let plan = literal_union_plan(py, &members, &pool).expect("every member is a literal");

            // Edge types the plan must defer on (returning `None`): a big
            // integer outside i64, a bool (exact-type rule), a float, and a NaN.
            let candidates: Vec<Bound<'_, PyAny>> = vec![
                1i64.into_pyobject(py).unwrap().into_any(),
                7i64.into_pyobject(py).unwrap().into_any(),
                42i64.into_pyobject(py).unwrap().into_any(),
                (1i128 << 70).into_pyobject(py).unwrap().into_any(),
                PyBool::new(py, true).to_owned().into_any(),
                1.0f64.into_pyobject(py).unwrap().into_any(),
                f64::NAN.into_pyobject(py).unwrap().into_any(),
                PyString::new(py, "ok").into_any(),
                PyString::new(py, "no").into_any(),
            ];

            let mut committed = 0;
            for value in &candidates {
                if let Some(decided) = plan.decide(&Value::Py(value)) {
                    committed += 1;
                    let scan = pool
                        .iter()
                        .any(|lit| literal_matches(value, lit.bind(py)).unwrap_or(false));
                    assert_eq!(decided, scan, "fast-path decision disagreed with the scan");
                }
            }
            // The exact int and str values commit, so the check is not vacuous.
            assert!(
                committed >= 4,
                "expected the fast path to commit on exact values"
            );
        });
    }

    /// A union of records one required key pins to constants no two share is
    /// tagged by it: an exact tag names its record or none, and any other tag
    /// is the scan's. Records sharing a constant, a key pinned to no constant,
    /// and fewer records than reading the tag first pays for are no tagged
    /// union.
    #[test]
    fn a_tagged_plan_names_the_record_its_tag_pins() {
        use valgebra_core::Field;

        Python::attach(|py| {
            let pool: Vec<Py<PyAny>> = vec![
                PyString::new(py, "a").into_any().unbind(),
                1i64.into_pyobject(py).unwrap().into_any().unbind(),
                PyString::new(py, "b").into_any().unbind(),
                PyString::new(py, "c").into_any().unbind(),
            ];
            let record = |tag: Schema| Schema::KeyedMap {
                fields: vec![
                    Field {
                        name: "v".into(),
                        schema: Schema::Int,
                        required: false,
                    },
                    Field {
                        name: "kind".into(),
                        schema: tag,
                        required: true,
                    },
                ]
                .into(),
                defaults: Vec::new().into(),
            };
            let literal = |slot: usize| Schema::Literal(ConstIx::new(slot));
            let both = Schema::Union(vec![literal(1), literal(2)].into());
            let three = [record(literal(0)), record(both), record(literal(3))];
            assert_eq!(three.len(), TAGGED_MIN_RECORDS);
            let plan = tagged_union_plan(py, &three, &pool).expect("one key pins every record");
            assert!(tagged_union_plan(py, &three[1..], &pool).is_none());
            assert_eq!(plan.key.bind(py).to_str().unwrap(), "kind");
            let tag = |value: Bound<'_, PyAny>| plan.branch(&value);
            assert_eq!(
                tag(PyString::new(py, "a").into_any()),
                Some(Tagged::Branch(0))
            );
            assert_eq!(
                tag(PyString::new(py, "b").into_any()),
                Some(Tagged::Branch(1))
            );
            assert_eq!(
                tag(1i64.into_pyobject(py).unwrap().into_any()),
                Some(Tagged::Branch(1))
            );
            assert_eq!(
                tag(PyString::new(py, "z").into_any()),
                Some(Tagged::Nothing)
            );
            assert_eq!(tag(PyBool::new(py, true).to_owned().into_any()), None);
            assert_eq!(
                tag((1i128 << 70).into_pyobject(py).unwrap().into_any()),
                None
            );

            let shared = [record(literal(0)), record(literal(0)), record(literal(3))];
            assert!(tagged_union_plan(py, &shared, &pool).is_none());
            let unpinned = [record(literal(0)), record(Schema::Str), record(literal(3))];
            assert!(tagged_union_plan(py, &unpinned, &pool).is_none());
        });
    }
}
