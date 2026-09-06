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

#[test]
fn compile_pattern_rejects_an_invalid_pattern() {
    assert!(compile_pattern("(unclosed").is_err());
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
}
