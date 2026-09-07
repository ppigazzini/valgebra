use super::*;

fn violation(code: &'static str, path: Vec<PathSegment>) -> Violation {
    Violation {
        code,
        path,
        expected: "int".to_owned(),
        value_summary: "'x'".to_owned(),
    }
}

/// A walk that reports a non-member without a violation is an internal
/// invariant break, and the two profiles answer it differently on purpose: a
/// debug build traps so the break is found, and a release build degrades to a
/// well-formed error rather than panicking across the language boundary.
///
/// This pins the debug half, which is the one a test profile can observe. The
/// release half needs no case of its own any more: the boundary reads the
/// first violation through `split_first`, so there is no index to be wrong --
/// the degradation is what the `Option` already means.
#[test]
#[should_panic(expected = "into_pyerr needs a failure")]
fn an_empty_violation_list_trips_the_debug_assert() {
    Python::attach(|py| {
        let _ = into_pyerr(py, Vec::new());
    });
}

#[test]
fn into_pyerr_maps_violations_to_the_structured_attributes() {
    Python::attach(|py| {
        // The six are built by the hooks, which the module installs on import.
        // Nothing imports the module here, so this test installs them itself --
        // through the same function, so what it reads is what a caller reads.
        install_lazy_attributes(py).expect("the hooks the module installs");
        let violations = vec![
            violation("int_type", vec![PathSegment::Key("a".to_owned())]),
            violation("missing", vec![PathSegment::Index(2)]),
        ];
        let err = into_pyerr(py, violations);
        let value = err.value(py);

        // The scalar attributes mirror the first violation; the path is the
        // built tuple.
        assert_eq!(
            value.getattr("code").unwrap().extract::<String>().unwrap(),
            "int_type"
        );
        assert_eq!(
            value
                .getattr("expected")
                .unwrap()
                .extract::<String>()
                .unwrap(),
            "int"
        );
        let path: Vec<String> = value.getattr("path").unwrap().extract().unwrap();
        assert_eq!(path, vec!["a".to_owned()]);

        // `errors` carries one item per violation, in order, each with its code.
        let errors = value.getattr("errors").unwrap();
        assert_eq!(errors.len().unwrap(), 2);
        let second_code: String = errors
            .get_item(1)
            .unwrap()
            .get_item("code")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(second_code, "missing");
    });
}
