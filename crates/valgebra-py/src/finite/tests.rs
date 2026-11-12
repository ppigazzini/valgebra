use std::sync::Arc;

use pyo3::types::PyList;
use valgebra_core::{ConstIx, DefIx, Field, PredIx, SeqShape, Spelling};

use super::*;

fn pool(py: Python<'_>, code: &str) -> Vec<Py<PyAny>> {
    py.eval(&std::ffi::CString::new(code).unwrap(), None, None)
        .unwrap()
        .cast_into::<PyList>()
        .unwrap()
        .iter()
        .map(Bound::unbind)
        .collect()
}

fn reprs(py: Python<'_>, values: Option<Vec<Bound<'_, PyAny>>>) -> Option<Vec<String>> {
    let _ = py;
    values.map(|values| {
        values
            .iter()
            .map(|value| value.repr().unwrap().to_string())
            .collect()
    })
}

fn literal(at: usize) -> Schema {
    Schema::Literal(ConstIx::new(at))
}

/// A subject of constants is enumerated as the Python values it holds: each
/// constant that is its whole set, both zeros for a float zero, and the
/// products a fixed sequence, a sequence of an exact length and a closed record
/// make of them.
#[test]
fn a_subject_of_constants_is_its_values() {
    Python::attach(|py| {
        let literals = pool(py, "[1, 0.0, __import__('decimal').Decimal(1), 'a', 2.5]");
        let read = |schema: &Schema| reprs(py, values_of(py, schema, &literals).unwrap());
        assert_eq!(read(&literal(0)), Some(vec!["1".to_owned()]));
        assert_eq!(
            read(&literal(1)),
            Some(vec!["0.0".to_owned(), "-0.0".to_owned()])
        );
        assert_eq!(read(&literal(4)), Some(vec!["2.5".to_owned()]));
        assert_eq!(read(&literal(2)), None, "a Decimal is no whole set");
        assert_eq!(
            read(&Schema::Bool),
            Some(vec!["False".to_owned(), "True".to_owned()])
        );
        assert_eq!(read(&Schema::NoneType), Some(vec!["None".to_owned()]));
        assert_eq!(read(&Schema::Nothing), Some(vec![]));
        assert_eq!(read(&Schema::Int), None);
        let either = Schema::Union(vec![literal(0), literal(3)].into());
        assert_eq!(read(&either), Some(vec!["1".to_owned(), "'a'".to_owned()]));
        let pair = Schema::list(SeqShape::fixed([literal(0), either.clone()]));
        assert_eq!(
            read(&pair),
            Some(vec!["[1, 1]".to_owned(), "[1, 'a']".to_owned()])
        );
        let couple = Schema::tuple(SeqShape::fixed([literal(3), literal(0)]));
        assert_eq!(read(&couple), Some(vec!["('a', 1)".to_owned()]));
        let any_list = Schema::list(SeqShape::homogeneous(Schema::Anything(Spelling::Any)));
        let exactly = |n| {
            Schema::refine(
                any_list.clone(),
                vec![Constraint::MinLen(n), Constraint::MaxLen(n)],
            )
        };
        let two_of = Schema::Intersection(
            vec![Schema::list(SeqShape::homogeneous(literal(0))), exactly(2)].into(),
        );
        assert_eq!(read(&two_of), Some(vec!["[1, 1]".to_owned()]));
        let reversed = Schema::Intersection(
            vec![
                exactly(1),
                Schema::list(SeqShape::homogeneous(either.clone())),
            ]
            .into(),
        );
        assert_eq!(
            read(&reversed),
            Some(vec!["[1]".to_owned(), "['a']".to_owned()])
        );
        let loose = Schema::refine(
            any_list.clone(),
            vec![Constraint::MinLen(1), Constraint::MaxLen(2)],
        );
        let ranged = Schema::Intersection(
            vec![Schema::list(SeqShape::homogeneous(literal(0))), loose].into(),
        );
        assert_eq!(read(&ranged), None, "a length range is no one length");
        let tuples = Schema::Intersection(
            vec![Schema::tuple(SeqShape::homogeneous(literal(0))), exactly(1)].into(),
        );
        assert_eq!(read(&tuples), None, "two containers");
        let record = Schema::record(
            vec![
                Field {
                    name: "a".into(),
                    schema: literal(0),
                    required: true,
                },
                Field {
                    name: "b".into(),
                    schema: literal(3),
                    required: false,
                },
            ],
            valgebra_core::Openness::Closed,
        );
        assert_eq!(
            read(&record),
            Some(vec!["{'a': 1}".to_owned(), "{'a': 1, 'b': 'a'}".to_owned()])
        );
    });
}

/// Past [`FINITE_VALUES`] a subject is left to the deciders, however it grows:
/// by a union, a fixed sequence or a record.
#[test]
fn a_subject_past_the_bound_is_not_enumerated() {
    Python::attach(|py| {
        let literals = pool(py, &format!("list(range({}))", FINITE_VALUES + 1));
        let read = |schema: &Schema| {
            values_of(py, schema, &literals)
                .unwrap()
                .map(|values| values.len())
        };
        let all = Schema::Union((0..=FINITE_VALUES).map(literal).collect::<Vec<_>>().into());
        let most = Schema::Union((0..FINITE_VALUES).map(literal).collect::<Vec<_>>().into());
        assert_eq!(read(&most), Some(FINITE_VALUES));
        assert_eq!(read(&all), None);
        let eight = Schema::Union((0..8).map(literal).collect::<Vec<_>>().into());
        let two = Schema::list(SeqShape::fixed([eight.clone(), eight.clone()]));
        assert_eq!(read(&two), Some(64));
        let three = Schema::list(SeqShape::fixed([
            eight.clone(),
            eight.clone(),
            eight.clone(),
        ]));
        assert_eq!(read(&three), None);
        let field = |name: &str| Field {
            name: name.into(),
            schema: eight.clone(),
            required: false,
        };
        let record = Schema::record(
            vec![field("a"), field("b")],
            valgebra_core::Openness::Closed,
        );
        assert_eq!(read(&record), None, "nine by nine is past the bound");
        let at_the_bound = Schema::record(
            vec![Field {
                name: "a".into(),
                schema: most,
                required: true,
            }],
            valgebra_core::Openness::Closed,
        );
        assert_eq!(read(&at_the_bound), Some(FINITE_VALUES));
    });
}

/// A supertype's walk reads every atom as a proof would only with no
/// predicate, class or attribute in it and no constant but a whole set, through
/// every definition a reference reaches.
#[test]
fn a_supertype_with_code_of_its_own_is_not_walked() {
    Python::attach(|py| {
        let literals = pool(py, "[1, __import__('decimal').Decimal(1)]");
        let alike = |schema: &Schema, defs: &[Schema]| walked_alike(py, schema, &literals, defs);
        let plain = Schema::Union(vec![Schema::Int, Schema::Str, literal(0)].into());
        assert!(alike(&plain, &[]));
        let predicate = Schema::refine(Schema::Int, vec![Constraint::Predicate(PredIx::new(0))]);
        assert!(!alike(
            &Schema::Union(vec![Schema::Str, predicate.clone()].into()),
            &[]
        ));
        let bounded = Schema::refine(Schema::Int, vec![Constraint::MinLen(0)]);
        assert!(alike(&bounded, &[]));
        assert!(!alike(
            &Schema::Instance(valgebra_core::ClassIx::new(0)),
            &[]
        ));
        assert!(!alike(&literal(1), &[]), "a Decimal constant");
        let reference = Schema::Ref(DefIx::new(0));
        let recursive = Schema::Union(
            vec![
                Schema::Int,
                Schema::list(SeqShape::homogeneous(reference.clone())),
            ]
            .into(),
        );
        assert!(alike(&reference, &[recursive]));
        let hidden =
            Schema::Union(vec![Schema::Int, Schema::list(SeqShape::homogeneous(predicate))].into());
        assert!(!alike(&reference, &[hidden]));
        assert!(
            !alike(&Schema::Ref(DefIx::new(1)), &[Schema::Int]),
            "a dangling reference"
        );
        let map = Schema::keyed_map(
            vec![],
            vec![valgebra_core::MapClause {
                key: Schema::Str,
                value: Schema::Complement(Arc::new(literal(1))),
            }],
        );
        assert!(!alike(&map, &[]), "inside a clause's value");
    });
}

/// A subject with a part no bound closes holds infinitely many values, and is
/// left to the deciders: a record open to more keys, a sequence of any length
/// past its prefix, and one met with an exact length whose items are not all
/// one schema.
#[test]
fn a_subject_with_an_unbounded_part_is_not_enumerated() {
    Python::attach(|py| {
        let literals = pool(py, "[1, 'a']");
        let read = |schema: &Schema| reprs(py, values_of(py, schema, &literals).unwrap());
        let open = Schema::keyed_map(
            vec![Field {
                name: "a".into(),
                schema: literal(0),
                required: true,
            }],
            vec![valgebra_core::MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        );
        assert_eq!(read(&open), None, "a record open to more keys");
        let longer = Schema::list(SeqShape::prefix_tail([literal(0)], literal(1)));
        assert_eq!(
            read(&longer),
            None,
            "a sequence of any length past its prefix"
        );
        let any_list = Schema::list(SeqShape::homogeneous(Schema::Anything(Spelling::Any)));
        let two = Schema::refine(any_list, vec![Constraint::MinLen(2), Constraint::MaxLen(2)]);
        let mixed = Schema::Intersection(vec![longer, two].into());
        assert_eq!(read(&mixed), None, "a prefix is no one item schema");
    });
}

/// The walk check reads every node a supertype holds: a class inside a set's
/// element or under a refinement's base is code a second call may answer
/// differently.
#[test]
fn a_class_inside_a_set_or_a_refinement_is_found() {
    Python::attach(|py| {
        let literals = pool(py, "[1]");
        let alike = |schema: &Schema| walked_alike(py, schema, &literals, &[]);
        let class = Schema::Instance(valgebra_core::ClassIx::new(0));
        assert!(alike(&Schema::set(Schema::Int)));
        assert!(
            !alike(&Schema::set(class.clone())),
            "inside a set's element"
        );
        let of_classes = Schema::list(SeqShape::homogeneous(class));
        let refined = Schema::Refine {
            base: Arc::new(of_classes),
            constraints: vec![Constraint::MinLen(1)].into(),
        };
        assert!(!alike(&refined), "under a refinement's base");
    });
}
