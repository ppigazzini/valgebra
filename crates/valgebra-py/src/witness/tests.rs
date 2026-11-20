use std::sync::Arc;

use pyo3::types::PyList;
use valgebra_core::{ConstIx, DefIx, Field, MapClause, SeqShape};

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

fn read(py: Python<'_>, schema: &Schema, literals: &[Py<PyAny>], defs: &[Schema]) -> Vec<String> {
    guesses(py, schema, literals, defs)
        .unwrap()
        .iter()
        .map(|value| value.repr().unwrap().to_string())
        .collect()
}

fn list_of(item: Schema) -> Schema {
    Schema::list(SeqShape::homogeneous(item))
}

/// A scalar kind guesses a value of it, a constant guesses itself, and the top
/// and a complement guess one value of each kind, which the subject's walk
/// sorts.
#[test]
fn a_scalar_guesses_a_value_of_its_kind() {
    Python::attach(|py| {
        let literals = pool(py, "[7]");
        let guessed = |schema: &Schema| read(py, schema, &literals, &[]);
        assert_eq!(guessed(&Schema::Int), ["0", "1"]);
        assert_eq!(guessed(&Schema::Str), ["''", "'a'"]);
        assert_eq!(guessed(&Schema::NoneType), ["None"]);
        assert_eq!(guessed(&Schema::Literal(ConstIx::new(0))), ["7"]);
        assert_eq!(guessed(&Schema::Nothing), Vec::<String>::new());
        let kinds = ["None", "False", "0", "0.5", "''", "[]", "{}", "()"];
        assert_eq!(guessed(&Schema::Complement(Arc::new(Schema::Int))), kinds);
        assert_eq!(guessed(&Schema::ANYTHING), kinds);
    });
}

/// A union hands up the first guess of each member before the second of any.
#[test]
fn a_union_interleaves_its_members_guesses() {
    Python::attach(|py| {
        let either = Schema::Union(vec![Schema::Int, Schema::Str, Schema::NoneType].into());
        assert_eq!(read(py, &either, &[], &[]), ["0", "''", "None", "1", "'a'"]);
    });
}

/// A list guesses two items together, each item alone and no item; a record
/// its required fields, then every field, and a key its clause names.
#[test]
fn a_container_guesses_from_its_items_and_fields() {
    Python::attach(|py| {
        assert_eq!(
            read(py, &list_of(Schema::Int), &[], &[]),
            ["[0, 1]", "[0]", "[1]", "[]"]
        );
        let record = Schema::KeyedMap {
            fields: vec![
                Field {
                    name: "a".into(),
                    schema: Schema::Int,
                    required: true,
                },
                Field {
                    name: "b".into(),
                    schema: Schema::Str,
                    required: false,
                },
            ]
            .into(),
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::NoneType,
            }]
            .into(),
        };
        assert_eq!(
            read(py, &record, &[], &[]),
            [
                "{'a': 0}",
                "{'a': 0, '': None}",
                "{'a': 0, 'b': ''}",
                "{'a': 1}",
                "{'a': 1, 'b': 'a'}"
            ]
        );
    });
}

/// A reference is read through its definition and spends no level, so a list
/// of a list is guessed within the depth.
#[test]
fn a_reference_is_read_through_its_definition() {
    Python::attach(|py| {
        let reference = Schema::Ref(DefIx::new(0));
        let defs = vec![Schema::Union(
            vec![Schema::NoneType, list_of(reference.clone())].into(),
        )];
        let guessed = read(py, &reference, &[], &defs);
        assert!(guessed.contains(&"[[None]]".to_owned()), "{guessed:?}");
    });
}

/// No guess nests more than [`WITNESS_DEPTH`] containers: a list of lists
/// deeper than that guesses the empty list at the level it stops.
#[test]
fn a_guess_nests_no_deeper_than_the_bound() {
    Python::attach(|py| {
        let mut deep = Schema::Int;
        for _ in 0..WITNESS_DEPTH + 2 {
            deep = list_of(deep);
        }
        let nesting = |text: &str| {
            let (mut level, mut deepest) = (0_usize, 0_usize);
            for character in text.chars() {
                match character {
                    '[' => {
                        level += 1;
                        deepest = deepest.max(level);
                    }
                    ']' => level -= 1,
                    _ => {}
                }
            }
            deepest
        };
        let deepest = read(py, &deep, &[], &[])
            .iter()
            .map(|text| nesting(text))
            .max()
            .unwrap();
        assert_eq!(deepest, WITNESS_DEPTH);
    });
}

/// A relation walks at most [`WITNESS_VALUES`] guesses, however many a union
/// of constants names.
#[test]
fn a_relation_walks_no_more_guesses_than_the_bound() {
    Python::attach(|py| {
        let literals = pool(py, "list(range(200))");
        let members: Vec<Schema> = (0..200)
            .map(|at| Schema::Literal(ConstIx::new(at)))
            .collect();
        let wide = Schema::Union(members.into());
        assert_eq!(read(py, &wide, &literals, &[]).len(), WITNESS_VALUES);
    });
}

/// A tuple or a record of more than [`WITNESS_PARTS`] positions or fields is
/// guessed at not at all, so no guess builds that many objects.
#[test]
fn a_value_past_the_bound_on_its_parts_is_not_guessed() {
    Python::attach(|py| {
        let wide = Schema::tuple(SeqShape::fixed(vec![Schema::Int; WITNESS_PARTS + 1]));
        assert_eq!(read(py, &wide, &[], &[]), Vec::<String>::new());
        let fits = Schema::tuple(SeqShape::fixed(vec![Schema::Int; WITNESS_PARTS]));
        assert_ne!(read(py, &fits, &[], &[]), Vec::<String>::new());
        let field = |at: usize| Field {
            name: format!("f{at}").into(),
            schema: Schema::Int,
            required: true,
        };
        let record = |width: usize| Schema::KeyedMap {
            fields: (0..width).map(field).collect::<Vec<_>>().into(),
            defaults: Vec::new().into(),
        };
        assert_eq!(
            read(py, &record(WITNESS_PARTS + 1), &[], &[]),
            Vec::<String>::new()
        );
        assert_ne!(
            read(py, &record(WITNESS_PARTS), &[], &[]),
            Vec::<String>::new()
        );
    });
}

/// Fixed positions are guessed together, the `j`-th guess of each.
#[test]
fn a_tuple_guesses_each_position_by_the_same_index() {
    Python::attach(|py| {
        let pair = Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Str]));
        assert_eq!(read(py, &pair, &[], &[]), ["(0, '')", "(1, 'a')"]);
    });
}

/// A node other than a union or a meet hands up at most [`WITNESS_FANOUT`]
/// guesses, however many its items guess: a list of a union of twenty
/// constants guesses the pair, the first few alone, and no item.
#[test]
fn a_node_hands_up_no_more_guesses_than_the_bound() {
    Python::attach(|py| {
        let literals = pool(py, "list(range(20))");
        let members: Vec<Schema> = (0..20)
            .map(|at| Schema::Literal(ConstIx::new(at)))
            .collect();
        let listed = list_of(Schema::Union(members.into()));
        let guessed = read(py, &listed, &literals, &[]);
        assert_eq!(guessed.len(), WITNESS_FANOUT);
        assert_eq!(guessed.first().map(String::as_str), Some("[0, 1]"));
    });
}
