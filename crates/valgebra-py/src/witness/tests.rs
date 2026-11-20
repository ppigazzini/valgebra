use std::sync::Arc;

use pyo3::types::PyList;
use valgebra_core::{ConstIx, Constraint, DefIx, Field, MapClause, OperandIx, SeqShape};

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

fn missed(py: Python<'_>, schema: &Schema, literals: &[Py<PyAny>], defs: &[Schema]) -> Vec<String> {
    misses(py, schema, literals, defs)
        .unwrap()
        .iter()
        .map(|value| value.repr().unwrap().to_string())
        .collect()
}

/// The values a scalar misses are the other kinds, the top misses nothing,
/// and a complement misses what its operand holds; a complement in a subject
/// is guessed as what its operand misses, a constant first missing the value
/// beside it.
#[test]
fn a_negation_is_pushed_through_the_kinds_and_the_complement() {
    Python::attach(|py| {
        let literals = pool(py, "['a', 7]");
        let kinds = ["None", "False", "0", "0.5", "''", "[]", "{}", "()"];
        assert_eq!(missed(py, &Schema::Int, &literals, &[]), kinds);
        assert_eq!(
            missed(py, &Schema::ANYTHING, &literals, &[]),
            Vec::<String>::new()
        );
        let not_int = Schema::Complement(Arc::new(Schema::Int));
        assert_eq!(missed(py, &not_int, &literals, &[]), ["0", "1"]);
        let not_a = Schema::Complement(Arc::new(Schema::Literal(ConstIx::new(0))));
        assert_eq!(
            read(py, &not_a, &literals, &[]).first().map(String::as_str),
            Some("'aa'")
        );
        let seven = Schema::Literal(ConstIx::new(1));
        assert_eq!(
            missed(py, &seven, &literals, &[])
                .first()
                .map(String::as_str),
            Some("8")
        );
    });
}

/// Outside a list of `X` is a list holding one value outside `X`, then the
/// other kinds; outside a tuple, one position outside its schema, or one too
/// few or too many. A value of each kind is a guess, and the one of `X`'s own
/// kind is left to the walks to discard.
#[test]
fn a_sequence_is_missed_on_one_part() {
    Python::attach(|py| {
        let list_missed = missed(py, &list_of(Schema::Int), &[], &[]);
        assert_eq!(list_missed.first().map(String::as_str), Some("[None]"));
        assert!(list_missed.contains(&"None".to_owned()), "{list_missed:?}");
        let pair = Schema::tuple(SeqShape::fixed(vec![Schema::Int, Schema::Str]));
        let pair_missed = missed(py, &pair, &[], &[]);
        for expected in ["(None, '')", "(0, None)", "(0,)", "(0, '', None)"] {
            assert!(
                pair_missed.contains(&expected.to_owned()),
                "{pair_missed:?}"
            );
        }
    });
}

/// Outside a record is a dict lacking a required key, holding a field outside
/// its type, or holding a key no field names: under `None` where the record
/// closes such keys out, under a value outside the clause where one governs it.
#[test]
fn a_record_is_missed_on_a_key_or_a_field() {
    Python::attach(|py| {
        let fields: Arc<[Field]> = vec![
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
        .into();
        let closed = Schema::KeyedMap {
            fields: fields.clone(),
            defaults: Vec::new().into(),
        };
        let closed_missed = missed(py, &closed, &[], &[]);
        for expected in [
            "{}",
            "{'a': None}",
            "{'a': 0, 'b': None}",
            "{'a': 0, '_0': None}",
        ] {
            assert!(
                closed_missed.contains(&expected.to_owned()),
                "{closed_missed:?}"
            );
        }
        let governed = Schema::KeyedMap {
            fields,
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }]
            .into(),
        };
        let governed_missed = missed(py, &governed, &[], &[]);
        assert!(
            governed_missed.contains(&"{'a': 0, '_0': None}".to_owned()),
            "{governed_missed:?}"
        );
    });
}

/// Outside a bound is a number beside it: below and at a lower bound, at and
/// above an upper one, as an integer where the operand is one.
#[test]
fn a_bound_is_missed_beside_it() {
    Python::attach(|py| {
        let literals = pool(py, "[5, 0.5]");
        let bounded = |constraint| Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![constraint].into(),
        };
        let at_least = missed(
            py,
            &bounded(Constraint::Ge(OperandIx::new(0))),
            &literals,
            &[],
        );
        assert_eq!(
            at_least.first().map(String::as_str),
            Some("4"),
            "{at_least:?}"
        );
        assert!(at_least.contains(&"5".to_owned()), "{at_least:?}");
        let at_most = missed(
            py,
            &bounded(Constraint::Lt(OperandIx::new(0))),
            &literals,
            &[],
        );
        assert_eq!(
            at_most.first().map(String::as_str),
            Some("6"),
            "{at_most:?}"
        );
        let above_half = missed(
            py,
            &bounded(Constraint::Gt(OperandIx::new(1))),
            &literals,
            &[],
        );
        assert_eq!(
            above_half.first().map(String::as_str),
            Some("0.0"),
            "{above_half:?}"
        );
    });
}

/// A relation walks at most [`WITNESS_VALUES`] values, the subject's guesses
/// and the supertype's misses taken a turn each, however many either names.
#[test]
fn guesses_against_a_supertype_take_turns_within_the_bound() {
    Python::attach(|py| {
        let literals = pool(py, "list(range(200))");
        let members: Vec<Schema> = (0..200)
            .map(|at| Schema::Literal(ConstIx::new(at)))
            .collect();
        let wide = Schema::Union(members.into());
        let all = guesses_against(py, &wide, &wide, &literals, &[]).unwrap();
        assert_eq!(all.len(), WITNESS_VALUES);
        let shown: Vec<String> = all.iter().map(|v| v.repr().unwrap().to_string()).collect();
        assert_eq!(&shown[..4], ["0", "1", "1", "2"], "{shown:?}");
        assert_eq!(missed(py, &wide, &literals, &[]).len(), WITNESS_VALUES);
    });
}

/// A miss nests no deeper than [`WITNESS_DEPTH`] containers holding a value:
/// below the last, a value of each kind may be an empty one, which builds
/// nothing further.
#[test]
fn a_miss_nests_no_deeper_than_the_bound() {
    Python::attach(|py| {
        let mut nested = Schema::Int;
        for _ in 0..WITNESS_DEPTH + 2 {
            nested = list_of(nested);
        }
        let deepest = misses(py, &nested, &[], &[])
            .unwrap()
            .iter()
            .map(|value| {
                let mut levels = 0;
                let mut at = value.clone();
                while let Ok(list) = at.clone().cast_into::<PyList>() {
                    match list.get_item(0) {
                        Ok(item) => {
                            levels += 1;
                            at = item;
                        }
                        Err(_) => break,
                    }
                }
                levels
            })
            .max()
            .unwrap();
        assert_eq!(deepest, WITNESS_DEPTH);
    });
}

/// Outside a set of `X` is a set holding one value outside `X` that hashes; a
/// reference misses what its definition does.
#[test]
fn a_set_and_a_reference_are_missed_through_their_parts() {
    Python::attach(|py| {
        let ints = Schema::Coll {
            container: valgebra_core::CollKind::Set,
            element: Arc::new(Schema::Int),
        };
        let set_missed = missed(py, &ints, &[], &[]);
        assert_eq!(set_missed.first().map(String::as_str), Some("{None}"));
        assert!(
            !set_missed.iter().any(|value| value == "{[]}"),
            "{set_missed:?}"
        );
        let defs = vec![list_of(Schema::Int)];
        let through = missed(py, &Schema::Ref(DefIx::new(0)), &[], &defs);
        assert_eq!(through.first().map(String::as_str), Some("[None]"));
    });
}

/// The edges of the misses beside a value: an upper float bound misses the
/// value above it, an integer bound at the end of `i64` misses itself alone, a
/// bytes constant the bytes one longer, and an integer constant at that end
/// nothing beside it.
#[test]
fn a_value_beside_a_bound_or_a_constant_stays_in_range() {
    Python::attach(|py| {
        let literals = pool(py, "[0.5, 2**63 - 1, b'x']");
        let bounded = |constraint| Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![constraint].into(),
        };
        let below_half = missed(
            py,
            &bounded(Constraint::Le(OperandIx::new(0))),
            &literals,
            &[],
        );
        assert_eq!(
            below_half.first().map(String::as_str),
            Some("1.0"),
            "{below_half:?}"
        );
        let at_the_end = missed(
            py,
            &bounded(Constraint::Le(OperandIx::new(1))),
            &literals,
            &[],
        );
        assert_eq!(
            at_the_end.first().map(String::as_str),
            Some("9223372036854775807"),
            "{at_the_end:?}"
        );
        let bytes = missed(py, &Schema::Literal(ConstIx::new(2)), &literals, &[]);
        assert_eq!(bytes.first().map(String::as_str), Some("b'xa'"));
        let last = missed(py, &Schema::Literal(ConstIx::new(1)), &literals, &[]);
        assert_eq!(last.first().map(String::as_str), Some("None"), "{last:?}");
    });
}

/// A tuple or a record of more than [`WITNESS_PARTS`] positions or fields is
/// missed only by the other kinds, so no miss builds that many objects, and
/// one of that many is missed on its parts; nor is a record whose required
/// field has no guess.
#[test]
fn a_value_past_the_bound_on_its_parts_is_not_missed() {
    Python::attach(|py| {
        let kinds = ["None", "False", "0", "0.5", "''", "[]", "{}", "()"];
        let tuple_of = |parts: usize| Schema::tuple(SeqShape::fixed(vec![Schema::Int; parts]));
        assert_eq!(missed(py, &tuple_of(WITNESS_PARTS + 1), &[], &[]), kinds);
        let at_the_bound = missed(py, &tuple_of(WITNESS_PARTS), &[], &[]);
        assert!(
            at_the_bound
                .first()
                .is_some_and(|miss| miss.starts_with("(None, 0"))
        );
        let field = |at: usize| Field {
            name: format!("f{at}").into(),
            schema: Schema::Int,
            required: true,
        };
        let record_of = |parts: usize| Schema::KeyedMap {
            fields: (0..parts).map(field).collect::<Vec<_>>().into(),
            defaults: Vec::new().into(),
        };
        assert_eq!(missed(py, &record_of(WITNESS_PARTS + 1), &[], &[]), kinds);
        let fields_at_the_bound = missed(py, &record_of(WITNESS_PARTS), &[], &[]);
        assert!(
            fields_at_the_bound
                .first()
                .is_some_and(|miss| miss.starts_with("{'f1': 0"))
        );
        let unguessed = Schema::KeyedMap {
            fields: vec![Field {
                name: "a".into(),
                schema: Schema::Nothing,
                required: true,
            }]
            .into(),
            defaults: Vec::new().into(),
        };
        assert_eq!(missed(py, &unguessed, &[], &[]), kinds);
    });
}

/// Outside a complement inside a union are the complement's operand's
/// members: outside `int | ~str` are the strings, the negation's negation.
#[test]
fn a_complement_inside_a_union_misses_its_operand() {
    Python::attach(|py| {
        let either =
            Schema::Union(vec![Schema::Int, Schema::Complement(Arc::new(Schema::Str))].into());
        let outside = missed(py, &either, &[], &[]);
        assert_eq!(&outside[..2], ["''", "'a'"], "{outside:?}");
    });
}

/// Where no item may follow the positions -- no tail, or a tail of `nothing` --
/// a sequence is missed one item too long, and an empty one on `[None]`.
#[test]
fn a_sequence_closed_after_its_positions_is_missed_one_too_long() {
    Python::attach(|py| {
        let closed = Schema::list(SeqShape {
            prefix: vec![Schema::Int].into(),
            tail: Some(Arc::new(Schema::Nothing)),
        });
        let closed_missed = missed(py, &closed, &[], &[]);
        assert!(
            closed_missed.contains(&"[0, None]".to_owned()),
            "{closed_missed:?}"
        );
        let empty = Schema::list(SeqShape::fixed(Vec::new()));
        assert_eq!(
            missed(py, &empty, &[], &[]).first().map(String::as_str),
            Some("[None]")
        );
    });
}
