use super::*;

/// A node built around another, so one pair can be asked of every container.
type Wrap = fn(Schema) -> Schema;

use crate::build::Pool;
use pyo3::types::{PyDict, PyInt};
use std::ffi::CString;
use valgebra_core::ConstIx;

/// Build a schema from an annotation, with the pool it indexes.
fn compile(py: Python<'_>, expression: &str) -> (Schema, Vec<Py<PyAny>>) {
    let namespace = PyDict::new(py);
    for module in ["typing", "types"] {
        namespace
            .set_item(module, py.import(module).expect("the module imports"))
            .expect("a namespace holds it");
    }
    let annotation = py
        .eval(
            &CString::new(expression).expect("no interior nul"),
            Some(&namespace),
            None,
        )
        .unwrap_or_else(|error| panic!("{expression} does not evaluate: {error}"));
    let mut pool = Pool::default();
    let mut definitions = Vec::new();
    let schema = crate::build::build_schema(&annotation, &mut pool, &mut definitions)
        .unwrap_or_else(|error| panic!("{expression} does not build: {error}"));
    (schema, pool.into_items())
}

fn same(py: Python<'_>, left: &str, right: &str) -> bool {
    let (a, a_pool) = compile(py, left);
    let (b, b_pool) = compile(py, right);
    schemas_equal(
        py,
        &a,
        &Side {
            definitions: &[],
            pool: &a_pool,
        },
        &b,
        &Side {
            definitions: &[],
            pool: &b_pool,
        },
    )
}

/// Two spellings of one schema are one schema, whatever order they pooled
/// their constants in.
#[test]
fn the_order_a_constant_was_pooled_in_is_not_part_of_the_schema() {
    Python::attach(|py| {
        for (left, right) in [
            // A union of literals: the members pool in the order written,
            // and the set they name does not.
            ("typing.Literal[1, 2]", "typing.Literal[2, 1]"),
            (
                "typing.Literal['a', 'b', 'c']",
                "typing.Literal['c', 'a', 'b']",
            ),
            // A record: the fields are a map.
            ("{'a': int, 'b': str}", "{'b': str, 'a': int}"),
            // A map's clauses are unordered here.
            ("{str: int, int: str}", "{int: str, str: int}"),
            // A refinement's markers are a set of constraints.
            (
                "typing.Annotated[int, types.SimpleNamespace(ge=0), \
                 types.SimpleNamespace(le=9)]",
                "typing.Annotated[int, types.SimpleNamespace(le=9), \
                 types.SimpleNamespace(ge=0)]",
            ),
            // And the plain cases, so the comparison is not passing
            // everything.
            ("int", "int"),
            ("list[dict[str, int]]", "list[dict[str, int]]"),
        ] {
            assert!(same(py, left, right), "{left} is not {right}");
        }
    });
}

/// And two schemas that differ are still different, including the ways this
/// comparison could paper over.
#[test]
fn reading_through_the_pool_does_not_make_everything_equal() {
    Python::attach(|py| {
        for (left, right) in [
            // Different constants at the same slot.
            ("typing.Literal[1]", "typing.Literal[2]"),
            ("typing.Literal[1, 2]", "typing.Literal[1, 3]"),
            // A literal is typed: `1 == True` in Python and these are two
            // singletons.
            ("typing.Literal[1]", "typing.Literal[True]"),
            ("typing.Literal[1]", "typing.Literal[1.0]"),
            // The same names, different types.
            ("{'a': int}", "{'a': str}"),
            // A missing field, an extra one, and a required one made
            // optional.
            ("{'a': int, 'b': int}", "{'a': int}"),
            ("{'a': int}", "{'a?': int}"),
            // The bound is the same number and the direction is not.
            (
                "typing.Annotated[int, types.SimpleNamespace(ge=0)]",
                "typing.Annotated[int, types.SimpleNamespace(le=0)]",
            ),
            // A sequence's positions are ordered, unlike a union's members.
            ("tuple[int, str]", "tuple[str, int]"),
            ("list[int]", "set[int]"),
            // One node holds both collection kinds, and the kind is part of the
            // value: a set is never a frozenset.
            ("set[int]", "frozenset[int]"),
            ("set[int]", "frozenset[str]"),
            // The constraints whose operand is written into the node rather
            // than pooled: their payloads are the whole difference.
            (
                "typing.Annotated[str, types.SimpleNamespace(min_length=1)]",
                "typing.Annotated[str, types.SimpleNamespace(min_length=2)]",
            ),
            (
                "typing.Annotated[str, types.SimpleNamespace(pattern='a+')]",
                "typing.Annotated[str, types.SimpleNamespace(pattern='b+')]",
            ),
        ] {
            assert!(!same(py, left, right), "{left} is read as {right}");
        }
    });
}

/// Every node that holds another must read the pool through it.
///
/// The comparison's fallback is the derived one, which compares slots. That
/// is right for a node with nothing inside it and wrong for every node that
/// carries a schema: a record whose field is a literal is structurally equal
/// to one whose field is a *different* literal at the same slot. So each
/// container is asked about a constant that pooled at a different slot on
/// each side, which no comparison of slots can answer.
#[test]
fn a_constant_inside_any_container_is_read_through_its_slot() {
    Python::attach(|py| {
        // One constant, two slots: the left pools it first, the right pools
        // a string ahead of it. Structurally these are `Literal(0)` and
        // `Literal(1)` -- different terms naming one singleton -- so a node
        // that stops recursing cannot answer by comparing slots.
        let (left, left_pool) = compile(py, "typing.Literal[1]");
        let (right_union, right_pool) = compile(py, "typing.Literal['z', 1]");
        let right = match &right_union {
            Schema::Union(members) => members
                .iter()
                .find(|member| **member == Schema::Literal(ConstIx::new(1)))
                .expect("the int pooled second")
                .clone(),
            other => panic!("a two-member Literal built {other:?}"),
        };
        assert_ne!(left, right, "the two sides must differ structurally");
        let wrappers: [(&str, Wrap); 8] = [
            ("complement", |s| Schema::Complement(Box::new(s))),
            ("set", |s| Schema::set(s)),
            ("frozenset", |s| Schema::frozen_set(s)),
            ("homogeneous list", |s| {
                Schema::list(valgebra_core::SeqShape::homogeneous(s))
            }),
            // A fixed-length sequence has no tail, which is its own arm.
            ("fixed tuple", |s| {
                Schema::tuple(valgebra_core::SeqShape::fixed([s, Schema::Str]))
            }),
            ("union", |s| Schema::union([s, Schema::Str])),
            ("record field", |s| {
                Schema::keyed_map(
                    vec![Field {
                        name: "x".into(),
                        schema: s,
                        required: true,
                    }],
                    Vec::new(),
                )
            }),
            // A clause is a key and a value, and both are read.
            ("map clause", |s| {
                Schema::mapping(MapClause {
                    key: Schema::Str,
                    value: s,
                })
            }),
        ];
        for (name, wrap) in wrappers {
            assert!(
                schemas_equal(
                    py,
                    &wrap(left.clone()),
                    &Side {
                        definitions: &[],
                        pool: &left_pool
                    },
                    &wrap(right.clone()),
                    &Side {
                        definitions: &[],
                        pool: &right_pool
                    },
                ),
                "a constant under a {name} is not read through its slot"
            );
        }
        // The rest of the containers are asked in their own case below,
        // where the pair being read through is built the same way.
    });
}

/// A map clause is two halves, and each is read on its own.
#[test]
fn a_clause_compares_its_key_and_its_value() {
    Python::attach(|py| {
        let (_, pool) = compile(py, "int");
        // A clause's halves are not interchangeable: swapping them is a
        // different map, and reading them as one condition would say
        // otherwise.
        let clause = |key: Schema, value: Schema| Schema::mapping(MapClause { key, value });
        let side = Side {
            definitions: &[],
            pool: &pool,
        };
        assert!(!schemas_equal(
            py,
            &clause(Schema::Str, Schema::Int),
            &side,
            &clause(Schema::Int, Schema::Str),
            &side,
        ));
        assert!(schemas_equal(
            py,
            &clause(Schema::Str, Schema::Int),
            &side,
            &clause(Schema::Str, Schema::Int),
            &side,
        ));
        // And each half is read on its own: a pair agreeing on the key and
        // differing on the value is a different map, which a clause read as
        // "either half matches" would call the same one.
        assert!(!schemas_equal(
            py,
            &clause(Schema::Str, Schema::Int),
            &side,
            &clause(Schema::Str, Schema::Str),
            &side,
        ));
        assert!(!schemas_equal(
            py,
            &clause(Schema::Str, Schema::Int),
            &side,
            &clause(Schema::Bytes, Schema::Int),
            &side,
        ));
    });
}

/// An attribute record and a definition behind a reference: the two nodes a
/// wrapper cannot stand in for.
#[test]
fn a_record_and_a_definition_are_read_through_the_pool_too() {
    Python::attach(|py| {
        let (left, left_pool) = compile(py, "typing.Literal[1]");
        let (right_union, right_pool) = compile(py, "typing.Literal['z', 1]");
        let right = match &right_union {
            Schema::Union(members) => members
                .iter()
                .find(|member| **member == Schema::Literal(ConstIx::new(1)))
                .expect("the int pooled second")
                .clone(),
            other => panic!("a two-member Literal built {other:?}"),
        };
        let field = |schema: Schema| {
            Schema::attr_record(vec![Field {
                name: "x".into(),
                schema,
                required: true,
            }])
        };
        assert!(schemas_equal(
            py,
            &field(left.clone()),
            &Side {
                definitions: &[],
                pool: &left_pool
            },
            &field(right.clone()),
            &Side {
                definitions: &[],
                pool: &right_pool
            },
        ));
        let reference = Schema::Ref(valgebra_core::DefIx::new(0));
        assert!(schemas_equal(
            py,
            &reference,
            &Side {
                definitions: core::slice::from_ref(&left),
                pool: &left_pool
            },
            &reference,
            &Side {
                definitions: core::slice::from_ref(&right),
                pool: &right_pool
            },
        ));
        // And a definition that differs is still a difference, so the walk
        // over the definitions is reading them rather than counting them.
        let (other, other_pool) = compile(py, "typing.Literal[3, 4]");
        assert!(!schemas_equal(
            py,
            &reference,
            &Side {
                definitions: core::slice::from_ref(&left),
                pool: &left_pool
            },
            &reference,
            &Side {
                definitions: core::slice::from_ref(&other),
                pool: &other_pool
            },
        ));
    });
}

/// The hash reads the *constant*, never the slot it sits in, and never the
/// order of a list whose order is not part of the schema.
#[test]
fn the_hash_is_blind_to_the_slot_and_to_the_order() {
    Python::attach(|py| {
        let one = PyInt::new(py, 1).into_any().unbind();
        let digest = |schema: &Schema, pool: &[Py<PyAny>]| {
            let mut hasher = DefaultHasher::new();
            hash_shape(py, schema, pool, &mut hasher);
            hasher.finish()
        };
        let literal = |at: usize| Schema::Literal(valgebra_core::ConstIx::new(at));
        // One constant, two slots: equality reads through a slot to the value, so
        // a hash that read the slot would deny two equal validators one bucket.
        let padded: Vec<Py<PyAny>> = (0..8).map(|_| one.clone_ref(py)).collect();
        assert_eq!(
            digest(&literal(0), std::slice::from_ref(&one)),
            digest(&literal(7), &padded)
        );
        // Order is not part of a union, so the fold over its members is not
        // allowed to see one.
        let left = Schema::Union(vec![Schema::Int, Schema::Str]);
        let right = Schema::Union(vec![Schema::Str, Schema::Int]);
        assert_eq!(digest(&left, &[]), digest(&right, &[]));
        // And a hash that ignored everything would pass the two lines above
        // having read nothing.
        assert_ne!(digest(&Schema::Int, &[]), digest(&Schema::Str, &[]));
        assert_ne!(
            digest(&left, &[]),
            digest(&Schema::Union(vec![Schema::Int]), &[])
        );
        // What a complement holds is part of its shape, and so is which collection
        // kind a node names.
        let not_int = Schema::Complement(Box::new(Schema::Int));
        let not_str = Schema::Complement(Box::new(Schema::Str));
        assert_ne!(digest(&not_int, &[]), digest(&not_str, &[]));
        assert_ne!(digest(&not_int, &[]), digest(&Schema::Int, &[]));
        assert_ne!(
            digest(&Schema::set(Schema::Int), &[]),
            digest(&Schema::frozen_set(Schema::Int), &[])
        );
        assert_ne!(
            digest(&Schema::set(Schema::Int), &[]),
            digest(&Schema::set(Schema::Str), &[])
        );
        // A reference names a definition, and two references to different ones
        // are different shapes.
        assert_ne!(
            digest(&Schema::Ref(valgebra_core::DefIx::new(0)), &[]),
            digest(&Schema::Ref(valgebra_core::DefIx::new(1)), &[])
        );
        // An attribute record's fields are its shape. No annotation builds one
        // without a class, so it is built here.
        let attributes = |name: &str, schema: Schema| {
            Schema::attr_record(vec![Field {
                name: name.into(),
                schema,
                required: true,
            }])
        };
        assert_ne!(
            digest(&attributes("x", Schema::Int), &[]),
            digest(&attributes("y", Schema::Int), &[])
        );
        assert_ne!(
            digest(&attributes("x", Schema::Int), &[]),
            digest(&attributes("x", Schema::Str), &[])
        );
        assert_eq!(
            digest(&attributes("x", Schema::Int), &[]),
            digest(&attributes("x", Schema::Int), &[])
        );
    });
}

/// The shapes a hash must keep apart, one per node that holds another.
///
/// The companion of the case above: a digest that stopped reading a node's
/// contents would put every record in one bucket, which is correct and
/// useless. Each pair below differs only inside the node named.
#[test]
fn the_shape_hash_reads_what_each_node_holds() {
    Python::attach(|py| {
        let digest = |expression: &str| {
            let (schema, pool) = compile(py, expression);
            let mut hasher = DefaultHasher::new();
            hash_shape(py, &schema, &pool, &mut hasher);
            hasher.finish()
        };
        for (left, right) in [
            ("list[int]", "list[str]"),
            ("set[int]", "set[str]"),
            ("tuple[int, str]", "tuple[int, int]"),
            ("tuple[int, str]", "tuple[int, ...]"),
            ("{'a': int}", "{'a': str}"),
            ("{'a': int}", "{'b': int}"),
            ("{'a': int}", "{'a?': int}"),
            ("{str: int}", "{str: str}"),
            ("{'a': int}", "{'a': int, str: int}"),
            (
                "typing.Annotated[str, types.SimpleNamespace(min_length=1)]",
                "typing.Annotated[str, types.SimpleNamespace(min_length=2)]",
            ),
            (
                "typing.Annotated[str, types.SimpleNamespace(min_length=1)]",
                "typing.Annotated[str, types.SimpleNamespace(max_length=1)]",
            ),
            (
                "typing.Annotated[str, types.SimpleNamespace(pattern='a+')]",
                "typing.Annotated[str, types.SimpleNamespace(pattern='b+')]",
            ),
            ("int | str", "int | bytes"),
            // The two pooled leaves. Every pair above differs in a node's
            // *shape*, which the discriminant alone tells apart, so none of them
            // asks whether the digest reads the constant behind a slot -- and
            // these two nodes are nothing but that constant. A literal and a
            // class are the whole of what `hash_constant` exists for: drop
            // either arm, or stop hashing the object it names, and each pair
            // here collides while every pair above still passes.
            ("typing.Literal[1]", "typing.Literal[2]"),
            ("typing.Literal['a']", "typing.Literal['b']"),
            ("types.SimpleNamespace", "types.ModuleType"),
        ] {
            assert_ne!(digest(left), digest(right), "{left} hashes as {right}");
        }
    });
}

/// A validator equals itself even when it pools a value that does not equal
/// itself, which is the case identity is checked first for.
#[test]
fn a_schema_over_a_nan_equals_itself() {
    Python::attach(|py| {
        let (schema, pool) = compile(py, "typing.Literal[float('nan')]");
        let side = Side {
            definitions: &[],
            pool: &pool,
        };
        assert!(schemas_equal(py, &schema, &side, &schema, &side));
        // Two separately pooled nans are two objects that are equal to
        // nothing, themselves included, so they are not one constant.
        assert!(!same(
            py,
            "typing.Literal[float('nan')]",
            "typing.Literal[float('nan')]"
        ));
    });
}
