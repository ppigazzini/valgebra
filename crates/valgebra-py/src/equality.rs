//! Equality of two compiled validators, modulo the constant pools they index.
//!
//! A schema's leaves name their constants by *pool slot*, and a slot is
//! construction order: `Literal[1, 2]` and `Literal[2, 1]` build the same two
//! nodes over pools that hold the same two values the other way round.
//! Comparing slot for slot answers "were these built the same way", and the
//! question `==` is asked is "are these the same schema" -- so the comparison
//! here reads *through* the slots to the values, and matches a union's members
//! and a refinement's constraints as the sets they are rather than as lists.
//!
//! What this does **not** do is decide anything. Two schemas that denote one set
//! by a theorem -- `int | ~int` and `anything`, a record and a mapping that
//! admit the same dicts -- are not equal here, and `is_equivalent` is the
//! question for that. This is the syntactic equality the constructors settle,
//! read modulo an accident of construction order.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use valgebra_core::{Constraint, Field, MapClause, Schema};

/// One side of a comparison: the schema's definitions and the pool it indexes.
pub(crate) struct Side<'a> {
    pub(crate) definitions: &'a [Schema],
    pub(crate) pool: &'a [Py<PyAny>],
}

/// Whether two compiled schemas are the same schema, modulo their pools.
///
/// The definition lists are compared alongside, pairwise: a `Ref` names a
/// definition by index, so two schemas whose bodies differ are different even
/// where the referring node is identical.
pub(crate) fn schemas_equal(
    py: Python<'_>,
    left_schema: &Schema,
    left: &Side<'_>,
    right_schema: &Schema,
    right: &Side<'_>,
) -> bool {
    if left.definitions.len() != right.definitions.len() {
        return false;
    }
    if !equal(py, left_schema, left, right_schema, right) {
        return false;
    }
    left.definitions
        .iter()
        .zip(right.definitions)
        .all(|(a, b)| equal(py, a, left, b, right))
}

/// Whether two pooled objects are one constant.
///
/// Identity first, so a validator equals itself even when it pools a value that
/// is not equal to itself -- a `nan` literal is the case, and comparing it by
/// `==` alone would make `v == v` false. Type then value after that, because the
/// literal rule is typed: `1` and `True` are equal in Python and name different
/// singletons here.
fn objects_equal(py: Python<'_>, left: Option<&Py<PyAny>>, right: Option<&Py<PyAny>>) -> bool {
    match (left, right) {
        (Some(a), Some(b)) => {
            let (a, b) = (a.bind(py), b.bind(py));
            a.is(b) || (a.get_type().is(b.get_type()) && a.eq(b).unwrap_or(false))
        }
        // A slot that is not in the pool is a schema the frontend could not have
        // built. Two of them are not evidence of anything, so they are not equal.
        _ => false,
    }
}

/// Whether every member of one list matches a distinct member of the other.
///
/// The lists are a union's members, a map's clauses or a refinement's
/// constraints: each is a *set* written as a list, and construction sorts them
/// by a key that reads a pool slot, so equal sets can be written in different
/// orders. Quadratic in the member count and paid only by `==`, which is not on
/// any hot path; the lists a schema builds are short, and the long ones -- a
/// wide `Literal` -- are the case this exists for.
fn same_multiset<T>(items: &[T], others: &[T], mut equal_item: impl FnMut(&T, &T) -> bool) -> bool {
    if items.len() != others.len() {
        return false;
    }
    let mut taken = vec![false; others.len()];
    for item in items {
        let found = others.iter().enumerate().position(|(at, other)| {
            !taken.get(at).copied().unwrap_or(true) && equal_item(item, other)
        });
        match found {
            Some(at) => {
                if let Some(slot) = taken.get_mut(at) {
                    *slot = true;
                }
            }
            None => return false,
        }
    }
    true
}

fn fields_equal(
    py: Python<'_>,
    a: &[Field],
    left: &Side<'_>,
    b: &[Field],
    right: &Side<'_>,
) -> bool {
    // Ordered rather than matched: construction sorts a record's fields by name,
    // which is a key no pool slot reaches, so two equal records are already in
    // the same order.
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.name == y.name
                && x.required == y.required
                && equal(py, &x.schema, left, &y.schema, right)
        })
}

fn clause_equal(
    py: Python<'_>,
    a: &MapClause,
    left: &Side<'_>,
    b: &MapClause,
    right: &Side<'_>,
) -> bool {
    equal(py, &a.key, left, &b.key, right) && equal(py, &a.value, left, &b.value, right)
}

fn constraint_equal(
    py: Python<'_>,
    a: &Constraint,
    left: &Side<'_>,
    b: &Constraint,
    right: &Side<'_>,
) -> bool {
    // The discriminants agree before anything is read: `Ge(0)` and `Le(0)` name
    // one constant and bound nothing alike.
    if core::mem::discriminant(a) != core::mem::discriminant(b) {
        return false;
    }
    let pooled = |i: usize, j: usize| objects_equal(py, left.pool.get(i), right.pool.get(j));
    match (a, b) {
        (Constraint::Ge(i), Constraint::Ge(j))
        | (Constraint::Gt(i), Constraint::Gt(j))
        | (Constraint::Le(i), Constraint::Le(j))
        | (Constraint::Lt(i), Constraint::Lt(j))
        | (Constraint::MultipleOf(i), Constraint::MultipleOf(j)) => pooled(i.get(), j.get()),
        (Constraint::Predicate(i), Constraint::Predicate(j)) => pooled(i.get(), j.get()),
        // A length and a pattern carry their operand inline, so the derived
        // comparison is the whole of it.
        (x, y) => x == y,
    }
}

/// The comparison itself, one node at a time.
fn equal(
    py: Python<'_>,
    left_schema: &Schema,
    left: &Side<'_>,
    right_schema: &Schema,
    right: &Side<'_>,
) -> bool {
    let recur = |a: &Schema, b: &Schema| equal(py, a, left, b, right);
    match (left_schema, right_schema) {
        // The pooled leaves: read through the slot to the object it names.
        (Schema::Literal(a), Schema::Literal(b)) => {
            objects_equal(py, left.pool.get(a.get()), right.pool.get(b.get()))
        }
        (Schema::Instance(a), Schema::Instance(b)) => {
            objects_equal(py, left.pool.get(a.get()), right.pool.get(b.get()))
        }
        // The set-shaped lists: matched rather than zipped.
        (Schema::Union(a), Schema::Union(b))
        | (Schema::Intersection(a), Schema::Intersection(b)) => {
            same_multiset(a, b, |x, y| recur(x, y))
        }
        (Schema::Complement(a), Schema::Complement(b)) => recur(a, b),
        (
            Schema::Coll {
                container: a_kind,
                element: a,
            },
            Schema::Coll {
                container: b_kind,
                element: b,
            },
        ) => a_kind == b_kind && recur(a, b),
        (
            Schema::Seq {
                container: a_kind,
                shape: a,
            },
            Schema::Seq {
                container: b_kind,
                shape: b,
            },
        ) => {
            // A sequence's elements are positional, so this one is a zip.
            a_kind == b_kind
                && a.prefix.len() == b.prefix.len()
                && a.prefix.iter().zip(&b.prefix).all(|(x, y)| recur(x, y))
                && match (&a.tail, &b.tail) {
                    (Some(x), Some(y)) => recur(x, y),
                    (None, None) => true,
                    _ => false,
                }
        }
        (
            Schema::KeyedMap {
                fields: a_fields,
                defaults: a_defaults,
            },
            Schema::KeyedMap {
                fields: b_fields,
                defaults: b_defaults,
            },
        ) => {
            fields_equal(py, a_fields, left, b_fields, right)
                && same_multiset(a_defaults, b_defaults, |x, y| {
                    clause_equal(py, x, left, y, right)
                })
        }
        (Schema::AttrRecord { fields: a }, Schema::AttrRecord { fields: b }) => {
            fields_equal(py, a, left, b, right)
        }
        (
            Schema::Refine {
                base: a_base,
                constraints: a,
            },
            Schema::Refine {
                base: b_base,
                constraints: b,
            },
        ) => {
            recur(a_base, b_base)
                && same_multiset(a, b, |x, y| constraint_equal(py, x, left, y, right))
        }
        // Everything left carries no pool slot and no set-shaped list, so the
        // derived comparison is the whole of it: the scalars, the two bounds,
        // and the two reference forms.
        (a, b) => a == b,
    }
}

/// Digest a schema's shape: everything about it a pool slot cannot reach.
///
/// The companion of [`schemas_equal`], and coarser on purpose. Equality reads
/// pooled *values*, which a hash cannot: reading one needs the interpreter, and
/// a constant that is not hashable would leave a validator that cannot be a
/// dictionary key. So the slots are skipped and their nodes contribute their
/// kind alone -- `Literal[1]` and `Literal[2]` hash together and equality tells
/// them apart, which is the direction a hash is allowed to be wrong in.
pub(crate) fn hash_shape<H: Hasher>(schema: &Schema, hasher: &mut H) {
    core::mem::discriminant(schema).hash(hasher);
    match schema {
        // The lists whose order is not part of the schema fold commutatively,
        // so two spellings of one set hash alike.
        Schema::Union(members) | Schema::Intersection(members) => {
            members.len().hash(hasher);
            unordered(members, hasher, hash_shape);
        }
        Schema::Complement(inner) => hash_shape(inner, hasher),
        Schema::Coll { container, element } => {
            container.hash(hasher);
            hash_shape(element, hasher);
        }
        Schema::Seq { container, shape } => {
            container.hash(hasher);
            shape.prefix.len().hash(hasher);
            for element in &shape.prefix {
                hash_shape(element, hasher);
            }
            shape.tail.is_some().hash(hasher);
            if let Some(tail) = &shape.tail {
                hash_shape(tail, hasher);
            }
        }
        Schema::KeyedMap { fields, defaults } => {
            hash_fields(fields, hasher);
            defaults.len().hash(hasher);
            unordered(defaults, hasher, |clause, one| {
                hash_shape(&clause.key, one);
                hash_shape(&clause.value, one);
            });
        }
        Schema::AttrRecord { fields } => hash_fields(fields, hasher),
        Schema::Refine { base, constraints } => {
            hash_shape(base, hasher);
            constraints.len().hash(hasher);
            unordered(constraints, hasher, |constraint, one| {
                core::mem::discriminant(constraint).hash(one);
                // The bounds a slot does not reach are part of the shape; the
                // ones it does are left to equality.
                match constraint {
                    Constraint::MinLen(n) | Constraint::MaxLen(n) => n.hash(one),
                    Constraint::Regex(pattern) => pattern.hash(one),
                    _ => {}
                }
            });
        }
        Schema::Ref(index) => index.hash(hasher),
        // The scalars, the two bounds, the pooled leaves and the build-time
        // marker: the discriminant above is the whole of their shape.
        _ => {}
    }
}

fn hash_fields<H: Hasher>(fields: &[Field], hasher: &mut H) {
    // Ordered, because construction orders them by name.
    fields.len().hash(hasher);
    for field in fields {
        field.name.hash(hasher);
        field.required.hash(hasher);
        hash_shape(&field.schema, hasher);
    }
}

/// Fold each item's own digest into one that does not depend on their order.
///
/// Addition rather than exclusive-or: a pair of equal items cancels under xor,
/// and while the lists here are deduplicated sets, a hash that turns two members
/// into none is a trap laid for the next person to relax that.
fn unordered<T, H: Hasher>(
    items: &[T],
    hasher: &mut H,
    mut each: impl FnMut(&T, &mut DefaultHasher),
) {
    let mut total: u64 = 0;
    for item in items {
        let mut one = DefaultHasher::new();
        each(item, &mut one);
        total = total.wrapping_add(one.finish());
    }
    total.hash(hasher);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_multiset_matches_a_permutation_and_refuses_a_repeat() {
        assert!(same_multiset(&[1, 2, 3], &[3, 1, 2], |a, b| a == b));
        assert!(!same_multiset(&[1, 1, 2], &[1, 2, 2], |a, b| a == b));
        assert!(!same_multiset(&[1, 2], &[1, 2, 3], |a, b| a == b));
        assert!(same_multiset::<u8>(&[], &[], |a, b| a == b));
    }

    /// The shape hash ignores what the pool holds and follows what equality
    /// reads: two spellings of one union hash alike, and two different literals
    /// are allowed to collide.
    #[test]
    fn the_shape_hash_is_blind_to_the_slot_and_to_the_order() {
        let digest = |schema: &Schema| {
            let mut hasher = DefaultHasher::new();
            hash_shape(schema, &mut hasher);
            hasher.finish()
        };
        let literal = |at: usize| Schema::Literal(valgebra_core::ConstIx::new(at));
        // Two slots, one shape: equality tells these apart by value, and a hash
        // that read the slot would deny two equal validators one bucket.
        assert_eq!(digest(&literal(0)), digest(&literal(7)));
        // Order is not part of a union, so the fold over its members is not
        // allowed to see one.
        let left = Schema::Union(vec![Schema::Int, Schema::Str]);
        let right = Schema::Union(vec![Schema::Str, Schema::Int]);
        assert_eq!(digest(&left), digest(&right));
        // And a hash that ignored everything would pass the two lines above
        // having read nothing.
        assert_ne!(digest(&Schema::Int), digest(&Schema::Str));
        assert_ne!(digest(&left), digest(&Schema::Union(vec![Schema::Int])));
        // A reference names a definition, and two references to different ones
        // are different shapes.
        assert_ne!(
            digest(&Schema::Ref(valgebra_core::DefIx::new(0))),
            digest(&Schema::Ref(valgebra_core::DefIx::new(1)))
        );
        // An attribute record's fields are its shape. No annotation builds one
        // without a class, so it is built here.
        let attributes = |name: &str, schema: Schema| {
            Schema::attr_record(vec![Field {
                name: name.to_owned(),
                schema,
                required: true,
            }])
        };
        assert_ne!(
            digest(&attributes("x", Schema::Int)),
            digest(&attributes("y", Schema::Int))
        );
        assert_ne!(
            digest(&attributes("x", Schema::Int)),
            digest(&attributes("x", Schema::Str))
        );
        assert_eq!(
            digest(&attributes("x", Schema::Int)),
            digest(&attributes("x", Schema::Int))
        );
    }
}

// Needs a live interpreter: the comparison reads pooled objects, which is the
// half a pure-Rust test cannot reach.
#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter {
    use super::*;

    /// A node built around another, so one pair can be asked of every container.
    type Wrap = fn(Schema) -> Schema;

    use crate::build::Pool;
    use pyo3::types::PyDict;
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
                            name: "x".to_owned(),
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
                    name: "x".to_owned(),
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

    /// The shapes a hash must keep apart, one per node that holds another.
    ///
    /// The companion of the case above: a digest that stopped reading a node's
    /// contents would put every record in one bucket, which is correct and
    /// useless. Each pair below differs only inside the node named.
    #[test]
    fn the_shape_hash_reads_what_each_node_holds() {
        Python::attach(|py| {
            let digest = |expression: &str| {
                let (schema, _) = compile(py, expression);
                let mut hasher = DefaultHasher::new();
                hash_shape(&schema, &mut hasher);
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
}
