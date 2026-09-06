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
mod tests;

// Needs a live interpreter: the comparison reads pooled objects, which is the
// half a pure-Rust test cannot reach.
#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter;
