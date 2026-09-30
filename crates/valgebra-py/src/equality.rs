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
//!
//! A constant is compared and hashed by its own `__eq__` and `__hash__`. One
//! raising an ordinary exception reads as unequal and adds nothing to the
//! digest; a fatal signal propagates, as it does from every other question
//! valgebra asks of user code.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use valgebra_core::{Constraint, Field, MapClause, Schema};

use crate::errors::unless_fatal;

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
) -> PyResult<bool> {
    Ok(left.definitions.len() == right.definitions.len()
        && equal(py, left_schema, left, right_schema, right)?
        && every(left.definitions.iter().zip(right.definitions), |(a, b)| {
            equal(py, a, left, b, right)
        })?)
}

/// Whether `test` holds of every item, stopping at the first it fails or raises.
fn every<T>(
    items: impl IntoIterator<Item = T>,
    mut test: impl FnMut(T) -> PyResult<bool>,
) -> PyResult<bool> {
    for item in items {
        if !test(item)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether two pooled objects are one constant.
///
/// Identity first, so a validator equals itself even when it pools a value that
/// is not equal to itself -- a `nan` literal is the case, and comparing it by
/// `==` alone would make `v == v` false. Type then value after that, because the
/// literal rule is typed: `1` and `True` are equal in Python and name different
/// singletons here.
fn objects_equal(
    py: Python<'_>,
    left: Option<&Py<PyAny>>,
    right: Option<&Py<PyAny>>,
) -> PyResult<bool> {
    match (left, right) {
        (Some(a), Some(b)) => {
            let (a, b) = (a.bind(py), b.bind(py));
            Ok(a.is(b) || (a.get_type().is(b.get_type()) && unless_fatal(a.eq(b), py, false)?))
        }
        // A slot that is not in the pool is a schema the frontend could not have
        // built. Two of them are not evidence of anything, so they are not equal.
        _ => Ok(false),
    }
}

/// Whether every member of one list matches a distinct member of the other.
///
/// The lists are a union's members, a map's clauses or a refinement's
/// constraints: each is a *set* written as a list, and construction sorts them
/// by a key that reads a pool slot, so equal sets can be written in different
/// orders. Quadratic in the member count where the two are written in
/// different orders, and one pass where they are written alike, which two
/// validators built the same way are; paid only by `==`, which is not on any
/// hot path, and the long lists -- a wide `Literal` -- are the case this exists
/// for.
fn same_multiset<T>(
    items: &[T],
    others: &[T],
    mut equal_item: impl FnMut(&T, &T) -> PyResult<bool>,
) -> PyResult<bool> {
    if items.len() != others.len() {
        return Ok(false);
    }
    let mut taken = vec![false; others.len()];
    // Where the untaken members begin. Every member before it is taken, and a
    // taken member is never compared, so starting there asks the same
    // questions in the same order as starting at the front -- and two lists
    // built alike, whose members match in place, are one pass rather than a
    // walk over the taken prefix per member.
    let mut untaken = 0;
    for item in items {
        let mut found = None;
        for (at, other) in others.iter().enumerate().skip(untaken) {
            #[cfg(test)]
            tests::scanned();
            if !taken.get(at).copied().unwrap_or(true) && equal_item(item, other)? {
                found = Some(at);
                break;
            }
        }
        let Some(at) = found else {
            return Ok(false);
        };
        if let Some(slot) = taken.get_mut(at) {
            *slot = true;
        }
        untaken += taken.iter().skip(untaken).take_while(|&&done| done).count();
    }
    Ok(true)
}

fn fields_equal(
    py: Python<'_>,
    a: &[Field],
    left: &Side<'_>,
    b: &[Field],
    right: &Side<'_>,
) -> PyResult<bool> {
    // Ordered rather than matched: construction sorts a record's fields by name,
    // which is a key no pool slot reaches, so two equal records are already in
    // the same order.
    Ok(a.len() == b.len()
        && every(a.iter().zip(b), |(x, y)| {
            Ok(x.name == y.name
                && x.required == y.required
                && equal(py, &x.schema, left, &y.schema, right)?)
        })?)
}

fn clause_equal(
    py: Python<'_>,
    a: &MapClause,
    left: &Side<'_>,
    b: &MapClause,
    right: &Side<'_>,
) -> PyResult<bool> {
    Ok(equal(py, &a.key, left, &b.key, right)? && equal(py, &a.value, left, &b.value, right)?)
}

fn constraint_equal(
    py: Python<'_>,
    a: &Constraint,
    left: &Side<'_>,
    b: &Constraint,
    right: &Side<'_>,
) -> PyResult<bool> {
    // The discriminants agree before anything is read: `Ge(0)` and `Le(0)` name
    // one constant and bound nothing alike.
    if core::mem::discriminant(a) != core::mem::discriminant(b) {
        return Ok(false);
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
        (x, y) => Ok(x == y),
    }
}

/// The comparison itself, one node at a time.
fn equal(
    py: Python<'_>,
    left_schema: &Schema,
    left: &Side<'_>,
    right_schema: &Schema,
    right: &Side<'_>,
) -> PyResult<bool> {
    let recur = |a: &Schema, b: &Schema| equal(py, a, left, b, right);
    Ok(match (left_schema, right_schema) {
        // The pooled leaves: read through the slot to the object it names.
        (Schema::Literal(a), Schema::Literal(b)) => {
            objects_equal(py, left.pool.get(a.get()), right.pool.get(b.get()))?
        }
        (Schema::Instance(a), Schema::Instance(b)) => {
            objects_equal(py, left.pool.get(a.get()), right.pool.get(b.get()))?
        }
        // The set-shaped lists: matched rather than zipped.
        (Schema::Union(a), Schema::Union(b))
        | (Schema::Intersection(a), Schema::Intersection(b)) => same_multiset(a, b, recur)?,
        (Schema::Complement(a), Schema::Complement(b)) => recur(a, b)?,
        (
            Schema::Coll {
                container: a_kind,
                element: a,
            },
            Schema::Coll {
                container: b_kind,
                element: b,
            },
        ) => a_kind == b_kind && recur(a, b)?,
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
                && every(a.prefix.iter().zip(b.prefix.iter()), |(x, y)| recur(x, y))?
                && match (&a.tail, &b.tail) {
                    (Some(x), Some(y)) => recur(x, y)?,
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
            fields_equal(py, a_fields, left, b_fields, right)?
                && same_multiset(a_defaults, b_defaults, |x, y| {
                    clause_equal(py, x, left, y, right)
                })?
        }
        (Schema::AttrRecord { fields: a }, Schema::AttrRecord { fields: b }) => {
            fields_equal(py, a, left, b, right)?
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
            recur(a_base, b_base)?
                && same_multiset(a, b, |x, y| constraint_equal(py, x, left, y, right))?
        }
        // Everything left carries no pool slot and no set-shaped list, so the
        // derived comparison is the whole of it: the scalars, the two bounds,
        // and the two reference forms.
        (a, b) => a == b,
    })
}

/// Fold the constant a pool slot names into the digest, where it has a hash.
///
/// Equality reads the *value* behind a slot, so the hash must too or two
/// schemas that differ only in a constant land on one bucket. They did:
/// `Literal[1]` through `Literal[1000]` were one hash, and a dictionary keyed by
/// validators -- the reason `__hash__` exists -- degenerated into a list, at
/// 98 microseconds per lookup over ten thousand entries.
///
/// [`objects_equal`] is `a is b`, or `type(a) is type(b)` and `a == b`, so
/// folding the constant's own hash beside its type's keeps the two consistent
/// for every type whose `__eq__` and `__hash__` agree -- which is the contract
/// Python's own dictionaries already require.
///
/// A constant with no hash contributes nothing and the shape stands alone,
/// which is what the one case needing it asks for: a validator is usable as a
/// key whatever it pools, and refusing to hash would be worse than a
/// collision.
fn hash_constant<H: Hasher>(
    py: Python<'_>,
    slot: usize,
    pool: &[Py<PyAny>],
    hasher: &mut H,
) -> PyResult<()> {
    let Some(object) = pool.get(slot) else {
        return Ok(());
    };
    let bound = object.bind(py);
    if let Some(hash) = unless_fatal(bound.hash().map(Some), py, None)? {
        hash.hash(hasher);
        if let Some(kind) = unless_fatal(bound.get_type().hash().map(Some), py, None)? {
            kind.hash(hasher);
        }
    }
    Ok(())
}

/// Digest a schema, reading the constants its pool slots name.
///
/// The companion of [`schemas_equal`]: every node equality separates is a node
/// this may separate, and the two are consistent where the constants behave.
/// What it does *not* read is the slot index -- that is construction order, and
/// two spellings of one schema pool their constants in different slots.
pub(crate) fn hash_shape<H: Hasher>(
    py: Python<'_>,
    schema: &Schema,
    pool: &[Py<PyAny>],
    hasher: &mut H,
) -> PyResult<()> {
    core::mem::discriminant(schema).hash(hasher);
    match schema {
        // The lists whose order is not part of the schema fold commutatively,
        // so two spellings of one set hash alike.
        Schema::Union(members) | Schema::Intersection(members) => {
            members.len().hash(hasher);
            unordered(members, hasher, |member, one| {
                hash_shape(py, member, pool, one)
            })?;
        }
        Schema::Complement(inner) => hash_shape(py, inner, pool, hasher)?,
        Schema::Coll { container, element } => {
            container.hash(hasher);
            hash_shape(py, element, pool, hasher)?;
        }
        Schema::Seq { container, shape } => {
            container.hash(hasher);
            shape.prefix.len().hash(hasher);
            for element in shape.prefix.iter() {
                hash_shape(py, element, pool, hasher)?;
            }
            shape.tail.is_some().hash(hasher);
            if let Some(tail) = &shape.tail {
                hash_shape(py, tail, pool, hasher)?;
            }
        }
        Schema::KeyedMap { fields, defaults } => {
            hash_fields(py, fields, pool, hasher)?;
            defaults.len().hash(hasher);
            unordered(defaults, hasher, |clause, one| {
                hash_shape(py, &clause.key, pool, one)?;
                hash_shape(py, &clause.value, pool, one)
            })?;
        }
        Schema::AttrRecord { fields } => hash_fields(py, fields, pool, hasher)?,
        Schema::Refine { base, constraints } => {
            hash_shape(py, base, pool, hasher)?;
            constraints.len().hash(hasher);
            unordered(constraints, hasher, |constraint, one| {
                core::mem::discriminant(constraint).hash(one);
                // The bounds a slot does not reach are part of the shape; the
                // ones it does are left to equality.
                match constraint {
                    Constraint::MinLen(n) | Constraint::MaxLen(n) => n.hash(one),
                    Constraint::Regex(pattern) => pattern.hash(one),
                    // A bound names its operand through the pool, and equality
                    // reads it, so `Ge(0)` and `Ge(1)` must not hash alike.
                    Constraint::Ge(i)
                    | Constraint::Gt(i)
                    | Constraint::Le(i)
                    | Constraint::Lt(i)
                    | Constraint::MultipleOf(i) => hash_constant(py, i.get(), pool, one)?,
                    Constraint::Predicate(i) => hash_constant(py, i.get(), pool, one)?,
                }
                Ok(())
            })?;
        }
        Schema::Ref(index) => index.hash(hasher),
        // The pooled leaves: the constant behind the slot, not the slot.
        Schema::Literal(index) => hash_constant(py, index.get(), pool, hasher)?,
        Schema::Instance(index) => hash_constant(py, index.get(), pool, hasher)?,
        // The scalars, the two bounds and the build-time marker: the
        // discriminant above is the whole of their shape.
        _ => {}
    }
    Ok(())
}

fn hash_fields<H: Hasher>(
    py: Python<'_>,
    fields: &[Field],
    pool: &[Py<PyAny>],
    hasher: &mut H,
) -> PyResult<()> {
    // Ordered, because construction orders them by name.
    fields.len().hash(hasher);
    for field in fields {
        field.name.hash(hasher);
        field.required.hash(hasher);
        hash_shape(py, &field.schema, pool, hasher)?;
    }
    Ok(())
}

/// Fold each item's own digest into one that does not depend on their order.
///
/// Addition rather than exclusive-or: a pair of equal items cancels under xor,
/// and while the lists here are deduplicated sets, a hash that turns two members
/// into none is a trap laid for the next person to relax that.
fn unordered<T, H: Hasher>(
    items: &[T],
    hasher: &mut H,
    mut each: impl FnMut(&T, &mut DefaultHasher) -> PyResult<()>,
) -> PyResult<()> {
    let mut total: u64 = 0;
    for item in items {
        let mut one = DefaultHasher::new();
        each(item, &mut one)?;
        total = total.wrapping_add(one.finish());
    }
    total.hash(hasher);
    Ok(())
}

#[cfg(test)]
mod tests;

// Needs a live interpreter: the comparison reads pooled objects, which is the
// half a pure-Rust test cannot reach.
#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter;
