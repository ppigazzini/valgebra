//! A relation from finitely many values, read value by value.
//!
//! `t ⊆ s` holds exactly where no value of `t` lies outside `s` [FCB08, §4.1],
//! so a subject of finitely many values is related by the membership of each:
//! a value outside refutes the pair, and every value inside proves it. The
//! relation asks this only where both deciders declined, of a subject built
//! from constants -- a literal, `None`, `bool`, and fixed sequences, closed
//! records and unions of them -- the shape a translation from JSON Schema
//! writes for `const` and `enum`.
//!
//! A value stands for its literal's whole set only where the constant is that
//! set alone (`docs/14-soundness.md`, "A literal is its constant's class under
//! `==`"): an exact `int`, `bool`, `str`, `bytes`, `None`, or a nonzero finite
//! `float`; a float zero is read as its two values. And the walk reads the supertype as the relation would only where
//! no atom of it runs code a second call may answer differently: no
//! predicate, no class, no attribute, and no constant past those types. A pair
//! holding either is left to the deciders, which decline it.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use valgebra_core::{Constraint, Field, Schema, SeqKind};

/// The values a subject may hold for this reading to enumerate them.
///
/// **This bound is cost.** Each value is built as a Python object and walked
/// through the supertype, where the deciders spend no step: a `const` of a
/// translation from JSON Schema is one value, an `enum` a handful, and a fixed
/// sequence of choices multiplies them.
pub(crate) const FINITE_VALUES: usize = 64;

/// The values `schema` denotes, each built as the Python object it is, where it
/// denotes at most [`FINITE_VALUES`] and every constant in it is its whole set;
/// `None` otherwise.
pub(crate) fn values_of<'py>(
    py: Python<'py>,
    schema: &Schema,
    literals: &[Py<PyAny>],
) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    Ok(match schema {
        Schema::Nothing => Some(Vec::new()),
        Schema::NoneType => Some(vec![py.None().into_bound(py)]),
        Schema::Bool => Some(vec![
            PyBool::new(py, false).to_owned().into_any(),
            PyBool::new(py, true).to_owned().into_any(),
        ]),
        Schema::Literal(index) => {
            let constant = literals.get(index.get()).map(|constant| constant.bind(py));
            constant.and_then(|constant| constant_values(py, constant))
        }
        Schema::Union(members) => {
            let mut all = Vec::new();
            for member in members.iter() {
                let Some(values) = values_of(py, member, literals)? else {
                    return Ok(None);
                };
                all.extend(values);
                if all.len() > FINITE_VALUES {
                    return Ok(None);
                }
            }
            Some(all)
        }
        Schema::Seq { container, shape } if shape.tail.is_none() => {
            let mut positions = Vec::with_capacity(shape.prefix.len());
            for item in shape.prefix.iter() {
                let Some(values) = values_of(py, item, literals)? else {
                    return Ok(None);
                };
                positions.push(values);
            }
            sequences(py, *container, &positions)?
        }
        // A sequence of one item schema met with a length bound fixing its
        // length: the spelling a translation from JSON Schema writes for an
        // array constant, `list[X] & Annotated[list[Any], Len(n, n)]`.
        Schema::Intersection(members) => match exact_length(members) {
            Some((container, item, length)) => match values_of(py, item, literals)? {
                Some(values) => sequences(py, container, &vec![values; length])?,
                None => None,
            },
            None => None,
        },
        Schema::KeyedMap { fields, defaults } if defaults.is_empty() => {
            records(py, fields, literals)?
        }
        _ => None,
    })
}

/// The values a literal's constant stands for: the constant where it is its
/// whole set, and both zeros for a float zero, which `==` joins.
fn constant_values<'py>(
    py: Python<'py>,
    constant: &Bound<'py, PyAny>,
) -> Option<Vec<Bound<'py, PyAny>>> {
    if a_whole_set(constant) {
        return Some(vec![constant.clone()]);
    }
    constant
        .cast_exact::<PyFloat>()
        .is_ok_and(|float| float.value() == 0.0)
        .then(|| {
            vec![
                PyFloat::new(py, 0.0).into_any(),
                PyFloat::new(py, -0.0).into_any(),
            ]
        })
}

/// Every sequence of `container` taking one of each position's values, or
/// `None` past the bound.
fn sequences<'py>(
    py: Python<'py>,
    container: SeqKind,
    positions: &[Vec<Bound<'py, PyAny>>],
) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    let mut rows: Vec<Vec<Bound<'py, PyAny>>> = vec![Vec::new()];
    for values in positions {
        let Some(grown) = extended(&rows, values) else {
            return Ok(None);
        };
        rows = grown;
    }
    let mut built = Vec::with_capacity(rows.len());
    for row in rows {
        built.push(match container {
            SeqKind::List => PyList::new(py, row)?.into_any(),
            SeqKind::Tuple => PyTuple::new(py, row)?.into_any(),
        });
    }
    Ok(Some(built))
}

/// Every dict a closed record of these fields holds, a field absent or taking
/// one of its values where it is optional, or `None` past the bound.
fn records<'py>(
    py: Python<'py>,
    fields: &[Field],
    literals: &[Py<PyAny>],
) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    let mut rows: Vec<Vec<(usize, Bound<'py, PyAny>)>> = vec![Vec::new()];
    for (at, field) in fields.iter().enumerate() {
        let Some(values) = values_of(py, &field.schema, literals)? else {
            return Ok(None);
        };
        let mut next = Vec::new();
        for row in &rows {
            if !field.required {
                next.push(row.clone());
            }
            for value in &values {
                let mut grown = row.clone();
                grown.push((at, value.clone()));
                next.push(grown);
            }
            if next.len() > FINITE_VALUES {
                return Ok(None);
            }
        }
        rows = next;
    }
    let mut built = Vec::with_capacity(rows.len());
    for row in rows {
        let dict = PyDict::new(py);
        for (at, value) in row {
            if let Some(field) = fields.get(at) {
                dict.set_item(PyString::new(py, &field.name), value)?;
            }
        }
        built.push(dict.into_any());
    }
    Ok(Some(built))
}

/// The container, item schema and length of a meet of two members: a sequence
/// whose every item is one schema, and a refinement of a sequence of anything
/// by a least and a greatest length that are one number, of the same container.
fn exact_length(members: &[Schema]) -> Option<(SeqKind, &Schema, usize)> {
    fn homogeneous(schema: &Schema) -> Option<(SeqKind, &Schema)> {
        match schema {
            Schema::Seq { container, shape } if shape.prefix.is_empty() => {
                Some((*container, shape.tail.as_deref()?))
            }
            _ => None,
        }
    }
    fn bounded(schema: &Schema) -> Option<(SeqKind, usize)> {
        let Schema::Refine { base, constraints } = schema else {
            return None;
        };
        let (container, item) = homogeneous(base)?;
        if !matches!(item, Schema::Anything(_)) {
            return None;
        }
        let least = constraints.iter().find_map(|constraint| match constraint {
            Constraint::MinLen(n) => Some(*n),
            _ => None,
        })?;
        let most = constraints.iter().find_map(|constraint| match constraint {
            Constraint::MaxLen(n) => Some(*n),
            _ => None,
        })?;
        (constraints.len() == 2 && least == most).then_some((container, least))
    }
    let [first, second] = members else {
        return None;
    };
    let ((container, item), (bound_container, length)) = homogeneous(first)
        .zip(bounded(second))
        .or_else(|| homogeneous(second).zip(bounded(first)))?;
    (container == bound_container).then_some((container, item, length))
}

/// Every row of `rows` with one of `values` appended, or `None` past the bound.
fn extended<'py>(
    rows: &[Vec<Bound<'py, PyAny>>],
    values: &[Bound<'py, PyAny>],
) -> Option<Vec<Vec<Bound<'py, PyAny>>>> {
    if rows.len().checked_mul(values.len())? > FINITE_VALUES {
        return None;
    }
    Some(
        rows.iter()
            .flat_map(|row| {
                values.iter().map(move |value| {
                    let mut grown = row.clone();
                    grown.push(value.clone());
                    grown
                })
            })
            .collect(),
    )
}

/// Whether `constant` is the whole set its literal denotes: an exact `int`,
/// `bool`, `str`, `bytes`, `None`, or a nonzero finite `float`. A float zero is
/// both zeros, which [`values_of`] builds and a supertype's constant is not.
fn a_whole_set(constant: &Bound<'_, PyAny>) -> bool {
    if constant.is_none()
        || constant.is_exact_instance_of::<PyInt>()
        || constant.is_exact_instance_of::<PyBool>()
        || constant.is_exact_instance_of::<PyString>()
        || constant.is_exact_instance_of::<PyBytes>()
    {
        return true;
    }
    constant.cast_exact::<PyFloat>().is_ok_and(|float| {
        let value = float.value();
        value.is_finite() && value != 0.0
    })
}

/// Whether a walk of `schema` reads every atom as a proof through it would:
/// no predicate, no class, no attribute, and no constant but one that is its
/// whole set, through every definition a reference reaches.
pub(crate) fn walked_alike(
    py: Python<'_>,
    schema: &Schema,
    literals: &[Py<PyAny>],
    definitions: &[Schema],
) -> bool {
    let mut pending: Vec<&Schema> = vec![schema];
    let mut seen = vec![false; definitions.len()];
    while let Some(node) = pending.pop() {
        match node {
            Schema::Instance(_) | Schema::AttrRecord { .. } | Schema::SelfRef(_) => return false,
            Schema::Literal(index) => {
                let whole = literals
                    .get(index.get())
                    .is_some_and(|constant| a_whole_set(constant.bind(py)));
                if !whole {
                    return false;
                }
            }
            Schema::Refine { constraints, .. }
                if constraints
                    .iter()
                    .any(|constraint| matches!(constraint, Constraint::Predicate(_))) =>
            {
                return false;
            }
            Schema::Ref(at) => {
                let Some(slot) = seen.get_mut(at.get()) else {
                    return false;
                };
                if !*slot {
                    *slot = true;
                    match definitions.get(at.get()) {
                        Some(body) => pending.push(body),
                        None => return false,
                    }
                }
            }
            _ => {}
        }
        pushed_children(node, &mut pending);
    }
    true
}

/// The schemas `node` holds directly, pushed onto `pending`.
fn pushed_children<'a>(node: &'a Schema, pending: &mut Vec<&'a Schema>) {
    match node {
        Schema::Union(members) | Schema::Intersection(members) => pending.extend(members.iter()),
        Schema::Complement(inner) => pending.push(inner),
        Schema::Refine { base, .. } => pending.push(base),
        Schema::Seq { shape, .. } => {
            pending.extend(shape.prefix.iter());
            pending.extend(shape.tail.as_deref());
        }
        Schema::Coll { element, .. } => pending.push(element),
        Schema::KeyedMap { fields, defaults } => {
            pending.extend(fields.iter().map(|field| &field.schema));
            for clause in defaults.iter() {
                pending.push(&clause.key);
                pending.push(&clause.value);
            }
        }
        _ => {}
    }
}

// Needs a live interpreter; compiled and run only under the `interpreter-tests`
// feature, which links an embedded Python.
#[cfg(all(test, feature = "interpreter-tests"))]
mod tests;
