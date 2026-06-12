//! The validation walk: membership testing of a Python value against the IR.
//!
//! [`check`] is the explain path: it walks the schema and returns the first
//! [`Violation`], threading a path so a nested failure reports its location.
//! [`matches`] is the membership fast path: a bool with no allocation, used by
//! `is_valid` and by the speculative combinators (`union` members, `complement`
//! inner) so a discarded branch never builds a `Violation`. `check` (whether it
//! produces a violation) and `matches` must stay membership-equivalent.

use std::cell::RefCell;
use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PySet, PyString, PyTuple,
};
use valgebra_core::{Constraint, Field, PathSegment, Schema, Violation};

use crate::errors::{class_label, summarize, truncate};

/// The read-only context threaded through a validation walk: the constants
/// pool, the recursion definitions, and the active recursion guard. The guard
/// records `(object id, definition index)` pairs currently on the path so a
/// value that contains itself fails with `recursion_loop` instead of looping.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub(crate) pool: &'a [Py<PyAny>],
    pub(crate) defs: &'a [Schema],
    pub(crate) guard: &'a RefCell<HashSet<(usize, usize)>>,
}

/// Walk the schema against `value`, returning the first [`Violation`] or `None`
/// if the value is a member. `path` accumulates the location of the current
/// value.
pub(crate) fn check(
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    match schema {
        Schema::Anything | Schema::Any => None,
        Schema::Nothing => Some(mismatch(schema, value, path)),
        Schema::NoneType => admit(value.is_none(), schema, value, path),
        Schema::Bool => admit(value.is_instance_of::<PyBool>(), schema, value, path),
        // bool subclasses int, so True/False are ints: Bool is a subset of Int.
        Schema::Int => admit(value.is_instance_of::<PyInt>(), schema, value, path),
        Schema::Float => admit(value.is_instance_of::<PyFloat>(), schema, value, path),
        Schema::Str => admit(value.is_instance_of::<PyString>(), schema, value, path),
        Schema::Bytes => admit(value.is_instance_of::<PyBytes>(), schema, value, path),
        Schema::Literal(index) => check_literal(*index, value, path, ctx),
        Schema::Sequence(element) => check_sequence(element, value, path, ctx),
        Schema::FixedSequence(elements) => check_fixed_sequence(elements, value, path, ctx),
        Schema::Tuple(elements) => check_tuple(elements, value, path, ctx),
        Schema::VariadicTuple(element) => check_variadic_tuple(element, value, path, ctx),
        Schema::Set(element) => check_set(element, value, path, ctx),
        Schema::FrozenSet(element) => check_frozenset(element, value, path, ctx),
        Schema::Mapping { key, value: val } => check_mapping(key, val, value, path, ctx),
        Schema::Record { fields, open } => check_record(fields, *open, value, path, ctx),
        Schema::Union(members) => check_union(members, value, path, ctx),
        Schema::Intersection(members) => check_intersection(members, value, path, ctx),
        Schema::Complement(inner) => check_complement(inner, value, path, ctx),
        Schema::Instance(index) => check_instance(*index, value, path, ctx),
        Schema::Object {
            class_index,
            fields,
        } => check_object(*class_index, fields, value, path, ctx),
        Schema::Refine { base, constraints } => check_refine(base, constraints, value, path, ctx),
        Schema::Ref(id) => check_ref(*id, value, path, ctx),
        // A SelfRef should have been resolved into a Ref at build time; reaching
        // one means an unresolved recursion marker leaked into a validator.
        Schema::SelfRef(_) => Some(Violation {
            code: "unresolved_recursion",
            path: path.clone(),
            expected: "a resolved recursive value".to_owned(),
            value_summary: summarize(value),
        }),
    }
}

/// Decide membership without building a violation or tracking a path.
///
/// The fast path: `is_valid` uses it, and the speculative combinators use it so
/// a discarded branch never pays for a `Violation` or a `repr`. It must stay
/// membership-equivalent to [`check`].
pub(crate) fn matches(schema: &Schema, value: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> bool {
    let py = value.py();
    match schema {
        Schema::Anything | Schema::Any => true,
        // Bottom admits nothing; an unresolved self-reference is never a member.
        Schema::Nothing | Schema::SelfRef(_) => false,
        Schema::NoneType => value.is_none(),
        Schema::Bool => value.is_instance_of::<PyBool>(),
        Schema::Int => value.is_instance_of::<PyInt>(),
        Schema::Float => value.is_instance_of::<PyFloat>(),
        Schema::Str => value.is_instance_of::<PyString>(),
        Schema::Bytes => value.is_instance_of::<PyBytes>(),
        Schema::Literal(i) => literal_matches(value, ctx.pool[*i].bind(py)),
        Schema::Sequence(e) => value
            .cast::<PyList>()
            .is_ok_and(|list| list.iter().all(|item| matches(e, &item, ctx))),
        Schema::FixedSequence(es) => value.cast::<PyList>().is_ok_and(|list| {
            list.len() == es.len()
                && es
                    .iter()
                    .zip(list.iter())
                    .all(|(s, item)| matches(s, &item, ctx))
        }),
        Schema::Tuple(es) => value.cast::<PyTuple>().is_ok_and(|tuple| {
            tuple.len() == es.len()
                && es
                    .iter()
                    .zip(tuple.iter())
                    .all(|(s, item)| matches(s, &item, ctx))
        }),
        Schema::VariadicTuple(e) => value
            .cast::<PyTuple>()
            .is_ok_and(|tuple| tuple.iter().all(|item| matches(e, &item, ctx))),
        Schema::Set(e) => value
            .cast::<PySet>()
            .is_ok_and(|set| set.iter().all(|item| matches(e, &item, ctx))),
        Schema::FrozenSet(e) => value
            .cast::<PyFrozenSet>()
            .is_ok_and(|set| set.iter().all(|item| matches(e, &item, ctx))),
        Schema::Mapping { key, value: val } => value.cast::<PyDict>().is_ok_and(|dict| {
            dict.iter()
                .all(|(k, v)| matches(key, &k, ctx) && matches(val, &v, ctx))
        }),
        Schema::Record { fields, open } => matches_record(fields, *open, value, ctx),
        Schema::Union(members) => members.iter().any(|m| matches(m, value, ctx)),
        Schema::Intersection(members) => members.iter().all(|m| matches(m, value, ctx)),
        Schema::Complement(inner) => !matches(inner, value, ctx),
        Schema::Instance(i) => value.is_instance(ctx.pool[*i].bind(py)).unwrap_or(false),
        Schema::Object {
            class_index,
            fields,
        } => {
            value
                .is_instance(ctx.pool[*class_index].bind(py))
                .unwrap_or(false)
                && fields.iter().all(|f| {
                    value
                        .getattr(f.name.as_str())
                        .is_ok_and(|attr| matches(&f.schema, &attr, ctx))
                })
        }
        Schema::Refine { base, constraints } => {
            matches(base, value, ctx) && constraints.iter().all(|c| constraint_holds(c, value, ctx))
        }
        Schema::Ref(id) => matches_ref(*id, value, ctx),
    }
}

/// Membership for a record on the fast path: declared fields match and required
/// keys are present, and no undeclared key is admitted (the record is closed).
fn matches_record(fields: &[Field], open: bool, value: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> bool {
    let Ok(dict) = value.cast::<PyDict>() else {
        return false;
    };
    for field in fields {
        match dict.get_item(field.name.as_str()) {
            Ok(Some(item)) => {
                if !matches(&field.schema, &item, ctx) {
                    return false;
                }
            }
            Ok(None) if field.required => return false,
            Ok(None) => {}
            Err(_) => return false,
        }
    }
    if open {
        return true;
    }
    let declared: HashSet<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    dict.iter().all(|(key, _)| {
        key.str()
            .is_ok_and(|text| declared.contains(text.to_string().as_str()))
    })
}

/// The most levels of recursive descent allowed before a value is rejected. A
/// finite value never reaches this; the bound exists so a pathologically deep
/// value fails with `recursion_limit` instead of overflowing the native stack.
const MAX_RECURSION_DEPTH: usize = 128;

/// Validate `value` against the definition `defs[id]`, guarding against value
/// cycles and unbounded depth. The guard holds `(object id, definition index)`
/// for every reference currently on the path: revisiting the same pair means
/// the value contains itself, and the number of entries is the current depth.
fn check_ref(
    id: usize,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let key = (value.as_ptr() as usize, id);
    let depth = {
        let mut guard = ctx.guard.borrow_mut();
        if !guard.insert(key) {
            return Some(Violation {
                code: "recursion_loop",
                path: path.clone(),
                expected: "a finite (non-cyclic) value".to_owned(),
                value_summary: summarize(value),
            });
        }
        guard.len()
    };
    if depth > MAX_RECURSION_DEPTH {
        ctx.guard.borrow_mut().remove(&key);
        return Some(Violation {
            code: "recursion_limit",
            path: path.clone(),
            expected: format!("at most {MAX_RECURSION_DEPTH} levels of recursion"),
            value_summary: summarize(value),
        });
    }
    let result = check(&ctx.defs[id], value, path, ctx);
    ctx.guard.borrow_mut().remove(&key);
    result
}

/// Membership for a recursion reference, with the same cycle and depth guards
/// as [`check_ref`] but reporting only yes/no.
fn matches_ref(id: usize, value: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> bool {
    let key = (value.as_ptr() as usize, id);
    let depth = {
        let mut guard = ctx.guard.borrow_mut();
        if !guard.insert(key) {
            return false;
        }
        guard.len()
    };
    if depth > MAX_RECURSION_DEPTH {
        ctx.guard.borrow_mut().remove(&key);
        return false;
    }
    let result = matches(&ctx.defs[id], value, ctx);
    ctx.guard.borrow_mut().remove(&key);
    result
}

fn check_refine(
    base: &Schema,
    constraints: &[Constraint],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    // Constraints narrow the base set, so they are meaningful only on a base
    // member: if the base fails, report that and do not run the constraints.
    if let Some(violation) = check(base, value, path, ctx) {
        return Some(violation);
    }
    for constraint in constraints {
        if let Some(violation) = check_constraint(constraint, value, path, ctx) {
            return Some(violation);
        }
    }
    None
}

fn check_constraint(
    constraint: &Constraint,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let py = value.py();
    let (ok, code, expected) = match constraint {
        Constraint::Ge(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.ge(bound)),
                "greater_than_equal",
                format!(">= {}", summarize(bound)),
            )
        }
        Constraint::Gt(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.gt(bound)),
                "greater_than",
                format!("> {}", summarize(bound)),
            )
        }
        Constraint::Le(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.le(bound)),
                "less_than_equal",
                format!("<= {}", summarize(bound)),
            )
        }
        Constraint::Lt(i) => {
            let bound = ctx.pool[*i].bind(py);
            (
                cmp(value.lt(bound)),
                "less_than",
                format!("< {}", summarize(bound)),
            )
        }
        Constraint::MinLen(n) => (
            value.len().is_ok_and(|len| len >= *n),
            "too_short",
            format!("length >= {n}"),
        ),
        Constraint::MaxLen(n) => (
            value.len().is_ok_and(|len| len <= *n),
            "too_long",
            format!("length <= {n}"),
        ),
        Constraint::Predicate(i) => {
            // Slow path: the user's Python callable runs at the boundary. A
            // raising predicate is surfaced as a distinct `predicate_error`
            // rather than masked as an ordinary failed match.
            let predicate = ctx.pool[*i].bind(py);
            match predicate_passes(predicate, value) {
                Ok(passed) => (passed, "predicate_failed", "a passing predicate".to_owned()),
                Err(err) => (
                    false,
                    "predicate_error",
                    format!("a predicate that does not raise (raised {err})"),
                ),
            }
        }
    };
    if ok {
        None
    } else {
        Some(Violation {
            code,
            path: path.to_vec(),
            expected,
            value_summary: summarize(value),
        })
    }
}

/// Whether `value` satisfies `constraint`, for the membership fast path.
///
/// A raising predicate is treated as not satisfied; the explain path
/// ([`check_constraint`]) is what distinguishes it as `predicate_error`.
fn constraint_holds(constraint: &Constraint, value: &Bound<'_, PyAny>, ctx: Ctx<'_>) -> bool {
    let py = value.py();
    match constraint {
        Constraint::Ge(i) => cmp(value.ge(ctx.pool[*i].bind(py))),
        Constraint::Gt(i) => cmp(value.gt(ctx.pool[*i].bind(py))),
        Constraint::Le(i) => cmp(value.le(ctx.pool[*i].bind(py))),
        Constraint::Lt(i) => cmp(value.lt(ctx.pool[*i].bind(py))),
        Constraint::MinLen(n) => value.len().is_ok_and(|len| len >= *n),
        Constraint::MaxLen(n) => value.len().is_ok_and(|len| len <= *n),
        Constraint::Predicate(i) => predicate_passes(ctx.pool[*i].bind(py), value).unwrap_or(false),
    }
}

/// Run a user predicate and report whether it returned a truthy result.
/// Returns `Err` if the predicate itself raised, so callers can distinguish a
/// false result from a broken predicate.
fn predicate_passes(predicate: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    predicate.call1((value,))?.is_truthy()
}

/// Interpret a rich-comparison result, treating an error as "did not hold".
fn cmp(result: PyResult<bool>) -> bool {
    result.unwrap_or(false)
}

fn check_instance(
    index: usize,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let class = ctx.pool[index].bind(value.py());
    if value.is_instance(class).unwrap_or(false) {
        None
    } else {
        Some(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            path,
        ))
    }
}

fn check_object(
    class_index: usize,
    fields: &[Field],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let class = ctx.pool[class_index].bind(value.py());
    if !value.is_instance(class).unwrap_or(false) {
        // Not an instance: the attribute checks below cannot be trusted.
        return Some(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            path,
        ));
    }
    for field in fields {
        match value.getattr(field.name.as_str()) {
            Ok(attr) => {
                path.push(PathSegment::Key(field.name.clone()));
                let result = check(&field.schema, &attr, path, ctx);
                path.pop();
                if result.is_some() {
                    return result;
                }
            }
            Err(_) => {
                return Some(located(
                    path,
                    field.name.clone(),
                    "missing_attribute",
                    format!("attribute {:?}", field.name),
                    "missing".to_owned(),
                ));
            }
        }
    }
    None
}

fn check_union(
    members: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    // A value is a member iff it matches at least one branch; decide that on the
    // fast path so a non-matching branch never builds a violation.
    if members.iter().any(|member| matches(member, value, ctx)) {
        return None;
    }
    let labels: Vec<&str> = members.iter().map(Schema::expected).collect();
    Some(Violation {
        code: "union_error",
        path: path.to_vec(),
        expected: format!("one of: {}", labels.join(", ")),
        value_summary: summarize(value),
    })
}

fn check_intersection(
    members: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    // Every member must hold; report the first that fails.
    for member in members {
        if let Some(violation) = check(member, value, path, ctx) {
            return Some(violation);
        }
    }
    None
}

fn check_complement(
    inner: &Schema,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    // The inner result is discarded, so decide membership on the fast path; a
    // value that matches the inner schema fails the complement.
    if matches(inner, value, ctx) {
        Some(Violation {
            code: "unexpected_match",
            path: path.to_vec(),
            expected: format!("not {}", inner.expected()),
            value_summary: summarize(value),
        })
    } else {
        None
    }
}

fn admit(
    ok: bool,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
) -> Option<Violation> {
    if ok {
        None
    } else {
        Some(mismatch(schema, value, path))
    }
}

fn mismatch(schema: &Schema, value: &Bound<'_, PyAny>, path: &[PathSegment]) -> Violation {
    Violation {
        code: schema.error_code(),
        path: path.to_vec(),
        expected: schema.expected().to_owned(),
        value_summary: summarize(value),
    }
}

fn type_mismatch(
    code: &'static str,
    expected: &str,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
) -> Violation {
    Violation {
        code,
        path: path.to_vec(),
        expected: expected.to_owned(),
        value_summary: summarize(value),
    }
}

/// Whether `value` is the typed singleton denoted by `literal`: same type and
/// equal. The same-type guard rules out Python's cross-type equality
/// (`1 == True == 1.0`), so `Literal[1]` denotes `{1}`, not `{1, True, 1.0}`.
fn literal_matches(value: &Bound<'_, PyAny>, literal: &Bound<'_, PyAny>) -> bool {
    value.get_type().is(literal.get_type()) && value.eq(literal).unwrap_or(false)
}

fn check_literal(
    index: usize,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let literal = ctx.pool[index].bind(value.py());
    if literal_matches(value, literal) {
        None
    } else {
        Some(Violation {
            code: "literal_value",
            path: path.to_vec(),
            expected: format!("the literal {}", summarize(literal)),
            value_summary: summarize(value),
        })
    }
}

fn check_sequence(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(list) = value.cast::<PyList>() else {
        return Some(type_mismatch("list_type", "list", value, path));
    };
    for (index, item) in list.iter().enumerate() {
        path.push(PathSegment::Index(index));
        let result = check(element, &item, path, ctx);
        path.pop();
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_fixed_sequence(
    elements: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(list) = value.cast::<PyList>() else {
        return Some(type_mismatch("list_type", "list", value, path));
    };
    if list.len() != elements.len() {
        return Some(Violation {
            code: "list_length",
            path: path.clone(),
            expected: format!("list of length {}", elements.len()),
            value_summary: summarize(value),
        });
    }
    for (index, (schema, item)) in elements.iter().zip(list.iter()).enumerate() {
        path.push(PathSegment::Index(index));
        let result = check(schema, &item, path, ctx);
        path.pop();
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_tuple(
    elements: &[Schema],
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(tuple) = value.cast::<PyTuple>() else {
        return Some(type_mismatch("tuple_type", "tuple", value, path));
    };
    if tuple.len() != elements.len() {
        return Some(Violation {
            code: "tuple_length",
            path: path.clone(),
            expected: format!("tuple of length {}", elements.len()),
            value_summary: summarize(value),
        });
    }
    for (index, (schema, item)) in elements.iter().zip(tuple.iter()).enumerate() {
        path.push(PathSegment::Index(index));
        let result = check(schema, &item, path, ctx);
        path.pop();
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_variadic_tuple(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(tuple) = value.cast::<PyTuple>() else {
        return Some(type_mismatch("tuple_type", "tuple", value, path));
    };
    for (index, item) in tuple.iter().enumerate() {
        path.push(PathSegment::Index(index));
        let result = check(element, &item, path, ctx);
        path.pop();
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_set(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(set) = value.cast::<PySet>() else {
        return Some(type_mismatch("set_type", "set", value, path));
    };
    // Set order is not meaningful, so element failures carry no index segment.
    for item in set.iter() {
        let result = check(element, &item, path, ctx);
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_frozenset(
    element: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(set) = value.cast::<PyFrozenSet>() else {
        return Some(type_mismatch("frozenset_type", "frozenset", value, path));
    };
    for item in set.iter() {
        let result = check(element, &item, path, ctx);
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_mapping(
    key_schema: &Schema,
    value_schema: &Schema,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(dict) = value.cast::<PyDict>() else {
        return Some(type_mismatch("dict_type", "dict", value, path));
    };
    for (key, val) in dict.iter() {
        path.push(PathSegment::Key(key_label(&key)));
        let result =
            check(key_schema, &key, path, ctx).or_else(|| check(value_schema, &val, path, ctx));
        path.pop();
        if result.is_some() {
            return result;
        }
    }
    None
}

fn check_record(
    fields: &[Field],
    open: bool,
    value: &Bound<'_, PyAny>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
) -> Option<Violation> {
    let Ok(dict) = value.cast::<PyDict>() else {
        return Some(type_mismatch("dict_type", "dict", value, path));
    };
    // Declared fields, in declared order: present values are checked, absent
    // required keys fail.
    for field in fields {
        match dict.get_item(field.name.as_str()) {
            Ok(Some(item)) => {
                path.push(PathSegment::Key(field.name.clone()));
                let result = check(&field.schema, &item, path, ctx);
                path.pop();
                if result.is_some() {
                    return result;
                }
            }
            Ok(None) if field.required => {
                return Some(located(
                    path,
                    field.name.clone(),
                    "missing_key",
                    format!("required key {:?}", field.name),
                    "missing".to_owned(),
                ));
            }
            Ok(None) => {}
            Err(_) => return Some(type_mismatch("dict_type", "dict", value, path)),
        }
    }
    // An open (lax) record admits undeclared keys; a closed one rejects them.
    if open {
        return None;
    }
    let declared: HashSet<&str> = fields.iter().map(|field| field.name.as_str()).collect();
    for (key, _) in dict.iter() {
        let key_text = key
            .str()
            .map_or_else(|_| String::new(), |text| text.to_string());
        if !declared.contains(key_text.as_str()) {
            return Some(located(
                path,
                key_text.clone(),
                "extra_key",
                "no unexpected key".to_owned(),
                format!("{key_text:?}"),
            ));
        }
    }
    None
}

/// Build a violation whose path is `path` extended by one key segment.
fn located(
    path: &[PathSegment],
    key: String,
    code: &'static str,
    expected: String,
    value_summary: String,
) -> Violation {
    let mut full = path.to_vec();
    full.push(PathSegment::Key(key));
    Violation {
        code,
        path: full,
        expected,
        value_summary,
    }
}

/// A short, printable label for a mapping key, used in error paths.
fn key_label(key: &Bound<'_, PyAny>) -> String {
    match key.str() {
        Ok(text) => truncate(&text.to_string(), 40),
        Err(_) => summarize(key),
    }
}
