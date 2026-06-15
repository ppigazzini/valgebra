//! The validation walk: one membership test of a value against the IR.
//!
//! [`member`] is the single walk. It returns whether the value belongs to the
//! schema's set, and in *explain* mode (`ctx.explain`) it also aggregates a
//! [`Violation`] for each independent failure into `out` (each record field,
//! each sequence element, each mapping entry), unless `ctx.fail_fast` stops it at
//! the first. In *fast* mode it allocates nothing and short-circuits as soon as
//! membership is decided — the path it took before this module fused the two
//! walks into one. There is no second walk to keep in sync.
//!
//! The walk runs over a [`Value`], so the object path and the in-place JSON path
//! share one traversal. The explain side only ever sees a Python value (the JSON
//! entry points materialize before explaining), so building a violation always
//! has a Python object in hand. The per-child path bookkeeping is gated on
//! `ctx.explain`, a flag constant for a whole walk, so the fast path pays nothing
//! for it.

use std::borrow::Cow;
use std::cell::RefCell;

use jiter::JsonValue;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};
use valgebra_core::{Constraint, Field, PathSegment, Schema, SeqKind, SeqRegex, Violation};

use crate::errors::{class_label, summarize, truncate};
use crate::input::Value;

/// The read-only context threaded through a validation walk: the constants pool,
/// the recursion definitions, the precomputed record index, the active recursion
/// guard, and the two mode flags. The guard records `(object id, definition
/// index)` pairs currently on the path so a value that contains itself fails with
/// `recursion_loop` instead of looping.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub(crate) pool: &'a [Py<PyAny>],
    pub(crate) defs: &'a [Schema],
    /// Per-record declared-field lookups, built once per validator and keyed by
    /// the address of each record's `fields` buffer. The keyed-map fast path
    /// reads it instead of rebuilding the name map on every call; a node absent
    /// from it falls back to building the map, so correctness never depends on
    /// it being complete.
    pub(crate) records: &'a RecordIndex,
    pub(crate) guard: &'a RefCell<FxHashSet<(usize, usize)>>,
    /// Build violations into `out`. When false the walk is the membership fast
    /// path: it never touches `out`, never builds a path, and short-circuits.
    pub(crate) explain: bool,
    /// In explain mode, stop at the first failure instead of aggregating siblings.
    pub(crate) fail_fast: bool,
}

/// A precomputed lookup for one record (`KeyedMap`) node: declared field name to
/// its index in the node's `fields`, plus the count of required fields. Built
/// once when a validator is first used and reused across calls, so a wide record
/// no longer rebuilds and rehashes its field-name map on every validation.
pub(crate) struct RecordPlan {
    by_name: FxHashMap<Box<str>, usize>,
    required: usize,
}

/// The record index for a whole validator: each record's `fields`-buffer address
/// mapped to its [`RecordPlan`]. The buffer address is stable for the life of the
/// (immutable) schema, and the index is rebuilt per validator from its own
/// schema, so an entry always refers to the same live node.
pub(crate) type RecordIndex = FxHashMap<usize, RecordPlan>;

/// Build the record index for a finished schema plus its recursion definitions.
/// A record with no declared fields is skipped: its plan is trivial and an empty
/// `Vec` carries no distinct buffer address to key on.
pub(crate) fn build_record_index(schema: &Schema, defs: &[Schema]) -> RecordIndex {
    let mut index = RecordIndex::default();
    collect_records(schema, &mut index);
    for def in defs {
        collect_records(def, &mut index);
    }
    index
}

fn collect_records(schema: &Schema, index: &mut RecordIndex) {
    match schema {
        Schema::KeyedMap { fields, defaults } => {
            if !fields.is_empty() {
                index
                    .entry(fields.as_ptr() as usize)
                    .or_insert_with(|| RecordPlan {
                        by_name: fields
                            .iter()
                            .enumerate()
                            .map(|(i, f)| (f.name.as_str().into(), i))
                            .collect(),
                        required: fields.iter().filter(|f| f.required).count(),
                    });
            }
            for f in fields {
                collect_records(&f.schema, index);
            }
            for (key_schema, value_schema) in defaults {
                collect_records(key_schema, index);
                collect_records(value_schema, index);
            }
        }
        Schema::Union(members) | Schema::Intersection(members) => {
            for member in members {
                collect_records(member, index);
            }
        }
        Schema::Set(inner) | Schema::FrozenSet(inner) | Schema::Complement(inner) => {
            collect_records(inner, index);
        }
        Schema::Refine { base, .. } => collect_records(base, index),
        Schema::Object { fields, .. } => {
            for f in fields {
                collect_records(&f.schema, index);
            }
        }
        Schema::Seq { regex, .. } => collect_seq_records(regex, index),
        _ => {}
    }
}

fn collect_seq_records(regex: &SeqRegex, index: &mut RecordIndex) {
    match regex {
        SeqRegex::Empty => {}
        SeqRegex::Elem(schema) => collect_records(schema, index),
        SeqRegex::Cat(parts) | SeqRegex::Or(parts) => {
            for part in parts {
                collect_seq_records(part, index);
            }
        }
        SeqRegex::Star(inner) => collect_seq_records(inner, index),
    }
}

/// Whether the sibling loop should stop after a failure: always in the fast path
/// (membership is already decided), and in explain mode only under `fail_fast`.
fn stop(ctx: Ctx<'_>) -> bool {
    !ctx.explain || ctx.fail_fast
}

/// Decide whether `value` is a member of `schema`'s set.
///
/// In explain mode a [`Violation`] is pushed into `out` for every independent
/// failure and `path` accumulates the location of the current value; in fast
/// mode nothing is allocated. The returned bool is authoritative: it is the same
/// answer `is_valid` and `validate` report.
pub(crate) fn member(
    schema: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    match schema {
        Schema::Anything | Schema::Any => true,
        // Bottom admits nothing; an unresolved self-reference is never a member.
        Schema::Nothing => admit(false, schema, value, path, ctx, out),
        Schema::SelfRef(_) => {
            if ctx.explain {
                out.push(Violation {
                    code: "unresolved_recursion",
                    path: path.clone(),
                    expected: "a resolved recursive value".to_owned(),
                    value_summary: summarize_value(value),
                });
            }
            false
        }
        Schema::NoneType => admit(value.is_none(), schema, value, path, ctx, out),
        Schema::Bool => admit(value.is_bool(), schema, value, path, ctx, out),
        // bool subclasses int, so True/False are ints: Bool is a subset of Int.
        Schema::Int => admit(value.is_int(), schema, value, path, ctx, out),
        Schema::Float => admit(value.is_float(), schema, value, path, ctx, out),
        Schema::Str => admit(value.is_str(), schema, value, path, ctx, out),
        Schema::Bytes => admit(value.is_bytes(), schema, value, path, ctx, out),
        Schema::Literal(index) => check_literal(*index, value, path, ctx, out),
        Schema::Seq { container, regex } => check_seq(*container, regex, value, path, ctx, out),
        Schema::Set(element) => check_set(element, value, path, ctx, out),
        Schema::FrozenSet(element) => check_frozenset(element, value, path, ctx, out),
        Schema::KeyedMap { fields, defaults } => {
            // Membership is the single-pass fast check; on failure the explain
            // pass re-walks in declared order to aggregate ordered violations.
            let ok = keyed_map_matches(fields, defaults, value, ctx);
            if !ok && ctx.explain {
                keyed_map_explain(fields, defaults, value, path, ctx, out);
            }
            ok
        }
        Schema::Union(members) => check_union(members, value, path, ctx, out),
        Schema::Intersection(members) => check_intersection(members, value, path, ctx, out),
        Schema::Complement(inner) => check_complement(inner, value, path, ctx, out),
        Schema::Instance(index) => check_instance(*index, value, path, ctx, out),
        Schema::Object {
            class_index,
            fields,
        } => check_object(*class_index, fields, value, path, ctx, out),
        Schema::Refine { base, constraints } => {
            check_refine(base, constraints, value, path, ctx, out)
        }
        Schema::Ref(id) => check_ref(*id, value, path, ctx, out),
    }
}

/// A leaf decision: pass `ok` through, recording a type/value mismatch when it is
/// false in explain mode.
fn admit(
    ok: bool,
    schema: &Schema,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if !ok && ctx.explain {
        out.push(mismatch(schema, value, path));
    }
    ok
}

fn check_literal(
    index: usize,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let literal = ctx.pool[index].bind(value.py());
    let ok = value
        .to_python()
        .is_ok_and(|obj| literal_matches(&obj, literal));
    if !ok && ctx.explain {
        out.push(Violation {
            code: "literal_value",
            path: path.to_vec(),
            expected: format!("the literal {}", summarize(literal)),
            value_summary: summarize_value(value),
        });
    }
    ok
}

/// Membership for a sequence node: the value is a list or tuple whose element
/// sequence matches the regex. The frontend emits only *linear* regexes — a
/// fixed positional prefix then an optional repeated tail — so the elements are
/// walked lazily with no automaton and no collection, identical in cost to a
/// direct positional or homogeneous check. JSON arrays are lists.
fn check_seq(
    container: SeqKind,
    regex: &SeqRegex,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let (kind_word, type_code, len_code) = match container {
        SeqKind::List => ("list", "list_type", "list_length"),
        SeqKind::Tuple => ("tuple", "tuple_type", "tuple_length"),
    };
    let Some((prefix, tail)) = regex.linear() else {
        // Alternation and nesting are built only inside the decision procedure;
        // such a regex never reaches value membership.
        return false;
    };
    match (container, value) {
        (SeqKind::List, Value::Py(v)) => {
            let Ok(list) = v.cast::<PyList>() else {
                return type_fail(type_code, kind_word, value, path, ctx, out);
            };
            if !seq_len_ok(list.len(), prefix.len(), tail.is_some()) {
                return seq_length_fail(len_code, kind_word, &prefix, tail, value, path, ctx, out);
            }
            let mut ok = true;
            for (i, item) in list.iter().enumerate() {
                ok &= seq_element(&prefix, tail, i, &Value::Py(&item), path, ctx, out);
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        (SeqKind::List, Value::Json(py, JsonValue::Array(items))) => {
            if !seq_len_ok(items.len(), prefix.len(), tail.is_some()) {
                return seq_length_fail(len_code, kind_word, &prefix, tail, value, path, ctx, out);
            }
            let mut ok = true;
            for (i, item) in items.iter().enumerate() {
                ok &= seq_element(&prefix, tail, i, &Value::Json(*py, item), path, ctx, out);
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        (SeqKind::Tuple, Value::Py(v)) => {
            let Ok(tuple) = v.cast::<PyTuple>() else {
                return type_fail(type_code, kind_word, value, path, ctx, out);
            };
            if !seq_len_ok(tuple.len(), prefix.len(), tail.is_some()) {
                return seq_length_fail(len_code, kind_word, &prefix, tail, value, path, ctx, out);
            }
            let mut ok = true;
            for (i, item) in tuple.iter().enumerate() {
                ok &= seq_element(&prefix, tail, i, &Value::Py(&item), path, ctx, out);
                if !ok && stop(ctx) {
                    return false;
                }
            }
            ok
        }
        // A tuple is never a JSON value; a list needs a JSON array.
        _ => type_fail(type_code, kind_word, value, path, ctx, out),
    }
}

/// Whether the element count fits the regex: exactly the prefix length with no
/// tail, or at least the prefix length when a repeated tail follows.
fn seq_len_ok(len: usize, prefix_len: usize, has_tail: bool) -> bool {
    if has_tail {
        len >= prefix_len
    } else {
        len == prefix_len
    }
}

/// Match one element at position `i`: the prefix schema at `i`, or the repeated
/// tail past the prefix. The index segment is pushed only in explain mode.
fn seq_element(
    prefix: &[&Schema],
    tail: Option<&Schema>,
    i: usize,
    item: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let schema = prefix
        .get(i)
        .copied()
        .unwrap_or_else(|| tail.expect("length already checked"));
    if ctx.explain {
        path.push(PathSegment::Index(i));
    }
    let ok = member(schema, item, path, ctx, out);
    if ctx.explain {
        path.pop();
    }
    ok
}

/// A sequence-length mismatch: terminal, since the positional match is then
/// meaningless. A tailless regex wants an exact length; a tailed one a minimum.
#[allow(clippy::too_many_arguments)]
fn seq_length_fail(
    len_code: &'static str,
    kind_word: &str,
    prefix: &[&Schema],
    tail: Option<&Schema>,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if ctx.explain {
        let expected = if tail.is_some() {
            format!("{kind_word} of length at least {}", prefix.len())
        } else {
            format!("{kind_word} of length {}", prefix.len())
        };
        out.push(Violation {
            code: len_code,
            path: path.to_vec(),
            expected,
            value_summary: summarize_value(value),
        });
    }
    false
}

/// A set whose every element matches `element`. Set order is not meaningful, so
/// element failures carry no index segment. JSON has no sets.
fn check_set(
    element: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Value::Py(v) = value else {
        return type_fail("set_type", "set", value, path, ctx, out);
    };
    let Ok(set) = v.cast::<PySet>() else {
        return type_fail("set_type", "set", value, path, ctx, out);
    };
    let mut ok = true;
    for item in set.iter() {
        ok &= member(element, &Value::Py(&item), path, ctx, out);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// A frozenset whose every element matches `element`. JSON has no frozensets.
fn check_frozenset(
    element: &Schema,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let Value::Py(v) = value else {
        return type_fail("frozenset_type", "frozenset", value, path, ctx, out);
    };
    let Ok(set) = v.cast::<PyFrozenSet>() else {
        return type_fail("frozenset_type", "frozenset", value, path, ctx, out);
    };
    let mut ok = true;
    for item in set.iter() {
        ok &= member(element, &Value::Py(&item), path, ctx, out);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// Membership for a keyed map: named fields, then a default clause for every
/// other key. The walk is inverted — it visits each entry once — and a JSON
/// object's keys are strings, a duplicate keeping its last value as
/// `json.loads` does.
fn keyed_map_matches(
    fields: &[Field],
    defaults: &[(Schema, Schema)],
    value: &Value<'_, '_>,
    ctx: Ctx<'_>,
) -> bool {
    match value {
        Value::Py(v) => keyed_map_matches_py(fields, defaults, v, ctx),
        Value::Json(py, JsonValue::Object(entries)) => {
            keyed_map_matches_json(fields, defaults, *py, entries, ctx)
        }
        Value::Json(..) => false,
    }
}

/// Whether `(key, val)` is covered by some default clause: the key belongs to a
/// clause's key schema and the value to that clause's value schema. The clauses
/// denote a union of key×value rectangles.
fn covered(
    defaults: &[(Schema, Schema)],
    key: &Value<'_, '_>,
    val: &Value<'_, '_>,
    ctx: Ctx<'_>,
) -> bool {
    let sub = fast(ctx);
    defaults.iter().any(|(key_schema, value_schema)| {
        member(key_schema, key, &mut Vec::new(), sub, &mut Vec::new())
            && member(value_schema, val, &mut Vec::new(), sub, &mut Vec::new())
    })
}

/// The keyed-map fast path over a Python dict. A string key naming a declared
/// field is checked against it; any other key (non-string, or undeclared) must
/// be covered by a default clause. Closed records have no clauses, so an
/// undeclared key is rejected; an open record's `anything` clause covers it.
///
/// The declared-field lookup comes from the validator's precomputed
/// [`RecordIndex`] when present, so a wide record skips rebuilding its name map
/// on every call; a record not in the index (an empty one, or a node the
/// build-time traversal did not reach) falls back to building the map here.
fn keyed_map_matches_py(
    fields: &[Field],
    defaults: &[(Schema, Schema)],
    dict: &Bound<'_, PyAny>,
    ctx: Ctx<'_>,
) -> bool {
    let Ok(dict) = dict.cast::<PyDict>() else {
        return false;
    };
    if let Some(plan) = ctx.records.get(&(fields.as_ptr() as usize)) {
        keyed_map_scan(fields, defaults, dict, ctx, plan.required, |name| {
            plan.by_name.get(name).copied()
        })
    } else {
        let declared: FxHashMap<&str, usize> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.as_str(), i))
            .collect();
        let required = fields.iter().filter(|f| f.required).count();
        keyed_map_scan(fields, defaults, dict, ctx, required, |name| {
            declared.get(name).copied()
        })
    }
}

/// Walk a dict once against a record's fields, resolving each string key to a
/// declared-field index through `lookup` (a precomputed plan or a freshly built
/// map). A key that resolves checks its value against that field; any other key
/// must be covered by a default clause. The record matches iff every entry
/// matches and every required field was seen.
fn keyed_map_scan(
    fields: &[Field],
    defaults: &[(Schema, Schema)],
    dict: &Bound<'_, PyDict>,
    ctx: Ctx<'_>,
    mut required_remaining: usize,
    lookup: impl Fn(&str) -> Option<usize>,
) -> bool {
    let sub = fast(ctx);
    for (key, val) in dict.iter() {
        let index = key
            .cast::<PyString>()
            .ok()
            .and_then(|s| s.to_str().ok())
            .and_then(&lookup);
        match index {
            Some(i) => {
                let field = &fields[i];
                if !member(
                    &field.schema,
                    &Value::Py(&val),
                    &mut Vec::new(),
                    sub,
                    &mut Vec::new(),
                ) {
                    return false;
                }
                if field.required {
                    required_remaining -= 1;
                }
            }
            None => {
                if !covered(defaults, &Value::Py(&key), &Value::Py(&val), ctx) {
                    return false;
                }
            }
        }
    }
    required_remaining == 0
}

/// The keyed-map fast path over a JSON object. Keys are strings; a duplicate key
/// keeps its last value (a reverse find), as `json.loads` does. Records are
/// small, so a linear scan beats building a per-object map.
fn keyed_map_matches_json(
    fields: &[Field],
    defaults: &[(Schema, Schema)],
    py: Python<'_>,
    entries: &[(Cow<'_, str>, JsonValue<'_>)],
    ctx: Ctx<'_>,
) -> bool {
    let sub = fast(ctx);
    for field in fields {
        match entries
            .iter()
            .rev()
            .find(|(key, _)| field.name == key.as_ref())
        {
            Some((_, val)) => {
                if !member(
                    &field.schema,
                    &Value::Json(py, val),
                    &mut Vec::new(),
                    sub,
                    &mut Vec::new(),
                ) {
                    return false;
                }
            }
            None if field.required => return false,
            None => {}
        }
    }
    // Every key that is not a declared field must be covered by a default clause,
    // testing each key's last value (json.loads semantics).
    for (i, (key, val)) in entries.iter().enumerate() {
        if fields.iter().any(|f| f.name == key.as_ref()) {
            continue;
        }
        if entries[i + 1..].iter().any(|(later, _)| later == key) {
            continue;
        }
        let key_value = JsonValue::Str(Cow::Borrowed(key.as_ref()));
        if !covered(
            defaults,
            &Value::Json(py, &key_value),
            &Value::Json(py, val),
            ctx,
        ) {
            return false;
        }
    }
    true
}

/// The explain pass over a keyed map, run only after [`keyed_map_matches`] has
/// reported the value is not a member. It walks in declared order — present
/// fields checked in order, then absent required keys — then reports each
/// undeclared key: an uncovered key with no clauses reads as an unexpected key,
/// and with clauses its key and value are checked against the first clause.
fn keyed_map_explain(
    fields: &[Field],
    defaults: &[(Schema, Schema)],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) {
    let Value::Py(v) = value else {
        // The explain pass only ever sees a Python value; a JSON value here is
        // unreachable, but keep the false-implies-a-violation invariant.
        out.push(type_mismatch("dict_type", "dict", value, path));
        return;
    };
    let Ok(dict) = v.cast::<PyDict>() else {
        out.push(type_mismatch("dict_type", "dict", value, path));
        return;
    };
    let declared: FxHashSet<&str> = fields.iter().map(|field| field.name.as_str()).collect();
    for field in fields {
        match dict.get_item(field.name.as_str()) {
            Ok(Some(item)) => {
                path.push(PathSegment::Key(field.name.clone()));
                member(&field.schema, &Value::Py(&item), path, ctx, out);
                path.pop();
            }
            Ok(None) if field.required => out.push(located(
                path,
                field.name.clone(),
                "missing_key",
                format!("required key {:?}", field.name),
                "missing".to_owned(),
            )),
            Ok(None) => {}
            Err(_) => out.push(type_mismatch("dict_type", "dict", value, path)),
        }
        if ctx.fail_fast && !out.is_empty() {
            return;
        }
    }
    for (key, val) in dict.iter() {
        if let Some(name) = key.cast::<PyString>().ok().and_then(|s| s.to_str().ok())
            && declared.contains(name)
        {
            continue;
        }
        if covered(defaults, &Value::Py(&key), &Value::Py(&val), ctx) {
            continue;
        }
        if let Some((key_schema, value_schema)) = defaults.first() {
            // A clause exists but did not cover this key: surface the key and
            // value violations against it (the homogeneous-mapping error).
            path.push(PathSegment::Key(key_label(&key)));
            member(key_schema, &Value::Py(&key), path, ctx, out);
            member(value_schema, &Value::Py(&val), path, ctx, out);
            path.pop();
        } else {
            // A closed record: the key is simply not allowed.
            let key_text = key
                .str()
                .map_or_else(|_| String::new(), |text| text.to_string());
            out.push(located(
                path,
                key_text.clone(),
                "extra_key",
                "no unexpected key".to_owned(),
                format!("{key_text:?}"),
            ));
        }
        if ctx.fail_fast && !out.is_empty() {
            return;
        }
    }
}

fn check_union(
    members: &[Schema],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // A value is a member iff it matches at least one branch; decide that on the
    // fast path, where a discarded branch pays for no path or violation.
    let sub = fast(ctx);
    if members
        .iter()
        .any(|m| member(m, value, &mut Vec::new(), sub, &mut Vec::new()))
    {
        return true;
    }
    if !ctx.explain {
        return false;
    }
    // No branch matches. Explain the *closest* branch — the one that descended
    // furthest into the value before failing — rather than dumping every branch.
    // "Furthest" is the greatest path depth past the union's own location. When
    // no branch makes progress (every branch is a flat type mismatch, e.g.
    // `int | str` against a float), fall back to a single union error. The probe
    // aggregates regardless of fail_fast so the deepest progress is visible; this
    // runs only on the error path.
    let base_depth = path.len();
    let probe = Ctx {
        explain: true,
        fail_fast: false,
        ..ctx
    };
    let mut best: Option<(usize, Vec<Violation>)> = None;
    for branch_schema in members {
        let mut branch = Vec::new();
        member(branch_schema, value, path, probe, &mut branch);
        let progress = branch
            .iter()
            .map(|v| v.path.len())
            .max()
            .unwrap_or(base_depth)
            .saturating_sub(base_depth);
        // Strictly greater keeps the earliest branch on a tie.
        let replace = best
            .as_ref()
            .is_none_or(|(best_progress, _)| progress > *best_progress);
        if replace {
            best = Some((progress, branch));
        }
    }
    match best {
        Some((progress, branch)) if progress > 0 => out.extend(branch),
        _ => {
            let labels: Vec<&str> = members.iter().map(Schema::expected).collect();
            out.push(Violation {
                code: "union_error",
                path: path.clone(),
                expected: format!("one of: {}", labels.join(", ")),
                value_summary: summarize_value(value),
            });
        }
    }
    false
}

fn check_intersection(
    members: &[Schema],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // Every member must hold; in explain mode each member's failure is collected.
    let mut ok = true;
    for member_schema in members {
        ok &= member(member_schema, value, path, ctx, out);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

fn check_complement(
    inner: &Schema,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // A value matches the complement iff it does not match the inner schema; the
    // inner explanation is irrelevant, so decide it on the fast path.
    if member(inner, value, &mut Vec::new(), fast(ctx), &mut Vec::new()) {
        if ctx.explain {
            out.push(Violation {
                code: "unexpected_match",
                path: path.to_vec(),
                expected: format!("not {}", inner.expected()),
                value_summary: summarize_value(value),
            });
        }
        return false;
    }
    true
}

fn check_instance(
    index: usize,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let class = ctx.pool[index].bind(value.py());
    let ok = value
        .to_python()
        .is_ok_and(|obj| obj.is_instance(class).unwrap_or(false));
    if !ok && ctx.explain {
        out.push(type_mismatch(
            "instance_type",
            &class_label(class),
            value,
            path,
        ));
    }
    ok
}

fn check_object(
    class_index: usize,
    fields: &[Field],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // A JSON value materializes to a builtin, never an instance of a user class.
    let Ok(obj) = value.to_python() else {
        return false;
    };
    let class = ctx.pool[class_index].bind(value.py());
    if !obj.is_instance(class).unwrap_or(false) {
        // Not an instance: the attribute checks below cannot be trusted.
        if ctx.explain {
            out.push(type_mismatch(
                "instance_type",
                &class_label(class),
                value,
                path,
            ));
        }
        return false;
    }
    let mut ok = true;
    for field in fields {
        if let Ok(attr) = obj.getattr(field.name.as_str()) {
            if ctx.explain {
                path.push(PathSegment::Key(field.name.clone()));
            }
            ok &= member(&field.schema, &Value::Py(&attr), path, ctx, out);
            if ctx.explain {
                path.pop();
            }
        } else {
            if ctx.explain {
                out.push(located(
                    path,
                    field.name.clone(),
                    "missing_attribute",
                    format!("attribute {:?}", field.name),
                    "missing".to_owned(),
                ));
            }
            ok = false;
        }
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

fn check_refine(
    base: &Schema,
    constraints: &[Constraint],
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    // Constraints narrow the base set, so they are meaningful only on a base
    // member: if the base fails, report that and do not run the constraints.
    if !member(base, value, path, ctx, out) {
        return false;
    }
    let Ok(obj) = value.to_python() else {
        return false;
    };
    let mut ok = true;
    for constraint in constraints {
        ok &= check_constraint(constraint, &obj, path, ctx, out);
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}

/// Whether `value` (already a base member, materialized once) satisfies one
/// constraint, recording a violation on failure in explain mode.
fn check_constraint(
    constraint: &Constraint,
    value: &Bound<'_, PyAny>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let py = value.py();
    let (ok, code, expected): (bool, &'static str, String) = match constraint {
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
        Constraint::MultipleOf(i) => {
            let operand = ctx.pool[*i].bind(py);
            (
                is_multiple_of(value, operand),
                "not_multiple_of",
                format!("a multiple of {}", summarize(operand)),
            )
        }
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
    if !ok && ctx.explain {
        out.push(Violation {
            code,
            path: path.to_vec(),
            expected,
            value_summary: summarize(value),
        });
    }
    ok
}

/// The most levels of recursive descent allowed before a value is rejected. A
/// finite value never reaches this; the bound exists so a pathologically deep
/// value fails with `recursion_limit` instead of overflowing the native stack.
const MAX_RECURSION_DEPTH: usize = 128;

fn check_ref(
    id: usize,
    value: &Value<'_, '_>,
    path: &mut Vec<PathSegment>,
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    let key = (value.id(), id);
    let depth = {
        let mut guard = ctx.guard.borrow_mut();
        if !guard.insert(key) {
            if ctx.explain {
                out.push(Violation {
                    code: "recursion_loop",
                    path: path.clone(),
                    expected: "a finite (non-cyclic) value".to_owned(),
                    value_summary: summarize_value(value),
                });
            }
            return false;
        }
        guard.len()
    };
    if depth > MAX_RECURSION_DEPTH {
        ctx.guard.borrow_mut().remove(&key);
        if ctx.explain {
            out.push(Violation {
                code: "recursion_limit",
                path: path.clone(),
                expected: format!("at most {MAX_RECURSION_DEPTH} levels of recursion"),
                value_summary: summarize_value(value),
            });
        }
        return false;
    }
    let result = member(&ctx.defs[id], value, path, ctx, out);
    ctx.guard.borrow_mut().remove(&key);
    result
}

/// A copy of `ctx` switched to the membership fast path (no explanation), for the
/// speculative sub-checks of union, complement, and the record fast walk.
fn fast(ctx: Ctx<'_>) -> Ctx<'_> {
    Ctx {
        explain: false,
        fail_fast: true,
        ..ctx
    }
}

/// A type/value mismatch for a leaf schema.
fn mismatch(schema: &Schema, value: &Value<'_, '_>, path: &[PathSegment]) -> Violation {
    Violation {
        code: schema.error_code(),
        path: path.to_vec(),
        expected: schema.expected().to_owned(),
        value_summary: summarize_value(value),
    }
}

/// Record a structural type mismatch and report non-membership.
fn type_fail(
    code: &'static str,
    expected: &str,
    value: &Value<'_, '_>,
    path: &[PathSegment],
    ctx: Ctx<'_>,
    out: &mut Vec<Violation>,
) -> bool {
    if ctx.explain {
        out.push(type_mismatch(code, expected, value, path));
    }
    false
}

fn type_mismatch(
    code: &'static str,
    expected: &str,
    value: &Value<'_, '_>,
    path: &[PathSegment],
) -> Violation {
    Violation {
        code,
        path: path.to_vec(),
        expected: expected.to_owned(),
        value_summary: summarize_value(value),
    }
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

/// A short repr-style summary of a value, materializing a JSON value first.
fn summarize_value(value: &Value<'_, '_>) -> String {
    match value.to_python() {
        Ok(obj) => summarize(&obj),
        Err(_) => "<unrepresentable>".to_owned(),
    }
}

/// Whether `value` is the typed singleton denoted by `literal`: same type and
/// equal. The same-type guard rules out Python's cross-type equality
/// (`1 == True == 1.0`), so `Literal[1]` denotes `{1}`, not `{1, True, 1.0}`.
fn literal_matches(value: &Bound<'_, PyAny>, literal: &Bound<'_, PyAny>) -> bool {
    value.get_type().is(literal.get_type()) && value.eq(literal).unwrap_or(false)
}

/// Whether `value % operand == 0`. A non-numeric value (whose modulo errors or
/// is not defined) is not a multiple. The remainder is zero iff it is falsy.
fn is_multiple_of(value: &Bound<'_, PyAny>, operand: &Bound<'_, PyAny>) -> bool {
    value
        .call_method1("__mod__", (operand,))
        .ok()
        .and_then(|remainder| remainder.is_truthy().ok())
        .is_some_and(|nonzero| !nonzero)
}

/// Run a user predicate and report whether it returned a truthy result.
fn predicate_passes(predicate: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    predicate.call1((value,))?.is_truthy()
}

/// Interpret a rich-comparison result, treating an error as "did not hold".
fn cmp(result: PyResult<bool>) -> bool {
    result.unwrap_or(false)
}

/// A short, printable label for a mapping key, used in error paths.
fn key_label(key: &Bound<'_, PyAny>) -> String {
    match key.str() {
        Ok(text) => truncate(&text.to_string(), 40),
        Err(_) => summarize(key),
    }
}
