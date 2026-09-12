//! The keyed maps and attribute records a value is read as.
//!
//! A record is the shape whose membership is a question per key rather than
//! per position: which keys the value carries, which of them the schema
//! declares, and what a key the schema does not declare is covered by. The
//! walk's other containers ask about elements and share none of that, which is
//! why these live together and beside rather than inside the dispatcher.

use std::borrow::Cow;
use std::ops::ControlFlow;
use std::sync::Arc;

use jiter::JsonValue;
use pyo3::prelude::*;
use pyo3::sync::critical_section::with_critical_section;
use pyo3::types::{PyDict, PyString};
use rustc_hash::{FxHashMap, FxHashSet};
use valgebra_core::{Field, MapClause, PathSegment, Schema};

use super::{Frame, Scan, fast, is_fatal, member, mutated, record_fatal, stop};
use crate::check::ctx::Ctx;
use crate::check::index::RecordPlan;
use crate::check::violation::{key_segment, located, type_mismatch};
use crate::input::Value;

/// Visit a dict's entries, refusing rather than panicking when the dict changes
/// size underneath the scan.
///
/// The iterator `PyO3` hands out panics when the dict's length moves, and the walk
/// runs Python at every entry, so that panic is reachable from an ordinary
/// schema — and a panic is not one of the answers this library gives: it crosses
/// the boundary as a `BaseException` that no caller catches as a validation
/// failure. The scan asks the same question one step earlier, before each step
/// rather than inside it, and stops at the entry count it began with, so the
/// iterator is never advanced into either of the states it panics in. The
/// critical section keeps a second thread out of the dict for the parts of the
/// scan that do not call back into the interpreter.
pub(super) fn scan_dict<'py>(
    dict: &Bound<'py, PyDict>,
    mut visit: impl FnMut(&Bound<'py, PyAny>, &Bound<'py, PyAny>) -> ControlFlow<()>,
) -> Scan {
    with_critical_section(dict.as_any(), || {
        let entries = dict.len();
        let mut iter = dict.iter();
        let mut seen = 0;
        while seen < entries {
            if dict.len() != entries {
                return Scan::Unreadable;
            }
            let Some((key, value)) = iter.next() else {
                break;
            };
            seen += 1;
            if visit(&key, &value).is_break() {
                return Scan::Stopped;
            }
        }
        if dict.len() == entries {
            Scan::Complete
        } else {
            Scan::Unreadable
        }
    })
}

/// Membership for a keyed map: named fields, then a default clause for every
/// other key. The walk is inverted — it visits each entry once — and a JSON
/// object's keys are strings, a duplicate keeping its last value as
/// `json.loads` does.
pub(super) fn keyed_map_matches(
    fields: &[Field],
    defaults: &[MapClause],
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
fn covered(defaults: &[MapClause], key: &Value<'_, '_>, val: &Value<'_, '_>, ctx: Ctx<'_>) -> bool {
    // One pair of scratch buffers for every clause rather than a pair per call
    // into the walk. Neither is written on this path -- a fast walk reports
    // nothing and records no location -- but each is a value with a destructor,
    // and building and dropping four of them per key is work the answer does
    // not depend on.
    let (mut path, mut out) = (Vec::new(), Vec::new());
    let mut sub = Frame::new(&mut path, &mut out, fast(ctx));
    defaults
        .iter()
        .any(|clause| member(&clause.key, key, &mut sub) && member(&clause.value, val, &mut sub))
}

/// A closed record's membership, asked key by key rather than read entry by
/// entry, or `None` where that reading does not settle it.
///
/// A closed record declares every key the value may carry, so the value belongs
/// exactly when each declared key it holds matches and it holds nothing else --
/// and "nothing else" is a count, since a dict cannot repeat a key. Asking for
/// the declared keys costs one probe each with the key's own hash, where
/// scanning the value costs an iteration step, a decode of the key's bytes, a
/// second hash of those bytes and a comparison against the name they matched.
///
/// A key is resolved the way Python resolves one -- by the dict's own lookup --
/// rather than by decoding its bytes and matching those, so a key of a `str`
/// subclass with an `__eq__` of its own is found exactly where indexing the
/// dict would find it.
///
/// `None` means "ask the scan instead": the record is open, so an undeclared
/// key may still be covered by a clause; the plan has no interned key for a
/// field; or a probe raised, which is not an answer. A value that changes size
/// under the probes is not one of those: it is answered here, as the scan
/// answers it, because there is no reading of it left to fall back to.
/// What a record's clauses say about a key it does not declare, where that can
/// be said without reading the key and its value together.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Undeclared {
    /// No clause: a closed record, and a key to spare refuses it.
    Refused,
    /// The top clause, `anything: anything`: any key, any value, by definition.
    Admitted,
    /// `str: anything` -- the clause a `TypedDict` carries, since its keys are
    /// strings by the typing spec -- admits a key exactly when the key is a
    /// `str`, whatever its value. Every declared key is a `str` too, so the
    /// question is asked of every key alike and no key need be resolved.
    AnyStr,
    /// A clause that reads the key or the value: each undeclared key is asked of
    /// it, which is the scan.
    Read,
}

impl Undeclared {
    fn of(defaults: &[MapClause]) -> Undeclared {
        match defaults {
            [] => Undeclared::Refused,
            [clause] if *clause == MapClause::top() => Undeclared::Admitted,
            [clause] if clause.key == Schema::Str && clause.value == Schema::ANYTHING => {
                Undeclared::AnyStr
            }
            _ => Undeclared::Read,
        }
    }
}

fn keyed_map_asks_for_its_keys(
    fields: &[Field],
    defaults: &[MapClause],
    dict: &Bound<'_, PyDict>,
    ctx: Ctx<'_>,
    plan: &RecordPlan,
) -> Option<bool> {
    // By its keys where the keys settle it. A record whose clause reads an
    // undeclared key together with its value has to be scanned, and the scan
    // walks the declared fields as it goes, so reading those by key first would
    // be work done twice; such a record is not read here.
    let undeclared = Undeclared::of(defaults);
    if undeclared == Undeclared::Read || plan.keys.len() != fields.len() {
        return None;
    }
    // One pair of scratch buffers for the record, as the scan takes: a fast
    // walk writes to neither.
    let (mut path, mut out) = (Vec::new(), Vec::new());
    let mut sub = Frame::new(&mut path, &mut out, fast(ctx));
    with_critical_section(dict.as_any(), || {
        let entries = dict.len();
        let mut present = 0usize;
        for (position, field) in fields.iter().enumerate() {
            let key = plan.keys.get(position)?.bind(dict.py());
            match dict.get_item(key) {
                Ok(Some(value)) => {
                    present += 1;
                    if !member(&field.schema, &Value::Py(&value), &mut sub) {
                        return Some(false);
                    }
                }
                // A key the value does not carry: the record still matches when
                // the field is optional.
                Ok(None) if !field.required => {}
                Ok(None) => return Some(false),
                // A failed probe is not an answer -- an unhashable key cannot be
                // in a dict, but a `__eq__` that raises can stop the lookup.
                Err(_) => return None,
            }
        }
        if dict.len() != entries {
            // The value changed while it was being read, so there is no reading
            // to answer from: not a member, exactly as the scan answers it, and
            // the explain pass names the mutation.
            return Some(false);
        }
        // Every key the value carries is one of the declared ones exactly when
        // the count of declared keys found equals the count it holds; there is
        // then nothing for a clause to govern, whatever the clause.
        if present == entries {
            return Some(true);
        }
        match undeclared {
            Undeclared::Refused => Some(false),
            Undeclared::Admitted => Some(true),
            // A `str` key is admitted whatever it names, so the keys are read for
            // their type and nothing else -- no name is resolved, no value is
            // walked -- and the first key of another type refuses the record.
            Undeclared::AnyStr => {
                let keys = scan_dict(dict, |key, _| {
                    if key.is_instance_of::<PyString>() {
                        ControlFlow::Continue(())
                    } else {
                        ControlFlow::Break(())
                    }
                });
                Some(matches!(keys, Scan::Complete))
            }
            // Kept out by the guard above; were it not, the scan still decides,
            // at the cost of the reading just done.
            Undeclared::Read => None,
        }
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
    defaults: &[MapClause],
    dict: &Bound<'_, PyAny>,
    ctx: Ctx<'_>,
) -> bool {
    let Ok(dict) = dict.cast::<PyDict>() else {
        return false;
    };
    if let Some(plan) = ctx.records.get(&(fields.as_ptr() as usize)) {
        if let Some(answered) = keyed_map_asks_for_its_keys(fields, defaults, dict, ctx, plan) {
            return answered;
        }
        keyed_map_scan(fields, defaults, dict, ctx, plan.required, |name| {
            plan.by_name.get(name).copied()
        })
    } else {
        let declared: FxHashMap<&str, usize> = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (&*f.name, i))
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
    defaults: &[MapClause],
    dict: &Bound<'_, PyDict>,
    ctx: Ctx<'_>,
    mut required_remaining: usize,
    lookup: impl Fn(&str) -> Option<usize>,
) -> bool {
    // Scratch buffers for the whole record, not one pair per field: a fast walk
    // writes to neither, and a fifty-field record was building and dropping a
    // hundred of them to answer one membership question.
    let (mut path, mut out) = (Vec::new(), Vec::new());
    let mut sub = Frame::new(&mut path, &mut out, fast(ctx));
    let scan = scan_dict(dict, |key, val| {
        // A non-string key, or a string carrying a lone surrogate (which cannot
        // equal a field name, since names are valid UTF-8 by build-time check),
        // resolves to no field and must instead be covered by a default clause.
        let index = key
            .cast::<PyString>()
            .ok()
            .and_then(|s| s.to_str().ok())
            .and_then(&lookup);
        match index.and_then(|i| fields.get(i)) {
            Some(field) => {
                if !member(&field.schema, &Value::Py(val), &mut sub) {
                    return ControlFlow::Break(());
                }
                if field.required {
                    // Saturating: the counter is the precomputed required-field
                    // count, so it cannot legitimately pass zero, but a malformed
                    // index must not wrap a release build into a false pass.
                    required_remaining = required_remaining.saturating_sub(1);
                }
            }
            None => {
                if !covered(defaults, &Value::Py(key), &Value::Py(val), ctx) {
                    return ControlFlow::Break(());
                }
            }
        }
        ControlFlow::Continue(())
    });
    matches!(scan, Scan::Complete) && required_remaining == 0
}

/// The keyed-map fast path over a JSON object. Keys are strings; a duplicate key
/// keeps its last value (a reverse find), as `json.loads` does. Records are
/// small, so a linear scan beats building a per-object map.
pub(super) fn keyed_map_matches_json(
    fields: &[Field],
    defaults: &[MapClause],
    py: Python<'_>,
    entries: &[(Cow<'_, str>, JsonValue<'_>)],
    ctx: Ctx<'_>,
) -> bool {
    let (mut path, mut out) = (Vec::new(), Vec::new());
    let mut sub = Frame::new(&mut path, &mut out, fast(ctx));
    // A record whose keys settle it resolves the document's keys through the
    // plan instead of searching the document once per field. The search is
    // quadratic in the width -- a fifty-field record read a fifty-entry object
    // fifty times -- and the plan already holds the name-to-position map the
    // resolution wants. A closed record refuses an undeclared key; the top
    // clause admits it, and so does `str: anything`, since a JSON key is a
    // string. A record whose clause reads a key takes the search below.
    let undeclared = Undeclared::of(defaults);
    let open = matches!(undeclared, Undeclared::Admitted | Undeclared::AnyStr);
    if let Some(plan) = ctx
        .records
        .get(&(fields.as_ptr() as usize))
        .filter(|plan| undeclared != Undeclared::Read && plan.by_name.len() == fields.len())
    {
        // The document's value for each declared field, last occurrence winning
        // as `json.loads` does, gathered before any of them is checked: an
        // earlier duplicate that fails is not the entry the document means.
        let mut found: Vec<Option<&JsonValue<'_>>> = vec![None; fields.len()];
        for (key, value) in entries {
            match plan.by_name.get(key.as_ref()) {
                Some(&at) => match found.get_mut(at) {
                    Some(slot) => *slot = Some(value),
                    // The plan and the field list disagree about a position,
                    // which the filter above rules out; answer conservatively
                    // rather than indexing.
                    None => return false,
                },
                // A closed record has no clause to cover an undeclared key; an
                // open one's top clause covers it by definition.
                None if !open => return false,
                None => {}
            }
        }
        for (field, value) in fields.iter().zip(found) {
            match value {
                Some(value) => {
                    if !member(&field.schema, &Value::Json(py, value), &mut sub) {
                        return false;
                    }
                }
                None if field.required => return false,
                None => {}
            }
        }
        return true;
    }
    for field in fields {
        match entries
            .iter()
            .rev()
            .find(|(key, _)| &*field.name == key.as_ref())
        {
            Some((_, val)) => {
                if !member(&field.schema, &Value::Json(py, val), &mut sub) {
                    return false;
                }
            }
            None if field.required => return false,
            None => {}
        }
    }
    // Every key that is not a declared field must be covered by a default clause,
    // testing each key's last value (json.loads semantics).
    //
    // Whether a key is a declared field is a question about the *schema*, so it
    // is answered from the record plan built once per validator rather than from
    // a name set rebuilt per object. A schema absent from the plan falls back to
    // scanning the field list, so correctness never depends on the plan being
    // complete.
    let plan = ctx.records.get(&(fields.as_ptr() as usize));
    let declares = |name: &str| match plan {
        Some(plan) => plan.by_name.contains_key(name),
        None => fields.iter().any(|f| &*f.name == name),
    };
    // A closed record has no clause to cover an undeclared key with, so the first
    // one decides and there is nothing to collapse.
    if defaults.is_empty() {
        return entries.iter().all(|(key, _)| declares(key.as_ref()));
    }
    undeclared_covered(defaults, py, entries, ctx, declares)
}

/// Whether every key of a parsed object that no field declares is covered by a
/// default clause, testing each key's last value (`json.loads` semantics).
///
/// Split from [`keyed_map_matches_json`] because it is the half with two
/// readings of its own -- which clause question to ask, and where to answer the
/// duplicate-key question -- and the caller is the field walk.
fn undeclared_covered(
    defaults: &[MapClause],
    py: Python<'_>,
    entries: &[(Cow<'_, str>, JsonValue<'_>)],
    ctx: Ctx<'_>,
    declares: impl Fn(&str) -> bool,
) -> bool {
    // A parsed object's keys are strings by construction, so a lone clause whose
    // key schema admits every string governs every undeclared key by its value
    // alone: the key half of the coverage question is already answered, and
    // asking it builds a `JsonValue` and walks a schema per key for a fixed yes.
    // `dict[str, V]` -- the shape a document's free-form section is written as
    // -- is exactly this clause.
    let value_only = match defaults {
        [clause] if matches!(clause.key, Schema::Str | Schema::Anything(_)) => Some(&clause.value),
        _ => None,
    };
    let (mut path, mut out) = (Vec::new(), Vec::new());
    let mut sub = Frame::new(&mut path, &mut out, fast(ctx));
    let mut covers = |key: &str, val: &JsonValue<'_>| {
        if let Some(schema) = value_only {
            return member(schema, &Value::Json(py, val), &mut sub);
        }
        let key_value = JsonValue::Str(Cow::Borrowed(key));
        covered(
            defaults,
            &Value::Json(py, &key_value),
            &Value::Json(py, val),
            ctx,
        )
    };
    // A narrow object is covered where it lies. The entry a document means by a
    // key is its last, and an entry is that one exactly when no entry after it
    // repeats the key -- which is a look forward, not a map of the whole
    // object. The map below answers the same question by allocating a table and
    // hashing every key into it, and a mapping of one or two keys pays that in
    // full: the free-form section of a record is written `dict[str, V]` and
    // usually carries a handful.
    if entries.len() <= SMALL_OBJECT {
        for (position, (key, val)) in entries.iter().enumerate() {
            if declares(key.as_ref())
                || entries
                    .iter()
                    .skip(position + 1)
                    .any(|(later, _)| later.as_ref() == key.as_ref())
            {
                continue;
            }
            if !covers(key.as_ref(), val) {
                return false;
            }
        }
        return true;
    }
    // Collapse the entries to each non-field key's last value in one pass, so a
    // document with many keys (or many duplicates) is covered linearly rather
    // than by rescanning the tail per key.
    let mut last_value: FxHashMap<&str, &JsonValue<'_>> = FxHashMap::default();
    for (key, val) in entries {
        if declares(key.as_ref()) {
            continue;
        }
        last_value.insert(key.as_ref(), val);
    }
    for (key, val) in last_value {
        if !covers(key, val) {
            return false;
        }
    }
    true
}

/// How many entries an object may carry and still be covered where it lies.
///
/// The look forward is quadratic in the entries and the map is linear with a
/// table and a hash per key, so the two cross somewhere -- but the constant on
/// the map is an allocation, and the comparisons the look forward makes are
/// mostly a length test that fails. The bound is set low enough that the
/// crossover is not the question: an object wider than this is a document's
/// payload rather than a record's free-form section, and pays for the table it
/// then uses.
const SMALL_OBJECT: usize = 8;

/// The explain pass over a keyed map, run only after [`keyed_map_matches`] has
/// reported the value is not a member. It walks in declared order — present
/// fields checked in order, then absent required keys — then reports each
/// undeclared key: an uncovered key with no clauses reads as an unexpected key,
/// and with clauses its key and value are checked against the first clause.
pub(super) fn keyed_map_explain(
    fields: &[Field],
    defaults: &[MapClause],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) {
    let ctx = frame.ctx;
    let Value::Py(v) = value else {
        // The explain pass only ever sees a Python value; a JSON value here is
        // unreachable, but keep the false-implies-a-violation invariant.
        frame
            .out
            .push(type_mismatch("dict_type", "dict", value, frame.path));
        return;
    };
    let Ok(dict) = v.cast::<PyDict>() else {
        frame
            .out
            .push(type_mismatch("dict_type", "dict", value, frame.path));
        return;
    };
    // The interned keys, in field order. Asking the dict by Rust text decodes a
    // fresh `PyString` and hashes it before the probe can start, once per field
    // per call; an interned key carries its hash. A schema absent from the index
    // falls back to its own text, so correctness never depends on the plan being
    // complete -- the two spellings name the same key.
    let interned = ctx.records.get(&(fields.as_ptr() as usize));
    let entries = dict.len();
    let mut present = 0usize;
    for (position, field) in fields.iter().enumerate() {
        let key = interned.and_then(|plan| plan.keys.get(position));
        let found = match key {
            Some(interned) => dict.get_item(interned.bind(dict.py())),
            None => dict.get_item(&*field.name),
        };
        match found {
            Ok(Some(item)) => {
                present += 1;
                frame.path.push(PathSegment::Key(Arc::clone(&field.name)));
                member(&field.schema, &Value::Py(&item), frame);
                frame.path.pop();
            }
            Ok(None) if field.required => frame.out.push(located(
                frame.path,
                Arc::clone(&field.name),
                "missing_key",
                format!("required key {:?}", field.name),
                "missing".to_owned(),
            )),
            Ok(None) => {}
            Err(_) => frame
                .out
                .push(type_mismatch("dict_type", "dict", value, frame.path)),
        }
        if ctx.mode.stops_at_first() && !frame.out.is_empty() {
            return;
        }
    }
    // A record that holds exactly the keys it declares has no undeclared key to
    // find, and the field loop above has already established it: a dict cannot
    // repeat a key, so finding as many declared keys as the value has entries
    // accounts for every one of them. Open or closed -- a clause governs the
    // keys a record does not declare, and there are none. The length is
    // re-read because the walk of a field's value runs Python, which can
    // resize the dict -- and a value that moved under the reading falls through
    // to the scan, which is where a mutation is reported.
    if present == entries && dict.len() == entries {
        return;
    }
    // Built here rather than above, because the scan is the only reader and a
    // record that answers by count never reaches it.
    let declared: FxHashSet<&str> = fields.iter().map(|field| &*field.name).collect();
    let scan = scan_dict(dict, |key, val| {
        if let Some(name) = key.cast::<PyString>().ok().and_then(|s| s.to_str().ok())
            && declared.contains(name)
        {
            return ControlFlow::Continue(());
        }
        if covered(defaults, &Value::Py(key), &Value::Py(val), ctx) {
            return ControlFlow::Continue(());
        }
        if let Some(clause) = defaults.first() {
            // A clause exists but did not cover this key: surface the key and
            // value violations against it (the homogeneous-mapping error).
            frame.path.push(key_segment(key));
            member(&clause.key, &Value::Py(key), frame);
            member(&clause.value, &Value::Py(val), frame);
            frame.path.pop();
        } else {
            // A closed record: the key is simply not allowed.
            let key_text = key
                .str()
                .map_or_else(|_| String::new(), |text| text.to_string());
            frame.out.push(located(
                frame.path,
                Arc::from(key_text.as_str()),
                "extra_forbidden",
                "no unexpected key".to_owned(),
                format!("{key_text:?}"),
            ));
        }
        if ctx.mode.stops_at_first() && !frame.out.is_empty() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    // The fast pass reported a non-member; if the dict moved underneath this one
    // there is nothing else to report, and the mutation is the finding.
    if matches!(scan, Scan::Unreadable) {
        mutated(value, frame);
    }
}

pub(super) fn check_attr_record(
    fields: &[Field],
    value: &Value<'_, '_>,
    frame: &mut Frame<'_, '_>,
) -> bool {
    let ctx = frame.ctx;
    // Attributes are read off a Python object, so a value that will not
    // materialize into one carries none and belongs to no record.
    let Ok(obj) = value.to_python() else {
        return false;
    };
    // The interned names, in field order. A schema absent from the index (an
    // incomplete build traversal) falls back to the field's own text, so
    // correctness never depends on the plan being complete.
    let interned = ctx.attrs.get(&(fields.as_ptr() as usize));
    let mut ok = true;
    for (position, field) in fields.iter().enumerate() {
        let name = interned.and_then(|plan| plan.names.get(position));
        let attribute = match name {
            Some(interned) => obj.getattr(interned.bind(value.py())),
            None => obj.getattr(&*field.name),
        };
        match attribute {
            Ok(attr) => {
                if ctx.mode.explains() {
                    frame.path.push(PathSegment::Key(Arc::clone(&field.name)));
                }
                ok &= member(&field.schema, &Value::Py(&attr), frame);
                if ctx.mode.explains() {
                    frame.path.pop();
                }
            }
            // A fatal signal during attribute access is the interpreter
            // unwinding, not a missing attribute: record it and stop.
            Err(err) if is_fatal(&err, value.py()) => {
                record_fatal(err, ctx);
                return false;
            }
            // A field the schema does not require is satisfied by its absence.
            Err(_) if !field.required => {}
            Err(_) => {
                if ctx.mode.explains() {
                    frame.out.push(located(
                        frame.path,
                        Arc::clone(&field.name),
                        "missing_attribute",
                        format!("attribute {:?}", field.name),
                        "missing".to_owned(),
                    ));
                }
                ok = false;
            }
        }
        if !ok && stop(ctx) {
            return false;
        }
    }
    ok
}
