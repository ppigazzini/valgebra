//! The per-validator precompute: record-field lookups, literal-union decision
//! tables, and compiled string patterns, built once and reused across calls.

use pyo3::prelude::*;
use pyo3::types::{PyInt, PyString};
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;
use valgebra_core::{Constraint, Schema};

use crate::input::Value;

pub(crate) struct RecordPlan {
    pub(crate) by_name: FxHashMap<Arc<str>, usize>,
    pub(crate) required: usize,
    /// The interned key of each declared field, in field order.
    ///
    /// The same thing [`AttrsPlan`] holds for `getattr`, for the other lookup a
    /// record does. The explain walk asks the dict for each declared key by
    /// name, and a Rust `&str` handed to `get_item` is decoded into a fresh
    /// `PyString` and hashed before the lookup can start -- per field, per
    /// call. An interned key carries its hash with it, so the lookup is the
    /// probe alone. The accepting walk does not need these: it scans the dict
    /// once and resolves each key it finds through `by_name`.
    pub(crate) keys: Vec<Py<PyString>>,
}

/// The interned attribute names of one [`Schema::AttrRecord`] node, in field
/// order.
///
/// `getattr` takes a name, and passing a Rust `&str` builds a fresh `PyString`
/// for every attribute of every value checked. The names are a property of the
/// schema, so they are built once here and the walk hands the interpreter the
/// same objects each time.
pub(crate) struct AttrsPlan {
    pub(crate) names: Vec<Py<PyString>>,
}

/// The attribute index for a whole validator, keyed and rebuilt like
/// [`RecordIndex`].
pub(crate) type AttrsIndex = FxHashMap<usize, AttrsPlan>;

/// The record index for a whole validator: each record's `fields`-buffer address
/// mapped to its [`RecordPlan`]. The buffer address is stable for the life of the
/// (immutable) schema, and the index is rebuilt per validator from its own
/// schema, so an entry always refers to the same live node.
pub(crate) type RecordIndex = FxHashMap<usize, RecordPlan>;

/// A precomputed decision table for one union whose members are all literals: the
/// `int`-typed literal values that fit a machine integer, and the `str`-typed
/// literal values. An exact `int`/`str` value's membership is then a single set
/// lookup instead of a scan of every branch.
pub(crate) struct UnionPlan {
    ints: FxHashSet<i64>,
    strs: FxHashSet<Box<str>>,
}

impl UnionPlan {
    /// Decide membership of `value` if the value's exact type is one this plan
    /// covers, else `None` to defer to the linear scan. Only an exact `int`
    /// matches `int` literals and only an exact `str` matches `str` literals
    /// (Python's cross-type equality is excluded by the same-type literal rule),
    /// so the set lookup is authoritative for those two types. A boolean, float,
    /// `None`, bytes, big integer, subclass instance, or JSON value returns `None`
    /// and is scanned linearly, matching `literal_matches` exactly.
    pub(crate) fn decide(&self, value: &Value<'_, '_>) -> Option<bool> {
        let Value::Py(v) = value else { return None };
        if v.is_exact_instance_of::<PyInt>() {
            // A big integer (outside i64) cannot equal any i64-valued literal, so
            // deferring to the scan handles it against any big-integer literal.
            let i = v.extract::<i64>().ok()?;
            return Some(self.ints.contains(&i));
        }
        if v.is_exact_instance_of::<PyString>() {
            let s = v.cast::<PyString>().ok()?.to_str().ok()?;
            return Some(self.strs.contains(s));
        }
        None
    }
}

/// The union index for a whole validator: each all-literal union's members-buffer
/// address mapped to its [`UnionPlan`]. Keyed and rebuilt like [`RecordIndex`].
pub(crate) type UnionIndex = FxHashMap<usize, UnionPlan>;

/// Each `Regex(...)` constraint's pattern-buffer address mapped to its compiled,
/// anchored regex, built once per validator so a string-pattern refinement
/// matches natively without recompiling on every call.
///
/// Keyed by address rather than by the pattern text, as the record and union
/// plans are keyed: the buffer is stable for the life of the (immutable) schema
/// and the index is rebuilt per validator from its own schema, so an entry
/// always refers to the same live constraint. The text is a key that must be
/// hashed in full for every value a pattern refinement checks; the address is
/// one integer. Two occurrences of the same pattern compile twice, once, at
/// build time.
pub(crate) type RegexIndex = FxHashMap<usize, Regex>;

/// Anchor a user pattern so the whole string must match (`re.fullmatch`
/// semantics): `\A` and `\z` are absolute string boundaries, and the
/// non-capturing group keeps the user's alternation from escaping them.
pub(crate) fn compile_pattern(pattern: &str) -> Result<Regex, regex::Error> {
    Regex::new(&format!(r"\A(?:{pattern})\z"))
}

/// The per-validator precompute: record-field lookups, literal-union decision
/// tables, and compiled string patterns, all built once from the finished schema
/// and reused across calls.
#[derive(Default)]
pub(crate) struct ValidatorIndex {
    pub(crate) records: RecordIndex,
    pub(crate) attrs: AttrsIndex,
    pub(crate) unions: UnionIndex,
    pub(crate) regexes: RegexIndex,
}

impl ValidatorIndex {
    /// Every Python object this index owns, for the validator's `__traverse__`.
    ///
    /// The interned attribute names and the interned record keys: the other
    /// plans hold Rust values -- integer and string sets, compiled patterns --
    /// copied out of the pool rather than referenced. A `str` joins no cycle,
    /// so nothing here can be the edge a collector needs; a traversal that
    /// skipped an owned reference would still be wrong on its own terms, which
    /// is why the keys join this the moment the record plan starts holding
    /// them.
    pub(crate) fn interned_names(&self) -> impl Iterator<Item = &Py<PyString>> {
        self.attrs
            .values()
            .flat_map(|plan| plan.names.iter())
            .chain(self.records.values().flat_map(|plan| plan.keys.iter()))
    }
}

/// Build the index for a finished schema plus its recursion definitions. `pool`
/// is the validator's constants pool, needed to read each literal's value while
/// building union plans. A record with no declared fields, and a union with a
/// non-literal member, are skipped; the walk falls back to its general path for
/// anything not indexed, so an incomplete traversal only costs speed.
pub(crate) fn build_index(
    py: Python<'_>,
    schema: &Schema,
    defs: &[Schema],
    pool: &[Py<PyAny>],
) -> ValidatorIndex {
    let mut index = ValidatorIndex::default();
    collect(py, schema, pool, &mut index);
    for def in defs {
        collect(py, def, pool, &mut index);
    }
    index
}

fn collect(py: Python<'_>, schema: &Schema, pool: &[Py<PyAny>], index: &mut ValidatorIndex) {
    match schema {
        Schema::KeyedMap { fields, defaults } => {
            if !fields.is_empty() {
                index
                    .records
                    .entry(fields.as_ptr() as usize)
                    .or_insert_with(|| RecordPlan {
                        by_name: fields
                            .iter()
                            .enumerate()
                            .map(|(i, f)| (Arc::clone(&f.name), i))
                            .collect(),
                        required: fields.iter().filter(|f| f.required).count(),
                        keys: fields
                            .iter()
                            .map(|f| PyString::new(py, &f.name).unbind())
                            .collect(),
                    });
            }
            for f in fields.iter() {
                collect(py, &f.schema, pool, index);
            }
            for clause in defaults.iter() {
                collect(py, &clause.key, pool, index);
                collect(py, &clause.value, pool, index);
            }
        }
        Schema::Union(members) => {
            if let Some(plan) = literal_union_plan(py, members, pool) {
                index
                    .unions
                    .entry(members.as_ptr() as usize)
                    .or_insert(plan);
            }
            for member in members.iter() {
                collect(py, member, pool, index);
            }
        }
        Schema::Intersection(members) => {
            for member in members.iter() {
                collect(py, member, pool, index);
            }
        }
        Schema::Coll { element: inner, .. } | Schema::Complement(inner) => {
            collect(py, inner, pool, index);
        }
        Schema::Refine { base, constraints } => {
            collect(py, base, pool, index);
            for constraint in constraints.iter() {
                if let Constraint::Regex(pattern) = constraint
                    && let Ok(compiled) = compile_pattern(pattern)
                {
                    index.regexes.insert(pattern.as_ptr() as usize, compiled);
                }
            }
        }
        Schema::AttrRecord { fields } => {
            if !fields.is_empty() {
                index
                    .attrs
                    .entry(fields.as_ptr() as usize)
                    .or_insert_with(|| AttrsPlan {
                        names: fields
                            .iter()
                            .map(|f| PyString::new(py, &f.name).unbind())
                            .collect(),
                    });
            }
            for f in fields.iter() {
                collect(py, &f.schema, pool, index);
            }
        }
        Schema::Seq { shape, .. } => {
            for element in shape.prefix.iter().chain(shape.tail.as_deref()) {
                collect(py, element, pool, index);
            }
        }
        _ => {}
    }
}

/// Build a [`UnionPlan`] when every union member is a literal, bucketing the
/// `int` and `str` literal values; returns `None` (so the union stays a linear
/// scan) when any member is not a literal. A big-integer or other-typed literal
/// is simply not bucketed — values of those types are scanned linearly.
fn literal_union_plan(py: Python<'_>, members: &[Schema], pool: &[Py<PyAny>]) -> Option<UnionPlan> {
    let mut ints = FxHashSet::default();
    let mut strs = FxHashSet::default();
    for member in members {
        let Schema::Literal(idx) = member else {
            return None;
        };
        // Bounds-check rather than index directly: a corrupt pool index abandons
        // the precompute (no fast path) instead of panicking across the FFI
        // boundary, matching the defensive `.get` posture used elsewhere.
        let constant = pool.get(idx.get()).map(|obj| obj.bind(py))?;
        if constant.is_exact_instance_of::<PyInt>() {
            if let Ok(i) = constant.extract::<i64>() {
                ints.insert(i);
            }
        } else if constant.is_exact_instance_of::<PyString>()
            && let Some(s) = constant
                .cast::<PyString>()
                .ok()
                .and_then(|s| s.to_str().ok())
        {
            strs.insert(s.into());
        }
    }
    Some(UnionPlan { ints, strs })
}

#[cfg(test)]
mod tests;
