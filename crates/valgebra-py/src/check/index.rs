//! The per-validator precompute: record-field lookups, literal-union decision
//! tables, and compiled string patterns, built once and reused across calls.

use pyo3::prelude::*;
use pyo3::types::{PyInt, PyString};
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use valgebra_core::{Constraint, Schema, SeqRegex};

use crate::input::Value;

pub(crate) struct RecordPlan {
    pub(crate) by_name: FxHashMap<Box<str>, usize>,
    pub(crate) required: usize,
}

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

/// Each `Regex(...)` constraint's source pattern mapped to its compiled,
/// anchored regex, built once per validator so a string-pattern refinement
/// matches natively without recompiling on every call.
pub(crate) type RegexIndex = FxHashMap<String, Regex>;

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
    pub(crate) unions: UnionIndex,
    pub(crate) regexes: RegexIndex,
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
                            .map(|(i, f)| (f.name.as_str().into(), i))
                            .collect(),
                        required: fields.iter().filter(|f| f.required).count(),
                    });
            }
            for f in fields {
                collect(py, &f.schema, pool, index);
            }
            for (key_schema, value_schema) in defaults {
                collect(py, key_schema, pool, index);
                collect(py, value_schema, pool, index);
            }
        }
        Schema::Union(members) => {
            if let Some(plan) = literal_union_plan(py, members, pool) {
                index
                    .unions
                    .entry(members.as_ptr() as usize)
                    .or_insert(plan);
            }
            for member in members {
                collect(py, member, pool, index);
            }
        }
        Schema::Intersection(members) => {
            for member in members {
                collect(py, member, pool, index);
            }
        }
        Schema::Set(inner) | Schema::FrozenSet(inner) | Schema::Complement(inner) => {
            collect(py, inner, pool, index);
        }
        Schema::Refine { base, constraints } => {
            collect(py, base, pool, index);
            for constraint in constraints {
                if let Constraint::Regex(pattern) = constraint
                    && !index.regexes.contains_key(pattern)
                    && let Ok(compiled) = compile_pattern(pattern)
                {
                    index.regexes.insert(pattern.clone(), compiled);
                }
            }
        }
        Schema::Attrs { fields, .. } => {
            for f in fields {
                collect(py, &f.schema, pool, index);
            }
        }
        Schema::Seq { regex, .. } => collect_seq(py, regex, pool, index),
        _ => {}
    }
}

fn collect_seq(py: Python<'_>, regex: &SeqRegex, pool: &[Py<PyAny>], index: &mut ValidatorIndex) {
    match regex {
        SeqRegex::Empty => {}
        SeqRegex::Elem(schema) => collect(py, schema, pool, index),
        SeqRegex::Cat(parts) | SeqRegex::Or(parts) => {
            for part in parts {
                collect_seq(py, part, pool, index);
            }
        }
        SeqRegex::Star(inner) => collect_seq(py, inner, pool, index),
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
        let constant = pool[*idx].bind(py);
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
mod tests {
    use super::*;

    #[test]
    fn compile_pattern_anchors_the_whole_string() {
        // Anchoring gives `re.fullmatch` semantics: the whole string must match,
        // and the non-capturing wrap keeps an alternation from escaping it.
        let alternation = compile_pattern("a|b").expect("valid pattern");
        assert!(alternation.is_match("a"));
        assert!(alternation.is_match("b"));
        assert!(!alternation.is_match("xa"));
        assert!(!alternation.is_match("ab"));
        let digits = compile_pattern("[0-9]+").expect("valid pattern");
        assert!(digits.is_match("123"));
        assert!(!digits.is_match("12a"));
        assert!(!digits.is_match(""));
    }

    #[test]
    fn compile_pattern_rejects_an_invalid_pattern() {
        assert!(compile_pattern("(unclosed").is_err());
    }

    // Tests that need a live interpreter; compiled and run only under the
    // `interpreter-tests` feature, which links an embedded Python.
    #[cfg(feature = "interpreter-tests")]
    mod interpreter {
        use super::super::*;
        use crate::check::walk::literal_matches;
        use pyo3::types::{PyBool, PyString};

        #[test]
        fn union_plan_decide_agrees_with_the_linear_scan() {
            // The literal-union fast path must never disagree with the linear scan
            // it replaces: whenever `decide` commits to an answer, that answer
            // equals membership decided by `literal_matches` over the same pooled
            // literals.
            Python::attach(|py| {
                let pool: Vec<Py<PyAny>> = vec![
                    1i64.into_pyobject(py).unwrap().into_any().unbind(),
                    7i64.into_pyobject(py).unwrap().into_any().unbind(),
                    PyString::new(py, "ok").into_any().unbind(),
                    PyString::new(py, "yes").into_any().unbind(),
                ];
                let members: Vec<Schema> = (0..pool.len()).map(Schema::Literal).collect();
                let plan =
                    literal_union_plan(py, &members, &pool).expect("every member is a literal");

                // Edge types the plan must defer on (returning `None`): a big
                // integer outside i64, a bool (exact-type rule), a float, and a NaN.
                let candidates: Vec<Bound<'_, PyAny>> = vec![
                    1i64.into_pyobject(py).unwrap().into_any(),
                    7i64.into_pyobject(py).unwrap().into_any(),
                    42i64.into_pyobject(py).unwrap().into_any(),
                    (1i128 << 70).into_pyobject(py).unwrap().into_any(),
                    PyBool::new(py, true).to_owned().into_any(),
                    1.0f64.into_pyobject(py).unwrap().into_any(),
                    f64::NAN.into_pyobject(py).unwrap().into_any(),
                    PyString::new(py, "ok").into_any(),
                    PyString::new(py, "no").into_any(),
                ];

                let mut committed = 0;
                for value in &candidates {
                    if let Some(decided) = plan.decide(&Value::Py(value)) {
                        committed += 1;
                        let scan = pool
                            .iter()
                            .any(|lit| literal_matches(value, lit.bind(py)).unwrap_or(false));
                        assert_eq!(decided, scan, "fast-path decision disagreed with the scan");
                    }
                }
                // The exact int and str values commit, so the check is not vacuous.
                assert!(
                    committed >= 4,
                    "expected the fast path to commit on exact values"
                );
            });
        }
    }
}
