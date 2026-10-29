//! The per-validator precompute: record-field lookups, literal-union decision
//! tables, and compiled string patterns, built once and reused across calls.

use pyo3::prelude::*;
use pyo3::types::{PyInt, PyString};
use regex_automata::util::syntax;
use regex_automata::{MatchKind, meta};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use valgebra_core::{Constraint, Schema};

use super::memo::{self, MemoPlan};
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
    /// probe alone. The deciding walk asks by these keys too wherever they
    /// settle the record (`keyed_map_asks_for_its_keys`), and scans the dict
    /// once, resolving each key it finds through `by_name`, where a clause reads
    /// an undeclared key together with its value or a probe raised.
    ///
    /// **Interned rather than merely built, and that is most of what the key
    /// is for.** A dict probe compares the key it is given with the key it
    /// holds by *pointer* first and only then by hash and contents, so two
    /// interned keys settle in one comparison and two equal strings that are
    /// not the same object walk their bytes. Every dict Python writes as a
    /// literal, every `**kwargs`, and every `__dict__` carries interned keys,
    /// so interning this side is what lets the probe take its fast path:
    /// measured on the fifty-field record walk, the two sides interned read
    /// **29.1% cheaper** than neither, and interning one side alone buys about
    /// a percent. Interning costs a hash and a probe of the interpreter's own
    /// table, once per field per validator.
    pub(crate) keys: Vec<Py<PyString>>,
    /// The fields the deciding walk reads before the declared order: each one
    /// decided by its own value and declared after one that is not, in
    /// declared order.
    ///
    /// Fields are stored sorted by name, so a tag such as `type` beside a
    /// recursive `left` is read after the child, and a union of records told
    /// apart by the tag walked the whole child in every branch it refused:
    /// branches to the power of the depth. Read first, a tag refuses a wrong
    /// branch in one comparison whatever it is called. Empty for every record
    /// whose declared order already reads them first, which is most of them,
    /// and then the deciding walk is the declared loop alone.
    pub(crate) early: Box<[usize]>,
}

/// Whether a field is decided by its own value, without a walk below it: a
/// scalar kind, a literal, or a union of them.
pub(crate) fn decided_by_its_value(schema: &Schema) -> bool {
    match schema {
        Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Str
        | Schema::Bytes
        | Schema::Literal(_) => true,
        Schema::Union(members) => members.iter().all(decided_by_its_value),
        _ => false,
    }
}

/// The fields a record's deciding walk reads first ([`RecordPlan::early`]).
fn early_fields(fields: &[valgebra_core::Field]) -> Box<[usize]> {
    let decided = |at: &usize| {
        fields
            .get(*at)
            .is_some_and(|f| decided_by_its_value(&f.schema))
    };
    (0..fields.len())
        .skip_while(decided)
        .filter(decided)
        .collect()
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
///
/// And the addresses of the pooled constants behind those values, so the
/// constant itself -- an interned string, a cached small integer, which is what
/// a program spelling the literal in its own source holds -- is found by its
/// address, without decoding or hashing its text. The pool holds each constant
/// for as long as the validator lives, so no other object can have its address
/// meanwhile, and an exact `int` or `str` is equal to itself: an address found
/// here is a value the set beside it holds.
pub(crate) struct UnionPlan {
    ints: FxHashSet<i64>,
    strs: FxHashSet<Box<str>>,
    pooled: FxHashSet<usize>,
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
            if self.pooled.contains(&(v.as_ptr() as usize)) {
                return Some(true);
            }
            // A big integer (outside i64) cannot equal any i64-valued literal, so
            // deferring to the scan handles it against any big-integer literal.
            let i = v.extract::<i64>().ok()?;
            return Some(self.ints.contains(&i));
        }
        if v.is_exact_instance_of::<PyString>() {
            if self.pooled.contains(&(v.as_ptr() as usize)) {
                return Some(true);
            }
            let s = v.cast::<PyString>().ok()?.to_str().ok()?;
            return Some(self.strs.contains(s));
        }
        None
    }
}

/// The union index for a whole validator: each all-literal union's members-buffer
/// address mapped to its [`UnionPlan`]. Keyed and rebuilt like [`RecordIndex`].
pub(crate) type UnionIndex = FxHashMap<usize, UnionPlan>;

/// A union of records that each require one key, pinned to `int` or `str`
/// constants no two records share: the one record a dict can be a member of is
/// the one its value at that key names, so the deciding walk reads the key once
/// and walks that record, where the scan reads it once a record. The design
/// reference is `pydantic-core`'s tagged union, a lookup and then one branch.
///
/// Every record's constants are in the tables, so a tag of an exact `int` or
/// `str` that no table holds is no record's. Any other tag is the scan's.
pub(crate) struct TaggedPlan {
    /// The key every record requires, interned.
    pub(crate) key: Py<PyString>,
    ints: FxHashMap<i64, usize>,
    strs: FxHashMap<Box<str>, usize>,
}

/// What a tag says of a tagged union's members.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tagged {
    /// The member at this position, the one record whose constants hold it.
    Branch(usize),
    /// No record's constants hold it, so no member admits the dict.
    Nothing,
}

impl Tagged {
    fn of(position: Option<&usize>) -> Tagged {
        position.map_or(Tagged::Nothing, |position| Tagged::Branch(*position))
    }
}

impl TaggedPlan {
    /// What `tag` says of the union's members, or `None` where it is no exact
    /// `int` or `str`, which the scan reads.
    pub(crate) fn branch(&self, tag: &Bound<'_, PyAny>) -> Option<Tagged> {
        if tag.is_exact_instance_of::<PyInt>() {
            // A big integer equals no constant the table holds, and a record
            // pinning one has no plan, so it is the scan's.
            return Some(Tagged::of(self.ints.get(&tag.extract::<i64>().ok()?)));
        }
        if tag.is_exact_instance_of::<PyString>() {
            let text = tag.cast::<PyString>().ok()?.to_str().ok()?;
            return Some(Tagged::of(self.strs.get(text)));
        }
        None
    }
}

/// Each tagged union's members-buffer address mapped to its [`TaggedPlan`].
pub(crate) type TaggedIndex = FxHashMap<usize, TaggedPlan>;

/// A compiled pattern: the engine `regex` builds a `Regex` on, built as it
/// builds one ([`compile_pattern`]), so it matches the same language and can
/// say what memory it keeps.
pub(crate) type Regex = meta::Regex;

/// Each `Regex(...)` constraint's pattern-buffer address mapped to its compiled,
/// anchored regex, built once per validator so a string-pattern refinement
/// matches natively without recompiling on every call.
///
/// Keyed by address rather than by the pattern text, as the record and union
/// plans are keyed: the buffer is stable for the life of the (immutable) schema
/// and the index is rebuilt per validator from its own schema, so an entry
/// always refers to the same live constraint. The text is a key that must be
/// hashed in full for every value a pattern refinement checks; the address is
/// one integer. Two occurrences of the same text share one compiled program
/// ([`compile_patterns`]).
pub(crate) type RegexIndex = FxHashMap<usize, Regex>;

/// Anchor a user pattern so the whole string must match (`re.fullmatch`
/// semantics): `\A` and `\z` are absolute string boundaries, and the
/// non-capturing group keeps the user's alternation from escaping them.
///
/// In extended mode a `#` runs to the end of the line, so a pattern that ends
/// inside a comment swallows the wrapper's own `)\z` and the anchored form
/// fails to parse while the pattern alone is fine. A newline ends the comment,
/// and in extended mode a newline is whitespace that denotes nothing -- so the
/// second wrapper is the same language as the first wherever the first parses,
/// and is reached only where it does not. Outside extended mode nothing is
/// line-terminated, so a pattern the first wrapper rejects the second rejects
/// too, and the caller gets the first wrapper's message either way.
///
/// The pattern is parsed on its own first. One that does not parse alone can
/// parse once wrapped -- `a)|(b` closes the wrapper's group and opens one the
/// wrapper closes -- and its alternation then escapes both anchors, so the
/// wrappers would compile a language the caller never wrote.
///
/// Built with the configuration `regex::RegexBuilder` hands the engine -- the
/// leftmost-first match, the UTF-8 syntax, and the lazy DFA's cache at the 2 MiB
/// the engine defaults to -- and its errors read as `regex`'s, so the language
/// and the refusal are the ones `regex` gives.
pub(crate) fn compile_pattern(pattern: &str) -> Result<Regex, regex::Error> {
    regex_syntax::Parser::new()
        .parse(pattern)
        .map_err(|err| regex::Error::Syntax(err.to_string()))?;
    let build = |anchored: &str| {
        meta::Builder::new()
            .configure(
                meta::Config::new()
                    .nfa_size_limit(Some(PATTERN_SIZE_LIMIT))
                    .match_kind(MatchKind::LeftmostFirst)
                    .utf8_empty(true),
            )
            .syntax(syntax::Config::new().utf8(true))
            .build(anchored)
            .map_err(|err| match (err.size_limit(), err.syntax_error()) {
                (Some(limit), _) => regex::Error::CompiledTooBig(limit),
                (None, Some(syntax)) => regex::Error::Syntax(syntax.to_string()),
                (None, None) => regex::Error::Syntax(err.to_string()),
            })
    };
    build(&format!(r"\A(?:{pattern})\z"))
        .or_else(|anchored| build(&format!("\\A(?:{pattern}\n)\\z")).map_err(|_| anchored))
}

/// A validator's patterns, each text compiled once, within [`PATTERN_SIZE_LIMIT`]
/// together.
///
/// A validator keeps the program of every pattern it holds for as long as it
/// lives, so the bound on what one pattern may keep is a bound on one of them
/// only: a hundred patterns of 16 MiB each held a gigabyte, and the allocation
/// past the memory there is aborts the interpreter rather than raising. Read
/// together, the memory each compiled pattern keeps is charged to one
/// allowance, and the pattern that would pass it is refused where the schema is
/// built, with the patterns compiled before it. A text met twice is one
/// program, charged once.
///
/// What it returns is the table the walk's index takes, so a validator's
/// patterns compile once, when it is built.
pub(crate) fn compile_patterns(
    schema: &Schema,
    definitions: &[Schema],
) -> Result<RegexIndex, String> {
    let mut compiled = RegexIndex::default();
    let mut by_text: FxHashMap<&str, Regex> = FxHashMap::default();
    let mut left = PATTERN_SIZE_LIMIT;
    for tree in std::iter::once(schema).chain(definitions) {
        for pattern in tree.patterns() {
            let regex = if let Some(regex) = by_text.get(pattern) {
                regex.clone()
            } else {
                let regex = compile_pattern(pattern)
                    .map_err(|err| format!("invalid regular expression {pattern:?}: {err}"))?;
                left = left.checked_sub(regex.memory_usage()).ok_or_else(|| {
                    format!(
                        "the patterns of one schema compile to at most {} MiB together, \
                         and {pattern:?} takes them past it: a validator keeps every \
                         pattern's program while it lives",
                        PATTERN_SIZE_LIMIT >> 20
                    )
                })?;
                by_text.insert(pattern, regex.clone());
                regex
            };
            compiled.insert(pattern.as_ptr() as usize, regex);
        }
    }
    Ok(compiled)
}

thread_local! {
    /// The patterns a walk met outside any index, each compiled under its text,
    /// a refusal among them, for the query that walked it.
    static UNINDEXED: RefCell<FxHashMap<Box<str>, Option<Regex>>> = RefCell::default();
}

/// Whether `text` is in the language of `pattern`, a pattern no index holds.
///
/// Only the relation oracle walks a schema no validator built an index for,
/// and it walks every constant of a table through one supertype: compiled per
/// value, a table against the alternation of its own words compiles the
/// pattern once a constant, quadratic in the table. So a pattern compiles once
/// a query, under its text -- a schema the query builds holds a copy of the
/// text at an address of its own, which no address key would find again -- and
/// [`forget_unindexed`] drops what the query compiled when its oracle goes.
///
/// Out of line and cold: a validator's own walk never reaches it, and inlined
/// into the pattern check every refined value reads, the table and the
/// thread-local around it lay out the walk's hot path anew.
#[cold]
#[inline(never)]
pub(crate) fn matches_unindexed(pattern: &str, text: &str) -> bool {
    UNINDEXED.with_borrow_mut(|compiled| {
        if let Some(regex) = compiled.get(pattern) {
            return regex.as_ref().is_some_and(|regex| regex.is_match(text));
        }
        let regex = compile_pattern(pattern).ok();
        let matched = regex.as_ref().is_some_and(|regex| regex.is_match(text));
        compiled.insert(pattern.into(), regex);
        matched
    })
}

/// Drop the patterns [`matches_unindexed`] compiled on this thread: the query
/// that walked them is over, and nothing outside one misses an index.
pub(crate) fn forget_unindexed() {
    UNINDEXED.take();
}

/// How many patterns [`matches_unindexed`] holds on this thread.
#[cfg(test)]
pub(crate) fn unindexed_held() -> usize {
    UNINDEXED.with_borrow(FxHashMap::len)
}

/// The memory a validator's patterns may cost: 32 MiB, three times the `regex`
/// crate's default program size, read twice.
///
/// The program one pattern compiles to is bounded at it as it is built, and the
/// patterns of one validator are bounded at it together, by what each keeps once
/// built ([`compile_patterns`]). The default refuses a counted Unicode class
/// written as an ordinary bound. The class holds some 700 ranges and a count
/// copies it once a repetition, so `\w{1,255}` keeps 14 MiB and
/// `\w{1,255}@\w{1,255}` 28, each built in 150 to 400 ms; `re` takes both. Past
/// this, `\w{1,1000}` is refused as before, and so is a third `\w{1,255}` beside
/// two. The ASCII class of the same counts, `(?-u)\w{1,255}`, keeps 40 KiB.
pub(crate) const PATTERN_SIZE_LIMIT: usize = 32 << 20;

/// The per-validator precompute: record-field lookups, literal-union decision
/// tables, and compiled string patterns, all built once from the finished schema
/// and reused across calls.
#[derive(Default)]
pub(crate) struct ValidatorIndex {
    /// Per-record declared-field lookups, keyed by the address of each record's
    /// `fields` buffer. The keyed-map fast path reads it instead of rebuilding
    /// the name map on every call; a node absent from it falls back to building
    /// the map.
    pub(crate) records: RecordIndex,
    /// Per-attribute-schema interned names. The attribute walk hands these to
    /// `getattr` instead of building a fresh `PyString` per attribute per value.
    pub(crate) attrs: AttrsIndex,
    /// Per-union value sets for unions whose members are all literals, keyed by
    /// the address of the union's members buffer. It answers only for an exact
    /// int or str value: the deciding walk takes its answer either way instead
    /// of scanning every branch, and the explaining walk takes only its yes, for
    /// the elements of a list of the union, and walks every other element. Any
    /// other case (a non-literal union, a value of another type, a JSON value)
    /// falls back to the linear scan.
    pub(crate) unions: UnionIndex,
    /// Per-union dispatch for unions of records pinned by one key to constants
    /// no two share ([`TaggedPlan`]), keyed as `unions` is. The deciding walk
    /// reads it and the explaining walk scans, so a report is the scan's.
    pub(crate) tagged: TaggedIndex,
    /// Compiled string patterns, keyed by the address of the pattern's own
    /// text in the schema; the refinement walk reads it for a `Regex(...)`
    /// constraint instead of recompiling.
    pub(crate) regexes: RegexIndex,
    /// Which definitions a walk may remember at a value, and which nodes arm
    /// the memo ([`MemoPlan`]). Empty for a schema with no definitions.
    pub(crate) memo: MemoPlan,
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
            .chain(self.tagged.values().map(|plan| &plan.key))
    }
}

/// Build the index for a finished schema plus its recursion definitions. `pool`
/// is the validator's constants pool, needed to read each literal's value while
/// building union plans. A record with no declared fields, and a union with a
/// non-literal member, are skipped; the walk falls back to its general path for
/// anything not indexed, which answers the same membership. A report is the
/// exception: a record with no plan is read by its entries rather than its
/// keys, and which branch of a union of records is explained follows that
/// reading, so the traversal descends into every node a record can sit under.
pub(crate) fn build_index(
    py: Python<'_>,
    schema: &Schema,
    defs: &[Schema],
    pool: &[Py<PyAny>],
) -> ValidatorIndex {
    build_index_from(py, schema, defs, pool, &RegexIndex::default())
}

/// [`build_index`] taking the patterns a validator compiled when it was built
/// ([`compile_patterns`]), so none compiles twice; a pattern the table does
/// not hold is compiled here.
pub(crate) fn build_index_from(
    py: Python<'_>,
    schema: &Schema,
    defs: &[Schema],
    pool: &[Py<PyAny>],
    compiled: &RegexIndex,
) -> ValidatorIndex {
    let mut index = ValidatorIndex::default();
    index.regexes.clone_from(compiled);
    collect(py, schema, pool, &mut index);
    for def in defs {
        collect(py, def, pool, &mut index);
    }
    index.memo = memo::plan(py, schema, defs, pool);
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
                            .map(|f| PyString::intern(py, &f.name).unbind())
                            .collect(),
                        early: early_fields(fields),
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
            } else if let Some(plan) = tagged_union_plan(py, members, pool) {
                index
                    .tagged
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
                    && let Entry::Vacant(slot) = index.regexes.entry(pattern.as_ptr() as usize)
                    && let Ok(compiled) = compile_pattern(pattern)
                {
                    slot.insert(compiled);
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
                            .map(|f| PyString::intern(py, &f.name).unbind())
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
/// is not bucketed — values of those types are scanned linearly.
fn literal_union_plan(py: Python<'_>, members: &[Schema], pool: &[Py<PyAny>]) -> Option<UnionPlan> {
    let mut ints = FxHashSet::default();
    let mut strs = FxHashSet::default();
    let mut pooled = FxHashSet::default();
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
                // Sized on its first member for the whole table, so a wide
                // table is one allocation rather than a rehash per doubling.
                if ints.is_empty() {
                    ints.reserve(members.len());
                }
                ints.insert(i);
                pooled.insert(constant.as_ptr() as usize);
            }
        } else if constant.is_exact_instance_of::<PyString>()
            && let Some(s) = constant
                .cast::<PyString>()
                .ok()
                .and_then(|s| s.to_str().ok())
        {
            if strs.is_empty() {
                strs.reserve(members.len());
            }
            strs.insert(s.into());
            pooled.insert(constant.as_ptr() as usize);
        }
    }
    Some(UnionPlan { ints, strs, pooled })
}

/// The fewest records a tagged union holds before its plan is built.
///
/// Below it, reading the tag and then the record it names costs more than
/// reading the records in turn, each of which reads the tag first and refuses
/// on it: the record the plan names reads the tag again. Measured on the
/// release build, averaged over the tags a value can carry, two records read in
/// turn take 0.115 us and dispatched 0.128 us, three 0.128 and 0.125, four
/// 0.144 and 0.125, and a thousand 48 and 0.14.
pub(crate) const TAGGED_MIN_RECORDS: usize = 3;

/// Build a [`TaggedPlan`] where at least [`TAGGED_MIN_RECORDS`] members are
/// records requiring a key the first record requires, each pinning it to exact
/// `int` or `str` constants and no two sharing one: the first such key of the
/// first record. `None` where no key is, and the union stays a scan.
fn tagged_union_plan(py: Python<'_>, members: &[Schema], pool: &[Py<PyAny>]) -> Option<TaggedPlan> {
    let Some(Schema::KeyedMap { fields: first, .. }) = members.first() else {
        return None;
    };
    if members.len() < TAGGED_MIN_RECORDS {
        return None;
    }
    first
        .iter()
        .filter(|candidate| candidate.required)
        .find_map(|candidate| {
            let mut ints = FxHashMap::default();
            let mut strs = FxHashMap::default();
            for (position, member) in members.iter().enumerate() {
                let Schema::KeyedMap { fields, .. } = member else {
                    return None;
                };
                let field = fields
                    .iter()
                    .find(|field| field.required && field.name == candidate.name)?;
                for constant in pinned_constants(&field.schema)? {
                    let constant = pool.get(constant)?.bind(py);
                    let other = if constant.is_exact_instance_of::<PyInt>() {
                        ints.insert(constant.extract::<i64>().ok()?, position)
                    } else if constant.is_exact_instance_of::<PyString>() {
                        let text = constant.cast::<PyString>().ok()?.to_str().ok()?;
                        strs.insert(text.into(), position)
                    } else {
                        return None;
                    };
                    if other.is_some_and(|other| other != position) {
                        return None;
                    }
                }
            }
            Some(TaggedPlan {
                key: PyString::intern(py, &candidate.name).unbind(),
                ints,
                strs,
            })
        })
}

/// The pool positions of the constants a field is pinned to: one literal, or
/// a union of nothing but literals.
fn pinned_constants(schema: &Schema) -> Option<Vec<usize>> {
    match schema {
        Schema::Literal(index) => Some(vec![index.get()]),
        Schema::Union(members) => members
            .iter()
            .map(|member| match member {
                Schema::Literal(index) => Some(index.get()),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
