//! The deterministic workloads the instruction gate runs, and nothing else.
//!
//! `scripts/perf_gate.py` builds `examples/binding_workload.rs` against an
//! embedded interpreter and counts what one of these shapes executes under
//! cachegrind. They are a measuring instrument: no caller reaches them, the
//! suites do not exercise them, and `#[doc(hidden)]` keeps them out of the
//! extension's documented surface.
//!
//! **They live apart from the crate root for a reason a number tells.** The
//! coverage lane holds the binding to 95% of its lines, measured by the Python
//! suite and the Rust unit tests -- and a workload is run by neither, so every
//! line here counts against a figure it says nothing about: held in a file the
//! lane reads, these shapes are lines nothing covers, so adding one lowers the
//! binding's coverage without a line of the caller's surface changing. The lane
//! skips this file by name, and what it measures is what it claims to: the
//! surface a caller reaches. `.github/workflows/ci.yml` owns the floor.
//!
//! The mutation sweep skips it for the same reason, beside `lib.rs`, which it
//! already skipped.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyModule, PyString};
use valgebra_core::{Constraint, DefIx, Field, MapClause, Schema, SeqKind, SeqShape};

use crate::build::{Pool, build_schema};
use crate::check::{Frame, WalkMode, WalkState, member};
use crate::input::Value;
use crate::validator::Validator;

/// Which deterministic workload to run.
///
/// The comparison gate -- `scripts/compare_gate.py` -- measures seven shapes
/// against pydantic-core on a wall clock; the instruction gate measured one of
/// them. The gap is how a shape regresses without a gate saying so: schema
/// construction grew twelve percent over one release cycle and nothing caught
/// it, because no deterministic workload built a schema.
///
/// Each variant below is the deterministic twin of a comparison shape, so a
/// wall-clock movement can be confirmed or refuted by an instruction count on
/// the same work. They are *not* the same code as the comparison shapes and are
/// not meant to be: the gate compares against another library and has to run
/// what that library can also run, while these run the thing being budgeted.
// A gate's own hook, like the workload it selects: not part of the extension's
// surface, and not in the crate's documentation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingShape {
    /// The membership walk over a homogeneous list of sixty-four integers: the
    /// original workload, and the twin of `large_array`.
    Walk,
    /// The walk over a single integer: the floor every other shape stands on,
    /// and the nearest deterministic twin of `scalar`.
    ///
    /// Not the FFI crossing itself. This runs inside one interpreter attachment
    /// and calls the walk directly, so what it counts is the dispatch and the
    /// type check with no container around them -- the part of `scalar` this
    /// crate owns. The crossing is `PyO3`'s and is measured by the comparison
    /// gate's wall clock, where it belongs.
    Boundary,
    /// The accepting walk over a fifty-field record: the twin of `wide_record`,
    /// and the path that resolves each key through the plan built with the
    /// validator rather than through the walk.
    Record,
    /// Building a fifty-field record validator from its Python spelling: the
    /// twin of `build`.
    ///
    /// The whole of what `Validator({"f0": int, ...})` does after the call
    /// lands: the annotation walk over the dict, the canonical form it imposes
    /// -- fields ordered, clauses deduplicated -- and the validator's own
    /// index. The spelling is built once, outside the loop, so the count is
    /// of reading it and of nothing the harness does to write it. An earlier
    /// form assembled the fields in Rust inside the loop, and three quarters
    /// of what it counted was the harness formatting fifty names and filling
    /// a dict per iteration, with the annotation walk in none of it.
    Build,
    /// The explaining walk over a fifty-field record with one bad field, read
    /// to the end: the twin of `error_report`, and the only shape here that
    /// builds violations rather than answering a bool.
    Explain,
    /// The accepting walk over the same fifty fields declared by a record that
    /// is **open** the way a `TypedDict` is: its clause admits any further
    /// `str` key. The shape the record walk takes for the annotation users
    /// write most, and the one the closed twin above could not see -- an open
    /// record was scanned key by key where a closed one was read by its keys,
    /// and nothing counted the difference.
    Open,
    /// The *accepting* walk over a fifty-field record in explain mode: the path
    /// `validate()` takes when the value is a member.
    ///
    /// The twin of `Explain` on the other answer, and the shape that was
    /// missing. Every other accepting shape runs in fast mode, so the cost of
    /// explain mode on a value that passes -- the mode a caller asking for a
    /// report is in, most of the time -- was measured by nothing. A change to
    /// the failing path that pays for itself there and charges the accepting
    /// one would have read as a pure win.
    ExplainAccept,
    /// Compiling a fifty-field `TypedDict` of `Annotated[int, Ge(0)]`: the
    /// frontend's *other* work, and the half [`Build`](BindingShape::Build)
    /// never reaches.
    ///
    /// `Build` reads a dict of plain type objects, which asks the frontend for
    /// none of what an annotation with any depth to it costs: a class read
    /// through `get_type_hints`, a field asked whether a qualifier states its
    /// required-ness, a marker read for the bound names it might carry. Every
    /// one of those was measured by nothing, which is how a frontend that
    /// raised an `AttributeError` per absent attribute went three releases
    /// unnoticed.
    Annotated,
    /// The fifty-field record walked with a value whose keys are **interned**:
    /// a dict written as a literal, `**kwargs`, an object's `__dict__`.
    ///
    /// The twin of [`Record`](BindingShape::Record), which walks the same
    /// schema over a dict built with `format!`. A dict probe compares by
    /// pointer before hash and bytes, so which of the two a caller brings
    /// decides whether fifty field lookups are fifty pointer comparisons or
    /// fifty hashes and memcmps -- 29% of the call. The pointer path needs
    /// *both* sides interned, so this shape is the one that can see the
    /// validator's own side stop being.
    Keys,
    /// Compiling a fifty-field **dataclass**: the one class form whose build
    /// asks a question of the standard library.
    ///
    /// The two build shapes beside it read a `dict` and a `TypedDict`, and
    /// neither reaches a class at all. A dataclass does: the frontend asks
    /// `dataclasses.is_dataclass` about it, reads its fields through
    /// `get_type_hints`, and reads the class's own declared order. Nothing
    /// counted any of that -- which is how putting the `dataclasses` import
    /// back at the top of the frontend cost a build that compiles *no*
    /// dataclass 6.45%, and had to be found with a profiler because no shape
    /// here would move.
    Object,
    /// Walking a `tuple` **subclass** that overrides nothing: a `NamedTuple`,
    /// against `tuple[int, ...]`.
    ///
    /// The walk cannot trust the C length accessor for a subclass, because
    /// `cpyext` answers it through the object's own `__len__`; it can trust the
    /// accessor for a subclass that *inherits* the base's slot, which is every
    /// `NamedTuple`. Telling the two apart by "is this exactly a tuple" instead
    /// copied every `NamedTuple` and cost 100 ns against a tuple's 57.
    ///
    /// That regression was invisible here: no shape walked a subclass, and the
    /// repair's own mutant is *answer*-equivalent -- reading every subclass as
    /// a liar decides the same things and only costs more -- so no test can
    /// hold it either. A count is the only instrument that can, which is what
    /// this shape is: the cheapest thing that would have caught it.
    Subclass,
    /// A JSON document of two hundred records, parsed and walked in one pass:
    /// the twin of `json_document`, and the shape that had none.
    ///
    /// It is the comparison gate's closest race -- 0.70 of pydantic-core's
    /// time against a ceiling of 1.00 -- and the tooling page records what a
    /// wide margin costs: this ratio drifted from 0.78 to 0.87 with no gate
    /// red, because a wall clock under a ceiling it clears by a third measures
    /// nothing until somebody looks. A count has no ceiling to hide under.
    ///
    /// What it counts is not all this crate's. Profiled at `9700181`, one call
    /// is about 1.41M instructions: 55% `jiter` building the value tree, 14%
    /// dropping it, and 31% the walk. `pydantic-core` parses to the same
    /// `jiter` tree before it validates, so the two-thirds is the floor both
    /// libraries stand on and the third is what the ratio is actually about.
    /// A regression in the walk is a sixth of what it would be in a shape
    /// measuring the walk alone -- which is the reason to count this rather
    /// than to infer it from [`Record`](BindingShape::Record), whose dict the
    /// interpreter hands over already built.
    Json,
    /// A string checked against a **pattern** refinement: the one refinement
    /// whose cost is a compiled object rather than a comparison.
    ///
    /// `check/index.rs` compiles the pattern once, when the validator's index
    /// is built, and the walk finds it by the pattern's address. Losing that
    /// precompute is not a wrong answer -- `check/walk/scalar.rs` compiles the
    /// pattern on the spot and decides the same thing -- so no test can hold
    /// it, and the index's mutation survivor for the arm that fills it is
    /// accepted on exactly that argument. This count is what makes accepting
    /// it honest: with the arm deleted, every validation compiles the regex
    /// instead of matching against one, and nothing else in the tree would
    /// say so.
    Pattern,
    /// A nested tree walked against a **recursive** schema: `mu X. int | list[X]`
    /// over a list nested eight deep.
    ///
    /// The one shape that enters a reference. Every other shape here walks a
    /// schema with no `Ref` in it, so the trail the walk keeps to refuse a
    /// cyclic value -- entered and left once per level, per value -- is reached
    /// by none of them, and a change to how it is held reads as free. The
    /// recursive schema is also the one a caller writes for a JSON document or
    /// a syntax tree, which is the shape most likely to carry real depth.
    Recursive,
    /// The JSON document with one record's `id` a string, halfway through: the
    /// *rejecting* path of `is_valid_json`, where the walk stops at the first
    /// field that fails and every shape beside it accepts.
    JsonReject,
    /// A document of records of two kinds against `list[JsonRecord |
    /// JsonEvent]`, alternating, so half the elements fail the first branch and
    /// are asked again: the union over parsed objects, which no accepting record
    /// shape reaches.
    JsonUnion,
    /// A document of records read through a typed catch-all, `{"id": int, str:
    /// str}`: every key but one is covered by the clause rather than named by a
    /// field, which is the reading an open record takes in place.
    JsonOpen,
    /// A recursive schema, `mu X. int | list[X]`, over a parsed document of
    /// twenty trees twelve levels deep: the reference and its trail on the JSON
    /// path, which the recursive shape measures only on Python values.
    JsonDeep,
    /// `list[list[...[int]...]]` twenty-five levels deep over a value nested as
    /// deep: the deterministic twin of the comparison gate's `deep_nesting`, a
    /// descent with no reference in it.
    Deep,
    /// Sixty-four integers against `list[Annotated[int, Ge(1000)]]`: a walk over
    /// refined values, where every element reads its bound. Every other walk
    /// shape checks a kind and nothing more.
    Refined,
}

impl BindingShape {
    /// The name the gate passes on the command line.
    #[must_use]
    pub fn named(name: &str) -> Option<BindingShape> {
        Some(match name {
            "walk" => BindingShape::Walk,
            "boundary" => BindingShape::Boundary,
            "record" => BindingShape::Record,
            "build" => BindingShape::Build,
            "explain" => BindingShape::Explain,
            "open" => BindingShape::Open,
            "explain-accept" => BindingShape::ExplainAccept,
            "annotated" => BindingShape::Annotated,
            "keys" => BindingShape::Keys,
            "object" => BindingShape::Object,
            "subclass" => BindingShape::Subclass,
            "json" => BindingShape::Json,
            "pattern" => BindingShape::Pattern,
            "recursive" => BindingShape::Recursive,
            "json-reject" => BindingShape::JsonReject,
            "json-union" => BindingShape::JsonUnion,
            "json-open" => BindingShape::JsonOpen,
            "json-deep" => BindingShape::JsonDeep,
            "deep" => BindingShape::Deep,
            "refined" => BindingShape::Refined,
            _ => return None,
        })
    }
}

/// The fifty-field record both record shapes use, as a schema and as a value.
///
/// Fifty fields is the comparison gate's width, kept identical so the two
/// measurements are of the same size of problem.
///
/// The *failing position* is not identical, and deliberately stays that way.
/// The explain shape below writes the wrong value at `f37` where the comparison
/// gate writes it at `f7` -- and the **name is not the position**, which is the
/// thing to read twice. [`Schema::keyed_map`] sorts fields by name, and `f0`
/// through `f49` sort as strings, so the walk visits
///
/// ```text
/// f0 f1 f10 f11 ... f19 f2 f20 ... f29 f3 f30 ... f39 f4 f40 ... f49 f5 f6 f7 f8 f9
/// ```
///
/// which puts `f37` at position **31** and `f7` at position **47**. The fast
/// pass stops at the first field that fails and the explain pass walks them
/// all, so this shape probes *sixteen fewer* fields than the gate's, not thirty
/// more. Read the 15% below accordingly: failing **later** is what costs more,
/// and the profile agrees.
///
/// Aligning them was tried and reverted: the count moves 15% and
/// `perf_gate.py --against` rebuilds the *base* to compare, so a workload whose
/// shape changed is measured against a different workload and reads as a
/// regression it is not. A shape is part of a workload's identity, and
/// re-recording its budget does not make two shapes one. What the two
/// instruments share is the size of the problem; that is what the sentence
/// above claims and all it claims.
fn wide_record(py: Python<'_>) -> (Schema, Py<PyAny>) {
    let (fields, value) = wide_fields(py);
    (Schema::keyed_map(fields, Vec::new()), value)
}

/// The same fifty fields and value under the clause a `TypedDict` carries.
fn open_record(py: Python<'_>) -> (Schema, Py<PyAny>) {
    let (fields, value) = wide_fields(py);
    let any_str_key = MapClause {
        key: Schema::Str,
        value: Schema::ANYTHING,
    };
    (Schema::keyed_map(fields, vec![any_str_key]), value)
}

/// The fifty-field record as the comparison gate spells it to `Validator`:
/// `{"f0": int, ..., "f49": int}`.
fn wide_spelling(py: Python<'_>) -> Bound<'_, PyDict> {
    let int = py.get_type::<PyInt>();
    let spelling = PyDict::new(py);
    for i in 0..50 {
        spelling
            .set_item(format!("f{i}"), &int)
            .expect("a fresh dict of fifty type objects always builds");
    }
    spelling
}

/// The fifty-field record's value with **interned** keys, which is what a dict
/// written as a literal or handed over as `**kwargs` carries.
fn wide_interned_value(py: Python<'_>) -> Py<PyAny> {
    let value = PyDict::new(py);
    for i in 0..50 {
        value
            .set_item(PyString::intern(py, &format!("f{i}")), i)
            .expect("a fresh dict of small ints always builds");
    }
    value.into_any().unbind()
}

/// The fifty-field `TypedDict` of refined integers the annotated build shape
/// compiles, built once.
///
/// The marker is defined here rather than imported from `annotated_types`: the
/// frontend reads a marker by its *shape* -- an object carrying `ge` is a lower
/// bound whoever wrote it -- so a workload that names no third-party package is
/// one this lane can run with whatever it has installed, and it compiles the
/// same constraint either way.
fn annotated_record(py: Python<'_>) -> Py<PyAny> {
    let module = PyModule::from_code(
        py,
        &std::ffi::CString::new(
            "from typing import Annotated, NotRequired, TypedDict\n\
             class Ge:\n\
             \x20   def __init__(self, ge):\n\
             \x20       self.ge = ge\n\
             \x20   def __repr__(self):\n\
             \x20       return f'Ge({self.ge})'\n\
             fields = {f'f{i}': Annotated[int, Ge(0)] for i in range(45)}\n\
             fields.update({f'o{i}': NotRequired[Annotated[int, Ge(0)]] for i in range(5)})\n\
             SPELLING = TypedDict('Wide', fields)\n",
        )
        .expect("no interior nul"),
        &std::ffi::CString::new("annotated.py").expect("no interior nul"),
        &std::ffi::CString::new("annotated").expect("no interior nul"),
    )
    .expect("the spelling compiles");
    module
        .getattr("SPELLING")
        .expect("the spelling is defined")
        .unbind()
}

/// The fifty-field dataclass the object build shape compiles, built once.
///
/// Made with `make_dataclass` rather than written out: fifty `@dataclass`
/// fields are fifty lines that say one thing, and the class the frontend reads
/// is the same either way -- `__dataclass_fields__`, annotations, and the
/// declared order.
fn object_record(py: Python<'_>) -> Py<PyAny> {
    let module = PyModule::from_code(
        py,
        &std::ffi::CString::new(
            "from dataclasses import make_dataclass\n\
             SPELLING = make_dataclass('Wide', [(f'f{i}', int) for i in range(50)])\n",
        )
        .expect("no interior nul"),
        &std::ffi::CString::new("object_shape.py").expect("no interior nul"),
        &std::ffi::CString::new("object_shape").expect("no interior nul"),
    )
    .expect("the spelling compiles");
    module
        .getattr("SPELLING")
        .expect("the spelling is defined")
        .unbind()
}

/// The three-field `NamedTuple` the subclass walk reads, built once.
///
/// A `NamedTuple` rather than a bare `tuple` subclass because it is the one a
/// program is most likely to hold, and the two take the same path: both inherit
/// `tuple.__len__`.
fn subclass_value(py: Python<'_>) -> Py<PyAny> {
    let module = PyModule::from_code(
        py,
        &std::ffi::CString::new(
            "from typing import NamedTuple\n\
             class Point(NamedTuple):\n\
             \x20   x: int\n\
             \x20   y: int\n\
             \x20   z: int\n\
             VALUE = Point(1, 2, 3)\n",
        )
        .expect("no interior nul"),
        &std::ffi::CString::new("subclass.py").expect("no interior nul"),
        &std::ffi::CString::new("subclass").expect("no interior nul"),
    )
    .expect("the spelling compiles");
    module
        .getattr("VALUE")
        .expect("the value is defined")
        .unbind()
}

/// The comparison gate's JSON document and the annotation it is read against,
/// built once: `list[JsonRecord]` over two hundred records.
///
/// Spelled in Python and compiled through the frontend rather than assembled as
/// `Schema` nodes here, for the same reason the document is `json.dumps`ed
/// rather than written out: the two gates have to measure one problem, and the
/// only way to be sure of that is to build it from the same source. Five fields
/// of four shapes -- an integer, two strings, a list of strings and a mapping
/// of strings -- reach the scalar arms, the sequence arm and the clause arm in
/// one record, which is what makes a document of them a walk rather than a
/// parse with a type check on the end.
///
/// Returns the annotation and the document as bytes. `is_valid_json` takes
/// `str` or `bytes` and decodes the first, so bytes is the form that measures
/// the parse and not the decode.
fn json_document(py: Python<'_>, source: &str) -> (Py<PyAny>, Vec<u8>) {
    let module = spelled_module(py, source);
    let document = module
        .getattr("DOCUMENT")
        .expect("the document is defined")
        .extract()
        .expect("the document is bytes");
    let spelling = module
        .getattr("SPELLING")
        .expect("the spelling is defined")
        .unbind();
    (spelling, document)
}

/// Run a shape's Python source as a module of its own.
///
/// The embedded interpreter has the standard library and nothing else, so a
/// shape spells its annotation with `typing` and a marker class it defines.
fn spelled_module<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyModule> {
    PyModule::from_code(
        py,
        &std::ffi::CString::new(source).expect("no interior nul"),
        &std::ffi::CString::new("shape.py").expect("no interior nul"),
        &std::ffi::CString::new("shape").expect("no interior nul"),
    )
    .expect("the spelling compiles")
}

/// A JSON shape's source: the record every one is written around, then its own
/// spelling and document.
macro_rules! json_shape {
    ($shape:literal) => {
        concat!(
            "import json\n\
     from typing import TypedDict\n\
     JsonRecord = TypedDict('JsonRecord', {'id': int, 'name': str, \
     'email': str, 'tags': list[str], 'meta': dict[str, str]})\n\
     def record(i):\n\
     \x20   return {'id': i, 'name': 'Ada', 'email': 'a@b.c', 'tags': ['x', 'y'], \
     'meta': {'k': 'v'}}\n",
            $shape
        )
    };
}

/// Two hundred records, every one a member.
const JSON_DOCUMENT: &str = json_shape!(
    "SPELLING = list[JsonRecord]\n\
     DOCUMENT = json.dumps([record(i) for i in range(200)]).encode()\n"
);

/// The same two hundred, the hundredth with a string for its `id`.
const JSON_REJECT: &str = json_shape!(
    "SPELLING = list[JsonRecord]\n\
     records = [record(i) for i in range(200)]\n\
     records[100]['id'] = 'x'\n\
     DOCUMENT = json.dumps(records).encode()\n"
);

/// Records and events alternating, against the union of the two.
const JSON_UNION: &str = json_shape!(
    "JsonEvent = TypedDict('JsonEvent', {'kind': str, 'at': int, 'tags': list[str]})\n\
     SPELLING = list[JsonRecord | JsonEvent]\n\
     DOCUMENT = json.dumps([record(i) if i % 2 else \
     {'kind': 'login', 'at': i, 'tags': ['x']} for i in range(200)]).encode()\n"
);

/// Records whose every key but `id` is read through the `str` clause.
const JSON_OPEN: &str = json_shape!(
    "SPELLING = list[{'id': int, str: str}]\n\
     DOCUMENT = json.dumps([{'id': i, 'name': 'Ada', 'email': 'a@b.c', \
     'city': 'Rome', 'lang': 'en'} for i in range(200)]).encode()\n"
);

fn wide_fields(py: Python<'_>) -> (Vec<Field>, Py<PyAny>) {
    let fields: Vec<Field> = (0..50)
        .map(|i| Field {
            name: format!("f{i}").into(),
            schema: Schema::Int,
            required: true,
        })
        .collect();
    let value = PyDict::new(py);
    for i in 0..50 {
        value
            .set_item(format!("f{i}"), i)
            .expect("a fresh dict of small ints always builds");
    }
    (fields, value.into_any().unbind())
}

/// A deterministic, binding-level instruction workload for the perf gate.
///
/// The shipped hot path is the membership walk over a live Python value, where
/// the core's deterministic core-only workload does not reach. This runs that walk
/// `iters` times over a fixed record value crossing the boundary at every node
/// kind on the hot path (an int, a str, and a homogeneous int list), returning a
/// checksum so the optimizer cannot discard the work.
///
/// Embedding `CPython` makes the absolute instruction count include a non-fixed
/// interpreter startup, so the gate ([`scripts/perf_gate.py`]) measures the
/// *difference* between two iteration counts: startup is identical in both runs
/// and cancels, leaving the deterministic per-iteration walk cost. This is the
/// budgeted signal, and it also covers the per-node `ctx.fatal.borrow()` tax.
#[doc(hidden)]
#[must_use]
pub fn binding_perf_workload_shape(py: Python<'_>, shape: BindingShape, iters: usize) -> u64 {
    match shape {
        BindingShape::Walk => binding_perf_workload(py, iters),
        BindingShape::Boundary => boundary_walk(py, iters),
        BindingShape::Record | BindingShape::Open | BindingShape::Keys | BindingShape::Subclass => {
            let (schema, value) = match shape {
                BindingShape::Record => wide_record(py),
                BindingShape::Keys => (wide_record(py).0, wide_interned_value(py)),
                BindingShape::Subclass => (
                    Schema::tuple(SeqShape::homogeneous(Schema::Int)),
                    subclass_value(py),
                ),
                _ => open_record(py),
            };
            let validator = Validator::new(schema, Vec::new(), Vec::new());
            let obj = value.bind(py).clone();
            settle_the_heap(py);
            let mut checksum: u64 = 0;
            for _ in 0..iters {
                let state = WalkState::new();
                let ok = member(
                    std::hint::black_box(&validator.schema),
                    &Value::Py(std::hint::black_box(&obj)),
                    &mut Frame::new(
                        &mut Vec::new(),
                        &mut Vec::new(),
                        validator.context(py, &state, WalkMode::Fast),
                    ),
                );
                checksum = checksum.wrapping_add(u64::from(ok));
            }
            checksum
        }
        BindingShape::Build => {
            let spelling = wide_spelling(py);
            settle_the_heap(py);
            let mut checksum: u64 = 0;
            for _ in 0..iters {
                let mut literals = Pool::default();
                let mut definitions = Vec::new();
                let schema = build_schema(
                    std::hint::black_box(&spelling).as_any(),
                    &mut literals,
                    &mut definitions,
                )
                .expect("fifty fields typed `int` always build");
                let validator = Validator::checked(schema, literals.into_items(), definitions)
                    .expect("a fifty-field record is within every limit");
                checksum = checksum.wrapping_add(validator.schema.node_count() as u64);
            }
            checksum
        }
        BindingShape::Recursive => recursive_walk(py, iters),
        BindingShape::Json => json_walk(py, iters, JSON_DOCUMENT, Expect::Member),
        BindingShape::JsonReject => json_walk(py, iters, JSON_REJECT, Expect::NonMember),
        BindingShape::JsonUnion => json_walk(py, iters, JSON_UNION, Expect::Member),
        BindingShape::JsonOpen => json_walk(py, iters, JSON_OPEN, Expect::Member),
        BindingShape::JsonDeep => json_deep_walk(py, iters),
        BindingShape::Deep => spelled_walk(py, iters, DEEP),
        BindingShape::Refined => spelled_walk(py, iters, REFINED),
        BindingShape::Pattern => pattern_walk(py, iters),
        BindingShape::ExplainAccept => explaining_record(py, iters, Wrong::No),
        BindingShape::Explain => explaining_record(py, iters, Wrong::Yes),
        BindingShape::Annotated | BindingShape::Object => {
            let spelling = if matches!(shape, BindingShape::Annotated) {
                annotated_record(py)
            } else {
                object_record(py)
            };
            settle_the_heap(py);
            let mut checksum: u64 = 0;
            for _ in 0..iters {
                let mut literals = Pool::default();
                let mut definitions = Vec::new();
                let schema = build_schema(
                    std::hint::black_box(spelling.bind(py)),
                    &mut literals,
                    &mut definitions,
                )
                .expect("fifty typed fields always build");
                let validator = Validator::checked(schema, literals.into_items(), definitions)
                    .expect("a fifty-field record is within every limit");
                checksum = checksum.wrapping_add(validator.schema.node_count() as u64);
            }
            checksum
        }
    }
}

/// The walk over a single integer, one validator call per iteration.
fn boundary_walk(py: Python<'_>, iters: usize) -> u64 {
    let validator = Validator::new(Schema::Int, Vec::new(), Vec::new());
    let obj = 42_i64
        .into_pyobject(py)
        .expect("an i64 always converts")
        .into_any();
    settle_the_heap(py);
    let mut checksum: u64 = 0;
    for _ in 0..iters {
        let state = WalkState::new();
        let ok = member(
            std::hint::black_box(&validator.schema),
            &Value::Py(std::hint::black_box(&obj)),
            &mut Frame::new(
                &mut Vec::new(),
                &mut Vec::new(),
                validator.context(py, &state, WalkMode::Fast),
            ),
        );
        checksum = checksum.wrapping_add(u64::from(ok));
    }
    checksum
}

/// Hand the allocator's free memory back before a shape starts counting.
///
/// The count is taken of the loop, but what a loop's allocations cost depends
/// on the heap the loop starts from, and nothing in the loop decides that:
/// whether glibc serves a large request straight from the top chunk or first
/// consolidates every small chunk the previous iteration freed is a fact about
/// where the long-lived blocks landed while the shape was being set up. The
/// size of the process environment alone moves it. Measured on one build, the
/// JSON shape reads 8.2% apart and the recursive one 7.2% apart between a
/// `uv run` and a bare shell, and a change to the *build* path -- which runs
/// before the loop and never inside it -- moved the JSON reading by the same
/// amount while its loop executed the same instructions.
///
/// `malloc_trim(0)` consolidates the free lists and returns the top of the
/// heap, so every shape counts from the same settled state whatever ran before
/// it. Called through `ctypes` because the crate forbids `unsafe`; a C library
/// without the symbol leaves the heap as it is, which is every platform the
/// gate does not run on.
fn settle_the_heap(py: Python<'_>) {
    py.run(
        c"import ctypes\n\
          try:\n\
          \x20   ctypes.CDLL(None).malloc_trim(0)\n\
          except (AttributeError, OSError):\n\
          \x20   pass\n",
        None,
        None,
    )
    .expect("trimming the heap raises nothing it does not catch");
}

/// The JSON document parsed and walked once per iteration.
///
/// The document and the validator are built outside the loop, as every shape
/// here builds what it reads: what is counted is one `matches_json`, which is
/// the parse and the walk and the drop of the tree between them. That is the
/// whole of what `is_valid_json` does after the argument is known to be bytes,
/// and it is the entry the comparison gate times.
/// What a JSON shape's document is, so the checksum counts the answer expected.
#[derive(Clone, Copy)]
enum Expect {
    Member,
    NonMember,
}

fn json_walk(py: Python<'_>, iters: usize, source: &str, expect: Expect) -> u64 {
    let (spelling, document) = json_document(py, source);
    let mut literals = Pool::default();
    let mut definitions = Vec::new();
    let schema = build_schema(spelling.bind(py), &mut literals, &mut definitions)
        .expect("a JSON shape's spelling always builds");
    let validator = Validator::checked(schema, literals.into_items(), definitions)
        .expect("a JSON shape's schema is within every limit");
    json_loop(py, iters, &validator, &document, expect)
}

/// Check one document `iters` times, counting the answers that were expected.
///
/// The count is the iteration count by construction, which is what the gate
/// asserts: a rejecting shape counts its rejections.
fn json_loop(
    py: Python<'_>,
    iters: usize,
    validator: &Validator,
    document: &[u8],
    expect: Expect,
) -> u64 {
    settle_the_heap(py);
    let mut checksum: u64 = 0;
    for _ in 0..iters {
        let ok = validator
            .matches_json(py, std::hint::black_box(document))
            .expect("a well-formed document raises nothing");
        checksum = checksum.wrapping_add(u64::from(ok == matches!(expect, Expect::Member)));
    }
    checksum
}

/// `mu X. int | list[X]`, built as the recursive shape builds it, over a parsed
/// document of twenty trees twelve levels deep.
fn json_deep_walk(py: Python<'_>, iters: usize) -> u64 {
    let body = Schema::union([
        Schema::Int,
        Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(0)))),
    ]);
    let validator = Validator::checked(Schema::Ref(DefIx::new(0)), Vec::new(), vec![body])
        .expect("one definition is within every limit");
    let document = spelled_module(
        py,
        "import json\n\
         def tree(depth):\n\
         \x20   return 1 if depth == 0 else [tree(depth - 1), depth]\n\
         DOCUMENT = json.dumps([tree(12) for _ in range(20)]).encode()\n",
    )
    .getattr("DOCUMENT")
    .expect("the document is defined")
    .extract::<Vec<u8>>()
    .expect("the document is bytes");
    json_loop(py, iters, &validator, &document, Expect::Member)
}

/// `list[...]` nested twenty-five deep, over a value nested as deep.
const DEEP: &str = "SPELLING = int\n\
     VALUE = 0\n\
     for _ in range(25):\n\
     \x20   SPELLING = list[SPELLING]\n\
     \x20   VALUE = [VALUE]\n";

/// Sixty-four integers at or above their refinement's lower bound.
const REFINED: &str = "from typing import Annotated\n\
     class Ge:\n\
     \x20   def __init__(self, ge):\n\
     \x20       self.ge = ge\n\
     SPELLING = list[Annotated[int, Ge(1000)]]\n\
     VALUE = list(range(1000, 1064))\n";

/// Build a shape's `SPELLING` through the frontend and walk its `VALUE`.
fn spelled_walk(py: Python<'_>, iters: usize, source: &str) -> u64 {
    let module = spelled_module(py, source);
    let mut literals = Pool::default();
    let mut definitions = Vec::new();
    let spelling = module.getattr("SPELLING").expect("the spelling is defined");
    let schema = build_schema(&spelling, &mut literals, &mut definitions)
        .expect("a walk shape's spelling always builds");
    let validator = Validator::checked(schema, literals.into_items(), definitions)
        .expect("a walk shape's schema is within every limit");
    let value = module.getattr("VALUE").expect("the value is defined");
    settle_the_heap(py);
    let mut checksum: u64 = 0;
    for _ in 0..iters {
        let state = WalkState::new();
        let ok = member(
            std::hint::black_box(&validator.schema),
            &Value::Py(std::hint::black_box(&value)),
            &mut Frame::new(
                &mut Vec::new(),
                &mut Vec::new(),
                validator.context(py, &state, WalkMode::Fast),
            ),
        );
        checksum = checksum.wrapping_add(u64::from(ok));
    }
    checksum
}

/// The recursive schema's own shape: `mu X. int | list[X]` over a nested list.
///
/// Eight levels of nesting, each a one-element list, so the walk enters and
/// leaves the reference eight times per iteration and the leaf is an integer.
/// A one-element list at each level rather than a wide one keeps what is
/// counted the *descent* -- the reference, the trail, and the union choice --
/// rather than a list loop the homogeneous shapes already measure.
///
/// The value is built once, outside the loop, like every shape here.
fn recursive_walk(py: Python<'_>, iters: usize) -> u64 {
    // mu X. int | list[X], as the frontend's own `recursive` spelling lowers it:
    // one definition, and a reference standing for the whole.
    let body = Schema::union([
        Schema::Int,
        Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(0)))),
    ]);
    let validator = Validator::checked(Schema::Ref(DefIx::new(0)), Vec::new(), vec![body])
        .expect("one definition is within every limit");
    let mut value = 42_i64
        .into_pyobject(py)
        .expect("an i64 always converts")
        .into_any();
    for _ in 0..8 {
        value = PyList::new(py, [value])
            .expect("a one-element list always builds")
            .into_any();
    }
    settle_the_heap(py);
    let mut checksum: u64 = 0;
    for _ in 0..iters {
        let state = WalkState::new();
        let ok = member(
            std::hint::black_box(&validator.schema),
            &Value::Py(std::hint::black_box(&value)),
            &mut Frame::new(
                &mut Vec::new(),
                &mut Vec::new(),
                validator.context(py, &state, WalkMode::Fast),
            ),
        );
        checksum = checksum.wrapping_add(u64::from(ok));
    }
    checksum
}

/// The pattern refinement's own shape: one string, one compiled match.
///
/// An address-like pattern rather than a trivial one, because what is being
/// counted is a *compiled* match against a lookup, and a pattern the engine
/// answers in three instructions would leave the lookup indistinguishable from
/// the compile. The value matches, so the match runs to the end rather than
/// failing at the first character.
fn pattern_walk(py: Python<'_>, iters: usize) -> u64 {
    let schema = Schema::refine(
        Schema::Str,
        vec![Constraint::Regex(
            r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}".to_owned(),
        )],
    );
    let validator = Validator::new(schema, Vec::new(), Vec::new());
    let obj = PyString::new(py, "ada.lovelace@example.com").into_any();
    settle_the_heap(py);
    let mut checksum: u64 = 0;
    for _ in 0..iters {
        let state = WalkState::new();
        let ok = member(
            std::hint::black_box(&validator.schema),
            &Value::Py(std::hint::black_box(&obj)),
            &mut Frame::new(
                &mut Vec::new(),
                &mut Vec::new(),
                validator.context(py, &state, WalkMode::Fast),
            ),
        );
        checksum = checksum.wrapping_add(u64::from(ok));
    }
    checksum
}

/// Whether the explaining record shape's value carries the wrong type.
///
/// The two shapes differ in one value and in nothing else, which is the point:
/// what separates them is the *answer*, so the pair measures what explain mode
/// costs on each.
#[derive(Clone, Copy)]
enum Wrong {
    /// `f37` holds a string, so the record is refused and the report is built.
    Yes,
    /// Every field holds an integer, so the walk accepts and reports nothing.
    No,
}

/// The fifty-field record walked in explain mode, accepting or refusing.
fn explaining_record(py: Python<'_>, iters: usize, wrong: Wrong) -> u64 {
    let (schema, value) = wide_record(py);
    let validator = Validator::new(schema, Vec::new(), Vec::new());
    let obj = value.bind(py).clone();
    if matches!(wrong, Wrong::Yes) {
        obj.cast::<PyDict>()
            .expect("the record value is a dict")
            .set_item("f37", "not an int")
            .expect("replacing one key always succeeds");
    }
    settle_the_heap(py);
    let mut checksum: u64 = 0;
    for _ in 0..iters {
        let state = WalkState::new();
        let mut out = Vec::new();
        let ok = member(
            std::hint::black_box(&validator.schema),
            &Value::Py(std::hint::black_box(&obj)),
            &mut Frame::new(
                &mut Vec::new(),
                &mut out,
                validator.context(py, &state, WalkMode::Explain),
            ),
        );
        checksum = checksum
            .wrapping_add(u64::from(ok))
            .wrapping_add(out.len() as u64);
    }
    checksum
}

#[doc(hidden)]
#[must_use]
pub fn binding_perf_workload(py: Python<'_>, iters: usize) -> u64 {
    // A homogeneous int list: the walk crosses the boundary at the container and
    // at each element (an `isinstance` check and the per-node `ctx.fatal.borrow()`
    // tax), the most common shape on the hot path.
    let schema = Schema::Seq {
        container: SeqKind::List,
        shape: SeqShape::homogeneous(Schema::Int),
    };
    let validator = Validator::new(schema, Vec::new(), Vec::new());

    // A fixed matching value, built once; the walk visits each list element.
    let items: Vec<i64> = (0..64).collect();
    let obj = PyList::new(py, items)
        .expect("a fresh list of i64 always builds")
        .into_any();

    settle_the_heap(py);

    let mut checksum: u64 = 0;
    for _ in 0..iters {
        let state = WalkState::new();
        // `black_box` the inputs so the optimizer cannot hoist the loop-invariant
        // walk out of the loop: the per-iteration walk is the signal being timed.
        let ok = member(
            std::hint::black_box(&validator.schema),
            &Value::Py(std::hint::black_box(&obj)),
            &mut Frame::new(
                &mut Vec::new(),
                &mut Vec::new(),
                validator.context(py, &state, WalkMode::Fast),
            ),
        );
        checksum = checksum.wrapping_add(u64::from(ok));
    }
    checksum
}
