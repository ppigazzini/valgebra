//! The readings chosen for their cost, each held to the values it is chosen
//! for. Every one answers as the reading beside it does, which the walk's
//! corpus holds, so what these read is the note each leaves
//! ([`taken`](super::taken)): a value sent to the other reading keeps every
//! answer and fails here.

use std::sync::Arc;

use jiter::JsonValue;
use pyo3::types::{PyDict, PyModule};
use valgebra_core::SeqShape;

use super::record::{RecordReading, SMALL_OBJECT, keyed_map_matches_json};
use super::sequence::SNAPSHOT_BAND;
use super::taken::{Reading, counted};
use super::*;
use crate::check::index::ValidatorIndex;
use crate::check::{WalkState, build_index};

/// How many times one walk of `value` in `mode` took `reading`, and its answer.
fn taken(
    reading: Reading,
    (schema, defs, pool): (&Schema, &[Schema], &[Py<PyAny>]),
    index: &ValidatorIndex,
    value: &Bound<'_, PyAny>,
    mode: WalkMode,
) -> (bool, usize) {
    let state = WalkState::new();
    let ctx = Ctx {
        pool,
        defs,
        index,
        state: &state,
        mode,
    };
    counted(reading, || {
        member(
            schema,
            &Value::Py(value),
            &mut Frame::new(&mut Vec::new(), &mut Vec::new(), ctx),
        )
    })
}

/// A module of the caller's naming, holding `source`.
fn module<'py>(py: Python<'py>, name: &str, source: &str) -> Bound<'py, PyModule> {
    PyModule::from_code(
        py,
        &std::ffi::CString::new(source).expect("no nul"),
        &std::ffi::CString::new(format!("{name}.py")).expect("no nul"),
        &std::ffi::CString::new(name).expect("no nul"),
    )
    .expect("the module compiles")
}

/// `source` evaluated, in `module` where there is one.
fn evaluated<'py>(
    py: Python<'py>,
    module: Option<&Bound<'py, PyModule>>,
    source: &str,
) -> Bound<'py, PyAny> {
    py.eval(
        &std::ffi::CString::new(source).expect("no nul"),
        module.map(PyModuleMethods::dict).as_ref(),
        None,
    )
    .expect("the value evaluates")
}

/// The values `schema` refuses.
fn not(schema: Schema) -> Schema {
    Schema::Complement(Arc::new(schema))
}

/// A list is read through a snapshot exactly where its length lies in the band
/// the interpreter this is built against compiles, both ends included: by the
/// loop over a scalar kind, by the reader of a class's instances, and by the
/// explaining reader, at each end of the band and one past it.
#[test]
fn a_list_is_snapshotted_from_one_end_of_the_band_to_the_other() {
    Python::attach(|py| {
        let module = module(py, "snapshotted_lists", "class P:\n    pass\n");
        let pool = vec![evaluated(py, Some(&module), "P").unbind()];
        let ints = Schema::list(SeqShape::homogeneous(Schema::Int));
        let points = Schema::list(SeqShape::homogeneous(Schema::Instance(ClassIx::new(0))));
        let (least, most) = SNAPSHOT_BAND;
        let lengths = [
            least.checked_sub(1),
            Some(least),
            Some(most),
            most.checked_add(1),
        ];
        for len in lengths.into_iter().flatten() {
            let snapshotted = usize::from((least..=most).contains(&len));
            let make = |element: &str| {
                evaluated(
                    py,
                    Some(&module),
                    &format!("[{element} for _ in range({len})]"),
                )
            };
            for (schema, list, modes) in [
                (&ints, make("1"), &[WalkMode::Fast, WalkMode::Explain][..]),
                (&points, make("P()"), &[WalkMode::Fast][..]),
            ] {
                let index = build_index(py, schema, &[], &pool);
                for &mode in modes {
                    let walk = (schema, &[][..], &pool[..]);
                    let read = taken(Reading::Snapshot, walk, &index, &list, mode);
                    assert_eq!(
                        read,
                        (true, snapshotted),
                        "{schema:?} of {len} in {mode:?}, the band {least}..={most}"
                    );
                }
            }
        }
    });
}

/// A mapping of `str` keys to `int` values is read by the loop over those two
/// type tests, which the instruction gate's mapping shape measures, admitted
/// or refused; a mapping of other kinds by the loop over any two.
#[test]
fn a_mapping_of_str_to_int_takes_the_loop_written_for_it() {
    Python::attach(|py| {
        let mapping = |key, value| Schema::mapping(MapClause { key, value });
        for (schema, source, admitted, written) in [
            (
                mapping(Schema::Str, Schema::Int),
                "{'a': 1, 'b': 2}",
                true,
                1,
            ),
            (
                mapping(Schema::Str, Schema::Int),
                "{'a': 1, 'b': 'x'}",
                false,
                1,
            ),
            (mapping(Schema::Str, Schema::Float), "{'a': 1.5}", true, 0),
            (mapping(Schema::Int, Schema::Int), "{1: 2}", true, 0),
        ] {
            let index = build_index(py, &schema, &[], &[]);
            let value = evaluated(py, None, source);
            assert!(value.is_instance_of::<PyDict>());
            let walk = (&schema, &[][..], &[][..]);
            assert_eq!(
                taken(Reading::StrIntMap, walk, &index, &value, WalkMode::Fast),
                (admitted, written),
                "{schema:?} at {source}"
            );
        }
    });
}

/// A parsed object is read through its record's plan, except by a record
/// whose clause reads keys and declares no more fields than a small object
/// holds, which searches the object for each field instead.
#[test]
fn a_record_reading_its_keys_searches_a_small_parsed_object() {
    Python::attach(|py| {
        let reads_keys = vec![MapClause {
            key: Schema::Str,
            value: Schema::Str,
        }];
        for (width, defaults, planned) in [
            (1, reads_keys.clone(), 0),
            (SMALL_OBJECT, reads_keys.clone(), 0),
            (SMALL_OBJECT + 1, reads_keys, 1),
            (1, Vec::new(), 1),
        ] {
            let fields = (0..width)
                .map(|at| Field {
                    name: format!("f{at}").into(),
                    schema: Schema::Int,
                    required: at == 0,
                })
                .collect();
            let schema = Schema::keyed_map(fields, defaults);
            let Schema::KeyedMap { fields, defaults } = &schema else {
                unreachable!("a keyed map")
            };
            let index = build_index(py, &schema, &[], &[]);
            let state = WalkState::new();
            let ctx = Ctx {
                pool: &[],
                defs: &[],
                index: &index,
                state: &state,
                mode: WalkMode::Fast,
            };
            let reading = RecordReading::of(fields, defaults, &ctx);
            let object = [("f0".into(), JsonValue::Int(1))];
            assert_eq!(
                counted(Reading::PlannedRecord, || {
                    keyed_map_matches_json(&reading, py, &object, ctx)
                }),
                (true, planned),
                "{width} fields beside {} clauses",
                defaults.len()
            );
        }
    });
}

/// An explained record that holds exactly the keys it declares has no
/// undeclared key to report, which the count of the fields it found shows, and
/// is explained without the scan for them; one holding a key more is scanned.
#[test]
fn a_record_holding_exactly_its_keys_is_explained_without_a_scan() {
    Python::attach(|py| {
        let field = |name: &str| Field {
            name: name.into(),
            schema: Schema::Int,
            required: true,
        };
        let schema = Schema::keyed_map(vec![field("a"), field("b")], Vec::new());
        let index = build_index(py, &schema, &[], &[]);
        for (source, scanned) in [("{'a': 1, 'b': 'x'}", 0), ("{'a': 1, 'b': 'x', 'c': 3}", 1)] {
            let value = evaluated(py, None, source);
            let walk = (&schema, &[][..], &[][..]);
            assert_eq!(
                taken(
                    Reading::UndeclaredScan,
                    walk,
                    &index,
                    &value,
                    WalkMode::Explain
                ),
                (false, scanned),
                "{source}"
            );
        }
    });
}

/// The list arm hands a union, a meet, a complement and a reference to the
/// readers of the index's readings only where the index holds one: with none,
/// no such reader is asked, and with one, each is.
#[test]
fn a_list_asks_the_readers_of_readings_only_where_the_index_holds_one() {
    Python::attach(|py| {
        let tree = Schema::Ref(DefIx::new(0));
        let defs = [Schema::Union(
            vec![
                Schema::Int,
                Schema::Str,
                Schema::list(SeqShape::homogeneous(tree.clone())),
            ]
            .into(),
        )];
        let union = Schema::Union(vec![Schema::NoneType, not(Schema::Str)].into());
        let meet = Schema::Intersection(vec![not(Schema::Bool), not(Schema::Str)].into());
        let complement = not(Schema::Union(vec![Schema::Int, Schema::Str].into()));
        for (element, source, reader) in [
            (union, "[None, 1]", Reading::LeafList),
            (meet, "[1, 0.5]", Reading::LeafList),
            (complement, "[None, 0.5]", Reading::LeafList),
            (tree, "[1, [2, 'x']]", Reading::TreeList),
        ] {
            let schema = Schema::list(SeqShape::homogeneous(element));
            let value = evaluated(py, None, source);
            let read = build_index(py, &schema, &defs, &[]);
            let mut walked = build_index(py, &schema, &defs, &[]);
            walked.forget_readings();
            assert!(read.has_readings() && !walked.has_readings());
            let walk = (&schema, &defs[..], &[][..]);
            let (admitted, asked) = taken(reader, walk, &read, &value, WalkMode::Fast);
            assert!(
                admitted && asked > 0,
                "{schema:?}: {reader:?} asked {asked}"
            );
            assert_eq!(
                taken(reader, walk, &walked, &value, WalkMode::Fast),
                (true, 0),
                "{schema:?} with no reading"
            );
        }
    });
}

/// A deciding walk of a list of leaves writes no location, the elements it
/// hands the walk included: the buffers [`Unwritten`] holds stay empty and own
/// nothing, which is what lets it forget them.
#[test]
fn a_deciding_walk_of_a_list_of_leaves_writes_no_location() {
    Python::attach(|py| {
        let schema = Schema::list(SeqShape::homogeneous(not(Schema::Bytes)));
        let index = build_index(py, &schema, &[], &[]);
        assert!(index.has_readings());
        let module = module(py, "leaves_left_to_the_walk", "class S(str):\n    pass\n");
        let value = evaluated(py, Some(&module), "[1, object(), S('x'), 2 ** 80]");
        let state = WalkState::new();
        let ctx = Ctx {
            pool: &[],
            defs: &[],
            index: &index,
            state: &state,
            mode: WalkMode::Fast,
        };
        let mut unwritten = Unwritten::default();
        assert!(member(
            &schema,
            &Value::Py(&value),
            &mut unwritten.frame(ctx)
        ));
        assert_eq!(unwritten.path.capacity(), 0, "a location was written");
        assert_eq!(unwritten.out.capacity(), 0, "a violation was written");
    });
}

/// In a walk a memo keeps, a list of a tree's reference reads each scalar it
/// holds by its type, as the walk without the memo does, and walks only what
/// the type does not settle.
#[test]
fn a_kept_list_of_a_tree_reads_its_scalars_by_their_type() {
    Python::attach(|py| {
        let tree = Schema::Ref(DefIx::new(0));
        let defs = [Schema::Union(
            vec![
                Schema::Int,
                Schema::list(SeqShape::homogeneous(tree.clone())),
            ]
            .into(),
        )];
        let schema = Schema::Union(
            vec![
                Schema::list(SeqShape::homogeneous(tree.clone())),
                Schema::list(SeqShape::homogeneous(Schema::Union(
                    vec![tree, Schema::Bytes].into(),
                ))),
            ]
            .into(),
        );
        let index = build_index(py, &schema, &defs, &[]);
        let value = evaluated(py, None, "[b'x', [1, 2, 3]]");
        let state = WalkState::new();
        let ctx = Ctx {
            pool: &[],
            defs: &defs,
            index: &index,
            state: &state,
            mode: WalkMode::Fast,
        };
        let read = counted(Reading::KeptScalar, || {
            member(
                &schema,
                &Value::Py(&value),
                &mut Frame::new(&mut Vec::new(), &mut Vec::new(), ctx),
            )
        });
        assert!(state.memo.in_use(), "the memo was not armed");
        assert_eq!(read, (true, 3));
    });
}
