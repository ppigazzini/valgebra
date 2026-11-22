//! The readings chosen for their cost, each held to the values it is chosen
//! for. Every one answers as the reading beside it does, which the walk's
//! corpus holds, so what these read is the note each leaves
//! ([`taken`](super::taken)): a value sent to the other reading keeps every
//! answer and fails here.

use pyo3::types::PyModule;
use valgebra_core::SeqShape;

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
