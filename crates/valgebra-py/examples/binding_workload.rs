//! The body of the binding-level instruction-count regression gate.
//!
//! Runs the membership walk over a live Python value `iters` times (the count is
//! the first CLI argument), then prints a checksum so the optimizer cannot
//! discard the work. Embedding `CPython` means the absolute count includes a
//! non-fixed interpreter startup, so `scripts/perf_gate.py` measures the
//! *difference* between two iteration counts: startup cancels, leaving the
//! deterministic per-iteration walk cost — the shipped hot path the core-only
//! workload does not reach.
//!
//! Requires the embedded interpreter, so it is built with
//! `--features interpreter-tests` and run with the interpreter's library
//! directory on the loader path.

use _valgebra::BindingShape;
use pyo3::Python;

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100_000);
    // The shape defaults to the walk, so the gate's original invocation and the
    // budget recorded for it keep their meaning.
    let name = std::env::args().nth(2).unwrap_or_else(|| "walk".to_owned());
    let Some(shape) = BindingShape::named(&name) else {
        eprintln!(
            "unknown shape {name:?}: walk, boundary, record, build, explain, explain-accept, open"
        );
        std::process::exit(2);
    };
    let checksum = Python::attach(|py| _valgebra::binding_perf_workload_shape(py, shape, iters));
    println!("{checksum}");
}
