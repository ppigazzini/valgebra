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

use pyo3::Python;

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(100_000);
    let checksum = Python::attach(|py| _valgebra::binding_perf_workload(py, iters));
    println!("{checksum}");
}
