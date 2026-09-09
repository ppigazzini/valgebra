//! Re-emit the interpreter's own configuration flags for this crate.
//!
//! `PyO3`'s build script decides what the interpreter it links against
//! supports and emits a `Py_3_x` flag per release it is at or past, plus
//! `Py_GIL_DISABLED` on the free-threaded build. Those flags reach the crate
//! that emits them and no other, so a `cfg!(Py_3_14)` written here reads
//! `false` against every interpreter until this script re-emits them --
//! silently, and in the direction that says "an older interpreter", which is
//! how a version-gated fast path ends up taken everywhere.
//!
//! The membership walk reads one of those flags to choose how it reads a list's
//! elements, because which reading is cheaper is a property of the interpreter
//! rather than of the code.

fn main() {
    pyo3_build_config::use_pyo3_cfgs();
}
