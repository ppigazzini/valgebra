//! Run the Python suite from `cargo test`, so a mutation sweep can observe it.
//!
//! Seven files of this crate are reached only through the shipped extension:
//! the entry points, the exception, the report the walk hands back, the render.
//! `cargo mutants` runs `cargo test`, which never loads the extension, so every
//! mutant of those files survives and says nothing about the tests. They were
//! excluded from the sweep by name, and the reason written beside each was
//! "pytest covers this" -- an assertion nothing measured.
//!
//! This makes it a measurement. Behind a feature, because the cost is a rebuild
//! and a suite run per mutant and no ordinary `cargo test` should pay it; the
//! sweep that enables the feature is scheduled rather than on the merge path.
//!
//! ## What it needs
//!
//! `VALGEBRA_SWEEP_VENV` names where the environments live: an absolute path
//! outside the tree, since a sweep runs from a copy and a path inside one names
//! a different directory per mutant.
//!
//! **One environment per worker, not one per sweep.** `cargo mutants -j N` runs
//! N workers in N copies of the tree, and two workers building the extension
//! into one environment write the same files at the same moment -- which fails
//! with `File exists` and takes the whole run down with it, a third of the way
//! through and with no verdict. Each worker's checkout has a directory name of
//! its own, so that name picks the environment, and the first mutant a worker
//! runs builds it from the lock file while every mutant after reuses it.
//!
//! Without the variable the test does nothing and says so. A sweep that lost
//! the variable would otherwise report every mutant caught, which is the
//! failure mode this whole file exists to end: a green number about a suite
//! that never ran.

#![cfg(feature = "pytest-sweep")]

use std::path::{Path, PathBuf};
use std::process::Command;

/// The checkout this test is running from: the mutated copy, under a sweep.
fn tree() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is `<tree>/crates/valgebra-py`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the manifest sits two levels below the tree root")
        .to_path_buf()
}

/// The environment this worker uses, beside the path the variable names.
///
/// Keyed by the checkout's own directory name, which `cargo mutants` makes
/// unique per worker: two workers then never share one environment, and the
/// mutants one worker runs all share its own.
fn worker_venv(base: &Path, at: &Path) -> PathBuf {
    let checkout = at
        .file_name()
        .map_or_else(|| "tree".to_owned(), |name| name.to_string_lossy().into());
    let base_name = base.file_name().map_or_else(
        || "sweep-venv".to_owned(),
        |name| name.to_string_lossy().into(),
    );
    base.with_file_name(format!("{base_name}-{checkout}"))
}

/// Build the worker's environment from the lock file, unless it is there.
///
/// Once per worker rather than once per mutant: the check is the whole of what
/// makes the cost amortise. The project itself is left out -- the extension is
/// built into this environment by `maturin develop` below, from the mutated
/// checkout, which is the thing being varied.
fn ensure(venv: &Path, at: &Path) -> Result<(), String> {
    if venv.join("bin").join("python").is_file() {
        return Ok(());
    }
    let output = Command::new("uv")
        .args(["sync", "--locked", "--no-install-project"])
        .current_dir(at)
        .env("UV_PROJECT_ENVIRONMENT", venv)
        .output()
        .map_err(|error| format!("uv did not start: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "uv sync into {} failed ({}): {}",
        venv.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr),
    ))
}

fn run(program: &Path, args: &[&str], venv: &Path, at: &Path) -> Result<(), String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(at)
        .env("VIRTUAL_ENV", venv)
        // A sweep runs many of these in sequence and the cache is a shared
        // directory: one writer per mutant is what it is not built for.
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .map_err(|error| format!("{} did not start: {error}", program.display()))?;
    if output.status.success() {
        return Ok(());
    }
    let tail = |bytes: &[u8]| {
        let text = String::from_utf8_lossy(bytes).to_string();
        text.chars().rev().take(4000).collect::<String>()
    };
    Err(format!(
        "{} failed ({}):\nstdout tail: {}\nstderr tail: {}",
        program.display(),
        output.status,
        tail(&output.stdout),
        tail(&output.stderr),
    ))
}

/// The Python suite, against the extension built from this checkout.
///
/// A mutant of a file only the extension reaches is caught here or nowhere: the
/// suite is the only caller those files have.
#[test]
fn the_python_suite_answers_for_the_extension_this_checkout_builds() {
    let Ok(venv) = std::env::var("VALGEBRA_SWEEP_VENV") else {
        panic!(
            "VALGEBRA_SWEEP_VENV is unset, so the Python suite did not run. A \
             sweep without it reports every mutant caught while measuring \
             nothing, so this refuses rather than passing."
        );
    };
    let base = PathBuf::from(venv);
    assert!(
        base.is_absolute(),
        "VALGEBRA_SWEEP_VENV must be absolute: a sweep runs from a copy of the \
         tree, and a relative path would name a directory inside it"
    );

    let at = tree();
    let venv = worker_venv(&base, &at);
    ensure(&venv, &at).expect("the worker's environment is built");
    let python = venv.join("bin").join("python");
    let maturin = venv.join("bin").join("maturin");
    assert!(python.is_file(), "{} holds no python", venv.display());
    assert!(maturin.is_file(), "{} holds no maturin", venv.display());
    run(&maturin, &["develop", "--uv"], &venv, &at).expect("the extension builds");
    run(
        &python,
        &[
            "-m",
            "pytest",
            "-x",
            "-q",
            "-m",
            "not repository",
            "-p",
            "no:cacheprovider",
            "--deselect",
            "tests/test_concurrency.py",
        ],
        &venv,
        &at,
    )
    .expect("the Python suite passes");
}
