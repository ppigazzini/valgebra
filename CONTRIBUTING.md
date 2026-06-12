# Contributing to valgebra

Thanks for your interest. valgebra is a Rust-core Python validation library;
this guide covers local setup and the checks every change must pass. The
project's design rules and non-negotiable invariants live in
[AGENTS.md](AGENTS.md) — read that first.

## Setup

Requirements: stable Rust (edition 2024, MSRV 1.88), Python >= 3.10, and
[`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                     # create .venv and install dev dependencies
uv run maturin develop      # build the Rust extension into the venv
uv run pre-commit install   # enable the git hooks
```

Verify the build:

```bash
uv run python -c "from valgebra import validator; print(validator(int).is_valid(7))"
```

## The gate

A change is not done until every command exits 0. CI runs the same set on
Linux, macOS, and Windows; local runs are previews of that source of truth.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
uv run maturin develop --uv
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
```

`pre-commit run --all-files` runs the file-hygiene, ruff, and cargo gates in
one step.

## Working on changes

- Branch off `main`; open a pull request. Both push and PR trigger CI, and the
  aggregated `ci` check must be green to merge.
- Keep the Python/Rust boundary explicit: the validator tree runs in Rust;
  Python predicates are a documented slow path, never a silent fallback.
- No schema combinator or annotation form lands without its denotation written
  in the same change and its algebra laws covered by property tests.

See [AGENTS.md](AGENTS.md) for the full rules and the rationale behind them.

## Commit messages

Conventional commits, body wrapped at 80 columns, authoritative mood (describe
what the system does after the change, not the act of changing it).

```
feat: short imperative summary

Body wrapped at 80 columns describing the resulting behavior.
```
