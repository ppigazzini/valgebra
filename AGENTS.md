# AGENTS

valgebra is a Rust-core Python library for runtime validation. A schema denotes
a set of Python values; validation is membership testing on the object the
caller already holds — zero-copy, zero-coercion. Schemas compile once into a
Rust validator tree that the hot path crosses into exactly once per call.

It is not a faster pydantic. pydantic and msgspec own parse-and-ingest; valgebra
owns check-and-contract. Keep that framing in code, docs, and commits.


## Setup

```bash
uv sync                 # resolve and install dev dependencies into .venv
maturin develop --uv    # build the Rust extension into the venv (rerun after
                        # any change under crates/)
```

Stable Rust (edition 2024, MSRV 1.88) and `uv` must be on PATH.


## Build Health Gate

A change is not done until every command exits 0. Trust exit codes, not log
text — a process can print progress and then fail. If a gate cannot run, say so
and list what was checked instead.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
maturin develop --uv                    # after any Rust change
ruff check . && ruff format --check .
ty check
pytest
```


## Layout

- `crates/valgebra-core/` — pure Rust: schema IR, validator tree, error model.
  No PyO3 here.
- `crates/valgebra-py/` — PyO3 bindings; built as the `_valgebra` extension.
- `python/valgebra/` — the importable package. `_valgebra` is private; import
  from `valgebra`, never `valgebra._valgebra`.
- `tests/` — pytest suites.

Style, naming, and formatting are enforced by ruff, clippy, and ty — follow
their output rather than restating rules here.


## Project Rules

These are the invariants tooling cannot enforce. Each pairs a prohibition with
what to do instead.

- **Write the denotation before the combinator.** A new schema construct must
  state, in the same change, which set of Python values it accepts. Do not land
  a combinator described only as "works like some other tool".

- **Prove laws, don't assert them.** Any claimed algebraic equivalence
  (associativity, De Morgan, top/bottom identity, simplifier rewrites) ships
  with a property test — proptest on the Rust side, hypothesis on the Python
  side. Do not document a law without covering it.

- **Keep the hot path allocation-free and Rust-only.** `validate` and
  `is_valid` check membership of the actual object; they do not copy or coerce.
  When a value must be returned after checking, use the explicit `ensure` (or
  `load` for JSON) mode. User predicates and custom types run as a documented
  Python-callback slow path — name it as
  such, never let it become a silent fallback on the default loop.

- **Cross the Python/Rust boundary once per call.** Push tree walks, key
  lookups, and bound checks into the Rust validator tree. Do not add Python work
  inside the per-element validation loop.

- **Prefer ecosystem crates over custom Rust.** Reach for PyO3, jiter, and
  rustc-hash first. If you write custom core Rust anyway, record why in
  the same change. Never add ruff or ty crates as dependencies — they are design
  references only.

- **Never vendor oracle or reference source.** pydantic-core, ruff, and ty
  source does not enter this tree; they are design references only. Pull any
  comparison oracle in as a pinned dev dependency rather than copying it.

- **Use plain names that don't shadow builtins.** Names like `compile`,
  `filter`, `time`, and `date` clash with Python builtins; the public API
  must not reuse them.

- **Lead every example with the typing-annotation form.** Show a valgebra
  combinator only for what annotations cannot express. Examples run as written
  with full imports, or are labelled planned.


## Commits

Conventional commits, body wrapped at 80 columns. Describe the system as it
stands after the change in authoritative mood — not "added X" or "this commit".
No meta commentary. No references to gitignored files.

```
feat: short imperative summary

Authoritative body wrapped at 80 columns describing what the system does
now, not the act of changing it.
```
