# Architecture

valgebra is a Rust-core Python library. A schema denotes a *set of Python
values*; validation is membership testing on the object the caller already holds
— zero-copy, zero-coercion. A schema compiles once into a validator tree, and
the hot path crosses into Rust exactly once per call.

This page is a map. Every claim it would carry lives on the page that owns its
zone, in the [developer documentation set](docs/dev/README.md), and the public
API is documented on the [docs site](https://ppigazzini.github.io/valgebra/).

| Component | Path | Owns |
| --- | --- | --- |
| Core | [`crates/valgebra-core/`](crates/valgebra-core/src/lib.rs) | the schema IR and what each node denotes, the two schema-to-schema deciders, the structured failure; pure Rust, no PyO3 |
| Bindings | [`crates/valgebra-py/`](crates/valgebra-py/src/lib.rs) | the schema frontend, the membership walk, the errors and `repr`; built as the `_valgebra` extension |
| Package | [`python/valgebra/`](python/valgebra/__init__.py) | the importable public surface; `_valgebra` is private |

## Where to look

- The zones, the direction their dependencies run, how a value and a pair of
  schemas flow, the invariants, every bound and the refused designs:
  [docs/dev/architecture.md](docs/dev/architecture.md).
- What a node denotes: the `Schema` enum in
  [`crates/valgebra-core/src/ir.rs`](crates/valgebra-core/src/ir.rs), whose
  variant comments are where meaning is written, and
  [docs/dev/schema-ir.md](docs/dev/schema-ir.md).
- What is decidable: [docs/dev/decision.md](docs/dev/decision.md), and the
  published boundary in [docs/explanation/decidability.md](docs/explanation/decidability.md).
- An annotation form: [docs/dev/frontend.md](docs/dev/frontend.md). What a
  value matches: [docs/dev/walk.md](docs/dev/walk.md). An error or a code:
  [docs/dev/errors.md](docs/dev/errors.md).
- The gate, the lanes and a release: [CONTRIBUTING.md](CONTRIBUTING.md),
  [docs/dev/tooling-ci.md](docs/dev/tooling-ci.md),
  [docs/dev/releasing.md](docs/dev/releasing.md).
- The theory behind the algebra: [docs/explanation/foundations.md](docs/explanation/foundations.md).
- The rules that bind agent and human contributors alike: [AGENTS.md](AGENTS.md).
