# References

What this project is checked against. A claim in this set is expected to be
traceable to something here or to the live tree.

Papers are [09-theory.md](09-theory.md). This page is implementations and
toolchain.

## Reference implementations

These are **read-only design references**, consulted as checkouts elsewhere on
disk. None is vendored into this tree, and none is a dependency.

- **pydantic-core** — <https://github.com/pydantic/pydantic-core>. The
  compile-once, validate-fast architecture valgebra's shape follows: a validator
  built once from a schema description and reused, with the walk in Rust and the
  boundary crossings named. It is also a differential oracle: the same schemas
  and values run through both over the fragment where the semantics agree.
- **jiter** — <https://github.com/pydantic/jiter>. The JSON parser valgebra
  uses, and the reason the JSON path can validate in place rather than
  materialising Python objects first.
- **ruff and ty** — <https://github.com/astral-sh/ruff>. ty's semantic crates are
  a set-theoretic type system in Rust, read for how a lattice of types is
  represented and decided. valgebra does not link them.

**The three divergences from pydantic-core are enumerated in
`tests/test_differential.py`** and are deliberate: `bool` as a subtype of `int`,
`int` and `float` as disjoint, and exact-match `Literal` membership. Read them
there — a second list drifts.

## The typing spec

- **The Python typing specification** —
  <https://typing.python.org/en/latest/spec/>. What an annotation means. Where
  valgebra's runtime semantics and a static reading differ, the difference is a
  valgebra decision and is stated in `docs/` as one.

The distinction this page keeps: a **fact** is verified from a reference; a
**decision** is a valgebra policy choice. Do not blur them.

## The binding

- **PyO3** — <https://github.com/PyO3/pyo3> and its guide. Classes, conversions,
  and the free-threading rules the validator's shared-across-threads guarantee
  rests on.
- **maturin** — <https://github.com/PyO3/maturin>. The mixed Rust/Python layout
  and the wheel build, including the profile-guided build the release lane uses.

## Rust

- **The Rust Reference** — <https://doc.rust-lang.org/reference/>. Particularly
  type layout, which is what makes `#[repr(transparent)]` a property of the type
  rather than a hope, and the overflow rules
  ([06-type-design.md](06-type-design.md)).
- **The Cargo book, on profiles** —
  <https://doc.rust-lang.org/cargo/reference/profiles.html>.
- **Rust API Guidelines** — <https://rust-lang.github.io/api-guidelines/>.
  Particularly the type-safety chapter: arguments convey meaning through types
  rather than through positional convention.

## Type design

The value-domain design rests on these; [06-type-design.md](06-type-design.md)
is where they are applied.

- **matklad, "Newtype Index Pattern"** —
  <https://matklad.github.io/2018/06/04/newtype-index-pattern.html>. One index
  type per collection, constructed in one place. This is the five index spaces.
- **`rustc_index::newtype_index!`** — the same pattern at compiler scale, named
  as the reference implementation and not as a dependency: valgebra writes the
  four-line struct by hand.
- **Wlaschin, "Designing with types: making illegal states unrepresentable"** —
  <https://fsharpforfunandprofit.com/posts/designing-with-types-making-illegal-states-unrepresentable/>.
  The sealed walk mode, where a pair of booleans admitted a fourth state nothing
  meant.
- **King, "Parse, Don't Validate"** —
  <https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/>. The
  frontend is a parser from annotations into the IR, and the indices it mints are
  the proofs that would otherwise evaporate into `usize`.
- **"When Zero Cost Abstractions Aren't Zero Cost"** —
  <https://blog.polybdenum.com/2021/08/09/when-zero-cost-abstractions-aren-t-zero-cost.html>.
  Read alongside every claim above: a newtype is free in layout and not always
  free in codegen, which is why adding one here ends with a budget run.

## Testing and measurement

- **Just et al., "Are Mutants a Valid Substitute for Real Faults?" (FSE 2014)**.
  Mutant detection correlates with real-fault detection independently of
  coverage — the reason the sweeps exist beside the coverage floors.
- **Inozemtseva & Holmes, "Coverage Is Not Strongly Correlated with Test Suite
  Effectiveness" (ICSE 2014)** —
  <https://www.cs.ubc.ca/~rtholmes/papers/icse_2014_inozemtseva.pdf>. Why a line
  floor is a floor and never the primary claim.
- **Papadakis et al., "Mutation Testing Advances" (2019)** —
  <https://mutationtesting.uni.lu/survey.pdf>. The equivalent-mutant problem, and
  why a zero-survivor target is the classic failure mode.
- **Mytkowicz et al., "Producing Wrong Data Without Doing Anything Obviously
  Wrong!" (ASPLOS 2009)**. Why a wall-clock CI budget cannot gate, and why the
  instruction count does.
- **McKeeman, "Differential Testing for Software" (1998)**.
- **Kaner & Bond, "Software Engineering Metrics: What Do They Measure and How Do
  We Know?" (2004)** — <https://kaner.com/pdfs/metrics2004.pdf>. Construct
  validity: ask what a number measures before asking whether it is high.

## Tooling

cargo-mutants <https://mutants.rs/>; cargo-llvm-cov; cargo-fuzz; hypothesis;
proptest; cachegrind
<https://valgrind.org/docs/manual/cg-manual.html>; cargo-deny; zizmor.

## The toolchain baseline

Read the pins from the files that own them, not from prose: `Cargo.toml` for the
Rust edition and MSRV, `pyproject.toml` for the supported Python range and the
pinned tool versions, and `.github/workflows/ci.yml` for the interpreter matrix
and the nightly the fuzz lane uses. Every one of those has been quoted in a doc
somewhere and gone stale within a release.
