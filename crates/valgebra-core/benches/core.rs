//! Wall-clock micro-benchmarks for the pure-Rust schema operations.
//!
//! These cover the transformations the compiler and combinators lean on:
//! `simplify` (the law-justified reducer), `shifted` (validator composition),
//! `with_records_open` (the `lax`/`strict` recursive transform), and the
//! decision procedures over the shapes a real annotation produces. The walk
//! over Python values lives in the bindings crate and is benchmarked from
//! Python; this harness isolates the work that is independent of `PyO3`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use valgebra_core::descr::classes::Class;
use valgebra_core::descr::lower::{Bounds, Constants, Operand, lower, lower_within};
use valgebra_core::{
    ClassIx, ConstIx, Constraint, DefShift, Field, Openness, OperandIx, PoolShift, Schema, SeqShape,
};

/// A redundant Boolean expression that exercises every simplifier rewrite:
/// nested unions and intersections, duplicate members, top/bottom identities,
/// and double-negated complements.
fn boolean_corpus(depth: usize) -> Schema {
    let mut node = Schema::Union(vec![
        Schema::Int,
        Schema::Int,
        Schema::Nothing,
        Schema::Complement(Box::new(Schema::Complement(Box::new(Schema::Str)))),
    ]);
    for _ in 0..depth {
        node = Schema::Complement(Box::new(Schema::Intersection(vec![
            node.clone(),
            Schema::Union(vec![Schema::Bool, Schema::ANYTHING, node]),
        ])));
    }
    node
}

/// A wide record whose fields carry pool-indexed leaves, so `shifted` has to
/// rewrite many indices in one pass.
fn wide_record(width: usize) -> Schema {
    let fields = (0..width)
        .map(|i| Field {
            name: format!("f{i}"),
            schema: Schema::Literal(ConstIx::new(i)),
            required: i % 2 == 0,
        })
        .collect();
    Schema::record(fields, Openness::Closed)
}

/// A record nested `depth` levels deep, each level holding a small record, so
/// `with_records_open` rebuilds the whole spine.
fn nested_records(depth: usize) -> Schema {
    let mut inner = Schema::record(
        vec![Field {
            name: "leaf".to_owned(),
            schema: Schema::Int,
            required: true,
        }],
        Openness::Closed,
    );
    for _ in 0..depth {
        inner = Schema::record(
            vec![
                Field {
                    name: "child".to_owned(),
                    schema: Schema::list(SeqShape::homogeneous(inner)),
                    required: true,
                },
                Field {
                    name: "tag".to_owned(),
                    schema: Schema::Str,
                    required: false,
                },
            ],
            Openness::Closed,
        );
    }
    inner
}

fn bench_simplify(c: &mut Criterion) {
    let schema = boolean_corpus(8);
    c.bench_function("simplify_boolean_depth8", |b| {
        b.iter(|| black_box(&schema).simplify());
    });
}

fn bench_shifted(c: &mut Criterion) {
    let schema = wide_record(64);
    c.bench_function("shifted_record_width64", |b| {
        b.iter(|| black_box(&schema).shifted(PoolShift::new(100), DefShift::new(0)));
    });
}

fn bench_with_records_open(c: &mut Criterion) {
    let schema = nested_records(32);
    c.bench_function("with_records_open_depth32", |b| {
        b.iter(|| black_box(&schema).with_records_open(Openness::Open));
    });
}

/// The shapes a real annotation produces, which is the population any change to
/// the decision procedure has to be measured against.
///
/// A decision on these costs tens of nanoseconds, so per-call setup a wider
/// schema would amortize is paid here in full and repays nothing. That is the
/// finding a previous effort recorded after building five commits of
/// memoization and reverting all of it, and this harness is what holds a future
/// one to it.
fn bench_decision(c: &mut Criterion) {
    let scalar = Schema::Bool;
    let scalar_sup = Schema::union([Schema::Int, Schema::Str]);
    c.bench_function("subtype_bool_below_int_or_str", |b| {
        b.iter(|| black_box(&scalar).is_subtype_of(black_box(&scalar_sup)));
    });

    let record = wide_record(8);
    c.bench_function("subtype_record_width8_reflexive", |b| {
        b.iter(|| black_box(&record).is_subtype_of(black_box(&record)));
    });

    let small_enum = |n: usize| Schema::union((0..n).map(|i| Schema::Literal(ConstIx::new(i))));
    let narrow = small_enum(8);
    let wide = small_enum(9);
    c.bench_function("subtype_enum8_below_enum9", |b| {
        b.iter(|| black_box(&narrow).is_subtype_of(black_box(&wide)));
    });

    let empty_meet = Schema::meet([Schema::Int, Schema::Str]);
    c.bench_function("is_empty_disjoint_meet", |b| {
        b.iter(|| black_box(&empty_meet).is_empty());
    });

    let nested = nested_records(8);
    c.bench_function("is_empty_nested_record_depth8", |b| {
        b.iter(|| black_box(&nested).is_empty());
    });
}

/// A pool that reads every operand as its own index, so a bound and a step have
/// values to be built from without an interpreter.
struct Indexed;

impl Constants for Indexed {
    fn operand(&self, index: OperandIx) -> Option<Operand> {
        i64::try_from(index.get()).ok().map(Operand::Integer)
    }

    fn constant(&self, index: ConstIx) -> Option<Operand> {
        i64::try_from(index.get()).ok().map(Operand::Integer)
    }

    fn class(&self, _index: ClassIx) -> Option<Class> {
        None
    }
}

/// What a *build* costs, which is the quantity the lowering's three bounds are
/// set from and the reason the structural rules answer first.
///
/// The four relations the descriptor decides and the rules do not are the cheap
/// side; the nested and sibling shapes are the expensive one. A bound whose
/// number lives only in a comment cannot be re-derived on another machine, and
/// cannot fail when the shape it guards against changes -- these are that
/// number's workload.
fn bench_lowering(c: &mut Criterion) {
    let list = |element| Schema::list(SeqShape::homogeneous(element));
    let pattern = |text: &str| Schema::Refine {
        base: Box::new(Schema::Str),
        constraints: vec![Constraint::Regex(text.to_owned())],
    };
    let step = |at| Schema::Refine {
        base: Box::new(Schema::Int),
        constraints: vec![Constraint::MultipleOf(OperandIx::new(at))],
    };

    // The wins: each is a difference the structural rules decline and the sets
    // decide, and each is what `WORK` has to leave room for.
    let wins: [(&str, Schema); 4] = [
        (
            "container_meet",
            Schema::meet([list(Schema::Int), list(Schema::Str)]),
        ),
        (
            "double_complement",
            Schema::meet([
                Schema::tuple(SeqShape::fixed([Schema::Int])),
                Schema::tuple(SeqShape::fixed([Schema::Str]))
                    .complement()
                    .complement(),
            ]),
        ),
        (
            "regular_language",
            Schema::meet([pattern("a"), pattern("ab?").complement()]),
        ),
        (
            "step_divides",
            Schema::meet([step(4), step(2).complement()]),
        ),
    ];
    for (name, schema) in &wins {
        c.bench_function(&format!("lower_{name}"), |b| {
            b.iter(|| lower(black_box(schema), &Indexed));
        });
    }

    // The blow-up: cost is exponential in nesting, which is what `DEPTH` bounds.
    // Measured unheld, so the growth is visible rather than clipped.
    for depth in [0usize, 2, 4, 6] {
        let schema = nested_records(depth);
        c.bench_function(&format!("lower_nested_records_depth{depth}"), |b| {
            b.iter(|| lower_within(Bounds::UNHELD, black_box(&schema), &Indexed));
        });
    }

    // Breadth multiplies too: a union of records minus a union of its siblings
    // is the shape that spent a third of a second and then refused because the
    // result was too wide.
    let members: Vec<Schema> = (0..4).map(|_| nested_records(3)).collect();
    let whole = Schema::Union(members.clone());
    let siblings = Schema::Union(members.into_iter().skip(1).collect());
    let difference = Schema::Intersection(vec![whole, siblings.complement()]);
    c.bench_function("lower_sibling_union_difference_unheld", |b| {
        b.iter(|| lower_within(Bounds::UNHELD, black_box(&difference), &Indexed));
    });
    // And the same shape under the allowance the tree ships, which is the number
    // this pair of benchmarks exists to justify.
    c.bench_function("lower_sibling_union_difference_held", |b| {
        b.iter(|| lower(black_box(&difference), &Indexed));
    });
}

criterion_group!(
    benches,
    bench_simplify,
    bench_shifted,
    bench_with_records_open,
    bench_decision,
    bench_lowering
);
criterion_main!(benches);
