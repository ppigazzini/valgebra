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
use valgebra_core::{ConstIx, DefShift, Field, Openness, PoolShift, Schema, SeqShape};

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

criterion_group!(
    benches,
    bench_simplify,
    bench_shifted,
    bench_with_records_open,
    bench_decision
);
criterion_main!(benches);
