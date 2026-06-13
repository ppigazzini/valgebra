//! Wall-clock micro-benchmarks for the pure-Rust schema operations.
//!
//! These cover the transformations the compiler and combinators lean on:
//! `simplify` (the law-justified reducer), `shifted` (validator composition),
//! and `with_records_open` (the `lax`/`strict` recursive transform). The walk
//! over Python values lives in the bindings crate and is benchmarked from
//! Python; this harness isolates the work that is independent of `PyO3`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use valgebra_core::{Field, Schema, SeqRegex};

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
            Schema::Union(vec![Schema::Bool, Schema::Anything, node]),
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
            schema: Schema::Literal(i),
            required: i % 2 == 0,
        })
        .collect();
    Schema::Record {
        fields,
        open: false,
    }
}

/// A record nested `depth` levels deep, each level holding a small record, so
/// `with_records_open` rebuilds the whole spine.
fn nested_records(depth: usize) -> Schema {
    let mut inner = Schema::Record {
        fields: vec![Field {
            name: "leaf".to_owned(),
            schema: Schema::Int,
            required: true,
        }],
        open: false,
    };
    for _ in 0..depth {
        inner = Schema::Record {
            fields: vec![
                Field {
                    name: "child".to_owned(),
                    schema: Schema::list(SeqRegex::homogeneous(inner)),
                    required: true,
                },
                Field {
                    name: "tag".to_owned(),
                    schema: Schema::Str,
                    required: false,
                },
            ],
            open: false,
        };
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
        b.iter(|| black_box(&schema).shifted(100, 0));
    });
}

fn bench_with_records_open(c: &mut Criterion) {
    let schema = nested_records(32);
    c.bench_function("with_records_open_depth32", |b| {
        b.iter(|| black_box(&schema).with_records_open(true));
    });
}

criterion_group!(
    benches,
    bench_simplify,
    bench_shifted,
    bench_with_records_open
);
criterion_main!(benches);
