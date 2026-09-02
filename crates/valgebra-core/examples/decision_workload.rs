//! A fixed, deterministic workload over the core *decision* procedures.
//!
//! `is_subtype_of`, `is_empty` and `is_equivalent`, which the sibling
//! `perf_workload` does not reach: that one covers the schema transformations,
//! and the two together are what the instruction-count gate measures.
//!
//! Run under cachegrind, the count is deterministic for a given build, so a
//! committed budget catches an algorithmic change without a wall clock.
//!
//! The corpus is the shapes whose cost differs: reflexivity on a wide record,
//! inclusion across a union, a literal union against a wider one, inclusion in a
//! complement, a meet of disjoint kinds, and emptiness through a nesting. Keep it
//! and `ITERATIONS` fixed; changing either moves the budget and requires
//! re-recording it.

use valgebra_core::{ConstIx, Field, Openness, Schema, SeqRegex};

/// Iterations per relation. Large enough that process startup is a rounding
/// error against the measured work.
const ITERATIONS: usize = 2_000;

/// A wide closed record whose fields carry pool-indexed leaves.
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

/// A record nested `depth` levels deep, each level behind a list.
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
            vec![Field {
                name: "child".to_owned(),
                schema: Schema::list(SeqRegex::homogeneous(inner)),
                required: true,
            }],
            Openness::Closed,
        );
    }
    inner
}

/// A union of `n` distinct literals: the error-code table shape.
fn literal_union(n: usize) -> Schema {
    Schema::union((0..n).map(|i| Schema::Literal(ConstIx::new(i))))
}

fn main() {
    let record = wide_record(8);
    let record_copy = wide_record(8);
    let nested = nested_records(8);
    let scalar = Schema::Bool;
    let scalar_sup = Schema::union([Schema::Int, Schema::Str]);
    let narrow = literal_union(8);
    let wide = literal_union(9);
    let list_int = Schema::list(SeqRegex::homogeneous(Schema::Int));
    let not_int = Schema::Complement(Box::new(Schema::Int));
    let disjoint = Schema::Intersection(vec![
        Schema::list(SeqRegex::homogeneous(Schema::Int)),
        Schema::Set(Box::new(Schema::Int)),
    ]);

    // Fold a checksum through each verdict so nothing is optimized away.
    let mut checksum: usize = 0;
    for _ in 0..ITERATIONS {
        let b = std::hint::black_box;
        // Reflexivity on the same node, and on an equal one that is a different
        // node: the two settle by different rules and cost differently.
        checksum += usize::from(b(&record).is_subtype_of(b(&record)));
        checksum += usize::from(b(&record).is_subtype_of(b(&record_copy)));
        // Inclusion decided by the scalar region partition.
        checksum += usize::from(b(&scalar).is_subtype_of(b(&scalar_sup)));
        // A union against a wider union, which set containment settles.
        checksum += usize::from(b(&narrow).is_subtype_of(b(&wide)));
        // A complement on the right, which reduces to disjointness.
        checksum += usize::from(b(&list_int).is_subtype_of(b(&not_int)));
        // Emptiness: a meet of disjoint kinds, and a walk through a nesting.
        checksum += usize::from(b(&disjoint).is_empty());
        checksum += usize::from(b(&nested).is_empty());
        // Both directions of one pair, which is what equivalence asks.
        checksum += usize::from(b(&record).is_equivalent(b(&record_copy)));
    }
    // Printing forces the checksum to be observed.
    println!("checksum={checksum}");
}
