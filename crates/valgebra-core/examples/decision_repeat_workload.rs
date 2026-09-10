//! A decision workload whose goals **repeat**.
//!
//! The sibling `decision_workload` asks eight relations and no two of its goals
//! are the same pair of nodes: a wide record's fields are distinct literals, a
//! nesting's levels are distinct, two literal tables share no subterm. So it
//! measures the decision procedure over shapes that a memo over goals would
//! leave exactly as they are -- and the budget that stands in for such a memo
//! (`DECISION_BUDGET`, and the descriptor's three) is argued over a workload
//! that could not show one working.
//!
//! These are the shapes that can. A record whose fields all carry one schema,
//! against a record whose fields all carry a wider one: the inclusion at every
//! field is the same goal, because the fields share their element schema and
//! construction shares equal subtrees. One key, asked `WIDTH` times. And the
//! same repetition in a tuple, whose elements are reached by position and
//! walked by a rule of their own.
//!
//! Run under cachegrind its count is deterministic for a given build, so the
//! saving a memo would make is the difference between this count and the one it
//! reads afterwards -- and the cost is what the sibling workloads read. Keep the
//! corpus and the counts fixed; changing either moves the budget.

use valgebra_core::{Field, Openness, Schema, SeqShape};

/// Iterations, chosen so the count is dominated by the work rather than by
/// process startup.
const ITERATIONS: usize = 2_000;

/// Fields, which is how many times the one goal is asked.
const WIDTH: usize = 8;

/// The nesting under each field, which is what one goal costs.
const DEPTH: usize = 2;

/// A list nested `depth` deep over `leaf`.
fn nested_lists(depth: usize, leaf: Schema) -> Schema {
    (0..depth).fold(leaf, |inner, _| Schema::list(SeqShape::homogeneous(inner)))
}

/// A tuple whose positions all carry `element`.
///
/// The record's repetition in the other container: elements reached by
/// position rather than by name, walked by the rule that aligns two shapes.
fn repeating_tuple(width: usize, element: &Schema) -> Schema {
    Schema::tuple(SeqShape::fixed((0..width).map(|_| element.clone())))
}

/// A record whose fields all carry `element`.
fn repeating(width: usize, element: &Schema) -> Schema {
    Schema::record(
        (0..width)
            .map(|index| Field {
                name: format!("f{index}").into(),
                schema: element.clone(),
                required: true,
            })
            .collect(),
        Openness::Closed,
    )
}

fn main() {
    let element = nested_lists(DEPTH, Schema::Int);
    let wider_element = nested_lists(DEPTH, Schema::union([Schema::Int, Schema::Str]));
    let narrow = repeating(WIDTH, &element);
    let wide = repeating(WIDTH, &wider_element);
    let narrow_tuple = repeating_tuple(WIDTH, &element);
    let wide_tuple = repeating_tuple(WIDTH, &wider_element);
    // Fold a checksum through each verdict so nothing is optimized away.
    let mut checksum: usize = 0;
    for _ in 0..ITERATIONS {
        let b = std::hint::black_box;
        // The proof: every field's element is inside the wider one, which is one
        // goal reached once per field.
        checksum += usize::from(b(&narrow).is_subtype_of(b(&wide)));
        // And the refutation, which walks the same repeated goal the other way:
        // the wider element is not inside the narrow one, and the first field
        // settles it -- so the pair is here to show that a memo may not turn a
        // walk that stops early into one that does not.
        checksum += usize::from(b(&wide).is_subtype_of(b(&narrow)));
        // The same pair of questions in the container whose elements are
        // reached by position, which repeats a goal the same way and is walked
        // by a different rule.
        checksum += usize::from(b(&narrow_tuple).is_subtype_of(b(&wide_tuple)));
        checksum += usize::from(b(&wide_tuple).is_subtype_of(b(&narrow_tuple)));
    }
    // Printing forces the checksum to be observed.
    println!("checksum={checksum}");
}
