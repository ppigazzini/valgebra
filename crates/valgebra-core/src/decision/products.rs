//! Sequences and products: a fixed arity split across a union, and a linear
//! shape compared position by position.
//!
//! A sequence of fixed length is a product of its positions, and the
//! decomposition of a product against a *union* of products is the rule that
//! needs care: stated the obvious way it enumerates every subset of the
//! negative side, which is where the exponent in the literature comes from.
//! The form here is the backtrack-free one -- narrow one component by the
//! branch and recur on the rest -- so a component narrowed to the empty set
//! ends the question, the empty set being below everything.
//!
//! A repeated tail has no such decomposition: it admits sequences of every
//! length, so there is no tuple of components to split over, and the shapes are
//! compared as the linear forms they are.

use crate::ir::{Schema, SeqShape};
use crate::verdict::{Relation, Verdict};

use super::{SubtypeCx, spend};

/// The element schemas of a *fixed-arity* sequence, or `None` when the shape has
/// a tail.
///
/// Lemma 6.5 decomposes a product, and a sequence is a product only when its
/// component count is fixed: a repeated tail admits sequences of every length,
/// so there is no tuple of components to split over. An empty prefix with no
/// tail is the nullary product.
pub(super) fn fixed_components(shape: &SeqShape) -> Option<Vec<Schema>> {
    shape.tail.is_none().then(|| shape.prefix.to_vec())
}

/// Whether the product `components` is contained in the union of the products in
/// `branches`, all of the same arity.
///
/// JACM Lemma 6.5 characterises this by splitting the negative set every way,
/// which is `2^|N|` subsets *and* needs a hypothesis retracted whenever a split
/// fails. ICFP §2.1.4 gives the equivalent formulation that does not backtrack,
/// and this is that function at `n` components rather than two:
///
/// ```text
/// Phi(P, [])       = false
/// Phi(P, [B, ..R]) = for every i:  P[i] <= B[i]  or  Phi(P with P[i] := P[i] \ B[i], R)
/// ```
///
/// Narrowing a component to the empty set makes the whole product empty, and the
/// empty set is below everything -- which is the base case that makes a value
/// split across branches decide, since no single branch contains it.
pub(super) fn product_subtype(
    components: &[Schema],
    branches: &[&[Schema]],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
    if !spend(cx.budget) {
        return false;
    }
    if components
        .iter()
        .any(|c| c.is_empty_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget))
    {
        return true;
    }
    let Some((branch, rest)) = branches.split_first() else {
        return false;
    };
    components
        .iter()
        .zip(branch.iter())
        .enumerate()
        .all(|(position, (mine, theirs))| {
            mine.is_subtype_rec(theirs, cx, assumptions).holds() || {
                // The component this branch does not cover, narrowed by what the
                // branch takes away, with the rest of the tuple as it was. Built
                // by mapping rather than by writing at an index: the position
                // comes from the same enumeration as the component, so an index
                // write cannot go out of range and cannot be seen not to.
                let narrowed: Vec<Schema> = components
                    .iter()
                    .enumerate()
                    .map(|(index, component)| {
                        if index == position {
                            Schema::meet([mine.clone(), theirs.clone().complement()])
                        } else {
                            component.clone()
                        }
                    })
                    .collect();
                product_subtype(&narrowed, rest, cx, assumptions)
            }
        })
}

/// Whether a fixed-arity sequence is covered by the sequence branches of a union.
///
/// A value that splits across branches -- `tuple[int|str, int]` covered by
/// `tuple[int, int] | tuple[str, int]` -- lands in no single branch, so the rule
/// that tries each branch alone cannot see it. Branches of another container or
/// another arity share no value with `self` by shape, so they drop out rather
/// than blocking the decomposition.
pub(super) fn seq_splits_across_union(
    schema: &Schema,
    members: &[Schema],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> bool {
    let Schema::Seq { container, shape } = schema else {
        return false;
    };
    let Some(components) = fixed_components(shape) else {
        return false;
    };
    let branches: Vec<Vec<Schema>> = members
        .iter()
        .filter_map(|member| match member {
            Schema::Seq {
                container: their_kind,
                shape: their_shape,
            } if their_kind == container => fixed_components(their_shape),
            _ => None,
        })
        .filter(|branch| branch.len() == components.len())
        .collect();
    if branches.is_empty() {
        return false;
    }
    let branches: Vec<&[Schema]> = branches.iter().map(Vec::as_slice).collect();
    product_subtype(&components, &branches, cx, assumptions)
}

/// Whether the language `pa · ta*` is included in `pb · tb*` — a fixed prefix
/// optionally followed by a repeated tail, which is every [`SeqShape`].
/// `ta`/`tb` of `None` mean no repeated tail.
pub(super) fn linear_subtype(
    pa: &[Schema],
    ta: Option<&Schema>,
    pb: &[Schema],
    tb: Option<&Schema>,
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    // A repeated tail with an empty element language never repeats, so the left
    // side is then just its fixed prefix. Emptiness is decided with the same
    // oracle and definitions as the rest of the decision, so a tail empty only
    // under a refinement bound or an uninhabited recursive reference is
    // recognised here too, consistent with the context-aware recursion around it.
    //
    // The verdict is kept in three values, because the one refutation below
    // that stands on the tail *repeating* needs the element proven to have a
    // value: a tail the rules cannot read either way may be no tail at all.
    // No tail repeats nothing, which is what `Empty` says of it.
    let repeats = ta.map_or(Verdict::Empty, |element| {
        element.verdict_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget)
    });
    let ta = ta.filter(|_| repeats != Verdict::Empty);
    // A's fixed prefix must align with B: against B's prefix where they overlap,
    // then against B's repeated tail past it (which B must therefore have). A
    // prefix shorter than B's cannot align at all, which is a refutation by
    // shape: the lengths one side admits are not among the other's.
    if pa.len() < pb.len() {
        return Relation::Fails;
    }
    // One goal, asked again -- the sequence's reading of what a record's fields
    // do, and remembered the same way. A tuple whose positions carry one schema
    // asks one question per position, and the positions are walked in order, so
    // a repeat is the position before this one -- under the same trail, since a
    // position's own decision pushes and pops its way back to it. See
    // `keyed_map_subtype` for that argument and for why the pair is compared by
    // equality rather than by address.
    let mut last: Option<(&Schema, &Schema, Relation)> = None;
    let mut aligns = |assumptions: &mut Vec<(Schema, Schema)>| {
        Relation::all(pa.iter().enumerate().map(|(i, element)| {
            let expected = match pb.get(i) {
                Some(expected) => expected,
                // Past B's prefix, B must repeat -- a fixed-length B admits no
                // such position at all.
                None => match tb {
                    Some(tail) => tail,
                    None => return Relation::Fails,
                },
            };
            match last {
                Some((sub, sup, answer)) if sub == element && sup == expected => answer,
                _ => {
                    let answer = element.is_subtype_rec(expected, cx, assumptions);
                    last = Some((element, expected, answer));
                    answer
                }
            }
        }))
    };
    match (ta, tb) {
        (None, None) if pa.len() != pb.len() => Relation::Fails,
        (None, None | Some(_)) => aligns(assumptions),
        // A repeats without bound but B is finite-length: no value with a
        // repeat is in B. That is a mismatch of the repeated element's values,
        // and it is read the way the query reads the subject's: refuted where
        // the element has one, declined where the rules cannot tell. An
        // element they cannot tell is also not proven empty, or the tail would
        // have been dropped above, so `Holds` is not an answer this arm gives.
        (Some(_), None) => Relation::of_mismatch(repeats),
        // A's repeated element must also land in B's repeated tail.
        (Some(a), Some(tail)) => {
            aligns(assumptions).and(|| a.is_subtype_rec(tail, cx, assumptions))
        }
    }
}
