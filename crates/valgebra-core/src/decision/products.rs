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
/// `branches`, all of the same arity, in the three values of [`Relation`].
///
/// JACM Lemma 6.5 characterises this by splitting the negative set every way,
/// which is `2^|N|` subsets *and* needs a hypothesis retracted whenever a split
/// fails. ICFP §2.1.4 gives the equivalent formulation that does not backtrack,
/// and this is that function at `n` components rather than two:
///
/// ```text
/// Phi(P, [])       = P is empty
/// Phi(P, [B, ..R]) = for every i:  P[i] <= B[i]  or  Phi(P with P[i] := P[i] \ B[i], R)
/// ```
///
/// Narrowing a component to the empty set makes the whole product empty, and the
/// empty set is below everything -- which is the base case that makes a value
/// split across branches decide, since no single branch contains it.
///
/// **A branch that shares no value with the product at some position is
/// dropped.** Where `P[j] & B[j]` is empty, narrowing `P[j]` by `B[j]` leaves it
/// as it was, so that conjunct is `Phi(P, R)`, and every other conjunct follows
/// from it because narrowing only makes a product easier to cover:
/// `Phi(P, [B, ..R]) = Phi(P, R)`, exactly. Without the drop the recursion
/// narrows by every branch at every position, arity to the power of the branch
/// count, and `tuple[K, K]` against the twenty-five pairs of a five-kind `K` ran
/// out of budget on an inclusion that holds. A branch is dropped only on a
/// *proof* that the two share nothing; one merely not shown to overlap stays.
///
/// **The characterisation is exact, so it refutes as well as proves.** A
/// conjunct fails where its position is refuted against the branch *and* the
/// narrowed product is refuted against the rest; the base case fails where every
/// narrowed component is proved inhabited, which is a product with a value and
/// no branch left. The conjuncts are asked in order and the first that does not
/// hold is the answer, as the boolean reading stopped at the first `false`.
pub(super) fn product_subtype(
    components: &[Schema],
    branches: &[&[Schema]],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    if !spend(cx.budget) {
        return Relation::Unknown;
    }
    let verdicts: Vec<Verdict> = components
        .iter()
        .map(|c| c.verdict_rec(cx.oracle, cx.defs, &mut Vec::new(), cx.budget))
        .collect();
    if verdicts.iter().any(|verdict| verdict.is_empty()) {
        return Relation::Holds;
    }
    let mut remaining = branches;
    let (branch, rest) = loop {
        let Some((branch, rest)) = remaining.split_first() else {
            return if verdicts.iter().all(|v| *v == Verdict::Inhabited) {
                Relation::Fails
            } else {
                Relation::Unknown
            };
        };
        let disjoint = components
            .iter()
            .zip(branch.iter())
            .any(|(mine, theirs)| mine.shares_no_value_with(theirs, cx));
        if !disjoint {
            break (branch, rest);
        }
        remaining = rest;
    };
    for (position, (mine, theirs)) in components.iter().zip(branch.iter()).enumerate() {
        let here = mine.is_subtype_rec(theirs, cx, assumptions);
        if here == Relation::Holds {
            continue;
        }
        // The component this branch does not cover, narrowed by what the
        // branch takes away, with the rest of the tuple as it was. Built by
        // mapping rather than by writing at an index: the position comes from
        // the same enumeration as the component, so an index write cannot go
        // out of range and cannot be seen not to.
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
        match (here, product_subtype(&narrowed, rest, cx, assumptions)) {
            (_, Relation::Holds) => {}
            (Relation::Fails, Relation::Fails) => return Relation::Fails,
            _ => return Relation::Unknown,
        }
    }
    Relation::Holds
}

/// Whether a fixed-arity sequence is covered by the sequence branches of a union.
///
/// A value that splits across branches -- `tuple[int|str, int]` covered by
/// `tuple[int, int] | tuple[str, int]` -- lands in no single branch, so the rule
/// that tries each branch alone cannot see it. Branches of another container or
/// another arity share no value with `self` by shape, so they drop out rather
/// than blocking the decomposition.
///
/// A refutation from the product rule is a refutation of the union only where
/// every member it set aside holds none of the subject's values. A sequence of
/// another kind or of another fixed arity holds none; one of this kind with a
/// repeated tail, a class, a complement or a reference may hold the value the
/// rule found outside the branches, so where one is set aside the refutation is
/// dropped and only a proof is carried.
pub(super) fn seq_splits_across_union(
    schema: &Schema,
    members: &[Schema],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    let Schema::Seq { container, shape } = schema else {
        return Relation::Unknown;
    };
    let Some(components) = fixed_components(shape) else {
        return Relation::Unknown;
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
    let set_aside_hold_none = members.iter().all(|member| match member {
        Schema::Seq {
            container: their_kind,
            shape: their_shape,
        } => their_kind != container || their_shape.tail.is_none(),
        _ => false,
    });
    // With no branch to split over, the rule can only answer from the product's
    // own emptiness, which the empty-subject bound asks anyway -- except where
    // nothing set aside may hold a value, and then `Phi(P, [])` is exact: a
    // product with a value is outside a union of none of its shape.
    if branches.is_empty() && !set_aside_hold_none {
        return Relation::Unknown;
    }
    let branches: Vec<&[Schema]> = branches.iter().map(Vec::as_slice).collect();
    let answer = product_subtype(&components, &branches, cx, assumptions);
    if set_aside_hold_none {
        answer
    } else {
        answer.proof_only()
    }
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
