//! The laws the three-valued answers' combinators are documented to obey.
//!
//! Each answer has three values, so every list up to a length is a finite set,
//! and these laws are checked over all of it rather than over a draw.

use std::cell::Cell;

use super::{Relation, Verdict};

const RELATIONS: [Relation; 3] = [Relation::Holds, Relation::Fails, Relation::Unknown];
const VERDICTS: [Verdict; 3] = [Verdict::Empty, Verdict::Inhabited, Verdict::Unknown];

/// Every list of `values` up to five long, the empty list included.
fn lists<T: Copy>(values: [T; 3]) -> Vec<Vec<T>> {
    let mut all = vec![Vec::new()];
    let mut last = vec![Vec::new()];
    for _ in 0..5 {
        last = last
            .iter()
            .flat_map(|list: &Vec<T>| {
                values
                    .iter()
                    .map(move |value| [list.as_slice(), &[*value]].concat())
            })
            .collect();
        all.extend(last.iter().cloned());
    }
    all
}

/// The negation of an answer: a proof and a refutation trade places, and a
/// decline stays one.
fn not(answer: Relation) -> Relation {
    match answer {
        Relation::Holds => Relation::Fails,
        Relation::Fails => Relation::Holds,
        Relation::Unknown => Relation::Unknown,
    }
}

/// An answer's place in the truth order: a refutation, then a decline, then a
/// proof.
fn rank(answer: Relation) -> u8 {
    match answer {
        Relation::Fails => 0,
        Relation::Unknown => 1,
        Relation::Holds => 2,
    }
}

/// Whether a set holds a value, read as an answer: proved inhabited is a
/// proof, proved empty a refutation.
fn inhabited(verdict: Verdict) -> Relation {
    match verdict {
        Verdict::Inhabited => Relation::Holds,
        Verdict::Empty => Relation::Fails,
        Verdict::Unknown => Relation::Unknown,
    }
}

// THEORY: the-decision-has-three-answers
/// `all` and `any` are Kleene's strong conjunction and disjunction: the least
/// and the greatest answer in the truth order, which is what makes their
/// answer the same whatever order the items come in, as `all`'s doc argues.
#[test]
fn the_relation_folds_are_kleenes_strong_connectives() {
    for items in lists(RELATIONS) {
        let least = items.iter().copied().min_by_key(|answer| rank(*answer));
        let greatest = items.iter().copied().max_by_key(|answer| rank(*answer));
        assert_eq!(
            Relation::all(items.clone()),
            least.unwrap_or(Relation::Holds),
            "{items:?}"
        );
        assert_eq!(
            Relation::any(items.clone()),
            greatest.unwrap_or(Relation::Fails),
            "{items:?}"
        );
        let reversed: Vec<Relation> = items.iter().rev().copied().collect();
        assert_eq!(
            Relation::all(reversed.clone()),
            Relation::all(items.clone()),
            "{items:?}"
        );
        assert_eq!(
            Relation::any(reversed),
            Relation::any(items.clone()),
            "{items:?}"
        );
        // `any` is the dual of `all`, as each one's doc says.
        assert_eq!(
            Relation::any(items.clone()),
            not(Relation::all(items.iter().copied().map(not))),
            "{items:?}"
        );
    }
}

// THEORY: the-decision-has-three-answers
/// The emptiness folds are the relation folds read through inhabitation, so
/// the laws above hold of them; and an inclusion's answer is the difference's
/// inhabitation negated, `a <= b` exactly when `a & ~b` holds no value.
#[test]
fn the_verdict_folds_are_the_relation_folds_read_as_inhabitation() {
    for parts in lists(VERDICTS) {
        let read: Vec<Relation> = parts.iter().copied().map(inhabited).collect();
        assert_eq!(
            inhabited(Verdict::every(parts.iter().copied())),
            Relation::all(read.clone())
        );
        assert_eq!(
            inhabited(Verdict::any(parts.iter().copied())),
            Relation::any(read)
        );
    }
    for verdict in VERDICTS {
        assert_eq!(Relation::of_difference(verdict), not(inhabited(verdict)));
    }
}

/// `and` chains steps and agrees with the conjunction on every proof, as the
/// decision page says; a step is reached only after a proof, and the second
/// decider only after a decline.
#[test]
fn a_chained_step_agrees_with_the_conjunction_on_every_proof() {
    for first in RELATIONS {
        for second in RELATIONS {
            let asked = Cell::new(false);
            let ask = || {
                asked.set(true);
                second
            };
            let chained = first.and(ask);
            assert_eq!(chained.holds(), Relation::all([first, second]).holds());
            assert_eq!(
                asked.get(),
                first == Relation::Holds,
                "{first:?} then {second:?}"
            );

            asked.set(false);
            let decided = first.or_else(ask);
            assert_eq!(
                asked.get(),
                first == Relation::Unknown,
                "{first:?} or {second:?}"
            );
            assert_eq!(
                decided,
                if first == Relation::Unknown {
                    second
                } else {
                    first
                }
            );
        }
    }
}

/// Each projection keeps one direction and drops the other, and the two are
/// one projection read through negation. The two-valued surface is a
/// projection too: only a proof is `true`, and only a proved emptiness empty.
#[test]
fn the_projections_keep_one_direction_each() {
    for verdict in VERDICTS {
        assert_eq!(verdict.is_empty(), verdict == Verdict::Empty);
    }
    for answer in RELATIONS {
        assert_eq!(answer.holds(), answer == Relation::Holds);
        let proof = answer.proof_only();
        assert_eq!(proof.holds(), answer.holds());
        assert_ne!(proof, Relation::Fails);
        assert_eq!(proof.proof_only(), proof);
        let refutation = answer.refutation_only();
        assert_eq!(refutation == Relation::Fails, answer == Relation::Fails);
        assert_ne!(refutation, Relation::Holds);
        assert_eq!(refutation, not(not(answer).proof_only()));
    }
}
