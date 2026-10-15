//! The walk's memo, driven through the real walk: it changes no answer and no
//! report, it walks a shared definition once per level, it keeps nothing over
//! a predicate, and a value nested past the bounds is refused by its stop, at
//! a cost the bounds fix.

use std::sync::Arc;

use pyo3::types::{PyDict, PyList, PyModule, PyTuple};
use valgebra_core::{Constraint, DefIx, MapClause, PredIx, SeqShape};

use super::*;
use crate::check::memo::{Answer, Memo, MemoPlan, PLANNED_DEFINITIONS, Room};
use crate::check::{WalkState, build_index};
use crate::codes::{RECURSION_LIMIT, RECURSION_LOOP};

/// What one walk answered, reported and met.
struct Walked {
    admitted: bool,
    violations: Vec<Violation>,
    entered: usize,
    stopped: Option<Code>,
    used: bool,
}

/// Walk `value` against `schema` in `mode`, with the memo the validator plans
/// or, where `remember` is false, with none.
fn walk(
    py: Python<'_>,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    at: (&[Py<PyAny>], &[Schema]),
    mode: WalkMode,
    remember: bool,
) -> Walked {
    let state = WalkState::new();
    walk_in(py, &state, schema, value, at, mode, remember)
}

/// [`walk`] in a state a test has placed: one whose trail holds definitions
/// already, so a shallow value meets the unfolding bound.
fn walk_in(
    py: Python<'_>,
    state: &WalkState,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    (pool, defs): (&[Py<PyAny>], &[Schema]),
    mode: WalkMode,
    remember: bool,
) -> Walked {
    let mut index = build_index(py, schema, defs, pool);
    if !remember {
        index.memo = MemoPlan::default();
    }
    let ctx = Ctx {
        pool,
        defs,
        index: &index,
        state,
        mode,
    };
    let mut violations = Vec::new();
    let admitted = member(
        schema,
        &Value::Py(value),
        &mut Frame::new(&mut Vec::new(), &mut violations, ctx),
    );
    Walked {
        admitted,
        violations,
        entered: state.memo.count(),
        stopped: state.signals.stopped.get(),
        used: state.memo.in_use(),
    }
}

fn reference() -> Schema {
    Schema::Ref(DefIx::new(0))
}

fn list_of(element: Schema) -> Schema {
    Schema::list(SeqShape::homogeneous(element))
}

fn union_of(members: Vec<Schema>) -> Schema {
    Schema::Union(members.into())
}

/// `T = int | list[str | T] | list[T]`, in the order the constructors write
/// it: two list branches read a list's element as `T`, and the first reaches
/// it a walk level deeper than the second.
fn two_lists() -> Vec<Schema> {
    vec![union_of(vec![
        Schema::Int,
        list_of(union_of(vec![Schema::Str, reference()])),
        list_of(reference()),
    ])]
}

/// `T = int | tuple[T, int] | tuple[T, str]`: two tuple branches read the
/// first element as `T`, and a value the second admits is walked by both.
fn two_tuples() -> Vec<Schema> {
    vec![union_of(vec![
        Schema::Int,
        Schema::tuple(SeqShape::fixed([reference(), Schema::Int])),
        Schema::tuple(SeqShape::fixed([reference(), Schema::Str])),
    ])]
}

/// `T = int | (list[T] & list[T | str])`: a meet's two members read one
/// element as `T`.
fn a_meet() -> Vec<Schema> {
    vec![union_of(vec![
        Schema::Int,
        Schema::Intersection(
            vec![
                list_of(reference()),
                list_of(union_of(vec![reference(), Schema::Str])),
            ]
            .into(),
        ),
    ])]
}

/// `T = int | {str: T, ~int: T | bytes}`: two clauses read one entry's value
/// as `T`.
fn two_clauses() -> Vec<Schema> {
    vec![union_of(vec![
        Schema::Int,
        Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![
                MapClause {
                    key: Schema::Str,
                    value: reference(),
                },
                MapClause {
                    key: Schema::Int.complement(),
                    value: union_of(vec![reference(), Schema::Bytes]),
                },
            ]
            .into(),
        },
    ])]
}

/// `T = {~int: U, str: T | bytes}` with `U = T`: two clauses read a string
/// key's value as `T`, the first through a second definition.
fn through_an_alias() -> Vec<Schema> {
    vec![
        Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![
                MapClause {
                    key: Schema::Int.complement(),
                    value: Schema::Ref(DefIx::new(1)),
                },
                MapClause {
                    key: Schema::Str,
                    value: union_of(vec![Schema::Bytes, reference()]),
                },
            ]
            .into(),
        },
        reference(),
    ]
}

/// `leaf` nested `depth` times by `wrap`.
fn nested<'py>(
    leaf: Bound<'py, PyAny>,
    depth: usize,
    wrap: impl Fn(Bound<'py, PyAny>) -> Bound<'py, PyAny>,
) -> Bound<'py, PyAny> {
    (0..depth).fold(leaf, |inner, _| wrap(inner))
}

fn in_a_list(inner: Bound<'_, PyAny>) -> Bound<'_, PyAny> {
    PyList::new(inner.py(), [inner]).expect("a list").into_any()
}

fn beside_text(inner: Bound<'_, PyAny>) -> Bound<'_, PyAny> {
    let text = "x".into_pyobject(inner.py()).expect("a str").into_any();
    PyTuple::new(inner.py(), [inner, text])
        .expect("a tuple")
        .into_any()
}

fn under_a_key(inner: Bound<'_, PyAny>) -> Bound<'_, PyAny> {
    let dict = PyDict::new(inner.py());
    dict.set_item("k", inner).expect("set_item");
    dict.into_any()
}

/// One way of nesting a value a level deeper.
type Wrap<'py> = fn(Bound<'py, PyAny>) -> Bound<'py, PyAny>;

/// Each shape, by name, with the way its value nests.
fn shapes(py: Python<'_>) -> Vec<(&'static str, Vec<Schema>, Wrap<'_>)> {
    let _ = py;
    vec![
        ("two lists", two_lists(), in_a_list),
        ("two tuples", two_tuples(), beside_text),
        ("a meet", a_meet(), in_a_list),
        ("two clauses", two_clauses(), under_a_key),
        (
            "two clauses, one through an alias",
            through_an_alias(),
            under_a_key,
        ),
    ]
}

/// The memo answers what the walk without it answers, and reports what it
/// reports, path for path, in every mode, at depths inside the bounds, where
/// that walk decides every value.
///
/// The claim the memo rests on is that a definition walked at a value answers
/// the same wherever the walk reaches it, which the comparison reads off the
/// walk itself: a violation reported at another path, a report from one mode
/// found by another, or an answer kept from a walk that stopped would each
/// show here as a difference.
#[test]
fn the_memo_changes_no_answer_and_no_report() {
    Python::attach(|py| {
        let leaves = [
            0i64.into_pyobject(py).expect("an int").into_any(),
            py.None().into_bound(py),
        ];
        for (name, defs, wrap) in shapes(py) {
            for depth in 0..=7 {
                for leaf in &leaves {
                    let chain = nested(leaf.clone(), depth, wrap);
                    // One object at two paths: a report the memo found at the
                    // first is said again at the second.
                    let twice = PyList::new(py, [chain.clone(), chain.clone()])
                        .expect("a list")
                        .into_any();
                    for value in [chain, wrap(twice)] {
                        for mode in [WalkMode::Fast, WalkMode::Explain, WalkMode::ExplainFailFast] {
                            let at = (&[][..], &defs[..]);
                            let kept = walk(py, &reference(), &value, at, mode, true);
                            let plain = walk(py, &reference(), &value, at, mode, false);
                            let case = format!("{name}, depth {depth}, {leaf}, {mode:?}");
                            assert_eq!(kept.admitted, plain.admitted, "{case}");
                            assert_eq!(kept.stopped, plain.stopped, "{case}");
                            assert_eq!(kept.violations.len(), plain.violations.len(), "{case}");
                            for (one, other) in kept.violations.iter().zip(&plain.violations) {
                                assert_eq!(one.code, other.code, "{case}");
                                assert_eq!(one.path, other.path, "{case}");
                                assert_eq!(one.expected, other.expected, "{case}");
                                assert_eq!(one.value_summary, other.value_summary, "{case}");
                            }
                        }
                    }
                }
            }
        }
    });
}

/// A definition two branches read at one value is walked once per level of
/// the value: twice the depth gives twice the entries and a few more, where a
/// walk that grew with the square of the depth would give four times, and the
/// walk without the memo doubles with each level.
///
/// Small depths, so a walk without its memo fails here rather than runs
/// for ever: the tests below that read a deep value call this first.
fn assert_linear_in_the_depth(py: Python<'_>) {
    let none = py.None().into_bound(py);
    let zero = 0i64.into_pyobject(py).expect("an int").into_any();
    for (name, defs, wrap) in shapes(py) {
        for leaf in [&none, &zero] {
            for mode in [WalkMode::Fast, WalkMode::Explain, WalkMode::ExplainFailFast] {
                // A meet's full report holds each member's failures, and both
                // members' hold the element's: the report itself doubles with
                // each level, which no memo can shrink.
                if name == "a meet" && mode == WalkMode::Explain && leaf.is_none() {
                    continue;
                }
                let walked = |depth: usize| {
                    let value = nested(leaf.clone(), depth, wrap);
                    let at = (&[][..], &defs[..]);
                    walk(py, &reference(), &value, at, mode, true)
                };
                let (shallow, deep) = (walked(6), walked(12));
                let case = format!("{name}, {leaf}, {mode:?}");
                // A walk enters a definition at each level at least, so a count
                // below the depth is one that stopped counting.
                assert!(shallow.entered >= 6, "{case}: {} entries", shallow.entered);
                assert!(
                    deep.entered <= 2 * shallow.entered + 16,
                    "{case}: {} entries at 6, {} at 12",
                    shallow.entered,
                    deep.entered
                );
            }
        }
    }
}

/// A walk that stopped is kept with the room it had, and a walk with no more
/// room finds it: a chain ending in a list that holds itself stops at the
/// loop on every path, and is walked a bounded number of times a level where
/// a walk that kept no stop would double with each.
///
/// Shallow too, so a memo that finds no stop fails here: the tests below that
/// read a value at the bounds call this first.
fn assert_a_stop_is_found_again(py: Python<'_>) {
    let defs = two_lists();
    let at = (&[][..], &defs[..]);
    let looped = PyList::empty(py);
    looped.append(&looped).expect("append");
    let value = nested(looped.into_any(), 10, in_a_list);
    for mode in [WalkMode::Fast, WalkMode::Explain] {
        let walked = walk(py, &reference(), &value, at, mode, true);
        assert!(!walked.admitted);
        assert_eq!(walked.stopped, Some(RECURSION_LOOP), "{mode:?}");
        assert!(
            walked.entered < 1_000,
            "{mode:?}: {} entries",
            walked.entered
        );
    }
}

#[test]
fn a_shared_definition_is_entered_a_bounded_number_of_times_per_level() {
    Python::attach(|py| {
        assert_linear_in_the_depth(py);
        // And the shape is one the walk without the memo doubles on, so the
        // bound is the memo's. Eight levels show the doubling: a mutation that
        // makes the deciding walk explain doubles the plain walk again at each
        // level, and twelve would run past the sweep's timeout.
        let value = nested(py.None().into_bound(py), 8, in_a_list);
        let defs = two_lists();
        let at = (&[][..], &defs[..]);
        let plain = walk(py, &reference(), &value, at, WalkMode::Fast, false);
        assert!(plain.entered >= 1 << 8, "{} entries", plain.entered);
        // Every branch walks a leaf no branch admits, so a later one arms the
        // memo the plan gives it, and nothing arms one with no plan.
        assert!(walk(py, &reference(), &value, at, WalkMode::Fast, true).used);
        assert!(!plain.used);
    });
}

#[test]
fn a_value_that_holds_itself_is_refused_by_its_loop_at_a_bounded_cost() {
    Python::attach(assert_a_stop_is_found_again);
}

/// A stop is kept with the room its walk had, and answers only a walk with no
/// more: the constructors write the deeper branch first, so a chain the
/// shallower one admits, nested where the deeper one stops, is reached by a
/// walk that stopped before the one that admits it.
#[test]
fn a_value_one_branch_admits_where_another_stops_is_admitted() {
    Python::attach(|py| {
        assert_linear_in_the_depth(py);
        assert_a_stop_is_found_again(py);
        let defs = two_lists();
        let at = (&[][..], &defs[..]);
        let zero = 0i64.into_pyobject(py).expect("an int").into_any();
        let value = nested(zero, 110, in_a_list);
        for mode in [WalkMode::Fast, WalkMode::Explain] {
            let walked = walk(py, &reference(), &value, at, mode, true);
            assert!(walked.admitted, "{mode:?}");
            assert!(walked.violations.is_empty(), "{mode:?}");
        }
    });
}

/// A room is two counts, and a walk with more of either can admit what one
/// with less refused: the levels open, and the definitions unfolded.
///
/// `T = {~int: U, str: T | bytes}` with `U = T`: both clauses read a string
/// key, and each costs a level of the value three walk levels, but the first
/// unfolds two definitions a level and the second one. With the trail nine
/// definitions short of its bound, a dict nested six deep is past the bound
/// along the first and inside it along the second, so the first clause's walk
/// of a level stops, and the second clause reaches the same pair with as many
/// levels and fewer definitions unfolded: a room that compared levels alone
/// found the stop there and refused the value. Nested nine deep, it is past
/// the bound along both. The memo answers as the walk without it does.
#[test]
fn a_value_one_path_admits_with_definitions_to_spare_is_admitted() {
    Python::attach(|py| {
        let defs = through_an_alias();
        let at = (&[][..], &defs[..]);
        for (depth, admitted) in [(6, true), (9, false)] {
            let value = nested(PyDict::new(py).into_any(), depth, under_a_key);
            for mode in [WalkMode::Fast, WalkMode::Explain] {
                for remember in [true, false] {
                    let state = WalkState::new();
                    // Pairs no walk enters: no definition has that index.
                    for open in 0..MAX_RECURSION_DEPTH - 9 {
                        let opened = state.guard.borrow_mut().enter((open, usize::MAX));
                        assert!(matches!(opened, Entered::Open));
                    }
                    let walked = walk_in(py, &state, &reference(), &value, at, mode, remember);
                    let case = format!("depth {depth}, {mode:?}, memo {remember}");
                    assert_eq!(walked.admitted, admitted, "{case}");
                    assert_eq!(walked.stopped, Some(RECURSION_LIMIT), "{case}");
                }
            }
        }
    });
}

/// A node whose members reach no definition in common arms nothing, though a
/// member before the one walked entered a definition and a node elsewhere in
/// the schema does share one: what one member kept no other could find.
#[test]
fn a_union_whose_branches_share_no_definition_arms_nothing() {
    Python::attach(|py| {
        // `A = int | list[A] | list[str | A]`, which shares `A` between its
        // list branches, and `B = None | dict[str, B]`; the root is
        // `list[A] | list[B]`, whose branches share nothing.
        let a = Schema::Ref(DefIx::new(0));
        let b = Schema::Ref(DefIx::new(1));
        let defs = vec![
            union_of(vec![
                Schema::Int,
                list_of(union_of(vec![Schema::Str, a.clone()])),
                list_of(a.clone()),
            ]),
            union_of(vec![
                Schema::NoneType,
                Schema::mapping(MapClause {
                    key: Schema::Str,
                    value: b.clone(),
                }),
            ]),
        ];
        let root = union_of(vec![list_of(a), list_of(b)]);
        let index = build_index(py, &root, &defs, &[]);
        let Schema::Union(members) = &root else {
            unreachable!("the root is a union");
        };
        assert!(!index.memo.shares(members.as_ptr() as usize));
        let tree = nested(py.None().into_bound(py), 3, under_a_key);
        let value = PyList::new(py, [tree]).expect("a list").into_any();
        for mode in [WalkMode::Fast, WalkMode::Explain, WalkMode::ExplainFailFast] {
            let walked = walk(py, &root, &value, (&[], &defs), mode, true);
            assert!(walked.admitted, "{mode:?}");
            assert!(
                !walked.used,
                "{mode:?}: a union sharing nothing armed the memo"
            );
        }
    });
}

/// A value nested past the bounds is refused by the bound, in every mode, and
/// the entries its walk makes stop growing with its depth.
///
/// Past the bounds every walk stops, and each is kept with its room; a value
/// is reached with a different room by each path to it, and walked once for
/// each, which the bounds cap. An explaining walk says a stopped report again
/// where its deciding walk stops too, rather than explaining the levels below
/// once per room.
#[test]
fn a_value_past_the_bounds_is_refused_by_its_stop_at_a_bounded_cost() {
    Python::attach(|py| {
        assert_linear_in_the_depth(py);
        assert_a_stop_is_found_again(py);
        let defs = two_lists();
        let none = py.None().into_bound(py);
        let mut entries = Vec::new();
        for depth in [300, 1_000] {
            let value = nested(none.clone(), depth, in_a_list);
            let at = (&[][..], &defs[..]);
            let mut at_depth = Vec::new();
            for mode in [WalkMode::Fast, WalkMode::Explain, WalkMode::ExplainFailFast] {
                let walked = walk(py, &reference(), &value, at, mode, true);
                assert!(!walked.admitted);
                assert_eq!(walked.stopped, Some(RECURSION_LIMIT), "{mode:?}");
                if mode.explains() {
                    assert_eq!(
                        walked.violations.first().map(|v| v.code),
                        Some(RECURSION_LIMIT.as_str()),
                        "{mode:?}"
                    );
                }
                assert!(
                    walked.entered < 20_000,
                    "{mode:?}: {} entries",
                    walked.entered
                );
                at_depth.push(walked.entered);
            }
            entries.push(at_depth);
        }
        assert_eq!(entries[0], entries[1], "the cost grew past the bounds");
    });
}

/// An answer the memo finds is said again with the stop its walk met, so a
/// complement or a kept walk above reads it as that walk did; and a report it
/// finds is said again at the path the walk finding it is at.
#[test]
fn an_answer_found_carries_its_stop_and_a_report_its_own_path() {
    Python::attach(|py| {
        let defs = two_lists();
        let index = build_index(py, &reference(), &defs, &[]);
        let value = PyList::empty(py).into_any();
        let key = (value.as_ptr() as usize, 0);
        let anywhere = Room {
            levels: 0,
            unfolded: 0,
        };
        let stopped = Answer {
            admitted: false,
            stopped: Some((RECURSION_LIMIT, anywhere)),
        };
        let find = |mode: WalkMode, state: &WalkState, path: &mut Vec<PathSegment>| {
            let ctx = Ctx {
                pool: &[],
                defs: &defs,
                index: &index,
                state,
                mode,
            };
            let mut out = Vec::new();
            let admitted = member(
                &reference(),
                &Value::Py(&value),
                &mut Frame::new(path, &mut out, ctx),
            );
            (admitted, out)
        };

        let state = WalkState::new();
        state.keep(|kept| kept.keep_decided(key, &Value::Py(&value), stopped));
        state.memo.armed(|| ());
        let (admitted, _) = find(WalkMode::Fast, &state, &mut Vec::new());
        assert!(!admitted);
        assert_eq!(state.signals.stopped.get(), Some(RECURSION_LIMIT));

        let state = WalkState::new();
        let kept = Violation {
            code: RECURSION_LIMIT.as_str(),
            path: vec![PathSegment::Index(7), PathSegment::Index(0)],
            expected: "a shallower value".to_owned(),
            value_summary: "[]".to_owned(),
        };
        state.keep(|memo| memo.keep_explained(key, &Value::Py(&value), stopped, 1, &[kept]));
        state.memo.armed(|| ());
        let mut path = vec![PathSegment::Index(3), PathSegment::Index(4)];
        let (admitted, out) = find(WalkMode::Explain, &state, &mut path);
        assert!(!admitted);
        assert_eq!(state.signals.stopped.get(), Some(RECURSION_LIMIT));
        let paths: Vec<_> = out.iter().map(|v| v.path.clone()).collect();
        assert_eq!(
            paths,
            vec![vec![
                PathSegment::Index(3),
                PathSegment::Index(4),
                PathSegment::Index(0)
            ]]
        );
    });
}

/// A definition whose body reaches a predicate is not remembered, so the
/// predicate runs at every occurrence the walk reaches, as the refinements
/// page promises: as many calls with the plan as without it.
#[test]
fn a_definition_reaching_a_predicate_is_walked_at_each_occurrence() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"calls = 0\ndef count(x):\n    global calls\n    calls += 1\n    return True\n",
            c"memo_counter.py",
            c"memo_counter",
        )
        .expect("the module compiles");
        let pool = vec![module.getattr("count").expect("count").unbind()];
        let counted = Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
        };
        // `T = U | tuple[T, int] | tuple[T, str]`, and `U` carries the
        // predicate: `T` reaches it through `U`, which is what a definition
        // reaching one means, and a value the second tuple admits walks the
        // first element under both.
        let defs = vec![
            union_of(vec![
                Schema::Ref(DefIx::new(1)),
                Schema::tuple(SeqShape::fixed([reference(), Schema::Int])),
                Schema::tuple(SeqShape::fixed([reference(), Schema::Str])),
            ]),
            counted,
        ];
        let index = build_index(py, &reference(), &defs, &pool);
        assert!(!index.memo.remembers(DefIx::new(0)));
        assert!(!index.memo.remembers(DefIx::new(1)));
        let calls = || -> i64 {
            module
                .getattr("calls")
                .and_then(|calls| calls.extract())
                .expect("calls")
        };
        let value = nested(
            0i64.into_pyobject(py).expect("an int").into_any(),
            9,
            beside_text,
        );
        let at = (&pool[..], &defs[..]);
        assert!(walk(py, &reference(), &value, at, WalkMode::Fast, true).admitted);
        let with_plan = calls();
        walk(py, &reference(), &value, at, WalkMode::Fast, false);
        assert_eq!(calls() - with_plan, with_plan);
        assert!(with_plan >= 1 << 9, "{with_plan} calls");
    });
}

/// Every definition a validator can hold has a bit in the plan
/// (`PLANNED_DEFINITIONS`): the last of a table at the validator's limit is
/// remembered as the first is, each naming the next.
#[test]
fn the_last_definition_a_validator_can_hold_is_planned() {
    Python::attach(|py| {
        let count = crate::validator::MAX_DEFINITIONS;
        assert!(count <= PLANNED_DEFINITIONS);
        let defs: Vec<Schema> = (0..count)
            .map(|at| {
                let next = Schema::Ref(DefIx::new((at + 1) % count));
                union_of(vec![
                    Schema::Int,
                    list_of(union_of(vec![Schema::Str, next.clone()])),
                    list_of(next),
                ])
            })
            .collect();
        let index = build_index(py, &reference(), &defs, &[]);
        assert!(index.memo.remembers(DefIx::new(0)));
        assert!(index.memo.remembers(DefIx::new(count - 1)));
        assert!(!index.memo.remembers(DefIx::new(PLANNED_DEFINITIONS)));
    });
}

/// What a definition reaches is closed under naming: one that names another
/// that names a predicate's reaches the predicate two definitions away, and
/// is not remembered; one that names none of them is. And a node whose members
/// reach a definition in common only where it is not remembered arms nothing,
/// since nothing a member walks there is kept.
#[test]
fn a_definition_is_planned_by_everything_it_reaches() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"def holds(x):\n    return True\n",
            c"memo_plan.py",
            c"memo_plan",
        )
        .expect("the module compiles");
        let pool = vec![module.getattr("holds").expect("holds").unbind()];
        let at = |ix: usize| Schema::Ref(DefIx::new(ix));
        // 0 names 1, 1 names 2, 2 carries the predicate; 3 names none of them.
        let defs = vec![
            union_of(vec![Schema::Int, list_of(at(1))]),
            union_of(vec![Schema::Str, list_of(at(2))]),
            Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
            },
            union_of(vec![Schema::Int, list_of(at(3))]),
        ];
        // The root's two branches share 0 alone, which is not remembered.
        let root = union_of(vec![
            list_of(at(0)),
            list_of(union_of(vec![at(0), Schema::Str])),
        ]);
        let index = build_index(py, &root, &defs, &pool);
        for (def, remembered) in [(0, false), (1, false), (2, false), (3, true)] {
            assert_eq!(index.memo.remembers(DefIx::new(def)), remembered, "{def}");
        }
        let Schema::Union(members) = &root else {
            unreachable!("the root is a union");
        };
        assert!(!index.memo.shares(members.as_ptr() as usize));
    });
}

/// A class whose metaclass is not `type` may answer `isinstance` itself, so a
/// definition reaching one is not remembered; one reaching a plain class is.
#[test]
fn a_definition_reaching_a_class_with_a_metaclass_of_its_own_is_not_remembered() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"class Meta(type):\n    pass\nclass Hooked(metaclass=Meta):\n    pass\nclass Plain:\n    pass\n",
            c"memo_classes.py",
            c"memo_classes",
        )
        .expect("the module compiles");
        for (class, remembered) in [("Hooked", false), ("Plain", true)] {
            let pool = vec![module.getattr(class).expect("the class").unbind()];
            let defs = vec![union_of(vec![
                Schema::Instance(valgebra_core::ClassIx::new(0)),
                list_of(reference()),
                list_of(union_of(vec![reference(), Schema::Str])),
            ])];
            let index = build_index(py, &reference(), &defs, &pool);
            assert_eq!(index.memo.remembers(DefIx::new(0)), remembered, "{class}");
        }
    });
}

/// A branch refused by its kind enters no definition, so the branch after it
/// arms nothing: a JSON schema's list and object branches share their element
/// definition and never read one value twice.
#[test]
fn a_branch_refused_by_its_kind_arms_nothing() {
    Python::attach(|py| {
        let defs = vec![union_of(vec![
            Schema::NoneType,
            Schema::Int,
            Schema::Str,
            list_of(reference()),
            Schema::mapping(MapClause {
                key: Schema::Str,
                value: reference(),
            }),
        ])];
        let index = build_index(py, &reference(), &defs, &[]);
        let Schema::Union(members) = &defs[0] else {
            unreachable!("the definition is a union");
        };
        assert!(index.memo.shares(members.as_ptr() as usize));
        let value = nested(py.None().into_bound(py), 20, under_a_key);
        let at = (&[][..], &defs[..]);
        for mode in [WalkMode::Fast, WalkMode::Explain, WalkMode::ExplainFailFast] {
            let walked = walk(py, &reference(), &value, at, mode, true);
            assert!(walked.admitted, "{mode:?}");
            assert!(
                !walked.used,
                "{mode:?}: a branch refused by its kind armed the memo"
            );
        }
    });
}

/// A meet walks its first member unarmed: what that member would keep only a
/// later member could find, and a deciding walk the first member refuses ends
/// there.
#[test]
fn a_meet_refused_by_its_first_member_arms_nothing() {
    Python::attach(|py| {
        let defs = a_meet();
        let value = nested(py.None().into_bound(py), 6, in_a_list);
        let walked = walk(py, &reference(), &value, (&[], &defs), WalkMode::Fast, true);
        assert!(!walked.admitted);
        assert!(!walked.used, "the meet's first member armed the memo");
    });
}

/// The memo holds a reference to each Python value it keys, and gives it back
/// when the call's state is dropped: an element freed while the walk runs could otherwise
/// give its address to a value the memo would answer for.
#[test]
fn the_memo_holds_each_value_it_keys_for_the_length_of_the_call() {
    Python::attach(|py| {
        let value = PyList::empty(py).into_any();
        let held = || -> i64 {
            py.import("sys")
                .and_then(|sys| sys.getattr("getrefcount"))
                .and_then(|count| count.call1((&value,)))
                .and_then(|count| count.extract())
                .expect("a reference count")
        };
        let before = held();
        let state = WalkState::new();
        let answer = Answer {
            admitted: false,
            stopped: None,
        };
        let key = (value.as_ptr() as usize, 0);
        state.keep(|kept| kept.keep_decided(key, &Value::Py(&value), answer));
        assert_eq!(held(), before + 1);
        drop(state);
        assert_eq!(held(), before);
    });
}

/// The memo's word: a definition entered is counted, and answers whether a
/// node has armed the memo yet; and a member armed inside another leaves the
/// outer one armed when it ends.
#[test]
fn a_member_armed_inside_another_leaves_it_armed() {
    let memo = Memo::default();
    assert!(!memo.enter(), "nothing has armed the memo");
    assert!(!memo.is_armed());
    memo.armed(|| {
        assert!(memo.is_armed());
        memo.armed(|| assert!(memo.is_armed()));
        assert!(memo.is_armed(), "the inner member disarmed the outer");
    });
    assert!(!memo.is_armed());
    assert!(memo.enter(), "a node armed the memo");
    assert_eq!(memo.count(), 2);
}

/// A stop is kept with the room its walk began with, and answers a later walk
/// with no more room on either count. Two walks can each have more of one
/// count than the other, so the room of each is kept, and either answers.
#[test]
fn a_stop_answers_a_walk_with_no_more_room_on_either_count() {
    Python::attach(|py| {
        let value = PyList::empty(py).into_any();
        let key = (value.as_ptr() as usize, 0);
        let room = |levels, unfolded| Room { levels, unfolded };
        let state = WalkState::new();
        for kept_at in [room(10, 4), room(6, 8)] {
            let stopped = Answer {
                admitted: false,
                stopped: Some((RECURSION_LIMIT, kept_at)),
            };
            state.keep(|kept| kept.keep_decided(key, &Value::Py(&value), stopped));
        }
        for (levels, unfolded, answered) in [
            (10, 4, true),
            (6, 8, true),
            (11, 5, true),
            (8, 8, true),
            // More levels left than the first walk had, or more definitions
            // left than the second, or both.
            (9, 4, false),
            (10, 3, false),
            (7, 7, false),
        ] {
            let found = state.recall(|kept| kept.decided(key, room(levels, unfolded)));
            assert_eq!(
                found.and_then(|answer| answer.stop()),
                answered.then_some(RECURSION_LIMIT),
                "{levels} levels, {unfolded} unfolded"
            );
        }
    });
}

/// An explaining walk's answer is found as a deciding walk's is: a refusal
/// that stopped by a walk with no more room than it had, and an admission by
/// any, though its walk met a stop on the way. A refusal that stopped is the
/// one a walk with more room asks its deciding walk about.
#[test]
fn an_explanation_that_stopped_answers_a_walk_with_no_more_room() {
    Python::attach(|py| {
        let value = PyList::empty(py).into_any();
        let key = (value.as_ptr() as usize, 0);
        let kept_at = Room {
            levels: 10,
            unfolded: 4,
        };
        let deeper = Room {
            levels: 12,
            unfolded: 5,
        };
        let shallower = Room {
            levels: 9,
            unfolded: 4,
        };
        for (admitted, stopped, by_the_shallower, asked_about) in [
            (false, true, false, true),
            (true, true, true, false),
            (false, false, true, false),
        ] {
            let answer = Answer {
                admitted,
                stopped: stopped.then_some((RECURSION_LIMIT, kept_at)),
            };
            let state = WalkState::new();
            state.keep(|kept| kept.keep_explained(key, &Value::Py(&value), answer, 0, &[]));
            let found = |at: Room| {
                state
                    .recall(|kept| kept.explained(key, at, &[], &mut Vec::new()))
                    .is_some()
            };
            let case = format!("admitted {admitted}, stopped {stopped}");
            assert!(found(deeper), "{case}");
            assert_eq!(found(shallower), by_the_shallower, "{case}");
            let a_stop = state
                .recall(|kept| kept.explained_a_stop(key).then_some(()))
                .is_some();
            assert_eq!(a_stop, asked_about, "{case}");
        }
    });
}

/// An explaining walk with more room than one that stopped asks its deciding
/// walk, and says the kept stop again only where that walk refuses and stops
/// too: a value its deciding walk admits, though a branch of it stopped, is
/// admitted, and one its deciding walk refuses by its shape is reported by its
/// shape.
#[test]
fn an_explaining_walk_says_a_stop_again_only_where_its_deciding_walk_stops() {
    Python::attach(|py| {
        let defs = two_lists();
        let index = build_index(py, &reference(), &defs, &[]);
        let at_depth = |state: &WalkState, value: &Bound<'_, PyAny>, depth, mode| {
            state.depth.set(depth);
            let ctx = Ctx {
                pool: &[],
                defs: &defs,
                index: &index,
                state,
                mode,
            };
            let mut out = Vec::new();
            let admitted = member(
                &reference(),
                &Value::Py(value),
                &mut Frame::new(&mut Vec::new(), &mut out, ctx),
            );
            state.depth.set(0);
            (admitted, out)
        };
        let zero = 0i64.into_pyobject(py).expect("an int").into_any();
        // `[0]`, read where the first list branch reaches the depth bound and
        // the second, a level shallower, admits it.
        let admitted_past_a_stop = in_a_list(zero);
        let depth = (MAX_WALK_DEPTH - 8..MAX_WALK_DEPTH)
            .find(|&depth| {
                let state = WalkState::new();
                let (admitted, _) = at_depth(&state, &admitted_past_a_stop, depth, WalkMode::Fast);
                admitted && state.signals.stopped.get().is_some()
            })
            .expect("a depth one branch stops at and the other admits at");
        // `[None]`, which no branch admits, read where nothing stops.
        let refused_by_its_shape = in_a_list(py.None().into_bound(py));
        for (value, depth, deeper, admitted) in [
            (&admitted_past_a_stop, depth, depth + 1, true),
            (&refused_by_its_shape, 0, MAX_WALK_DEPTH - 1, false),
        ] {
            // An explaining walk begun deeper stops, and is kept.
            let state = WalkState::new();
            let (kept, report) = state
                .memo
                .armed(|| at_depth(&state, value, deeper, WalkMode::Explain));
            assert!(!kept, "{value}");
            assert!(
                report.iter().any(|v| v.code == RECURSION_LIMIT.as_str()),
                "{value}: the deeper walk did not stop"
            );
            let (found, out) = at_depth(&state, value, depth, WalkMode::Explain);
            assert_eq!(found, admitted, "{value}");
            assert!(
                out.iter().all(|v| v.code != RECURSION_LIMIT.as_str()),
                "{value}: the kept stop was said again"
            );
        }
    });
}

/// A clause reaches what its key names as well as what its value names: a map
/// whose first clause reads a definition as its key and as its value, and
/// whose second reads it as its value, has two clauses reaching it.
#[test]
fn a_clause_reaches_what_its_key_and_its_value_both_name() {
    Python::attach(|py| {
        let defs = vec![union_of(vec![
            Schema::Int,
            Schema::KeyedMap {
                fields: Vec::new().into(),
                defaults: vec![
                    MapClause {
                        key: reference(),
                        value: reference(),
                    },
                    MapClause {
                        key: Schema::Str,
                        value: reference(),
                    },
                ]
                .into(),
            },
        ])];
        let index = build_index(py, &reference(), &defs, &[]);
        let Some(Schema::Union(members)) = defs.first() else {
            unreachable!("the definition is a union");
        };
        let Some(Schema::KeyedMap { defaults, .. }) = members.get(1) else {
            unreachable!("the second member is a map");
        };
        assert!(index.memo.shares(defaults.as_ptr() as usize));
    });
}

/// A report that holds a violation already reads a shared definition once per
/// level too: the explaining walk of a list whose first element is refused
/// reaches the chain beside it with that violation in hand.
#[test]
fn a_report_holding_a_violation_walks_a_shared_definition_once_per_level() {
    Python::attach(|py| {
        let defs = two_lists();
        let at = (&[][..], &defs[..]);
        let entries = |depth: usize| {
            let chain = nested(py.None().into_bound(py), depth, in_a_list);
            let refused = 1.5f64.into_pyobject(py).expect("a float").into_any();
            let value = PyList::new(py, [refused, chain])
                .expect("a list")
                .into_any();
            walk(py, &reference(), &value, at, WalkMode::Explain, true).entered
        };
        let (shallow, deep) = (entries(6), entries(12));
        assert!(
            deep <= 2 * shallow + 16,
            "{shallow} entries at 6, {deep} at 12"
        );
    });
}
