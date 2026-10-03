use std::sync::Arc;

use super::record::{keyed_map_matches_json, scan_dict};
use super::sequence::scan_list;
use super::*;
use crate::check::ctx::MAX_WALK_DEPTH;
use crate::check::index::ValidatorIndex;
use crate::check::{WalkMode, WalkState, build_index};
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple};
use std::borrow::Cow;
use std::ops::ControlFlow;

use jiter::JsonValue;
use pyo3::types::{PyFrozenSet, PySet};
use valgebra_core::SeqShape;
use valgebra_core::{Constraint, DefIx, Field, MapClause, Openness, PathSegment};

/// Decide membership of a Python value against a schema, through the real
/// walk, in the mode a validator's `is_valid` uses.
fn holds(
    py: Python<'_>,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    pool: &[Py<PyAny>],
    defs: &[Schema],
) -> bool {
    let index = build_index(py, schema, defs, pool);
    let state = WalkState::new();
    let ctx = Ctx {
        pool,
        defs,
        records: &index.records,
        attrs: &index.attrs,
        unions: &index.unions,
        regexes: &index.regexes,
        guard: &state.guard,
        depth: &state.depth,
        fatal: &state.fatal,
        fatal_seen: &state.fatal_seen,
        mode: WalkMode::Fast,
    };
    member(
        schema,
        &Value::Py(value),
        &mut Frame::new(&mut Vec::new(), &mut Vec::new(), ctx),
    )
}

/// The same decision in explain mode, returning the violations it aggregated
/// alongside the verdict. The two modes must agree on the verdict — the "one
/// walk" invariant — so every case below is driven through both.
fn explain(
    py: Python<'_>,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    pool: &[Py<PyAny>],
    defs: &[Schema],
) -> (bool, Vec<Violation>) {
    explain_in(py, schema, value, pool, defs, WalkMode::Explain)
}

/// [`explain`] in a mode of the caller's choosing.
fn explain_in(
    py: Python<'_>,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    pool: &[Py<PyAny>],
    defs: &[Schema],
    mode: WalkMode,
) -> (bool, Vec<Violation>) {
    let index = build_index(py, schema, defs, pool);
    let state = WalkState::new();
    let ctx = Ctx {
        pool,
        defs,
        records: &index.records,
        attrs: &index.attrs,
        unions: &index.unions,
        regexes: &index.regexes,
        guard: &state.guard,
        depth: &state.depth,
        fatal: &state.fatal,
        fatal_seen: &state.fatal_seen,
        mode,
    };
    let mut out = Vec::new();
    let ok = member(
        schema,
        &Value::Py(value),
        &mut Frame::new(&mut Vec::new(), &mut out, ctx),
    );
    (ok, out)
}

/// Drive one case through both modes and assert they agree, then return the
/// verdict. A case that only ran fast would leave the explain arms — half of
/// every composite in this file — unobserved.
fn decide(
    py: Python<'_>,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    pool: &[Py<PyAny>],
    defs: &[Schema],
) -> bool {
    let fast = holds(py, schema, value, pool, defs);
    let (explained, violations) = explain(py, schema, value, pool, defs);
    assert_eq!(fast, explained, "fast and explain modes disagree");
    assert_eq!(
        violations.is_empty(),
        fast,
        "a rejected value must report at least one violation, an accepted one none"
    );
    fast
}

/// A dict scan stops at the entry count it began with, and reports rather
/// than reads a dict whose size moved.
///
/// The count is what keeps the iterator away from the state `PyO3` panics in,
/// and a panic is not one of the answers this library gives: it crosses the
/// FFI boundary as a `BaseException` no caller catches as a validation
/// failure. Driven against the scan rather than through a schema, because
/// the schema path needs a value that mutates itself mid-walk and the
/// question here is what the scan does with the count.
#[test]
fn a_dict_scan_visits_each_entry_once_and_stops_where_it_is_told() {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        for i in 0..5 {
            dict.set_item(i, i).expect("set_item");
        }

        // Every entry, exactly once: a count that advanced by more than one
        // per entry would visit fewer, and one that compared loosely would
        // step past the end.
        let mut seen = 0;
        let scan = scan_dict(&dict, |_, _| {
            seen += 1;
            ControlFlow::Continue(())
        });
        assert!(matches!(scan, Scan::Complete));
        assert_eq!(seen, 5, "each of the five entries is visited once");

        // A visitor that breaks stops the scan, and the answer says so: a
        // stop is not a complete reading, and the caller reports the miss
        // that caused it rather than the container.
        let mut before_break = 0;
        let scan = scan_dict(&dict, |_, _| {
            before_break += 1;
            if before_break == 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert!(matches!(scan, Scan::Stopped));
        assert_eq!(before_break, 2);

        // A dict that grows under the scan has no reading to answer from.
        let moving = PyDict::new(py);
        for i in 0..4 {
            moving.set_item(i, i).expect("set_item");
        }
        let scan = scan_dict(&moving, |key, _| {
            if key.extract::<i64>().unwrap_or(-1) == 0 {
                moving.set_item("added", 1).expect("set_item");
            }
            ControlFlow::Continue(())
        });
        assert!(
            matches!(scan, Scan::Unreadable),
            "a dict whose size moved is not readable"
        );

        // And one that shrinks, which is the same fact reached at the other
        // end: the scan began expecting entries that are no longer there.
        let shrinking = PyDict::new(py);
        for i in 0..4 {
            shrinking.set_item(i, i).expect("set_item");
        }
        let scan = scan_dict(&shrinking, |key, _| {
            if key.extract::<i64>().unwrap_or(-1) == 0 {
                shrinking.del_item(3).expect("del_item");
            }
            ControlFlow::Continue(())
        });
        assert!(matches!(scan, Scan::Unreadable));

        // A key the scan has passed, swapped at the same size for a new one,
        // leaves the iterator one entry past the count: it yields the new key,
        // and PyO3 panics on the step after. The scan stops at the count it
        // began with, so it takes neither step.
        let swapped = PyDict::new(py);
        for i in 0..4 {
            swapped.set_item(i, i).expect("set_item");
        }
        let mut visited = 0;
        let scan = scan_dict(&swapped, |key, _| {
            visited += 1;
            if key.extract::<i64>().unwrap_or(-1) == 1 {
                swapped.del_item(0).expect("del_item");
                swapped.set_item("new", 0).expect("set_item");
            }
            ControlFlow::Continue(())
        });
        assert_eq!(visited, 4, "the scan stops at the count it began with");
        assert!(!matches!(scan, Scan::Stopped));
    });
}

/// A list scan visits each position once and refuses a list that resizes.
///
/// The sequence walk reads a length once and matches positions against it, so a
/// list that grows hides its new items from the walk and one that shrinks leaves
/// it answering about items that are gone -- and `is_valid` said `True` for a
/// value that is not a member. Driven against the scan for the reason the dict
/// case is: the question is what the scan does with the count.
#[test]
fn a_list_scan_visits_each_position_once_and_refuses_one_that_resizes() {
    Python::attach(|py| {
        let list = PyList::new(py, 0..5).expect("a list of five");

        // Every position, exactly once, and in order: the index the visitor is
        // handed is what picks a prefix schema, so a drifting one would match
        // the wrong element rather than fail.
        let mut seen = Vec::new();
        let scan = scan_list(&list, |at, _| {
            seen.push(at);
            ControlFlow::Continue(())
        });
        assert!(matches!(scan, Scan::Complete));
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);

        // A visitor that breaks stops the scan, and the answer says so.
        let mut before_break = 0;
        let scan = scan_list(&list, |_, _| {
            before_break += 1;
            if before_break == 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        assert!(matches!(scan, Scan::Stopped));
        assert_eq!(before_break, 2);

        // Grown under the scan: the items past the length read at entry would
        // never be visited, so there is no reading to answer from.
        let growing = PyList::new(py, 0..4).expect("a list of four");
        let scan = scan_list(&growing, |at, _| {
            if at == 0 {
                growing.append(9).expect("append");
            }
            ControlFlow::Continue(())
        });
        assert!(
            matches!(scan, Scan::Unreadable),
            "a list whose size moved is not readable"
        );

        // And shrunk, which is the same fact at the other end.
        let shrinking = PyList::new(py, 0..4).expect("a list of four");
        let scan = scan_list(&shrinking, |at, _| {
            if at == 0 {
                shrinking.del_item(3).expect("del_item");
            }
            ControlFlow::Continue(())
        });
        assert!(matches!(scan, Scan::Unreadable));

        // A list nobody touched reads to the end, so the guard costs no answer.
        let still = PyList::new(py, 0..4).expect("a list of four");
        assert!(matches!(
            scan_list(&still, |_, _| ControlFlow::Continue(())),
            Scan::Complete
        ));
    });
}

/// A value that changed under the walk is a non-member, and in explain mode
/// it says which failure it was.
///
/// The code is valgebra-coined because it reports a failure of the *check*:
/// nothing about the value's contents was decided. A verdict of `true` here
/// would admit a value no reading of it supports.
#[test]
fn a_changed_container_is_a_non_member_that_names_itself() {
    Python::attach(|py| {
        let value = PyDict::new(py);
        let state = WalkState::new();
        let index = build_index(py, &Schema::ANYTHING, &[], &[]);
        let ctx = |mode| Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode,
        };

        let mut out = Vec::new();
        let held = mutated(
            &Value::Py(&value),
            &mut Frame::new(&mut Vec::new(), &mut out, ctx(WalkMode::Explain)),
        );
        assert!(!held, "a value that changed under the walk is a non-member");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].code, MUTATED_CODE.as_str());
        assert_eq!(out[0].expected, MUTATED_EXPECTED);

        // Fast mode reports the same verdict and writes nothing: the
        // violations it would build go to a buffer nothing reads.
        let mut fast_out = Vec::new();
        let held = mutated(
            &Value::Py(&value),
            &mut Frame::new(&mut Vec::new(), &mut fast_out, ctx(WalkMode::Fast)),
        );
        assert!(!held);
        assert!(fast_out.is_empty());
    });
}

/// A union's `expected` names each branch the way that branch names itself,
/// and a nested union contributes its members rather than itself.
///
/// `Literal[...]` builds a union of its constants, so without the nesting
/// rule a single-constant literal would name itself `union` and a table of
/// permitted strings would read `one of: literal, literal`.
#[test]
fn a_union_names_its_branches_by_their_constants() {
    Python::attach(|py| {
        let pool: Vec<Py<PyAny>> = ["torch", "jax"]
            .iter()
            .map(|name| PyString::new(py, name).into_any().unbind())
            .collect();
        let table = Schema::Union(
            vec![
                Schema::Literal(ConstIx::new(0)),
                Schema::Literal(ConstIx::new(1)),
            ]
            .into(),
        );
        // Nested, which is the shape `Literal[...]` beside another branch
        // builds: the inner union's members are the branches, not the union.
        let schema = Schema::Union(vec![table, Schema::Int].into());
        let index = build_index(py, &schema, &[], &pool);
        let state = WalkState::new();
        let ctx = Ctx {
            pool: &pool,
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Explain,
        };
        let mut labels = BranchLabels::new();
        push_branch_label(&schema, ctx, py, &mut labels);
        assert_eq!(
            labels.render(),
            "one of: the literal 'torch', the literal 'jax', int",
            "each branch names itself, and the nested union names its members"
        );
    });
}

/// A meet collects every member's failure until one rejects the value
/// *itself*, and then stops: what a later member would say describes a value
/// already known to be the wrong kind of thing. A member that fails *inside*
/// the value leaves the others meaningful, and they are still collected.
///
/// This is what keeps a class with declared attributes -- the meet of an
/// `isinstance` atom and an attribute record -- reporting one
/// `instance_type` for a foreign object rather than that violation plus the
/// attributes the object never had to carry.
#[test]
fn a_meet_stops_at_the_member_that_rejects_the_value() {
    Python::attach(|py| {
        let module = classes(py);
        let point = module.getattr("Point").expect("Point");
        let pool: Vec<Py<PyAny>> = vec![point.clone().unbind()];
        let object = Schema::meet([
            Schema::Instance(ClassIx::new(0)),
            Schema::AttrRecord {
                fields: vec![field("x", Schema::Int, true)].into(),
            },
        ]);

        // Not a Point: the class atom rejects the value at the meet's own
        // path, so the record is not asked about attributes it does not have.
        let foreign = PyInt::new(py, 1i64).into_any();
        let (ok, violations) = explain(py, &object, &foreign, &pool, &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].code, "instance_type");
        assert_eq!(violations[0].path, Vec::new());

        // A Point whose attribute is wrong fails *inside* the value: the
        // class atom held, and the record reports the attribute.
        let bad = point
            .call1((PyString::new(py, "x"), PyInt::new(py, 2i64)))
            .expect("Point(str, int)");
        let (ok, violations) = explain(py, &object, &bad, &pool, &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].location(), "x");

        // Two members that each fail inside the value both report: neither
        // rejected the value itself, so neither silences the other.
        let deep = |element| Schema::list(SeqShape::homogeneous(element));
        let both = Schema::Intersection(vec![deep(Schema::Int), deep(Schema::Bool)].into());
        let list = PyList::new(py, [PyString::new(py, "a")]).expect("list");
        let (ok, violations) = explain(py, &both, &list.into_any(), &[], &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 2, "{violations:?}");

        // And a member that rejects the value itself stops the rest even
        // when no class is involved.
        let scalars = Schema::Intersection(vec![Schema::Int, Schema::Str].into());
        let number = PyFloat::new(py, 1.5).into_any();
        let (ok, violations) = explain(py, &scalars, &number, &[], &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].code, "int_type");
    });
}

/// A class branch names its class, and a class with declared attributes --
/// the meet of an atom and a record -- names that same class rather than the
/// algebra's spelling of it. A meet that is not an object has no class to
/// name, and names each member instead: its kind would name no set.
#[test]
fn a_union_names_a_class_branch_by_its_class() {
    Python::attach(|py| {
        let module = classes(py);
        let point = module.getattr("Point").expect("Point");
        let other = module.getattr("Other").expect("Other");
        let pool: Vec<Py<PyAny>> = vec![point.unbind(), other.unbind()];
        let object = Schema::meet([
            Schema::Instance(ClassIx::new(0)),
            Schema::AttrRecord {
                fields: vec![Field {
                    name: "x".into(),
                    schema: Schema::Int,
                    required: true,
                }]
                .into(),
            },
        ]);
        let schema = Schema::Union(
            vec![
                object,
                Schema::Instance(ClassIx::new(1)),
                Schema::meet([Schema::Int, Schema::Str]),
                Schema::meet([
                    Schema::union([Schema::Int, Schema::Str]),
                    Schema::Bool.complement(),
                ]),
            ]
            .into(),
        );
        let index = build_index(py, &schema, &[], &pool);
        let state = WalkState::new();
        let ctx = Ctx {
            pool: &pool,
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Explain,
        };
        let mut labels = BranchLabels::new();
        push_branch_label(&schema, ctx, py, &mut labels);
        assert_eq!(
            labels.render(),
            "one of: Point, Other, int and str, (int or str) and not bool",
            "an object meet names its class; any other meet names its members"
        );
    });
}

/// A value a union admits is never summarized.
///
/// Explaining a union walked every branch in explain mode, and a branch that
/// refused the value by its kind built a violation nothing kept, summarizing
/// the value in it -- which ran the value's `__repr__` once for each branch
/// before the one that matched. The repr here counts. Each union below admits
/// the value through its last branch and must leave the count at zero, in both
/// explaining modes; a union the value does not belong to summarizes it, which
/// is what the count is able to see.
#[test]
fn a_value_a_union_admits_is_never_summarized() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"class Seen:\n\
              \x20   reprs = 0\n\
              \x20   def __repr__(self):\n\
              \x20       Seen.reprs += 1\n\
              \x20       return 'Seen()'\n",
            c"seen.py",
            c"seen",
        )
        .expect("the module compiles");
        let seen = module.getattr("Seen").expect("the class");
        let value = seen.call0().expect("an instance");
        let one = 1i64.into_pyobject(py).unwrap().into_any().unbind();
        let pool: Vec<Py<PyAny>> = vec![seen.clone().unbind(), one];
        let reprs = || {
            seen.getattr("reprs")
                .and_then(|count| count.extract::<i64>())
                .expect("the count reads")
        };
        let ints = Schema::list(SeqShape::homogeneous(Schema::Int));
        let refused = [
            Schema::Int,
            Schema::Str,
            Schema::Literal(ConstIx::new(1)),
            ints.clone(),
            Schema::Refine {
                base: Arc::new(ints),
                constraints: vec![Constraint::MinLen(1)].into(),
            },
        ];
        for branch in refused {
            let union = Schema::Union(vec![branch, Schema::Instance(ClassIx::new(0))].into());
            for mode in [WalkMode::Explain, WalkMode::ExplainFailFast] {
                let (admitted, violations) = explain_in(py, &union, &value, &pool, &[], mode);
                assert!(admitted, "{union:?} admits the instance");
                assert!(violations.is_empty(), "{union:?} reports nothing");
            }
        }
        assert_eq!(reprs(), 0, "no branch summarized a value the union admits");
        let (admitted, _) = explain(
            py,
            &Schema::Union(vec![Schema::Int, Schema::Str].into()),
            &value,
            &pool,
            &[],
        );
        assert!(!admitted);
        assert!(reprs() > 0, "a union the value is outside summarizes it");
    });
}

/// The classes the union corpus below reads: a repr that counts, two plain
/// classes and a subclass, and a class whose metaclass answers `isinstance`
/// with code of its own that counts its calls.
///
/// Each test names its own module. `PyModule::from_code` runs its source in
/// the `sys.modules` entry of the name it is given when there is one, so a
/// second test under the same name rebinds the first's classes and resets its
/// counts -- on a free-threaded interpreter, between the first reading a class
/// and reading its count.
fn union_classes<'py>(py: Python<'py>, name: &std::ffi::CStr) -> Bound<'py, PyModule> {
    PyModule::from_code(
        py,
        c"class Seen:\n\
          \x20   reprs = 0\n\
          \x20   def __repr__(self):\n\
          \x20       Seen.reprs += 1\n\
          \x20       return 'Seen()'\n\
          class Other:\n\
          \x20   pass\n\
          class Base:\n\
          \x20   def __init__(self):\n\
          \x20       self.x = 1\n\
          class Sub(Base):\n\
          \x20   pass\n\
          class Meta(type):\n\
          \x20   calls = 0\n\
          \x20   def __instancecheck__(cls, value):\n\
          \x20       Meta.calls += 1\n\
          \x20       return True\n\
          class Hooked(metaclass=Meta):\n\
          \x20   pass\n\
          def interrupts(value):\n\
          \x20   raise KeyboardInterrupt\n",
        c"union_classes.py",
        name,
    )
    .expect("the module compiles")
}

/// A class attribute of the union corpus's module, read as a count.
fn count_of(module: &Bound<'_, PyModule>, class: &str, name: &str) -> i64 {
    module
        .getattr(class)
        .and_then(|class| class.getattr(name))
        .and_then(|count| count.extract::<i64>())
        .expect("the count reads")
}

/// A record branch or a class branch builds no report for a value its union
/// admits.
///
/// A record branch is decided by its deciding pass beside the others, and
/// explained only once no branch admits the value; a class branch is its
/// instance test, and a class met with its attributes is refused by its class
/// without being explained. Each union below admits the value through a later
/// branch, a subclass instance through the class it derives from among them,
/// and none may summarize it.
#[test]
fn a_record_or_class_branch_builds_no_report_for_a_value_the_union_admits() {
    Python::attach(|py| {
        let module = union_classes(py, c"union_unreported");
        let class = |name: &str| module.getattr(name).expect("the class");
        let pool: Vec<Py<PyAny>> = ["Seen", "Other", "Base"]
            .map(|name| class(name).unbind())
            .into();
        let instance = |slot| Schema::Instance(ClassIx::new(slot));
        let object = |slot| {
            Schema::meet([
                instance(slot),
                Schema::AttrRecord {
                    fields: vec![field("x", Schema::Int, true)].into(),
                },
            ])
        };
        let record = |schema| Schema::record(vec![field("a", schema, true)], Openness::Closed);
        let union = |branches: Vec<Schema>| Schema::Union(branches.into());
        let seen = class("Seen").call0().expect("an instance");
        let holding = PyDict::new(py);
        holding.set_item("a", &seen).expect("a field");
        let sub = class("Sub").call0().expect("a subclass instance");
        let cases = [
            (
                union(vec![record(Schema::Int), record(instance(0))]),
                holding.as_any(),
            ),
            (union(vec![instance(1), instance(0)]), &seen),
            (union(vec![object(1), instance(0)]), &seen),
            (union(vec![Schema::Int, instance(2)]), &sub),
            (union(vec![Schema::Int, object(2)]), &sub),
        ];
        for (schema, value) in &cases {
            for mode in [WalkMode::Explain, WalkMode::ExplainFailFast] {
                let (admitted, violations) = explain_in(py, schema, value, &pool, &[], mode);
                assert!(admitted, "{schema:?} admits {value}");
                assert!(violations.is_empty(), "{schema:?} reports nothing");
            }
        }
        assert_eq!(count_of(&module, "Seen", "reprs"), 0);
        let refused = union(vec![record(Schema::Int), record(Schema::Str)]);
        let (admitted, _) = explain(py, &refused, holding.as_any(), &pool, &[]);
        assert!(!admitted);
        assert!(
            count_of(&module, "Seen", "reprs") > 0,
            "a union the value is outside summarizes it"
        );
    });
}

/// A union no branch admits reports what its chosen branch reports walked on
/// its own, in either explaining mode.
///
/// The branch whose first failure lies deepest is chosen, the earliest on a
/// tie, and a record branch is explained after every branch is decided while
/// any other is explained as it is decided: the cases below put a record
/// beside a record nested deeper, beside a record that ties it, and beside a
/// refined record explained in its own place, in both orders.
#[test]
fn a_refused_union_reports_what_its_chosen_branch_reports_alone() {
    Python::attach(|py| {
        let record = |fields| Schema::record(fields, Openness::Closed);
        let flat = record(vec![field("a", Schema::Int, true)]);
        let wide = record(vec![
            field("a", Schema::Int, true),
            field("b", Schema::Int, true),
        ]);
        let text = record(vec![field("a", Schema::Str, true)]);
        let deep = record(vec![field(
            "a",
            record(vec![field("c", Schema::Int, true)]),
            true,
        )]);
        let refined = Schema::Refine {
            base: Arc::new(text.clone()),
            constraints: vec![Constraint::MinLen(1)].into(),
        };
        let dict = |code: &str| {
            py.eval(&std::ffi::CString::new(code).expect("no nul"), None, None)
                .expect("the value evaluates")
        };
        let cases = [
            (wide.clone(), text.clone(), dict("{'a': 1, 'b': 'x'}"), 0),
            (text, wide, dict("{'a': 1, 'b': 'x'}"), 0),
            (flat.clone(), deep, dict("{'a': {'c': 'x'}}"), 1),
            (flat.clone(), refined.clone(), dict("{'a': 1.5}"), 0),
            (refined, flat, dict("{'a': 1.5}"), 0),
        ];
        for (first, second, value, chosen) in cases {
            let union = Schema::Union(vec![first.clone(), second.clone()].into());
            let alone = [first, second][chosen].clone();
            for mode in [WalkMode::Explain, WalkMode::ExplainFailFast] {
                let (admitted, report) = explain_in(py, &union, &value, &[], &[], mode);
                let (_, expected) = explain_in(py, &alone, &value, &[], &[], mode);
                assert!(!admitted, "{union:?} refuses {value}");
                assert!(!expected.is_empty(), "{alone:?} reports {value}");
                assert_eq!(
                    format!("{report:?}"),
                    format!("{expected:?}"),
                    "{union:?} against {value} in {mode:?}"
                );
            }
        }
    });
}

/// A class whose metaclass answers `isinstance` with code of its own is asked
/// once for a value a union admits through it, alone or met with the
/// attributes it declares.
///
/// The meet's walk asks the class itself, so a branch led by such a class is
/// explained rather than decided first: deciding it would run the metaclass's
/// code twice for one member.
#[test]
fn a_class_with_its_own_instance_test_is_asked_once_through_a_union() {
    Python::attach(|py| {
        let module = union_classes(py, c"union_asked_once");
        let hooked = module.getattr("Hooked").expect("the class");
        let pool = vec![hooked.unbind()];
        let value = module
            .getattr("Base")
            .and_then(|class| class.call0())
            .expect("an instance");
        let alone = Schema::Instance(ClassIx::new(0));
        let met = Schema::meet([
            alone.clone(),
            Schema::AttrRecord {
                fields: vec![field("x", Schema::Int, true)].into(),
            },
        ]);
        for branch in [alone, met] {
            let union = Schema::Union(vec![Schema::Int, branch].into());
            for mode in [WalkMode::Explain, WalkMode::ExplainFailFast] {
                let before = count_of(&module, "Meta", "calls");
                let (admitted, _) = explain_in(py, &union, &value, &pool, &[], mode);
                assert!(admitted, "{union:?} admits {value}");
                assert_eq!(
                    count_of(&module, "Meta", "calls") - before,
                    1,
                    "{union:?} asks its class once"
                );
            }
        }
    });
}

/// A fatal signal one branch of a union raises stops the report of a record
/// branch decided before it, as it stops every later walk: the record's
/// explaining pass, which would summarize a key it does not declare, never
/// runs.
///
/// The refined record between them is explained as it is decided, before the
/// signal, and is the branch the report chooses; it admits the undeclared key,
/// so nothing it reports summarizes it either.
#[test]
fn a_signal_a_later_branch_raises_stops_an_earlier_record_report() {
    use pyo3::exceptions::PyKeyboardInterrupt;
    Python::attach(|py| {
        let module = union_classes(py, c"union_signalled");
        let pool = vec![
            module
                .getattr("interrupts")
                .expect("the predicate")
                .unbind(),
        ];
        let interrupted = Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
        };
        let refined = Schema::Refine {
            base: Arc::new(Schema::record(
                vec![field("a", Schema::Str, true)],
                Openness::Open,
            )),
            constraints: vec![Constraint::MinLen(1)].into(),
        };
        let union = Schema::Union(
            vec![
                Schema::record(vec![field("a", Schema::Int, true)], Openness::Closed),
                refined,
                Schema::record(vec![field("b", interrupted, true)], Openness::Closed),
            ]
            .into(),
        );
        let value = PyDict::new(py);
        value.set_item("a", 1.5f64).expect("a field");
        value.set_item("b", 1i64).expect("a field");
        let seen = module
            .getattr("Seen")
            .and_then(|class| class.call0())
            .expect("an instance");
        value.set_item("z", seen).expect("an undeclared key");
        let (fatal, report) = recorded(py, &union, value.as_any(), &pool);
        assert!(fatal.is_some_and(|err| err.is_instance_of::<PyKeyboardInterrupt>(py)));
        assert_eq!(
            report.iter().map(|v| v.code).collect::<Vec<_>>(),
            ["string_type"]
        );
        assert_eq!(count_of(&module, "Seen", "reprs"), 0);
    });
}

/// `(schema, value, expected)` over an empty pool and no definitions.
fn case(py: Python<'_>, schema: &Schema, value: &Bound<'_, PyAny>, expected: bool) {
    assert_eq!(
        decide(py, schema, value, &[], &[]),
        expected,
        "schema {schema:?} against {value}"
    );
}

fn list_of(py: Python<'_>, items: Vec<i64>) -> Bound<'_, PyAny> {
    PyList::new(py, items)
        .expect("a list of i64 builds")
        .into_any()
}

#[test]
fn the_scalar_atoms_admit_their_own_kind_and_no_other() {
    Python::attach(|py| {
        let none = py.None().into_bound(py);
        let boolean = PyBool::new(py, true).to_owned().into_any();
        let integer = PyInt::new(py, 7i64).into_any();
        let float = PyFloat::new(py, 1.5).into_any();
        let text = PyString::new(py, "x").into_any();
        let raw = PyBytes::new(py, b"x").into_any();
        let values = [&none, &boolean, &integer, &float, &text, &raw];

        // Each atom admits exactly its own column, with one exception the
        // typing spec forces: `bool` is a subclass of `int`, so a boolean is
        // an integer and `Int` admits it.
        let rows: [(Schema, [bool; 6]); 6] = [
            (Schema::NoneType, [true, false, false, false, false, false]),
            (Schema::Bool, [false, true, false, false, false, false]),
            (Schema::Int, [false, true, true, false, false, false]),
            (Schema::Float, [false, false, false, true, false, false]),
            (Schema::Str, [false, false, false, false, true, false]),
            (Schema::Bytes, [false, false, false, false, false, true]),
        ];
        for (schema, expected) in &rows {
            for (value, want) in values.iter().zip(expected) {
                case(py, schema, value, *want);
            }
        }

        // The lattice bounds, over the same column set. The top is checked
        // in both spellings: one node admits every value whichever way the
        // user wrote it.
        for value in values {
            case(py, &Schema::ANYTHING, value, true);
            case(py, &Schema::ANY, value, true);
            case(py, &Schema::Nothing, value, false);
            // A self-reference never survives compilation, and is never a
            // member if one is reached anyway.
            case(py, &Schema::SelfRef(0), value, false);
        }
    });
}

#[test]
fn a_literal_admits_its_own_value_at_its_own_type() {
    Python::attach(|py| {
        let one = PyInt::new(py, 1i64).into_any();
        let pool = vec![one.clone().unbind()];
        let schema = Schema::Literal(ConstIx::new(0));

        assert!(decide(
            py,
            &schema,
            &PyInt::new(py, 1i64).into_any(),
            &pool,
            &[]
        ));
        assert!(!decide(
            py,
            &schema,
            &PyInt::new(py, 2i64).into_any(),
            &pool,
            &[]
        ));
        // Python's `==` conflates across types (`1 == True == 1.0`), so the
        // same-type test is what makes this a singleton rather than a class.
        let truth = PyBool::new(py, true).to_owned().into_any();
        assert!(!decide(py, &schema, &truth, &pool, &[]));
        let float_one = PyFloat::new(py, 1.0).into_any();
        assert!(!decide(py, &schema, &float_one, &pool, &[]));
    });
}

#[test]
fn a_sequence_matches_its_regex_and_its_container_kind() {
    Python::attach(|py| {
        let homogeneous = Schema::list(SeqShape::homogeneous(Schema::Int));
        case(py, &homogeneous, &list_of(py, vec![]), true);
        case(py, &homogeneous, &list_of(py, vec![1, 2, 3]), true);
        let mixed = PyList::new(py, [1i64])
            .expect("a one-element list builds")
            .into_any();
        mixed
            .cast::<PyList>()
            .expect("a list")
            .append(PyString::new(py, "x"))
            .expect("append");
        case(py, &homogeneous, &mixed, false);

        // The container kind is part of the denotation: a tuple is not a list.
        let tuple = PyTuple::new(py, [1i64, 2])
            .expect("a tuple builds")
            .into_any();
        case(py, &homogeneous, &tuple, false);
        case(
            py,
            &Schema::tuple(SeqShape::homogeneous(Schema::Int)),
            &tuple,
            true,
        );

        // Fixed arity: exactly the prefix length, no more and no fewer.
        let fixed = Schema::list(SeqShape::fixed([Schema::Int, Schema::Str]));
        let ok = PyList::new(py, [1i64]).expect("builds").into_any();
        ok.cast::<PyList>()
            .expect("a list")
            .append(PyString::new(py, "x"))
            .expect("append");
        case(py, &fixed, &ok, true);
        case(py, &fixed, &list_of(py, vec![1]), false);
        case(py, &fixed, &list_of(py, vec![1, 2, 3]), false);

        // Prefix plus tail: at least the prefix length, and the tail repeats.
        let prefixed = Schema::list(SeqShape::prefix_tail([Schema::Int], Schema::Int));
        case(py, &prefixed, &list_of(py, vec![]), false);
        case(py, &prefixed, &list_of(py, vec![1]), true);
        case(py, &prefixed, &list_of(py, vec![1, 2, 3]), true);
    });
}

/// A sequence with a prefix is not one schema at every position, so the loop
/// must not take it.
///
/// `tuple[str, int, ...]` says the first element is a string and the rest are
/// integers. A loop that read the tail as the whole shape would test the first
/// element against `int` and reject the value the schema admits.
#[test]
fn a_prefix_is_not_tested_against_the_repeated_tail() {
    Python::attach(|py| {
        let prefixed = Schema::tuple(SeqShape::prefix_tail([Schema::Str], Schema::Int));
        let good = PyTuple::new(py, [PyString::new(py, "a").into_any()])
            .expect("a tuple builds")
            .into_any();
        let good = good
            .cast::<PyTuple>()
            .expect("a tuple")
            .as_sequence()
            .concat(
                PyTuple::new(py, [1i64, 2])
                    .expect("a tuple builds")
                    .as_sequence(),
            )
            .expect("two tuples concatenate")
            .to_tuple()
            .expect("a sequence of tuples is a tuple")
            .into_any();
        case(py, &prefixed, &good, true);
        // The tail's schema at the prefix's position is not the shape: an int
        // first is a non-member, and a loop over the tail alone would admit it.
        let wrong = PyTuple::new(py, [1i64, 2])
            .expect("a tuple builds")
            .into_any();
        case(py, &prefixed, &wrong, false);
    });
}

/// A sequence of one scalar kind is walked by a loop of its own, and that loop
/// answers what the general walk answers.
///
/// The loop reads the element schema once and tests each element against the
/// kind, without the depth guard, the signal check and the dispatch the general
/// walk pays per element. What it must not lose is the answer: one element that
/// is not of the kind makes the value a non-member, and the loop has to stop
/// there rather than fold the elements together.
#[test]
fn a_sequence_of_one_scalar_kind_rejects_an_element_that_is_not_one() {
    Python::attach(|py| {
        let bad = |first: bool| {
            let items = PyList::new(py, [1i64]).expect("a one-element list builds");
            if first {
                items.insert(0, PyString::new(py, "x")).expect("insert");
            } else {
                items.append(PyString::new(py, "x")).expect("append");
            }
            items.into_any()
        };
        let list_of_int = Schema::list(SeqShape::homogeneous(Schema::Int));
        case(py, &list_of_int, &list_of(py, vec![1, 2, 3]), true);
        // Wherever the element that is not an int sits: first, so the loop must
        // stop; last, so it must not have folded the earlier answers away.
        case(py, &list_of_int, &bad(true), false);
        case(py, &list_of_int, &bad(false), false);

        // The same for the immutable container, which the loop walks without a
        // scan because a tuple cannot resize under it.
        let tuple_of_int = Schema::tuple(SeqShape::homogeneous(Schema::Int));
        let mixed = PyTuple::new(py, [1i64, 2])
            .expect("a tuple builds")
            .into_any();
        case(py, &tuple_of_int, &mixed, true);
        let spoiled = PyTuple::new(py, [PyString::new(py, "x")])
            .expect("a tuple builds")
            .into_any();
        case(py, &tuple_of_int, &spoiled, false);

        // And for a set, whose elements the same loop tests.
        let set_of_int = Schema::set(Schema::Int);
        let set = PySet::new(py, [1i64, 2]).expect("a set builds").into_any();
        case(py, &set_of_int, &set, true);
        let spoiled = PySet::new(py, [PyString::new(py, "x")])
            .expect("a set builds")
            .into_any();
        case(py, &set_of_int, &spoiled, false);
    });
}

#[test]
fn a_set_and_a_frozenset_are_distinct_containers() {
    Python::attach(|py| {
        let set_of_int = Schema::set(Schema::Int);
        let frozen_of_int = Schema::frozen_set(Schema::Int);
        let set = PySet::new(py, [1i64, 2]).expect("a set builds").into_any();
        let frozen = PyFrozenSet::new(py, [1i64, 2])
            .expect("a frozenset builds")
            .into_any();

        case(py, &set_of_int, &set, true);
        case(py, &set_of_int, &frozen, false);
        case(py, &frozen_of_int, &frozen, true);
        case(py, &frozen_of_int, &set, false);

        let mixed = PySet::new(py, [1i64]).expect("a set builds").into_any();
        mixed
            .cast::<PySet>()
            .expect("a set")
            .add(PyString::new(py, "x"))
            .expect("add");
        case(py, &set_of_int, &mixed, false);
    });
}

/// A list wide enough to be read through a snapshot of it answers what a
/// narrow one answers.
///
/// The walk copies a list of one scalar kind into a tuple and reads the tuple's
/// elements borrowed, on the interpreters where owning them is dear, and only
/// between two widths: a list too narrow cannot pay for the copy's allocation.
/// Every list in the rest of this file is under that floor, so the copy is a
/// path nothing here walked -- and the verdict it produces, and the mutation it
/// reports when the list moves underneath it, went unread.
#[test]
fn a_list_wide_enough_for_a_snapshot_answers_as_a_narrow_one_does() {
    Python::attach(|py| {
        let ints = Schema::list(SeqShape::homogeneous(Schema::Int));
        let wide: Vec<i64> = (0..64).collect();

        let good = PyList::new(py, &wide).expect("a list builds").into_any();
        case(py, &ints, &good, true);

        // One element of another kind, at the end, so the answer depends on the
        // whole copy being read rather than on where the walk gives up.
        let mixed = PyList::new(py, &wide).expect("a list builds");
        mixed
            .set_item(63, PyString::new(py, "x"))
            .expect("a list takes an item");
        case(py, &ints, &mixed.into_any(), false);

        // And a kind that is not the element's, at the front.
        let front = PyList::new(py, &wide).expect("a list builds");
        front
            .set_item(0, PyString::new(py, "x"))
            .expect("a list takes an item");
        case(py, &ints, &front.into_any(), false);
    });
}

/// A closed record holding exactly the keys it declares has no undeclared key
/// to find, and the report reaches that by counting what the field walk found
/// against the entries the value held. Checking a field runs Python, and Python
/// can add a key, so the count is a claim about the value as it was: the length
/// is read again before it is believed.
///
/// The arrangement below is the one where a stale count would say there is
/// nothing to look for -- the first field fails and the second grows the value,
/// which leaves as many declared fields found as there were entries to begin
/// with. The key the growth added is still reported.
#[test]
fn a_key_added_while_a_record_is_explained_is_still_reported() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "payload = {'a': 1, 'b': 2}\n\
                 def grow(x):\n\
                 \x20   payload['x%d' % len(payload)] = 0\n\
                 \x20   return True\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("grow.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("grow")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let payload = module.getattr("payload").expect("payload");
        let pool = vec![module.getattr("grow").expect("grow").unbind()];

        let field = |name: &str, schema| Field {
            name: name.into(),
            schema,
            required: true,
        };
        let schema = Schema::keyed_map(
            vec![
                field("a", Schema::Str),
                field(
                    "b",
                    Schema::Refine {
                        base: Arc::new(Schema::Int),
                        constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
                    },
                ),
            ],
            Vec::new(),
        );

        let (ok, violations) = explain(py, &schema, &payload, &pool, &[]);
        assert!(!ok);
        let codes: Vec<&str> = violations.iter().map(|v| v.code).collect();
        assert!(codes.contains(&"string_type"), "{codes:?}");
        assert!(codes.contains(&"extra_forbidden"), "{codes:?}");
    });
}

/// A tuple's elements are borrowed rather than owned, and the walk runs Python
/// between the borrow and the answer -- an `isinstance` reaches a metaclass that
/// can run anything at all. What keeps the borrow good is the tuple: it is
/// frozen, so an element cannot be replaced, and the caller's own handle holds
/// it for the whole walk, so nothing it contains can be freed. The check here
/// drops every other reference to the tuple and collects, which is the strongest
/// form of that pressure the interpreter offers.
#[test]
fn a_tuple_element_survives_the_python_its_own_check_runs() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "import gc\n\
                 held = None\n\
                 class Meta(type):\n\
                 \x20   def __instancecheck__(cls, obj):\n\
                 \x20       global held\n\
                 \x20       held = None\n\
                 \x20       gc.collect()\n\
                 \x20       return type(obj).__name__ == 'Thing'\n\
                 class Thing(metaclass=Meta):\n\
                 \x20   pass\n\
                 class Other:\n\
                 \x20   pass\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("borrowed.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("borrowed")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let thing_class = module.getattr("Thing").expect("Thing");
        let other_class = module.getattr("Other").expect("Other");
        let pool = vec![thing_class.clone().unbind()];
        let schema = Schema::tuple(SeqShape::homogeneous(Schema::Instance(ClassIx::new(0))));

        let thing = || thing_class.call0().expect("Thing()");
        let good = PyTuple::new(py, [thing(), thing(), thing()])
            .expect("a tuple builds")
            .into_any();
        module
            .setattr("held", &good)
            .expect("the module holds the tuple");
        assert!(decide(py, &schema, &good, &pool, &[]));

        // And the same pressure on the rejecting answer, where the walk stops
        // part-way through the elements it borrowed.
        let mixed = PyTuple::new(
            py,
            [thing(), other_class.call0().expect("Other()"), thing()],
        )
        .expect("a tuple builds")
        .into_any();
        module
            .setattr("held", &mixed)
            .expect("the module holds the tuple");
        assert!(!decide(py, &schema, &mixed, &pool, &[]));
    });
}

/// A sequence whose elements are not one scalar kind is walked element by
/// element, and that walk answers the same way.
///
/// The scalar loop covers the shapes it can, and everything else -- an element
/// schema that is itself a container, a JSON array of them, a tuple of them --
/// takes the general walk. The rules are the same rules: one element outside
/// the schema makes the value a non-member, and a walk that stops at the first
/// failure must not stop before it.
#[test]
fn a_sequence_of_a_container_element_is_walked_element_by_element() {
    Python::attach(|py| {
        let rows = |element| Schema::list(SeqShape::homogeneous(element));
        let list_of_lists = rows(Schema::list(SeqShape::homogeneous(Schema::Int)));

        // Over a parsed JSON array, where the element is not a scalar.
        let array = |items: Vec<JsonValue<'static>>| JsonValue::Array(std::sync::Arc::new(items));
        let good = array(vec![
            array(vec![JsonValue::Int(1)]),
            array(vec![JsonValue::Int(2)]),
        ]);
        assert!(holds_json(py, &list_of_lists, &good));
        let bad = array(vec![
            array(vec![JsonValue::Int(1)]),
            array(vec![JsonValue::Str("x".into())]),
        ]);
        assert!(!holds_json(py, &list_of_lists, &bad));

        // And over a tuple of them, which is the third of the three loops.
        let tuple_of_lists = Schema::tuple(SeqShape::homogeneous(Schema::list(
            SeqShape::homogeneous(Schema::Int),
        )));
        let int = |n: i64| n.into_pyobject(py).expect("an int").into_any();
        let one = PyList::new(py, [int(1)]).expect("a list builds").into_any();
        let two = PyList::new(py, [int(2)]).expect("a list builds").into_any();
        let text = PyList::new(py, [PyString::new(py, "x").into_any()])
            .expect("a list builds")
            .into_any();
        let good = PyTuple::new(py, [one.clone(), two])
            .expect("a tuple builds")
            .into_any();
        case(py, &tuple_of_lists, &good, true);
        let bad = PyTuple::new(py, [one, text])
            .expect("a tuple builds")
            .into_any();
        case(py, &tuple_of_lists, &bad, false);
    });
}

/// A length bound on a container is read before the elements it bounds.
///
/// A container too long for its bound is refused by the bound alone, as a fixed
/// shape refuses one by its arity: the wrong element inside it is never read.
/// One row per kind the walk counts, through both inputs, and a value of another
/// kind is refused by the base first, since a bound is asked only of a value
/// the base would read.
#[test]
fn a_length_bound_is_read_before_the_elements_it_bounds() {
    Python::attach(|py| {
        let capped = |base: Schema| Schema::Refine {
            base: Arc::new(base),
            constraints: vec![Constraint::MaxLen(2)].into(),
        };
        let ints = || Schema::list(SeqShape::homogeneous(Schema::Int));
        let mapping = || {
            Schema::mapping(MapClause {
                key: Schema::Str,
                value: Schema::Int,
            })
        };
        let codes = |violations: &[Violation]| -> Vec<(&str, usize)> {
            violations.iter().map(|v| (v.code, v.path.len())).collect()
        };
        let items = [
            PyInt::new(py, 1i64).into_any(),
            PyString::new(py, "x").into_any(),
            PyInt::new(py, 3i64).into_any(),
        ];
        let dict = PyDict::new(py);
        for (key, item) in ["a", "b", "c"].into_iter().zip(&items) {
            dict.set_item(key, item).expect("a dict takes an item");
        }
        let rows = [
            (ints(), PyList::new(py, &items).expect("a list").into_any()),
            (
                Schema::tuple(SeqShape::homogeneous(Schema::Int)),
                PyTuple::new(py, &items).expect("a tuple").into_any(),
            ),
            (
                Schema::set(Schema::Int),
                PySet::new(py, &items).expect("a set").into_any(),
            ),
            (
                Schema::frozen_set(Schema::Int),
                PyFrozenSet::new(py, &items)
                    .expect("a frozenset")
                    .into_any(),
            ),
            (mapping(), dict.into_any()),
        ];
        for (base, value) in rows {
            let schema = capped(base);
            let (ok, violations) = explain(py, &schema, &value, &[], &[]);
            assert!(!ok, "{value} is too long");
            assert_eq!(codes(&violations), [("too_long", 0)], "{value}");
            assert!(!holds(py, &schema, &value, &[], &[]));
        }
        let text = || JsonValue::Str("x".into());
        let array = JsonValue::Array(Arc::new(vec![JsonValue::Int(1), text(), JsonValue::Int(3)]));
        let object = json_object(vec![
            ("a", JsonValue::Int(1)),
            ("b", text()),
            ("c", JsonValue::Int(3)),
        ]);
        for (base, json) in [(ints(), &array), (mapping(), &object)] {
            let (ok, violations) = explain_json(py, &capped(base), json);
            assert!(!ok);
            assert_eq!(codes(&violations), [("too_long", 0)]);
        }
        // Another kind is the base's to refuse, and a container inside the
        // bound has its elements read.
        let long_text = PyString::new(py, "abcdef").into_any();
        let (_, violations) = explain(py, &capped(ints()), &long_text, &[], &[]);
        assert_eq!(codes(&violations), [("list_type", 0)]);
        let short = PyList::new(py, &items[..2]).expect("a list").into_any();
        let (_, violations) = explain(py, &capped(ints()), &short, &[], &[]);
        assert_eq!(codes(&violations), [("int_type", 1)]);
        let fits = PyList::new(py, [1i64, 2]).expect("a list").into_any();
        assert!(decide(py, &capped(ints()), &fits, &[], &[]));
        // A lower bound is read at the same step: a list too short for it is
        // refused by the bound, whatever it holds.
        let at_least = Schema::Refine {
            base: Arc::new(ints()),
            constraints: vec![Constraint::MinLen(4)].into(),
        };
        let (_, violations) = explain(py, &at_least, &short, &[], &[]);
        assert_eq!(codes(&violations), [("too_short", 0)]);
        let enough = PyList::new(py, [1i64, 2, 3, 4]).expect("a list").into_any();
        assert!(decide(py, &at_least, &enough, &[], &[]));
    });
}

/// A JSON array of one scalar kind takes the same loop as a Python list.
#[test]
fn a_json_array_of_one_scalar_kind_rejects_an_element_that_is_not_one() {
    Python::attach(|py| {
        let list_of_int = Schema::list(SeqShape::homogeneous(Schema::Int));
        let good = JsonValue::Array(std::sync::Arc::new(vec![
            JsonValue::Int(1),
            JsonValue::Int(2),
        ]));
        assert!(holds_json(py, &list_of_int, &good));
        let bad = JsonValue::Array(std::sync::Arc::new(vec![
            JsonValue::Int(1),
            JsonValue::Str("x".into()),
        ]));
        assert!(!holds_json(py, &list_of_int, &bad));
    });
}

#[test]
fn a_keyed_map_separates_fields_from_the_catch_all() {
    Python::attach(|py| {
        let field = |name: &str, schema, required| Field {
            name: name.into(),
            schema,
            required,
        };
        let closed = Schema::record(
            vec![
                field("x", Schema::Int, true),
                field("y", Schema::Str, false),
            ],
            Openness::Closed,
        );
        let open = Schema::record(vec![field("x", Schema::Int, true)], Openness::Open);
        let mapping = Schema::mapping(MapClause {
            key: Schema::Str,
            value: Schema::Int,
        });

        let build = |pairs: &[(&str, Bound<'_, PyAny>)]| {
            let dict = PyDict::new(py);
            for (key, value) in pairs {
                dict.set_item(key, value).expect("set_item");
            }
            dict.into_any()
        };
        let int = |n: i64| PyInt::new(py, n).into_any();
        let text = |s: &str| PyString::new(py, s).into_any();

        // The required field must be present and match; the optional one need
        // not be present, but must match when it is.
        case(py, &closed, &build(&[("x", int(1))]), true);
        case(
            py,
            &closed,
            &build(&[("x", int(1)), ("y", text("a"))]),
            true,
        );
        case(py, &closed, &build(&[("x", int(1)), ("y", int(2))]), false);
        case(py, &closed, &build(&[("y", text("a"))]), false);
        case(py, &closed, &build(&[("x", text("a"))]), false);
        // A closed record forbids an undeclared key; an open one admits it.
        case(py, &closed, &build(&[("x", int(1)), ("z", int(2))]), false);
        case(py, &open, &build(&[("x", int(1)), ("z", int(2))]), true);
        // A pure mapping judges every key and value by the clause.
        case(py, &mapping, &build(&[("k", int(1))]), true);
        case(py, &mapping, &build(&[("k", text("a"))]), false);
        case(py, &mapping, &build(&[]), true);
        // Not a dict at all.
        case(py, &closed, &list_of(py, vec![1]), false);
    });
}

/// A closed record asks the value for the keys it declares, and the answer is
/// the one the general scan gives.
///
/// Four rules meet on that path: a declared key's value must match, a required
/// key must be there, an optional one need not be, and a key the record does
/// not declare makes the value a non-member however good the rest of it is.
#[test]
fn a_closed_record_answers_for_each_of_its_keys() {
    Python::attach(|py| {
        let record = Schema::record(
            vec![
                Field {
                    name: "a".into(),
                    schema: Schema::Int,
                    required: true,
                },
                Field {
                    name: "b".into(),
                    schema: Schema::Str,
                    required: false,
                },
            ],
            Openness::Closed,
        );
        let dict = |pairs: Vec<(&str, Bound<'_, PyAny>)>| {
            let value = PyDict::new(py);
            for (key, item) in pairs {
                value.set_item(key, item).expect("a fresh dict takes a key");
            }
            value.into_any()
        };
        let int = |n: i64| n.into_pyobject(py).expect("an int").into_any();
        let text = |s: &str| PyString::new(py, s).into_any();

        case(py, &record, &dict(vec![("a", int(1))]), true);
        case(
            py,
            &record,
            &dict(vec![("a", int(1)), ("b", text("x"))]),
            true,
        );
        // A declared key whose value is not the field's schema.
        case(py, &record, &dict(vec![("a", text("x"))]), false);
        case(
            py,
            &record,
            &dict(vec![("a", int(1)), ("b", int(2))]),
            false,
        );
        // A required key the value does not carry.
        case(py, &record, &dict(vec![("b", text("x"))]), false);
        case(py, &record, &dict(vec![]), false);
        // A key the record does not declare: closed means closed.
        case(
            py,
            &record,
            &dict(vec![("a", int(1)), ("z", int(2))]),
            false,
        );
    });
}

/// A violation says what the value was measured against, for every kind of
/// constraint. The message is built only on the failing path, so nothing
/// else pins its text: a `render` returning a constant would satisfy every
/// other test in this file.
#[test]
fn a_violation_names_the_constraint_the_value_failed() {
    Python::attach(|py| {
        let pool = vec![PyInt::new(py, 10).into_any().unbind()];
        let ten = OperandIx::new(0);
        let refine = |base: Schema, constraint: Constraint| Schema::Refine {
            base: Arc::new(base),
            constraints: vec![constraint].into(),
        };
        let int = |n: i64| PyInt::new(py, n).into_any();
        let text = |s: &str| PyString::new(py, s).into_any();

        // Each row is a constraint, a value that fails it, and the whole of
        // the message that failure must carry.
        for (schema, value, want) in [
            (refine(Schema::Int, Constraint::Ge(ten)), int(1), ">= 10"),
            (refine(Schema::Int, Constraint::Gt(ten)), int(1), "> 10"),
            (refine(Schema::Int, Constraint::Le(ten)), int(11), "<= 10"),
            (refine(Schema::Int, Constraint::Lt(ten)), int(11), "< 10"),
            (
                refine(Schema::Int, Constraint::MultipleOf(ten)),
                int(3),
                "a multiple of 10",
            ),
            (
                refine(Schema::Str, Constraint::MinLen(2)),
                text("a"),
                "length >= 2",
            ),
            (
                refine(Schema::Str, Constraint::MaxLen(1)),
                text("abc"),
                "length <= 1",
            ),
            (
                refine(Schema::Str, Constraint::Regex("[0-9]+".to_owned())),
                text("x"),
                "a string matching '[0-9]+'",
            ),
        ] {
            let (ok, violations) = explain(py, &schema, &value, &pool, &[]);
            assert!(!ok, "{want}: the value must fail for a message to exist");
            let [violation] = violations.as_slice() else {
                panic!("{want}: expected exactly one violation, got {violations:?}")
            };
            assert_eq!(violation.expected, want);
        }
    });
}

#[test]
fn the_boolean_combinators_compose_the_member_sets() {
    Python::attach(|py| {
        let int = PyInt::new(py, 1i64).into_any();
        let text = PyString::new(py, "x").into_any();
        let float = PyFloat::new(py, 1.5).into_any();

        let union = Schema::Union(vec![Schema::Int, Schema::Str].into());
        case(py, &union, &int, true);
        case(py, &union, &text, true);
        case(py, &union, &float, false);

        let intersection = Schema::Intersection(
            vec![Schema::Int, Schema::Complement(Arc::new(Schema::Bool))].into(),
        );
        case(py, &intersection, &int, true);
        let truth = PyBool::new(py, true).to_owned().into_any();
        case(py, &intersection, &truth, false);

        let complement = Schema::Complement(Arc::new(Schema::Int));
        case(py, &complement, &int, false);
        case(py, &complement, &text, true);
        // Double negation returns the original set.
        let doubled = Schema::Complement(Arc::new(complement));
        case(py, &doubled, &int, true);
        case(py, &doubled, &text, false);
    });
}

#[test]
fn a_refinement_narrows_its_base_by_every_constraint() {
    Python::attach(|py| {
        let five = PyInt::new(py, 5i64).into_any();
        let pool = vec![five.clone().unbind()];
        let refine = |constraints: Vec<Constraint>| Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: constraints.into(),
        };
        let int = |n: i64| PyInt::new(py, n).into_any();

        // Each comparison arm, at and around its bound.
        for (constraint, at, above, below) in [
            (Constraint::Ge(OperandIx::new(0)), true, true, false),
            (Constraint::Gt(OperandIx::new(0)), false, true, false),
            (Constraint::Le(OperandIx::new(0)), true, false, true),
            (Constraint::Lt(OperandIx::new(0)), false, false, true),
        ] {
            let schema = refine(vec![constraint]);
            assert_eq!(decide(py, &schema, &int(5), &pool, &[]), at);
            assert_eq!(decide(py, &schema, &int(6), &pool, &[]), above);
            assert_eq!(decide(py, &schema, &int(4), &pool, &[]), below);
        }

        // The base is checked first: a bound on a non-int rejects rather than
        // raising through the comparison.
        let ge = refine(vec![Constraint::Ge(OperandIx::new(0))]);
        let text = PyString::new(py, "x").into_any();
        assert!(!decide(py, &ge, &text, &pool, &[]));

        // A multiple-of divides; length bounds measure `len`.
        let multiple = refine(vec![Constraint::MultipleOf(OperandIx::new(0))]);
        assert!(decide(py, &multiple, &int(10), &pool, &[]));
        assert!(!decide(py, &multiple, &int(11), &pool, &[]));

        let sized = Schema::Refine {
            base: Arc::new(Schema::Str),
            constraints: vec![Constraint::MinLen(2), Constraint::MaxLen(3)].into(),
        };
        for (text, want) in [("a", false), ("ab", true), ("abc", true), ("abcd", false)] {
            let value = PyString::new(py, text).into_any();
            assert_eq!(decide(py, &sized, &value, &pool, &[]), want, "{text:?}");
        }

        // A pattern is anchored: `re.fullmatch` semantics, not a search.
        let pattern = Schema::Refine {
            base: Arc::new(Schema::Str),
            constraints: vec![Constraint::Regex("a+".to_owned())].into(),
        };
        for (text, want) in [("a", true), ("aaa", true), ("ab", false), ("ba", false)] {
            let value = PyString::new(py, text).into_any();
            assert_eq!(decide(py, &pattern, &value, &pool, &[]), want, "{text:?}");
        }

        // Every constraint must hold, not merely one.
        let both = refine(vec![
            Constraint::Ge(OperandIx::new(0)),
            Constraint::Le(OperandIx::new(0)),
        ]);
        assert!(decide(py, &both, &int(5), &pool, &[]));
        assert!(!decide(py, &both, &int(6), &pool, &[]));
    });
}

#[test]
fn a_reference_unfolds_its_definition_and_a_cycle_is_refused() {
    Python::attach(|py| {
        // `T = None | {"next": T}`: a finite chain is a member.
        let defs = vec![Schema::Union(
            vec![
                Schema::NoneType,
                Schema::record(
                    vec![Field {
                        name: "next".into(),
                        schema: Schema::Ref(DefIx::new(0)),
                        required: true,
                    }],
                    Openness::Closed,
                ),
            ]
            .into(),
        )];
        let schema = Schema::Ref(DefIx::new(0));

        let none = py.None().into_bound(py);
        assert!(decide(py, &schema, &none, &[], &defs));

        let one = PyDict::new(py);
        one.set_item("next", py.None()).expect("set_item");
        assert!(decide(py, &schema, &one.clone().into_any(), &[], &defs));

        let two = PyDict::new(py);
        two.set_item("next", &one).expect("set_item");
        assert!(decide(py, &schema, &two.into_any(), &[], &defs));

        let wrong = PyDict::new(py);
        wrong.set_item("next", 1i64).expect("set_item");
        assert!(!decide(py, &schema, &wrong.into_any(), &[], &defs));

        // A value that contains itself is refused rather than looped on.
        let cyclic = PyDict::new(py);
        cyclic.set_item("next", &cyclic).expect("set_item");
        assert!(!decide(py, &schema, &cyclic.into_any(), &[], &defs));

        // A reference past the definitions table is an internal invariant
        // break, and degrades to a non-member rather than panicking. Checked
        // in release only: the walk `debug_assert`s it.
        #[cfg(not(debug_assertions))]
        assert!(!decide(py, &Schema::Ref(DefIx::new(9)), &none, &[], &defs));
    });
}

/// Define a small class hierarchy in the embedded interpreter, for the two
/// class-based arms. `Point` carries `x: int` and `y: int`; `Sub` is a
/// subclass of it; `Other` is unrelated.
fn classes(py: Python<'_>) -> Bound<'_, PyAny> {
    let module = PyModule::from_code(
        py,
        std::ffi::CString::new(
            "class Point:\n\
             \x20   def __init__(self, x, y):\n\
             \x20       self.x = x\n\
             \x20       self.y = y\n\
             class Sub(Point):\n\
             \x20   pass\n\
             class Other:\n\
             \x20   pass\n\
             class NoAttrs:\n\
             \x20   pass\n",
        )
        .expect("no interior nul")
        .as_c_str(),
        std::ffi::CString::new("classes.py")
            .expect("no interior nul")
            .as_c_str(),
        std::ffi::CString::new("classes")
            .expect("no interior nul")
            .as_c_str(),
    )
    .expect("the module compiles");
    module.into_any()
}

#[test]
fn an_instance_atom_admits_the_class_and_its_subclasses() {
    Python::attach(|py| {
        let module = classes(py);
        let point_class = module.getattr("Point").expect("Point");
        let sub_class = module.getattr("Sub").expect("Sub");
        let other_class = module.getattr("Other").expect("Other");
        let pool = vec![point_class.clone().unbind()];
        let schema = Schema::Instance(ClassIx::new(0));

        let point = point_class.call1((1i64, 2i64)).expect("Point(1, 2)");
        let sub = sub_class.call1((1i64, 2i64)).expect("Sub(1, 2)");
        let other = other_class.call0().expect("Other()");

        // `isinstance`, so a subclass instance is a member and an unrelated
        // one is not. A non-object value is not a member either.
        assert!(decide(py, &schema, &point, &pool, &[]));
        assert!(decide(py, &schema, &sub, &pool, &[]));
        assert!(!decide(py, &schema, &other, &pool, &[]));
        assert!(!decide(
            py,
            &schema,
            &PyInt::new(py, 1i64).into_any(),
            &pool,
            &[]
        ));
        // The class itself is not one of its instances.
        assert!(!decide(py, &schema, &point_class, &pool, &[]));
    });
}

#[test]
fn an_attribute_record_checks_the_class_then_every_attribute() {
    Python::attach(|py| {
        let module = classes(py);
        let point_class = module.getattr("Point").expect("Point");
        let other_class = module.getattr("Other").expect("Other");
        let bare_class = module.getattr("NoAttrs").expect("NoAttrs");
        let pool = vec![point_class.clone().unbind(), bare_class.clone().unbind()];
        let field = |name: &str, schema| Field {
            name: name.into(),
            schema,
            required: true,
        };
        let object = |class, fields| {
            Schema::meet([
                Schema::Instance(ClassIx::new(class)),
                Schema::AttrRecord { fields },
            ])
        };
        let schema = object(
            0,
            vec![field("x", Schema::Int), field("y", Schema::Int)].into(),
        );

        let good = point_class.call1((1i64, 2i64)).expect("Point(1, 2)");
        assert!(decide(py, &schema, &good, &pool, &[]));

        // Every attribute must match: one wrong value rejects the whole.
        let text = PyString::new(py, "x").into_any();
        let wrong = point_class.call1((1i64, text)).expect("Point(1, \"x\")");
        assert!(!decide(py, &schema, &wrong, &pool, &[]));

        // The isinstance check is not rescued by the attributes matching: an
        // unrelated object carrying x and y is still not a Point.
        let impostor = other_class.call0().expect("Other()");
        impostor.setattr("x", 1i64).expect("setattr x");
        impostor.setattr("y", 2i64).expect("setattr y");
        assert!(!decide(py, &schema, &impostor, &pool, &[]));

        // A missing attribute is a rejection, not a raise.
        let missing = object(1, vec![field("absent", Schema::Int)].into());
        let bare = bare_class.call0().expect("NoAttrs()");
        assert!(!decide(py, &missing, &bare, &pool, &[]));

        // The class atom is what the frontend emits when nothing is declared.
        let nominal = Schema::Instance(ClassIx::new(0));
        assert!(decide(py, &nominal, &good, &pool, &[]));
        assert!(!decide(py, &nominal, &impostor, &pool, &[]));

        // An optional attribute is satisfied by its absence, and still
        // checked when the value carries it. No annotation builds one -- a
        // declared attribute is one an instance has -- so the record's own
        // denotation is what holds the walk to it.
        let optional = Schema::AttrRecord {
            fields: vec![Field {
                name: "absent".into(),
                schema: Schema::Int,
                required: false,
            }]
            .into(),
        };
        assert!(decide(py, &optional, &bare, &pool, &[]));
        bare.setattr("absent", "not an int")
            .expect("setattr absent");
        assert!(!decide(py, &optional, &bare, &pool, &[]));
    });
}

/// Decide membership and report whether a fatal interpreter signal was
/// recorded on the way. The signal is what the entry point re-raises, so a
/// corpus that only reads the verdict cannot tell a refused value from an
/// interrupted walk.
fn decide_with_fatal(
    py: Python<'_>,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    pool: &[Py<PyAny>],
) -> (bool, bool) {
    let index = build_index(py, schema, &[], pool);
    let state = WalkState::new();
    let ctx = Ctx {
        pool,
        defs: &[],
        records: &index.records,
        attrs: &index.attrs,
        unions: &index.unions,
        regexes: &index.regexes,
        guard: &state.guard,
        depth: &state.depth,
        fatal: &state.fatal,
        fatal_seen: &state.fatal_seen,
        mode: WalkMode::Fast,
    };
    let ok = member(
        schema,
        &Value::Py(value),
        &mut Frame::new(&mut Vec::new(), &mut Vec::new(), ctx),
    );
    (ok, state.fatal.borrow().is_some())
}

/// Decide membership of a parsed JSON value, in the mode `is_valid_json` uses.
fn holds_json(py: Python<'_>, schema: &Schema, json: &JsonValue<'_>) -> bool {
    walk_json(py, schema, json, WalkMode::Fast).0
}

/// The same decision in explain mode, with the violations it aggregated.
fn explain_json(py: Python<'_>, schema: &Schema, json: &JsonValue<'_>) -> (bool, Vec<Violation>) {
    walk_json(py, schema, json, WalkMode::Explain)
}

/// Walk a parsed JSON value in `mode`.
fn walk_json(
    py: Python<'_>,
    schema: &Schema,
    json: &JsonValue<'_>,
    mode: WalkMode,
) -> (bool, Vec<Violation>) {
    let index = build_index(py, schema, &[], &[]);
    let state = WalkState::new();
    let ctx = Ctx {
        pool: &[],
        defs: &[],
        records: &index.records,
        attrs: &index.attrs,
        unions: &index.unions,
        regexes: &index.regexes,
        guard: &state.guard,
        depth: &state.depth,
        fatal: &state.fatal,
        fatal_seen: &state.fatal_seen,
        mode,
    };
    let mut out = Vec::new();
    let ok = member(
        schema,
        &Value::Json(py, json),
        &mut Frame::new(&mut Vec::new(), &mut out, ctx),
    );
    (ok, out)
}

fn json_object<'a>(pairs: Vec<(&'a str, JsonValue<'a>)>) -> JsonValue<'a> {
    JsonValue::Object(std::sync::Arc::new(
        pairs
            .into_iter()
            .map(|(k, v)| (Cow::Borrowed(k), v))
            .collect(),
    ))
}

fn field(name: &str, schema: Schema, required: bool) -> Field {
    Field {
        name: name.into(),
        schema,
        required,
    }
}

#[test]
fn a_json_keyed_map_decides_like_its_object_form() {
    Python::attach(|py| {
        // The JSON path has its own keyed-map walk -- it reads entries in
        // document order rather than a dict -- so every rule the object path
        // holds is asserted against it separately.
        let closed = Schema::record(
            vec![
                field("x", Schema::Int, true),
                field("y", Schema::Str, false),
            ],
            Openness::Closed,
        );
        let open = Schema::record(vec![field("x", Schema::Int, true)], Openness::Open);
        // A record with fields *and* a catch-all: the shape the closed record's
        // key-by-key path does not answer, so the document search still does.
        let mixed = Schema::keyed_map(
            vec![
                field("x", Schema::Int, true),
                field("y", Schema::Str, false),
            ],
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        );
        let mapping = Schema::mapping(MapClause {
            key: Schema::Str,
            value: Schema::Int,
        });

        for (schema, entries, want) in [
            // The required field must be present and match.
            (&closed, vec![("x", JsonValue::Int(1))], true),
            (&closed, vec![("y", JsonValue::Str("a".into()))], false),
            (&closed, vec![("x", JsonValue::Str("a".into()))], false),
            // The optional one may be absent, and must match when present.
            (
                &closed,
                vec![("x", JsonValue::Int(1)), ("y", JsonValue::Str("a".into()))],
                true,
            ),
            (
                &closed,
                vec![("x", JsonValue::Int(1)), ("y", JsonValue::Int(2))],
                false,
            ),
            // A closed record forbids an undeclared key; an open one admits it.
            (
                &closed,
                vec![("x", JsonValue::Int(1)), ("z", JsonValue::Int(2))],
                false,
            ),
            (
                &open,
                vec![("x", JsonValue::Int(1)), ("z", JsonValue::Int(2))],
                true,
            ),
            // The mixed record, whose required-ness the document search reads:
            // a required key must be there, an optional one need not be, and a
            // key neither names is the catch-all's to judge.
            (&mixed, vec![("x", JsonValue::Int(1))], true),
            (
                &mixed,
                vec![("x", JsonValue::Int(1)), ("y", JsonValue::Str("a".into()))],
                true,
            ),
            (&mixed, vec![("y", JsonValue::Str("a".into()))], false),
            (
                &mixed,
                vec![("x", JsonValue::Int(1)), ("k", JsonValue::Int(2))],
                true,
            ),
            (
                &mixed,
                vec![("x", JsonValue::Int(1)), ("k", JsonValue::Str("a".into()))],
                false,
            ),
            // A pure mapping judges every key and value by the clause.
            (&mapping, vec![("k", JsonValue::Int(1))], true),
            (&mapping, vec![("k", JsonValue::Str("a".into()))], false),
            (&mapping, vec![], true),
        ] {
            let json = json_object(entries.clone());
            assert_eq!(holds_json(py, schema, &json), want, "{entries:?}");
        }

        // A duplicate key takes its LAST value, which is what `json.loads`
        // would have produced, so the two input paths cannot disagree here.
        let last_wins = json_object(vec![
            ("x", JsonValue::Str("a".into())),
            ("x", JsonValue::Int(1)),
        ]);
        assert!(holds_json(py, &closed, &last_wins));
        let last_loses = json_object(vec![
            ("x", JsonValue::Int(1)),
            ("x", JsonValue::Str("a".into())),
        ]);
        assert!(!holds_json(py, &closed, &last_loses));
        // The same rule for a key the default clause covers.
        let default_last = json_object(vec![
            ("k", JsonValue::Int(1)),
            ("k", JsonValue::Str("a".into())),
        ]);
        assert!(!holds_json(py, &mapping, &default_last));
    });
}

#[test]
fn a_composite_rejects_when_any_one_element_fails() {
    Python::attach(|py| {
        // Each container folds its element verdicts with a conjunction. A
        // disjunction there accepts a container whose first element happens
        // to match, which is the shape a single all-good case cannot see --
        // so every container is driven with a value that is part-good.
        let int_text = PyList::new(py, [1i64]).expect("builds");
        int_text.append(PyString::new(py, "x")).expect("append");

        let list_schema = Schema::list(SeqShape::homogeneous(Schema::Int));
        case(py, &list_schema, &int_text.clone().into_any(), false);

        let tuple_schema = Schema::tuple(SeqShape::homogeneous(Schema::Int));
        let tuple = PyTuple::new(py, [1i64, 2]).expect("builds").into_any();
        case(py, &tuple_schema, &tuple, true);
        let mixed_tuple = int_text.to_tuple().into_any();
        case(py, &tuple_schema, &mixed_tuple, false);

        // Both container kinds on the JSON path too, where the fold is a
        // separate arm.
        let good = JsonValue::Array(std::sync::Arc::new(vec![
            JsonValue::Int(1),
            JsonValue::Int(2),
        ]));
        let part_good = JsonValue::Array(std::sync::Arc::new(vec![
            JsonValue::Int(1),
            JsonValue::Str("x".into()),
        ]));
        assert!(holds_json(py, &list_schema, &good));
        assert!(!holds_json(py, &list_schema, &part_good));
        // JSON has no tuple: `json.loads` produces a list, so the JSON path
        // has no tuple arm at all and a tuple schema rejects an array
        // whatever its elements are. Pinned here because it is the one place
        // the two input paths deliberately decide differently.
        assert!(!holds_json(py, &tuple_schema, &good));
        assert!(!holds_json(py, &tuple_schema, &part_good));
        assert!(holds(py, &tuple_schema, &tuple, &[], &[]));

        // Sets and frozensets fold the same way.
        let mixed_set = PySet::new(py, [1i64]).expect("builds");
        mixed_set.add(PyString::new(py, "x")).expect("add");
        case(
            py,
            &Schema::set(Schema::Int),
            &mixed_set.clone().into_any(),
            false,
        );
        let mixed_frozen = PyFrozenSet::new(py, mixed_set.iter())
            .expect("builds")
            .into_any();
        case(py, &Schema::frozen_set(Schema::Int), &mixed_frozen, false);
        let good_frozen = PyFrozenSet::new(py, [1i64, 2]).expect("builds").into_any();
        case(py, &Schema::frozen_set(Schema::Int), &good_frozen, true);
    });
}

#[test]
fn a_union_explains_the_branch_that_descended_furthest() {
    Python::attach(|py| {
        // No branch matches, and the two fail at different depths: one is a
        // flat type mismatch, the other descends into a field. The report is
        // the deeper branch's, so a reader is shown the branch the value was
        // closest to rather than every branch's noise.
        let deep = Schema::record(vec![field("x", Schema::Int, true)], Openness::Closed);
        let schema = Schema::Union(vec![Schema::Int, deep].into());
        let value = PyDict::new(py);
        value.set_item("x", PyString::new(py, "a")).expect("set");
        let (ok, violations) = explain(py, &schema, &value.into_any(), &[], &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].location(), "x");
        assert_ne!(violations[0].code, "union_error");

        // No branch makes any progress: a single union error, not two flat
        // mismatches. This is the arm the depth comparison selects between.
        let flat = Schema::Union(vec![Schema::Int, Schema::Str].into());
        let number = PyFloat::new(py, 1.5).into_any();
        let (ok, violations) = explain(py, &flat, &number, &[], &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].code, "union_error");

        // Stopping at the first violation, the branch is walked to its first
        // failure, which is all the choice between branches reads.
        let index = build_index(py, &schema, &[], &[]);
        let deep_value = PyDict::new(py);
        deep_value
            .set_item("x", PyString::new(py, "a"))
            .expect("set");
        let deep_value = deep_value.into_any();
        let state = WalkState::new();
        let mut out = Vec::new();
        let ctx = Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::ExplainFailFast,
        };
        let ok = member(
            &schema,
            &Value::Py(&deep_value),
            &mut Frame::new(&mut Vec::new(), &mut out, ctx),
        );
        assert!(!ok);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].location(), "x");

        // The caller's mode decides how much of the chosen branch is walked
        // and reported: `fail_fast` is one violation, which is what the error
        // model promises at every site. A branch with two failing fields is
        // what tells the two modes apart -- one is reported, then both.
        let wide = Schema::record(
            vec![field("p", Schema::Int, true), field("q", Schema::Int, true)],
            Openness::Closed,
        );
        let union_wide = Schema::Union(vec![Schema::Int, wide].into());
        let wide_value = PyDict::new(py);
        for key in ["p", "q"] {
            wide_value
                .set_item(key, PyString::new(py, "s"))
                .expect("set");
        }
        let wide_value = wide_value.into_any();
        assert_eq!(
            run_mode(py, &union_wide, &wide_value, WalkMode::ExplainFailFast),
            (false, 1)
        );
        // And the aggregating mode beside it reports both.
        assert_eq!(
            run_mode(py, &union_wide, &wide_value, WalkMode::Explain),
            (false, 2)
        );

        // A tie keeps the earliest branch, so the choice is deterministic.
        let left = Schema::record(vec![field("a", Schema::Int, true)], Openness::Closed);
        let right = Schema::record(vec![field("b", Schema::Int, true)], Openness::Closed);
        let tied = Schema::Union(vec![left, right].into());
        let value = PyDict::new(py);
        value.set_item("a", PyString::new(py, "s")).expect("set");
        value.set_item("b", PyString::new(py, "s")).expect("set");
        let (_, violations) = explain(py, &tied, &value.into_any(), &[], &[]);
        assert_eq!(violations[0].location(), "a");
    });
}

#[test]
fn an_explaining_walk_aggregates_every_independent_failure() {
    Python::attach(|py| {
        // Three fields fail independently. Explain mode reports all three;
        // fail-fast reports the first; the fast path reports none and
        // allocates nothing. All three modes agree on the verdict.
        let schema = Schema::record(
            vec![
                field("a", Schema::Int, true),
                field("b", Schema::Int, true),
                field("c", Schema::Int, true),
            ],
            Openness::Closed,
        );
        let value = PyDict::new(py);
        for key in ["a", "b", "c"] {
            value.set_item(key, PyString::new(py, "s")).expect("set");
        }
        let value = value.into_any();

        let index = build_index(py, &schema, &[], &[]);
        let run = |mode: WalkMode| {
            let state = WalkState::new();
            let ctx = Ctx {
                pool: &[],
                defs: &[],
                records: &index.records,
                attrs: &index.attrs,
                unions: &index.unions,
                regexes: &index.regexes,
                guard: &state.guard,
                depth: &state.depth,
                fatal: &state.fatal,
                fatal_seen: &state.fatal_seen,
                mode,
            };
            let mut out = Vec::new();
            let ok = member(
                &schema,
                &Value::Py(&value),
                &mut Frame::new(&mut Vec::new(), &mut out, ctx),
            );
            (ok, out.len())
        };
        assert_eq!(run(WalkMode::Explain), (false, 3));
        assert_eq!(run(WalkMode::ExplainFailFast), (false, 1));
        assert_eq!(run(WalkMode::Fast), (false, 0));
    });
}

/// Run one membership walk in a given mode and report the verdict and how
/// many violations it aggregated.
fn run_mode(
    py: Python<'_>,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    mode: WalkMode,
) -> (bool, usize) {
    let index = build_index(py, schema, &[], &[]);
    let state = WalkState::new();
    let ctx = Ctx {
        pool: &[],
        defs: &[],
        records: &index.records,
        attrs: &index.attrs,
        unions: &index.unions,
        regexes: &index.regexes,
        guard: &state.guard,
        depth: &state.depth,
        fatal: &state.fatal,
        fatal_seen: &state.fatal_seen,
        mode,
    };
    let mut out = Vec::new();
    let ok = member(
        schema,
        &Value::Py(value),
        &mut Frame::new(&mut Vec::new(), &mut out, ctx),
    );
    (ok, out.len())
}

#[test]
fn a_composite_stops_at_its_first_failing_child_only_when_asked_to() {
    Python::attach(|py| {
        // Two elements fail independently. Aggregating mode reports both;
        // fail-fast reports the first; the fast path reports none. Every
        // composite consults one predicate for this, so a sequence pins it
        // for the arms a record does not reach.
        let schema = Schema::list(SeqShape::homogeneous(Schema::Int));
        let value = PyList::new(py, [1i64]).expect("builds");
        value.append(PyString::new(py, "a")).expect("append");
        value.append(PyString::new(py, "b")).expect("append");
        let value = value.into_any();

        assert_eq!(run_mode(py, &schema, &value, WalkMode::Explain), (false, 2));
        assert_eq!(
            run_mode(py, &schema, &value, WalkMode::ExplainFailFast),
            (false, 1)
        );
        assert_eq!(run_mode(py, &schema, &value, WalkMode::Fast), (false, 0));

        // The same for a set, whose fold is a separate arm.
        let set_schema = Schema::set(Schema::Int);
        let set = PySet::new(py, [1i64]).expect("builds");
        set.add(PyString::new(py, "a")).expect("add");
        set.add(PyString::new(py, "b")).expect("add");
        let set = set.into_any();
        assert_eq!(
            run_mode(py, &set_schema, &set, WalkMode::Explain),
            (false, 2)
        );
        assert_eq!(
            run_mode(py, &set_schema, &set, WalkMode::ExplainFailFast),
            (false, 1)
        );
    });
}

#[test]
fn a_raising_comparison_folds_and_a_fatal_signal_does_not() {
    Python::attach(|py| {
        // A value that cannot answer "are you in this set?" is not in it --
        // unless the interpreter is unwinding, which is not an answer at all.
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "class Rude:\n\
                 \x20   def __eq__(self, other):\n\
                 \x20       raise ValueError('no')\n\
                 class Stopping:\n\
                 \x20   def __eq__(self, other):\n\
                 \x20       raise KeyboardInterrupt\n\
                 class NoLen:\n\
                 \x20   def __len__(self):\n\
                 \x20       raise TypeError('no')\n\
                 class OutOfMemory:\n\
                 \x20   def __eq__(self, other):\n\
                 \x20       raise MemoryError\n\
                 class TooDeep:\n\
                 \x20   def __eq__(self, other):\n\
                 \x20       raise RecursionError\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("raising.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("raising")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");

        let literal = Schema::Literal(ConstIx::new(0));
        let instance = |name: &str| {
            module
                .getattr(name)
                .expect("the class")
                .call0()
                .expect("the instance")
        };

        // A literal's same-type test runs BEFORE `==`, so a value of another
        // type never reaches the comparison at all. Pinned first, because it
        // is why the two cases below have to pool an instance of the raising
        // class rather than an int.
        let one = PyInt::new(py, 1i64).into_any();
        let int_pool = vec![one.unbind()];
        let rude = instance("Rude");
        assert_eq!(
            decide_with_fatal(py, &literal, &rude, &int_pool),
            (false, false)
        );

        // An ordinary exception folds to a non-member, and records nothing.
        let rude_pool = vec![instance("Rude").unbind()];
        assert_eq!(
            decide_with_fatal(py, &literal, &rude, &rude_pool),
            (false, false)
        );

        // A fatal signal is recorded so the entry point re-raises it. The
        // local answer is still a non-member so the frame returns.
        let stopping = instance("Stopping");
        let stopping_pool = vec![instance("Stopping").unbind()];
        assert_eq!(
            decide_with_fatal(py, &literal, &stopping, &stopping_pool),
            (false, true)
        );

        // MemoryError and RecursionError ARE ordinary exceptions, so the
        // base-exception test alone misses them -- and they still mean the
        // interpreter cannot continue. Each is a separate disjunct of the
        // classifier, so each needs its own case.
        for name in ["OutOfMemory", "TooDeep"] {
            let value = instance(name);
            let pool = vec![instance(name).unbind()];
            assert_eq!(
                decide_with_fatal(py, &literal, &value, &pool),
                (false, true),
                "{name}"
            );
        }

        // The same split at a length bound, which reaches the value through
        // `__len__` rather than `__eq__`.
        let sized = Schema::Refine {
            base: Arc::new(Schema::ANYTHING),
            constraints: vec![Constraint::MinLen(1)].into(),
        };
        let no_len = module.getattr("NoLen").expect("NoLen").call0().expect("()");
        let (ok, fatal) = decide_with_fatal(py, &sized, &no_len, &[]);
        assert!(
            !ok,
            "a value whose __len__ raises cannot satisfy a length bound"
        );
        assert!(!fatal, "a TypeError is an ordinary exception, not a signal");
    });
}

#[test]
fn a_predicate_constraint_runs_the_pooled_callable() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new("def is_even(x):\n\x20   return x % 2 == 0\n")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("pred.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("pred")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let is_even = module.getattr("is_even").expect("is_even");
        let pool = vec![is_even.unbind()];
        let schema = Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
        };
        assert!(decide(
            py,
            &schema,
            &PyInt::new(py, 4i64).into_any(),
            &pool,
            &[]
        ));
        assert!(!decide(
            py,
            &schema,
            &PyInt::new(py, 3i64).into_any(),
            &pool,
            &[]
        ));
    });
}

#[test]
fn a_literal_union_decides_alike_through_the_fast_plan_and_the_scan() {
    Python::attach(|py| {
        // A union whose members are all literals is decided by a precomputed
        // set lookup on the membership path, and by the linear scan
        // everywhere else. The two must agree, and the plan must not be
        // consulted while explaining -- an early return there would report a
        // rejection with no violation behind it.
        let pool: Vec<Py<PyAny>> = (1i64..=3)
            .map(|n| PyInt::new(py, n).into_any().unbind())
            .collect();
        let schema = Schema::Union((0..3).map(|i| Schema::Literal(ConstIx::new(i))).collect());
        for n in 1i64..=3 {
            assert!(decide(
                py,
                &schema,
                &PyInt::new(py, n).into_any(),
                &pool,
                &[]
            ));
        }
        // Rejections, which are what an explain walk must produce a violation
        // for. `decide` runs both modes and holds them to agreeing.
        for n in [0i64, 4, 99] {
            assert!(!decide(
                py,
                &schema,
                &PyInt::new(py, n).into_any(),
                &pool,
                &[]
            ));
        }
        // A value of a type the plan does not cover falls to the scan.
        let text = PyString::new(py, "1").into_any();
        assert!(!decide(py, &schema, &text, &pool, &[]));
    });
}

#[test]
fn the_explain_pass_reports_only_the_fields_that_actually_failed() {
    Python::attach(|py| {
        // The explain pass re-walks a record that already failed. An absent
        // OPTIONAL field is not a failure, so it must not be reported -- and
        // the only way to see that is a record that fails for another reason
        // while an optional field is absent.
        let schema = Schema::record(
            vec![
                field("x", Schema::Int, true),
                field("y", Schema::Str, false),
            ],
            Openness::Closed,
        );
        let value = PyDict::new(py);
        value.set_item("x", PyString::new(py, "s")).expect("set");
        let (ok, violations) = explain(py, &schema, &value.into_any(), &[], &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].location(), "x");

        // A required field that IS absent is reported, and only once.
        let empty = PyDict::new(py);
        let (ok, violations) = explain(py, &schema, &empty.into_any(), &[], &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].code, "missing_key");
    });
}

#[test]
fn a_closed_record_reports_every_extra_key_unless_fail_fast_stops_it() {
    Python::attach(|py| {
        // Two undeclared keys, so the loop that reports them is driven past
        // its first iteration: aggregating mode reports both, fail-fast the
        // first only.
        let schema = Schema::record(vec![field("x", Schema::Int, true)], Openness::Closed);
        let value = PyDict::new(py);
        value.set_item("x", 1i64).expect("set");
        value.set_item("extra1", 1i64).expect("set");
        value.set_item("extra2", 1i64).expect("set");
        let value = value.into_any();

        let index = build_index(py, &schema, &[], &[]);
        let run = |mode: WalkMode| {
            let state = WalkState::new();
            let ctx = Ctx {
                pool: &[],
                defs: &[],
                records: &index.records,
                attrs: &index.attrs,
                unions: &index.unions,
                regexes: &index.regexes,
                guard: &state.guard,
                depth: &state.depth,
                fatal: &state.fatal,
                fatal_seen: &state.fatal_seen,
                mode,
            };
            let mut out = Vec::new();
            let ok = member(
                &schema,
                &Value::Py(&value),
                &mut Frame::new(&mut Vec::new(), &mut out, ctx),
            );
            (ok, out.len())
        };
        assert_eq!(run(WalkMode::Explain), (false, 2));
        assert_eq!(run(WalkMode::ExplainFailFast), (false, 1));
    });
}

/// A fatal signal raised by a key's own `__eq__` is kept for the entry point to
/// re-raise, on every path that reads one; an ordinary error there answers as
/// before. The key raises once, as an interrupt does, so a site that dropped it
/// would find the next reading answer and nothing re-raised.
#[test]
fn a_fatal_signal_propagates_from_a_key() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "class StoppingKey(str):\n\
                 \x20   __hash__ = str.__hash__\n\
                 \x20   def __eq__(self, other):\n\
                 \x20       if not self.__dict__.get('raised'):\n\
                 \x20           self.raised = True\n\
                 \x20           raise KeyboardInterrupt\n\
                 \x20       return str.__eq__(self, other)\n\
                 class RudeKey(str):\n\
                 \x20   __hash__ = str.__hash__\n\
                 \x20   def __eq__(self, other):\n\
                 \x20       raise ValueError('no')\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("stopping_keys.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("stopping_keys")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let keyed = |class: &str| {
            let dict = PyDict::new(py);
            let key = module
                .getattr(class)
                .expect("class")
                .call1(("a",))
                .expect("key");
            dict.set_item(key, 1i64).expect("set");
            dict.into_any()
        };
        // A closed record asks the dict for its declared keys; a record beside a
        // clause scans the entries and resolves each key to a field name.
        let closed = Schema::record(vec![field("a", Schema::Int, true)], Openness::Closed);
        let clausal = Schema::keyed_map(
            vec![field("a", Schema::Int, true)],
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        );
        for schema in [&closed, &clausal] {
            for (class, want_fatal) in [("RudeKey", false), ("StoppingKey", true)] {
                assert_eq!(
                    decide_with_fatal(py, schema, &keyed(class), &[]),
                    (false, want_fatal),
                    "{class} against {schema:?}"
                );
            }
        }
    });
}

/// A length bound reads a tuple subclass's length as `PyObject_Size` reads it,
/// which runs the type's own `__len__`, so it asks the type whether that
/// `__len__` is the tuple's own on every interpreter -- where the tuple walk
/// beside it asks nothing on `CPython`. Asking runs the metaclass: a fatal
/// signal it raises is kept for the entry point, and an ordinary error reads
/// the type as a liar, whose length is read through the base.
#[test]
fn a_length_bound_asks_a_tuple_subclass_its_type() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"class StoppingMeta(type):\n\
              \x20   def __getattribute__(cls, name):\n\
              \x20       if name == '__len__':\n\
              \x20           raise KeyboardInterrupt\n\
              \x20       return super().__getattribute__(name)\n\
              class RudeMeta(type):\n\
              \x20   def __getattribute__(cls, name):\n\
              \x20       if name == '__len__':\n\
              \x20           raise ValueError('no')\n\
              \x20       return super().__getattribute__(name)\n\
              class StoppingPair(tuple, metaclass=StoppingMeta):\n\
              \x20   pass\n\
              class RudePair(tuple, metaclass=RudeMeta):\n\
              \x20   pass\n",
            c"bounded_types.py",
            c"bounded_types",
        )
        .expect("the module compiles");
        let bounded = Schema::Refine {
            base: Arc::new(Schema::tuple(SeqShape::homogeneous(Schema::Int))),
            constraints: vec![Constraint::MinLen(1)].into(),
        };
        for (class, want) in [("RudePair", (true, false)), ("StoppingPair", (false, true))] {
            let pair = module
                .getattr(class)
                .expect("class")
                .call1(((1i64, 2i64),))
                .expect("pair");
            assert_eq!(decide_with_fatal(py, &bounded, &pair, &[]), want, "{class}");
        }
    });
}

/// A fatal signal raised by a type's metaclass when the walk asks for its
/// `__len__` or `__iter__` is kept for the entry point to re-raise; an ordinary
/// error there reads the type as a liar, and the value is read through its base.
///
/// Where the walk asks is the other half of the claim. A set subclass is
/// iterated where it lies only if its type's `__iter__` is the set's own, on
/// every interpreter. A tuple subclass is asked for its `__len__` on `PyPy`
/// alone: `CPython` reads the storage whatever the type overrides, so there the
/// metaclass never runs and the pair is read for what it holds.
#[test]
fn a_fatal_signal_propagates_from_a_type() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "class StoppingMeta(type):\n\
                 \x20   def __getattribute__(cls, name):\n\
                 \x20       if name == '__len__':\n\
                 \x20           raise KeyboardInterrupt\n\
                 \x20       return super().__getattribute__(name)\n\
                 class RudeMeta(type):\n\
                 \x20   def __getattribute__(cls, name):\n\
                 \x20       if name == '__len__':\n\
                 \x20           raise ValueError('no')\n\
                 \x20       return super().__getattribute__(name)\n\
                 class StoppingPair(tuple, metaclass=StoppingMeta):\n\
                 \x20   pass\n\
                 class RudePair(tuple, metaclass=RudeMeta):\n\
                 \x20   pass\n\
                 class StoppingIterMeta(type):\n\
                 \x20   def __getattribute__(cls, name):\n\
                 \x20       if name == '__iter__':\n\
                 \x20           raise KeyboardInterrupt\n\
                 \x20       return super().__getattribute__(name)\n\
                 class RudeIterMeta(type):\n\
                 \x20   def __getattribute__(cls, name):\n\
                 \x20       if name == '__iter__':\n\
                 \x20           raise ValueError('no')\n\
                 \x20       return super().__getattribute__(name)\n\
                 class StoppingBag(set, metaclass=StoppingIterMeta):\n\
                 \x20   pass\n\
                 class RudeBag(set, metaclass=RudeIterMeta):\n\
                 \x20   pass\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("stopping_types.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("stopping_types")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let ints = Schema::tuple(SeqShape::homogeneous(Schema::Int));
        let stopping = (!cfg!(PyPy), cfg!(PyPy));
        for (class, want) in [("RudePair", (true, false)), ("StoppingPair", stopping)] {
            let pair = module
                .getattr(class)
                .expect("class")
                .call1(((1i64, 2i64),))
                .expect("pair");
            assert_eq!(decide_with_fatal(py, &ints, &pair, &[]), want, "{class}");
        }
        let int_set = Schema::set(Schema::Int);
        for (class, want) in [("RudeBag", (true, false)), ("StoppingBag", (false, true))] {
            let bag = module
                .getattr(class)
                .expect("class")
                .call1(((1i64, 2i64),))
                .expect("bag");
            assert_eq!(decide_with_fatal(py, &int_set, &bag, &[]), want, "{class}");
        }
    });
}

/// A list is read for what it holds on every interpreter, whatever its type's
/// `__iter__` yields: the snapshot a narrow list is read through below 3.14 is
/// taken of an exact list only.
#[test]
fn a_list_subclass_is_read_for_what_it_holds() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "class Liar(list):\n\
                 \x20   def __iter__(self):\n\
                 \x20       return iter(['x'] * len(self))\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("liar.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("liar")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let ints: Vec<i64> = (0..40).collect();
        let liar = module
            .getattr("Liar")
            .expect("Liar")
            .call1((ints,))
            .expect("a list");
        let schema = Schema::list(SeqShape::homogeneous(Schema::Int));
        assert_eq!(decide_with_fatal(py, &schema, &liar, &[]), (true, false));
    });
}

#[test]
fn a_fatal_signal_propagates_from_an_attribute_and_from_a_predicate() {
    Python::attach(|py| {
        // The two sites the literal and length cases do not reach: attribute
        // access on an object schema, and a user predicate. Both fold an
        // ordinary exception to a non-member and record a fatal signal.
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "class Base:\n\
                 \x20   pass\n\
                 class RudeAttr(Base):\n\
                 \x20   def __getattr__(self, name):\n\
                 \x20       raise ValueError('no')\n\
                 class StoppingAttr(Base):\n\
                 \x20   def __getattr__(self, name):\n\
                 \x20       raise KeyboardInterrupt\n\
                 def rude(x):\n\
                 \x20   raise ValueError('no')\n\
                 def stopping(x):\n\
                 \x20   raise KeyboardInterrupt\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("fatal.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("fatal")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let base = module.getattr("Base").expect("Base");

        let attrs = Schema::meet([
            Schema::Instance(ClassIx::new(0)),
            Schema::AttrRecord {
                fields: vec![field("missing", Schema::Int, true)].into(),
            },
        ]);
        let pool = vec![base.clone().unbind()];
        for (name, want_fatal) in [("RudeAttr", false), ("StoppingAttr", true)] {
            let value = module.getattr(name).expect("class").call0().expect("()");
            assert_eq!(
                decide_with_fatal(py, &attrs, &value, &pool),
                (false, want_fatal),
                "{name}"
            );
        }

        let one = PyInt::new(py, 1i64).into_any();
        for (name, want_fatal) in [("rude", false), ("stopping", true)] {
            let predicate = module.getattr(name).expect("callable");
            let pool = vec![predicate.unbind()];
            let schema = Schema::Refine {
                base: Arc::new(Schema::Int),
                constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
            };
            assert_eq!(
                decide_with_fatal(py, &schema, &one, &pool),
                (false, want_fatal),
                "{name}"
            );
        }
    });
}

// SWEEP-SKIP: this case exists to prove a bound, so a mutation that removes
// the bound makes it run without end. It stays in the test lane and leaves
// the mutation sweep, where a run that returns no verdict is a rig fault.
#[test]
fn recursion_deeper_than_the_bound_is_refused() {
    Python::attach(|py| {
        // `T = None | {"next": T}`. A chain the walk can carry is a member; a
        // chain past the guard's depth bound is refused rather than recursed
        // into, because the walk descends one native frame per level.
        let defs = vec![Schema::Union(
            vec![
                Schema::NoneType,
                Schema::record(
                    vec![field("next", Schema::Ref(DefIx::new(0)), true)],
                    Openness::Closed,
                ),
            ]
            .into(),
        )];
        let schema = Schema::Ref(DefIx::new(0));

        let chain = |depth: usize| {
            let mut node = py.None().into_bound(py);
            for _ in 0..depth {
                let dict = PyDict::new(py);
                dict.set_item("next", &node).expect("set_item");
                node = dict.into_any();
            }
            node
        };
        assert!(decide(py, &schema, &chain(8), &[], &defs));
        assert!(decide(
            py,
            &schema,
            &chain(MAX_RECURSION_DEPTH - 1),
            &[],
            &defs
        ));
        assert!(!decide(
            py,
            &schema,
            &chain(MAX_RECURSION_DEPTH + 2),
            &[],
            &defs
        ));
    });
}

/// The JSON record path answers the same with the plan and without it.
///
/// Whether a key is a declared field is read from the per-validator record
/// plan, and a schema absent from that plan falls back to scanning the field
/// list. The fallback is what keeps correctness from depending on the index
/// being complete, so it has to answer the same -- and nothing exercises it
/// through the ordinary entry points, because the index is always built.
#[test]
fn the_json_record_path_agrees_with_and_without_its_plan() {
    Python::attach(|py| {
        let schema = Schema::keyed_map(
            vec![Field {
                name: "a".into(),
                schema: Schema::Int,
                required: true,
            }],
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Str,
            }],
        );
        let Schema::KeyedMap { fields, defaults } = &schema else {
            panic!("the schema is a keyed map")
        };

        // `a` is the declared field and takes an int; `b` is undeclared and
        // must go to the clause, which takes a string. A reading that
        // confused the two would accept the first and reject the second.
        let good = [
            ("a".into(), JsonValue::Int(1)),
            ("b".into(), JsonValue::Str("x".into())),
        ];
        let bad = [
            ("a".into(), JsonValue::Int(1)),
            ("b".into(), JsonValue::Int(2)),
        ];

        let built = build_index(py, &schema, &[], &[]);
        let empty = ValidatorIndex::default();
        for index in [&built, &empty] {
            let state = WalkState::new();
            let ctx = Ctx {
                pool: &[],
                defs: &[],
                records: &index.records,
                attrs: &index.attrs,
                unions: &index.unions,
                regexes: &index.regexes,
                guard: &state.guard,
                depth: &state.depth,
                fatal: &state.fatal,
                fatal_seen: &state.fatal_seen,
                mode: WalkMode::Fast,
            };
            assert!(keyed_map_matches_json(fields, defaults, py, &good, ctx));
            assert!(!keyed_map_matches_json(fields, defaults, py, &bad, ctx));
        }
        // The plan really was absent for the second pass, so the two answers
        // came from the two readings rather than from one of them twice.
        assert!(built.records.contains_key(&(fields.as_ptr() as usize)));
        assert!(empty.records.is_empty());
    });
}

#[test]
fn the_json_path_and_the_object_path_agree() {
    Python::attach(|py| {
        // The two input paths share one walk, so they must decide alike. This
        // drives the `Value::Json` arms the object corpus above never reaches.
        let schema = Schema::list(SeqShape::homogeneous(Schema::Int));
        let json = JsonValue::Array(std::sync::Arc::new(vec![
            JsonValue::Int(1),
            JsonValue::Int(2),
        ]));
        let index = build_index(py, &schema, &[], &[]);
        let state = WalkState::new();
        let ctx = Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Fast,
        };
        assert!(member(
            &schema,
            &Value::Json(py, &json),
            &mut Frame::new(&mut Vec::new(), &mut Vec::new(), ctx)
        ));
        assert!(holds(py, &schema, &list_of(py, vec![1, 2]), &[], &[]));

        let bad = JsonValue::Array(std::sync::Arc::new(vec![JsonValue::Str("x".into())]));
        assert!(!member(
            &schema,
            &Value::Json(py, &bad),
            &mut Frame::new(&mut Vec::new(), &mut Vec::new(), ctx)
        ));
    });
}

/// The scalar loop and the walk admit the same values.
///
/// `scalar_admits` states the scalar rules a second time so a homogeneous list
/// can test its elements without the walk's per-element bookkeeping -- the
/// difference between the two paths is 70% of a list of integers. Two
/// statements of one rule is how a value comes to be decided two ways, so this
/// asks both about every scalar schema and every kind of value the walk can be
/// handed, and requires the same answer.
#[test]
fn the_scalar_loop_and_the_walk_admit_the_same_values() {
    Python::initialize();
    Python::attach(|py| {
        let scalars = [
            Schema::ANYTHING,
            Schema::ANY,
            Schema::Nothing,
            Schema::NoneType,
            Schema::Bool,
            Schema::Int,
            Schema::Float,
            Schema::Str,
            Schema::Bytes,
        ];
        let values: Vec<Bound<'_, PyAny>> = vec![
            py.None().into_bound(py),
            PyBool::new(py, true).to_owned().into_any(),
            PyBool::new(py, false).to_owned().into_any(),
            0_i64.into_pyobject(py).expect("an int").into_any(),
            (-7_i64).into_pyobject(py).expect("an int").into_any(),
            PyFloat::new(py, 1.5).into_any(),
            PyString::new(py, "a").into_any(),
            PyBytes::new(py, b"a").into_any(),
            PyList::empty(py).into_any(),
            PyDict::new(py).into_any(),
        ];
        for schema in &scalars {
            let kind = scalar_of(schema).expect("every schema above is a scalar");
            for value in &values {
                let walked = holds(py, schema, value, &[], &[]);
                let looped = scalar_admits(kind, &Value::Py(value));
                assert_eq!(
                    walked, looped,
                    "the walk and the scalar loop disagree about {schema:?} and {value:?}"
                );
            }
        }
        // And the other direction: a schema the loop calls a scalar is one the
        // walk answers without descending, so nothing that needs a descent may
        // be named here.
        assert!(scalar_of(&Schema::list(SeqShape::homogeneous(Schema::Int))).is_none());
        assert!(scalar_of(&Schema::Literal(ConstIx::new(0))).is_none());
        assert!(scalar_of(&Schema::Instance(ClassIx::new(0))).is_none());
    });
}

/// A record open under `str: anything` -- the clause a `TypedDict` carries,
/// since the typing spec makes its keys strings -- is answered by its keys.
///
/// Reading a record by its declared keys was a closed record's path alone,
/// and every `TypedDict` value was scanned instead: each key resolved by name
/// and asked of the clause, for a clause that admits any string. The keys
/// settle this record as they settle a closed one: every declared field is
/// probed, and a key to spare is admitted when it is a `str` and refuses the
/// record when it is not -- which no key need be resolved to say. The general
/// scan and the reading by keys must agree on every one of these, and `case`
/// asks both walks.
#[test]
fn a_typed_dict_shaped_record_answers_for_each_of_its_keys() {
    Python::attach(|py| {
        let field = |name: &str, schema, required| Field {
            name: name.into(),
            schema,
            required,
        };
        let any_str_key = MapClause {
            key: Schema::Str,
            value: Schema::ANYTHING,
        };
        let record = Schema::keyed_map(
            vec![
                field("a", Schema::Int, true),
                field("b", Schema::Str, false),
            ],
            vec![any_str_key],
        );
        let dict = |pairs: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>| {
            let value = PyDict::new(py);
            for (key, item) in pairs {
                value.set_item(key, item).expect("a fresh dict takes a key");
            }
            value.into_any()
        };
        let int = |n: i64| n.into_pyobject(py).expect("an int").into_any();
        let text = |s: &str| PyString::new(py, s).into_any();

        // Exactly the declared keys, and the optional one absent.
        case(py, &record, &dict(vec![(text("a"), int(1))]), true);
        case(
            py,
            &record,
            &dict(vec![(text("a"), int(1)), (text("b"), text("x"))]),
            true,
        );
        // A declared key's value is the field's business, not the clause's.
        case(py, &record, &dict(vec![(text("a"), text("x"))]), false);
        case(
            py,
            &record,
            &dict(vec![(text("a"), int(1)), (text("b"), int(2))]),
            false,
        );
        // A required key the value does not carry.
        case(py, &record, &dict(vec![(text("b"), text("x"))]), false);
        // A key to spare: admitted with any value when it is a string ...
        case(
            py,
            &record,
            &dict(vec![(text("a"), int(1)), (text("z"), int(2))]),
            true,
        );
        case(
            py,
            &record,
            &dict(vec![(text("a"), int(1)), (text("z"), list_of(py, vec![]))]),
            true,
        );
        // ... and refused, whatever its value, when it is not.
        case(
            py,
            &record,
            &dict(vec![(text("a"), int(1)), (int(3), int(2))]),
            false,
        );
    });
}

/// The same record over a JSON document, whose keys are strings by the
/// grammar: a key to spare is admitted outright, and the document is read
/// through the plan rather than searched once per field.
#[test]
fn a_typed_dict_shaped_record_reads_a_document_by_its_keys() {
    Python::attach(|py| {
        let field = |name: &str, schema, required| Field {
            name: name.into(),
            schema,
            required,
        };
        let record = Schema::keyed_map(
            vec![
                field("a", Schema::Int, true),
                field("b", Schema::Str, false),
            ],
            vec![MapClause {
                key: Schema::Str,
                value: Schema::ANYTHING,
            }],
        );
        for (entries, expected) in [
            (vec![("a", JsonValue::Int(1))], true),
            (
                vec![("a", JsonValue::Int(1)), ("b", JsonValue::Str("x".into()))],
                true,
            ),
            (vec![("a", JsonValue::Str("x".into()))], false),
            (vec![("b", JsonValue::Str("x".into()))], false),
            (
                vec![("a", JsonValue::Int(1)), ("z", JsonValue::Int(2))],
                true,
            ),
            // The last of a repeated key is the one the document means.
            (
                vec![("a", JsonValue::Str("x".into())), ("a", JsonValue::Int(1))],
                true,
            ),
        ] {
            let json = json_object(entries);
            assert_eq!(
                holds_json(py, &record, &json),
                expected,
                "{record:?} against {json:?}"
            );
        }
    });
}

/// A record with fields *and* a clause that reads a key is scanned, and the
/// scan answers every rule the reading by keys answers for a closed record.
///
/// The by-keys path is for a record whose keys settle it -- closed, or open
/// under a clause that admits any string. `{str: int}` beside declared fields
/// reads each undeclared key with its value, so the scan is the only walk such
/// a record takes, and the four rules must hold there: a declared key's value
/// is the field's business, a required key must be present, an undeclared key
/// is the clause's to admit or refuse, and a key of another type is refused.
#[test]
fn a_record_whose_clause_reads_a_key_is_scanned_for_every_rule() {
    Python::attach(|py| {
        let field = |name: &str, schema, required| Field {
            name: name.into(),
            schema,
            required,
        };
        let mixed = Schema::keyed_map(
            vec![
                field("a", Schema::Str, true),
                field("b", Schema::Int, false),
            ],
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        );
        let dict = |pairs: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)>| {
            let value = PyDict::new(py);
            for (key, item) in pairs {
                value.set_item(key, item).expect("a fresh dict takes a key");
            }
            value.into_any()
        };
        let int = |n: i64| n.into_pyobject(py).expect("an int").into_any();
        let text = |s: &str| PyString::new(py, s).into_any();

        // A declared field's value is the field's, not the clause's: `a` holds
        // a string the clause would refuse, and the record admits it.
        case(py, &mixed, &dict(vec![(text("a"), text("x"))]), true);
        case(py, &mixed, &dict(vec![(text("a"), int(1))]), false);
        case(
            py,
            &mixed,
            &dict(vec![(text("a"), text("x")), (text("b"), text("y"))]),
            false,
        );
        // The required key.
        case(py, &mixed, &dict(vec![(text("b"), int(1))]), false);
        // An undeclared key is the clause's: an int value is covered, a string
        // value is not, and a key that is not a string is not.
        case(
            py,
            &mixed,
            &dict(vec![(text("a"), text("x")), (text("k"), int(2))]),
            true,
        );
        case(
            py,
            &mixed,
            &dict(vec![(text("a"), text("x")), (text("k"), text("v"))]),
            false,
        );
        case(
            py,
            &mixed,
            &dict(vec![(text("a"), text("x")), (int(3), int(2))]),
            false,
        );

        // A clause over keys of another type is read, not assumed: `int:
        // anything` admits an int key the string rule would refuse, and
        // refuses a string key it would admit. No frontend spells this
        // record -- a `TypedDict`'s keys are strings by the spec -- and the
        // walk decides it by what the clause says all the same.
        let int_keyed = Schema::keyed_map(
            vec![field("a", Schema::Str, true)],
            vec![MapClause {
                key: Schema::Int,
                value: Schema::ANYTHING,
            }],
        );
        case(
            py,
            &int_keyed,
            &dict(vec![(text("a"), text("x")), (int(3), text("v"))]),
            true,
        );
        case(
            py,
            &int_keyed,
            &dict(vec![(text("a"), text("x")), (text("k"), text("v"))]),
            false,
        );
    });
}

/// Both readings of a parsed object's undeclared keys answer alike.
///
/// A narrow object is covered where it lies -- an entry is the one the document
/// means exactly when no entry after it repeats its key -- and a wide one
/// collapses to a table of last values first, because the look forward is
/// quadratic. They are one question asked two ways, so the boundary between
/// them may show in what a walk costs and never in what it answers.
///
/// The rows run either side of `SMALL_OBJECT` and cross it, because that is the
/// only way a bound can be held: a suite that asks one width cannot tell the
/// bound from a constant, and one that never repeats a key cannot tell the look
/// forward from a walk that ignores repeats.
#[test]
fn a_parsed_object_reads_its_repeats_alike_at_every_width() {
    Python::attach(|py| {
        let mapping = Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }]
            .into(),
        };
        let names: Vec<String> = (0..24).map(|i| format!("k{i}")).collect();
        let filler = |width: usize| -> Vec<(&str, JsonValue<'_>)> {
            names
                .iter()
                .take(width)
                .map(|name| (name.as_str(), JsonValue::Int(1)))
                .collect()
        };

        // Every width either side of the bound, and across it.
        for width in [0, 1, 2, 7, 8, 9, 10, 24] {
            let good = json_object(filler(width));
            assert!(holds_json(py, &mapping, &good), "width {width}");

            // One entry of the wrong type refuses at every width.
            let mut wrong = filler(width);
            wrong.push(("bad", JsonValue::Str("a".into())));
            assert!(
                !holds_json(py, &mapping, &json_object(wrong)),
                "width {width}"
            );

            // A repeated key is the last one the document means, both ways
            // round, at every width: the earlier entry decides nothing.
            let mut last_wins = filler(width);
            last_wins.push(("r", JsonValue::Str("a".into())));
            last_wins.push(("r", JsonValue::Int(1)));
            assert!(
                holds_json(py, &mapping, &json_object(last_wins)),
                "width {width}"
            );

            let mut last_loses = filler(width);
            last_loses.push(("r", JsonValue::Int(1)));
            last_loses.push(("r", JsonValue::Str("a".into())));
            assert!(
                !holds_json(py, &mapping, &json_object(last_loses)),
                "width {width}"
            );
        }
    });
}

/// The key a violation points at, so a row reads as the field it means.
fn at(violation: &Violation) -> String {
    match violation.path.first() {
        Some(PathSegment::Key(name)) => name.to_string(),
        other => format!("{other:?}"),
    }
}

/// The explaining walk resumes where the deciding walk stopped, and reports the
/// same thing as one that starts over.
///
/// A record that fails is walked twice, and the second walk skips the fields the
/// first one passed -- they matched, so they have no violation to report, and a
/// probe through the interpreter is the dear half of reading one. What must not
/// change is the report, so every row here is asserted against the whole
/// violation list rather than against its length: a resumption that lost a
/// field would drop a violation, and one that lost the *count* it carries would
/// send a complete record to a scan that finds nothing.
///
/// The positions matter and the names do not resemble them: `Schema::keyed_map`
/// sorts fields by name, so a record of `f0..f9` is walked
/// `f0 f1 f2 ... f9` while one of `f0..f11` is walked `f0 f1 f10 f11 f2 ...`.
/// Every row below states the position it means.
#[test]
fn an_explaining_walk_resumes_where_the_deciding_one_stopped() {
    Python::attach(|py| {
        let named = |count: usize, required: bool| -> Vec<Field> {
            (0..count)
                .map(|i| field(&format!("f{i}"), Schema::Int, required))
                .collect()
        };
        // Ten fields, so the name order and the position order agree.
        let fields = named(10, true);
        let schema = Schema::keyed_map(fields.clone(), Vec::new());
        let state = WalkState::new();
        let index = build_index(py, &schema, &[], &[]);
        let explain = || Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Explain,
        };
        let report = |value: &Bound<'_, PyDict>| {
            let mut out = Vec::new();
            let held = member(
                &schema,
                &Value::Py(value.as_any()),
                &mut Frame::new(&mut Vec::new(), &mut out, explain()),
            );
            (held, out)
        };
        let whole = |wrong: Option<(&str, &str)>, drop: Option<&str>| {
            let value = PyDict::new(py);
            for i in 0..10 {
                value
                    .set_item(format!("f{i}"), i)
                    .expect("a fresh dict of small ints always builds");
            }
            if let Some((key, text)) = wrong {
                value.set_item(key, text).expect("replacing a key succeeds");
            }
            if let Some(key) = drop {
                value
                    .del_item(key)
                    .expect("deleting a present key succeeds");
            }
            value
        };

        // A refusal at the last position: nine fields are skipped and the tenth
        // is the whole report.
        let (held, late) = report(&whole(Some(("f9", "not an int")), None));
        assert!(!held);
        assert_eq!(late.len(), 1, "{late:?}");
        assert_eq!(at(&late[0]), "f9");

        // A refusal at the first position: nothing is skipped, and the nine
        // fields after it are read by the explaining walk itself.
        let (held, early) = report(&whole(Some(("f0", "not an int")), None));
        assert!(!held);
        assert_eq!(early.len(), 1, "{early:?}");
        assert_eq!(at(&early[0]), "f0");

        // Two wrong fields either side of the resumption: the report carries
        // both, in declared order, which is what a walk that resumed at the
        // first would lose.
        let two = whole(Some(("f2", "not an int")), None);
        two.set_item("f7", "not an int")
            .expect("replacing succeeds");
        let (held, both) = report(&two);
        assert!(!held);
        assert_eq!(both.len(), 2, "{both:?}");
        assert_eq!(at(&both[0]), "f2");
        assert_eq!(at(&both[1]), "f7");

        // A required key missing: the deciding walk refuses at its position and
        // the explaining walk reports it as missing rather than as a mismatch.
        let (held, absent) = report(&whole(None, Some("f4")));
        assert!(!held);
        assert_eq!(absent.len(), 1, "{absent:?}");
        assert_eq!(at(&absent[0]), "f4");

        // An undeclared key, so the deciding walk reads every field, finds each
        // one matching, and refuses on the count alone. The explaining walk then
        // resumes past *all* of them, and the count it resumes with is what lets
        // it reach the scan that names the extra key rather than reporting
        // nothing and calling the value mutated.
        let extra = whole(None, None);
        extra.set_item("surplus", 1).expect("adding a key succeeds");
        let (held, spare) = report(&extra);
        assert!(!held);
        assert_eq!(spare.len(), 1, "{spare:?}");
        assert_eq!(at(&spare[0]), "surplus");

        // A wrong field *and* undeclared keys: the count the resumption carries
        // must be the count the deciding walk had, exactly. Too high by any
        // amount and the explaining walk can read the record as holding exactly
        // its own keys, skip the scan, and never name the extra ones.
        //
        // One extra key catches a count too high by one and two catches one too
        // high by two, because what the early return compares is a *sum*: a
        // count wrong by `n` fires it on a record carrying `n` undeclared keys
        // and misses on every other. So the row asks both rather than one.
        for extras in 1..=2 {
            let crowded = whole(Some(("f5", "not an int")), None);
            for spare in 0..extras {
                crowded
                    .set_item(format!("surplus{spare}"), 1)
                    .expect("adding a key succeeds");
            }
            let (held, reported) = report(&crowded);
            assert!(!held);
            let named: Vec<String> = reported.iter().map(at).collect();
            assert_eq!(named.len(), 1 + extras, "{extras} extra: {named:?}");
            assert_eq!(named[0], "f5", "{named:?}");
            for spare in 0..extras {
                assert!(
                    named.contains(&format!("surplus{spare}")),
                    "{extras} extra: {named:?}"
                );
            }
        }
    });
}

/// The count a resumption carries is the count the deciding walk had, and an
/// absent optional field is not in it.
///
/// The resumption hands over two numbers, and the second is the one that is easy
/// to lose: how many declared keys had been *found*. That count is what lets a
/// record holding exactly its own keys skip the scan for undeclared ones, so a
/// resumption that over-counted would send a complete record to a scan, and one
/// that under-counted would let an extra key through unreported. An optional
/// field the value does not carry is the case that separates them: the deciding
/// walk passes it without counting it.
#[test]
fn a_resumption_carries_the_count_the_deciding_walk_had() {
    Python::attach(|py| {
        let state = WalkState::new();
        let mixed = vec![
            field("a", Schema::Int, false),
            field("b", Schema::Int, true),
            field("c", Schema::Int, true),
        ];
        let small = Schema::keyed_map(mixed, Vec::new());
        let small_index = build_index(py, &small, &[], &[]);
        let value = PyDict::new(py);
        value.set_item("b", 1).expect("a fresh dict builds");
        value
            .set_item("c", "not an int")
            .expect("a fresh dict builds");
        let mut out = Vec::new();
        let held = member(
            &small,
            &Value::Py(value.as_any()),
            &mut Frame::new(
                &mut Vec::new(),
                &mut out,
                Ctx {
                    pool: &[],
                    defs: &[],
                    records: &small_index.records,
                    attrs: &small_index.attrs,
                    unions: &small_index.unions,
                    regexes: &small_index.regexes,
                    guard: &state.guard,
                    depth: &state.depth,
                    fatal: &state.fatal,
                    fatal_seen: &state.fatal_seen,
                    mode: WalkMode::Explain,
                },
            ),
        );
        assert!(!held);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(at(&out[0]), "c");
    });
}

/// A resumption is refused where the value changed size between the two walks.
///
/// The deciding walk runs the schema against each field, and a field's schema
/// can run Python -- a predicate here, an `__eq__` or an `__instancecheck__` in
/// general -- which can change the dict it is being read out of. The resumption
/// then names a position in a value that no longer exists, so it is guarded by
/// the entry count, which is the same guard the deciding walk applies to itself
/// before answering.
///
/// Without the guard the explaining walk skips fields of a value it never read,
/// and reports about a dict that has moved on. With it, the two walks disagree,
/// nothing is found, and the walk says the value changed -- which is the true
/// report and the one a caller can act on.
#[test]
fn a_value_that_changes_size_between_the_walks_is_not_resumed() {
    Python::attach(|py| {
        // A predicate that deletes a key the first time it is asked, so the
        // dict shrinks while the deciding walk is part way through it.
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "seen = []\n\
                 def drops(x):\n\
                 \x20   if not seen:\n\
                 \x20       seen.append(x)\n\
                 \x20       target.pop('d', None)\n\
                 \x20       target['a'] = 'not an int'\n\
                 \x20   return True\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("drops.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("drops")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");

        let value = PyDict::new(py);
        for key in ["a", "b", "c", "d"] {
            value.set_item(key, 1).expect("a fresh dict builds");
        }
        module
            .setattr("target", &value)
            .expect("the module takes the dict to change");
        let drops = module.getattr("drops").expect("drops");
        let pool = vec![drops.unbind()];

        let guarded = Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
        };
        // Fields are walked in name order. `b` runs the predicate, which drops
        // `d` -- changing the size -- and spoils `a`, which the deciding walk
        // has already passed. `c` is fine, `d` is now missing, so the deciding
        // walk refuses at `d` and would hand over a resumption naming a value
        // that no longer exists.
        let schema = Schema::keyed_map(
            vec![
                field("a", Schema::Int, true),
                field("b", guarded, true),
                field("c", Schema::Int, true),
                field("d", Schema::Int, true),
            ],
            Vec::new(),
        );
        let state = WalkState::new();
        let index = build_index(py, &schema, &[], &[]);
        let mut out = Vec::new();
        let held = member(
            &schema,
            &Value::Py(value.as_any()),
            &mut Frame::new(
                &mut Vec::new(),
                &mut out,
                Ctx {
                    pool: &pool,
                    defs: &[],
                    records: &index.records,
                    attrs: &index.attrs,
                    unions: &index.unions,
                    regexes: &index.regexes,
                    guard: &state.guard,
                    depth: &state.depth,
                    fatal: &state.fatal,
                    fatal_seen: &state.fatal_seen,
                    mode: WalkMode::Explain,
                },
            ),
        );
        assert!(!held, "a value that lost a required key is not a member");
        // Read for itself, the value now fails at `a` as well as at `d`. A walk
        // that trusted the resumption would skip `a` -- passed against a value
        // that has since changed -- and report only what came after it.
        let named: Vec<String> = out.iter().map(at).collect();
        assert!(named.contains(&"a".to_owned()), "{named:?}");
        assert!(named.contains(&"d".to_owned()), "{named:?}");
    });
}

/// A lone clause governs a parsed object's keys by its value alone only where
/// its key schema admits every key there could be.
///
/// A parsed JSON object's keys are strings by construction, so a single clause
/// keyed by `str` -- or by anything -- answers the key half of the coverage
/// question before it is asked, and the walk reads only the values. The whole
/// reading rests on that guard: a clause keyed by anything *else* governs some
/// keys and not others, so each key must be asked of the clause as a whole.
///
/// Read as always true, the value half alone would admit a key no clause covers.
/// A clause keyed by `int` is the sharpest case, since a parsed key is never an
/// integer: it covers nothing, so an object carrying any key at all is refused,
/// and a walk that skipped the key would accept every one of them.
///
/// This is an interpreter-backed row on purpose. The rows that first held this
/// reading are in the Python suite, which the mutation sweep cannot observe, so
/// a nightly full-file sweep read both of the guard's mutants as new survivors.
#[test]
fn a_clause_reads_a_parsed_object_by_value_alone_only_where_its_key_admits_every_key() {
    Python::attach(|py| {
        let mapping = |key: Schema| Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause {
                key,
                value: Schema::Int,
            }]
            .into(),
        };
        let one = json_object(vec![("a", JsonValue::Int(1))]);
        let empty = json_object(Vec::new());

        // A clause keyed by `str` admits every key a parsed object can carry, so
        // the values decide and this one holds.
        assert!(holds_json(py, &mapping(Schema::Str), &one));
        assert!(!holds_json(
            py,
            &mapping(Schema::Str),
            &json_object(vec![("a", JsonValue::Str("x".into()))])
        ));

        // A clause keyed by `int` admits none of them: a parsed key is a string,
        // so nothing covers `"a"` and the object is refused. An object with no
        // keys has nothing to cover and holds.
        assert!(
            !holds_json(py, &mapping(Schema::Int), &one),
            "a clause that covers no key of this object cannot admit it"
        );
        assert!(holds_json(py, &mapping(Schema::Int), &empty));

        // The same for another kind sharing no value with a string, so the
        // guard is held by more than one witness.
        assert!(!holds_json(py, &mapping(Schema::Bool), &one));
        assert!(holds_json(py, &mapping(Schema::Bool), &empty));
    });
}

#[test]
fn a_tuple_subclass_is_walked_over_the_elements_it_holds() {
    // `PyTuple_Size` reads the storage on CPython and goes through the object's
    // own `__len__` on PyPy's `cpyext`, so a subclass that overrides it answers
    // the C accessor with whatever it likes -- and a walk that indexed against
    // that read past the end of the allocation and took the process down. The
    // walk reads the base type's own slot instead, and the rows below are the
    // answers that reading has to give: the elements the value holds, on every
    // interpreter, whatever the subclass reports for its own length.
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "class Lying(tuple):\n\
                 \x20   def __len__(self):\n\
                 \x20       return 10\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("lying.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("lying")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let lying = |items: Vec<i64>| {
            module
                .getattr("Lying")
                .expect("the class")
                .call1((PyTuple::new(py, items).expect("a tuple builds"),))
                .expect("the subclass builds")
        };

        let one_int = lying(vec![1]);
        let ints = Schema::tuple(SeqShape::homogeneous(Schema::Int));
        let strs = Schema::tuple(SeqShape::homogeneous(Schema::Str));
        assert!(
            holds(py, &ints, &one_int, &[], &[]),
            "one int, and it is one"
        );
        assert!(!holds(py, &strs, &one_int, &[], &[]), "and it is not a str");

        // The arity is the storage's too: one element is not two.
        let exactly_one = Schema::tuple(SeqShape::fixed([Schema::Int]));
        let exactly_two = Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Int]));
        assert!(holds(py, &exactly_one, &one_int, &[], &[]));
        assert!(!holds(py, &exactly_two, &one_int, &[], &[]));

        // And a copy that cannot be made is a value the walk could not read,
        // which is not the same as a value it decided against: the elements
        // here are readable, so every row above is an answer rather than a
        // refusal standing in for one.
        let two = lying(vec![1, 2]);
        assert!(holds(py, &exactly_two, &two, &[], &[]));
    });
}

#[test]
fn a_tuple_subclass_that_overrides_nothing_is_read_where_it_lies() {
    // The copy above exists for a subclass that answers the C accessor for
    // itself. A `NamedTuple` does not: it inherits `tuple.__len__`, so the
    // accessor reads its storage on every interpreter, as it does for an exact
    // tuple. Telling the two apart by `is_exact_instance_of` read the common
    // subclass as the rare one and copied every value; the walk asks whether
    // the type's `__len__` is the base's instead.
    //
    // The rows are membership, and the reading they pin is the *absence* of a
    // copy -- which no answer can show. What an answer shows is that the
    // cheaper path decides the same things, which is what makes it takeable.
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "from typing import NamedTuple\n\
                 class Point(NamedTuple):\n\
                 \x20   x: int\n\
                 \x20   y: int\n\
                 class Widened(tuple):\n\
                 \x20   pass\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("inheriting.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("inheriting")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");

        let point = module
            .getattr("Point")
            .expect("the class")
            .call1((1i64, 2i64))
            .expect("the namedtuple builds");
        let two_ints = Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Int]));
        assert!(
            holds(py, &two_ints, &point, &[], &[]),
            "two ints, and it is"
        );
        assert!(!holds(
            py,
            &Schema::tuple(SeqShape::fixed([Schema::Int])),
            &point,
            &[],
            &[]
        ));
        assert!(holds(
            py,
            &Schema::tuple(SeqShape::homogeneous(Schema::Int)),
            &point,
            &[],
            &[]
        ));
        assert!(!holds(
            py,
            &Schema::tuple(SeqShape::homogeneous(Schema::Str)),
            &point,
            &[],
            &[]
        ));

        // A bare subclass inherits the slot too, and is read the same way.
        let widened = module
            .getattr("Widened")
            .expect("the class")
            .call1((PyTuple::new(py, [1i64, 2, 3]).expect("a tuple builds"),))
            .expect("the subclass builds");
        assert!(holds(
            py,
            &Schema::tuple(SeqShape::homogeneous(Schema::Int)),
            &widened,
            &[],
            &[]
        ));
    });
}

/// A record resolves a key the way the dict it reads does.
///
/// The declared names are interned and an exact `str` is compared by text. A
/// subclass carries the field's text and may still be a key of its own: a dict
/// reaches an entry by hash and then by equality, so a subclass that hashes
/// elsewhere, or that equals nothing, is not the field it spells. Reading it as
/// that field would admit a value the dict does not carry under the name.
#[test]
fn a_record_resolves_a_key_the_way_the_dict_does() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "class Plain(str):\n\
                 \x20   __slots__ = ()\n\
                 class OtherHash(str):\n\
                 \x20   __slots__ = ()\n\
                 \x20   def __hash__(self):\n\
                 \x20       return 0\n\
                 class NeverEqual(str):\n\
                 \x20   __slots__ = ()\n\
                 \x20   __hash__ = str.__hash__\n\
                 \x20   def __eq__(self, other):\n\
                 \x20       return False\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("keys.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("keys")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");

        // An open record: a key the field lookup does not find falls to the
        // scan, which is where a key is read as the name it spells. A closed
        // record answers by count before the scan and never asks.
        let record = Schema::keyed_map(
            vec![Field {
                name: "a".into(),
                schema: Schema::Int,
                required: false,
            }],
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Str,
            }],
        );
        let keyed = |class: &str| {
            let key = module
                .getattr(class)
                .expect("the class")
                .call1((PyString::new(py, "a"),))
                .expect("the subclass builds");
            let value = PyDict::new(py);
            value
                .set_item(key, PyInt::new(py, 1i64))
                .expect("a fresh dict takes a key");
            value.into_any()
        };

        // A faithful subclass is the field, and the field admits the integer
        // the clause would refuse.
        case(py, &record, &keyed("Plain"), true);
        // A key that hashes elsewhere is a key of its own: the field does not
        // govern it, the clause does, and the clause takes strings.
        case(py, &record, &keyed("OtherHash"), false);
        // The same for a key that equals nothing: the hash reaches the bucket
        // and the comparison refuses the entry.
        case(py, &record, &keyed("NeverEqual"), false);
    });
}

/// The scalar shortcut is for a shape whose every element is the same scalar.
///
/// A prefix fixes what its positions hold, so a shape carrying one is not that
/// shape: taking the shortcut over it would read the prefix positions against
/// the *tail's* element and refuse a value the shape admits.
#[test]
fn a_prefix_is_not_read_against_the_tail_it_precedes() {
    Python::attach(|py| {
        let prefixed = Schema::list(SeqShape::prefix_tail([Schema::Str], Schema::Int));
        let text = |s: &str| PyString::new(py, s).into_any();
        let int = |n: i64| PyInt::new(py, n).into_any();
        macro_rules! list {
            ($($item:expr),* $(,)?) => {
                PyList::new(py, [$($item),*]).expect("a list builds").into_any()
            };
        }

        // The prefix holds a string and the tail repeats ints.
        case(py, &prefixed, &list![text("a")], true);
        case(py, &prefixed, &list![text("a"), int(1), int(2)], true);
        // A value the shortcut would admit and the shape does not: every
        // element is an int, and the first position is not a string.
        case(py, &prefixed, &list![int(1), int(2)], false);
        // And one the shortcut would refuse and the shape admits is the row
        // above it: a string in the prefix, past the tail's element kind.
        case(py, &prefixed, &list![text("a"), text("b")], false);
    });
}

/// The scalar shortcut and the explaining walk agree at every depth.
///
/// The shortcut reads a homogeneous list of a scalar kind without opening a
/// level and the explaining walk opens one per element, so the level an element
/// sits at is taken in one reading and skipped in the other unless the shortcut
/// takes it too. At the walk's ceiling that is the difference between an answer
/// and a refusal, and the two readings are one answer or the library has two.
///
/// A fixpoint is what drives a walk that deep -- the schema depth bound stops a
/// spelled one long before -- and its branches carry the scalar-tailed list the
/// shortcut is for.
#[test]
fn the_two_readings_agree_at_the_walks_depth_bound() {
    Python::attach(|py| {
        // `mu X. [X, ...] & MinLen(1) | [int, ...] & MinLen(1)`
        let non_empty = |element: Schema| {
            Schema::refine(
                Schema::list(SeqShape::homogeneous(element)),
                vec![Constraint::MinLen(1)],
            )
        };
        let defs = [Schema::Union(
            vec![
                non_empty(Schema::Ref(DefIx::new(0))),
                non_empty(Schema::Int),
            ]
            .into(),
        )];
        let schema = Schema::Ref(DefIx::new(0));
        let nested = |levels: usize| {
            let mut value = PyInt::new(py, 1i64).into_any();
            for _ in 0..levels {
                value = PyList::new(py, [value]).expect("a list builds").into_any();
            }
            value
        };

        // `decide` holds the two readings to one answer, which is the property
        // under test; the value of that answer is the row beside it.
        for levels in [1, 8, 127, 128, 129, MAX_WALK_DEPTH + 1] {
            decide(py, &schema, &nested(levels), &[], &defs);
        }
        assert!(holds(py, &schema, &nested(1), &[], &defs));
        assert!(holds(py, &schema, &nested(8), &[], &defs));
        // Past the ceiling there is no level to open, and the answer is a
        // refusal rather than a reading taken without one.
        assert!(!holds(py, &schema, &nested(MAX_WALK_DEPTH + 1), &[], &defs));
    });
}

/// A set subclass is walked over the members it holds, not the ones it yields.
///
/// A set has no positions, so it is read through an iterator -- and an iterator
/// is a slot a subclass may override. One that does was walked through its own
/// `__iter__`, so `set[int]` admitted a value whose storage held a `str`: an
/// accept with no value under it, which is the one direction the contract
/// forbids. The base type's slot is asked instead, and the rows below are the
/// answers that reading has to give.
#[test]
fn a_set_subclass_is_walked_over_the_members_it_holds() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "class LyingIter(set):\n\
                 \x20   def __iter__(self):\n\
                 \x20       return iter([1, 2, 3])\n\
                 class LyingLen(set):\n\
                 \x20   def __len__(self):\n\
                 \x20       return 9\n\
                 class Quiet(set):\n\
                 \x20   pass\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("lying_set.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("lying_set")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let build = |name: &str, members: Vec<&str>| {
            module
                .getattr(name)
                .expect("the class")
                .call1((PySet::new(py, members).expect("a set builds"),))
                .expect("the subclass builds")
        };

        let ints = Schema::set(Schema::Int);
        let strs = Schema::set(Schema::Str);

        // The iterator says three integers; the storage holds one string, and
        // the storage is the value.
        let lying = build("LyingIter", vec!["a"]);
        case(py, &ints, &lying, false);
        case(py, &strs, &lying, true);

        // A length bound counts the storage for the same reason.
        let one = build("LyingLen", vec!["a"]);
        let at_least_three = Schema::refine(Schema::set(Schema::Str), vec![Constraint::MinLen(3)]);
        let at_most_one = Schema::refine(Schema::set(Schema::Str), vec![Constraint::MaxLen(1)]);
        case(py, &at_least_three, &one, false);
        case(py, &at_most_one, &one, true);

        // A subclass overriding neither slot is read where it lies, which is
        // what keeps the common subclass costing what a set costs.
        let quiet = build("Quiet", vec!["a"]);
        case(py, &strs, &quiet, true);
        case(py, &ints, &quiet, false);
    });
}

/// A list wide enough to be read from a snapshot decides what the in-place
/// scan decides, and reports a list that moved under the reading.
///
/// Past a width the scalar walk copies the list and reads the copy borrowed,
/// which is a second implementation of the same answer: it must accept what
/// the scan accepts, refuse what it refuses, and compare the count again
/// afterwards so a value that resized while it was read is reported rather
/// than answered for. The width is what selects it, so the rows are wide.
#[test]
fn a_wide_list_of_scalars_is_decided_the_same_read_from_a_copy() {
    Python::attach(|py| {
        let schema = Schema::list(SeqShape::homogeneous(Schema::Int));
        // Wider than the copy's floor, so this is the snapshot reading rather
        // than the scan the narrow rows above take.
        let wide = PyList::new(py, (0..64i64).collect::<Vec<_>>()).expect("builds");
        assert!(decide(py, &schema, &wide.clone().into_any(), &[], &[]));

        // One element of another kind, and the answer is the scan's.
        let spoiled = PyList::new(py, (0..64i64).collect::<Vec<_>>()).expect("builds");
        spoiled
            .set_item(40, PyString::new(py, "x"))
            .expect("set_item");
        assert!(!decide(py, &schema, &spoiled.into_any(), &[], &[]));
    });
}

/// A key a clause does not cover reports both halves of the mismatch, whatever
/// the record has already found.
///
/// A homogeneous mapping is two schemas, and an entry outside it can be
/// outside either: reporting only the key leaves a reader with a key that
/// looks right. The failure this pins is the report going quiet once anything
/// else has failed -- a declared field's mismatch is not a reason to stop
/// describing the entries beside it.
#[test]
fn a_clause_reports_the_value_it_refused_beside_a_field_that_already_failed() {
    Python::attach(|py| {
        let schema = Schema::keyed_map(
            vec![field("a", Schema::Int, true)],
            vec![MapClause {
                key: Schema::Str,
                value: Schema::Int,
            }],
        );
        let value = PyDict::new(py);
        value.set_item("a", PyString::new(py, "no")).expect("set");
        value.set_item("b", PyString::new(py, "nope")).expect("set");
        let value = value.into_any();

        let (ok, violations) = explain(py, &schema, &value, &[], &[]);
        assert!(!ok);
        // The declared field, then the undeclared entry's value: the key `b`
        // is a string and belongs, so what is left to say is about `"nope"`.
        assert_eq!(violations.len(), 2, "{violations:?}");
        assert_eq!(violations[0].location(), "a");
        assert_eq!(violations[1].location(), "b");
        assert_eq!(violations[1].code, "int_type");
    });
}

/// A branch's name follows a definition a bounded number of times, and says
/// the node's own kind past the bound.
///
/// A union that matched nothing names its branches, and a branch built by
/// `recursive` is a reference whose kind is the word `value`. Following the
/// definition gives a reader the set rather than the word, and following it
/// without a bound gives a definition naming another one a walk that does not
/// end. The bound is what makes both true at once, so it is pinned from both
/// sides: a chain shorter than it is followed to the end, and a chain one link
/// longer stops and names the reference.
#[test]
fn a_branch_label_follows_a_definition_to_a_bounded_depth() {
    Python::attach(|py| {
        // Two chains of references in one table. The first is one link longer
        // than the bound follows, the second one link shorter.
        let defs = vec![
            Schema::Ref(DefIx::new(1)),
            Schema::Ref(DefIx::new(2)),
            Schema::Ref(DefIx::new(3)),
            Schema::Ref(DefIx::new(4)),
            Schema::Int,
            Schema::Ref(DefIx::new(6)),
            Schema::Ref(DefIx::new(7)),
            Schema::Ref(DefIx::new(8)),
            Schema::Str,
        ];
        let schema =
            Schema::Union(vec![Schema::Ref(DefIx::new(0)), Schema::Ref(DefIx::new(5))].into());
        // A float belongs to neither end, and neither branch descends into it,
        // so the report is the summary that names them.
        let value = PyFloat::new(py, 1.5).into_any();
        let (ok, violations) = explain(py, &schema, &value, &[], &defs);
        assert!(!ok);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].code, "union_error");
        // The long chain stops at the reference it reached the bound on; the
        // short one arrives at the set and names it.
        assert_eq!(violations[0].expected, "one of: value, str");
    });
}

/// A union reports the branch whose *walk* stopped, rather than folding it
/// into "this value matched no branch".
///
/// A branch that raised inside a predicate has not said the value is outside
/// its set -- it has said the question was not answered -- and it fails at the
/// union's own location, which is where the summary would swallow it. The
/// sentence that says what to do about it is the only one the report has.
#[test]
fn a_union_reports_a_branch_whose_walk_stopped_rather_than_a_summary() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new("def raises(x):\n\x20   raise ValueError('no')\n")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("raiser.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("raiser")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let pool = vec![module.getattr("raises").expect("raises").unbind()];
        let raising = Schema::Refine {
            base: Arc::new(Schema::ANYTHING),
            constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
        };
        let schema = Schema::Union(vec![raising, Schema::Int].into());
        // Neither branch descends: the predicate raises where it stands, and
        // the integer branch refuses the value itself.
        let value = PyString::new(py, "s").into_any();
        let (ok, violations) = explain(py, &schema, &value, &pool, &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].code, "predicate_error");
    });
}

/// A branch is measured by where its first failure lies, which a walk stopped
/// there has measured. So each branch is walked in the caller's mode, and a
/// fail-fast report costs a fail-fast walk of each branch rather than the size
/// of the value.
///
/// A branch that fails shallowly and again deep inside is the case that tells
/// the first failure from the deepest. Its first failure lies nearer than the
/// other branch's only one, so the other branch is the closer, in either mode:
/// the one failure fail-fast reports is the one the full report leads with.
#[test]
fn a_branch_is_measured_by_its_first_failure_in_either_mode() {
    Python::attach(|py| {
        let inner = PyDict::new(py);
        inner.set_item("d", 1i64).expect("set");
        let middle = PyDict::new(py);
        middle.set_item("c", &inner).expect("set");
        let value = PyDict::new(py);
        value.set_item("a", 1i64).expect("set");
        value.set_item("b", &middle).expect("set");
        let value = value.into_any();

        // The near branch fails at `a` and again three levels down at `b.c.d`.
        let near = Schema::record(
            vec![
                field("a", Schema::Str, true),
                field(
                    "b",
                    Schema::record(
                        vec![field(
                            "c",
                            Schema::record(vec![field("d", Schema::Str, true)], Openness::Closed),
                            true,
                        )],
                        Openness::Closed,
                    ),
                    true,
                ),
            ],
            Openness::Closed,
        );
        // The far branch fails once, two levels down: further than the near
        // branch's *first* failure, nearer than its last.
        let far = Schema::record(
            vec![
                field("a", Schema::Int, true),
                field(
                    "b",
                    Schema::record(vec![field("c", Schema::Str, true)], Openness::Closed),
                    true,
                ),
            ],
            Openness::Closed,
        );
        let schema = Schema::Union(vec![near, far].into());

        // Aggregating, the far branch is chosen: its one failure lies deeper
        // than the near branch's first.
        let (ok, violations) = explain(py, &schema, &value, &[], &[]);
        assert!(!ok);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].location(), "b.c");

        // Stopping at the first, the same branch and the same failure.
        let (ok, violations) = explain_in(py, &schema, &value, &[], &[], WalkMode::ExplainFailFast);
        assert!(!ok);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].location(), "b.c");
    });
}

/// A fail-fast report walks no branch past its first failure: a predicate on
/// the elements after it never runs, where a full report runs it on each.
#[test]
fn a_fail_fast_report_walks_no_branch_past_its_first_failure() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            std::ffi::CString::new(
                "calls = []\ndef counts(x):\n\x20   calls.append(x)\n\x20   return True\n",
            )
            .expect("no interior nul")
            .as_c_str(),
            std::ffi::CString::new("counter.py")
                .expect("no interior nul")
                .as_c_str(),
            std::ffi::CString::new("counter")
                .expect("no interior nul")
                .as_c_str(),
        )
        .expect("the module compiles");
        let pool = vec![module.getattr("counts").expect("counts").unbind()];
        let counted = Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![Constraint::Predicate(PredIx::new(0))].into(),
        };
        let schema =
            Schema::Union(vec![Schema::Int, Schema::list(SeqShape::homogeneous(counted))].into());
        let value = PyList::new(py, [PyString::new(py, "x").into_any()])
            .expect("builds")
            .into_any();
        let items = value.cast::<PyList>().expect("a list");
        for n in 1i64..=3 {
            items.append(n).expect("append");
        }
        let calls = || {
            module
                .getattr("calls")
                .expect("calls")
                .len()
                .expect("a list")
        };

        let (ok, violations) =
            explain_in(py, &schema, &value, &pool, &[], WalkMode::ExplainFailFast);
        assert!(!ok);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].location(), "[0]");
        assert_eq!(calls(), 0, "the branch walked past its first failure");

        let (ok, _) = explain(py, &schema, &value, &pool, &[]);
        assert!(!ok);
        assert_eq!(calls(), 3, "a full report walks the branch whole");
    });
}

/// A module whose objects run caller code when a message renders them: two
/// whose `repr` is interrupted, one whose `repr` fails the ordinary way, and a
/// class whose metaclass answers `__name__` with an interrupt.
fn loud(py: Python<'_>) -> Bound<'_, PyAny> {
    PyModule::from_code(
        py,
        c"class Loud:\n    def __repr__(self):\n        raise KeyboardInterrupt\n\
          class LoudInt(int):\n    def __repr__(self):\n        raise KeyboardInterrupt\n\
          class Awkward:\n    def __repr__(self):\n        raise ValueError('no repr')\n\
          class Meta(type):\n    def __getattribute__(cls, name):\n\
          \x20       if name == '__name__':\n            raise KeyboardInterrupt\n\
          \x20       return super().__getattribute__(name)\n\
          class LoudlyNamed(metaclass=Meta):\n    pass\n",
        c"loud.py",
        c"loud",
    )
    .expect("the module compiles")
    .into_any()
}

/// The first fatal signal an explaining walk of `value` recorded, and what it
/// reported.
fn recorded(
    py: Python<'_>,
    schema: &Schema,
    value: &Bound<'_, PyAny>,
    pool: &[Py<PyAny>],
) -> (Option<PyErr>, Vec<Violation>) {
    let index = build_index(py, schema, &[], pool);
    let state = WalkState::new();
    let ctx = Ctx {
        pool,
        defs: &[],
        records: &index.records,
        attrs: &index.attrs,
        unions: &index.unions,
        regexes: &index.regexes,
        guard: &state.guard,
        depth: &state.depth,
        fatal: &state.fatal,
        fatal_seen: &state.fatal_seen,
        mode: WalkMode::Explain,
    };
    let mut out = Vec::new();
    member(
        schema,
        &Value::Py(value),
        &mut Frame::new(&mut Vec::new(), &mut out, ctx),
    );
    (state.fatal.take(), out)
}

/// A message renders the constant a literal names, the bound or step a value
/// missed, the value a constraint refused, a key the path names by its repr,
/// and a class's name, and each of those runs caller code. A fatal signal
/// raised there is recorded for the entry point to raise.
#[test]
fn a_fatal_signal_from_any_part_of_a_message_is_recorded() {
    use pyo3::exceptions::PyKeyboardInterrupt;
    Python::attach(|py| {
        let module = loud(py);
        let make = |name: &str, arg: Option<i64>| -> Py<PyAny> {
            let class = module.getattr(name).expect("the class");
            match arg {
                Some(n) => class.call1((n,)),
                None => class.call0(),
            }
            .expect("an instance")
            .unbind()
        };
        let int = |n: i64| PyInt::new(py, n).into_any();
        let refine = |constraint| Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: vec![constraint].into(),
        };
        let first = OperandIx::new(0);
        let loud_key = PyDict::new(py);
        loud_key
            .set_item(make("Loud", None), "x")
            .expect("a hashable key");
        let named = module.getattr("LoudlyNamed").expect("the class").unbind();

        let cases = [
            (
                "the constant a literal names",
                Schema::Literal(ConstIx::new(0)),
                int(1),
                vec![make("Loud", None)],
            ),
            (
                "the bound a value missed",
                refine(Constraint::Ge(first)),
                int(1),
                vec![make("LoudInt", Some(11))],
            ),
            (
                "the step a value missed",
                refine(Constraint::MultipleOf(first)),
                int(1),
                vec![make("LoudInt", Some(3))],
            ),
            (
                "the value a bound refused",
                refine(Constraint::Ge(first)),
                make("LoudInt", Some(1)).into_bound(py),
                vec![int(10).unbind()],
            ),
            (
                "a key the path names by its repr",
                Schema::mapping(MapClause {
                    key: Schema::ANYTHING,
                    value: Schema::Int,
                }),
                loud_key.into_any(),
                Vec::new(),
            ),
            (
                "a class's name",
                Schema::Instance(ClassIx::new(0)),
                int(1),
                vec![named.clone_ref(py)],
            ),
        ];
        for (site, schema, value, pool) in cases {
            let (fatal, _) = recorded(py, &schema, &value, &pool);
            assert!(
                fatal.is_some_and(|err| err.is_instance_of::<PyKeyboardInterrupt>(py)),
                "{site} was not carried out"
            );
        }
    });
}

/// A union's branch label, read on its own, renders the literal it names and
/// the class it names, and carries out a fatal signal from either. An ordinary
/// error is a value that cannot render, and reads as `<unrepresentable>`.
#[test]
fn a_fatal_signal_from_a_branch_label_is_recorded() {
    Python::attach(|py| {
        let module = loud(py);
        let make = |name: &str| -> Py<PyAny> {
            module
                .getattr(name)
                .and_then(|class| class.call0())
                .expect("an instance")
                .unbind()
        };
        let named = module.getattr("LoudlyNamed").expect("the class").unbind();
        for (site, branch, pool) in [
            (
                "a literal branch",
                Schema::Literal(ConstIx::new(0)),
                vec![make("Loud")],
            ),
            (
                "a class branch",
                Schema::Instance(ClassIx::new(0)),
                vec![named.clone_ref(py)],
            ),
        ] {
            let schema = Schema::union([branch, Schema::Str]);
            let index = build_index(py, &schema, &[], &pool);
            let state = WalkState::new();
            let ctx = Ctx {
                pool: &pool,
                defs: &[],
                records: &index.records,
                attrs: &index.attrs,
                unions: &index.unions,
                regexes: &index.regexes,
                guard: &state.guard,
                depth: &state.depth,
                fatal: &state.fatal,
                fatal_seen: &state.fatal_seen,
                mode: WalkMode::Explain,
            };
            push_branch_label(&schema, ctx, py, &mut BranchLabels::new());
            assert!(state.fatal.take().is_some(), "{site} was not carried out");
        }

        // The control: an ordinary error is a value that cannot render.
        let pool = vec![PyInt::new(py, 10).into_any().unbind()];
        let awkward = make("Awkward").into_bound(py);
        let (fatal, out) = recorded(py, &Schema::Int, &awkward, &pool);
        assert!(fatal.is_none());
        assert_eq!(
            out.first().map(|v| v.value_summary.as_str()),
            Some("<unrepresentable>")
        );
    });
}

/// A scalar branch or clause is answered as the walk answers it: the verdict
/// `member` gives in a fast walk for every scalar schema and every kind of
/// value, a refusal where no level is free or a fatal signal is recorded, and
/// no answer for a schema that is not a scalar.
#[test]
fn a_scalar_is_answered_as_the_walk_answers_it() {
    Python::attach(|py| {
        let values = [
            py.None().into_bound(py),
            PyBool::new(py, true).to_owned().into_any(),
            PyInt::new(py, 7i64).into_any(),
            PyFloat::new(py, 1.5).into_any(),
            PyString::new(py, "x").into_any(),
            PyBytes::new(py, b"y").into_any(),
            list_of(py, vec![1]),
        ];
        let scalars = [
            Schema::ANY,
            Schema::Nothing,
            Schema::NoneType,
            Schema::Bool,
            Schema::Int,
            Schema::Float,
            Schema::Str,
            Schema::Bytes,
        ];
        let index = build_index(py, &Schema::Int, &[], &[]);
        let state = WalkState::new();
        let ctx = Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Fast,
        };
        for schema in &scalars {
            for value in &values {
                let value = Value::Py(value);
                let walked = member(
                    schema,
                    &value,
                    &mut Frame::new(&mut Vec::new(), &mut Vec::new(), ctx),
                );
                assert_eq!(scalar_member(schema, &value, ctx, true), Some(walked));
                assert_eq!(scalar_member(schema, &value, ctx, false), Some(false));
            }
        }
        let seven = Value::Py(&values[2]);
        assert_eq!(
            scalar_member(
                &Schema::list(SeqShape::homogeneous(Schema::Int)),
                &seven,
                ctx,
                true
            ),
            None
        );
        state.fatal_seen.set(true);
        assert_eq!(scalar_member(&Schema::Int, &seven, ctx, true), Some(false));
    });
}

/// A mapping of one clause of two scalars answers as the walk does: its keys
/// are not read as field names, since it declares none, and each half of an
/// entry is its type test, which refuses the entry that fails either.
#[test]
fn a_mapping_of_one_scalar_clause_is_read_entry_by_entry() {
    Python::attach(|py| {
        let mapping = Schema::mapping(MapClause {
            key: Schema::Str,
            value: Schema::Int,
        });
        let dict = |source: &str| {
            py.eval(&std::ffi::CString::new(source).expect("no nul"), None, None)
                .expect("the dict evaluates")
        };
        case(py, &mapping, &dict("{'a': 1, 'b': 2}"), true);
        case(py, &mapping, &dict("{}"), true);
        case(py, &mapping, &dict("{'a': 1, 'b': 'x'}"), false);
        case(py, &mapping, &dict("{'a': 1, 2: 2}"), false);
    });
}

/// A sequence whose repeated element is a scalar is read as its type test where
/// one level is free under it, and one whose element is a union of scalars as
/// a test per branch where two are -- the union's and its branch's -- whether
/// the walk explains or not; neither is read so behind a fixed prefix, nor for
/// a union holding a container. Its verdict is the walk's, for a list, a tuple
/// and a parsed array, and an explaining reading records one violation for an
/// element that fails and none for a sequence that belongs.
#[test]
fn a_sequence_of_a_union_of_scalars_is_read_as_its_tests() {
    use super::scalar::{Scalar, homogeneous_scalar, homogeneous_scalar_union};
    use super::sequence::{scalar_union_list_matches, scalar_union_tuple_matches};
    Python::attach(|py| {
        let nullable = Schema::union([Schema::Int, Schema::NoneType]);
        let index = build_index(py, &Schema::Int, &[], &[]);
        let state = WalkState::new();
        let ctx = |mode| Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode,
        };
        let Schema::Union(branches) = &nullable else {
            unreachable!("a union of two scalars stays a union")
        };
        let of_lists = Schema::list(SeqShape::homogeneous(Schema::Int));
        let holding_a_list = Schema::union([Schema::Int, of_lists.clone()]);
        let Schema::Union(holding_a_list) = &holding_a_list else {
            unreachable!("a union of a scalar and a list stays a union")
        };
        let union = |members: &[Schema], prefix: &[Schema], mode| {
            homogeneous_scalar_union(prefix, members, ctx(mode)).map(<[Schema]>::len)
        };
        let kind = |tail: &Schema, prefix: &[Schema], mode| {
            homogeneous_scalar(prefix, Some(tail), ctx(mode)).map(|(kind, schema)| {
                assert!(
                    std::ptr::eq(schema, tail),
                    "the kind comes with its own schema"
                );
                kind
            })
        };
        assert_eq!(union(branches, &[], WalkMode::Fast), Some(2));
        assert_eq!(union(branches, &[], WalkMode::Explain), Some(2));
        assert_eq!(union(branches, &[Schema::Int], WalkMode::Fast), None);
        assert_eq!(union(holding_a_list, &[], WalkMode::Fast), None);
        assert_eq!(kind(&Schema::Int, &[], WalkMode::Fast), Some(Scalar::Int));
        assert_eq!(
            kind(&Schema::Int, &[], WalkMode::Explain),
            Some(Scalar::Int)
        );
        assert_eq!(kind(&Schema::Int, &[Schema::Int], WalkMode::Fast), None);
        assert_eq!(kind(&of_lists, &[], WalkMode::Fast), None);
        assert_eq!(kind(&nullable, &[], WalkMode::Fast), None);
        assert_eq!(homogeneous_scalar(&[], None, ctx(WalkMode::Fast)), None);
        state.depth.set(MAX_WALK_DEPTH - 2);
        assert_eq!(union(branches, &[], WalkMode::Fast), Some(2));
        state.depth.set(MAX_WALK_DEPTH - 1);
        assert_eq!(union(branches, &[], WalkMode::Fast), None);
        assert_eq!(kind(&Schema::Int, &[], WalkMode::Fast), Some(Scalar::Int));
        state.depth.set(MAX_WALK_DEPTH);
        assert_eq!(kind(&Schema::Int, &[], WalkMode::Fast), None);
        state.depth.set(0);

        let eval = |source: &str| {
            py.eval(&std::ffi::CString::new(source).expect("no nul"), None, None)
                .expect("the value evaluates")
        };
        let list = Schema::list(SeqShape::homogeneous(nullable.clone()));
        let tuple = Schema::tuple(SeqShape::homogeneous(nullable.clone()));
        case(py, &list, &eval("[1, None, 2]"), true);
        case(py, &list, &eval("[1, 'x']"), false);
        case(py, &tuple, &eval("(None, 3)"), true);
        case(py, &tuple, &eval("(None, 1.5)"), false);
        // Each reading answers where it is taken, and declines where it is not:
        // a decline hands the walk to the general path, which answers the same,
        // so only asking the reading itself shows it was taken.
        let direct = |source: &str, mode| {
            let value = eval(source);
            let (mut path, mut out) = (Vec::new(), Vec::new());
            let mut frame = Frame::new(&mut path, &mut out, ctx(mode));
            let answer = if let Ok(list) = value.cast::<PyList>() {
                let value = Value::Py(&value);
                scalar_union_list_matches(list, &[], &nullable, branches, &value, &mut frame)
            } else {
                let tuple = value.cast::<PyTuple>().expect("a list or a tuple");
                scalar_union_tuple_matches(tuple, &[], &nullable, branches, &mut frame)
            };
            (answer, out.len())
        };
        assert_eq!(direct("[1, None, 2]", WalkMode::Fast), (Some(true), 0));
        assert_eq!(direct("[1, 'x']", WalkMode::Fast), (Some(false), 0));
        assert_eq!(direct("[1, None]", WalkMode::Explain), (Some(true), 0));
        assert_eq!(direct("[1, 'x']", WalkMode::Explain), (Some(false), 1));
        assert_eq!(direct("(None, 3)", WalkMode::Fast), (Some(true), 0));
        assert_eq!(direct("(None, 1.5)", WalkMode::Fast), (Some(false), 0));
        assert_eq!(direct("(None, 3)", WalkMode::Explain), (Some(true), 0));
        assert_eq!(direct("(None, 1.5)", WalkMode::Explain), (Some(false), 1));
        let json = |source: &str| {
            let parsed = JsonValue::parse(source.as_bytes(), false).expect("the JSON parses");
            holds_json(py, &list, &parsed)
        };
        assert!(json("[1, null, 2]"));
        assert!(!json("[1, \"x\"]"));
    });
}

/// The constant itself is its literal where equality is the builtin's and holds
/// of every object -- an exact `str`, `int`, `bool` or `bytes`, and `None` --
/// and nowhere else: not for the same `nan`, which is not equal to itself, not
/// for a class answering `==` for itself, not for a `str` subclass, and not for
/// an equal object that is another object. Where it does not answer, the
/// comparison does, alone and inside a union's table.
#[test]
fn the_constant_itself_is_its_literal() {
    use super::scalar::is_the_constant;
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"class Never:\n\
              \x20   def __eq__(self, other):\n\
              \x20       return False\n\
              \x20   __hash__ = object.__hash__\n\
              class Text(str):\n\
              \x20   pass\n\
              NAN = float('nan')\n\
              NEVER = Never()\n\
              TEXT = Text('ab')\n\
              AB = 'ab'\n\
              BUILT = ''.join(['a', 'b'])\n",
            c"constants.py",
            c"constants",
        )
        .expect("the module compiles");
        let get = |name: &str| module.getattr(name).expect("the constant is defined");
        let reflexive = [
            get("AB"),
            PyInt::new(py, 7i64).into_any(),
            PyBool::new(py, true).to_owned().into_any(),
            PyBytes::new(py, b"x").into_any(),
            py.None().into_bound(py),
        ];
        for constant in &reflexive {
            assert!(is_the_constant(constant, constant), "{constant}");
        }
        for name in ["NAN", "NEVER", "TEXT"] {
            assert!(!is_the_constant(&get(name), &get(name)), "{name}");
        }
        let (ab, built) = (get("AB"), get("BUILT"));
        assert!(!ab.is(&built), "two objects of one text");
        assert!(!is_the_constant(&built, &ab));

        let alone = |name: &str| {
            let pool = vec![get(name).unbind()];
            decide(
                py,
                &Schema::Literal(ConstIx::new(0)),
                &get(name),
                &pool,
                &[],
            )
        };
        assert!(alone("AB"));
        assert!(!alone("NAN"), "the same nan is not equal to itself");
        assert!(!alone("NEVER"), "a class's own `==` is asked");
        let pool = vec![
            ab.clone().unbind(),
            PyString::new(py, "cd").into_any().unbind(),
        ];
        let either = Schema::Union((0..2).map(|i| Schema::Literal(ConstIx::new(i))).collect());
        assert!(decide(py, &either, &ab, &pool, &[]));
        assert!(decide(py, &either, &built, &pool, &[]));
        assert!(!decide(
            py,
            &either,
            &PyString::new(py, "ef").into_any(),
            &pool,
            &[]
        ));
    });
}

/// A tuple whose every position is a scalar is read as a type test a position
/// where a level is free under it, whether the walk explains or not, and
/// nowhere else: not at the bound, not where a position or the tail holds a
/// container. Its verdict is the walk's, for a fixed tuple and for a prefix
/// before a tail.
#[test]
fn a_tuple_of_scalar_positions_is_read_as_its_tests() {
    use super::sequence::scalar_positions_tuple_matches;
    Python::attach(|py| {
        let index = build_index(py, &Schema::Int, &[], &[]);
        let state = WalkState::new();
        let ctx = |mode| Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode,
        };
        let eval = |source: &str| {
            py.eval(&std::ffi::CString::new(source).expect("no nul"), None, None)
                .expect("the value evaluates")
        };
        let fixed = [Schema::Int, Schema::Str, Schema::Float];
        let tuple = Schema::tuple(SeqShape::fixed(fixed.clone()));
        case(py, &tuple, &eval("(1, 'a', 1.5)"), true);
        case(py, &tuple, &eval("(True, 'a', 1.5)"), true);
        case(py, &tuple, &eval("(1, 'a', 'x')"), false);
        case(py, &tuple, &eval("(1, 'a')"), false);
        let headed = Schema::tuple(SeqShape::prefix_tail([Schema::Str], Schema::Int));
        case(py, &headed, &eval("('a', 1, 2)"), true);
        case(py, &headed, &eval("('a',)"), true);
        case(py, &headed, &eval("('a', 1, 'x')"), false);

        let read = |source: &str, prefix: &[Schema], tail: Option<&Schema>, mode| {
            let value = eval(source);
            let tuple = value.cast::<PyTuple>().expect("a tuple");
            let (mut path, mut out) = (Vec::new(), Vec::new());
            let mut frame = Frame::new(&mut path, &mut out, ctx(mode));
            scalar_positions_tuple_matches(tuple, prefix, tail, &mut frame)
        };
        assert_eq!(
            read("(1, 'a', 1.5)", &fixed, None, WalkMode::Fast),
            Some(true)
        );
        assert_eq!(
            read("(1, 'a', 'x')", &fixed, None, WalkMode::Fast),
            Some(false)
        );
        assert_eq!(
            read("(1, 'a', 1.5)", &fixed, None, WalkMode::Explain),
            Some(true)
        );
        assert_eq!(
            read("(1, 'a', 'x')", &fixed, None, WalkMode::Explain),
            Some(false)
        );
        let tail = Schema::Int;
        let head = [Schema::Str];
        assert_eq!(
            read("('a', 1, 2)", &head, Some(&tail), WalkMode::Fast),
            Some(true)
        );
        assert_eq!(
            read("('a', 1, 'x')", &head, Some(&tail), WalkMode::Fast),
            Some(false)
        );
        let of_ints = Schema::list(SeqShape::homogeneous(Schema::Int));
        let holding = [Schema::Int, of_ints.clone()];
        assert_eq!(read("(1, [2])", &holding, None, WalkMode::Fast), None);
        assert_eq!(
            read("('a', [1])", &head, Some(&of_ints), WalkMode::Fast),
            None
        );
        state.depth.set(MAX_WALK_DEPTH - 1);
        assert_eq!(
            read("(1, 'a', 1.5)", &fixed, None, WalkMode::Fast),
            Some(true)
        );
        state.depth.set(MAX_WALK_DEPTH);
        assert_eq!(read("(1, 'a', 1.5)", &fixed, None, WalkMode::Fast), None);
        state.depth.set(0);
    });
}

/// A value whose type is the class a schema names is an instance of it, read
/// off the type pointer on `CPython`; a subclass instance and any other value
/// are asked of `isinstance`, and so is every value on `PyPy`. The class's own
/// `__instancecheck__` is not consulted for its exact instances, as `isinstance`
/// does not consult it.
#[test]
fn an_instance_of_the_class_itself_is_read_off_its_type() {
    use super::is_exactly_a;
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"class Refusing(type):\n\
              \x20   def __instancecheck__(cls, instance):\n\
              \x20       return False\n\
              class Base:\n\
              \x20   pass\n\
              class Sub(Base):\n\
              \x20   pass\n\
              class Odd(metaclass=Refusing):\n\
              \x20   pass\n\
              BASE, SUB, ODD = Base(), Sub(), Odd()\n",
            c"instances.py",
            c"instances",
        )
        .expect("the module compiles");
        let get = |name: &str| module.getattr(name).expect("defined");
        let (base, odd) = (get("Base"), get("Odd"));
        assert_eq!(is_exactly_a(&get("BASE"), &base), !cfg!(PyPy));
        assert_eq!(is_exactly_a(&get("ODD"), &odd), !cfg!(PyPy));
        assert!(!is_exactly_a(&get("SUB"), &base));
        assert!(!is_exactly_a(&PyInt::new(py, 1i64).into_any(), &base));

        let instance = Schema::Instance(ClassIx::new(0));
        let pool = vec![base.unbind()];
        assert!(decide(py, &instance, &get("BASE"), &pool, &[]));
        assert!(decide(py, &instance, &get("SUB"), &pool, &[]));
        assert!(!decide(py, &instance, &get("ODD"), &pool, &[]));
        let pool = vec![odd.unbind()];
        assert!(decide(py, &instance, &get("ODD"), &pool, &[]));
        assert!(!decide(py, &instance, &get("BASE"), &pool, &[]));
    });
}

/// An explaining walk answers an element it would admit by the element's test
/// alone -- a scalar, or a union of scalars -- where the levels it would take
/// are free and no fatal signal is recorded, and walks every other element as
/// before: the report names each failing element at its own index, whole or
/// fail-fast, and a passing element names nothing.
#[test]
fn an_admitted_scalar_element_is_answered_quietly_when_explaining() {
    use super::scalar::admitted_quietly;
    Python::attach(|py| {
        let index = build_index(py, &Schema::Int, &[], &[]);
        let state = WalkState::new();
        let ctx = Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode: WalkMode::Explain,
        };
        let one = PyInt::new(py, 1i64).into_any();
        let text = PyString::new(py, "x").into_any();
        let (one, text) = (Value::Py(&one), Value::Py(&text));
        let nullable = Schema::union([Schema::Int, Schema::NoneType]);
        let of_ints = Schema::list(SeqShape::homogeneous(Schema::Int));
        let holding = Schema::union([Schema::Int, of_ints.clone()]);
        assert!(admitted_quietly(&Schema::Int, &one, ctx));
        assert!(!admitted_quietly(&Schema::Int, &text, ctx));
        assert!(admitted_quietly(&nullable, &one, ctx));
        assert!(!admitted_quietly(&nullable, &text, ctx));
        assert!(!admitted_quietly(&of_ints, &one, ctx));
        assert!(!admitted_quietly(&holding, &one, ctx));
        state.depth.set(MAX_WALK_DEPTH - 2);
        assert!(admitted_quietly(&nullable, &one, ctx));
        state.depth.set(MAX_WALK_DEPTH - 1);
        assert!(!admitted_quietly(&nullable, &one, ctx));
        assert!(admitted_quietly(&Schema::Int, &one, ctx));
        state.depth.set(MAX_WALK_DEPTH);
        assert!(!admitted_quietly(&Schema::Int, &one, ctx));
        state.depth.set(0);
        state.fatal_seen.set(true);
        assert!(!admitted_quietly(&Schema::Int, &one, ctx));
        assert!(!admitted_quietly(&nullable, &one, ctx));
        state.fatal_seen.set(false);
    });
}

/// An explaining walk of a sequence or a set of one scalar kind, or of a union
/// of scalars, reports each failing element at its own index -- a set's in the
/// order of what they say -- whole or fail-fast, and a passing element names
/// nothing.
#[test]
fn an_explaining_walk_reports_each_failing_element_of_one_kind() {
    Python::attach(|py| {
        let eval = |source: &str| {
            py.eval(&std::ffi::CString::new(source).expect("no nul"), None, None)
                .expect("the value evaluates")
        };
        let located = |schema: &Schema, source: &str, mode| {
            let (ok, violations) = explain_in(py, schema, &eval(source), &[], &[], mode);
            let at: Vec<Vec<PathSegment>> = violations.into_iter().map(|v| v.path).collect();
            (ok, at)
        };
        let index_at = |i: usize| vec![PathSegment::Index(i)];
        let ints = Schema::list(SeqShape::homogeneous(Schema::Int));
        let nullable = Schema::union([Schema::Int, Schema::NoneType]);
        assert_eq!(
            located(&ints, "[1, 'x', 2, 'y']", WalkMode::Explain),
            (false, vec![index_at(1), index_at(3)])
        );
        assert_eq!(
            located(&ints, "[1, 'x', 2, 'y']", WalkMode::ExplainFailFast),
            (false, vec![index_at(1)])
        );
        assert_eq!(
            located(&ints, "[1, 2, 3]", WalkMode::Explain),
            (true, vec![])
        );
        let nullables = Schema::list(SeqShape::homogeneous(nullable.clone()));
        assert_eq!(
            located(&nullables, "[1, None, 'x']", WalkMode::Explain),
            (false, vec![index_at(2)])
        );
        let pairs = Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Str]));
        assert_eq!(
            located(&pairs, "(1, 2)", WalkMode::Explain),
            (false, vec![index_at(1)])
        );
        // A document is never explained by an entry point, and its readings
        // refuse the mode: explained, a parsed array reports as a list does.
        let parsed_at = |schema: &Schema, source: &str| {
            let parsed = JsonValue::parse(source.as_bytes(), false).expect("the JSON parses");
            let (ok, violations) = explain_json(py, schema, &parsed);
            let at: Vec<Vec<PathSegment>> = violations.into_iter().map(|v| v.path).collect();
            (ok, at)
        };
        assert_eq!(
            parsed_at(&ints, "[1, \"x\", 2, \"y\"]"),
            (false, vec![index_at(1), index_at(3)])
        );
        assert_eq!(
            parsed_at(&nullables, "[1, null, \"x\"]"),
            (false, vec![index_at(2)])
        );
        let nullable_tuple = Schema::tuple(SeqShape::homogeneous(nullable.clone()));
        assert_eq!(
            located(&nullable_tuple, "(None, 'x', 1, 1.5)", WalkMode::Explain),
            (false, vec![index_at(1), index_at(3)])
        );
        let int_tuple = Schema::tuple(SeqShape::homogeneous(Schema::Int));
        assert_eq!(
            located(&int_tuple, "(1, 'x', 2, 'y')", WalkMode::Explain),
            (false, vec![index_at(1), index_at(3)])
        );
        assert_eq!(
            located(&int_tuple, "(1, 'x', 2, 'y')", WalkMode::ExplainFailFast),
            (false, vec![index_at(1)])
        );
        assert_eq!(
            located(&int_tuple, "(1, 2)", WalkMode::Explain),
            (true, vec![])
        );
        // A set's failures carry no index, and are ordered by what they say.
        let int_set = Schema::set(Schema::Int);
        let summaries = |source: &str, mode| {
            let (ok, violations) = explain_in(py, &int_set, &eval(source), &[], &[], mode);
            let said: Vec<String> = violations.into_iter().map(|v| v.value_summary).collect();
            (ok, said)
        };
        let said = |items: &[&str]| items.iter().map(|&s| s.to_owned()).collect::<Vec<_>>();
        assert_eq!(
            summaries("{1, 'y', 2, 'x'}", WalkMode::Explain),
            (false, said(&["'x'", "'y'"]))
        );
        assert_eq!(
            summaries("{1, 'y', 2, 'x'}", WalkMode::ExplainFailFast),
            (false, said(&["'x'"]))
        );
        assert_eq!(summaries("{1, 2, 3}", WalkMode::Explain), (true, vec![]));
    });
}

/// An explaining walk reads a list that belongs as the deciding walk does,
/// through a snapshot where one pays, and a list holding an element that fails
/// in place: the report names each failure at its index, and a subclass is read
/// for what it holds, whatever its `__iter__` yields -- one that yields
/// integers over a storage of strings is refused.
#[test]
fn an_explaining_walk_reads_a_list_that_belongs_through_its_snapshot() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"class Flatterer(list):\n\
              \x20   def __iter__(self):\n\
              \x20       return iter([0] * len(self))\n",
            c"flatterer.py",
            c"flatterer",
        )
        .expect("the module compiles");
        let ints = Schema::list(SeqShape::homogeneous(Schema::Int));
        let wide: Vec<i64> = (0..40).collect();
        let (ok, violations) = explain(
            py,
            &ints,
            &PyList::new(py, &wide).expect("a list"),
            &[],
            &[],
        );
        assert!(ok && violations.is_empty(), "{violations:?}");

        let failing = py
            .eval(c"[*range(30), 'x', *range(9)]", None, None)
            .expect("the list builds");
        let (ok, violations) = explain(py, &ints, &failing, &[], &[]);
        let at: Vec<Vec<PathSegment>> = violations.into_iter().map(|v| v.path).collect();
        assert_eq!((ok, at), (false, vec![vec![PathSegment::Index(30)]]));

        let flatterer = module
            .getattr("Flatterer")
            .expect("the class")
            .call1((vec!["x"; 40],))
            .expect("the subclass builds");
        let (ok, violations) = explain(py, &ints, &flatterer, &[], &[]);
        assert!(!ok && violations.len() == 40, "{violations:?}");
    });
}

/// Each scalar kind with a value of it and a value outside it, where one
/// exists: the readers of a list or a set of one kind take a test per kind,
/// and each is its own code.
fn scalar_kinds() -> [(Schema, Option<&'static str>, Option<&'static str>); 8] {
    [
        (Schema::Int, Some("1"), Some("'x'")),
        (Schema::Str, Some("'a'"), Some("1")),
        (Schema::Float, Some("1.5"), Some("'x'")),
        (Schema::Bool, Some("True"), Some("2")),
        (Schema::Bytes, Some("b'a'"), Some("'a'")),
        (Schema::NoneType, Some("None"), Some("0")),
        (Schema::ANY, Some("1"), None),
        (Schema::Nothing, None, Some("1")),
    ]
}

/// Evaluate a Python expression with no names of its own.
fn evaluate<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyAny> {
    py.eval(&std::ffi::CString::new(source).expect("no nul"), None, None)
        .expect("the value evaluates")
}

/// A list of each scalar kind answers alike in both modes through the kind's
/// own test, read in place and wide enough for a snapshot: every element
/// passing, or the last failing, which the report names at its index.
#[test]
fn a_list_of_each_scalar_kind_is_explained_by_its_own_test() {
    Python::attach(|py| {
        for (kind, good, bad) in scalar_kinds() {
            let list = Schema::list(SeqShape::homogeneous(kind));
            for width in [3, 40] {
                if let Some(good) = good {
                    let value = evaluate(py, &format!("[{good}] * {width}"));
                    assert!(decide(py, &list, &value, &[], &[]), "{good} x{width}");
                }
                let Some(bad) = bad else {
                    continue;
                };
                let source = match good {
                    Some(good) => format!("[{good}] * {} + [{bad}]", width - 1),
                    None => format!("[{bad}]"),
                };
                let value = evaluate(py, &source);
                assert!(!decide(py, &list, &value, &[], &[]), "{source}");
                let (_, violations) = explain(py, &list, &value, &[], &[]);
                let refused_at = if good.is_some() { width - 1 } else { 0 };
                assert_eq!(
                    violations[0].path,
                    vec![PathSegment::Index(refused_at)],
                    "{source}"
                );
            }
        }
    });
}

/// A set and a frozenset of each scalar kind answer alike in both modes
/// through the kind's own scan, deciding and explaining: every element
/// passing, or one failing, which the report names.
#[test]
fn a_set_of_each_scalar_kind_is_scanned_by_its_own_test() {
    Python::attach(|py| {
        for (kind, good, bad) in scalar_kinds() {
            for (schema, wrap) in [
                (Schema::set(kind.clone()), "set"),
                (Schema::frozen_set(kind.clone()), "frozenset"),
            ] {
                if let Some(good) = good {
                    let value = evaluate(py, &format!("{wrap}([{good}])"));
                    assert!(decide(py, &schema, &value, &[], &[]), "{wrap} of {good}");
                }
                let Some(bad) = bad else {
                    continue;
                };
                let source = match good {
                    Some(good) => format!("{wrap}([{good}, {bad}])"),
                    None => format!("{wrap}([{bad}])"),
                };
                let value = evaluate(py, &source);
                assert!(!decide(py, &schema, &value, &[], &[]), "{source}");
                let (_, violations) = explain(py, &schema, &value, &[], &[]);
                assert_eq!(violations.len(), 1, "{source}");
            }
        }
    });
}

/// A list whose element is a union of literals is read by the union's table,
/// found once for the list, where a level is free under it: an element the
/// table decides is its answer, one it does not decide is walked, and the
/// verdict is the walk's in both modes. The reading declines behind a fixed
/// prefix, at the bound, and for a union without a table.
#[test]
fn a_list_of_literals_is_read_by_its_table() {
    use super::sequence::literal_list_matches;
    Python::attach(|py| {
        let pool: Vec<Py<PyAny>> = ["a", "b", "c"]
            .iter()
            .map(|s| PyString::new(py, s).into_any().unbind())
            .collect();
        let union = Schema::Union((0..3).map(|i| Schema::Literal(ConstIx::new(i))).collect());
        let Schema::Union(members) = &union else {
            unreachable!("a union of three literals stays a union")
        };
        let list = Schema::list(SeqShape::homogeneous(union.clone()));
        let eval = |source: &str| {
            py.eval(&std::ffi::CString::new(source).expect("no nul"), None, None)
                .expect("the value evaluates")
        };
        for (source, want) in [
            ("['a', 'b', 'c', 'a']", true),
            ("['a', 'd']", false),
            ("['a', 1.5]", false),
            ("['a', 1]", false),
            ("[''.join(['a', 'b'])[:1]] * 40", true),
            ("['a'] * 39 + ['z']", false),
        ] {
            assert_eq!(
                decide(py, &list, &eval(source), &pool, &[]),
                want,
                "{source}"
            );
        }

        let index = build_index(py, &list, &[], &pool);
        let state = WalkState::new();
        let ctx = |mode| Ctx {
            pool: &pool,
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode,
        };
        let read = |source: &str, prefix: &[Schema], members: &[Schema], mode| {
            let value = eval(source);
            let listed = value.cast::<PyList>().expect("a list");
            let (mut path, mut out) = (Vec::new(), Vec::new());
            let mut frame = Frame::new(&mut path, &mut out, ctx(mode));
            let answer = literal_list_matches(
                listed,
                prefix,
                &union,
                members,
                &Value::Py(&value),
                &mut frame,
            );
            (answer, out.len())
        };
        assert_eq!(
            read("['a', 'c']", &[], members, WalkMode::Fast),
            (Some(true), 0)
        );
        assert_eq!(
            read("['a', 'd']", &[], members, WalkMode::Fast),
            (Some(false), 0)
        );
        assert_eq!(
            read("['a', 2.5]", &[], members, WalkMode::Fast),
            (Some(false), 0)
        );
        assert_eq!(
            read("['a', 'c']", &[], members, WalkMode::Explain),
            (Some(true), 0)
        );
        assert_eq!(
            read("['a', 'd']", &[], members, WalkMode::Explain),
            (Some(false), 1)
        );
        assert_eq!(
            read("['a']", &[Schema::Str], members, WalkMode::Fast),
            (None, 0)
        );
        let unplanned = [Schema::Literal(ConstIx::new(0)), Schema::Str];
        assert_eq!(read("['a']", &[], &unplanned, WalkMode::Fast), (None, 0));
        state.depth.set(MAX_WALK_DEPTH - 1);
        assert_eq!(read("['a']", &[], members, WalkMode::Fast), (Some(true), 0));
        state.depth.set(MAX_WALK_DEPTH);
        assert_eq!(read("['a']", &[], members, WalkMode::Fast), (None, 0));
        state.depth.set(0);
    });
}

/// A list whose element is a class is read off each element's type where a
/// level is free under it: an element of exactly the class is an instance of
/// it, any other is walked, which asks `isinstance`, and the verdict is the
/// walk's in both modes. The reading declines behind a fixed prefix and at the
/// bound, and the reader a list arm hands its union and class tails to reaches
/// both readings.
#[test]
fn a_list_of_one_class_is_read_off_its_elements_types() {
    use super::sequence::{element_list_matches, instance_list_matches};
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"class Base:\n\
              \x20   pass\n\
              class Sub(Base):\n\
              \x20   pass\n\
              ONE, SUB = Base(), Sub()\n",
            c"classes_listed.py",
            c"classes_listed",
        )
        .expect("the module compiles");
        let base = module.getattr("Base").expect("the class");
        let pool = vec![base.unbind()];
        let element = Schema::Instance(ClassIx::new(0));
        let list = Schema::list(SeqShape::homogeneous(element.clone()));
        let values = |source: &str| {
            let globals = module.dict();
            py.eval(
                &std::ffi::CString::new(source).expect("no nul"),
                Some(&globals),
                None,
            )
            .expect("the value evaluates")
        };
        for (source, want) in [
            ("[ONE, ONE]", true),
            ("[ONE, SUB]", true),
            ("[ONE, 1]", false),
            ("[ONE] * 40", true),
            ("[ONE] * 39 + [SUB]", true),
            ("[ONE] * 39 + [2]", false),
        ] {
            assert_eq!(
                decide(py, &list, &values(source), &pool, &[]),
                want,
                "{source}"
            );
        }

        let index = build_index(py, &list, &[], &pool);
        let state = WalkState::new();
        let ctx = |mode| Ctx {
            pool: &pool,
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode,
        };
        let read = |source: &str, prefix: &[Schema], mode| {
            let value = values(source);
            let listed = value.cast::<PyList>().expect("a list");
            let (mut path, mut out) = (Vec::new(), Vec::new());
            let mut frame = Frame::new(&mut path, &mut out, ctx(mode));
            let answer = instance_list_matches(
                listed,
                prefix,
                &element,
                ClassIx::new(0),
                &Value::Py(&value),
                &mut frame,
            );
            (answer, out.len())
        };
        assert_eq!(read("[ONE, SUB]", &[], WalkMode::Fast), (Some(true), 0));
        assert_eq!(read("[ONE, 1]", &[], WalkMode::Fast), (Some(false), 0));
        assert_eq!(read("[ONE, SUB]", &[], WalkMode::Explain), (Some(true), 0));
        assert_eq!(read("[ONE, 1]", &[], WalkMode::Explain), (Some(false), 1));
        assert_eq!(read("[ONE]", &[Schema::Int], WalkMode::Fast), (None, 0));
        state.depth.set(MAX_WALK_DEPTH - 1);
        assert_eq!(read("[ONE]", &[], WalkMode::Fast), (Some(true), 0));
        state.depth.set(MAX_WALK_DEPTH);
        assert_eq!(read("[ONE]", &[], WalkMode::Fast), (None, 0));
        state.depth.set(0);

        // Both kinds of tail reach their reading through the one reader.
        let handed = |source: &str, element: &Schema| {
            let value = values(source);
            let listed = value.cast::<PyList>().expect("a list");
            let (mut path, mut out) = (Vec::new(), Vec::new());
            let mut frame = Frame::new(&mut path, &mut out, ctx(WalkMode::Fast));
            element_list_matches(listed, &[], element, &Value::Py(&value), &mut frame)
        };
        let nullable = Schema::union([Schema::Int, Schema::NoneType]);
        assert_eq!(handed("[ONE]", &element), Some(true));
        assert_eq!(handed("[1, None]", &nullable), Some(true));
        assert_eq!(
            handed("[[1]]", &Schema::list(SeqShape::homogeneous(Schema::Int))),
            None
        );
    });
}

/// A module holding `SUB`, an instance of a `tuple` subclass the tuple-list
/// tests read beside exact tuples, under the name its test gives, for the
/// reason `union_classes` takes one.
fn tuple_subclass<'py>(py: Python<'py>, name: &std::ffi::CStr) -> Bound<'py, PyModule> {
    PyModule::from_code(
        py,
        c"class Pair(tuple):\n\
          \x20   pass\n\
          SUB = Pair((1, 'a'))\n",
        c"tuples_listed.py",
        name,
    )
    .expect("the module compiles")
}

/// Evaluate `source` against `module`'s globals.
fn evaluated<'py>(module: &Bound<'py, PyModule>, source: &str) -> Bound<'py, PyAny> {
    module
        .py()
        .eval(
            &std::ffi::CString::new(source).expect("no nul"),
            Some(&module.dict()),
            None,
        )
        .expect("the value evaluates")
}

/// A list whose element is a tuple of scalar positions answers what the walk
/// answers in both modes: a tuple subclass is admitted through its storage, a
/// wrong arity, position or container refused, in place and through a
/// snapshot, and a refused position is named at its own index within its
/// element.
#[test]
fn a_list_of_scalar_tuples_answers_as_the_walk_does() {
    Python::attach(|py| {
        let module = tuple_subclass(py, c"tuples_answered");
        let element = Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Str]));
        let list = Schema::list(SeqShape::homogeneous(element));
        for (source, want) in [
            ("[(1, 'a'), (2, 'b')]", true),
            ("[(True, 'a')]", true),
            ("[(1, 'a'), SUB]", true),
            ("[]", true),
            ("[(1, 'a'), (1, 2)]", false),
            ("[(1, 'a'), (1,)]", false),
            ("[(1, 'a'), (1, 'a', 2)]", false),
            ("[(1, 'a'), [1, 'a']]", false),
            ("[(1, 'a')] * 40", true),
            ("[(1, 'a')] * 39 + [SUB]", true),
            ("[(1, 'a')] * 39 + [(1, b'a')]", false),
        ] {
            assert_eq!(
                decide(py, &list, &evaluated(&module, source), &[], &[]),
                want,
                "{source}"
            );
        }
        let refused = evaluated(&module, "[(1, 'a'), (1, 2)]");
        let (ok, violations) = explain(py, &list, &refused, &[], &[]);
        let at: Vec<Vec<PathSegment>> = violations.into_iter().map(|v| v.path).collect();
        assert_eq!(
            (ok, at),
            (
                false,
                vec![vec![PathSegment::Index(1), PathSegment::Index(1)]]
            )
        );
    });
}

/// The tuple-list reader settles an exact tuple whose positions pass and walks
/// any other element, in both modes. It declines behind a fixed prefix, for a
/// tuple with a tail or a position that is not a scalar, and where the two
/// levels below the list are not free; the reader a list arm hands its tails
/// to reaches it.
#[test]
fn a_list_of_scalar_tuples_is_read_by_a_test_of_each_element() {
    use super::sequence::{element_list_matches, tuple_list_matches};
    Python::attach(|py| {
        let module = tuple_subclass(py, c"tuples_read");
        let element = Schema::tuple(SeqShape::fixed([Schema::Int, Schema::Str]));
        let list = Schema::list(SeqShape::homogeneous(element.clone()));
        let index = build_index(py, &list, &[], &[]);
        let state = WalkState::new();
        let ctx = |mode| Ctx {
            pool: &[],
            defs: &[],
            records: &index.records,
            attrs: &index.attrs,
            unions: &index.unions,
            regexes: &index.regexes,
            guard: &state.guard,
            depth: &state.depth,
            fatal: &state.fatal,
            fatal_seen: &state.fatal_seen,
            mode,
        };
        let read = |source: &str, prefix: &[Schema], element: &Schema, mode| {
            let Schema::Seq { shape, .. } = element else {
                unreachable!("a tuple schema")
            };
            let value = evaluated(&module, source);
            let listed = value.cast::<PyList>().expect("a list");
            let (mut path, mut out) = (Vec::new(), Vec::new());
            let mut frame = Frame::new(&mut path, &mut out, ctx(mode));
            let answer = tuple_list_matches(
                listed,
                prefix,
                element,
                shape,
                &Value::Py(&value),
                &mut frame,
            );
            (answer, out.len())
        };
        let fast = WalkMode::Fast;
        assert_eq!(
            read("[(1, 'a'), SUB]", &[], &element, fast),
            (Some(true), 0)
        );
        assert_eq!(
            read("[(1, 'a'), (1, 2)]", &[], &element, fast),
            (Some(false), 0)
        );
        assert_eq!(
            read("[(1, 'a')]", &[], &element, WalkMode::Explain),
            (Some(true), 0)
        );
        assert_eq!(
            read("[(1, 'a'), (1, 2)]", &[], &element, WalkMode::Explain),
            (Some(false), 1)
        );
        assert_eq!(
            read("[(1, 'a')]", &[Schema::Int], &element, fast),
            (None, 0)
        );
        let tailed = Schema::tuple(SeqShape::homogeneous(Schema::Int));
        assert_eq!(read("[(1, 2)]", &[], &tailed, fast), (None, 0));
        let nested = Schema::tuple(SeqShape::fixed([
            Schema::Int,
            Schema::list(SeqShape::homogeneous(Schema::Int)),
        ]));
        assert_eq!(read("[(1, [2])]", &[], &nested, fast), (None, 0));
        state.depth.set(MAX_WALK_DEPTH - 2);
        assert_eq!(read("[(1, 'a')]", &[], &element, fast), (Some(true), 0));
        state.depth.set(MAX_WALK_DEPTH - 1);
        assert_eq!(read("[(1, 'a')]", &[], &element, fast), (None, 0));
        state.depth.set(MAX_WALK_DEPTH);
        assert_eq!(read("[(1, 'a')]", &[], &element, fast), (None, 0));
        state.depth.set(0);

        let value = evaluated(&module, "[(1, 'a')]");
        let listed = value.cast::<PyList>().expect("a list");
        let (mut path, mut out) = (Vec::new(), Vec::new());
        let mut frame = Frame::new(&mut path, &mut out, ctx(fast));
        assert_eq!(
            element_list_matches(listed, &[], &element, &Value::Py(&value), &mut frame),
            Some(true)
        );
    });
}
