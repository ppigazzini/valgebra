use super::*;
use pyo3::exceptions::PyTypeError;
use valgebra_core::SeqShape;

#[test]
fn intern_deduplicates_by_identity() {
    Python::attach(|py| {
        let mut pool = Pool::default();
        let a = PyString::new(py, "x").into_any();

        // The same object interns to one slot.
        let first = pool.intern(&a);
        let again = pool.intern(&a);
        assert_eq!(first, again);
        assert_eq!(pool.items().len(), 1);

        // A distinct object takes a new slot.
        let b = PyList::empty(py).into_any();
        let second = pool.intern(&b);
        assert_ne!(first, second);
        assert_eq!(pool.items().len(), 2);

        // Dedup is by identity, not value: a fresh equal-but-distinct object
        // gets its own slot rather than collapsing onto the first.
        let c = PyList::empty(py).into_any();
        let third = pool.intern(&c);
        assert_ne!(second, third);
        assert_eq!(pool.items().len(), 3);
    });
}

/// A definition block is placed once: a block already present at some offset
/// is reused at that offset, shifted references included, and a new one is
/// appended at the end.
#[test]
fn a_definition_block_is_placed_once() {
    let body = |index: usize| {
        Schema::union([
            Schema::Int,
            Schema::list(SeqShape::homogeneous(Schema::Ref(DefIx::new(index)))),
        ])
    };
    let mut defs = Vec::new();

    // Into an empty list: offset zero.
    assert_eq!(place_definitions(&[Schema::Str], &[], &mut defs), 0);
    assert_eq!(defs, vec![Schema::Str]);

    // A block whose body names itself is shifted to the offset it lands at.
    assert_eq!(place_definitions(&[body(0)], &[], &mut defs), 1);
    assert_eq!(defs, vec![Schema::Str, body(1)]);

    // The same block again is found where it already is, and nothing grows.
    assert_eq!(place_definitions(&[body(0)], &[], &mut defs), 1);
    assert_eq!(defs.len(), 2);

    // A block that matches nowhere is appended after the last one.
    assert_eq!(place_definitions(&[Schema::Bytes], &[], &mut defs), 2);
    assert_eq!(defs, vec![Schema::Str, body(1), Schema::Bytes]);
}

/// The variant a node is, by the name the enum gives it.
///
/// Exhaustive on purpose: a variant added to the IR fails to compile here
/// until the table below has a row that builds it, which is the obligation
/// that the IR is exactly as expressive as its producers, held by the
/// compiler rather than by a parse of the enum's source.
fn variant(schema: &Schema) -> &'static str {
    match schema {
        Schema::Anything(_) => "Anything",
        Schema::Nothing => "Nothing",
        Schema::NoneType => "NoneType",
        Schema::Bool => "Bool",
        Schema::Int => "Int",
        Schema::Float => "Float",
        Schema::Str => "Str",
        Schema::Bytes => "Bytes",
        Schema::Literal(_) => "Literal",
        Schema::Seq { .. } => "Seq",
        Schema::Coll { .. } => "Coll",
        Schema::KeyedMap { .. } => "KeyedMap",
        Schema::Union(_) => "Union",
        Schema::Intersection(_) => "Intersection",
        Schema::Complement(_) => "Complement",
        Schema::Instance(_) => "Instance",
        Schema::AttrRecord { .. } => "AttrRecord",
        Schema::Refine { .. } => "Refine",
        Schema::Ref(_) => "Ref",
        Schema::SelfRef(_) => "SelfRef",
    }
}

/// Every variant reachable from `schema`, following each definition once.
fn variants_reached(
    schema: &Schema,
    defs: &[Schema],
    followed: &mut Vec<usize>,
    out: &mut std::collections::BTreeSet<&'static str>,
) {
    out.insert(variant(schema));
    if let Schema::Ref(id) = schema {
        if !followed.contains(&id.get()) {
            followed.push(id.get());
            if let Some(body) = defs.get(id.get()) {
                variants_reached(body, defs, followed, out);
            }
        }
        return;
    }
    let mut descend = |child: &Schema| variants_reached(child, defs, followed, out);
    match schema {
        Schema::Seq { shape, .. } => {
            shape.prefix.iter().for_each(&mut descend);
            if let Some(tail) = &shape.tail {
                descend(tail);
            }
        }
        Schema::Coll { element, .. } => descend(element),
        Schema::KeyedMap { fields, defaults } => {
            fields.iter().for_each(|field| descend(&field.schema));
            for clause in defaults.iter() {
                descend(&clause.key);
                descend(&clause.value);
            }
        }
        Schema::Union(members) | Schema::Intersection(members) => {
            members.iter().for_each(&mut descend);
        }
        Schema::Complement(inner) => descend(inner),
        Schema::AttrRecord { fields } => fields.iter().for_each(|field| descend(&field.schema)),
        Schema::Refine { base, .. } => descend(base),
        Schema::Ref(_)
        | Schema::Anything(_)
        | Schema::Nothing
        | Schema::NoneType
        | Schema::Bool
        | Schema::Int
        | Schema::Float
        | Schema::Str
        | Schema::Bytes
        | Schema::Literal(_)
        | Schema::Instance(_)
        | Schema::SelfRef(_) => {}
    }
}

// THEORY: the-ir-matches-its-producers
/// Every variant of the IR is built by a producer, and one is built by none.
///
/// Each row is an annotation the frontend reads, or a combinator the package
/// exports, compiled through the same entry points a caller reaches; the
/// variants the compiled schema holds are collected by walking it, and the
/// union over the rows is every variant but `SelfRef`, which is the build's
/// own placeholder and is resolved before a schema is returned. The match
/// above is exhaustive, so a variant added to the enum is a compile error
/// here before it is a missing row.
#[test]
fn every_variant_is_built_by_a_producer_and_the_placeholder_by_none() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"from dataclasses import dataclass\n\
              from typing import Annotated, Any, Literal, NoReturn\n\
              class Ge:\n\
              \x20   def __init__(self, ge):\n\
              \x20       self.ge = ge\n\
              Ge.__module__ = 'annotated_types'\n\
              @dataclass\n\
              class Point:\n\
              \x20   x: int\n\
              ROWS = [\n\
              \x20   Any, NoReturn, None, bool, int, float, str, bytes,\n\
              \x20   Literal[1], list[int], set[int], dict[str, int],\n\
              \x20   int | str, Point, Annotated[int, Ge(0)],\n\
              ]\n\
              BUILDER = lambda node: int | list[node]\n",
            c"producers.py",
            c"producers",
        )
        .expect("the rows compile");
        let rows = module.getattr("ROWS").expect("the rows are defined");
        let mut reached = std::collections::BTreeSet::new();
        for row in rows
            .try_iter()
            .expect("a list")
            .map(|row| row.expect("a row"))
        {
            let mut lits = Pool::default();
            let mut defs = Vec::new();
            let schema = build_schema(&row, &mut lits, &mut defs)
                .unwrap_or_else(|error| panic!("{row}: {error}"));
            variants_reached(&schema, &defs, &mut Vec::new(), &mut reached);
        }
        // The two combinators the annotations cannot spell.
        let int = py.get_type::<pyo3::types::PyInt>().into_any();
        let negated = crate::complement(&int).expect("the complement builds");
        variants_reached(
            &negated.schema,
            &negated.definitions,
            &mut Vec::new(),
            &mut reached,
        );
        let builder = module.getattr("BUILDER").expect("the builder is defined");
        let fixpoint = crate::recursive(&builder).expect("the fixpoint builds");
        variants_reached(
            &fixpoint.schema,
            &fixpoint.definitions,
            &mut Vec::new(),
            &mut reached,
        );

        let every: std::collections::BTreeSet<&'static str> = [
            "Anything",
            "Nothing",
            "NoneType",
            "Bool",
            "Int",
            "Float",
            "Str",
            "Bytes",
            "Literal",
            "Seq",
            "Coll",
            "KeyedMap",
            "Union",
            "Intersection",
            "Complement",
            "Instance",
            "AttrRecord",
            "Refine",
            "Ref",
        ]
        .into_iter()
        .collect();
        assert_eq!(
            reached, every,
            "the rows reach these variants and no others"
        );
        assert!(!reached.contains("SelfRef"));
    });
}

/// Classes whose annotations `get_type_hints` hands back unchanged, and classes
/// where it changes one, each row naming which it is. `AtTheBound` and `Deep`
/// have nothing to evaluate: the first is nested as far as
/// `MAX_ANNOTATION_DEPTH` reads and the second past it. `Meta` is a
/// metaclass, whose `__mro__` holds `type` and the descriptor `type` keeps as
/// its `__annotations__`, which the call skips.
const TYPE_HINTS_CORPUS: &str = r#"
import collections.abc, dataclasses, sys, typing
from typing import Annotated, ClassVar, Generic, List, Literal, NamedTuple, Optional, TypeVar, TypedDict

class Ge:
    def __init__(self, ge):
        self.ge = ge
Ge.__module__ = "annotated_types"

T = TypeVar("T")

@dataclasses.dataclass
class Plain:
    a: int
    b: None
    c: Optional[list[int]]
    d: Annotated[int, Ge(0)]
    e: Literal["x", 1]
    f: tuple[int, ...]
    g: int | None
    h: List[Literal["q"]]
    i: Annotated[str, "meta"]

class Derived(Plain):
    j: ClassVar[int]
    a: str

class Record(TypedDict, total=False):
    a: int
    b: Annotated[int, Ge(0)]

class Pair(NamedTuple):
    a: int
    b: Optional[str] = None

@dataclasses.dataclass
class Box(Generic[T]):
    item: T
    items: list[T]

class Bare:
    pass

class Written:
    a: "int"

class Nested:
    a: dict[str, "int"]

class Referenced:
    a: Optional["Plain"]

class Called:
    a: collections.abc.Callable[[int], str]

@typing.no_type_check
class Unchecked:
    a: int

def nested(levels):
    alias = int
    for _ in range(levels):
        alias = list[alias]
    return alias

class AtTheBound:
    a: nested(64)

class Deep:
    a: nested(70)

class Meta(type):
    x: int

class Refusal(ValueError):
    reason: str

class Annotations(dict):
    pass

class Held:
    a: int

Held.__annotations__ = Annotations(Held.__annotations__)

FAST = [Plain, Derived, Record, Pair, Box, Bare, AtTheBound, Meta, Refusal, Held]
DECLINED = [Written, Nested, Referenced, Called, Unchecked, Deep]
if sys.version_info >= (3, 11):
    exec("class Unpacked:\n    a: tuple[int, *tuple[str, ...]]")
    DECLINED.append(Unpacked)
"#;

/// The annotations as written are `get_type_hints`' answer wherever the reading
/// takes them, and the reading declines every class where the call would
/// change a value.
///
/// Equal key order as well as equal values, since a record's fields keep the
/// order the hints give them. The decline half is what keeps the fast path
/// honest: a reading that took `list["int"]` would hand the builder a string
/// where the call hands it `int`.
#[test]
fn annotations_as_written_are_the_hints_get_type_hints_returns() {
    Python::attach(|py| {
        let source = std::ffi::CString::new(TYPE_HINTS_CORPUS).expect("no interior nul");
        let module =
            PyModule::from_code(py, &source, c"type_hints_corpus.py", c"type_hints_corpus")
                .expect("the corpus compiles");
        let get_type_hints = py
            .import("typing")
            .and_then(|typing| typing.getattr("get_type_hints"))
            .expect("typing.get_type_hints");
        let kwargs = PyDict::new(py);
        kwargs.set_item("include_extras", true).expect("a kwarg");
        let classes = |name: &str| {
            module
                .getattr(name)
                .expect("the row list is defined")
                .cast_into::<PyList>()
                .expect("a list")
        };
        for class in classes("FAST").iter() {
            let class = class.cast_into::<PyType>().expect("a class");
            let expected = get_type_hints
                .call((&class,), Some(&kwargs))
                .expect("get_type_hints answers");
            let read = annotations_as_written(&class)
                .expect("the reading answers")
                .unwrap_or_else(|| panic!("{class} was declined"));
            let order = |hints: &Bound<'_, PyAny>| {
                py.get_type::<PyList>()
                    .call1((hints,))
                    .expect("a list of the keys")
                    .unbind()
            };
            assert!(
                read.eq(&expected).expect("dicts compare")
                    && PyAnyMethods::eq(order(read.as_any()).bind(py), order(&expected))
                        .expect("lists compare"),
                "{class}: read {read}, get_type_hints {expected}"
            );
        }
        for class in classes("DECLINED").iter() {
            let class = class.cast_into::<PyType>().expect("a class");
            assert!(
                annotations_as_written(&class)
                    .expect("the reading answers")
                    .is_none(),
                "{class} is one the reading declines, and was read as written"
            );
        }
    });
}

/// The builtin bases a class's annotations are not read from are the ones
/// that hold none on every release, and no other class is skipped.
///
/// Each named builtin reads the empty table through the call the reading
/// makes for it on this release, so skipping it moves no answer; a class of
/// the program's own, a `NamedTuple`, a `TypedDict` and `type` itself, which
/// holds the two descriptors in its namespace, are each still read.
#[test]
fn a_builtin_base_holding_no_annotations_is_not_asked_for_them() {
    use super::classes::annotates_nothing;
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"from typing import NamedTuple, TypedDict\n\
              class Plain:\n\
              \x20   a: int\n\
              class Pair(NamedTuple):\n\
              \x20   x: int\n\
              class Row(TypedDict):\n\
              \x20   a: int\n\
              import sys\n\
              if sys.version_info >= (3, 14):\n\
              \x20   from annotationlib import get_annotations as read\n\
              else:\n\
              \x20   def read(base):\n\
              \x20       return base.__dict__.get('__annotations__', {})\n\
              builtins = [object, tuple, dict, list, int, float, str, bytes, set,\n\
              \x20           frozenset, BaseException, Exception]\n\
              others = [Plain, Pair, Row, type]\n",
            c"annotated_bases.py",
            c"annotated_bases",
        )
        .expect("the module compiles");
        let read = module.getattr("read").expect("the reading");
        let listed = |name: &str| -> Vec<Bound<'_, PyAny>> {
            module
                .getattr(name)
                .and_then(|classes| classes.try_iter()?.collect())
                .expect("the classes read")
        };
        for base in listed("builtins") {
            assert!(annotates_nothing(&base), "{base} is skipped");
            let own = read.call1((&base,)).expect("the reading answers");
            assert!(
                own.cast::<PyDict>().is_ok_and(PyDictMethods::is_empty),
                "{base} holds no annotations"
            );
        }
        for class in listed("others") {
            assert!(!annotates_nothing(&class), "{class} is read");
        }
    });
}

/// The classes `instance_of` reads, one per arm: the two that declare fields,
/// the two shapes that look like a named tuple and are not, the classes the
/// class step already reads as their atom, and the two it refuses.
const INSTANCE_OF_CORPUS: &str = r#"
import dataclasses, enum
from typing import NamedTuple, Protocol, TypedDict

@dataclasses.dataclass
class Box:
    a: int

class Pair(NamedTuple):
    x: int

class Row(tuple):
    pass

class Fielded:
    _fields = ("a",)

class Plain:
    a: int

class Colour(enum.Enum):
    RED = 1

class Record(TypedDict):
    a: int

class HasA(Protocol):
    a: int
"#;

/// `instance_of` reads a class alone: a dataclass and a named tuple are their
/// atom where `Validator` meets it with the fields, every other class is the
/// node `Validator` builds, and a `TypedDict`, a `Protocol` and anything that
/// is not a class are refused.
#[test]
fn instance_of_reads_the_class_alone() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            &std::ffi::CString::new(INSTANCE_OF_CORPUS).expect("no interior NUL"),
            c"instance_of_reads_the_class_alone.py",
            c"instance_of_reads_the_class_alone",
        )
        .expect("the corpus compiles");
        let class = |name: &str| module.getattr(name).expect("the corpus defines it");
        let declared = |class: &Bound<'_, PyAny>| {
            build_schema(class, &mut Pool::default(), &mut Vec::new()).expect("it builds")
        };

        for name in ["Box", "Pair"] {
            let built = crate::instance_of(&class(name)).expect("a class");
            assert!(matches!(built.schema, Schema::Instance(_)), "{name}");
            assert!(
                matches!(declared(&class(name)), Schema::Intersection(_)),
                "{name}"
            );
        }
        // A tuple subclass carrying no `_fields`, and a `_fields` on a class that
        // is no tuple: neither declares fields, so each is the class step's atom.
        for name in ["Row", "Fielded", "Plain", "Colour"] {
            let built = crate::instance_of(&class(name)).expect("a class");
            assert!(matches!(built.schema, Schema::Instance(_)), "{name}");
            assert_eq!(built.schema, declared(&class(name)), "{name}");
        }
        let int = py.get_type::<PyInt>().into_any();
        assert_eq!(
            crate::instance_of(&int).expect("a class").schema,
            Schema::Int
        );

        for (name, says) in [("Record", "is a TypedDict"), ("HasA", "is a Protocol")] {
            let Err(error) = crate::instance_of(&class(name)) else {
                panic!("{name} was read as a class alone");
            };
            assert!(error.is_instance_of::<PyTypeError>(py), "{name}");
            assert!(error.to_string().contains(says), "{name}: {error}");
        }
        let validator = Bound::new(py, crate::instance_of(&int).expect("a class"))
            .expect("it binds")
            .into_any();
        let three = 3_i32.into_pyobject(py).expect("an int").into_any();
        for (value, says) in [(&validator, "the validator int"), (&three, " 3 ")] {
            let Err(error) = crate::instance_of(value) else {
                panic!("{value} was read as a class");
            };
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert!(error.to_string().contains(says), "{error}");
        }
    });
}

/// The atom of a class that declares fields prints as the call that builds it,
/// since its name alone builds the class met with the fields; every other atom
/// prints as the name.
#[test]
fn the_atom_of_a_class_with_fields_prints_as_instance_of() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            &std::ffi::CString::new(INSTANCE_OF_CORPUS).expect("no interior NUL"),
            c"the_atom_of_a_class_with_fields_prints_as_instance_of.py",
            c"the_atom_of_a_class_with_fields_prints_as_instance_of",
        )
        .expect("the corpus compiles");
        for (name, printed) in [
            ("Box", "instance_of(Box)"),
            ("Pair", "instance_of(Pair)"),
            ("Row", "Row"),
            ("Plain", "Plain"),
        ] {
            let class = module.getattr(name).expect("the corpus defines it");
            let built =
                Bound::new(py, crate::instance_of(&class).expect("a class")).expect("it binds");
            assert_eq!(built.repr().expect("it renders").to_string(), printed);
        }
    });
}

/// A `Final[T]` field of a dataclass holds a `T`: the qualifier says the name is
/// not rebound, and the field is read through it. A field whose hint has an
/// origin other than `Final` is read as written, and a bare `Final` names no
/// type and is refused.
#[test]
fn a_final_field_of_a_dataclass_is_read_as_its_type() {
    Python::attach(|py| {
        let module = PyModule::from_code(
            py,
            c"import dataclasses\n\
              from typing import Final\n\
              @dataclasses.dataclass\n\
              class Fixed:\n\
              \x20   a: Final[int] = 1\n\
              @dataclasses.dataclass\n\
              class Listed:\n\
              \x20   a: list[int]\n\
              @dataclasses.dataclass\n\
              class Bare:\n\
              \x20   a: Final = 1\n",
            c"a_final_field_of_a_dataclass_is_read_as_its_type.py",
            c"a_final_field_of_a_dataclass_is_read_as_its_type",
        )
        .expect("the corpus compiles");
        let field = |name: &str| {
            let class = module.getattr(name).expect("the corpus defines it");
            let schema = build_schema(&class, &mut Pool::default(), &mut Vec::new())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let Schema::Intersection(members) = schema else {
                panic!("{name} is not a class met with its fields: {schema:?}");
            };
            members
                .iter()
                .find_map(|member| match member {
                    Schema::AttrRecord { fields } => Some(fields[0].schema.clone()),
                    _ => None,
                })
                .expect("a record of the fields")
        };
        assert_eq!(field("Fixed"), Schema::Int);
        assert!(matches!(field("Listed"), Schema::Seq { .. }));
        let bare = module.getattr("Bare").expect("the corpus defines it");
        assert!(build_schema(&bare, &mut Pool::default(), &mut Vec::new()).is_err());
    });
}

/// A corpus module whose names are the annotations a test builds, named for its
/// test so two tests never share one.
fn corpus<'py>(py: Python<'py>, test: &str, code: &str) -> Bound<'py, PyModule> {
    let file = std::ffi::CString::new(format!("{test}.py")).expect("a module name");
    let name = std::ffi::CString::new(test).expect("a module name");
    let code = std::ffi::CString::new(code).expect("a module body");
    PyModule::from_code(py, &code, &file, &name).expect("the corpus compiles")
}

/// The nodes the schema the frontend builds from the corpus name `annotation`
/// spans, or the refusal it met.
fn held(module: &Bound<'_, PyModule>, annotation: &str) -> Result<usize, String> {
    let annotation = module.getattr(annotation).expect("the corpus defines it");
    build_schema(&annotation, &mut Pool::default(), &mut Vec::new())
        .map(|schema| schema.node_count())
        .map_err(|refusal| refusal.to_string())
}

/// What `Validator::checked` makes of the schema the frontend builds from the
/// corpus name `annotation`: the nodes it spans, or the refusal either met.
fn checked(module: &Bound<'_, PyModule>, annotation: &str) -> Result<usize, String> {
    let annotation = module.getattr(annotation).expect("the corpus defines it");
    let mut pool = Pool::default();
    let mut defs = Vec::new();
    build_schema(&annotation, &mut pool, &mut defs)
        .and_then(|schema| Validator::checked(schema, pool.into_items(), defs))
        .map(|validator| validator.schema.node_count())
        .map_err(|refusal| refusal.to_string())
}

/// An annotation spanning the node bound builds, twice in a row.
///
/// A flat `tuple` of `n` `int`s spans `n + 1` nodes and counts `n` while it is
/// read, its leaves. The outermost descent leaves nothing counted, so a build
/// starts from an empty count whatever the one before it built.
#[test]
fn an_annotation_at_the_node_bound_builds() {
    Python::attach(|py| {
        let test = "an_annotation_at_the_node_bound_builds";
        let module = corpus(
            py,
            test,
            &format!("at_the_bound = tuple[(int,) * {}]\n", MAX_SCHEMA_NODES - 1),
        );
        for _ in 0..2 {
            assert_eq!(held(&module, "at_the_bound"), Ok(MAX_SCHEMA_NODES));
            assert_eq!(checked(&module, "at_the_bound"), Ok(MAX_SCHEMA_NODES));
        }
    });
}

/// The descent after the one that passes the bound is refused, and a build
/// passing it at its last step is refused by `Validator::checked`.
///
/// One `int` past the bound's count of leaves, the tuple has no descent left
/// to refuse it, and the frontend hands back what it built; one more, and that
/// descent refuses, with the count it found.
#[test]
fn the_descent_after_the_bound_refuses_and_the_last_is_checked() {
    Python::attach(|py| {
        let test = "the_descent_after_the_bound_refuses_and_the_last_is_checked";
        let module = corpus(
            py,
            test,
            &format!(
                "last = tuple[(int,) * {}]\nafter = tuple[(int,) * {}]\n",
                MAX_SCHEMA_NODES + 1,
                MAX_SCHEMA_NODES + 2,
            ),
        );
        assert_eq!(held(&module, "last"), Ok(MAX_SCHEMA_NODES + 2));
        let refusal = checked(&module, "last").expect_err("past the bound");
        assert!(refusal.contains("spans 100002 nodes"), "{refusal}");
        let refusal = held(&module, "after").expect_err("past the bound");
        assert!(refusal.contains("counts 100001 nodes"), "{refusal}");
        // A refused build leaves nothing counted either: the thread builds the
        // next annotation from an empty count and the depth it started at.
        assert_eq!(held(&module, "last"), Ok(MAX_SCHEMA_NODES + 2));
    });
}

/// The parts a build holds count together before their parent is built from
/// them.
///
/// A part of 40,000 nodes named twice fits; named three times, the build is
/// refused at the descent after the leaf that passes the bound, before the
/// tuple holding them is built.
#[test]
fn the_parts_a_build_holds_count_together() {
    Python::attach(|py| {
        let test = "the_parts_a_build_holds_count_together";
        let module = corpus(
            py,
            test,
            "part = tuple[(int,) * 39_999]\n\
             twice = tuple[part, part]\n\
             three_times = tuple[part, part, part]\n",
        );
        assert_eq!(held(&module, "twice"), Ok(80_001));
        let refusal = held(&module, "three_times").expect_err("past the bound");
        assert!(refusal.contains("counts 100001 nodes"), "{refusal}");
    });
}

/// A named tuple's fields are read once, as the positions they lay out, so a
/// build of named tuples counts what it builds.
///
/// The class is its `isinstance` atom met with the tuple of its positions;
/// the record of its attributes says the same of the same values and is not
/// built beside it. Built and dropped, that record was still counted, because
/// the node guard holds every node a build has read: sixty uses of a
/// thousand-field named tuple span about sixty thousand nodes and counted
/// twice that, past the bound a schema of their size is under.
#[test]
fn a_named_tuple_is_read_once_and_counted_once() {
    Python::attach(|py| {
        let test = "a_named_tuple_is_read_once_and_counted_once";
        let module = corpus(
            py,
            test,
            "from typing import NamedTuple\n\
             Wide = NamedTuple('Wide', [(f'f{i}', int) for i in range(1000)])\n\
             sixty = tuple[(Wide,) * 60]\n",
        );
        let spans = 1 + 60 * (1 + 1 + 1 + 1000);
        assert!(spans < MAX_SCHEMA_NODES && 2 * spans > MAX_SCHEMA_NODES);
        assert_eq!(
            held(&module, "Wide"),
            Ok(1003),
            "the class met with its positions"
        );
        assert_eq!(held(&module, "sixty"), Ok(spans));
        assert_eq!(checked(&module, "sixty"), Ok(spans));
    });
}

/// A class named from many places is refused at the node bound rather than
/// built whole.
///
/// Four records a level, each holding the union of the level below: the
/// classes are a graph of four a level, and the schema is a tree whose size
/// multiplies by four each level -- 436,901 nodes at eight, 134 million at
/// twelve. Held as it is read, the build is refused a step past the bound. Six
/// levels fit. Eight rather than twelve, so that a mutant that stops the count
/// builds the tree it names and fails, rather than outlasting the sweep;
/// `tests/test_adversarial_bounds.py` reads twelve in a child process.
#[test]
fn a_class_named_from_many_places_is_refused_before_it_is_built() {
    Python::attach(|py| {
        let test = "a_class_named_from_many_places_is_refused_before_it_is_built";
        let module = corpus(
            py,
            test,
            "from typing import Literal, TypedDict\n\
             def levels(depth):\n\
             \x20   below = int\n\
             \x20   for level in range(depth):\n\
             \x20       tagged = [TypedDict(f'L{level}{tag}', {'type': Literal[tag], 'left': below}) for tag in 'abcd']\n\
             \x20       below = tagged[0] | tagged[1] | tagged[2] | tagged[3]\n\
             \x20   return below\n\
             six = levels(6)\n\
             eight = levels(8)\n",
        );
        assert!(checked(&module, "six").is_ok_and(|nodes| nodes <= MAX_SCHEMA_NODES));
        let refusal = held(&module, "eight").expect_err("past the bound");
        assert!(refusal.contains("schema is too large"), "{refusal}");
    });
}

/// A compiled validator named in an annotation counts every node of its
/// schema.
///
/// It is one descent however large its schema: a `tuple` of 59,999 `int`s is
/// 60,000 nodes in one step, so naming it twice counts 120,000, which the next
/// descent finds past the bound.
#[test]
fn a_compiled_validator_counts_every_node_it_brings() {
    Python::attach(|py| {
        let test = "a_compiled_validator_counts_every_node_it_brings";
        let module = corpus(py, test, "");
        let part = Validator::new(
            Schema::tuple(SeqShape::fixed(vec![Schema::Int; 59_999])),
            Vec::new(),
            Vec::new(),
        );
        module
            .add("part", Py::new(py, part).expect("a validator object"))
            .expect("the corpus takes it");
        py.run(
            c"twice = tuple[part, part]\nthen_more = tuple[part, part, int]",
            Some(&module.dict()),
            None,
        )
        .expect("the annotations are written");
        assert_eq!(held(&module, "part"), Ok(60_000));
        assert_eq!(held(&module, "twice"), Ok(120_001));
        let refusal = held(&module, "then_more").expect_err("past the bound");
        assert!(refusal.contains("counts 120000 nodes"), "{refusal}");
    });
}

/// A part a fold drops still counts toward the union that dropped it.
///
/// A union counts its members' leaves, read without walking what the union
/// folded them into: `Union[part, Any]` is the top, one node, and the 99,998
/// leaves of the part it absorbed are counted all the same -- which is what
/// lets a result built from parts be counted without a pass over it.
#[test]
fn a_part_a_fold_drops_still_counts_toward_the_union_that_dropped_it() {
    Python::attach(|py| {
        let test = "a_part_a_fold_drops_still_counts_toward_the_union_that_dropped_it";
        let module = corpus(
            py,
            test,
            "from typing import Any, Union\n\
             part = tuple[(int,) * 99_998]\n\
             absorbed = Union[part, Any]\n\
             beside = tuple[absorbed, int, int, int]\n",
        );
        assert_eq!(held(&module, "absorbed"), Ok(1));
        let refusal = held(&module, "beside").expect_err("counted with its part");
        assert!(refusal.contains("counts 100001 nodes"), "{refusal}");
    });
}

/// Past as many descents as the bound, a container above a leaf counts too.
///
/// `list[int]` is two nodes and one leaf, so a part of 30,000 of them counts
/// half of what it spans while the build is small. Three such parts span
/// 180,003 nodes and 90,000 leaves; the build passes as many descents as the
/// bound in the second, walks each result after, and is refused in the third,
/// where a count of leaves alone stays under the bound and hands back a schema
/// past it.
#[test]
fn a_chain_of_containers_is_counted_once_the_build_is_large() {
    Python::attach(|py| {
        let test = "a_chain_of_containers_is_counted_once_the_build_is_large";
        let module = corpus(
            py,
            test,
            "part = tuple[(list[int],) * 30_000]\n\
             three = tuple[part, part, part, int]\n",
        );
        assert_eq!(held(&module, "part"), Ok(60_001));
        let refusal = held(&module, "three").expect_err("past the bound");
        assert!(refusal.contains("schema is too large"), "{refusal}");
    });
}

/// The count turns exact at the descent after the bound's number of descents,
/// and not at the one that reaches it.
///
/// The union below is the hundred-thousandth descent of its build: it counts
/// its parts, though it folds them into `Any`, and the `int` after it finds the
/// build past the bound. Its parts are the descents after, so they are walked,
/// and walking the union too would count the one node it is.
#[test]
fn the_count_turns_exact_at_the_descent_past_the_bounds_count_of_descents() {
    Python::attach(|py| {
        let test = "the_count_turns_exact_at_the_descent_past_the_bounds_count_of_descents";
        let module = corpus(
            py,
            test,
            &format!(
                "from typing import Any, Union\n\
                 build = tuple[(int,) * {} + (Union[tuple[int], Any], int)]\n",
                MAX_SCHEMA_NODES - 2,
            ),
        );
        let refusal = held(&module, "build").expect_err("past the bound");
        assert!(refusal.contains("counts 100001 nodes"), "{refusal}");
    });
}
