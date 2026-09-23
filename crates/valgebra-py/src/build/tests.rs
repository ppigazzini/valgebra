use super::*;
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

FAST = [Plain, Derived, Record, Pair, Box, Bare, AtTheBound, Meta]
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
