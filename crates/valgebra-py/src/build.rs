//! The schema frontend: build the IR from Python types, typing annotations,
//! native container forms, and already-compiled validators.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use pyo3::PyTypeInfo;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyModule, PySet, PyString,
    PyTuple, PyType,
};
use valgebra_core::{
    ClassIx, ConstIx, DefIx, DefShift, Guarded, OperandIx, PredIx, Schema, fresh_self_token,
};

use crate::errors::summarize;
use crate::validator::{MAX_SCHEMA_NODES, Validator};

/// The `build_schema` recursion running on this thread, read and written once on
/// each side of a descent.
struct Build {
    /// How deep it is, bounding it so a self-referential class (whose field type
    /// names the class) fails cleanly instead of recursing until the native
    /// stack overflows.
    depth: Cell<usize>,
    /// How many descents it has made, which decides how a result is counted.
    descents: Cell<usize>,
    /// The schema nodes it holds: what its finished descents returned that
    /// their parents have not yet been built from.
    ///
    /// An annotation names a class, an alias or any annotation object once,
    /// and the frontend builds it once per place it is named -- so a part
    /// shared through several levels is built as a tree whose size multiplies
    /// with each level. Holding the count to the node bound as the build goes
    /// is what refuses that tree instead of building it.
    held: Cell<usize>,
}

thread_local! {
    static BUILD: Build = const {
        Build {
            depth: Cell::new(0),
            descents: Cell::new(0),
            held: Cell::new(0),
        }
    };

    /// The PEP 695 aliases whose bodies are being built on this thread.
    ///
    /// One entry per alias, holding the object's address, the self-reference
    /// token standing for it while its body is read, and whether that token was
    /// handed out. An alias reached again while its own body is being built is
    /// the fixpoint's back edge, and this is what tells the two apart -- the
    /// address, because an alias is one object and `__value__` yields the same
    /// one however many times it is read.
    static OPEN_ALIASES: RefCell<Vec<(usize, u64, Cell<bool>)>> =
        const { RefCell::new(Vec::new()) };
}

/// Build the body of a PEP 695 alias, tying the knot where it names itself.
///
/// `type Json = int | str | list[Json] | dict[str, Json]` is the standard
/// spelling of a recursive type, and the annotation is the whole definition: the
/// alias object is reached again while its own body is being read, and there is
/// no lambda to carry the fixpoint. So the alias *is* the binder. A token stands
/// for it while the body is built, every occurrence of the token becomes a
/// reference to the definition the body turns into, and an alias that never
/// names itself builds exactly what it did before -- one schema, no definition,
/// nothing to resolve.
///
/// The contractivity check is the same one `recursive` runs, for the same
/// reason: `type X = int | X` names a set no value settles, and a walk over it
/// would not terminate.
fn build_alias(
    obj: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let address = obj.as_ptr() as usize;
    if let Some(token) = OPEN_ALIASES.with_borrow(|open| {
        open.iter()
            .find(|(at, _, _)| *at == address)
            .map(|(_, token, used)| {
                used.set(true);
                *token
            })
    }) {
        return Ok(Schema::SelfRef(token));
    }

    let token = fresh_self_token();
    OPEN_ALIASES.with_borrow_mut(|open| open.push((address, token, Cell::new(false))));
    // The value is a type *argument*, as a `NewType`'s supertype is: a string
    // there is a forward reference, and `TypeAliasType("A", "int")` and `type A
    // = "int"` were read as the literal `'int'`.
    let body =
        generics::build_type_argument(&obj.getattr(intern!(obj.py(), "__value__"))?, lits, defs);
    let recursive = OPEN_ALIASES
        .with_borrow_mut(Vec::pop)
        .is_some_and(|(_, _, used)| used.get());
    let body = body?;
    if !recursive {
        return Ok(body);
    }

    // The body becomes a definition and every occurrence of the token becomes a
    // reference to it -- in the body, and in any definition the body's own build
    // appended, since an inner fixpoint may name this one.
    let ref_id = DefIx::new(defs.len());
    for definition in defs.iter_mut() {
        *definition = definition.resolve_self(token, ref_id);
    }
    let resolved = body.resolve_self(token, ref_id);
    if resolved.occurs_unguarded_under(ref_id, Guarded::No, defs) {
        return Err(PyValueError::new_err(
            "recursive type alias is not contractive: the alias names itself \
             outside any structural constructor, so it denotes no set. Put the \
             self-reference under a list, tuple, set, dict, record, or object",
        ));
    }
    defs.push(resolved);
    Ok(Schema::Ref(ref_id))
}

/// The most levels of schema nesting the frontend descends while compiling.
///
/// One level above the published construction depth, so the frontend never
/// rejects a schema the construction bounds would accept: below this,
/// `Validator::checked` owns the verdict and reports which bound tripped. A
/// self-referential class is the shape that reaches this one, because its field
/// type names the class and the descent does not terminate.
///
/// The margin is what makes the ownership hold rather than happen to hold. Each
/// level the frontend descends builds at least one schema node, so the deepest
/// schema `checked` accepts is reached in at most that many descents; one more
/// lets the schema *past* the bound be built and refused by name. The two
/// constants were equal while a sequence counted two levels of depth per level of
/// descent, which hid the question: no list chain could reach this bound before
/// `checked` had already refused it.
const MAX_BUILD_DEPTH: usize = crate::validator::MAX_SCHEMA_DEPTH + 1;

/// RAII guard over one `build_schema` descent: it bounds the recursion's depth
/// and holds what the build has built to the node bound. Entering past either
/// bound is an error; leaving (including on an early `?`) restores the depth and
/// leaves what the descent built counted.
///
/// **What a result counts for.** A descent's result is built from the parts
/// the descents below it returned and replaces them, so the count is every
/// finished part still waiting for its parent. A leaf counts one; a result
/// built from parts counts them and nothing of its own, which is what keeps an
/// alias or `NotRequired[...]` -- a descent passing its one part through -- from
/// counting that part twice; a compiled validator named in an annotation, the
/// one form bringing many nodes in one descent, counts every node of its schema
/// (`hold_whole`), as a `Literal`'s callable constant, built without a descent
/// of its own, counts its one. So the count is never more than `Validator::checked`
/// counts of the schema returned, except where a fold drops a part -- a member a
/// union absorbs into `Any` still counts -- and a schema within the bound is
/// never refused here.
///
/// **Past as many descents as the bound, a result is counted by a walk.** A
/// count of leaves leaves out the nodes above them, and a build that names a
/// chain of containers from many places builds many more nodes than leaves. A
/// build that has made more descents than the bound walks each result it
/// returns from then on, whose nodes then count exactly; a smaller one never
/// walks.
///
/// **The bound is asked as a descent begins.** A descent that finds the build
/// holding more than the bound refuses, which is the step after the one that
/// passed it, and the result a finished descent hands back is never replaced:
/// the build makes nothing past the bound but the step that crossed it, and a
/// crossing at the last step is `Validator::checked`'s to refuse.
struct BuildGuard {
    /// What the build held when this descent began.
    before: usize,
    /// Whether the build had made more descents than the node bound when this
    /// one began, so that its result is counted by a walk.
    walks: bool,
    /// The nodes its result spans, where it was counted by a walk.
    walked: Option<usize>,
}

impl BuildGuard {
    fn enter() -> PyResult<Self> {
        let (depth, descents, before) = BUILD.with(|build| {
            let depth = build.depth.get() + 1;
            build.depth.set(depth);
            let descents = build.descents.get() + 1;
            build.descents.set(descents);
            (depth, descents, build.held.get())
        });
        if depth > MAX_BUILD_DEPTH {
            BUILD.with(|build| build.depth.set(depth - 1));
            return Err(not_implemented(
                format!(
                    "schema nesting is too deep to compile: the frontend descended \
                 {MAX_BUILD_DEPTH} levels without reaching a leaf. A class whose \
                 own type appears in its fields is recursive and never bottoms \
                 out; write it with recursive(...), which ties the fixpoint \
                 explicitly."
                )
                .as_str(),
            ));
        }
        if before > MAX_SCHEMA_NODES {
            BUILD.with(|build| build.depth.set(depth - 1));
            return Err(too_large(before));
        }
        Ok(BuildGuard {
            before,
            walks: descents > MAX_SCHEMA_NODES,
            walked: None,
        })
    }
}

/// The refusal of a build holding `held` nodes, out of the descent's line: it
/// is built once per refused annotation and never on the way to a schema.
#[cold]
#[inline(never)]
fn too_large(held: usize) -> PyErr {
    PyValueError::new_err(format!(
        "schema is too large: reading this annotation counts {held} nodes before it \
         finishes, past the limit of {MAX_SCHEMA_NODES}. Each class, alias or \
         annotation object is built once per place it is named, so a part shared \
         through several levels multiplies the size with each; the frontend \
         refuses at the step after the one that passes the limit rather than \
         building the rest."
    ))
}

/// Hold a schema a descent brings whole as nodes of that descent: every one of
/// them, the top included. A compiled validator named in an annotation is one,
/// and a `Literal`'s callable constant ([`build_constant`]) another.
fn hold_whole(schema: &Schema) {
    let nodes = schema.node_count();
    BUILD.with(|build| build.held.set(build.held.get() + nodes));
}

impl Drop for BuildGuard {
    fn drop(&mut self) {
        BUILD.with(|build| {
            let depth = build.depth.get() - 1;
            build.depth.set(depth);
            let held = build.held.get();
            // A failed descent fails every descent above it, so what it leaves
            // counted is read by no other.
            build.held.set(if depth == 0 {
                // The outermost leaves nothing, so the next build starts from
                // an empty count.
                build.descents.set(0);
                0
            } else if let Some(nodes) = self.walked {
                self.before + nodes
            } else if held == self.before {
                // A leaf: nothing was built below it.
                held + 1
            } else {
                held
            });
        });
    }
}

/// Per-interpreter cache of the typing and builtins special-form objects the
/// frontend compares schema descriptions against. Resolved once on first
/// compile rather than re-fetched on every node, so a deep schema does not
/// re-import a module and re-`getattr` the same singleton dozens of times. The
/// handles are immutable interpreter singletons; the optional ones are absent on
/// older Pythons (`Never`/`TypeAliasType`).
struct Forms {
    any: Py<PyAny>,
    never: Option<Py<PyAny>>,
    noreturn: Option<Py<PyAny>>,
    type_alias_type: Option<Py<PyAny>>,
    union: Py<PyAny>,
    optional: Py<PyAny>,
    union_type: Py<PyAny>,
    /// `types.GenericAlias`, the class of `list[int]`, which `PyPy` defines in a
    /// module of its own, so it is known by identity rather than by module.
    generic_alias: Py<PyAny>,
    literal: Py<PyAny>,
    /// `typing.Annotated` itself, which is a class below 3.13: written bare it
    /// read as an `isinstance` test no value passes.
    annotated: Py<PyAny>,
    /// The class of an `Annotated[T, ...]` alias, `typing._AnnotatedAlias`,
    /// which `typing_extensions` spells with `typing`'s own `Annotated`.
    annotated_alias: Py<PyType>,
    /// `typing.get_origin` and `typing.get_args`, the spec's own introspection,
    /// held as the callables they are.
    ///
    /// Every node asks both, and asking them through the module means importing
    /// `typing` and resolving the name per node: the import machinery alone was
    /// 7.7% of building a fifty-field validator, for a module `sys.modules` has
    /// held since the first one.
    get_origin: Py<PyAny>,
    get_args: Py<PyAny>,
    /// `typing.ForwardRef`, and the four classes a type variable or special
    /// form is an instance of, in the order they are asked.
    forward_ref: Option<Py<PyAny>>,
    type_variables: Vec<Py<PyAny>>,
    /// The `TypedDict` field qualifiers: `Required`, `NotRequired`,
    /// `ReadOnly`. Absent where the runtime is older than the one that spells
    /// them.
    required: Option<Py<PyAny>>,
    not_required: Option<Py<PyAny>>,
    read_only: Option<Py<PyAny>>,
    /// `typing.Unpack`, the origin of an unpacked tuple.
    unpack: Option<Py<PyAny>>,
    /// `typing.get_type_hints`, which resolves a class's annotations.
    get_type_hints: Py<PyAny>,
    object: Py<PyAny>,
    /// `builtins.frozendict` (PEP 814), absent below 3.15.
    frozendict: Option<Py<PyAny>>,
    /// `typing.TypedDict` and `typing.NamedTuple`, the bases a class is
    /// declared from, which a bare annotation names in place of a class.
    class_factories: Vec<Py<PyAny>>,
    enum_class: Py<PyAny>,
    callable: Py<PyAny>,
    ellipsis: Py<PyAny>,
    /// `typing.Protocol`, the base a protocol is declared from, which names no
    /// set of its own.
    protocol: Py<PyAny>,
    /// `typing._get_protocol_attrs`, the derivation of a protocol's member
    /// names that `typing` caches as `__protocol_attrs__` from 3.12. Read only
    /// where a protocol carries no cache, and absent on a release that has
    /// neither.
    get_protocol_attrs: Option<Py<PyAny>>,
    /// `typing.ClassVar` and `typing.Final`, which say where a protocol
    /// member's value lives rather than what it is.
    class_var: Py<PyAny>,
    final_qualifier: Py<PyAny>,
    /// `builtins.property`, the one class attribute a protocol member is read
    /// through its getter for.
    property: Py<PyAny>,
}

static FORMS: PyOnceLock<Forms> = PyOnceLock::new();

/// The special-form cache for this interpreter, built once. `PyOnceLock` makes
/// the one-time initialization safe under free-threading.
///
/// **`annotationlib` is imported before `typing` is read**, and `ForwardRef` is
/// read out of it. On 3.15 `typing` serves `ForwardRef` through its module
/// `__getattr__`, which reaches `annotationlib` through a lazy import, and
/// resolving a lazy import holds the interpreter's global import lock while it
/// waits for the module. A thread importing `annotationlib` at that moment --
/// through `dataclasses` or `inspect`, say -- needs that lock for the imports
/// its body makes, so read through `typing` the two wait on each other for
/// good, and building the first validator hangs the process. An ordinary
/// import waits on the module's own lock alone, and once it returns every lazy
/// import of the module finds it loaded and waits on nothing. The module exists
/// from 3.14, where the two `ForwardRef`s are one class; a release without it
/// reads `typing`'s, as every optional form here is read.
fn forms(py: Python<'_>) -> PyResult<&'static Forms> {
    FORMS.get_or_try_init(py, || {
        let annotationlib = py.import("annotationlib").ok();
        let typing = py.import("typing")?;
        let builtins = py.import("builtins")?;
        let optional_form = |module: &Bound<'_, PyModule>, name: &str| -> Option<Py<PyAny>> {
            module.getattr(name).ok().map(Bound::unbind)
        };
        Ok(Forms {
            any: typing.getattr("Any")?.unbind(),
            never: optional_form(&typing, "Never"),
            noreturn: optional_form(&typing, "NoReturn"),
            type_alias_type: optional_form(&typing, "TypeAliasType"),
            union: typing.getattr("Union")?.unbind(),
            optional: typing.getattr("Optional")?.unbind(),
            union_type: py.import("types")?.getattr("UnionType")?.unbind(),
            generic_alias: py.import("types")?.getattr("GenericAlias")?.unbind(),
            literal: typing.getattr("Literal")?.unbind(),
            annotated: typing.getattr("Annotated")?.unbind(),
            annotated_alias: typing
                .getattr("Annotated")?
                .get_item((py.get_type::<PyInt>(), 0))?
                .get_type()
                .unbind(),
            get_origin: typing.getattr("get_origin")?.unbind(),
            get_args: typing.getattr("get_args")?.unbind(),
            forward_ref: optional_form(annotationlib.as_ref().unwrap_or(&typing), "ForwardRef"),
            type_variables: ["TypeVar", "ParamSpec", "TypeVarTuple", "_SpecialForm"]
                .iter()
                .filter_map(|name| optional_form(&typing, name))
                .collect(),
            required: optional_form(&typing, "Required"),
            not_required: optional_form(&typing, "NotRequired"),
            read_only: optional_form(&typing, "ReadOnly"),
            unpack: optional_form(&typing, "Unpack"),
            get_type_hints: typing.getattr("get_type_hints")?.unbind(),
            object: builtins.getattr("object")?.unbind(),
            frozendict: optional_form(&builtins, "frozendict"),
            class_factories: ["TypedDict", "NamedTuple"]
                .iter()
                .filter_map(|name| optional_form(&typing, name))
                .collect(),
            enum_class: py.import("enum")?.getattr("Enum")?.unbind(),
            callable: py.import("collections.abc")?.getattr("Callable")?.unbind(),
            ellipsis: builtins.getattr("Ellipsis")?.unbind(),
            protocol: typing.getattr("Protocol")?.unbind(),
            get_protocol_attrs: optional_form(&typing, "_get_protocol_attrs"),
            class_var: typing.getattr("ClassVar")?.unbind(),
            final_qualifier: typing.getattr("Final")?.unbind(),
            property: builtins.getattr("property")?.unbind(),
        })
    })
}

/// The special forms `typing_extensions` spells with objects of its own.
///
/// Before the release that adds a form to `typing`, `typing_extensions` defines
/// it itself, so the name is a different object from anything [`Forms`] holds:
/// `Any`, `Never`, `Self` and the `TypedDict` qualifiers on 3.10, and
/// `TypeAliasType` before 3.15. Asked against `typing` alone, such a form reads
/// as something it is not -- `Never` as a literal of the form object, `Any` as
/// a class nothing is an instance of, `Required[int]` as an unsupported form.
/// Where a release spells the form in `typing`, the two names are one object
/// and nothing here changes a reading.
///
/// Asked only where a form would otherwise be read as something else, so a
/// schema that names none of them pays nothing for it.
pub(crate) struct Extensions {
    pub(crate) any: Option<Py<PyAny>>,
    pub(crate) never: Option<Py<PyAny>>,
    type_alias_type: Option<Py<PyAny>>,
    /// The class `Self`, `LiteralString` and `Never` are instances of below
    /// 3.11, which is not `typing._SpecialForm`.
    special_form: Option<Py<PyAny>>,
    pub(crate) required: Option<Py<PyAny>>,
    pub(crate) not_required: Option<Py<PyAny>>,
    pub(crate) read_only: Option<Py<PyAny>>,
    pub(crate) unpack: Option<Py<PyAny>>,
    /// Its `Protocol`, which is its own below 3.14.
    protocol: Option<Py<PyAny>>,
    /// Its `TypedDict` and `NamedTuple`, which are its own on some releases.
    class_factories: Vec<Py<PyAny>>,
}

static EXTENSIONS: PyOnceLock<Extensions> = PyOnceLock::new();

/// `sys.modules`, held so that asking whether a module is loaded is one
/// dictionary lookup rather than an import of `sys` per question.
static MODULES: PyOnceLock<Py<PyDict>> = PyOnceLock::new();

/// The table of loaded modules, `sys.modules`, read once per interpreter.
///
/// An import of `sys` is `__import__` called through the import machinery,
/// with its arguments built by format string, every time it is asked: some two
/// thousand instructions, which a lookup the build makes per class pays again
/// for every class it compiles.
pub(crate) fn loaded_modules(py: Python<'_>) -> PyResult<&Bound<'_, PyDict>> {
    MODULES
        .get_or_try_init(py, || -> PyResult<Py<PyDict>> {
            Ok(py
                .import(intern!(py, "sys"))?
                .getattr(intern!(py, "modules"))?
                .cast_into::<PyDict>()?
                .unbind())
        })
        .map(|modules| modules.bind(py))
}

/// The `typing_extensions` forms, once the module is loaded.
///
/// Looked up in `sys.modules` rather than imported: an object of the module
/// exists only once the module is loaded, and a lookup runs no module's code.
/// Held once found loaded; until then every call looks again, since a program
/// may import the module after its first validator, or on another thread
/// while one is built.
pub(crate) fn extensions(py: Python<'_>) -> PyResult<Option<&'static Extensions>> {
    if let Some(held) = EXTENSIONS.get(py) {
        return Ok(Some(held));
    }
    let Some(module) = loaded_modules(py)?.get_item(intern!(py, "typing_extensions"))? else {
        return Ok(None);
    };
    // A module is in `sys.modules` before its body has run, and another thread
    // may be running it: a form the body has not reached yet would be held as
    // absent for good. `importlib` marks the module's spec while it loads, and
    // until the mark is gone the module is read as not loaded.
    if let Some(spec) = module.getattr_opt(intern!(py, "__spec__"))?
        && let Some(loading) = spec.getattr_opt(intern!(py, "_initializing"))?
        && loading.is_truthy()?
    {
        return Ok(None);
    }
    let form = |name: &str| -> PyResult<Option<Py<PyAny>>> {
        Ok(module.getattr_opt(name)?.map(Bound::unbind))
    };
    let found = Extensions {
        any: form("Any")?,
        never: form("Never")?,
        type_alias_type: form("TypeAliasType")?,
        special_form: form("_SpecialForm")?,
        required: form("Required")?,
        not_required: form("NotRequired")?,
        read_only: form("ReadOnly")?,
        unpack: form("Unpack")?,
        protocol: form("Protocol")?,
        class_factories: ["TypedDict", "NamedTuple"]
            .into_iter()
            .map(form)
            .filter_map(Result::transpose)
            .collect::<PyResult<_>>()?,
    };
    Ok(Some(EXTENSIONS.get_or_init(py, || found)))
}

/// Whether `obj` is the `typing_extensions` form `pick` names.
pub(crate) fn is_extension(
    obj: &Bound<'_, PyAny>,
    pick: fn(&Extensions) -> &Option<Py<PyAny>>,
) -> PyResult<bool> {
    let py = obj.py();
    Ok(extensions(py)?
        .and_then(|held| pick(held).as_ref())
        .is_some_and(|form| obj.is(form.bind(py))))
}

/// What a `typing_extensions` form with no `typing` counterpart here reads as:
/// `Never` is the bottom and a `TypeAliasType` its aliased type, as their
/// `typing` spellings are.
fn read_extension(
    obj: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Option<Schema>> {
    let py = obj.py();
    let Some(held) = extensions(py)? else {
        return Ok(None);
    };
    if held
        .never
        .as_ref()
        .is_some_and(|never| obj.is(never.bind(py)))
    {
        return Ok(Some(Schema::Nothing));
    }
    if let Some(alias_type) = &held.type_alias_type
        && obj.is_instance(alias_type.bind(py))?
    {
        return build_alias(obj, lits, defs).map(Some);
    }
    Ok(None)
}

/// Build the IR from a native Python schema description.
///
/// Recognized forms: the scalar types and `None`/`type(None)`; `object` as the
/// top schema and `Never`/`NoReturn` as the bottom; the list literal `[T]`
/// (homogeneous), `[A, B]` (fixed-length), and `[A, B, ...]` (prefix-plus-tail);
/// a single `{KeyType: ValueType}` entry as a mapping; an all-string-key dict as
/// a closed record (a trailing `"?"` marks an optional key); any other value as
/// an exact-value literal. Set literals (`{T}`) and tuple literals (`(A, B)`)
/// are rejected: they duplicate `set[T]`/`tuple[...]`, which typing spells.
pub(crate) fn build_schema(
    obj: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let mut guard = BuildGuard::enter()?;
    if !guard.walks {
        return read_annotation(obj, lits, defs);
    }
    let built = read_apart(obj, lits, defs);
    guard.walked = built.as_ref().ok().map(Schema::node_count);
    built
}

/// [`read_annotation`] in a frame of its own, for a build past as many descents
/// as the node bound, which walks each result.
///
/// The dispatch is inlined into [`build_schema`] for the descent every build
/// takes, and a second copy there is a second set of slots in every unoptimized
/// descent: with both in one frame, the 129-level chain of
/// `the_build_descends_one_level_past_the_construction_bound` needed 2,146 KiB
/// of stack on 3.11, past the 2 MiB a test thread is given, against 1,673 KiB
/// with one.
#[cold]
#[inline(never)]
fn read_apart(obj: &Bound<'_, PyAny>, lits: &mut Pool, defs: &mut Vec<Schema>) -> PyResult<Schema> {
    read_annotation(obj, lits, defs)
}

/// One descent of [`build_schema`] inside its guard: the dispatch on what `obj`
/// is.
///
/// Inlined at both of its call sites -- the one every descent takes, and
/// [`read_apart`] -- because a call to a function this size costs each
/// descent a frame of its own: the fifty-field record build of
/// `scripts/perf_gate.py --binding-build` read +2.44% against its base with the
/// call and +1.64% inlined.
#[expect(
    clippy::inline_always,
    reason = "measured: a call per descent read +2.44% on the record build, inlined +1.64%"
)]
#[inline(always)]
fn read_annotation(
    obj: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(Schema::NoneType);
    }
    let forms = forms(py)?;

    // `typing.Any` is a singleton special form: the gradual dynamic type. It is
    // checked before the type-object dispatch below because on 3.11+ `Any` is
    // itself a class and would otherwise be taken for an ordinary type.
    if obj.is(forms.any.bind(py)) {
        return Ok(Schema::ANY);
    }

    // `typing.Never`/`NoReturn` are the empty type: the lattice bottom, the
    // typing-native spelling of `nothing`. (`object` maps to the top below.)
    // `Never` is 3.11+, so it is absent (skipped) on older Pythons.
    for form in [&forms.never, &forms.noreturn].into_iter().flatten() {
        if obj.is(form.bind(py)) {
            return Ok(Schema::Nothing);
        }
    }

    // `Annotated` written bare annotates nothing. Below 3.13 it is a class, so
    // the dispatch below read it as an `isinstance` test no value passes; from
    // 3.13 it is a special form and was refused as one, in words about type
    // variables. Asked by identity, once, ahead of both.
    if obj.is(forms.annotated.bind(py)) {
        return Err(not_implemented(
            "typing.Annotated written bare annotates nothing: write \
             Annotated[T, metadata] for the type T with its metadata",
        ));
    }

    // A plain type or class (a scalar, `object`, TypedDict, dataclass, enum,
    // protocol, ...) is dispatched here, before the typing introspection below.
    // A type never has a typing origin, so taking this path first skips a
    // `get_origin` call per scalar and class node on the common compile path.
    if let Ok(ty) = obj.cast::<PyType>() {
        return build_type_object(ty, lits, defs);
    }

    // Two readings every arm below would reach, answered before any of them,
    // and after the type branch, because the fields of a record are types and
    // a type is answered above for a flag test.
    //
    // An exact `bool`, `int`, `float`, `str` or `bytes` is a constant: its type
    // carries no `__metadata__` or `__supertype__`, `get_origin` answers `None`
    // for it, and it is no container and no special form, so the fallthrough
    // interns it -- after an attribute read, a call into `typing` and a dozen
    // tests, which were most of compiling a wide `Literal`. A validator
    // composes in, and a validator has no subclass and matches no arm below
    // before its own.
    if let Some(constant) = builtin_constant(obj, lits) {
        return Ok(constant);
    }
    // Asked exactly, which a class with no subclass answers as `cast` would,
    // without the walk of the object's `__mro__` a subclass test takes.
    if let Ok(compiled) = obj.cast_exact::<Validator>() {
        let schema = compose(py, compiled.get(), lits, defs);
        hold_whole(&schema);
        return Ok(schema);
    }

    // Annotated[T, m1, ...]: the base type T with refinement metadata.
    //
    // Asked once rather than twice, and asked with names the interpreter
    // already holds. Spelled `hasattr` then `getattr`, this read the metadata
    // twice and decoded three attribute names from UTF-8 and hashed them --
    // per node, on a path every non-class form crosses, whether or not it is
    // annotated at all. Both names are asked optionally: an object carrying
    // `__metadata__` and no `__origin__` is not an `Annotated` form, and is
    // read as whatever else it is rather than failing on the name it lacks.
    if let Some(metadata) = obj.getattr_opt(intern!(py, "__metadata__"))?
        && let Some(base) = obj.getattr_opt(intern!(py, "__origin__"))?
    {
        return build_refine(&base, metadata.cast::<PyTuple>()?, lits, defs);
    }

    // Typing constructs (list[int], dict[K, V], tuple[...], X | Y, Literal,
    // ...) are read through the typing spec's own introspection, so the builtin
    // and legacy aliases share one path. A non-typing object has origin None and
    // falls through to the native handling below.
    let origin = forms.get_origin.bind(py).call1((obj,))?;
    if !origin.is_none() {
        let args = forms.get_args.bind(py).call1((obj,))?;
        let args = args.cast::<PyTuple>()?;
        // A *bare* legacy alias -- `typing.List`, `typing.Tuple` -- is the class
        // it aliases, so it is compiled as that class rather than as a
        // parametrization with no arguments.
        //
        // The two are told apart by whether the form carries a type argument
        // list at all, rather than by whether that list is empty. A bare alias
        // carries none; a parametrisation carries one even where it is empty,
        // and `typing.Tuple[()]` is exactly that -- the empty tuple, a different
        // type from every tuple. `get_args` answers `()` for both, so reading it
        // made `typing.Tuple[()]` admit `(1,)`.
        if !obj.hasattr(intern!(py, "__args__"))?
            && let Ok(class) = origin.cast::<PyType>()
        {
            return build_type_object(class, lits, defs);
        }
        return build_parametrized(obj, &origin, args, lits, defs);
    }

    // PEP 695 `type X = ...` alias (3.12+): validate the aliased type, tying
    // the fixpoint where the alias names itself.
    if let Some(alias_type) = &forms.type_alias_type
        && obj.is_instance(alias_type.bind(py))?
    {
        return build_alias(obj, lits, defs);
    }

    // NewType: validate the supertype it wraps.
    // Asked once, with a name the interpreter already holds -- the reading the
    // `__metadata__` arm above states the reason for. Spelled `hasattr` then
    // `getattr`, this decoded the name from UTF-8 and hashed it twice per node,
    // on the path every form that is not a class and has no origin crosses.
    //
    // The supertype is a type *argument*: a string there is a forward reference,
    // and read as the constant fallthrough reads one, `NewType("N", "int")` was
    // the literal `'int'`.
    if let Some(supertype) = obj.getattr_opt(intern!(py, "__supertype__"))? {
        return generics::build_type_argument(&supertype, lits, defs);
    }

    if let Ok(list) = obj.cast::<PyList>() {
        return build_sequence(list, lits, defs);
    }
    // A tuple literal duplicates `tuple[...]`, which typing spells, so it is not
    // a native form. (The list literal exists only because typing cannot spell a
    // fixed-length list: `[A, B]` is the fixed list, `tuple[A, B]` the tuple.)
    if obj.is_instance_of::<PyTuple>() {
        return Err(not_implemented(
            "a tuple literal is not a schema; write a fixed-length tuple as \
             tuple[A, B] (the list literal [A, B] is the fixed-length list)",
        ));
    }
    // A set literal duplicates `set[T]`, which typing spells, so it is not a
    // native form either.
    if obj.is_instance_of::<PySet>() {
        return Err(not_implemented(
            "a set literal is not a schema; write a set as set[T]",
        ));
    }
    // And the frozen one for the same reason. Asked separately because a
    // `frozenset` is not a `set`: the arm above does not see it, so it fell
    // through to the constant below and was interned. `frozenset({int})` --
    // which names a frozen set of integers -- became a schema admitting one
    // frozen set holding the `int` *type object*, and no value a caller has.
    if obj.is_instance_of::<PyFrozenSet>() {
        return Err(not_implemented(
            "a frozen set literal is not a schema; write a frozen set as \
             frozenset[T]",
        ));
    }
    // And the dict literal's frozen sibling. A `frozendict` is not a `dict`, so
    // the arm below does not see it, and past it the constant fallthrough would
    // intern `frozendict(a=int)` as a schema admitting that one mapping of a
    // type object. It names a record, which the dict literal spells.
    if let Some(frozendict) = &forms.frozendict
        && obj.is_instance(frozendict.bind(py))?
    {
        return Err(not_implemented(
            "a frozen dict literal is not a schema; write a record as a dict \
             literal {\"key\": T}, or a mapping as dict[K, V]",
        ));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return build_dict(dict, lits, defs);
    }

    build_unrecognised(obj, lits, defs)
}

/// The literal an exact `bool`, `int`, `float`, `str` or `bytes` is, or `None`
/// for any other object: a value [`build_schema`] can only read as the constant
/// it is. A subclass is not one -- an `IntEnum` member is an `int` -- and takes
/// the walk every other object takes.
fn builtin_constant(obj: &Bound<'_, PyAny>, lits: &mut Pool) -> Option<Schema> {
    is_builtin_scalar(obj).then(|| Schema::Literal(lits.intern_const(obj)))
}

/// Whether `obj` is an exact `bool`, `int`, `float`, `str` or `bytes`: a value
/// and never a typing form, whichever reading asks.
pub(crate) fn is_builtin_scalar(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_exact_instance_of::<PyInt>()
        || obj.is_exact_instance_of::<PyString>()
        || obj.is_exact_instance_of::<PyBool>()
        || obj.is_exact_instance_of::<PyFloat>()
        || obj.is_exact_instance_of::<PyBytes>()
}

/// An already-compiled validator, composed into the schema being built: its
/// pooled constants interned (so a constant shared by identity with one already
/// present collapses to a single index, which keeps structurally-equal schemas
/// equal across a merge), its definitions appended, and its schema's indices
/// remapped.
fn compose(py: Python<'_>, inner: &Validator, lits: &mut Pool, defs: &mut Vec<Schema>) -> Schema {
    // Keys a relation already read for this validator are read again from the
    // cache rather than from the interpreter; composing validators reads them
    // fresh and caches nothing.
    let lit_map: Vec<usize> = match inner.keys.get(py) {
        Some(keys) => inner
            .literals
            .iter()
            .zip(&keys.keys)
            .map(|(o, key)| lits.intern_keyed(o.bind(py), key.clone()))
            .collect(),
        None => inner
            .literals
            .iter()
            .map(|o| lits.intern(o.bind(py)))
            .collect(),
    };
    let offset = DefShift::new(place_definitions(&inner.definitions, &lit_map, defs));
    inner.schema.reindexed(&lit_map, offset)
}

/// An object [`build_schema`] has no other reading for: a literal of itself,
/// unless it is a typing form that a literal would misread.
fn build_unrecognised(
    obj: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    // A type variable or typing special form reaches the constant fallthrough
    // only because it has no typing origin and is not a value. Reject it with a
    // clear message rather than interning it as a literal that would match
    // almost nothing (a free `T` accepts only objects equal to the TypeVar).
    // A forward reference names a type rather than being one, and the constant
    // fallthrough would intern it as a literal of the `ForwardRef` object --
    // matching nothing a caller has. The same refusal a type *argument* already
    // gives, reached at the top level too.
    if is_forward_reference(obj)? {
        return Err(not_implemented(&format!(
            "{} is a forward reference, and a schema is built from the types \
             themselves: resolve the annotation first with typing.get_type_hints(\
             ..., include_extras=True), or write the type rather than its name",
            summarize(obj)?
        )));
    }

    if let Some(schema) = read_extension(obj, lits, defs)? {
        return Ok(schema);
    }

    if is_init_var(obj)? {
        return Err(not_implemented(&format!(
            "{} names a dataclass's constructor parameter, which is no field and \
             no value of one: write the type it carries",
            summarize(obj)?
        )));
    }

    if is_class_factory(obj)? {
        return Err(not_implemented(&format!(
            "{} is the base a class is declared from, not a type: pass the \
             TypedDict or named tuple class itself, or tuple[...] for the fields a \
             named tuple lays out",
            summarize(obj)?
        )));
    }

    if is_typing_construct(obj)? {
        return Err(not_implemented(&format!(
            "{} is a typing construct, not a value: a type variable, ParamSpec, \
             TypeVarTuple, or special form (such as Final or ClassVar) cannot be a \
             schema; use a concrete type",
            summarize(obj)?
        )));
    }

    // A callable that is no class, which the type branch answered: a function,
    // a bound method, a `partial`, an object defining `__call__`. The typing
    // spec's grammar for a type has no production for one, and the constant
    // reading gave it the set holding that one object, which no value a caller
    // checks is: `intersection(record, check)` was the record met with a
    // function, a set with no member, and nothing said so. What was meant is
    // one of two spellings, and each reads it: a predicate in `Annotated`
    // metadata, or the object itself in `Literal` ([`build_constant`]).
    if obj.is_callable() {
        return Err(not_implemented(&format!(
            "{} is callable, and a callable is not a schema: write \
             Annotated[T, fn] for the values of T it accepts, or Literal[fn] for \
             the object itself",
            summarize(obj)?
        )));
    }

    Ok(Schema::Literal(lits.intern_const(obj)))
}

/// A `Literal` argument: the one object it is.
///
/// Read as [`build_schema`] reads an object, except a callable, which the
/// constant fallthrough refuses where a schema is read and which a `Literal`
/// names as itself. It is interned here and counted as the leaf it is, without
/// a descent: a second caller of the guard's entry costs every descent of a
/// build its inlining, and the fifty-field record build of
/// `scripts/perf_gate.py --binding-build` read +1.77% with one, +0.00% without.
/// Two kinds of callable keep the reading the dispatch gives them, which is a
/// refusal of their own: `TypedDict` and `NamedTuple`, which are functions, and
/// a special form such as `ClassVar` -- `Annotated` written bare among them from
/// 3.13, and a class below it, which the `Literal` arm refuses as a type.
pub(super) fn build_constant(
    obj: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    if obj.is_callable() && !is_class_factory(obj)? && !is_typing_construct(obj)? {
        let constant = Schema::Literal(lits.intern_const(obj));
        hold_whole(&constant);
        return Ok(constant);
    }
    build_schema(obj, lits, defs)
}

/// True if `obj` is a `dataclasses.InitVar[...]`, which is an instance of the
/// class rather than a typing form, and so reached the constant fallthrough.
///
/// Looked up in `sys.modules` rather than imported: an instance exists only once
/// the module is loaded, and importing `dataclasses` from here is the import
/// that deadlocks against a lazy import on 3.15 (`docs/dev/08-testing.md`).
fn is_init_var(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    let Some(module) = loaded_modules(py)?.get_item(intern!(py, "dataclasses"))? else {
        return Ok(false);
    };
    match module.getattr_opt(intern!(py, "InitVar"))? {
        Some(class) => obj.is_instance(&class),
        None => Ok(false),
    }
}

/// True if `obj` is `TypedDict` or `NamedTuple` itself, from either module.
///
/// Each is a function or a form at runtime, so the literal fallback would read
/// it as the one object it is, which no value a caller has belongs to. A
/// checker reads `NamedTuple` as every named tuple class and refuses
/// `TypedDict` as a type; neither is a question `isinstance` can ask.
fn is_class_factory(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    let named = |factories: &[Py<PyAny>]| factories.iter().any(|factory| obj.is(factory.bind(py)));
    Ok(named(&forms(py)?.class_factories)
        || extensions(py)?.is_some_and(|held| named(&held.class_factories)))
}

/// True if `obj` is a `typing.ForwardRef`, on a runtime that has one.
fn is_forward_reference(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    match &forms(py)?.forward_ref {
        Some(class) => obj.is_instance(class.bind(py)),
        None => Ok(false),
    }
}

/// True if `obj` is a type variable or a typing special form (`Final`,
/// `ClassVar`, a bare `Optional`/`Union`/`Literal`, ...): a type-system
/// construct carrying no runtime value, so it cannot denote a set of values.
fn is_typing_construct(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    for class in &forms(py)?.type_variables {
        if obj.is_instance(class.bind(py))? {
            return Ok(true);
        }
    }
    match extensions(py)?.and_then(|held| held.special_form.as_ref()) {
        Some(class) => obj.is_instance(class.bind(py)),
        None => Ok(false),
    }
}

/// Compile arguments into one shared pool and combine them with `make`.
pub(crate) fn combine(
    args: &Bound<'_, PyTuple>,
    make: impl FnOnce(Vec<Schema>, &[Schema]) -> Schema,
) -> PyResult<Validator> {
    let mut literals = Pool::default();
    let mut definitions = Vec::new();
    let mut members = Vec::with_capacity(args.len());
    for arg in args.iter() {
        members.push(build_schema(&arg, &mut literals, &mut definitions)?);
    }
    // The definitions go to the fold as well as to the validator: a member that
    // is a reference is not a set on its own evidence, and the complement law
    // needs to read what it names.
    let schema = make(members, &definitions);
    Validator::checked(schema, literals.into_items(), definitions)
}

/// Where a merged validator's definitions live in the target list.
///
/// Appended, unless the same definitions are already there -- in which case the
/// offset they are already at is returned and nothing is added. That is what
/// makes a recursive schema equal to itself once combined: without it,
/// `intersection(json, complement(json))` copies one definition twice, the two
/// occurrences become `Ref(0)` and `Ref(1)`, and the fold that cancels a schema
/// against its own complement compares terms and sees two. Every law about a
/// recursive schema failed on that, including the one a reader tries first.
///
/// The whole block is matched rather than one definition at a time, because a
/// definition's body names its siblings by index: a block is self-consistent
/// only at the offset it was built for, so it is shifted to each candidate
/// offset and compared there. Quadratic in the definition count, which
/// `MAX_DEFINITIONS` holds at 128, and paid once per merged validator.
fn place_definitions(inner: &[Schema], lit_map: &[usize], defs: &mut Vec<Schema>) -> usize {
    if inner.is_empty() {
        return defs.len();
    }
    let at_offset = |start: usize| -> Vec<Schema> {
        inner
            .iter()
            .map(|d| d.reindexed(lit_map, DefShift::new(start)))
            .collect()
    };
    for start in 0..=defs.len().saturating_sub(inner.len()) {
        if defs.get(start..start + inner.len()) == Some(at_offset(start).as_slice()) {
            return start;
        }
    }
    let start = defs.len();
    defs.extend(at_offset(start));
    start
}

/// What two pooled objects have to agree on to be one constant.
///
/// A builtin scalar is keyed by its exact type and its value, which is the rule
/// a literal is read by everywhere else: `Literal[1]` and `Literal[True]` name
/// different singletons although `1 == True`, and two equal strings name one
/// however they were built. Anything else is keyed by address, because `==` on
/// it is the object's own and an object that answers it inconsistently would
/// merge two constants that are not one.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Constant {
    /// An object this cannot read by value: a class, a callable, a container, a
    /// scalar too wide for the key below.
    Address(usize),
    NoneType,
    Bool(bool),
    Int(i64),
    /// A float by its bits, with the one value whose bits are not its identity
    /// folded: `0.0 == -0.0`, so the two are one constant. A `nan` never reaches
    /// here -- it equals nothing, itself included, so it is keyed by address.
    Float(u64),
    /// Shared rather than owned, so a key a [`PoolKeys`] holds is handed to a
    /// second pool for a reference count rather than a copy.
    Str(Arc<str>),
    Bytes(Arc<[u8]>),
}

/// The key an object interns under.
fn constant_of(obj: &Bound<'_, PyAny>) -> Constant {
    let py = obj.py();
    let address = Constant::Address(obj.as_ptr() as usize);
    if obj.is_none() {
        return Constant::NoneType;
    }
    // Exact types only. A subclass carries its own `__eq__`, so two of its
    // instances comparing equal is not the literal rule holding of them.
    let ty = obj.get_type();
    let is = |exact: Bound<'_, PyType>| ty.is(&exact);
    if is(PyBool::type_object(py)) {
        return obj.extract::<bool>().map_or(address, Constant::Bool);
    }
    if is(PyInt::type_object(py)) {
        // An `int` wider than the key falls back to its address, which pools it
        // once per object rather than once per value -- correct, just coarser.
        return obj.extract::<i64>().map_or(address, Constant::Int);
    }
    if is(PyFloat::type_object(py)) {
        return match obj.extract::<f64>() {
            Ok(value) if !value.is_nan() => Constant::Float((value + 0.0).to_bits()),
            _ => address,
        };
    }
    if let Ok(text) = obj.cast_exact::<PyString>() {
        return text
            .to_str()
            .map_or(address, |text| Constant::Str(Arc::from(text)));
    }
    if let Ok(raw) = obj.cast_exact::<PyBytes>() {
        return Constant::Bytes(Arc::from(raw.as_bytes()));
    }
    address
}

/// A pool's keys, slot for slot, and the index a new constant is looked up in.
///
/// Built once per validator, by the first relation it is a side of, and shared
/// by every relation after it. Rebuilt per relation, it read each constant back
/// out of the interpreter and copied each string: 70% of relating two tables of
/// ten thousand codes, against the 20% the rules take to decide the pair.
pub(crate) struct PoolKeys {
    keys: Vec<Constant>,
    index: rustc_hash::FxHashMap<Constant, usize>,
}

impl PoolKeys {
    /// The keys of `items`, the later of two slots under one key winning it, as
    /// [`Pool::seeded`] has it.
    pub(crate) fn of(py: Python<'_>, items: &[Py<PyAny>]) -> Self {
        let keys: Vec<Constant> = items.iter().map(|obj| constant_of(obj.bind(py))).collect();
        let index = keys.iter().cloned().zip(0..).collect();
        PoolKeys { keys, index }
    }
}

/// The constants pool a compile builds, plus an index into it. Pooling
/// deduplicates by [`Constant`]; the index makes each `intern` a hash lookup
/// rather than a linear scan, so compiling a wide `Literal[...]` or merging many
/// validators stays linear in the number of constants instead of quadratic.
///
/// The address key is stable for the pooled objects: every interned value is
/// kept alive by `items`, so no live pool entry is freed and reallocated during a
/// compile.
#[derive(Default)]
pub(crate) struct Pool {
    items: Vec<Py<PyAny>>,
    index: rustc_hash::FxHashMap<Constant, usize>,
    /// The keys of the validator this pool was seeded from, shared rather than
    /// rebuilt. A constant is looked up here after `index`, and one new to both
    /// goes into `index`, so the two read as one index.
    seed: Option<Arc<PoolKeys>>,
}

impl Pool {
    /// Seed a pool with an existing validator's constants, rebuilding the index,
    /// so a second schema interns into the same pool when validators merge.
    ///
    /// A seeded pool can hold two slots under one key -- the constants were
    /// pooled before this rule, or by another pool -- and the later slot wins the
    /// key. Both stay in `items`, because a compiled schema already names the
    /// earlier one; what the key decides is only which slot a *new* occurrence
    /// joins.
    pub(crate) fn seeded(py: Python<'_>, items: Vec<Py<PyAny>>) -> Self {
        let index = items
            .iter()
            .enumerate()
            .map(|(at, obj)| (constant_of(obj.bind(py)), at))
            .collect();
        Pool {
            items,
            index,
            seed: None,
        }
    }

    /// Seed a pool with a validator's constants and the keys it already holds
    /// for them: [`seeded`](Self::seeded) without rebuilding the index.
    pub(crate) fn seeded_by(items: Vec<Py<PyAny>>, keys: Arc<PoolKeys>) -> Self {
        Pool {
            items,
            index: rustc_hash::FxHashMap::default(),
            seed: Some(keys),
        }
    }

    /// Pool `obj` and return its slot, deduplicating by [`Constant`].
    ///
    /// Private, and reached only through the four typed forms below. One pool
    /// serves four index spaces, so the slot acquires its meaning here, at the
    /// line that decides what the object is being pooled *as*. Two occurrences
    /// of one constant land in one slot whichever space they arrive through,
    /// which is what makes two spellings of a literal one schema node.
    fn intern(&mut self, obj: &Bound<'_, PyAny>) -> usize {
        self.intern_keyed(obj, constant_of(obj))
    }

    /// [`intern`](Self::intern), with the key already read.
    ///
    /// The seed is asked first. The two indices share no key, so the order
    /// answers nothing, and a relation pools mostly constants its subject
    /// already holds: asked second, each of those was hashed twice.
    fn intern_keyed(&mut self, obj: &Bound<'_, PyAny>, key: Constant) -> usize {
        let seeded = self.seed.as_ref().and_then(|seed| seed.index.get(&key));
        if let Some(&index) = seeded.or_else(|| self.index.get(&key)) {
            return index;
        }
        let index = self.items.len();
        self.items.push(obj.clone().unbind());
        self.index.insert(key, index);
        index
    }

    /// Make room for `additional` constants, so a wide `Literal` is one
    /// allocation of the index rather than a rehash per doubling.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.items.reserve(additional);
        self.index.reserve(additional);
    }

    /// Pool `obj` as the constant of a typed singleton.
    fn intern_const(&mut self, obj: &Bound<'_, PyAny>) -> ConstIx {
        ConstIx::new(self.intern(obj))
    }

    /// Pool `obj` as the class of an `isinstance` atom or an attribute record.
    fn intern_class(&mut self, obj: &Bound<'_, PyAny>) -> ClassIx {
        ClassIx::new(self.intern(obj))
    }

    /// Pool `obj` as a comparison or multiple-of operand.
    fn intern_operand(&mut self, obj: &Bound<'_, PyAny>) -> OperandIx {
        OperandIx::new(self.intern(obj))
    }

    /// Pool `obj` as a user predicate's callable.
    fn intern_predicate(&mut self, obj: &Bound<'_, PyAny>) -> PredIx {
        PredIx::new(self.intern(obj))
    }

    /// The pooled constants, for reading against a compiled schema.
    pub(crate) fn items(&self) -> &[Py<PyAny>] {
        &self.items
    }

    /// Consume the pool, yielding the constants a validator stores.
    pub(crate) fn into_items(self) -> Vec<Py<PyAny>> {
        self.items
    }
}

/// Refuse a map key schema narrowed by a constraint or a predicate.
///
/// A clause's key says which keys it governs, and a map reads that as whole
/// *kinds* of key -- `str`, `int`, and the rest -- or as the constants a
/// `Literal` names. A key narrowed by a bound, a pattern or a callback is
/// neither: it names part of a kind, and two such clauses can overlap without
/// either containing the other. Overlapping key domains are a different theory
/// from the one this library's maps are built on -- with them "there would not
/// be any difference between record types and an intersection of function types"
/// (Castagna, ICFP 2023, §4.5) -- and reading them as this library does, one
/// clause at a time, answers a question the two spellings do not agree on.
///
/// So it is refused where it is written, which is the rule the zero divisor and
/// the invalid pattern already follow.
fn checked_key(schema: &Schema, spelling: &Bound<'_, PyAny>) -> PyResult<()> {
    if !narrows_its_keys(schema) {
        return Ok(());
    }
    Err(not_implemented(&format!(
        "{} narrows the keys it governs, and a map key must be a key type or a \
         Literal: write dict[str, V] to key every string, or dict[Literal[\"a\"], V] \
         (or {{\"a\": V}}) to key one. To constrain the keys themselves, check them \
         beside the mapping rather than inside it",
        summarize(spelling)?
    )))
}

/// Whether a key schema narrows its keys by a constraint or a predicate, at any
/// depth a union or a complement can hide one.
fn narrows_its_keys(schema: &Schema) -> bool {
    match schema {
        Schema::Refine { constraints, .. } => !constraints.is_empty(),
        Schema::Union(members) | Schema::Intersection(members) => {
            members.iter().any(narrows_its_keys)
        }
        Schema::Complement(inner) => narrows_its_keys(inner),
        _ => false,
    }
}

pub(crate) fn not_implemented(message: &str) -> PyErr {
    PyNotImplementedError::new_err(message.to_owned())
}

// Needs a live interpreter; compiled and run only under the `interpreter-tests`
// feature, which links an embedded Python.
/// The frontend, driven against real annotations through a real interpreter.
///
/// Why this module exists rather than leaving the frontend to pytest: pytest
/// exercises this file thoroughly and a mutation harness cannot observe pytest,
/// so every mutant of this file read as a survivor and the file was excluded
/// from the sweep by name. An exclusion is a hole in the coverage claim, and
/// this one covered the widest file in the binding -- the one place a wrong arm
/// turns a written annotation into a schema that denotes something else,
/// silently, with every downstream test still passing.
///
/// The corpus is a table rather than a test per case, because the frontend is a
/// dispatch and the thing worth asserting is that each spelling lands on its own
/// arm. Each row is an annotation as a caller writes it and the schema it must
/// build, spelled as the render of that schema -- which is the same string
/// `repr(Validator(...))` prints, so a row can be read against the docs.
///
/// The refinement markers are built here rather than imported. An embedded
/// interpreter starts on the base prefix and does not see a virtual
/// environment's packages, so importing `annotated_types` would make this corpus
/// depend on how the harness was launched. It costs nothing to do without: the
/// frontend reads a marker by *attribute* -- `ge`, `min_length`, `pattern` --
/// which is the contract the vocabulary's classes meet, so an object carrying
/// the attribute exercises the same arm. The one arm that reads a marker's
/// identity is the refusal for a vocabulary member this frontend does not check,
/// and it reads `type(marker).__module__`, which a class here can set.
mod classes;
mod dialect;
mod generics;
mod refine;

#[cfg(all(test, feature = "interpreter-tests"))]
use classes::annotations_as_written;
use classes::build_type_object;
pub(crate) use classes::{build_instances, declares_fields};
use generics::{build_dict, build_parametrized, build_sequence};
use refine::build_refine;

#[cfg(all(test, feature = "interpreter-tests"))]
mod interpreter;

#[cfg(all(test, feature = "interpreter-tests"))]
mod tests;
