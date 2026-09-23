//! What a class declares: dispatch step 4, and the record the class's own
//! attributes describe.
//!
//! A `TypedDict` says so by carrying `__required_keys__`, an enum by subclassing
//! `Enum`, a dataclass by `dataclasses.is_dataclass`, a `Protocol` by
//! `_is_protocol`; every other class names its instances and is an `isinstance`
//! atom. The section "What a class declares" in `docs/dev/03-frontend.md` is
//! this module.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyNone, PySet, PyString, PyTuple,
    PyType,
};
use valgebra_core::{Field, MapClause, Schema, SeqShape};

use super::generics::is_field_qualifier;
use super::{MAX_BUILD_DEPTH, Pool, build_schema, forms, not_implemented};
use crate::errors::summarize;

/// `dataclasses.is_dataclass`, held apart from [`Forms`] and resolved on the
/// first class node that reaches the question.
///
/// Not in the cache beside the other forms, because that cache is built the
/// first time anything is compiled and `dataclasses` is a module most programs
/// never import: it pulls `inspect`, `copy`, `functools` and their own imports
/// in with it, and the objects they leave behind are *tracked* -- so every
/// later collection walks them. Measured on the shape that compiles a
/// fifty-field record of plain types, which reaches no dataclass and never asks
/// this question: importing it with the rest reads **6.45% dearer**, all of it
/// after the import, in the generational walks a build's own allocations
/// trigger.
static IS_DATACLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `dataclasses.fields`, held beside [`IS_DATACLASS`] and for its reason.
///
/// Reached only from [`declared_fields`], which asks it after `is_dataclass`
/// has answered yes -- so the module is already imported by the time this cell
/// is filled, and a program that compiles no dataclass still never imports it.
static DATACLASS_FIELDS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `dataclasses.is_dataclass`, imported on first use.
pub(super) fn is_dataclass(ty: &Bound<'_, PyType>) -> PyResult<bool> {
    let py = ty.py();
    IS_DATACLASS
        .get_or_try_init(py, || {
            Ok::<_, PyErr>(py.import("dataclasses")?.getattr("is_dataclass")?.unbind())
        })?
        .bind(py)
        .call1((ty,))?
        .is_truthy()
}

/// Build the schema for a Python type object (a builtin, `TypedDict`, `Enum`,
/// dataclass, `NamedTuple`, runtime-checkable `Protocol`, or `object`).
pub(super) fn build_type_object(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = ty.py();
    if ty.is(py.get_type::<PyBool>()) {
        return Ok(Schema::Bool);
    }
    if ty.is(py.get_type::<PyInt>()) {
        return Ok(Schema::Int);
    }
    if ty.is(py.get_type::<PyFloat>()) {
        return Ok(Schema::Float);
    }
    if ty.is(py.get_type::<PyString>()) {
        return Ok(Schema::Str);
    }
    if ty.is(py.get_type::<PyBytes>()) {
        return Ok(Schema::Bytes);
    }
    if ty.is(py.None().bind(py).get_type()) {
        return Ok(Schema::NoneType);
    }
    // A bare container class is its kind: `list` admits every list, which is the
    // set `list[object]` names and the set the typing spec assigns an
    // unparameterised generic. Read as an `isinstance` atom it was a different
    // sort of thing from the sequence node beside it -- in no kind at all -- and
    // neither spelling was decided below the other. See "What a bare builtin
    // class denotes" in `docs/dev/01-schema-ir.md`.
    if ty.is(py.get_type::<PyList>()) {
        return Ok(Schema::list(SeqShape::homogeneous(Schema::ANYTHING)));
    }
    if ty.is(py.get_type::<PyTuple>()) {
        return Ok(Schema::tuple(SeqShape::homogeneous(Schema::ANYTHING)));
    }
    if ty.is(py.get_type::<PySet>()) {
        return Ok(Schema::set(Schema::ANYTHING));
    }
    if ty.is(py.get_type::<PyFrozenSet>()) {
        return Ok(Schema::frozen_set(Schema::ANYTHING));
    }
    if ty.is(py.get_type::<PyDict>()) {
        return Ok(Schema::KeyedMap {
            fields: Vec::new().into(),
            defaults: vec![MapClause::top()].into(),
        });
    }
    let forms = forms(py)?;
    if ty.is(forms.object.bind(py)) {
        return Ok(Schema::ANYTHING);
    }
    // A bare typing special form is a class on some Pythons (notably `Union`).
    // It is not a value, so reject it rather than building an instance check that
    // accepts nothing; the special forms that are not types are rejected on the
    // value fallthrough in build_schema.
    for form in [&forms.union, &forms.optional] {
        if ty.is(form.bind(py)) {
            return Err(not_implemented(&format!(
                "{} is a typing special form, not a value; write a concrete type \
                 (for a union, X | Y or Union[X, Y])",
                summarize(ty.as_any())
            )));
        }
    }
    // TypedDict: a closed record whose required keys come from the class.
    if ty.hasattr(intern!(ty.py(), "__required_keys__"))? {
        return build_typed_dict(ty, lits, defs);
    }
    // Enum: an instance of the enumeration class (any of its members).
    if ty.is_subclass(forms.enum_class.bind(py))? {
        return Ok(Schema::Instance(lits.intern_class(ty.as_any())));
    }
    // dataclass / NamedTuple: isinstance plus a deep check of each field.
    if is_dataclass(ty)?
        || (ty.is_subclass_of::<PyTuple>()? && ty.hasattr(intern!(py, "_fields"))?)
    {
        return build_object(ty, lits, defs);
    }
    // Protocol: a runtime-checkable protocol validates by isinstance.
    if is_truthy_attr(ty, intern!(py, "_is_protocol")) {
        if is_truthy_attr(ty, intern!(py, "_is_runtime_protocol")) {
            return Ok(Schema::Instance(lits.intern_class(ty.as_any())));
        }
        return Err(not_implemented(
            "a Protocol must be @runtime_checkable to be used as a schema",
        ));
    }
    // Any other class names its instances: a bare class is an isinstance check.
    // This covers the remaining builtins (complex, bytearray, memoryview, range,
    // the `collections.abc` ABCs including Callable, ...) and arbitrary user
    // classes uniformly.
    Ok(Schema::Instance(lits.intern_class(ty.as_any())))
}

/// True if `obj.<name>` exists and is truthy; false on absence or error.
///
/// The name arrives interned and the attribute is asked for *optionally*, which
/// is what absence costs here: every one of these names is absent on an
/// ordinary class, and a bare `getattr` answers that by building an
/// `AttributeError`, raising it and dropping it -- per class node, for an answer
/// that is `false`.
pub(super) fn is_truthy_attr(obj: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> bool {
    obj.getattr_opt(name)
        .ok()
        .flatten()
        .and_then(|value| value.is_truthy().ok())
        .unwrap_or(false)
}

/// Resolve a class's type hints with `Annotated` metadata preserved.
///
/// `include_extras=True` keeps `Annotated[...]` field types intact so a field's
/// refinement markers reach [`build_refine`]; without it `get_type_hints` strips
/// the metadata and the field's constraints are silently lost.
///
/// **The annotations as written are the answer wherever evaluation would hand
/// each one back unchanged**, and [`annotations_as_written`] reads them without
/// the call. `get_type_hints` exists to evaluate forward references, and on a
/// class with none it still copies every base's namespace and walks every
/// annotation in Python: 60 to 70% of compiling a dataclass or a `TypedDict`.
/// Anything the reading is not certain of goes to `get_type_hints`, so an
/// answer or an error from this function is always the one it gives.
pub(super) fn resolve_type_hints<'py>(ty: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyAny>> {
    // A failure inside the reading is a decline and never an answer: the
    // exception a caller sees is the one `get_type_hints` raises.
    if let Ok(Some(hints)) = annotations_as_written(ty) {
        return Ok(hints.into_any());
    }
    let py = ty.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "include_extras"), true)?;
    forms(py)?
        .get_type_hints
        .bind(py)
        .call((ty,), Some(&kwargs))
}

/// The objects [`annotations_as_written`] reads a class and its annotations
/// through, resolved once per interpreter.
struct Evaluation {
    /// `(typing._GenericAlias, types.GenericAlias, types.UnionType)`: the
    /// classes whose `__args__` `typing._eval_type` descends into. Every other
    /// object it returns as it was given.
    aliases: Py<PyTuple>,
    /// `types.GenericAlias`, the one alias `_eval_type` rebuilds whatever its
    /// arguments are.
    generic_alias: Py<PyAny>,
    /// `types.GetSetDescriptorType`, which `get_type_hints` reads as a class
    /// with no annotations of its own, below 3.14.
    getset_descriptor: Py<PyAny>,
    /// `annotationlib.get_annotations`, how `get_type_hints` reads one class's
    /// own annotations from 3.14; `None` below it, where the class's namespace
    /// holds them.
    get_annotations: Option<Py<PyAny>>,
}

static EVALUATION: PyOnceLock<Evaluation> = PyOnceLock::new();

fn evaluation(py: Python<'_>) -> PyResult<&'static Evaluation> {
    EVALUATION.get_or_try_init(py, || {
        let types = py.import("types")?;
        let generic_alias = types.getattr("GenericAlias")?;
        let aliases = PyTuple::new(
            py,
            [
                py.import("typing")?.getattr("_GenericAlias")?,
                generic_alias.clone(),
                types.getattr("UnionType")?,
            ],
        )?;
        let get_annotations = if py.version_info() >= (3, 14) {
            Some(
                py.import("annotationlib")?
                    .getattr("get_annotations")?
                    .unbind(),
            )
        } else {
            None
        };
        Ok(Evaluation {
            aliases: aliases.unbind(),
            generic_alias: generic_alias.unbind(),
            getset_descriptor: types.getattr("GetSetDescriptorType")?.unbind(),
            get_annotations,
        })
    })
}

/// How deep an annotation is walked before the walk declines. `_eval_type`
/// recurses without a bound of its own and meets the interpreter's recursion
/// limit; a walk on the Rust stack stops well before its own.
const MAX_ANNOTATION_DEPTH: usize = 64;

/// A class's type hints read as `typing.get_type_hints(ty, include_extras=True)`
/// returns them, or `None` where that call would change a value.
///
/// The reading is the call's own, step for step. Each base in reversed
/// `__mro__` contributes its own annotations -- from its namespace's
/// `__annotations__` below 3.14, from `annotationlib.get_annotations` from 3.14
/// -- a later name replacing an earlier one in place, and `None` read as
/// `type(None)`. What the call adds is `_eval_type` over each value, and that
/// returns the value unchanged unless [`evaluates`] says otherwise. A class
/// marked `__no_type_check__` is left to the call, which answers `{}` for it.
///
/// **Equal to the call's answer, and the objects themselves.** Two values
/// differ only where `_eval_type` rebuilds a builtin alias, and every such
/// alias is declined. `annotations_as_written_are_the_hints_get_type_hints_returns`
/// in `build/tests.rs` holds the reading to the call, value for value and in
/// order, and `tests/test_classes.py` holds the compiled validators to it on
/// every interpreter the matrix runs.
pub(super) fn annotations_as_written<'py>(
    ty: &Bound<'py, PyType>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let py = ty.py();
    let names = evaluation(py)?;
    if let Some(flag) = ty.getattr_opt(intern!(py, "__no_type_check__"))?
        && flag.is_truthy()?
    {
        return Ok(None);
    }
    let hints = PyDict::new(py);
    let mro = ty.getattr(intern!(py, "__mro__"))?;
    for base in mro.cast::<PyTuple>()?.iter().rev() {
        let own = if let Some(get_annotations) = &names.get_annotations {
            get_annotations.bind(py).call1((&base,))?
        } else {
            // `base.__dict__.get('__annotations__', {})`, the call's own
            // spelling, so a namespace a metaclass supplies is read the way the
            // call reads it.
            let own = base.getattr(intern!(py, "__dict__"))?.call_method1(
                intern!(py, "get"),
                (intern!(py, "__annotations__"), PyDict::new(py)),
            )?;
            if own.is_instance(names.getset_descriptor.bind(py))? {
                continue;
            }
            own
        };
        // `get_type_hints` reads any mapping through `.items()`; a dict is the
        // one this reading takes, and anything else is left to the call.
        let Ok(own) = own.cast::<PyDict>() else {
            return Ok(None);
        };
        for (name, value) in own.iter() {
            if value.is_instance_of::<PyString>() || evaluates(&value, names, 0)? {
                return Ok(None);
            }
            if value.is_none() {
                hints.set_item(name, py.get_type::<PyNone>())?;
            } else {
                hints.set_item(name, value)?;
            }
        }
    }
    Ok(Some(hints))
}

/// Whether `typing._eval_type` would hand back anything other than `value`.
///
/// It resolves a forward reference, and descends only into the `__args__` of
/// the three alias classes. A builtin `types.GenericAlias` it rebuilds as
/// `origin[args]` from 3.11: a string argument becomes a forward reference, an
/// unpacked alias becomes `Unpack[...]`, and a `collections.abc.Callable`'s
/// argument list is re-nested. Each of those is a change, so each is `true`. A
/// string under any other alias -- a `Literal`'s value -- is returned as it is.
fn evaluates(value: &Bound<'_, PyAny>, names: &Evaluation, depth: usize) -> PyResult<bool> {
    let py = value.py();
    if depth > MAX_ANNOTATION_DEPTH {
        return Ok(true);
    }
    if let Some(forward_ref) = &forms(py)?.forward_ref
        && value.is_instance(forward_ref.bind(py))?
    {
        return Ok(true);
    }
    if !value.is_instance(names.aliases.bind(py))? {
        return Ok(false);
    }
    let args = value.getattr(intern!(py, "__args__"))?;
    let args = args.cast::<PyTuple>()?;
    if value.is_instance(names.generic_alias.bind(py))?
        && (is_truthy_attr(value, intern!(py, "__unpacked__"))
            || value
                .getattr(intern!(py, "__origin__"))?
                .is(forms(py)?.callable.bind(py))
            || args.iter().any(|arg| arg.is_instance_of::<PyString>()))
    {
        return Ok(true);
    }
    for arg in args.iter() {
        if evaluates(&arg, names, depth + 1)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Read a record field name as Rust text, refusing a key that is not valid
/// Unicode (one carrying a lone surrogate). A field name is stored as UTF-8 and
/// matched against dict keys as UTF-8, so a surrogate key cannot round-trip;
/// refusing it at build time turns silent corruption — a lossy replacement that
/// makes the field unmatchable — into an explicit error.
///
/// The name is handed back borrowed from the Python string rather than copied
/// into one of its own. Every caller turns it into the shared name a field
/// holds, and that conversion copies the text anyway, so owning it here would
/// buy an allocation per field and free it one line later.
pub(super) fn field_name<'a>(name: &'a Bound<'_, PyString>) -> PyResult<&'a str> {
    name.to_str().map_err(|_| {
        PyValueError::new_err(
            "a record key must be valid Unicode; a field name cannot contain a lone surrogate",
        )
    })
}

/// Build the record a `TypedDict` denotes: its keys, and what it says about the
/// ones it does not name.
///
/// **Open unless the class says otherwise**, which is the set the typing spec
/// assigns it -- "By default, `TypedDict`s are open". `Validator(TD)` reads an
/// annotation whose meaning is fixed elsewhere, and reading it as a narrower set
/// is a deviation a caller has no way to see, because the class carries no mark
/// of it. The dict-literal form `{"a": int}` stays closed: it is this library's
/// own spelling, and a schema written as a *shape* means that shape
/// (`docs/dev/01-schema-ir.md`, "What a `TypedDict` denotes").
///
/// `ReadOnly` needs no arm here. It constrains writers and a value has none, and
/// `get_type_hints` has already resolved it away.
pub(super) fn build_typed_dict(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let hints = resolve_type_hints(ty)?;
    let hints = hints.cast::<PyDict>()?;
    let required = ty.getattr(intern!(ty.py(), "__required_keys__"))?;
    let mut fields = Vec::with_capacity(hints.len());
    for (name, hint) in hints.iter() {
        fields.push(Field {
            name: field_name(&name.str()?)?.into(),
            schema: build_schema(&hint, lits, defs)?,
            // The qualifier on the *resolved* hint wins over the class's key
            // sets. CPython fills `__required_keys__` when the class is created,
            // from the annotations as written -- and under `from __future__
            // import annotations` those are strings, so `NotRequired[...]` is
            // invisible to it and every optional key was compiled required.
            // `get_type_hints` resolves the string and keeps the qualifier, so
            // reading it there is reading what the author wrote.
            required: match qualified_required(&hint)? {
                Some(stated) => stated,
                None => required.contains(&name)?,
            },
        });
    }
    Ok(Schema::keyed_map(fields, unnamed_keys(ty, lits, defs)?))
}

/// Whether a resolved hint states its own required-ness, and which.
///
/// `Required[T]` and `NotRequired[T]` say it; every other form, `ReadOnly[T]`
/// included, says nothing and leaves the class's key sets to answer. A qualifier
/// may wrap another -- `ReadOnly[NotRequired[T]]` is legal -- so the search goes
/// through the ones that carry no answer rather than stopping at the first.
pub(super) fn qualified_required(hint: &Bound<'_, PyAny>) -> PyResult<Option<bool>> {
    let py = hint.py();
    let forms = forms(py)?;
    let mut current = hint.clone();
    for _ in 0..MAX_BUILD_DEPTH {
        // Asked rather than tried: a field that carries no qualifier is the
        // common one, and an attribute that is absent answers by *raising* --
        // an exception built, thrown and dropped per field. `getattr_opt`
        // reads the same absence without one.
        let Some(origin) = current.getattr_opt(intern!(py, "__origin__"))? else {
            return Ok(None);
        };
        for (marker, answer) in [(&forms.required, true), (&forms.not_required, false)] {
            if let Some(marker) = marker
                && origin.is(marker.bind(py))
            {
                return Ok(Some(answer));
            }
        }
        if !is_field_qualifier(&origin)? {
            return Ok(None);
        }
        let Some(args) = current.getattr_opt(intern!(py, "__args__"))? else {
            return Ok(None);
        };
        let Ok(inner) = args.get_item(0) else {
            return Ok(None);
        };
        current = inner;
    }
    Ok(None)
}

/// Whether `extra_items` carries the sentinel for "the author gave none".
///
/// A runtime with PEP 728 fills `__extra_items__` in either way: with the type
/// its author wrote, or with `NoExtraItems` to say there was none. The sentinel
/// is not a type and reading it as one makes the record admit exactly the
/// sentinel -- which is a closed record wearing an open one's spelling, and is
/// what 3.15 turned this into before the check was here.
pub(super) fn gave_no_extra_items(extra: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Some(sentinel) = forms(extra.py())?.no_extra_items.as_ref() else {
        return Ok(false);
    };
    Ok(extra.is(sentinel.bind(extra.py())))
}

/// What a `TypedDict` says about the keys it does not name.
///
/// `closed=True` shuts them and `extra_items=T` gives them a type -- PEP 728,
/// which the typing spec carries. Neither marker is in `typing` yet, so both are
/// read where a runtime that has them puts them and are absent otherwise:
/// a `TypedDict` written for a runtime without PEP 728 cannot have said either,
/// and the spec's default is what is left.
pub(super) fn unnamed_keys(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Vec<MapClause>> {
    let py = ty.py();
    if let Some(flag) = ty.getattr_opt(intern!(py, "__closed__"))?
        && flag.is_truthy()?
    {
        return Ok(Vec::new());
    }
    if let Some(extra) = ty.getattr_opt(intern!(py, "__extra_items__"))?
        && !extra.is_none()
        && !gave_no_extra_items(&extra)?
    {
        // A `TypedDict`'s keys are strings, so the type it gives the extra ones
        // governs the string keys and leaves no other kind admitted.
        return Ok(vec![MapClause {
            key: Schema::Str,
            value: build_schema(&extra, lits, defs)?,
        }]);
    }
    // A `TypedDict`'s keys are strings -- the spec relates one to
    // `Mapping[str, object]` and to nothing wider -- so being open is being open
    // to further *string* keys, not to keys of every kind.
    Ok(vec![MapClause {
        key: Schema::Str,
        value: Schema::ANYTHING,
    }])
}

/// The attribute names a class declares, in declaration order.
///
/// Not every annotation on a dataclass names an attribute of its instances:
/// `ClassVar` annotates the class, and `InitVar` names a constructor parameter
/// the instance does not keep. Reading the hints alone asks an instance for
/// attributes it cannot have, which is a schema no instance of the class
/// satisfies — the empty set wearing the class's name. Each kind of class keeps
/// its own list of what it declares, so that list is what is read: a dataclass's
/// `fields()` and a named tuple's `_fields`. A field declared `init=False` is on
/// the instance and stays.
pub(super) fn declared_fields<'py>(ty: &Bound<'py, PyType>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let py = ty.py();
    // Both questions go through the held handles. Spelled as a module import
    // and two `call_method1`s, this asked `sys.modules` for `dataclasses` and
    // decoded three names per class node -- and asked `is_dataclass` a second
    // time, the caller having just had its answer.
    if is_dataclass(ty)? {
        return DATACLASS_FIELDS
            .get_or_try_init(py, || {
                Ok::<_, PyErr>(py.import("dataclasses")?.getattr("fields")?.unbind())
            })?
            .bind(py)
            .call1((ty,))?
            .try_iter()?
            .map(|field| field?.getattr(intern!(py, "name")))
            .collect();
    }
    ty.getattr(intern!(py, "_fields"))?.try_iter()?.collect()
}

/// Build the schema of a class with declared attributes: the meet of its
/// `isinstance` atom and a record of its fields, whose types come from the
/// resolved hints. Every attribute an instance declares is required, because an
/// instance carries it.
///
/// A class with no annotated field is the atom alone. The meet is what the form
/// means, and spelling it out is what lets each half relate on its own; the
/// surface does not change, because `render` reads the pair back as the class
/// name and the walk stops at the failing conjunct, which is what
/// `docs/dev/01-schema-ir.md` records under "What a class with attributes is, on
/// the surface".
pub(super) fn build_object(
    ty: &Bound<'_, PyType>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let hints = resolve_type_hints(ty)?;
    let hints = hints.cast::<PyDict>()?;
    let class_index = lits.intern_class(ty.as_any());
    let declared = declared_fields(ty)?;
    let mut fields = Vec::with_capacity(declared.len());
    for name in declared {
        // A name the hints do not carry is unannotated -- a `collections`
        // namedtuple's fields are the case -- so there is no type to check and
        // the class's own isinstance test is the whole of it.
        let Some(hint) = hints.get_item(&name)? else {
            continue;
        };
        fields.push(Field {
            name: field_name(&name.str()?)?.into(),
            schema: build_schema(&hint, lits, defs)?,
            required: true,
        });
    }
    let instance = Schema::Instance(class_index);
    let mut parts = vec![instance];
    let positions = named_tuple_positions(ty, hints, lits, defs)?;
    if positions.is_none() && !fields.is_empty() {
        parts.push(Schema::attr_record(fields));
    }
    if let Some(positions) = positions {
        parts.push(positions);
    }
    if parts.len() == 1 {
        return Ok(parts.remove(0));
    }
    Ok(Schema::meet(parts))
}

/// The tuple shape a named tuple's fields lay out, or `None` for a class that
/// lays out none.
///
/// A named tuple's positions *are* its attributes: an instance is a tuple of
/// exactly as many elements as the class declares, the i-th holding the i-th
/// field. The class says so and the frontend reads it, so the schema says it
/// too -- otherwise the set the schema denotes is wider than the set the class
/// has, admitting an instance whose attributes are integers and whose positions
/// are anything, which no instance is.
///
/// It is the shape *instead of* an attribute record rather than beside one,
/// because the two would say the same thing about the same values: a wrong
/// field would be reported twice, and every passing value would be read twice.
/// The shape is the more precise of the pair -- it carries the arity as well as
/// the types -- and every rule that reads a shape reads it, so a relation
/// between a named tuple and the tuple it lays out is decided structurally
/// rather than declined. What the attribute record carried and this does not is
/// the field's *name* in a failure's path, which becomes its position.
///
/// A field the annotations do not carry -- a `collections.namedtuple` has none
/// -- takes `anything` at its position, which is the arity without the types.
pub(super) fn named_tuple_positions(
    ty: &Bound<'_, PyType>,
    hints: &Bound<'_, PyDict>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Option<Schema>> {
    if !(ty.is_subclass_of::<PyTuple>()? && ty.hasattr(intern!(ty.py(), "_fields"))?) {
        return Ok(None);
    }
    let mut positions = Vec::new();
    for name in declared_fields(ty)? {
        positions.push(match hints.get_item(&name)? {
            Some(hint) => build_schema(&hint, lits, defs)?,
            None => Schema::ANYTHING,
        });
    }
    Ok(Some(Schema::tuple(SeqShape::fixed(positions))))
}
