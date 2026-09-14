//! What a parametrized form says: dispatch step 6, the typing spec's
//! introspection, and the two native literals that spell what it cannot.
//!
//! The origin is read before the arguments and compared by identity against the
//! forms resolved once at import, so a form this frontend does not know is a
//! refusal rather than a guess. A list literal and a dict literal answer the
//! same question from the other side -- `[A, B]` is the fixed-length list
//! `typing` has no spelling for -- so they are read here. The section "What a
//! parametrized form says" in `docs/dev/03-frontend.md` is this module.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple};
use valgebra_core::{Field, MapClause, Schema, SeqShape};

use super::classes::{field_name, is_truthy_attr};
use super::{Pool, build_schema, checked_key, forms, is_forward_reference, not_implemented};
use crate::errors::summarize;

/// Build the IR for a parametrized typing generic given its origin and args.
pub(super) fn build_parametrized(
    origin: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let py = origin.py();
    if origin.is(py.get_type::<PyList>()) {
        return Ok(Schema::list(SeqShape::homogeneous(build_type_argument(
            &single_arg(args)?,
            lits,
            defs,
        )?)));
    }
    if origin.is(py.get_type::<PySet>()) {
        return Ok(Schema::set(build_type_argument(
            &single_arg(args)?,
            lits,
            defs,
        )?));
    }
    if origin.is(py.get_type::<PyFrozenSet>()) {
        return Ok(Schema::frozen_set(build_type_argument(
            &single_arg(args)?,
            lits,
            defs,
        )?));
    }
    if origin.is(py.get_type::<PyDict>()) {
        if args.len() != 2 {
            return Err(not_implemented(
                "dict[...] needs a key type and a value type",
            ));
        }
        // Named rather than positional: `dict[K, V]` compiled with the two
        // transposed is `dict[V, K]`, which typechecks and validates real values.
        let key_argument = args.get_item(0)?;
        let key = build_type_argument(&key_argument, lits, defs)?;
        checked_key(&key, &key_argument)?;
        return Ok(Schema::mapping(MapClause {
            key,
            value: build_type_argument(&args.get_item(1)?, lits, defs)?,
        }));
    }
    if origin.is(py.get_type::<PyTuple>()) {
        return build_tuple(args, lits, defs);
    }
    if is_field_qualifier(origin)? {
        // A field qualifier survives hint resolution because field metadata is
        // kept (include_extras), so the frontend unwraps it and compiles the type
        // it qualifies.
        return build_type_argument(&single_arg(args)?, lits, defs);
    }
    if is_union_origin(origin)? {
        let mut members = Vec::with_capacity(args.len());
        for arg in args.iter() {
            members.push(build_type_argument(&arg, lits, defs)?);
        }
        return Ok(Schema::union(members));
    }
    if is_literal_origin(origin)? {
        // Literal args are constant values; each becomes a literal, unioned when
        // there is more than one.
        let mut members = Vec::with_capacity(args.len());
        for arg in args.iter() {
            refuse_unhashable_literal(&arg)?;
            members.push(build_schema(&arg, lits, defs)?);
        }
        return Ok(Schema::union(members));
    }
    if origin.is(forms(py)?.callable.bind(py)) {
        // Callable[...] checks only callability at runtime; the argument and
        // return types cannot be inspected, so the parameters are ignored and
        // the schema is the opaque `isinstance(x, Callable)` test.
        return Ok(Schema::Instance(lits.intern_class(origin)));
    }
    Err(not_implemented(&format!(
        "unsupported typing form with origin {}; supported: list, set, dict, \
         tuple, Union, Optional, Literal, Callable",
        summarize(origin)
    )))
}

/// Refuse a `Literal` argument the typing spec does not allow, where reading it
/// on would mean something else entirely.
///
/// The spec's `Literal` takes `None`, an enum member, or an `int`, `bool`, `str`
/// or `bytes` value. A **list, dict or set** is none of those, and Python does
/// not reject the subscription -- so `Literal[[1]]` arrived here as a list, and
/// the constant fallthrough read it as this library's own native list *schema*:
/// `Literal[[1]]` became `list[Literal[1]]`, and `Literal[{}]` the empty record.
/// Those are sets a caller who wrote `Literal` did not ask for, and no message
/// said so.
///
/// A float is deliberately not refused here. It is not a spelling the spec
/// allows either, but it is a *constant*, and this library pools it as one --
/// which is what `docs/05-refinements.md` says and what a caller writing
/// `Literal[1.5]` means. The line is between a value with no interpretation but
/// itself and one this library already reads as a schema.
pub(super) fn refuse_unhashable_literal(arg: &Bound<'_, PyAny>) -> PyResult<()> {
    let (kind, instead) = if arg.is_instance_of::<PyList>() {
        (
            "a list",
            "list[T] for a list of values, or a tuple of them in the Literal",
        )
    } else if arg.is_instance_of::<PyDict>() {
        (
            "a dict",
            "dict[K, V], or a record written as a dict literal",
        )
    } else if arg.is_instance_of::<PySet>() {
        ("a set", "set[T]")
    } else {
        return Ok(());
    };
    Err(not_implemented(&format!(
        "{kind} is not a Literal argument: the typing spec allows None, an enum \
         member, or an int, bool, str or bytes value, and this one would be read \
         as a schema of its own rather than as a constant. Write {instead}"
    )))
}

/// True if `origin` is `typing.Union` (from Union/Optional) or
/// `types.UnionType` (from `X | Y`).
pub(super) fn is_union_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = origin.py();
    let forms = forms(py)?;
    Ok(origin.is(forms.union.bind(py)) || origin.is(forms.union_type.bind(py)))
}

/// True if `origin` is `typing.Literal`.
pub(super) fn is_literal_origin(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = origin.py();
    Ok(origin.is(forms(py)?.literal.bind(py)))
}

/// True if `origin` is one of the `TypedDict` field qualifiers, which
/// `include_extras` keeps in the resolved hints.
///
/// `Required`/`NotRequired` say whether the key must be present, which
/// [`qualified_required`] reads from here, and `ReadOnly` says whether a
/// consumer may write the key back — a statement about use, not about which
/// values belong. None of the three narrows the field's *set*, so each is
/// unwrapped to the type it qualifies.
pub(super) fn is_field_qualifier(origin: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = origin.py();
    let forms = forms(py)?;
    for marker in [&forms.required, &forms.not_required, &forms.read_only] {
        if let Some(marker) = marker
            && origin.is(marker.bind(py))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Compile a *type argument* of a typing form.
///
/// A string here is a forward reference, which the typing spec asks a consumer to
/// resolve against the namespace the annotation was written in. valgebra has no
/// such namespace in hand, and the constant fallthrough would read `list["int"]`
/// as a list of the string `"int"` — a schema that refuses what the annotation
/// admits and admits what it refuses. So the position is refused, and the caller
/// is pointed at the resolution the spec provides.
///
/// `Literal["a"]`'s arguments are not this position: they are values, which is
/// why the literal arm compiles them straight through. Nor is the native list or
/// dict literal, where a bare value is a literal by design.
pub(super) fn build_type_argument(
    arg: &Bound<'_, PyAny>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    if arg.is_instance_of::<PyString>() || is_forward_reference(arg)? {
        return Err(not_implemented(&format!(
            "{} is a forward reference, and a schema is built from the types \
             themselves: resolve the annotation first with typing.get_type_hints(\
             ..., include_extras=True), or write the type rather than its name",
            summarize(arg)
        )));
    }
    build_schema(arg, lits, defs)
}

/// What an unpacked tuple argument contributes to the tuple that carries it.
pub(super) enum Unpacked<'py> {
    /// `*tuple[A, B]`: its elements splice in where it stands.
    Fixed(Vec<Bound<'py, PyAny>>),
    /// `*tuple[B, ...]`: the tuple repeats `B` from here on.
    Tail(Bound<'py, PyAny>),
}

/// Read `arg` as an unpacked tuple, or `None` when it is an ordinary element.
///
/// Two spellings say the same thing: `*tuple[B, ...]` is the tuple alias itself
/// carrying a flag, and `Unpack[tuple[B, ...]]` wraps it. Both have to be read,
/// because a reader who writes one and a reader who writes the other mean one
/// annotation, and reading only the second leaves the first looking like a
/// nested tuple — a schema that admits a different set than the annotation says.
///
/// Only a tuple can be unpacked into a tuple. `*Ts` over a `TypeVarTuple` binds
/// no element types at runtime, so it is refused here rather than read as an
/// empty splice.
pub(super) fn unpacked_tuple<'py>(arg: &Bound<'py, PyAny>) -> PyResult<Option<Unpacked<'py>>> {
    let py = arg.py();
    let forms = forms(py)?;
    let origin_of = |of: &Bound<'py, PyAny>| forms.get_origin.bind(py).call1((of,));
    let inner = if is_truthy_attr(arg, "__unpacked__") {
        arg.clone()
    } else {
        let Some(unpack) = &forms.unpack else {
            return Ok(None);
        };
        if !origin_of(arg)?.is(unpack.bind(py)) {
            return Ok(None);
        }
        let wrapped = forms.get_args.bind(py).call1((arg,))?;
        single_arg(wrapped.cast::<PyTuple>()?)?
    };
    if !origin_of(&inner)?.is(py.get_type::<PyTuple>()) {
        return Err(not_implemented(&format!(
            "only a tuple can be unpacked into a tuple schema; {} binds no \
             element types at runtime",
            summarize(&inner)
        )));
    }
    let args = forms.get_args.bind(py).call1((&inner,))?;
    let args = args.cast::<PyTuple>()?;
    let len = args.len();
    if len == 2 && is_ellipsis(&args.get_item(1)?) {
        return Ok(Some(Unpacked::Tail(args.get_item(0)?)));
    }
    Ok(Some(Unpacked::Fixed(args.iter().collect())))
}

/// `tuple[...]`, in every shape typing spells it.
///
/// A trailing `...` repeats the element before it after a fixed prefix, and an
/// unpacked variadic tuple — `tuple[A, *tuple[B, ...]]` — is that same shape said
/// another way, so both compile to the prefix-and-tail form. An unpacked *fixed*
/// tuple splices its elements in where it stands.
///
/// What a sequence carries is a fixed prefix and then a repeating tail, with
/// nothing after it. An element following the tail is therefore refused: the set
/// it names is one this algebra cannot spell, and reading it as anything else
/// would admit a different one.
pub(super) fn build_tuple(
    args: &Bound<'_, PyTuple>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let mut prefix: Vec<Bound<'_, PyAny>> = Vec::with_capacity(args.len());
    let mut tail: Option<Bound<'_, PyAny>> = None;
    for arg in args.iter() {
        if tail.is_some() {
            return Err(not_implemented(
                "a tuple schema carries a fixed prefix and then a repeating \
                 tail, so nothing may follow the tail: write the repeating part \
                 last",
            ));
        }
        match unpacked_tuple(&arg)? {
            Some(Unpacked::Fixed(items)) => prefix.extend(items),
            Some(Unpacked::Tail(item)) => tail = Some(item),
            None if is_ellipsis(&arg) => {
                let Some(repeated) = prefix.pop() else {
                    return Err(not_implemented(
                        "`...` repeats the element before it, so a tuple schema \
                         cannot begin with one",
                    ));
                };
                tail = Some(repeated);
            }
            None => prefix.push(arg),
        }
    }
    let mut elements = Vec::with_capacity(prefix.len());
    for element in &prefix {
        elements.push(build_type_argument(element, lits, defs)?);
    }
    let regex = match tail {
        None => SeqShape::fixed(elements),
        Some(tail) => {
            let tail = build_type_argument(&tail, lits, defs)?;
            if elements.is_empty() {
                SeqShape::homogeneous(tail)
            } else {
                SeqShape::prefix_tail(elements, tail)
            }
        }
    };
    Ok(Schema::tuple(regex))
}

pub(super) fn single_arg<'py>(args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyAny>> {
    if args.len() == 1 {
        args.get_item(0)
    } else {
        Err(not_implemented("expected exactly one type argument"))
    }
}

pub(super) fn build_sequence(
    list: &Bound<'_, PyList>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    let len = list.len();
    // [T]: a homogeneous list of T (the single-element idiom).
    if len == 1 && !is_ellipsis(&list.get_item(0)?) {
        let element = build_schema(&list.get_item(0)?, lits, defs)?;
        return Ok(Schema::list(SeqShape::homogeneous(element)));
    }
    // [p0, ..., tail, ...]: a trailing `...` repeats the element before it, after
    // a fixed prefix of the earlier elements. [T, ...] is the prefix-free case,
    // and [T, T, ...] is the non-empty list.
    if len >= 2 && is_ellipsis(&list.get_item(len - 1)?) {
        let mut elements = Vec::with_capacity(len - 1);
        for index in 0..len - 1 {
            let item = list.get_item(index)?;
            if is_ellipsis(&item) {
                return Err(not_implemented(
                    "`...` may appear only as the last element of a list schema",
                ));
            }
            elements.push(build_schema(&item, lits, defs)?);
        }
        let tail = elements.pop().expect("at least one element precedes `...`");
        let regex = if elements.is_empty() {
            SeqShape::homogeneous(tail)
        } else {
            SeqShape::prefix_tail(elements, tail)
        };
        return Ok(Schema::list(regex));
    }
    // [A, B]: a fixed-length list matched positionally (and `[]` the empty list).
    // typing cannot spell a fixed-length list (`list[A, B]` is illegal), so the
    // list literal carries this shape; `tuple[A, B]` is the tuple counterpart.
    let mut elements = Vec::with_capacity(len);
    for index in 0..len {
        let item = list.get_item(index)?;
        if is_ellipsis(&item) {
            return Err(not_implemented(
                "`...` may appear only as the last element of a list schema",
            ));
        }
        elements.push(build_schema(&item, lits, defs)?);
    }
    Ok(Schema::list(SeqShape::fixed(elements)))
}

pub(super) fn build_dict(
    dict: &Bound<'_, PyDict>,
    lits: &mut Pool,
    defs: &mut Vec<Schema>,
) -> PyResult<Schema> {
    // A string key is a named field (with the `"key?"` optional convention); any
    // other key is a schema keying a default clause for the rest. All string keys
    // give a record; a single schema key with no fields gives `dict[K, V]`;
    // several schema keys a heterogeneous mapping; a mix a record with a typed
    // catch-all; the empty dict the empty closed record.
    let mut fields = Vec::new();
    let mut defaults = Vec::new();
    for (key, value) in dict.iter() {
        if let Ok(name) = key.cast::<PyString>() {
            let raw = field_name(name)?;
            let (name, required) = match raw.strip_suffix('?') {
                Some(stripped) => (stripped.into(), false),
                None => (raw.into(), true),
            };
            fields.push(Field {
                name,
                schema: build_schema(&value, lits, defs)?,
                required,
            });
        } else {
            let key_schema = build_schema(&key, lits, defs)?;
            checked_key(&key_schema, &key)?;
            defaults.push(MapClause {
                key: key_schema,
                value: build_schema(&value, lits, defs)?,
            });
        }
    }
    Ok(Schema::keyed_map(fields, defaults))
}

pub(super) fn is_ellipsis(obj: &Bound<'_, PyAny>) -> bool {
    let py = obj.py();
    forms(py).is_ok_and(|forms| obj.is(forms.ellipsis.bind(py)))
}
