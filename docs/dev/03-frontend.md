# The schema frontend

`crates/valgebra-py/src/build.rs` turns a Python schema description into the IR.
It is the only place a Python object becomes a pooled index, and the only place
that decides what an annotation means.

The file is the dispatch, the pool and the guard; each section below that names
a *surface* is a module beside it:

| module | the section it follows |
|---|---|
| `build/refine.rs` | How `Annotated` metadata is read |
| `build/classes.rs` | What a class declares |
| `build/generics.rs` | What a parametrized form says |
| `build/dialect.rs` | Three rejections that belong at compile time (the regex dialect) |

## How `Annotated` metadata is read

`parse_constraint` reads the constraint a marker carries by **attribute
protocol**, and the frontend never imports `annotated_types`. A marker carrying
`ge`, `gt`, `le`, `lt`, `min_length`, `max_length`, `multiple_of` or `pattern`
(with its `flags`) contributes the matching constraint, so any library's marker
of that shape works; `Probe` in `build/refine.rs` owns the names. The one
reading by name is the refusal of a vocabulary marker the frontend does not
check: `is_constraint_vocabulary` compares its type's `__module__` with
`annotated_types` and its `__name__` with `DocInfo`.

**A class is never read as a marker**, and is settled before any attribute of
it is read as a value. A marker *class* exposes descriptors where an instance
exposes values: `at.Ge(0)` carries `ge = 0`, while `at.Ge` carries the slot
descriptor that reads it, and taking that for a bound builds a comparison no
value is ordered against. Calling one is the same trap a step later —
`Kilograms(1.5)` constructs a unit marker rather than answering whether `1.5`
belongs.

A class whose instances would be read or refused as a constraint — one from
the vocabulary, or one carrying a name a constraint is read through — is a
marker written without its parentheses, and is **refused** with the spelling
that was meant: ignored, it would widen the schema to its base in silence. The
names are asked of the class itself, since the probe table asks them of a
marker's type, which for a class is its metaclass. Any other class is metadata
this frontend does not recognise, and is ignored.

**A compiled validator narrows by its set**, and is read before any attribute:
`build_refine` meets the refined base with every validator the metadata holds,
grouped or not, so `Annotated[T, v]` is `intersection(T, v)`. It is this
library's own statement of a set, written to narrow the base; ignored as
unrecognised metadata, it would widen the schema to `T` in silence.

The rest are read in this order:

1. a marker that is itself callable — the marker becomes the predicate;
2. otherwise, a marker carrying a callable `func` — its `.func` becomes the
   predicate;
3. otherwise, a marker that contributed nothing above and carries a true
   `__is_annotated_types_grouped_metadata__` — each marker it yields is read in
   turn, to `MAX_GROUPING_DEPTH` levels of grouping;
4. otherwise, a marker that contributed nothing above and comes from the
   `annotated_types` vocabulary — refused, since it was written to narrow this
   schema and ignoring it would admit what it excludes.

Metadata matching none of these is ignored, which the typing spec says a
consumer should do with metadata it has no logic for.

**Reading a constant, a mapping, a class or a sequence in the metadata as a
schema is refused.** The spec leaves the reading to the consumer -- "deciding
how to interpret the metadata (if at all) is the responsibility of the tool or
library" -- so the refusal is this frontend's rule, not the spec's: what it
reads there is its own statement of a set, a compiled validator, and the marker
protocol other libraries share. Each of the four is a form other libraries
write for their own reading -- a description, a mapping of options, a unit
class, a list of tags -- and read as a schema, `Annotated[int, "user id"]`
would be `int` met with the literal `"user id"`, a set with no member, and no
library would say so. The set a schema author means is spelled
`Annotated[T, Validator(s)]` or `intersection(T, s)`. A proposal starts from
the test it fails: `test_metadata_another_library_writes_is_ignored` in
`tests/test_refinements.py`, which holds each of the four kinds to the base it
annotates.

**A question the frontend asks of user code carries a fatal signal out.** A
marker's attributes, a bound's conversion to a float, a length bound's
`__index__`, a step's comparison with zero, the class a bound says it is, a
class's protocol and unpacking flags, and the repr a refusal names each read an
ordinary exception as a documented fallback -- a marker from somewhere else, a
bound that is not a float or not a length, a flag that is not set, an object
that is `<unrepresentable>`. `errors::unless_fatal` is that reading, and it
raises a fatal signal rather than reading it: read as a fallback, an
interrupted `isinstance(bound, Number)` is a bound with no order, and the build
refuses the marker with a sentence about the wrong cause.

**A constraint no value of the base can answer is refused where it is
written.** Reading a length off an `int` raises, and the walk reads a raise as
a non-member, so `Annotated[int, MinLen(1)]` would compile to a set that
admits nothing and says nothing about why. The rule that says which bases can
answer which constraints is the core's `carries` module, with three answers:
every value can, no value can, or the base does not say. Only the second is a
refusal. A union answers for its members, so `Annotated[int | str,
MinLen(1)]` is the non-empty strings; an intersection, a complement and a
reference do not say; and an order bound is asked about the pair, since
Python orders a value only within its own group. The laws' buildable fragment
draws its refinements through the same module, so the pairs the generator
draws are the pairs the frontend builds by construction rather than by two
tables kept alike.

**A name is a handle, and an absence is not an exception.** Every attribute the
protocol asks for is asked by an interned `PyString` the interpreter already
holds, because text would be decoded into a fresh string and hashed before the
lookup could begin, once per name per marker. The rule is the whole frontend's
and not the marker protocol's: the dispatch asks `__metadata__`, `__origin__`
and `__supertype__` the same way, a class node asks `_is_protocol` and a
protocol `__protocol_attrs__` the same way, and each is asked *optionally* --
`getattr_opt` rather than `hasattr` and then `getattr`, which is one lookup
instead of two and no exception where the answer is no. Below 3.13 `PyO3`
reads that answer as an `AttributeError` raised and cleared, and a class
formats the error's message on the way, so on 3.12 a class is asked through
the builtin `getattr` with a default (`optional_attribute` in
`build/classes.rs`), whose lookup answers a class's miss without building it;
3.10 and 3.11 build the error on either road. `__args__`, whose
presence is all the dispatch reads, is asked by `hasattr`, one lookup through
`PyObject_HasAttrWithError`. And a marker carries one or two
of the ten names and not the rest, so absence is the common answer, and giving
it by *raising* costs an exception built, thrown and dropped — four hundred of
them to compile fifty fields. Which names a marker can carry is a property of
its type (`Ge` is a `slots` dataclass, so `Ge.ge` is the descriptor
that reads the slot and `Ge.gt` does not exist), so the type is read once and
its answer kept, and a marker that keeps its values in a dictionary of its own
is read from that dictionary. A type with a `__getattr__` hook answers for
names neither holds, and is asked for everything.

**Callability is how `annotated_types` tells its two marker shapes apart**, so
the order is its rule rather than a heuristic. `Not` defines `__call__` because
calling is what applies the negation; `Predicate` deliberately does not, and
carries its callable on `.func`. Both carry a `.func`, so reading that attribute
first would strip `Not` of its negation — and a `functools.partial` of its bound
arguments, since it has one too.

A class is excluded from the first arm although it is callable, because calling
one **constructs** rather than asks. `Kilograms(1.5)` builds a unit marker; it
does not answer whether `1.5` belongs, and a constructor that rejects the value
would leave the schema uninhabited. Every other callable — a function, a lambda,
a bound method, an object with `__call__` — is asked.

No other library in the ecosystem reads a bare callable as a constraint:
pydantic requires `AfterValidator`, beartype `Is[...]`, and `annotated_types`
supplies `Predicate`; msgspec reads no callable constraint at all, since its
`Meta` carries none. The typing spec leaves the choice to the
consumer, so this arm is a deviation rather than a defect — and the class
exclusion is where the deviation stops being ergonomic and starts being a trap.

### What the order buys

Three markers turn on it, and the wrong order is wrong for two:

| marker | callable | `.func` | read as |
|---|---|---|---|
| `annotated_types.Not(f)` | yes | yes | the marker — calling it applies the negation |
| `functools.partial(eq, 1)` | yes | yes | the marker — calling it supplies the bound arguments |
| `annotated_types.Predicate(f)` | no | yes | `.func` — the marker raises if called |

`tests/test_refinements.py` holds one row each, because a marker of a shape the
suite never receives is a defect nothing reports: read by its `.func`, `Not`
inverts every verdict under it, and a suite with no `Not` row passes.

## The dispatch order is load-bearing

`build_schema` tries forms in an order chosen for the common path, not for
symmetry:

1. `None` — before anything, since it is the most common leaf.
2. `typing.Any` — before the type-object branch, because on 3.11+ `Any` is
   itself a class and would otherwise be taken for an ordinary type.
3. `Never`/`NoReturn` — the lattice bottom, absent on older Pythons and skipped
   there.
4. **A plain type or class** — a scalar, `object`, a TypedDict, a dataclass, an
   enum, a protocol. Taken before the typing introspection below because a type
   never has a typing origin, so this skips a `get_origin` call per scalar node.
5. **An exact `bool`, `int`, `float`, `str` or `bytes`** — a literal of itself.
6. **An already-compiled validator**, whose pool is interned into this one.
   Both after the type branch, because a record's fields are types and a type
   is answered there for a flag test.
7. `Annotated[T, ...]` — the refinement metadata.
8. Anything with a typing origin — `list[int]`, `dict[K, V]`, `tuple[...]`,
   `X | Y`, `Literal`.
9. PEP 695 aliases, `NewType`, native list and dict literals.
10. Anything else — a literal of itself.

Moving a branch earlier is a behaviour change, not a refactor. `Any` above the
type branch is the sharp one. Steps 5 and 6 are the moves that are not, and the
argument is what makes them safe: an exact builtin scalar's type carries no
`__metadata__` or `__supertype__`, `get_origin` answers `None` for it, and it is
no container and no special form, so every arm before the fallthrough passed it
there; a validator has no subclass and matches no arm before its own. Taken
last, each pays an attribute read, a call into `typing` and a dozen tests
first, which measures at three quarters of compiling a two-thousand-constant
`Literal` and two thirds of a `union` of five hundred validators.
`an_exact_builtin_scalar_is_its_own_constant`
holds the first reading to exactly the five types.

**A `typing_extensions` spelling reads as its `typing` one.** Before the release
that adds a form to `typing`, `typing_extensions` defines it with an object of
its own, and an identity check against `typing` alone reads it as something
else: on 3.10 `Never` as a literal of the form object, `Any` as a class
nothing is an instance of, and `Required[int]` as an unsupported form, and
before 3.15 a `TypeAliasType` as a literal of the alias. `Extensions` in
`build.rs` holds those objects, looked up in `sys.modules` once the module has
finished loading -- a module is in `sys.modules` while its body runs, and a
form read then would be kept as absent -- and is asked only where a form would
otherwise be misread: the class fallback (`Any`), the literal fallback
(`Never`, `TypeAliasType`, its own `TypedDict` and `NamedTuple`, and its
`_SpecialForm` for `Self` and `LiteralString`), a `TypedDict` field's
`Required` or `NotRequired` (`qualified_required` in `build/classes.rs`), and
an origin no other arm reads (the three field qualifiers and `Unpack`). A
schema naming none of them pays nothing for it.

## What a class declares

Dispatch step 4 takes any plain type, and what it builds depends on what the
class *says about itself* rather than on what it is called. The order is the
order of the questions, and each is asked of an attribute the runtime fills in:

1. **A builtin scalar** — `bool`, `int`, `float`, `str`, `bytes`, `NoneType` —
   is its own node.
2. **A bare container class** is its kind: `list` admits every list, which is
   the set `list[object]` names and the set the typing spec assigns an
   unparameterised generic. Read as an `isinstance` atom it would be a
   different sort of thing from the sequence node beside it, and neither
   spelling would be decided below the other.
3. **`object`** is the lattice top, and a bare `Union`/`Optional` — a class on
   some Pythons — is refused, because it is a form rather than a value.
4. **A `TypedDict`** declares itself by carrying `__required_keys__`, and
   becomes a keyed map. Its fields are what `get_type_hints` returns, not the
   raw `__annotations__` (see the reading below): under `from __future__ import annotations` those are
   strings, so a `NotRequired[...]` is invisible to the class's own key sets,
   which read every such key as required. A qualifier on the resolved hint wins
   over the key sets, and a qualifier may wrap another. The keys it does not
   name are what PEP 728's two attributes say, read off the class:
   `__closed__` shuts them, `__extra_items__` types them, and neither leaves
   the record open over string keys. The runtime writes both on every class
   with that class's own keywords, and the spec's open default holds "except
   when inheriting from another TypedDict that is not open", so a class that
   gives neither reads its bases: `inherited_tail` walks `__orig_bases__`
   depth first and takes the first base that states one. "No `extra_items` given" is written with
   the `NoExtraItems` sentinel of the implementation that built the class, the
   one in its metaclass's module -- `typing`'s and `typing_extensions`' are two
   objects before 3.15. An implementation older than the sentinel writes `None`
   for it; with one, `None` is the type the extra values have.
5. **An enum** is an instance check against the enumeration class.
6. **A dataclass or a `NamedTuple`** is an instance check *plus* a deep check
   of each declared field, so a value of the right class with a field of the
   wrong type is not a member. A dataclass's fields are an attribute record; a
   named tuple's are the fixed tuple shape of its positions instead
   (`named_tuple_positions` in `build/classes.rs`), which carries the arity as
   well as the types. A field the hints do not carry — a `collections`
   namedtuple's — takes `anything` at its position, which checks the arity
   without the types.
7. **A `Protocol`** is the record of the members it declares: an
   `AttrRecord` with one required field per member and no class beside it, so
   it denotes every value whose attribute, read by `getattr`, holds what the
   member declares. The members are the names `typing` lists --
   `__protocol_attrs__`, which every protocol carries from 3.12 and a
   `typing_extensions` protocol on every release, and otherwise
   `typing._get_protocol_attrs`, the derivation that attribute caches.
   `protocol_members` in `build/classes.rs` classifies each one, and
   `each_protocol_member_is_classified_as_typing_lists_it` holds the
   classification to `typing`'s own caches on the releases that have them:

   | member | where the classifier finds it | its field |
   |---|---|---|
   | data | an annotation names it | the annotation's set |
   | method, special method | the class attribute is callable | the `Callable` atom, as `Callable[...]` reads |
   | property | the first class on the `__mro__` defining it holds a `property` | the getter's return annotation, or `anything` |
   | value | the class attribute is neither callable nor a property | `anything`: present, holding anything |

   `@runtime_checkable` is not read. It is what lets `isinstance` answer, and
   nothing here asks `isinstance`, so one declaration is one set with the
   decorator, without it, and with its mark inherited from a base -- and the
   same set on every release, where `isinstance` finds a member by `hasattr`
   on 3.10 and 3.11 and by `inspect.getattr_static` from 3.12.
   `test_a_protocol_admits_the_same_values_on_every_release` in
   `tests/test_frontend_forms.py` holds the values those two answer apart.

   Three shapes are refused, each in `tests/test_refusal_messages.py`:
   `typing.Protocol` itself, the base a protocol is declared from, which
   declares nothing; a generic `Protocol[T]`, which names one set per type
   argument and has no reading of its parameters yet; and a member declared
   `ClassVar` or `Final`, whose qualifier says where the value lives rather
   than what it is. A protocol naming itself in a member is refused by the
   depth guard, with `recursive(...)` named, as a dataclass is.
8. **Any other class** names its instances: the remaining builtins, the
   `collections.abc` ABCs, and every user class, uniformly.

**A class's annotations are read as written where evaluating them would change
nothing.** `get_type_hints` evaluates forward references, and on a class with
none it hands back the objects it was given -- after copying every base's
namespace and walking every annotation in Python, which is 60 to 70% of
compiling a dataclass or a `TypedDict`. `annotations_as_written` in
`build/classes.rs` takes the call's own reading step for step: reversed
`__mro__`, each base's own annotations, `None` as `type(None)`. It declines, and
the call runs, wherever `typing._eval_type` would rebuild a value: a string, a
`ForwardRef`, a builtin alias with a string argument, an unpacked alias, a
`collections.abc.Callable`, or a class marked `__no_type_check__`. A failure
inside the reading is a decline too, so an error a caller sees is the call's.
A base among `object`, `tuple`, `dict` and the other static builtins
`annotates_nothing` names is not read at all: no assignment reaches such a type
and its namespace holds neither `__annotations__` nor `__annotate__`, so it
contributes the empty table on every release. From 3.14 reading it anyway is a
call into `annotationlib.get_annotations`, where those two names raise an
`AttributeError` with its message formatted for the call to drop -- most of a
`NamedTuple`'s build on 3.14 and 3.15, whose `__mro__` ends in `tuple` and
`object`. `a_builtin_base_holding_no_annotations_is_not_asked_for_them` holds
the list to the reading it skips.
The invariant is equality with the call, and two tests hold it: the Rust
interpreter test compares the dicts on the interpreter the coverage lane
builds, and `tests/test_classes.py` compares the validators on every
interpreter the matrix runs.

```bash
uv run --no-sync python scripts/perf_gate.py --against HEAD~1 --binding-annotated --binding-object
```

**A dataclass's fields are read as `dataclasses.fields` reads them.** The call
is `tuple(f for f in cls.__dataclass_fields__.values() if f._field_type is
_FIELD)` on every supported interpreter, PyPy included, and running that
generator in Python is 15% of compiling a fifty-field dataclass.
`fields_as_declared` in `build/classes.rs` takes the same steps in
the same order -- the table's values, each kept by identity with the marker
before any name is read -- and the call answers wherever a step does not read as
its own: a table that is not exactly a `dict`, a `dataclasses` with no marker.
`a_dataclass_declares_what_fields_returns` holds the reading to the call.

**A module is a handle too.** `dataclasses.is_dataclass` and `dataclasses.fields`
are asked of handles held after the first class that asks and *only* after one
asks: importing `dataclasses` pulls `inspect`, `copy` and `functools` in with
it, and the tracked objects they leave behind are walked by every later garbage
collection. `numbers.Number` -- the register both a multiple-of's remainder and
an order bound's comparison follow -- is held the same way, and for the ordinary
reason rather than that one: written as an import it asks `sys.modules` and
decodes two names **per bound**, which a fifty-field record of
`Annotated[int, Ge(0)]` pays fifty times. A bound of exactly `int`, `float` or
`bool` is a number without asking: `numbers` registers those types and an ABC
keeps what it registers, where the question runs `ABCMeta.__instancecheck__`, a
Python function, once a bound -- 15% of compiling that record. A subclass is
asked, because the ABC also reads a value's `__class__`, which a subclass may
answer with code.

## What a parametrized form says

Dispatch step 8 takes anything with a typing origin, and reads the origin
before the arguments. The origins are compared by identity against the forms
resolved once, at the first build — `typing` is imported once, not once per
node — and a form this frontend does not know is a refusal rather than a guess.
One unknown origin is refused by name: `Validator[int]` is the alias the class's
`__class_getitem__` builds for a static checker, and its origin is `Validator`
itself, which no schema reads. The arm sits after every container, union,
`Literal` and `Callable` arm and before the field qualifiers, whose origins it
never matches, so a form that builds never reaches it.

**The forms are resolved without waiting on another thread's import.** On 3.15
`typing` serves `ForwardRef` through its module `__getattr__`, which reaches
`annotationlib` through a lazy import, and the interpreter resolves a lazy
import holding its global import lock while it waits for the module. A thread
importing `annotationlib` at that moment -- `dataclasses` and `inspect` both do
-- needs that lock for the imports the module's body makes, so the two wait on
each other for good, and the process hangs at its first build. `forms` imports
`annotationlib` itself before it reads `typing`, where the release has it, and
reads `ForwardRef` out of it: an ordinary import waits on the module's own lock
alone, and once it returns every lazy import of the module finds it loaded.
`test_the_first_validator_builds_beside_a_thread_importing_annotationlib` holds
the import open in a fresh interpreter while the first validator builds.

`Union` and `X | Y` are the same origin in two spellings and build the same
node. `Literal` interns each argument as a constant, and refuses a type, a list,
a dict or a set: the typing spec allows `None`, an enum member, or an `int`,
`bool`, `str` or `bytes` value, and a type or a container there would be read as
a schema of its own rather than as a constant (`refuse_unhashable_literal` in
`build/generics.rs`). The refusal names the spelling that was meant. A
constant the spec does not admit there -- a float, an instance -- is read all
the same, as the constant `Validator(c)` reads it: the set is the same
singleton, and a static checker is what refuses the spelling.

A `tuple` reads its arguments as a *shape*: `tuple[int, str]` is a fixed
sequence of two, `tuple[int, ...]` is a homogeneous one, and `tuple[()]` is the
empty tuple. `Unpack[Ts]` and `*tuple[int, ...]` are the same unpacking in two
spellings. A shape is a fixed prefix and then a repeating tail, so nothing may
follow the tail and a tuple cannot begin with `...`; both are refused by naming
the spelling that was meant.

Five origins are read as containers — `list`, `set`, `frozenset`, `dict` and
`tuple` — and their arguments become element and key types. A parametrized
`collections.abc` generic other than `Callable` is **refused**, not read:
`Sequence[int]` names a protocol whose instances a check cannot enumerate
without consuming them, and the bare abstract class is available as an
`isinstance` atom instead ([the API page](../16-api.md) records the refusal).
`Callable[...]`, from either module, is read as the bare `Callable` atom: a
callable's parameter and return types cannot be read at runtime, so the
arguments are dropped and the schema asks only `isinstance(x, Callable)`.
`dict[K, V]`'s two are read by name rather than by position, because the two
transposed is `dict[V, K]`, which typechecks and validates real values.

The two **native literals** are read here too, because they answer the same
question: `[A, B]` is the fixed-length list, which `typing` cannot spell, and a
`dict` literal is a record — string keys are named fields, with `"key?"` for an
optional one, and any other key is a schema governing the rest. A tuple literal
and a set literal are refused rather than accepted, since `tuple[A, B]` and
`set[T]` already spell them and two spellings for one set is a fork. The frozen
siblings are refused for the same reason and need arms of their own: a
`frozenset` is not a `set` and, on 3.15, a `frozendict` is not a `dict`, so
without one each falls through to the constant reading and becomes a schema
admitting that one object.

**A `frozendict` is a `Mapping` and not a `dict`.** The record and mapping node
denotes dicts (`ir.rs`), so `dict[K, V]`, a record and a `TypedDict` refuse a
`frozendict` value, and the bare `collections.abc.Mapping` atom admits it.
`frozendict[K, V]` names a set of frozen dicts, a carrier the node set does not
have, and is refused as an unknown origin rather than read as `dict[K, V]`:
admitting it is a question for the admission test of
[01-schema-ir.md](01-schema-ir.md), not for this page.
`test_a_frozen_dict_is_a_mapping_and_not_a_dict` in
`tests/test_frontend_forms.py` holds all three, on the interpreters that have
the type. A key
narrowed by a constraint is refused where "Three rejections that belong at
compile time" says why.

## Where an index acquires its meaning

One pool holds four kinds of object, so the slot means nothing until the frontend
says what it pooled. That decision is at the mint, not at the read:

```rust
lits.intern_const(obj)      // the constant of a typed singleton
lits.intern_class(ty)       // an isinstance atom
lits.intern_operand(bound)  // a comparison or multiple-of operand
lits.intern_predicate(func) // a user callback
```

Each returns its own index type. The private `intern` beneath them deduplicates
through a hashed map keyed by what the object is as a constant -- an exact
builtin scalar by its type and value, anything else by its address -- so
compiling a wide `Literal[...]` or merging many validators stays linear rather
than quadratic. The address key is stable because every interned value is kept
alive by the pool.

**A relation reads each validator's keys once.** Relating two validators pools
the other side's constants into a pool seeded with the subject's, and reading a
key reads the constant out of the interpreter -- for a string, a copy of its
text. A validator keeps the keys of its own pool, as `PoolKeys`, from the first
relation it is a side of; a pool seeded from them shares them rather than
rebuilding them, and keeps only the constants new to it in a map of its own.
Composition -- `union`, `|` -- reads the keys a relation has kept where there
are some, reads the rest fresh, and keeps none (`compose` in `build.rs`), so a
validator nobody relates carries nothing for it.
`a_pool_seeded_by_shared_keys_pools_as_one_seeded_by_rebuilding` holds the two
seedings to one answer.

## Three rejections that belong at compile time

**A typing construct that carries no runtime value** — a `TypeVar`, a
`ParamSpec`, a bare `Final` or `ClassVar` — is refused rather than interned as a
literal. Interning it would produce a schema that admits only objects equal to
the TypeVar, which is almost nothing, and the user would see a validation failure
instead of a compile error.

**A zero divisor.** `MultipleOf(0)` is unsatisfiable and checking it would divide
by zero at validation time, so the error is raised where the schema is built.

**A map key narrowed by a constraint.** A clause's key says which keys it
governs, and a map reads that as whole *kinds* of key or as the constants a
`Literal` names. `Annotated[str, MinLen(2)]` is neither: it names part of a kind,
and two such clauses can overlap without either containing the other. Overlapping
key domains are a different theory from the one these maps are built on --
Castagna's §4.5 says that with them "there would not be any difference between
record types and an intersection of function types" -- and this library reads
overlapping clauses disjunctively, which answers a question the two theories do
not agree on.

The same principle governs the regex: the pattern is compiled and anchored at
build time, so an invalid expression fails at construction rather than at first
validation. The pattern is parsed on its own before the anchors wrap it: `a)|(b`
does not parse, and wrapped it closes the anchors' group and opens one they
close, so its alternation would escape both of them. A pattern this engine
compiles and `re` reads as a different set is refused before the compile:
`reject_reserved_class_syntax` in `build/dialect.rs` refuses a character class
carrying `--`, `&&`, `~~` or a nested `[` other than a POSIX class, and a space
or a `#` inside a class under verbose mode, each named by the reading this
engine would give it.

## Recursion is tied by `recursive` or by an alias

A class whose own type appears in its fields is recursive, and the frontend
refuses to chase it — a depth guard bounds `build_schema` and returns a message
naming `recursive(...)` as the way to express it. The bound exists so that case
fails cleanly instead of overflowing the native stack.

A PEP 695 alias is its own binder, so the fixpoint it names needs no call.
`build_alias` in `build.rs` stands a token for the alias while its body is
built -- the alias is one object, found again by its address in
`OPEN_ALIASES` -- turns every occurrence of the token into a reference to the
definition the body becomes, and refuses a body in which the alias occurs
outside a structural constructor, since `type X = int | X` names no set a value
settles. An alias that never names itself builds its body and no definition.

## The limit

**The frontend decides meaning; nothing checks it against the typing spec
mechanically.** A form compiled to the wrong node is a defect no gate in this
tree catches — the differential lane compares against pydantic-core and
jsonschema over the fragment where the semantics agree, with the divergences
enumerated in `tests/test_differential.py`, and that is the closest thing to an
external judge.

**A transposition inside one call is not closed by a type.** A clause's key
schema and value schema are both `Schema`, so `dict[K, V]` compiled with them
swapped would typecheck and validate real values. The frontend spells each
clause as the struct literal `MapClause { key, value }`, where a swap has to be
written out at the call site; [06-type-design.md](06-type-design.md) records the
positional constructor the core keeps as the sharpest residual hazard in the
tree.
