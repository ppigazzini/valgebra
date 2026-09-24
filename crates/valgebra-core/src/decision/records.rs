//! Records and mappings: what a keyed map admits, and when two of them relate.
//!
//! A dict is a function from keys to values that is constant on all but
//! finitely many of them, so a *set* of dicts is written by naming the keys it
//! constrains and saying what every other key maps to. That is the node these
//! rules read, and they read it in the shape the frontend writes rather than in
//! a canonical form: the fields as spelled, and the clauses unordered, so a key
//! belongs when *some* clause admits it.
//!
//! The cursor is what keeps a field lookup from being a scan. Both field lists
//! are sorted by name where the constructors built them, so two sorted lists
//! are walked together and the cursor falls back to a scan only where one is
//! not -- which it asks once, on a miss, rather than per lookup.

use std::cell::Cell;

use rustc_hash::FxHashMap;

use crate::ir::{DefIx, Field, MapClause, Schema};
use crate::kind::Kind;
use crate::verdict::Relation;

use super::{LeafRelations, SubtypeCx};

/// Whether the keyed maps meeting in an intersection admit no dict between them.
///
/// ICFP formula (12) meets two record atoms pointwise -- field by field, clause
/// by clause -- and formula (11) makes the result empty when a field's type is.
/// A dict in the meet carries every key some side requires, with a value in every
/// type its sides give that key, so two rules follow:
///
/// - a key required somewhere whose types meet to nothing admits no dict;
/// - a key required somewhere and absent from a *closed* map admits none either,
///   since a closed map is exactly its declared keys.
///
/// Only a **required** key can empty a meet. Footnote 11 of the same paper is the
/// guard: the meet of two mappings "is never empty since it always contains at
/// least the empty record expression", and two optional fields are the same case
/// -- the empty dict satisfies both.
///
/// A map with clauses is not read as closed here. Deciding whether a clause
/// admits a given name means comparing a bare `String` against a key schema,
/// which the core cannot do, so any clause at all leaves the map open and the
/// second rule declines.
///
/// **The meet of a key's types is read under the caller's `visiting`.** It is a
/// required position of the node being decided, so a reference already being
/// resolved is read as the cycle it is, as the field of a single map reads it.
/// A fresh list there unfolds the reference again: a recursive meet of two maps
/// reaches this rule once per unfolding, with nothing to stop it but the stack.
pub(super) fn keyed_map_meet_empty(
    members: &[Schema],
    oracle: &dyn LeafRelations,
    defs: &[Schema],
    visiting: &mut Vec<DefIx>,
    budget: &Cell<u32>,
) -> bool {
    let maps: Vec<(&[Field], bool)> = members
        .iter()
        .filter_map(|member| match member {
            Schema::KeyedMap { fields, defaults } => Some((&fields[..], defaults.is_empty())),
            _ => None,
        })
        .collect();
    if maps.len() < 2 {
        return false;
    }
    // Every type the maps give a key, and whether any of them requires it.
    let mut keys: FxHashMap<&str, (Vec<&Schema>, bool)> = FxHashMap::default();
    for (fields, _) in &maps {
        for field in *fields {
            let entry = keys.entry(&*field.name).or_default();
            entry.0.push(&field.schema);
            entry.1 |= field.required;
        }
    }
    keys.iter()
        .filter(|(_, (_, required))| *required)
        .any(|(name, (types, _))| {
            let types_cannot_hold = types.len() > 1 && {
                let meet = Schema::Intersection(types.iter().copied().cloned().collect());
                meet.is_empty_rec(oracle, defs, visiting, budget)
            };
            types_cannot_hold
                || maps.iter().any(|(fields, closed)| {
                    *closed && !fields.iter().any(|field| *field.name == **name)
                })
        })
}

/// Whether keyed-map `a` (fields `fa`, default clauses `da`) is a subtype of
/// keyed-map `b`. Sound everywhere; complete on three shapes, conservative
/// (returns `false`) outside them:
///
/// 1. **Closed record ≤ anything** (`da` empty): holds by width and depth (each
///    field of `a` maps into a like-named field of `b` with a subtype schema) and
///    by required-ness (every field `b` requires is required in `a`).
/// 2. **Pure mapping ≤ pure mapping** (`fa` and `fb` empty): every clause of `a`
///    is subsumed by a clause of `b` with both key and value narrower.
/// 3. **Mixed record-and-catch-all ≤ mixed** (general): each shared field narrows
///    and respects required-ness; each field `a` declares that `b` does not is
///    covered by `b`'s catch-all; each field `b` requires that `a` lacks is
///    governed by `a`'s catch-all — decidable when it is **optional** and every
///    catch-all value of `a` fits it, and *refuted* when `a` carries no
///    catch-all at all, since a closed record admits no value with a key it does
///    not declare; and every catch-all clause of `a` is subsumed by one of `b`.
///
/// Sound throughout — a required supertype field a subject with a catch-all
/// cannot guarantee present, or a clause an oracle cannot relate, is undecided
/// rather than an unsound proof, and a refutation stands on a value of the
/// subject, which the query's witness guard reads against its emptiness.
pub(super) fn keyed_map_subtype(
    fa: &[Field],
    da: &[MapClause],
    fb: &[Field],
    db: &[MapClause],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    // Index both field lists by name once, so the cross-list lookups below are O(1)
    // each rather than a fresh linear scan per field (O(fields²) per comparison).
    let order = Cell::new(None);
    let mut a_by_name = FieldCursor::over(fa, fb, &order);
    let mut b_by_name = FieldCursor::over(fb, fa, &order);
    {
        // One rule for every shape a keyed map takes. The closed record and the
        // pure mapping are not special cases needing a branch of their own: each
        // is this rule with one of the two lists empty, and a dedicated branch
        // for either can only answer the same question less well. The closed
        // record did -- it read a field the supertype covered through a catch-all
        // as undecided, so `{"x": int}` was not seen below `dict[str, int]`.
        //
        // Every supertype field is checked against `a`: a field `a` declares is
        // matched field-wise; a field `a` lacks is governed by `a`'s catch-all.
        // One goal, asked again. A record whose fields repeat a type asks the
        // same question once per field, and answering it once is the whole of
        // what a memo over goals would do here.
        //
        // Compared by *equality* rather than by address: two fields carrying one
        // schema hold two `Schema` values, each in its own slot of the field
        // list, and what they share is everything under them. Equality between
        // two nodes whose payloads are the same allocations is a discriminant
        // and a pointer per payload, which is what makes this cheaper than the
        // walk it skips.
        //
        // One entry rather than a table: the fields are walked in order, so a
        // repeat is the field before this one. A table of goals over the whole
        // query was measured beside this and cost the shapes with nothing to
        // repeat more than it saved the shapes with something.
        //
        // The coinductive hypothesis needs no thought here, which is the other
        // reason this is the place for it: the trail is the same at every field
        // of one record -- a field's own decision pushes and pops its way back
        // to it -- so the answer read is the answer to the same goal under the
        // same hypotheses. A table spanning a whole query cannot say that, and
        // has to refuse every goal decided under a hypothesis at all.
        let mut last: Option<(&Schema, &Schema, Relation)> = None;
        let fields_ok = Relation::all(fb.iter().map(|b_field| {
            match a_by_name.named(&b_field.name) {
                // Shared field: it must narrow in depth, and a field `b` requires
                // must be required in `a` too. A key the supertype requires and
                // the subtype does not is a value of the subtype -- the one
                // leaving that key out -- that the supertype rejects.
                Some(a_field) => {
                    let depth = match last {
                        Some((sub, sup, answer))
                            if sub == &a_field.schema && sup == &b_field.schema =>
                        {
                            answer
                        }
                        _ => {
                            let answer =
                                a_field
                                    .schema
                                    .is_subtype_rec(&b_field.schema, cx, assumptions);
                            last = Some((&a_field.schema, &b_field.schema, answer));
                            answer
                        }
                    };
                    depth.and(|| Relation::decided(!b_field.required || a_field.required))
                }
                // A field `b` requires that `a` does not declare. A clause
                // governs the keys a value carries and never requires one, so
                // `a` admits a value without this key whatever its clauses say:
                // take a value of `a` and drop the key, and every required
                // field is still there and every key left is one a clause
                // already covered. That value is one `b` rejects, which is a
                // refutation by the same reading as a key `a` declares optional
                // two arms above. It stands on `a` having a value at all, which
                // is what the reading around this rule settles -- an empty `a`
                // is below every schema, this one included.
                None if b_field.required => Relation::Fails,
                // An optional field `b` declares that `a` does not: a value of
                // `a` carries that key only where one of `a`'s clauses produces
                // it, so only a clause whose **key admits the name** has
                // anything to say about it. A clause keyed by another kind
                // never spells a string, so it governs nothing here and its
                // value type is beside the point; reading it anyway refutes on
                // a value `a` does not have.
                None => Relation::all(da.iter().map(|clause| {
                    let covers = clause
                        .value
                        .is_subtype_rec(&b_field.schema, cx, assumptions);
                    match &clause.key {
                        // Every string key, so this clause does spell the name.
                        Schema::Str | Schema::Anything(_) => covers,
                        // A key of a settled kind that is not a string: no value
                        // of `a` carries this name through this clause.
                        key if key
                            .type_tag_with(cx.oracle)
                            .is_some_and(|kind| kind != Kind::Str) =>
                        {
                            Relation::Holds
                        }
                        // A key the rules cannot read -- a string literal, a
                        // union of them -- might admit the name, so a proof
                        // carries and a refutation does not.
                        _ => covers.proof_only(),
                    }
                })),
            }
        }));
        // Each field `a` declares that `b` does not is read by `b` through its
        // catch-all, so a `str`/`anything`-keyed clause of `b` must cover it.
        //
        // A clause whose key the rules cannot read might admit the name, so
        // where `b` carries one the reading is a proof or nothing. Where every
        // clause's key is one of the two spellings that plainly admit a string
        // name -- and where `b` carries no clause at all, which is the closed
        // record -- the covering clauses are all of them, and a field whose
        // values none of them accepts is a refutation: fields are independent,
        // so a value of `a` carrying that key with that value is a value `b`
        // rejects. It stands on the field having a value, read the way the
        // query reads the subject's, so a field the rules cannot tell either
        // way declines and an empty *optional* field proves nothing against.
        let readable_keys = db
            .iter()
            .all(|clause| matches!(clause.key, Schema::Str | Schema::Anything(_)));
        let extra_covered = Relation::all(
            fa.iter()
                .filter(|a_field| b_by_name.named(&a_field.name).is_none())
                .map(|a_field| {
                    let covering = db
                        .iter()
                        .filter(|clause| matches!(clause.key, Schema::Str | Schema::Anything(_)));
                    let answer = Relation::any(covering.map(|clause| {
                        a_field
                            .schema
                            .is_subtype_rec(&clause.value, cx, assumptions)
                    }));
                    match answer {
                        Relation::Holds => Relation::Holds,
                        Relation::Fails if readable_keys => {
                            Relation::of_mismatch(a_field.schema.verdict_of(cx))
                        }
                        _ => Relation::Unknown,
                    }
                }),
        );
        // Every catch-all clause of `a` (governing its non-field keys) is subsumed
        // by a clause of `b` with both key and value narrower.
        let defaults = Relation::all(da.iter().map(|mine| {
            // One clause of `b` subsuming this one settles it; none of them
            // doing so is a decline, since a clause pair the rules cannot
            // relate is not a pair they have refuted.
            Relation::proven(db.iter().any(|theirs| {
                mine.key
                    .is_subtype_rec(&theirs.key, cx, assumptions)
                    .and(|| mine.value.is_subtype_rec(&theirs.value, cx, assumptions))
                    .holds()
            }))
        }));
        // Three conjuncts of one claim about one pair, so any one of them
        // refutes it and the order they are read in decides nothing.
        Relation::all([fields_ok, extra_covered, defaults])
    }
}

/// Whether the attribute record `fa` is a subtype of `fb`: width and depth.
///
/// Every attribute the supertype declares must be one the subtype declares, and
/// no less narrowly. There is no class in it -- a record denotes every value
/// carrying its attributes, whatever the value is -- so the nominal question the
/// old node asked first is now a conjunct of its own, and two records that came
/// from unrelated classes still relate.
///
/// The rule is set inclusion read off the denotation: an attribute the supertype
/// does not name constrains nothing, and one it names constrains every value of
/// the subtype exactly when the subtype names it too, at least as narrowly.
pub(super) fn attr_record_subtype(
    fa: &[Field],
    fb: &[Field],
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    let order = Cell::new(None);
    let mut a_by_name = FieldCursor::over(fa, fb, &order);
    Relation::all(fb.iter().map(|b| {
        match a_by_name.named(&b.name) {
            // An attribute the supertype names and the subtype does not: the
            // subtype holds values without it, and those are outside the
            // supertype. So is a supertype attribute the subtype only *may*
            // carry.
            None => Relation::Fails,
            Some(a) if b.required && !a.required => Relation::Fails,
            Some(a) => a.schema.is_subtype_rec(&b.schema, cx, assumptions),
        }
    }))
}

/// Whether a field list is in name order, which is what lets a cursor over it
/// move forward only.
pub(super) fn sorted_by_name(list: &[Field]) -> bool {
    list.is_sorted_by(|one, two| one.name <= two.name)
}

/// One walk over a field list, finding each name of another list as it passes.
///
/// Both lists are in name order, so a partner is found by moving forward: no
/// allocation, and one string compare per field passed. The table this replaces
/// was built and freed once per side per comparison, and was a fifth of the
/// repeating workload.
///
/// **Unique names are the caller's invariant**, and the order is what makes it
/// load-bearing here: two fields of one name would let the cursor answer with
/// whichever it reached first, and the `required` and width readings that
/// consume the answer would decide on a field the other shadows. The frontend
/// refuses a duplicate; the `debug_assert` names the dependency and catches a
/// malformed IR in debug rather than deciding on the wrong field.
pub(super) struct FieldCursor<'a, 'o> {
    fields: &'a [Field],
    rest: &'a [Field],
    /// The list this cursor is asked the names of, kept so the order can be
    /// read later rather than now.
    queries: &'a [Field],
    /// Whether both lists are in name order, once that has been asked --
    /// shared with the sibling cursor over the same pair, which asks the
    /// identical question. `None` until a lookup misses, because until then
    /// nothing depends on the answer.
    order: &'o Cell<Option<bool>>,
}

impl<'a, 'o> FieldCursor<'a, 'o> {
    /// A cursor over `fields`, to be asked for the names of `queries` in turn.
    ///
    /// Neither list is read for order here. The cursor never moves back, so a
    /// list out of order can make it look past a field that is behind it --
    /// but only into answering that the list does not carry the name, never
    /// into answering with the wrong field: names are unique, and the cursor
    /// stops on an *equal* name. So a lookup that finds something is right
    /// whatever the order is, and the order matters to a lookup that finds
    /// nothing. That is where it is read ([`missed`](Self::missed)), once, and
    /// through a cell the sibling cursor over the same pair shares -- each of
    /// the two needs *both* lists sorted, which is one question.
    fn over(
        fields: &'a [Field],
        queries: &'a [Field],
        order: &'o Cell<Option<bool>>,
    ) -> FieldCursor<'a, 'o> {
        debug_assert!(
            !sorted_by_name(fields)
                || fields
                    .windows(2)
                    .filter_map(|pair| Some((pair.first()?, pair.get(1)?)))
                    .all(|(one, two)| one.name != two.name),
            "record has duplicate field names; the frontend must reject them"
        );
        FieldCursor {
            fields,
            rest: fields,
            queries,
            order,
        }
    }

    /// The field called `name`.
    ///
    /// One *ordering* comparison per field the cursor passes, and one for the
    /// field it stops on: the walk forward and the test that the field it
    /// stopped on is the one wanted are the same question asked once, where a
    /// `take_while` on `<` followed by a `filter` on `==` asks the stopping
    /// field twice. Each of those is a string compare, which is a call.
    fn named(&mut self, name: &str) -> Option<&'a Field> {
        if self.order.get() == Some(false) {
            return self.fields.iter().find(|field| &*field.name == name);
        }
        while let Some((first, rest)) = self.rest.split_first() {
            match (*first.name).cmp(name) {
                core::cmp::Ordering::Less => self.rest = rest,
                core::cmp::Ordering::Equal => return Some(first),
                core::cmp::Ordering::Greater => break,
            }
        }
        self.missed(name)
    }

    /// The field called `name`, for a walk that reached no such name.
    ///
    /// The one answer the cursor cannot give without knowing the order, so the
    /// order is read here and nowhere else. Read once: both lists in order
    /// makes every later miss a real one, and either out of order puts this
    /// cursor and its sibling on the scan for the rest of the comparison.
    fn missed(&mut self, name: &str) -> Option<&'a Field> {
        if self.order.get().is_some() {
            return None;
        }
        let ordered = sorted_by_name(self.fields) && sorted_by_name(self.queries);
        self.order.set(Some(ordered));
        if ordered {
            return None;
        }
        self.rest = self.fields;
        self.fields.iter().find(|field| &*field.name == name)
    }
}
