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

use crate::descr::maps::KEY_KINDS;
use crate::ir::{DefIx, Field, MapClause, Schema};
use crate::kind::Kind;
use crate::verdict::{Relation, Verdict};

use super::{Budget, LeafRelations, SubtypeCx};

/// What the keyed maps meeting in an intersection admit between them: `Empty`
/// where no dict is in every map, `Inhabited` where every member is a closed
/// map and the dict the rules below name is in all of them, and `Unknown`
/// otherwise -- every member a map or not.
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
/// **The converse is read off the same keys, where every member is a closed
/// map.** The dict holding exactly the required keys, each with a value in the
/// meet of the types its maps give it, is in every map: each map declares each
/// of those keys (a closed map lacking one empties the meet above), requires
/// none besides, and admits the value it is given. So the meet is inhabited
/// where every required key's types meet in a value, and nothing else of a
/// member needs reading: every closed map declares every required key, or the
/// meet is empty above, so no key is one map's alone. A map with clauses gives
/// a key it does not declare a value
/// the clause decides, which the core cannot read off a name, so the converse
/// declines on one.
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
pub(super) fn keyed_map_meet_verdict(
    members: &[Schema],
    oracle: &dyn LeafRelations,
    defs: &[Schema],
    visiting: &mut Vec<DefIx>,
    budget: &Budget,
) -> Verdict {
    let maps: Vec<(&[Field], bool)> = members
        .iter()
        .filter_map(|member| match member {
            Schema::KeyedMap { fields, defaults } => Some((&fields[..], defaults.is_empty())),
            _ => None,
        })
        .collect();
    if maps.len() < 2 {
        return Verdict::Unknown;
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
    let mut required_hold = Verdict::Inhabited;
    for (name, (types, _)) in keys.iter().filter(|(_, (_, required))| *required) {
        let held = if types.len() > 1 {
            let meet = Schema::Intersection(types.iter().copied().cloned().collect());
            meet.verdict_rec(oracle, defs, visiting, budget)
        } else {
            Verdict::Inhabited
        };
        if held.is_empty()
            || maps.iter().any(|(fields, closed)| {
                *closed && !fields.iter().any(|field| *field.name == **name)
            })
        {
            return Verdict::Empty;
        }
        required_hold = Verdict::every([required_hold, held].into_iter());
    }
    if maps.len() == members.len() && maps.iter().all(|(_, closed)| *closed) {
        required_hold
    } else {
        Verdict::Unknown
    }
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
/// A catch-all clause of `a` that no clause of `b` subsumes is asked for a key
/// it admits that `b` reads one way only ([`clause_escapes`]): a refutation
/// where some value under that key is outside `b`, a decline otherwise.
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
                // anything to say about it. A clause whose key holds no string
                // -- another kind, or the complement of `str` that `open` writes
                // -- never spells the name, so it governs nothing here and its
                // value type is beside the point; reading it anyway refutes on
                // a value `a` does not have.
                None => Relation::all(da.iter().map(|clause| {
                    let covers = clause
                        .value
                        .is_subtype_rec(&b_field.schema, cx, assumptions);
                    if clause.key.holds_every_value_of(Kind::Str, cx.oracle) {
                        // Every string key, so this clause does spell the name.
                        covers
                    } else if clause.key.holds_no_value_of(Kind::Str, cx.oracle) {
                        Relation::Holds
                    } else {
                        // A key the rules cannot read -- a string literal, a
                        // union of them -- might admit the name, so a proof
                        // carries and a refutation does not.
                        covers.proof_only()
                    }
                })),
            }
        }));
        // Each field `a` declares that `b` does not is read by `b` through its
        // catch-all, so a clause of `b` whose key holds every string must cover
        // it.
        //
        // A clause whose key the rules cannot read might admit the name, so
        // where `b` carries one the reading is a proof or nothing. Where every
        // clause's key either holds every string or holds none, the covering
        // clauses are the first kind, and a refutation needs a value of the
        // field that every one of them rejects: fields are independent, so a
        // value of `a` carrying that key with that value is a value `b`
        // rejects, and a clause holding no string never reads the name. Each
        // covering clause's own refutation stands on a value of its own, so
        // they make one only where there is at most one of them -- the closed
        // record has none. Two may cover the field together, `int | str` under
        // `str: int` and `str: str`, and refuting each in turn proves nothing
        // about the pair. Asking the field against the union of their values
        // would, and is not done: that union is a term no side spelled, so the
        // goal would fall outside the closure of the query's subterms that
        // `DECISION_BUDGET` stands on. A refutation also stands on the field
        // having a value, read the way the query reads the subject's, so a
        // field the rules cannot tell either way declines and an empty
        // *optional* field proves nothing against.
        let (covering_clauses, unread_clause) =
            db.iter()
                .fold((0usize, false), |(covering, unread), clause| {
                    if clause.key.holds_every_value_of(Kind::Str, cx.oracle) {
                        (covering + 1, unread)
                    } else {
                        (
                            covering,
                            unread || !clause.key.holds_no_value_of(Kind::Str, cx.oracle),
                        )
                    }
                });
        let one_witness = !unread_clause && covering_clauses <= 1;
        let extra_covered = Relation::all(
            fa.iter()
                .filter(|a_field| b_by_name.named(&a_field.name).is_none())
                .map(|a_field| {
                    let covering = db
                        .iter()
                        .filter(|clause| clause.key.holds_every_value_of(Kind::Str, cx.oracle));
                    let answer = Relation::any(covering.map(|clause| {
                        a_field
                            .schema
                            .is_subtype_rec(&clause.value, cx, assumptions)
                    }));
                    match answer {
                        Relation::Holds => Relation::Holds,
                        Relation::Fails if one_witness => {
                            Relation::of_mismatch(a_field.schema.verdict_of(cx))
                        }
                        _ => Relation::Unknown,
                    }
                }),
        );
        // Every catch-all clause of `a` (governing its non-field keys) is subsumed
        // by a clause of `b` with both key and value narrower.
        let defaults = Relation::all(da.iter().map(|mine| {
            // One clause of `b` subsuming this one settles it. None of them
            // doing so is not yet a refutation -- the clause may be covered by
            // several of `b`'s between them, or by a clause pair the rules
            // cannot relate -- so the clause is asked for a key it admits and
            // `b` reads one way only.
            //
            // The value inclusions the search below asks are kept for that
            // second question, which asks one of them again whenever the key
            // that clause reads is one `mine` holds. Asking it twice doubles
            // the work at every level of a nested map, and a chain twenty
            // `dict[str, ...]` deep then spends the budget on a refutation one
            // reading names. Kept only where the answer is not a proof, since a
            // proof ends the search, so a subsumed clause allocates nothing.
            let mut values_asked: Vec<(&Schema, Relation)> = Vec::new();
            let subsumed = db.iter().any(|theirs| {
                if !mine
                    .key
                    .is_subtype_rec(&theirs.key, cx, assumptions)
                    .holds()
                {
                    return false;
                }
                let answer = mine.value.is_subtype_rec(&theirs.value, cx, assumptions);
                if !answer.holds() {
                    values_asked.push((&theirs.value, answer));
                }
                answer.holds()
            });
            if subsumed {
                Relation::Holds
            } else {
                clause_escapes(mine, db, &mut values_asked, cx, assumptions)
            }
        }));
        // Three conjuncts of one claim about one pair, so any one of them
        // refutes it and the order they are read in decides nothing.
        Relation::all([fields_ok, extra_covered, defaults])
    }
}

/// Whether the catch-all clause `mine` of a subject admits an entry no dict of
/// the supertype's clauses `db` holds: `Fails` where one is found, `Unknown`
/// where none is.
///
/// ICFP Lemma 4.7 refutes a map below another one key-type at a time: a key
/// the subject's clause admits and no clause of the supertype does, or one
/// both admit whose value the subject's clause allows and the supertype's
/// forbids. This is that, read one **key kind** at a time. For a kind whose
/// every value the clause's key holds, each clause of `b` either holds every
/// value of the kind too or holds none ([`holds_every_value_of`],
/// [`holds_no_value_of`]); a clause doing neither declines the kind. Then a
/// key of the kind is read by at most one clause of `b`, and the witness is a
/// value of `a` with one entry added, under a key the rule chooses:
///
/// - **no clause reads the kind**: any value of `mine`'s value type, which
///   needs that type to have one;
/// - **one clause reads it**: a value `mine` allows and that clause forbids,
///   which is the clause's value refuted below it.
///
/// The key is a plain value of the kind -- `None`, `True`, `7`, `0.5`, `b""`,
/// `()`, `frozenset()` -- and for `str` a name neither map declares as a field,
/// which exists because fields are finitely many and the clause holds every
/// string. Each of those is hashable, and none is a field name of `b`, so `b`
/// reads it through its clauses alone and reads it as this says; and none is
/// a field name of `a`, so `a` admits the entry through `mine`. The rest of
/// the witness is a value of `a` with its undeclared keys dropped, which the
/// witness guard around the query reads as `a` being inhabited.
///
/// **Two clauses reading one kind decline**, as two covering one field do
/// above: each refutes on a value of its own, and the union of their values
/// is a term neither side spelled, outside the closure of subterms the
/// budget's argument counts goals in. **A kind is read whole or not at all**:
/// a key of `tuple[list[int]]` or of a class whose instances are unhashable
/// holds no key at all, and a key reading by values rather than kinds would
/// refute `{U: int} <= {str: int}`, which holds because `{U: int}` admits only
/// `{}`.
///
/// The finite kinds need nothing of their own. Lemma 4.7's proof assumes
/// every key type is infinite, and `NoneType` and `bool` are not; but the rule
/// refutes on one key, and a finite kind has one as surely as an infinite one.
/// What a finite kind changes is the *proof*, which this does not attempt.
///
/// [`holds_every_value_of`]: Schema::holds_every_value_of
/// [`holds_no_value_of`]: Schema::holds_no_value_of
///
/// `values_asked` holds the inclusions of `mine.value` in a clause value the
/// caller has already asked, and gains the ones asked here: the same goal asked
/// twice is the same work twice, and at every level of a nested map.
fn clause_escapes<'b>(
    mine: &MapClause,
    db: &'b [MapClause],
    values_asked: &mut Vec<(&'b Schema, Relation)>,
    cx: SubtypeCx<'_>,
    assumptions: &mut Vec<(Schema, Schema)>,
) -> Relation {
    // The questions a kind asks of `mine.value` repeat across kinds -- a
    // complement key reads seven of them -- so each is asked once: its own
    // inhabitance, and its inclusion in each clause value it is held to.
    let mut inhabited: Option<Relation> = None;
    for kind in KEY_KINDS {
        if !mine.key.holds_every_value_of(kind, cx.oracle) {
            continue;
        }
        let mut reader: Option<&Schema> = None;
        let mut read_one_way = true;
        for theirs in db {
            if theirs.key.holds_no_value_of(kind, cx.oracle) {
                continue;
            }
            if reader.is_none() && theirs.key.holds_every_value_of(kind, cx.oracle) {
                reader = Some(&theirs.value);
                continue;
            }
            read_one_way = false;
            break;
        }
        if !read_one_way {
            continue;
        }
        let answer = match reader {
            None => {
                *inhabited.get_or_insert_with(|| Relation::of_mismatch(mine.value.verdict_of(cx)))
            }
            Some(value) => {
                if let Some((_, answer)) = values_asked
                    .iter()
                    .find(|(asked, _)| core::ptr::eq(*asked, value))
                {
                    *answer
                } else {
                    let answer = mine.value.is_subtype_rec(value, cx, assumptions);
                    values_asked.push((value, answer));
                    answer
                }
            }
        };
        if answer == Relation::Fails {
            return Relation::Fails;
        }
    }
    Relation::Unknown
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
