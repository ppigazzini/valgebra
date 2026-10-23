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

use crate::descr::lower::key_cover;
use crate::descr::maps::{KEY_KINDS, Label, PARTS, key_slot};
use crate::ir::{Attribute, Field, MapClause, Schema};
use crate::kind::Kind;
use crate::oracle::denotes_a_set_within;
use crate::verdict::{Relation, Verdict};

use super::literals::finite_set;
use super::{Cx, descending, spend};

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
pub(super) fn keyed_map_meet_verdict(members: &[Schema], cx: &mut Cx<'_>) -> Verdict {
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
            meet.verdict_rec(cx)
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
/// keyed-map `b`, read piece by piece. Sound everywhere, and complete on no
/// syntactic shape: it proves where one piece of `b` covers each piece of `a`,
/// refutes on a value it names, and declines where pieces of `b` cover one of
/// `a` only together. `dict[int | str, int]` is below `{int: int, str: int}`,
/// each key kind under a clause of its own, and no one clause covers the
/// first's: ICFP 2023 Lemma 4.7 needs the clauses read jointly, which is the
/// set representation's to decide, and it does.
///
/// One rule, the conjunction of three readings, for every shape a keyed map
/// takes: a closed record is the rule with `da` empty, and a pure mapping the
/// rule with both field lists empty.
///
/// 1. **Every field of `b`**: one `a` declares narrows into it and is required
///    where `b` requires it; one `a` lacks is refuted where `b` requires it,
///    since a clause never requires a key, and otherwise is held to the clauses
///    of `a` whose key can spell its name.
/// 2. **Every field of `a` that `b` lacks** is covered by a clause of `b` whose
///    key holds every string.
/// 3. **Every clause of `a`** is subsumed by one clause of `b`, narrower in
///    both key and value, or asked for a key it admits that `b` reads one way
///    only ([`clause_by_key`]): a refutation where some value under that key
///    is outside `b`, a decline otherwise.
///
/// Sound throughout — a clause an oracle cannot relate is undecided rather
/// than an unsound proof, and a refutation stands on a value of the subject,
/// which the query's witness guard reads against its emptiness.
pub(super) fn keyed_map_subtype(
    fa: &[Field],
    da: &[MapClause],
    fb: &[Field],
    db: &[MapClause],
    cx: &mut Cx<'_>,
) -> Relation {
    // A cursor over each field list, so finding every name of one list in the
    // other is one forward walk of each rather than a scan per field.
    let order = Cell::new(None);
    let mut a_by_name = FieldCursor::over(fa, fb, &order);
    let mut b_by_name = FieldCursor::over(fb, fa, &order);
    {
        // One rule for every shape a keyed map takes. The closed record and the
        // pure mapping are not special cases needing a branch of their own: each
        // is this rule with one of the two lists empty, and a dedicated branch
        // for either can only answer the same question less well -- a closed
        // record read alone has no catch-all to read a field through, and would
        // leave `{"x": int}` undecided against `dict[str, int]`.
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
        // query costs the shapes with nothing to repeat more than it saves the
        // shapes with something.
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
                // A field `b` requires must be required in `a` too. A key the
                // supertype requires and the subtype does not is a value of the
                // subtype -- the one leaving that key out -- that the supertype
                // rejects, whatever the field's types say of each other: the
                // value carries no entry for them to judge. So the refutation
                // stands beside a depth the rules decline.
                Some(a_field) if b_field.required && !a_field.required => Relation::Fails,
                // Shared field: it must narrow in depth.
                Some(a_field) => match last {
                    Some((sub, sup, answer))
                        if sub == &a_field.schema && sup == &b_field.schema =>
                    {
                        answer
                    }
                    _ => {
                        let answer = a_field.schema.is_subtype_rec(&b_field.schema, cx);
                        last = Some((&a_field.schema, &b_field.schema, answer));
                        answer
                    }
                },
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
                    if clause.key.holds_every_value_of(Kind::Str, cx.oracle) {
                        // Every string key, so this clause does spell the name.
                        clause.value.is_subtype_rec(&b_field.schema, cx)
                    } else if clause.key.holds_no_value_of(Kind::Str, cx.oracle) {
                        // Asked of the key first, so a clause that never spells
                        // the name costs no goal about its value.
                        Relation::Holds
                    } else {
                        spelled_by(clause, b_field, cx)
                    }
                })),
            }
        }));
        // The three conjuncts below are one claim about one pair, so a
        // refutation from one is the answer whatever the others say, as
        // `Relation::all` reads them; the ones after it are not asked.
        if fields_ok == Relation::Fails {
            return Relation::Fails;
        }
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
        //
        // A clause keyed by constants reads no key but those exact constants,
        // so the witness takes a `str` subclass's instance equal to the name
        // as its key: the field reads it, since the walk looks the name up,
        // and a literal, which pins the type, does not. A clause holding every
        // string reads it as it reads the name, so the count above stands.
        let (covering_clauses, unread_clause) =
            db.iter()
                .fold((0usize, false), |(covering, unread), clause| {
                    if clause.key.holds_every_value_of(Kind::Str, cx.oracle) {
                        (covering + 1, unread)
                    } else {
                        let unread = unread
                            || !(clause.key.holds_no_value_of(Kind::Str, cx.oracle)
                                || finite_set(&clause.key).is_some());
                        (covering, unread)
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
                    let answer = Relation::any(
                        covering.map(|clause| a_field.schema.is_subtype_rec(&clause.value, cx)),
                    );
                    match answer {
                        Relation::Holds => Relation::Holds,
                        Relation::Fails if one_witness => {
                            Relation::of_mismatch(a_field.schema.verdict_rec(cx))
                        }
                        _ => Relation::Unknown,
                    }
                }),
        );
        // Every catch-all clause of `a` (governing its non-field keys) is subsumed
        // by a clause of `b` with both key and value narrower.
        if extra_covered == Relation::Fails {
            return Relation::Fails;
        }
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
                if !mine.key.is_subtype_rec(&theirs.key, cx).holds() {
                    return false;
                }
                let answer = mine.value.is_subtype_rec(&theirs.value, cx);
                if !answer.holds() {
                    values_asked.push((&theirs.value, answer));
                }
                answer.holds()
            });
            if subsumed {
                Relation::Holds
            } else {
                clause_by_key(mine, fa, fb, db, &mut values_asked, cx)
            }
        }));
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
/// budget's argument counts goals in. **A kind is read whole or not at all**,
/// because a whole kind holds the plain key the rule names and a part of one
/// need not: `tuple[list[int]]` and a class whose own instances are unhashable
/// each hold a key only through a subclass that hashes. `{U: int} <= {str:
/// int}` is refuted by `{U2(): 1}` for such a subclass and by no plain value,
/// so the pair declines.
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
fn clause_by_key<'b>(
    mine: &MapClause,
    fa: &[Field],
    fb: &'b [Field],
    db: &'b [MapClause],
    values_asked: &mut Vec<(&'b Schema, Relation)>,
    cx: &mut Cx<'_>,
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
        if read_one_way
            && entered(mine, reader, &mut inhabited, values_asked, cx) == Relation::Fails
        {
            return Relation::Fails;
        }
    }
    // The constants a literal key names, each a key of its own: `a` admits it
    // through `mine` unless a field of `a` reads it, and `b` reads it through
    // its field of that name or through the clauses whose key holds it -- a
    // literal naming it, or the whole kind. One reader or none, as for a kind.
    // A key naming constants and nothing else holds no key but those, so it is
    // covered where each lands inside its reader: exact, since a literal pins
    // the key's type and the reader reads that very key.
    //
    // A key names a constant only through a literal, alone or in a union, so a
    // key of kinds -- the common one -- is not read into the partition again.
    let literal = |key: &Schema| matches!(key, Schema::Literal(_));
    let names_constants = literal(&mine.key)
        || matches!(&mine.key, Schema::Union(members) if members.iter().any(literal));
    if !names_constants {
        return Relation::Unknown;
    }
    by_constants(mine, fa, fb, db, &mut inhabited, values_asked, cx)
}

/// The constants `mine`'s key names, each read as a key of its own: refuted by
/// one whose entry escapes the supertype, and proved where the key names
/// constants alone and each lands inside its reader.
///
/// Out of line, behind the discriminant test its caller makes, so a key of
/// kinds pays nothing for it.
#[inline(never)]
fn by_constants<'b>(
    mine: &MapClause,
    fa: &[Field],
    fb: &'b [Field],
    db: &'b [MapClause],
    inhabited: &mut Option<Relation>,
    values_asked: &mut Vec<(&'b Schema, Relation)>,
    cx: &mut Cx<'_>,
) -> Relation {
    let Some((labels, parts)) = key_cover(&mine.key, cx.oracle) else {
        return Relation::Unknown;
    };
    let mut covered = parts.is_empty();
    for label in labels {
        let named = |field: &&Field| Label::str(&field.name) == label;
        if fa.iter().any(|field| named(&field)) {
            continue;
        }
        let (reader, read_one_way) = match fb.iter().find(named) {
            Some(field) => (Some(&field.schema), true),
            None => read_once(&label, db, cx),
        };
        if !read_one_way {
            covered = false;
            continue;
        }
        match entered(mine, reader, inhabited, values_asked, cx) {
            Relation::Fails => return Relation::Fails,
            Relation::Unknown => covered = false,
            Relation::Holds => {}
        }
    }
    if covered {
        Relation::Holds
    } else {
        Relation::Unknown
    }
}

/// The clause of `db` reading the key `label` names, or none, and whether it
/// is read one way: `false` where two clauses read it or one cannot be placed.
fn read_once<'b>(label: &Label, db: &'b [MapClause], cx: &Cx<'_>) -> (Option<&'b Schema>, bool) {
    let mut reader = None;
    for theirs in db {
        let Some((labels, parts)) = key_cover(&theirs.key, cx.oracle) else {
            return (None, false);
        };
        if !labels.contains(label) && !parts.contains(&Some(label.kind())) {
            continue;
        }
        if reader.replace(&theirs.value).is_some() {
            return (None, false);
        }
    }
    (reader, true)
}

/// Whether the entries `mine` admits under a key `reader` reads land inside the
/// supertype: refuted by a value of `mine`'s where nothing reads the key, or by
/// one `reader` forbids. The two questions repeat across keys, so each is asked
/// once.
fn entered<'b>(
    mine: &MapClause,
    reader: Option<&'b Schema>,
    inhabited: &mut Option<Relation>,
    values_asked: &mut Vec<(&'b Schema, Relation)>,
    cx: &mut Cx<'_>,
) -> Relation {
    match reader {
        None => *inhabited.get_or_insert_with(|| Relation::of_mismatch(mine.value.verdict_rec(cx))),
        Some(value) => {
            if let Some((_, answer)) = values_asked
                .iter()
                .find(|(asked, _)| core::ptr::eq(*asked, value))
            {
                *answer
            } else {
                let answer = mine.value.is_subtype_rec(value, cx);
                values_asked.push((value, answer));
                answer
            }
        }
    }
}

/// The field `field` of `b` against a clause of `a` whose key holds some strings
/// and not every one.
///
/// A key naming constants spells the name exactly where one of them is it: a
/// literal pins the type, and the field reads the key the name is. A key the
/// rules cannot place -- a class, a narrower tuple -- might admit it, so a proof
/// carries and a refutation does not. Out of line, since the loop over `b`'s
/// fields it is asked from is the hot part of comparing two records and this
/// case is the rare one.
#[inline(never)]
fn spelled_by(clause: &MapClause, field: &Field, cx: &mut Cx<'_>) -> Relation {
    let spelled = key_cover(&clause.key, cx.oracle)
        .filter(|(_, parts)| !parts.contains(&Some(Kind::Str)))
        .map(|(labels, _)| labels.contains(&Label::str(&field.name)));
    match spelled {
        Some(true) => clause.value.is_subtype_rec(&field.schema, cx),
        Some(false) => Relation::Holds,
        None => clause.value.is_subtype_rec(&field.schema, cx).proof_only(),
    }
}

/// Whether the keyed map `schema` is below the union of `members`, split
/// across the keyed maps among them: ICFP 2023's backtrack-free `Φ`, formula
/// (17), with the defaults read part by part as (22) reads them.
///
/// A record whose tag takes three values lands in no single branch of a union
/// of three records that each fix the tag, and [`keyed_map_subtype`] compares
/// one map with one map, so this is the rule that sees it. It is
/// [`product_subtype`](super::products::product_subtype) one component over:
/// labels in place of positions, and the defaults as the guard (22) puts on
/// each branch.
///
/// ```text
/// Φ(R, [])       = R is empty
/// Φ(R, [B, ..N]) = if def(R) ≤ def(B) on every part
///                  then for every label ℓ:  R(ℓ) ≤ B(ℓ)  or  Φ(R with ℓ := R(ℓ) ∖ B(ℓ), N)
///                  else Φ(R, N)
/// ```
///
/// A label is read in `T⊥`, where an optional field holds absence as a value
/// (`Slot`), and the labels are those the record and the branch name: a label
/// both read through their defaults is below on the guard the branch was
/// entered on, so the paper's union over every map adds nothing.
///
/// **A branch is set aside exactly where some label shares no value**, as in
/// the product rule: narrowing by it leaves the record as it was. Without the
/// drop a record below its sixteen corners narrows by every corner at every
/// label.
///
/// **The else-branch refutes only through an infinite part.** Theorem 4.5
/// rests on Lemma 4.4, whose proof adds to a witness a fresh key of the part
/// whose default escapes the branch, and the paper says the hypothesis that
/// every key type is infinite "is actually used". `None` has one key and
/// `bool` two, so a branch set aside on those parts alone keeps the proof the
/// rest give and drops their refutation. Every other part, the one for keys of
/// no listed kind included, has a key no label takes.
///
/// A member that is no map this reads is set aside, and a refutation stands
/// only where each such member shares no kind with the subject, which is
/// [`Schema::disjoint_with`]'s two discriminants.
///
/// The rule reads the clauses the way the set representation does
/// ([`key_cover`]), so the two deciders agree on which key falls in which part.
/// A clause naming a constant, holding part of a kind, or reading a part
/// another clause reads declines, as [`clause_by_key`] declines it.
pub(super) fn keyed_map_splits_across_union(
    schema: &Schema,
    members: &[Schema],
    cx: &mut Cx<'_>,
) -> Relation {
    let Schema::KeyedMap { fields, defaults } = schema else {
        return Relation::Unknown;
    };
    // One branch is the pair the union rule asked already, and most unions
    // carry no second map, so this costs them one walk of the members.
    if members
        .iter()
        .filter(|member| matches!(member, Schema::KeyedMap { .. }))
        .nth(1)
        .is_none()
    {
        return Relation::Unknown;
    }
    split_across(fields, defaults, members.iter(), schema, cx)
}

/// The meet `R ∧ ¬R₁ ∧ … ∧ ¬Rₙ` of one keyed map with the complements of
/// keyed maps: empty exactly where `R` is below their union, which is
/// [`keyed_map_splits_across_union`] read as emptiness.
///
/// Without it the two routes to one question part: `a <= b` would decide
/// where `a & ~b` is left to the set representation, whose difference spends
/// its allowance on the chain this decides one complement at a time.
/// `Unknown` for any other meet, and asked only where the meet's own rules
/// have declined.
///
/// **A complement is read here only where its inner denotes a set**, the
/// guard the complement law takes ([`denotes_a_set_within`]). The rule
/// relates the subject to each removed map through the subtyping rules, which
/// prove a predicate below itself by identity, so `R ∧ ¬R` would read as
/// empty for an `R` carrying one -- and a predicate may answer its two
/// occurrences apart, which is why that law declines it.
pub(super) fn keyed_map_minus_keyed_maps(members: &[Schema], cx: &mut Cx<'_>) -> Verdict {
    // The shape first, which reads discriminants alone; the walk the set guard
    // takes is paid only by a meet of this shape.
    let mut subject = None;
    let mut removed = 0usize;
    for member in members {
        match member {
            Schema::KeyedMap { .. } if subject.is_none() => subject = Some(member),
            Schema::Complement(_) => removed += 1,
            _ => return Verdict::Unknown,
        }
    }
    let Some(schema @ Schema::KeyedMap { fields, defaults }) = subject else {
        return Verdict::Unknown;
    };
    let sets = members.iter().all(|member| match member {
        Schema::Complement(inner) => denotes_a_set_within(inner, cx.oracle, cx.defs),
        _ => true,
    });
    if removed == 0 || !sets {
        return Verdict::Unknown;
    }
    let removed = members.iter().flat_map(|member| match member {
        Schema::Complement(inner) => match &**inner {
            Schema::Union(branches) => branches.iter(),
            other => core::slice::from_ref(other).iter(),
        },
        _ => [].iter(),
    });
    match split_across(fields, defaults, removed, schema, cx) {
        Relation::Holds => Verdict::Empty,
        Relation::Fails => Verdict::Inhabited,
        Relation::Unknown => Verdict::Unknown,
    }
}

/// The two readings' shared body: the subject's fields and clauses, the
/// members of the union it is asked below, and `schema` itself, which a member
/// set aside is held disjoint from.
fn split_across<'s>(
    fields: &'s [Field],
    defaults: &'s [MapClause],
    members: impl Iterator<Item = &'s Schema>,
    schema: &Schema,
    cx: &mut Cx<'_>,
) -> Relation {
    let Some(parts) = parts_of(defaults, cx) else {
        return Relation::Unknown;
    };
    let Some(subject) = Labels::of(fields, names_read(&parts)) else {
        return Relation::Unknown;
    };
    let mut branches = Vec::new();
    let mut set_aside_hold_none = true;
    for member in members {
        let read = match member {
            Schema::KeyedMap { fields, defaults } => parts_of(defaults, cx).and_then(|theirs| {
                Some(Branch {
                    labels: Labels::of(fields, names_read(&theirs))?,
                    defaults: defaults_against(&parts, &theirs, cx),
                })
            }),
            _ => None,
        };
        match read {
            Some(branch) => branches.push(branch),
            None => set_aside_hold_none &= schema.disjoint_with(member, cx.oracle),
        }
    }
    if branches.is_empty() {
        return Relation::Unknown;
    }
    let answer = phi(&subject, &branches, cx);
    if set_aside_hold_none {
        answer
    } else {
        answer.proof_only()
    }
}

/// What a key of the `str` part no field names reads: the part a record's
/// field names fall in.
fn names_read<'s>(parts: &[Option<&'s Schema>; PARTS]) -> Option<&'s Schema> {
    key_slot(Some(Kind::Str))
        .and_then(|part| parts.get(part))
        .copied()
        .flatten()
}

/// Whether `part` holds finitely many keys -- `None`'s one and `bool`'s two --
/// where the else-branch of `Φ` keeps only its proof.
fn finite(part: usize) -> bool {
    [Kind::NoneType, Kind::Bool]
        .into_iter()
        .any(|kind| key_slot(Some(kind)) == Some(part))
}

/// What each part of the key partition maps to under `defaults`: the value of
/// the one clause holding every key of it, or `None` where no clause holds one
/// and a key there is absent. `None` overall where a clause names a constant
/// or holds part of a kind, or two clauses read one part.
fn parts_of<'s>(defaults: &'s [MapClause], cx: &Cx<'_>) -> Option<[Option<&'s Schema>; PARTS]> {
    let mut parts = [None; PARTS];
    for clause in defaults {
        let (labels, covered) = key_cover(&clause.key, cx.oracle)?;
        if !labels.is_empty() {
            return None;
        }
        for part in covered {
            let slot = parts.get_mut(key_slot(part)?)?;
            if slot.replace(&clause.value).is_some() {
                return None;
            }
        }
    }
    Some(parts)
}

/// How a subject's defaults stand against a branch's, which is the condition
/// (22) branches on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Defaults {
    /// Every part's default is within the branch's: split across its labels.
    Within,
    /// Some part's default holds a value the branch's does not, so a value of
    /// the subject escapes the branch under a key no label names. `exact` where
    /// one such part is infinite, which is where a refutation stands.
    Exceed { exact: bool },
    /// Neither is proved.
    Unknown,
}

/// Read once per branch, since narrowing a label never moves a default.
fn defaults_against(
    mine: &[Option<&Schema>; PARTS],
    theirs: &[Option<&Schema>; PARTS],
    cx: &mut Cx<'_>,
) -> Defaults {
    let mut exceeds: Option<bool> = None;
    let mut declined = false;
    for (part, (mine, theirs)) in mine.iter().zip(theirs).enumerate() {
        let Some(mine) = mine else {
            continue;
        };
        let answer = match theirs {
            Some(theirs) => mine.is_subtype_rec(theirs, cx),
            // The branch admits no key here, so the default escapes it exactly
            // where it holds a value.
            None => Relation::of_mismatch(mine.verdict_rec(cx)),
        };
        match answer {
            Relation::Holds => {}
            Relation::Fails => {
                let exact = !finite(part);
                exceeds = Some(exceeds.unwrap_or(false) || exact);
            }
            Relation::Unknown => declined = true,
        }
    }
    match exceeds {
        Some(exact) => Defaults::Exceed { exact },
        None if declined => Defaults::Unknown,
        None => Defaults::Within,
    }
}

/// One label as `Φ` reads it: the values its key may hold, and whether the key
/// may be absent -- ICFP's `T⊥`, where an optional field holds `⊥`.
#[derive(Clone, Debug)]
struct Slot {
    ty: Schema,
    absent: bool,
}

impl Slot {
    /// `self ≤ theirs` in `T⊥`. Absence the other slot does not allow is a
    /// value outside it, whatever the types say of each other.
    fn below(&self, theirs: &Slot, cx: &mut Cx<'_>) -> Relation {
        if self.absent && !theirs.absent {
            return Relation::Fails;
        }
        self.ty.is_subtype_rec(&theirs.ty, cx)
    }

    /// `self ∖ theirs` in `T⊥`: absent where this allows it and that does not.
    fn without(&self, theirs: &Slot) -> Slot {
        Slot {
            ty: Schema::meet([self.ty.clone(), theirs.ty.clone().complement()]),
            absent: self.absent && !theirs.absent,
        }
    }

    /// Whether the two slots share no value, absence included.
    fn apart(&self, theirs: &Slot, cx: &mut Cx<'_>) -> bool {
        if self.absent && theirs.absent {
            return false;
        }
        self.ty.shares_no_value_with(&theirs.ty, cx)
    }
}

/// A keyed map's labels in name order, and what a key of the `str` part no
/// label names reads: the clause there, or absence.
#[derive(Clone, Debug)]
struct Labels<'s> {
    named: Vec<(&'s str, Slot)>,
    rest: Option<&'s Schema>,
}

impl<'s> Labels<'s> {
    /// `None` where two fields share a name, which the frontend never builds and
    /// a label cannot hold twice.
    fn of(fields: &'s [Field], rest: Option<&'s Schema>) -> Option<Self> {
        let mut named: Vec<(&'s str, Slot)> = fields
            .iter()
            .map(|field| {
                let slot = Slot {
                    ty: field.schema.clone(),
                    absent: !field.required,
                };
                (&*field.name, slot)
            })
            .collect();
        named.sort_by(|a, b| a.0.cmp(b.0));
        if named
            .windows(2)
            .any(|pair| matches!(pair, [(a, _), (b, _)] if a == b))
        {
            return None;
        }
        Some(Self { named, rest })
    }

    /// The slot a key of this name reads.
    fn slot(&self, name: &str) -> Slot {
        let named = self
            .named
            .binary_search_by(|(label, _)| (*label).cmp(name))
            .ok()
            .and_then(|at| self.named.get(at));
        match named {
            Some((_, slot)) => slot.clone(),
            None => Slot {
                ty: self.rest.map_or(Schema::Nothing, Schema::clone),
                absent: true,
            },
        }
    }

    /// The names either map labels, in order and once.
    fn names_with(&self, other: &Labels<'s>) -> Vec<&'s str> {
        let mut names: Vec<&'s str> = self
            .named
            .iter()
            .chain(&other.named)
            .map(|(name, _)| *name)
            .collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// The same map with `name` read as `slot`.
    fn with(&self, name: &'s str, slot: Slot) -> Labels<'s> {
        let mut named = self.named.clone();
        match named.binary_search_by(|(label, _)| (*label).cmp(name)) {
            Ok(at) => {
                if let Some(entry) = named.get_mut(at) {
                    entry.1 = slot;
                }
            }
            Err(at) => named.insert(at, (name, slot)),
        }
        Labels {
            named,
            rest: self.rest,
        }
    }

    /// `Φ(R, [])`: refuted where every key the map requires holds a value,
    /// proved where one holds none.
    fn base_case(&self, cx: &mut Cx<'_>) -> Relation {
        let mut verdict = Verdict::Inhabited;
        for (_, slot) in &self.named {
            if slot.absent {
                continue;
            }
            match slot.ty.verdict_rec(cx) {
                Verdict::Empty => return Relation::Holds,
                Verdict::Unknown => verdict = Verdict::Unknown,
                Verdict::Inhabited => {}
            }
        }
        Relation::of_mismatch(verdict)
    }
}

/// A map of the union, read for `Φ`.
struct Branch<'s> {
    labels: Labels<'s>,
    defaults: Defaults,
}

/// `Φ(subject, branches)` in three values.
///
/// Each conjunct a label asks is the paper's: covered where the label is
/// below the branch's or where the record narrowed there is below the rest.
/// It refutes where both refute, since the label's refutation stands on a
/// value the narrowed record holds and the rest's on that record having one
/// outside them, and the first conjunct that does not hold is the answer.
/// Each branch narrowed by is a level of the query's depth, as in the product
/// rule.
fn phi(subject: &Labels<'_>, branches: &[Branch<'_>], cx: &mut Cx<'_>) -> Relation {
    if !spend(cx.budget) {
        return Relation::Unknown;
    }
    let mut refutes = true;
    let mut remaining = branches;
    let (branch, rest) = loop {
        let Some((branch, rest)) = remaining.split_first() else {
            let answer = subject.base_case(cx);
            return if refutes { answer } else { answer.proof_only() };
        };
        remaining = rest;
        match branch.defaults {
            Defaults::Within => {
                let apart = subject
                    .names_with(&branch.labels)
                    .into_iter()
                    .any(|name| subject.slot(name).apart(&branch.labels.slot(name), cx));
                if !apart {
                    break (branch, rest);
                }
            }
            Defaults::Exceed { exact } => refutes &= exact,
            Defaults::Unknown => refutes = false,
        }
    };
    let answer = 'labels: {
        for name in subject.names_with(&branch.labels) {
            let mine = subject.slot(name);
            let theirs = branch.labels.slot(name);
            let here = mine.below(&theirs, cx);
            if here == Relation::Holds {
                continue;
            }
            let narrowed = mine.without(&theirs);
            // A required key narrowed to no value empties the record, which
            // every union covers; asked here, for the one slot that moved,
            // rather than of every slot at every level.
            let covered = if !narrowed.absent && narrowed.ty.verdict_rec(cx).is_empty() {
                Relation::Holds
            } else {
                let narrowed = subject.with(name, narrowed);
                descending(cx.budget, Relation::Unknown, || phi(&narrowed, rest, cx))
            };
            match (here, covered) {
                (_, Relation::Holds) => {}
                (Relation::Fails, Relation::Fails) => break 'labels Relation::Fails,
                _ => break 'labels Relation::Unknown,
            }
        }
        Relation::Holds
    };
    if refutes { answer } else { answer.proof_only() }
}

/// Whether the attribute record `fa` is a subtype of `fb`: width and depth.
///
/// Every attribute the supertype declares must be one the subtype declares, and
/// no less narrowly. There is no class in it -- a record denotes every value
/// carrying its attributes, whatever the value is -- so a nominal question is a
/// conjunct of its own, and two records read off unrelated classes relate.
///
/// The rule is set inclusion read off the denotation: an attribute the supertype
/// does not name constrains nothing, and one it names constrains every value of
/// the subtype exactly when the subtype names it too, at least as narrowly.
pub(super) fn attr_record_subtype(fa: &[Attribute], fb: &[Attribute], cx: &mut Cx<'_>) -> Relation {
    let order = Cell::new(None);
    let mut a_by_name = FieldCursor::over(fa, fb, &order);
    Relation::all(fb.iter().map(|b| {
        match a_by_name.named(&b.name) {
            // An attribute the supertype names and the subtype does not: the
            // subtype holds values without it, and those are outside the
            // supertype.
            None => Relation::Fails,
            Some(a) => a.schema.is_subtype_rec(&b.schema, cx),
        }
    }))
}

/// What a cursor finds by name: a record's field, or an attribute record's
/// attribute. The two lists are kept in name order alike and looked up alike;
/// what differs is what the caller reads off the entry it found.
pub(super) trait Named {
    /// The entry's name.
    fn name(&self) -> &str;
}

impl Named for Field {
    fn name(&self) -> &str {
        &self.name
    }
}

impl Named for Attribute {
    fn name(&self) -> &str {
        &self.name
    }
}

/// Whether a field list is in name order, which is what lets a cursor over it
/// move forward only.
pub(super) fn sorted_by_name<T: Named>(list: &[T]) -> bool {
    list.is_sorted_by(|one, two| one.name() <= two.name())
}

/// One walk over a field list, finding each name of another list as it passes.
///
/// Both lists are in name order, so a partner is found by moving forward: no
/// allocation, and one string compare per field passed. A name table would be
/// built and freed once per side per comparison, which is a fifth of the
/// repeating workload.
///
/// **Unique names are the caller's invariant**, and the order is what makes it
/// load-bearing here: two fields of one name would let the cursor answer with
/// whichever it reached first, and the `required` and width readings that
/// consume the answer would decide on a field the other shadows. The frontend
/// refuses a duplicate; the `debug_assert` names the dependency and catches a
/// malformed IR in debug rather than deciding on the wrong field.
pub(super) struct FieldCursor<'a, 'o, T> {
    fields: &'a [T],
    rest: &'a [T],
    /// The list this cursor is asked the names of, kept so the order can be
    /// read later rather than now.
    queries: &'a [T],
    /// Whether both lists are in name order, once that has been asked --
    /// shared with the sibling cursor over the same pair, which asks the
    /// identical question. `None` until a lookup misses, because until then
    /// nothing depends on the answer.
    order: &'o Cell<Option<bool>>,
}

impl<'a, 'o, T: Named> FieldCursor<'a, 'o, T> {
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
        fields: &'a [T],
        queries: &'a [T],
        order: &'o Cell<Option<bool>>,
    ) -> FieldCursor<'a, 'o, T> {
        debug_assert!(
            !sorted_by_name(fields)
                || fields
                    .windows(2)
                    .filter_map(|pair| Some((pair.first()?, pair.get(1)?)))
                    .all(|(one, two)| one.name() != two.name()),
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
    fn named(&mut self, name: &str) -> Option<&'a T> {
        if self.order.get() == Some(false) {
            return self.fields.iter().find(|field| field.name() == name);
        }
        while let Some((first, rest)) = self.rest.split_first() {
            match first.name().cmp(name) {
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
    fn missed(&mut self, name: &str) -> Option<&'a T> {
        if self.order.get().is_some() {
            return None;
        }
        let ordered = sorted_by_name(self.fields) && sorted_by_name(self.queries);
        self.order.set(Some(ordered));
        if ordered {
            return None;
        }
        self.rest = self.fields;
        self.fields.iter().find(|field| field.name() == name)
    }
}
