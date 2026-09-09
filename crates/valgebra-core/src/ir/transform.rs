//! Rewriting a schema into another schema.
//!
//! Everything here takes a node and gives back a node with one thing changed:
//! an index moved to another validator's pool, a definition renumbered, a
//! reference unfolded once, a record opened, a definitions table pruned of what
//! nothing reaches. None of it decides anything about a value -- the shapes and
//! the questions asked of them are the parent module's -- and none of it edits
//! in place: a rewrite that changes nothing gives back `None` (or the node it
//! was handed), which is what lets an unchanged subtree keep its `Arc` instead
//! of being rebuilt beside itself.
//!
//! The child set of each variant is declared once, in [`Schema::map_children`],
//! and every rewrite below is written against it rather than against the
//! variants a second time.

use std::sync::Arc;

use super::{
    Clauses, Constraint, Constraints, DefIx, DefShift, Fields, MapClause, Members, Openness,
    OperandIx, PoolShift, Remap, Schema, SeqShape, already_said, clauses_for, with_field_buffer,
    with_member_buffer,
};

/// Move a constraint list's indices, rebuilding it only where one moves.
fn remapped_constraints(constraints: &Constraints, remap: Remap<'_>) -> Option<Constraints> {
    let mut rebuilt: Option<Vec<Constraint>> = None;
    for (at, constraint) in constraints.iter().enumerate() {
        match constraint.remapped_where_moved(remap) {
            None => {
                if let Some(buffer) = rebuilt.as_mut() {
                    buffer.push(constraint.clone());
                }
            }
            Some(moved) => {
                let buffer = rebuilt.get_or_insert_with(|| {
                    constraints.iter().take(at).cloned().collect::<Vec<_>>()
                });
                buffer.push(moved);
            }
        }
    }
    rebuilt.map(|buffer| Constraints::from(&buffer[..]))
}

/// Map a member list, rebuilding it only where `f` changed a member.
///
/// Nothing is allocated until a member actually changes: a list `f` has nothing
/// to say about costs one call per member and no memory at all, which is the
/// whole point of asking whether it changed.
fn mapped_members(members: &Members, f: &impl Fn(&Schema) -> Option<Schema>) -> Option<Members> {
    with_member_buffer(|buffer| {
        let mut changed = false;
        for (at, member) in members.iter().enumerate() {
            match f(member) {
                None => {
                    if changed {
                        buffer.push(member.clone());
                    }
                }
                Some(mapped) => {
                    if !changed {
                        buffer.extend(members.iter().take(at).cloned());
                        changed = true;
                    }
                    buffer.push(mapped);
                }
            }
        }
        // Drained rather than copied: the finished list is moved into the
        // node's slice, so a member does not pay a clone -- which is a
        // reference count on every handle it carries -- to be put where it
        // was going anyway.
        changed.then(|| buffer.drain(..).collect())
    })
}

/// Map a field list, rebuilding it only where `f` changed a field's schema.
///
/// `f` is asked once per field, and the list is allocated only once a field has
/// answered: asking twice -- once to find out and once to build -- would run a
/// recursive transform twice per level, which is exponential in the depth.
fn mapped_fields(fields: &Fields, f: &impl Fn(&Schema) -> Option<Schema>) -> Option<Fields> {
    with_field_buffer(|buffer| {
        let mut changed = false;
        for (at, field) in fields.iter().enumerate() {
            match f(&field.schema) {
                None => {
                    if changed {
                        buffer.push(field.clone());
                    }
                }
                Some(schema) => {
                    if !changed {
                        buffer.extend(fields.iter().take(at).cloned());
                        changed = true;
                    }
                    buffer.push(field.with_schema(schema));
                }
            }
        }
        changed.then(|| buffer.drain(..).collect())
    })
}

/// Map a clause list, rebuilding it only where `f` changed a key or a value.
///
/// Asked once per key and once per value, for the same reason as
/// [`mapped_fields`].
fn mapped_clauses(clauses: &Clauses, f: &impl Fn(&Schema) -> Option<Schema>) -> Option<Clauses> {
    let mut rebuilt: Option<Vec<MapClause>> = None;
    for (at, clause) in clauses.iter().enumerate() {
        let key = f(&clause.key);
        let value = f(&clause.value);
        if key.is_none() && value.is_none() {
            if let Some(buffer) = rebuilt.as_mut() {
                buffer.push(clause.clone());
            }
            continue;
        }
        let buffer =
            rebuilt.get_or_insert_with(|| clauses.iter().take(at).cloned().collect::<Vec<_>>());
        buffer.push(MapClause::of(
            key.unwrap_or_else(|| clause.key.clone()),
            value.unwrap_or_else(|| clause.value.clone()),
        ));
    }
    rebuilt.map(|buffer| Clauses::from(&buffer[..]))
}

impl SeqShape {
    /// The same, rebuilding only what `f` changed, and `None` when it changed
    /// nothing: the sequence half of [`Schema::mapped_children`].
    pub(crate) fn mapped_elems(&self, f: &impl Fn(&Schema) -> Option<Schema>) -> Option<SeqShape> {
        let prefix = mapped_members(&self.prefix, f);
        let tail = self.tail.as_deref().map(f);
        match (&prefix, &tail) {
            (None, None | Some(None)) => None,
            _ => Some(SeqShape {
                prefix: prefix.unwrap_or_else(|| self.prefix.clone()),
                tail: match (tail, &self.tail) {
                    (Some(Some(mapped)), _) => Some(Arc::new(mapped)),
                    (_, held) => held.clone(),
                },
            }),
        }
    }
}

impl Schema {
    /// This schema with its references unfolded, and the cut replaced by the
    /// bound the position makes sound.
    ///
    /// A set representation holds no cycle, so a recursive schema could not be
    /// lowered at all and every relation over one was left to the structural
    /// rules -- including relations a single unfolding settles, like a meet with
    /// a kind the body never admits. Unfolding `unfolds` times and putting a
    /// *bound* where the reference would have been gives a schema the descriptor
    /// can hold and an answer that still holds of the original, provided the
    /// bound is chosen for the position.
    ///
    /// `positive` says which. In a positive position the top is the cut, so the
    /// result denotes a superset: proving *that* empty proves the original
    /// empty. In a negative one -- under a complement -- the bottom is the cut
    /// and the result denotes a subset, which is what keeps a difference sound.
    /// `Complement` is the only node that flips the polarity: every other one is
    /// monotone in what it holds, a map clause included, since widening a clause
    /// only makes more dicts covered by it.
    ///
    /// A schema with no reference is returned as it stands, so the caller pays
    /// nothing for the common case.
    #[must_use]
    pub fn unfolded(&self, definitions: &[Schema], unfolds: u32, positive: bool) -> Schema {
        let cut = || {
            if positive {
                Schema::ANYTHING
            } else {
                Schema::Nothing
            }
        };
        match self {
            Schema::Ref(index) => match definitions.get(index.get()) {
                Some(body) if unfolds > 0 => body.unfolded(definitions, unfolds - 1, positive),
                // Out of unfoldings, or a reference to a definition this caller
                // does not hold: the bound stands in for what is not read.
                _ => cut(),
            },
            // A marker for a definition still being built. Nothing can be read
            // from it, so the bound stands in for it too.
            Schema::SelfRef(_) => cut(),
            Schema::Complement(inner) => {
                Schema::Complement(Arc::new(inner.unfolded(definitions, unfolds, !positive)))
            }
            other => other.map_children(&|child| child.unfolded(definitions, unfolds, positive)),
        }
    }

    /// This schema with every definition index rewritten by `table`.
    ///
    /// An index the table drops cannot be reached from a schema this is called
    /// on -- that is what made it droppable -- so it is left as it stands rather
    /// than guessed at.
    fn renumbered(&self, table: &[Option<DefIx>]) -> Schema {
        if let Schema::Ref(index) = self {
            return match table.get(index.get()).copied().flatten() {
                Some(moved) => Schema::Ref(moved),
                None => self.clone(),
            };
        }
        self.map_children(&|child| child.renumbered(table))
    }

    /// Rebuild this node with every child schema mapped through `f`, leaving the
    /// node's own payloads -- the container kind, a field's name and
    /// required-ness, a pooled index, a constraint -- exactly as they are.
    ///
    /// **This is the one place the child set of each variant is written down.** A
    /// walk that only descends -- moving indices, resolving a self-reference --
    /// used to spell the whole descent out per pass, and the compiler forced an
    /// arm without being able to check the arm recursed into everything: a
    /// forgotten child was a silent stale subtree. Written once, every such pass
    /// inherits the child set.
    ///
    /// A new variant carrying a child schema must map it here, or every pass
    /// built on this drops it. A new variant carrying a *pooled index* must also
    /// be handled in [`remapped_by`](Self::remapped_by), which is why that match
    /// takes no wildcard.
    pub(crate) fn map_children(&self, f: &impl Fn(&Schema) -> Schema) -> Schema {
        self.mapped_children(&|child| Some(f(child)))
            .unwrap_or_else(|| self.clone())
    }

    /// The same descent, rebuilding only what `f` actually changed.
    ///
    /// `f` answers `None` for a child it leaves alone, and this answers `None`
    /// for a node every child of which came back that way -- so a pass over a
    /// subtree it has nothing to say about allocates nothing and the caller
    /// keeps the handle it already had. The pass that has something to say
    /// about every node pays one comparison per child for the privilege, which
    /// is the trade the shared representation makes worth taking: what it
    /// saves is a rebuild of everything beneath the node, and what it costs is
    /// an `Option` the optimiser sees through.
    ///
    /// [`map_children`](Self::map_children) is this with `f` that always
    /// answers `Some`, so the child set is still written down once.
    pub(crate) fn mapped_children(&self, f: &impl Fn(&Schema) -> Option<Schema>) -> Option<Schema> {
        match self {
            Schema::Anything(_)
            | Schema::Nothing
            | Schema::NoneType
            | Schema::Bool
            | Schema::Int
            | Schema::Float
            | Schema::Str
            | Schema::Bytes
            | Schema::Literal(_)
            | Schema::Instance(_)
            | Schema::Ref(_)
            | Schema::SelfRef(_) => None,
            Schema::Seq { container, shape } => shape.mapped_elems(f).map(|shape| Schema::Seq {
                container: *container,
                shape,
            }),
            Schema::Coll { container, element } => f(element).map(|element| Schema::Coll {
                container: *container,
                element: Arc::new(element),
            }),
            Schema::Complement(inner) => f(inner).map(|inner| Schema::Complement(Arc::new(inner))),
            Schema::Union(members) => mapped_members(members, f).map(Schema::Union),
            Schema::Intersection(members) => mapped_members(members, f).map(Schema::Intersection),
            Schema::KeyedMap { fields, defaults } => {
                let mapped_fields = mapped_fields(fields, f);
                let mapped_clauses = mapped_clauses(defaults, f);
                if mapped_fields.is_none() && mapped_clauses.is_none() {
                    return None;
                }
                Some(Schema::KeyedMap {
                    fields: mapped_fields.unwrap_or_else(|| fields.clone()),
                    defaults: mapped_clauses.unwrap_or_else(|| defaults.clone()),
                })
            }
            Schema::AttrRecord { fields } => {
                mapped_fields(fields, f).map(|fields| Schema::AttrRecord { fields })
            }
            Schema::Refine { base, constraints } => f(base).map(|base| Schema::Refine {
                base: Arc::new(base),
                constraints: constraints.clone(),
            }),
        }
    }

    /// Rebuild this schema against another validator's pools, moving every
    /// payload the way `remap` moves its index space.
    ///
    /// The arms here are exactly the nodes that *carry* an index; the structural
    /// descent is [`map_children`](Self::map_children). The match deliberately
    /// takes **no wildcard**: a variant that carries a pooled index and reaches
    /// this by a catch-all would keep its index and read a real object of the
    /// wrong kind, which is a plausible wrong verdict rather than a crash. Listing
    /// the structural variants costs a line each and makes a new variant a
    /// compile error here, where the decision belongs.
    fn remapped_by(&self, remap: Remap<'_>) -> Schema {
        self.remapped_where_moved(remap)
            .unwrap_or_else(|| self.clone())
    }

    /// The same rebuild, answering `None` where no index below this node moves.
    ///
    /// Combining two validators shifts one side's indices and leaves the
    /// other's where they are, so half of every composition was rebuilding a
    /// whole tree to arrive at the tree it started from. An index that does not
    /// move says so, and a node all of whose children said so is returned as
    /// the handle the caller already held.
    fn remapped_where_moved(&self, remap: Remap<'_>) -> Option<Schema> {
        match self {
            Schema::Literal(index) => {
                let moved = index.remapped_by(remap);
                (moved != *index).then_some(Schema::Literal(moved))
            }
            Schema::Instance(index) => {
                let moved = index.remapped_by(remap);
                (moved != *index).then_some(Schema::Instance(moved))
            }
            Schema::Ref(index) => {
                let moved = index.remapped_by(remap);
                (moved != *index).then_some(Schema::Ref(moved))
            }
            Schema::Refine { base, constraints } => {
                let moved_base = base.remapped_where_moved(remap);
                let moved_constraints = remapped_constraints(constraints, remap);
                if moved_base.is_none() && moved_constraints.is_none() {
                    return None;
                }
                Some(Schema::Refine {
                    base: moved_base.map_or_else(|| Arc::clone(base), Arc::new),
                    constraints: moved_constraints.unwrap_or_else(|| Arc::clone(constraints)),
                })
            }
            // No payload of its own: descend, and let the child set live in one
            // place. Spelled out rather than caught by `_` so a new variant with
            // an index cannot arrive here silently.
            Schema::Anything(_)
            | Schema::Nothing
            | Schema::NoneType
            | Schema::Bool
            | Schema::Int
            | Schema::Float
            | Schema::Str
            | Schema::Bytes
            | Schema::SelfRef(_)
            | Schema::Seq { .. }
            | Schema::Coll { .. }
            | Schema::Complement(_)
            | Schema::Union(_)
            | Schema::Intersection(_)
            | Schema::KeyedMap { .. }
            | Schema::AttrRecord { .. } => self.mapped_children(&|s| s.remapped_where_moved(remap)),
        }
    }

    /// Return a copy with pool indices shifted by `pool` and definition
    /// references shifted by `defs`.
    ///
    /// Used when composing two compiled validators: their constants pools and
    /// definitions tables are concatenated, so the second schema's
    /// `Literal`/`Instance`/`Refine` indices move past the first pool's length
    /// and its `Ref` indices past the first definitions' length.
    #[must_use]
    pub fn shifted(&self, pool: PoolShift, defs: DefShift) -> Schema {
        self.remapped_by(Remap::Append { pool, defs })
    }

    /// Like [`shifted`](Self::shifted), but remapping pool indices through
    /// `lit_map` (an old->new table from interning one pool into another, so
    /// identity-shared constants collapse to one index) while still offsetting
    /// definition indices by `def_offset`.
    ///
    /// The same walk as `shifted`, over the same payload sites: the two differ
    /// only in how a pool index moves, which is what `Remap` names.
    #[must_use]
    pub fn reindexed(&self, lit_map: &[usize], def_offset: DefShift) -> Schema {
        self.remapped_by(Remap::Intern {
            lit_map,
            defs: def_offset,
        })
    }

    /// Replace each `SelfRef(token)` with `Ref(ref_id)`, leaving other tokens
    /// (from enclosing `recursive` definitions) untouched.
    #[must_use]
    pub fn resolve_self(&self, token: u64, ref_id: DefIx) -> Schema {
        match self {
            Schema::SelfRef(t) if *t == token => Schema::Ref(ref_id),
            // Every other node keeps its payloads and passes the rewrite down. A
            // wildcard is right here, unlike in `remapped_by`: this pass rewrites
            // one marker and touches nothing else, so it is already correct for a
            // variant that does not exist yet -- provided that variant's children
            // are mapped in `map_children`, which is the one place to add them.
            other => other.map_children(&|s| s.resolve_self(token, ref_id)),
        }
    }

    /// Return a copy with every record-shaped [`Schema::KeyedMap`] in the tree
    /// set to `open`.
    ///
    /// This backs the `open`/`close` methods: `open` opens every record in a
    /// subtree (undeclared keys allowed via an `anything` catch-all), `close`
    /// closes them. A pure mapping keeps its clauses -- it is a map from a key
    /// *type*, not a record, and opening it would say something it does not.
    ///
    /// **The labels are read on the semantic `dom` first**, which is what makes
    /// this a function on sets. Naming a key and giving it exactly what the
    /// record already gives every key it does not name says nothing, so
    /// `{"a?": nothing}` and `{}` are one record -- and unless the redundant name
    /// is dropped they open to different ones, which would make `open` map equal
    /// sets to unequal sets and put it outside the algebra.
    #[must_use]
    pub fn with_records_open(&self, open: Openness) -> Schema {
        self.records_opened(open).unwrap_or_else(|| self.clone())
    }

    /// The same transform, answering `None` for a subtree it leaves alone.
    ///
    /// A schema with no record under it is returned to the caller as the handle
    /// it already had: opening a list of integers rebuilt every node of it to
    /// arrive back where it started, and a record whose catch-all already says
    /// what the openness asks for is in the same position.
    fn records_opened(&self, open: Openness) -> Option<Schema> {
        match self {
            // The one node this transform is about: a record replaces its
            // catch-all. Having no field does not make one a mapping -- the empty
            // *closed* record is a record, and the empty clause list is what says
            // so; a mapping has a clause and no field.
            Schema::KeyedMap { fields, defaults } if !fields.is_empty() || defaults.is_empty() => {
                // Sized from the field list rather than collected into a
                // vector that grows. A filter cannot say in advance how many
                // elements survive it, so `collect` starts a record's fields at
                // capacity zero and reallocates its way up, copying every field
                // it has built so far each time; a wide record pays that on
                // every pass over it. Dropping a field here is the rare case --
                // it happens only when a name says exactly what the catch-all
                // already says -- so the field list's own length is the right
                // guess, and over-reserving by the one or two it drops costs a
                // few unused slots and no allocation.
                let wanted = clauses_for(open);
                let dropping = fields.iter().any(|field| already_said(field, defaults));
                let opened = mapped_fields(fields, &|schema| schema.records_opened(open));
                if !dropping && opened.is_none() && *defaults == wanted {
                    return None;
                }
                let updated = opened.unwrap_or_else(|| fields.clone());
                if !dropping {
                    // The names and their order are the ones this node already
                    // had, so the list is canonical without being sorted again.
                    return Some(Schema::KeyedMap {
                        fields: updated,
                        defaults: wanted,
                    });
                }
                // A field a name says twice -- once as its own and once through
                // the catch-all the openness adds -- is dropped, and the list
                // has to be assembled again around the hole.
                Some(with_field_buffer(|kept| {
                    kept.extend(
                        updated
                            .iter()
                            .filter(|field| !already_said(field, defaults))
                            .cloned(),
                    );
                    Schema::keyed_map_from(kept, wanted)
                }))
            }
            // Every other node carries the transform to its children and keeps
            // its own payloads. Spelling the descent out here again is what let
            // it end in a wildcard, where a new child-carrying variant would be
            // cloned unopened rather than failing to compile.
            // Refolded, because this transform can *make* a shape the
            // constructors promise never survives them: opening the records in
            // `{a: int} | ~{a: int}` maps both sides to one schema beside its own
            // complement, and a rule downstream is entitled to assume no such
            // shape exists. [`map_children`] itself stays raw -- reindexing uses
            // it to relabel pool slots, and a relabelling that changed the shape
            // would not be one.
            _ => self
                .mapped_children(&|s| s.records_opened(open))
                .map(Schema::refolded),
        }
    }

    /// This node rebuilt through its own smart constructor.
    ///
    /// The children are already whatever they should be; only the node itself
    /// may have become a shape construction folds. Written as one step so a
    /// transform can descend with [`map_children`] and still leave the tree in
    /// the shape the constructors guarantee.
    fn refolded(self) -> Schema {
        match self {
            Schema::Union(members) => Schema::union(members.iter().cloned()),
            Schema::Intersection(members) => Schema::meet(members.iter().cloned()),
            Schema::Complement(inner) => Arc::unwrap_or_clone(inner).complement(),
            other => other,
        }
    }
}

impl Constraint {
    /// This constraint against another validator's pool.
    ///
    /// A length bound and a regex pattern are not pool indices and are carried
    /// through untouched. The type says so for the length: a `usize` has no
    /// `remapped_by`, so an arm that must not move an index cannot.
    fn remapped_where_moved(&self, remap: Remap<'_>) -> Option<Constraint> {
        let moved = |index: OperandIx| {
            let moved = index.remapped_by(remap);
            (moved != index).then_some(moved)
        };
        match self {
            Constraint::Ge(index) => moved(*index).map(Constraint::Ge),
            Constraint::Gt(index) => moved(*index).map(Constraint::Gt),
            Constraint::Le(index) => moved(*index).map(Constraint::Le),
            Constraint::Lt(index) => moved(*index).map(Constraint::Lt),
            // Neither carries an index, so neither can move.
            Constraint::MinLen(_) | Constraint::MaxLen(_) | Constraint::Regex(_) => None,
            Constraint::MultipleOf(index) => moved(*index).map(Constraint::MultipleOf),
            Constraint::Predicate(index) => {
                let moved = index.remapped_by(remap);
                (moved != *index).then_some(Constraint::Predicate(moved))
            }
        }
    }
}

/// A schema and the definitions it can still reach, with the rest dropped.
///
/// A fold can leave a definition behind: `json | ~json` is the top and carries
/// the fixpoint's body no reference names any more. Dead weight in itself, and
/// worse than that for equality -- the definitions are part of what two
/// validators are compared on, so the top built that way was not the top built
/// any other way. Two schemas differ when their *reachable* definitions differ,
/// which is what this makes true by leaving nothing else.
///
/// The traversal is the reachability of a graph whose edges are `Ref` nodes:
/// what the schema names, what those name, and no further.
#[must_use]
pub fn pruned(schema: Schema, definitions: Vec<Schema>) -> (Schema, Vec<Schema>) {
    let mut reachable = vec![false; definitions.len()];
    let mut pending = vec![&schema];
    while let Some(node) = pending.pop() {
        // A definition's body may name another, so reaching one puts its body on
        // the stack and the walk continues from there.
        if let Schema::Ref(index) = node
            && let Some(seen) = reachable.get_mut(index.get())
            && !*seen
        {
            *seen = true;
            if let Some(body) = definitions.get(index.get()) {
                pending.push(body);
            }
        }
        node.push_children(&mut pending);
    }
    if reachable.iter().all(|kept| *kept) {
        return (schema, definitions);
    }
    // The kept definitions keep their relative order, so a validator that loses
    // none is untouched and one that loses some reads the same way.
    let mut next = 0;
    let renumbered: Vec<Option<DefIx>> = reachable
        .iter()
        .map(|kept| {
            kept.then(|| {
                let moved = DefIx::new(next);
                next += 1;
                moved
            })
        })
        .collect();
    let bodies: Vec<Schema> = definitions
        .iter()
        .zip(&reachable)
        .filter(|(_, kept)| **kept)
        .map(|(body, _)| body.renumbered(&renumbered))
        .collect();
    (schema.renumbered(&renumbered), bodies)
}
