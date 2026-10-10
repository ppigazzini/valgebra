//! A pool the decision can look into: the constants, bounds and classes a
//! schema's indices name, drawn from the fuzzer's bytes and answered exactly.
//!
//! The `decision` target asks every question under `NoLeafRelations`, which
//! declines all of them, so every rule that turns on what a literal is, how two
//! bounds order or what a class can hold is a rule that target never reaches:
//! the literal table, the class readings, the bound and step rules, and every
//! lowering of a schema naming a constant. This module is the other half. It
//! draws a small Python world -- three operands, three constants, three classes
//! -- and answers each [`LeafRelations`] question about it as the interpreter
//! would, so the decision is asked what the bindings ask it.
//!
//! **Every answer is a fact about one world, or a decline.** The decision is
//! sound for every truthful oracle, so a law that fails here is a defect in the
//! decision or a lie in this module, and the second is what the unit tests
//! below hold: two questions about one value are answered from that value, so
//! they cannot disagree. Where Python's own answer turns on something this
//! does not model -- a hooked class's `isinstance`, a `NoneType` builtin --
//! the question is declined, as the bindings decline it.
//!
//! The numbers are small and exact: integers from -3 to 5 and floats that are
//! sums of halves, so each comparison, floor and remainder below is Python's
//! to the bit. `nan` and the infinities are left out on purpose: the frontend
//! refuses a bound against `nan`, and a remainder against an infinity is a
//! rule of its own that an exact model would have to restate.

use valgebra_core::descr::classes::{Attributes, Class, Hook, Member};
use valgebra_core::descr::lower::{Constants, Operand};
use valgebra_core::{ClassIx, ConstIx, Kind, LeafRelations, OperandIx, Schema};

use arbitrary::{Arbitrary, Result, Unstructured};

use crate::NAMES;

/// One of the three tables, each as wide as the indices the generator's
/// `small` draws, so every index it writes names an entry; one past it would
/// be declined, as the bindings decline an index past their pool.
type Table<T> = [T; 3];

/// A bound or a step, as a caller writes one in `Ge(...)` or `MultipleOf(...)`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Number {
    Int(i64),
    Float(f64),
}

/// The floats a number is drawn from: each a multiple of a half, so every
/// floor, comparison and remainder on them is exact in an `f64`.
const FLOATS: [f64; 8] = [-2.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.5, 4.0];

impl Number {
    fn draw(u: &mut Unstructured) -> Result<Number> {
        let byte = u.arbitrary::<u8>()?;
        Ok(if byte & 0x80 == 0 {
            Number::Int(i64::from(byte % 9) - 3)
        } else {
            Number::Float(FLOATS[usize::from(byte) % FLOATS.len()])
        })
    }

    /// The value as Python's mixed arithmetic reads it: an `int` against a
    /// `float` is converted, which is exact for every integer drawn here.
    #[allow(clippy::cast_precision_loss)]
    fn real(self) -> f64 {
        match self {
            Number::Int(n) => n as f64,
            Number::Float(x) => x,
        }
    }

    fn kind(self) -> Kind {
        match self {
            Number::Int(_) => Kind::Int,
            Number::Float(_) => Kind::Float,
        }
    }

    fn operand(self) -> Operand {
        match self {
            Number::Int(n) => Operand::Integer(n),
            Number::Float(x) => Operand::Float(x),
        }
    }
}

/// A constant a `Literal` names: one value of one exact builtin type.
#[derive(Debug, Clone, PartialEq)]
enum Constant {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'static str),
    Bytes(&'static [u8]),
}

impl Constant {
    /// Drawn so that the pairs Python's `==` conflates across types are
    /// reached: `0`, `False` and `0.0` are three constants and three disjoint
    /// singletons.
    fn draw(u: &mut Unstructured) -> Result<Constant> {
        let value = u.arbitrary::<u8>()?;
        Ok(match u.arbitrary::<u8>()? % 6 {
            0 => Constant::None,
            1 => Constant::Bool(value & 1 == 1),
            2 => Constant::Int(i64::from(value % 5) - 2),
            3 => Constant::Float([0.0, 1.0, -0.5, 2.5][usize::from(value) % 4]),
            4 => Constant::Str(["", "a", "ab"][usize::from(value) % 3]),
            _ => Constant::Bytes([b"".as_slice(), b"a"][usize::from(value) % 2]),
        })
    }

    /// The kind of the constant's type, which is exact: a literal pins
    /// `type(x)`, and every constant here is a builtin's own value.
    fn kind(&self) -> Kind {
        match self {
            Constant::None => Kind::NoneType,
            Constant::Bool(_) => Kind::Bool,
            Constant::Int(_) => Kind::Int,
            Constant::Float(_) => Kind::Float,
            Constant::Str(_) => Kind::Str,
            Constant::Bytes(_) => Kind::Bytes,
        }
    }

    /// Whether the constant is a member of a scalar atom. `Int` holds the
    /// booleans too, since `bool` subclasses `int`.
    fn in_kind(&self, kind: Kind) -> bool {
        self.kind() == kind || (kind == Kind::Int && self.kind() == Kind::Bool)
    }

    fn operand(&self) -> Operand {
        match self {
            Constant::None => Operand::NoneType,
            Constant::Bool(b) => Operand::Boolean(*b),
            Constant::Int(n) => Operand::Integer(*n),
            Constant::Float(x) => Operand::Float(*x),
            Constant::Str(text) => Operand::Word(text.as_bytes().to_vec(), Kind::Str),
            Constant::Bytes(raw) => Operand::Word(raw.to_vec(), Kind::Bytes),
        }
    }
}

/// What one class body defines that a record met with the class asks about.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Defines {
    Nothing,
    /// A class attribute holding a value, which an instance entry replaces.
    Value(&'static str),
    /// A property, which answers every read.
    Property(&'static str),
    Getattr,
    Getattribute,
}

/// One class of the world: where it derives from, what it lays out, and
/// whether its metaclass answers `isinstance` by running code.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ClassFacts {
    /// A class earlier in the table it derives from, which is how a table
    /// drawn in order stays acyclic.
    parent: Option<usize>,
    /// The builtin whose layout the class carries, its own or its parent's.
    builtin: Option<Kind>,
    /// A metaclass hooking `isinstance`, which every subclass inherits.
    hooked: bool,
    defines: Defines,
}

/// The builtins a class may derive from, which are the ones laying down a
/// layout: `bool` and `NoneType` cannot be subclassed.
const LAYOUTS: [Kind; 9] = [
    Kind::Int,
    Kind::Str,
    Kind::Bytes,
    Kind::Float,
    Kind::Tuple,
    Kind::FrozenSet,
    Kind::List,
    Kind::Set,
    Kind::Dict,
];

/// The id a builtin enters a class's order under: past the table's ids and
/// below the ones `Class::exact` takes, so no two classes share one.
fn builtin_id(kind: Kind) -> u32 {
    16 + kind as u32
}

/// The pool a schema's indices name, and the oracle that reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct World {
    operands: Table<Number>,
    constants: Table<Constant>,
    classes: Table<ClassFacts>,
}

impl<'a> Arbitrary<'a> for World {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let operands = [Number::draw(u)?, Number::draw(u)?, Number::draw(u)?];
        let constants = [Constant::draw(u)?, Constant::draw(u)?, Constant::draw(u)?];
        let mut classes: Table<ClassFacts> = [ClassFacts {
            parent: None,
            builtin: None,
            hooked: false,
            defines: Defines::Nothing,
        }; 3];
        for at in 0..classes.len() {
            let shape = u.arbitrary::<u8>()?;
            let body = u.arbitrary::<u8>()?;
            let parent = (at > 0 && shape & 1 == 1).then(|| usize::from(shape >> 1) % at);
            let inherited = parent.map(|p| classes[p]);
            // A class carrying a layout keeps it: Python refuses a second
            // builtin base whose layout does not extend the first. One that
            // carries none may take one, `class D(Plain, int)`.
            let builtin = inherited.and_then(|p| p.builtin).or_else(|| {
                (shape & 0x18 == 0x08).then(|| LAYOUTS[usize::from(body >> 4) % LAYOUTS.len()])
            });
            // A metaclass is inherited, so a subclass of a hooked class is
            // hooked too.
            let hooked = inherited.is_some_and(|p| p.hooked) || shape & 0xe0 == 0xe0;
            let name = NAMES[usize::from(body) % NAMES.len()];
            let defines = match (body >> 2) % 8 {
                0 => Defines::Value(name),
                1 => Defines::Property(name),
                2 => Defines::Getattr,
                3 => Defines::Getattribute,
                _ => Defines::Nothing,
            };
            classes[at] = ClassFacts {
                parent,
                builtin,
                hooked,
                defines,
            };
        }
        Ok(World {
            operands,
            constants,
            classes,
        })
    }
}

impl World {
    fn number(&self, index: OperandIx) -> Option<Number> {
        self.operands.get(index.get()).copied()
    }

    fn constant_at(&self, index: ConstIx) -> Option<&Constant> {
        self.constants.get(index.get())
    }

    /// A class the decision may reason about, or `None` for one whose
    /// `isinstance` runs code -- and for an index past the table.
    fn pure(&self, index: ClassIx) -> Option<(usize, &ClassFacts)> {
        let facts = self.classes.get(index.get())?;
        (!facts.hooked).then_some((index.get(), facts))
    }

    /// The class and every class it derives from, in `__mro__` order.
    fn lineage(&self, at: usize) -> Vec<usize> {
        let mut chain = vec![at];
        while let Some(parent) = self.classes[*chain.last().unwrap_or(&at)].parent {
            chain.push(parent);
        }
        chain
    }

    /// What a direct instance can carry, read off the lineage as an attribute
    /// lookup reads it: no class here declares `__slots__`, so every instance
    /// carries a dictionary, and the first class to define a name is the one
    /// a lookup finds.
    fn attributes(&self, lineage: &[usize]) -> Attributes {
        let defines = |hook| lineage.iter().any(|&c| self.classes[c].defines == hook);
        let hook = if defines(Defines::Getattribute) {
            Hook::Getattribute
        } else if defines(Defines::Getattr) {
            Hook::Getattr
        } else {
            Hook::Neither
        };
        let mut attributes = Attributes::new(true, hook);
        for &c in lineage {
            match self.classes[c].defines {
                Defines::Value(name) => attributes.define(name, Member::Plain),
                Defines::Property(name) => attributes.define(name, Member::Descriptor),
                _ => {}
            }
        }
        attributes
    }
}

impl Constants for World {
    fn operand(&self, index: OperandIx) -> Option<Operand> {
        self.number(index).map(Number::operand)
    }

    fn constant(&self, index: ConstIx) -> Option<Operand> {
        self.constant_at(index).map(Constant::operand)
    }

    /// The order snapshot the bindings take: the class's id, every id on its
    /// `__mro__` but `object`'s, the builtin it is laid out as, and what its
    /// direct instances carry.
    fn class(&self, index: ClassIx) -> Option<Class> {
        let (at, facts) = self.pure(index)?;
        let lineage = self.lineage(at);
        let mut bases: Vec<Class> = lineage[1..]
            .iter()
            .map(|&c| Class::plain(u32::try_from(c).unwrap_or(u32::MAX)))
            .collect();
        bases.extend(facts.builtin.map(|kind| Class::plain(builtin_id(kind))));
        let id = u32::try_from(at).unwrap_or(u32::MAX);
        let class = Class::new(id, facts.builtin.map(builtin_id), &bases)
            .carrying(std::sync::Arc::new(self.attributes(&lineage)));
        Some(match facts.builtin {
            Some(kind) => class.of_kind(kind),
            None => class,
        })
    }
}

impl LeafRelations for World {
    fn leaf_subtype(&self, sub: &Schema, sup: &Schema) -> Option<bool> {
        match (sub, sup) {
            // A literal is the singleton of its constant, so it is below `sup`
            // exactly when the constant is a member of it.
            (Schema::Literal(c), _) => {
                let constant = self.constant_at(*c)?;
                match sup {
                    Schema::Anything(_) => Some(true),
                    Schema::Nothing => Some(false),
                    Schema::NoneType => Some(constant.in_kind(Kind::NoneType)),
                    Schema::Bool => Some(constant.in_kind(Kind::Bool)),
                    Schema::Int => Some(constant.in_kind(Kind::Int)),
                    Schema::Float => Some(constant.in_kind(Kind::Float)),
                    Schema::Str => Some(constant.in_kind(Kind::Str)),
                    Schema::Bytes => Some(constant.in_kind(Kind::Bytes)),
                    Schema::Literal(d) => Some(self.constant_at(*d)? == constant),
                    // A builtin's own value is no instance of a class the
                    // table defines, whatever builtin that class derives from:
                    // its type is the builtin and not the class.
                    Schema::Instance(k) => self.pure(*k).map(|_| false),
                    _ => None,
                }
            }
            (Schema::Instance(a), Schema::Instance(b)) => {
                let (at, _) = self.pure(*a)?;
                let (above, _) = self.pure(*b)?;
                Some(self.lineage(at).contains(&above))
            }
            _ => None,
        }
    }

    fn compare(&self, left: OperandIx, right: OperandIx) -> Option<core::cmp::Ordering> {
        let (left, right) = (self.number(left)?, self.number(right)?);
        match (left, right) {
            (Number::Int(a), Number::Int(b)) => Some(a.cmp(&b)),
            _ => left.real().partial_cmp(&right.real()),
        }
    }

    fn literal_sets_disjoint(&self, left: &[ConstIx], right: &[ConstIx]) -> Option<bool> {
        let mut shared = false;
        for l in left {
            for r in right {
                shared |= !self.literals_disjoint(*l, *r)?;
            }
        }
        Some(!shared)
    }

    /// `floor` and `ceil` of a number drawn here are exact integers, so the
    /// least and the greatest integer the interval admits are too.
    #[allow(clippy::cast_possible_truncation)]
    fn no_int_between(
        &self,
        lo: OperandIx,
        lo_strict: bool,
        hi: OperandIx,
        hi_strict: bool,
    ) -> Option<bool> {
        let (lo, hi) = (self.number(lo)?.real(), self.number(hi)?.real());
        let least = if lo_strict {
            lo.floor() as i64 + 1
        } else {
            lo.ceil() as i64
        };
        let greatest = if hi_strict {
            hi.ceil() as i64 - 1
        } else {
            hi.floor() as i64
        };
        Some(least > greatest)
    }

    /// `multiple % step == 0`, by Python's rules: two integers divide as
    /// integers, a float on either side makes both floats, and a zero step
    /// raises, which answers nothing.
    fn divides(&self, step: OperandIx, multiple: OperandIx) -> Option<bool> {
        match (self.number(step)?, self.number(multiple)?) {
            (Number::Int(0), Number::Int(_)) => None,
            (Number::Int(s), Number::Int(m)) => Some(m % s == 0),
            (step, multiple) => {
                let s = step.real();
                (s != 0.0).then(|| multiple.real() % s == 0.0)
            }
        }
    }

    fn atom_denotes_a_set(&self, atom: &Schema) -> Option<bool> {
        let Schema::Instance(index) = atom else {
            return None;
        };
        self.classes.get(index.get()).map(|facts| !facts.hooked)
    }

    fn literal_kind(&self, constant: ConstIx) -> Option<Kind> {
        self.constant_at(constant).map(Constant::kind)
    }

    fn operand_kind(&self, operand: OperandIx) -> Option<Kind> {
        self.number(operand).map(Number::kind)
    }

    /// A class laid out as a builtin holds that builtin's values and no
    /// other's, and so does every subclass. One laying out nothing declines:
    /// a subclass of it may derive from any builtin.
    fn class_admits_kind(&self, class: ClassIx, kind: Kind) -> Option<bool> {
        let (_, facts) = self.pure(class)?;
        facts.builtin.map(|own| own == kind)
    }

    /// A direct instance of a class laid out as a builtin has that builtin's
    /// kind, and one of a class laying out nothing is a plain object.
    fn direct_instance_of_kind(&self, class: ClassIx, kind: Kind) -> Option<bool> {
        let (_, facts) = self.pure(class)?;
        Some(facts.builtin == Some(kind))
    }

    /// No builtin derives from a class the table defines. `NoneType` has no
    /// builtin to ask about, and the bindings decline it.
    fn kind_derives_from(&self, kind: Kind, class: ClassIx) -> Option<bool> {
        self.pure(class)?;
        (kind != Kind::NoneType).then_some(false)
    }

    /// Two constants of different types share no value, and two of one type
    /// share one exactly when they are equal: every type here is a builtin
    /// whose `==` is Python's own.
    fn literals_disjoint(&self, left: ConstIx, right: ConstIx) -> Option<bool> {
        Some(self.constant_at(left)? != self.constant_at(right)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use valgebra_core::{Constraint, NoLeafRelations, Relation};

    /// Worlds from a spread of byte fills, so every arm of every draw is taken.
    fn worlds() -> Vec<World> {
        let mut drawn = Vec::new();
        for seed in 0u8..=255 {
            let bytes: Vec<u8> = (0..64u8)
                .map(|i| seed.wrapping_mul(31).wrapping_add(i.wrapping_mul(seed | 1)))
                .collect();
            if let Ok(world) = World::arbitrary(&mut Unstructured::new(&bytes)) {
                drawn.push(world);
            }
        }
        drawn
    }

    /// The pairwise and the whole-set literal answers are one relation, and a
    /// literal is below another exactly where the two are not disjoint.
    #[test]
    fn the_literal_answers_are_one_reading_of_the_constants() {
        for world in worlds() {
            for l in 0..world.constants.len() {
                for r in 0..world.constants.len() {
                    let (left, right) = (ConstIx::new(l), ConstIx::new(r));
                    let apart = world.literals_disjoint(left, right);
                    assert_eq!(world.literal_sets_disjoint(&[left], &[right]), apart);
                    assert_eq!(
                        world.leaf_subtype(&Schema::Literal(left), &Schema::Literal(right)),
                        apart.map(|apart| !apart)
                    );
                }
                // A constant is in the kind its type names.
                let at = ConstIx::new(l);
                let kind = world.literal_kind(at).expect("every constant has a kind");
                assert_eq!(
                    world.constant(at).map(|operand| operand_kind(&operand)),
                    Some(kind)
                );
            }
        }
    }

    fn operand_kind(operand: &Operand) -> Kind {
        match operand {
            Operand::Boolean(_) => Kind::Bool,
            Operand::Integer(_) => Kind::Int,
            Operand::Float(_) => Kind::Float,
            Operand::Word(_, kind) => *kind,
            Operand::NoneType => Kind::NoneType,
            Operand::Instance(_) => unreachable!("no constant here is a class"),
        }
    }

    /// The order and the integer gap read the same numbers: an interval whose
    /// bounds cross holds no integer, and one closed on an integer holds it.
    #[test]
    fn the_bound_answers_are_one_reading_of_the_numbers() {
        for world in worlds() {
            for lo in 0..world.operands.len() {
                for hi in 0..world.operands.len() {
                    let (low, high) = (OperandIx::new(lo), OperandIx::new(hi));
                    let order = world.compare(low, high).expect("every pair orders");
                    assert_eq!(world.compare(high, low), Some(order.reverse()));
                    if order.is_gt() {
                        assert_eq!(world.no_int_between(low, false, high, false), Some(true));
                    }
                    if order.is_eq() && matches!(world.number(low), Some(Number::Int(_))) {
                        assert_eq!(world.no_int_between(low, false, high, false), Some(false));
                        assert_eq!(world.no_int_between(low, true, high, false), Some(true));
                        // A nonzero number divides itself.
                        if world.number(low) != Some(Number::Int(0)) {
                            assert_eq!(world.divides(low, high), Some(true));
                        }
                    }
                }
            }
        }
    }

    /// A class's answers agree with the snapshot it hands the descriptor: a
    /// class derives from itself, a direct instance has the kind its class
    /// admits, and a hooked class answers nothing.
    #[test]
    fn the_class_answers_are_one_reading_of_the_snapshot() {
        let mut reached = [false; 3];
        for world in worlds() {
            for at in 0..world.classes.len() {
                let index = ClassIx::new(at);
                let atom = Schema::Instance(index);
                let Some(class) = world.class(index) else {
                    reached[0] = true;
                    assert_eq!(world.atom_denotes_a_set(&atom), Some(false));
                    assert_eq!(world.leaf_subtype(&atom, &atom), None);
                    continue;
                };
                assert_eq!(world.atom_denotes_a_set(&atom), Some(true));
                assert_eq!(world.leaf_subtype(&atom, &atom), Some(true));
                assert_eq!(class.kind(), world.classes[at].builtin);
                reached[1 + usize::from(class.lays_down_a_layout())] = true;
                for kind in Kind::ALL {
                    if world.direct_instance_of_kind(index, kind) == Some(true) {
                        assert_ne!(world.class_admits_kind(index, kind), Some(false));
                    }
                }
                for above in 0..world.classes.len() {
                    let Some(other) = world.class(ClassIx::new(above)) else {
                        continue;
                    };
                    assert_eq!(
                        world.leaf_subtype(&atom, &Schema::Instance(ClassIx::new(above))),
                        Some(class.derives_from(&other))
                    );
                }
            }
        }
        assert_eq!(
            reached, [true; 3],
            "a hooked, a plain or a laid-out class was never drawn"
        );
    }

    /// What the target is for: a world decides what the blind oracle cannot,
    /// in the literal, bound and class rules alike.
    #[test]
    fn a_world_decides_what_no_oracle_leaves_open() {
        let world = World {
            operands: [Number::Int(5), Number::Int(3), Number::Float(0.5)],
            constants: [Constant::Int(1), Constant::Bool(true), Constant::Str("a")],
            classes: [
                ClassFacts {
                    parent: None,
                    builtin: Some(Kind::Int),
                    hooked: false,
                    defines: Defines::Nothing,
                },
                ClassFacts {
                    parent: Some(0),
                    builtin: Some(Kind::Int),
                    hooked: false,
                    defines: Defines::Nothing,
                },
                ClassFacts {
                    parent: None,
                    builtin: None,
                    hooked: true,
                    defines: Defines::Nothing,
                },
            ],
        };
        let refine = |constraints: Vec<Constraint>| Schema::Refine {
            base: Arc::new(Schema::Int),
            constraints: constraints.into(),
        };
        let literal = |i| Schema::Literal(ConstIx::new(i));
        let class = |i| Schema::Instance(ClassIx::new(i));
        let crossed = refine(vec![
            Constraint::Ge(OperandIx::new(0)),
            Constraint::Le(OperandIx::new(1)),
        ]);
        let decided = [
            (literal(0), Schema::Int),
            (literal(1), Schema::Int),
            (crossed.clone(), Schema::Nothing),
            (class(1), class(0)),
        ];
        for (sub, sup) in &decided {
            assert!(
                sub.is_subtype_of_under(sup, &world, &[]),
                "{sub:?} <= {sup:?}"
            );
            assert!(
                !sub.is_subtype_of_under(sup, &NoLeafRelations, &[]),
                "{sub:?} <= {sup:?}"
            );
        }
        // And refuted, where a value stands outside.
        assert_eq!(
            literal(0).subtype_relation_under(&literal(1), &world, &[]),
            Relation::Fails
        );
        assert!(crossed.is_empty_with(&world, &[]));
    }
}
