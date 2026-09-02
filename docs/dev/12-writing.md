# Writing rules

These govern the shipped prose — this set, `README.md`, `CONTRIBUTING.md`,
`AGENTS.md`, the user guide under `docs/` — and the comments in the source. They
live here, in the set that ships, because a rule nobody can read is a rule nobody
can follow.

A second surface is untracked: the engineering contract, the operator prompt, the
milestone backlog, user-requested analyses. **Do not converge the two.** A
shipped page must not carry campaign history; an untracked note must not be the
only place a shipped fact lives. And a shipped file must not name that surface's
**location** — it is gitignored, so the reference dangles for every reader but
its author. `scripts/docs_lint.py` sweeps every tracked file for it, prose and
source alike, since a dangling reference costs a reader the same either way.

## The rules

**State the page contract first.** The shortest accurate sentence saying what the
page covers. A reader who is on the wrong page should learn it in one line.

**Say what a schema means as a set of values.** A combinator, a node, or an
annotation form may not appear without its denotation. "Works like some other
library" is not a denotation, and neither is a list of examples.

**Separate a verified fact from a valgebra decision.** "pydantic-core builds a
validator once and reuses it" is checkable against a checkout. "valgebra exposes
typing annotations as the primary notation" is a policy choice. Blur them and
nobody can tell what they are allowed to change.

**Describe a gap as a gap, never as a design.** Framing a hole as a decision is
what keeps it alive — nobody fixes a design. Say unimplemented, say what the
absence costs today, and say which kind it is: not yet built, or ruled out with a
reason.

**Never rationalise a defect into a convention.** The same rule one level down.
When you find yourself writing the sentence that makes the odd thing sound
intended, stop and check whether it is.

**Name the owner and the invariant, not just the mechanism.** Say which file and
symbol owns the behaviour and what must stay true about it. "The pool holds the
literals" is accurate and useless; what a reader needs is that four index spaces
address that one pool, so a payload used against the wrong kind retrieves a real
object of the wrong kind. Write the sentence a reader needs before they delete
your line.

**Verify the claim against the tree; run it when it is behavioural.** Not "read
it carefully" — `grep -n` for the symbol, `python scripts/perf_gate.py` for a
count, `uv run pytest -k` for a behaviour. Claims that took seconds to disprove
have shipped in documentation sets.

**A claim about *meaning* is not behavioural, and running it does not check it.**
What a schema denotes is stated by the variant's doc comment in
`crates/valgebra-core/src/ir.rs`. A `Validator` probe shows what the walk does,
and the walk can agree with the denotation by accident — a node that refuses a
value because its carrier is wrong looks exactly like one that refuses it because
the structure is wrong. So:

| the claim is about | check it against |
|---|---|
| what a schema admits, in this build | a probe, a test |
| what a node *denotes* | the doc comment in `ir.rs` |
| whether a set is expressible at all | the node set, and [01-schema-ir.md](01-schema-ir.md)'s admission test |

The failure this prevents has a shape: arguing from **behaviour** or from **what
the implementation computes internally** to **what the algebra denotes**. Three
false claims in this project's own analysis had that shape, and each survived a
REPL check because a REPL cannot see a denotation.

**Never pin a number a gate computes.** The instruction budgets, a mutation
score, a test count, a line count. Quote the file or the gate that owns it.
`scripts/docs_lint.py` fails on a budget number copied into prose, because a
stale number is worse than an absent one: it tells a reader to hold the wrong
invariant.

**Never pin a list a gate owns, either.** The jobs in CI, the commands in the
local gate, the divergences from a differential oracle. A list that drifts by one
entry reads exactly like one that has not, and the lint cannot count. Name the
file that owns the list beside it.

**State the limit.** A page that omits its own boundary invites over-trust. Say
what the thing does *not* cover: a coverage floor cannot see a wrong answer; a
mutation score is a statement about one test command; a gate that could not run
has proven nothing.

**Show the command.** "It is faster" is not a claim; an instruction count with
the command that produced it is. A performance or behaviour claim ships with what
produced it, so the next reader re-runs it instead of trusting you.

**No history in shipped prose.** "It used to be X", "this was fixed in Y", "we
tried Z" is out of date the day after it is written. The before-and-after belongs
in the commit message, which is the durable per-task record. A page states what
is true **now**.

A measurement is the exception, and only as a rule: where a number explains why
the code has its shape — the walk mode's discriminant order in
[06-type-design.md](06-type-design.md) — write it as the rule a reader applies
now, and let the number be the evidence.

**One example beats three paragraphs**, and **pair every prohibition with an
alternative.** "Do not pin the budget" leaves a reader stuck; "do not pin the
budget — name `scripts/perf_budget.json`, which owns it" does not.

**Cut anything that does not help implement or verify.** Background a reader
could get from the typing spec belongs in [11-references.md](11-references.md) as
a link. Length is not thoroughness; it is where rot hides. This binds a generated
page exactly as it binds a hand-written one: add no section that exists to look
complete — a summary restating the section above it, a recap of what a gate
prints, a next-steps list nobody asked for.

## Hot and cold

These pages do not age alike, and treating them the same is why they rot. A page
is **hot** when it describes code that moves, **cold** when what it describes
barely does.

**Change hot code, fix its page in the same commit.** A doc is wrong from the
moment the code lands, and nobody knows which claim broke better than the person
who broke it.

| Page | Temperature |
|---|---|
| [00-architecture.md](00-architecture.md) | warm — the crate boundary is compiler-checked, so it moves rarely |
| [01-schema-ir.md](01-schema-ir.md) | hot — a node added here is an edit there |
| [02-decision.md](02-decision.md) | hot — the decided fragment grows |
| [03-frontend.md](03-frontend.md) | hot — every new annotation form lands here |
| [04-walk.md](04-walk.md) | hot — where soundness is decided |
| [05-errors.md](05-errors.md) | warm — the codes are API and move slowly |
| [06-type-design.md](06-type-design.md) | warm — a type added without a row makes the page wrong |
| [07-tooling-ci.md](07-tooling-ci.md) | hot — a gate added here is a page edit there |
| [08-testing.md](08-testing.md) | warm |
| [09-releasing.md](09-releasing.md) | warm — a change to the release workflow's inputs makes the page wrong |
| [10-theory.md](10-theory.md) | cold — a theorem does not expire |
| [11-references.md](11-references.md) | cold |
| this page | cold |
| [13-glossary.md](13-glossary.md) | warm — every entry names an owner, and a rename dates it |

Cold does not mean unowned. It means the claim outlives a release, so when it
*is* wrong it has usually been wrong for a long time.

## Code comments

Same rules, plus these. Rust states more in the types than most languages can — a
slice carries its length, a `Result` carries its error set — so a comment that
restates the signature earns nothing. What it must carry is what the type cannot.

**Imperative mood, leading with a verb.** "Return the set of…", not "This
function returns…".

**Write only the constraint the code cannot show.** Never restate the next line.
Never say where the change came from or why it is right — that is the commit
message's job, and it is noise the moment the commit merges.

**Name the invariant, and what breaks without it.** "Build the index once" says
nothing; "keyed by the address of each record's own buffer, so a copied validator
never inherits another's index" survives a refactor.

**Give a denotation where a node has one.** A `Schema` variant's comment says
which set of Python values it admits. That is the comment the whole design rests
on and it is not optional.

**Keep the integer-semantics comments.** Where a line relies on saturation or on
a checked operation, that note is the whole reason it looks the way it does.

**No history, no meta.** Not "was a bool", not "changed in the type campaign",
not "the following block does". A comment describes the code as it is, to someone
who has never seen it.

## The gate, and what it cannot see

```bash
python scripts/docs_lint.py
```

It reads every tracked Markdown file and fails on six things: a dead internal
link, a named path that does not exist, a reference into the untracked surface, a
budget number quoted in prose, a developer page missing from its index, and a
ledger missing from the table of ledgers (or a spelled count that disagrees with
it). Read the script's own docstring for the exact rules — a second copy here
would drift.

A path `.gitignore` names is exempt from the second rule, because the repository
decided not to carry it and a page naming one is usually documenting the tool
that writes it. **The exemption is asked of git rather than of the filesystem**,
so the answer is the same on a fresh clone and on a machine that has run that
tool. It was not, once, and the lane that caught it is the only reason anyone
knew: a check whose verdict changes with local build output is measuring the
machine rather than the tree.

That exemption is also what makes the third rule necessary. The untracked working
area is gitignored, so every reference into it would be exempt from the path
check by design; the rule that sweeps for it by name is what covers the class the
exemption creates.

**Three classes stay out of its reach, and they are the common ones:**

- a real symbol attributed to the **wrong file**;
- a list with the wrong **count or order** — the jobs in CI, the commands in the
  local gate. Each lints perfectly clean;
- a behaviour or flag described as absent from a build that has it.

### It cannot tell you a sentence is false

That is the whole point of this section. A page can link cleanly, name only real
paths, quote no budget — and still describe code replaced three commits ago, or
frame an unbuilt capability as a design decision. Neither is mechanically
detectable.

The gate buys the mechanical half so review can spend its attention on the half
that needs a reader. Prefer the claim that stays true: name the owner and the
invariant, name the file that owns the number, and say what the thing does not
cover.
