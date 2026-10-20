# valgebra developer documentation

What the codebase does, for a contributor or an agent reading it cold. It
describes the tree **as it is**, not as it is intended to become; where a
capability is unbuilt, its page says so and says what the absence costs.

This set is not published. The site under [../](../README.md) is the user guide —
what valgebra does, for someone using it. These pages are what it is made of.

Read in order. Each page owns a zone of the source and is the live claim about
it. Which page a source file belongs to is the zone table in
[00-architecture.md](00-architecture.md), which `scripts/docs_lint.py` holds to
the tree; this table says what each page is about.

| Page | Owns |
|---|---|
| [00-architecture.md](00-architecture.md) | the zones and the page each source file belongs to, the direction their dependencies run, how a value and a pair of schemas flow, the invariants every zone keeps, every bound, and the designs refused for the shape of the whole |
| [01-schema-ir.md](01-schema-ir.md) | the node set and what each node denotes |
| [02-decision.md](02-decision.md) | what is decided and what stays conservative |
| [03-frontend.md](03-frontend.md) | typing annotations into the IR |
| [04-walk.md](04-walk.md) | the membership walk, both input paths, and its modes |
| [05-errors.md](05-errors.md) | the violation model, the error codes, the annotation render |
| [06-type-design.md](06-type-design.md) | the value domain: what each type denotes, why it has that shape, what it does not promise |
| [07-tooling-ci.md](07-tooling-ci.md) | the gates, the lanes, the exit codes, and each instrument's blind spots |
| [08-testing.md](08-testing.md) | the test layers, the three mutation sweeps, the ledgers, the completeness probe |
| [09-releasing.md](09-releasing.md) | `.github/workflows/release.yml` and the version surfaces — the order a release is cut in |
| [10-theory.md](10-theory.md) | the papers the design rests on, which code each touches, and the ledger of every result, obligation and deviation with the test that holds it |
| [11-references.md](11-references.md) | the reference implementations and the toolchain baseline |
| [12-writing.md](12-writing.md) | how to write a page here, and a comment in the source |
| [13-glossary.md](13-glossary.md) | the words this set uses without stopping to define them |

## Docs are part of the change, not after it

Each page is a live claim about code someone is about to touch. Change a zone,
fix its page **in the same commit**; [12-writing.md](12-writing.md) says why.

```bash
python scripts/docs_lint.py
```

catches, among its rules, a dead link, a named path that does not exist, a
reference into the untracked working area, and a budget number copied into
prose. It **cannot** tell you a sentence has become false. That part is yours,
and [12-writing.md](12-writing.md) names the three classes it cannot reach.

## Two documentation surfaces

- **This set ships**, with the rest of the surface
  [12-writing.md](12-writing.md) names and owns: [../../README.md](../../README.md),
  [../../CONTRIBUTING.md](../../CONTRIBUTING.md),
  [../../AGENTS.md](../../AGENTS.md), the user guide under [../](../README.md),
  and the comments in the source. A clone carries it.
- **A second surface is untracked**, so a clone does not. It holds the
  engineering contract, the operator prompt, the milestone backlog, and
  user-requested analyses.

[12-writing.md](12-writing.md) owns how the two are kept apart.
