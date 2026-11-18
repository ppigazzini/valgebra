# valgebra developer documentation

What the codebase does, for a contributor or an agent reading it cold. It
describes the tree **as it is**, not as it is intended to become; where a
capability is unbuilt, its page says so and says what the absence costs.

This set is not published. The site under [../](../index.md) is the user guide —
what valgebra does, for someone using it. These pages are what it is made of.

Read in order. Each page owns a zone of the source and is the live claim about
it. Which page a source file belongs to is the zone table in
[the architecture page](architecture.md), which `scripts/docs_lint.py` holds to
the tree; this table says what each page is about.

| Page | Owns |
|---|---|
| [The architecture page](architecture.md) | the zones and the page each source file belongs to, the direction their dependencies run, how a value and a pair of schemas flow, the invariants every zone keeps, every bound, and the designs refused for the shape of the whole |
| [The schema IR page](schema-ir.md) | the node set and what each node denotes |
| [The decision page](decision.md) | what is decided and what stays conservative |
| [The frontend page](frontend.md) | typing annotations into the IR |
| [The walk page](walk.md) | the membership walk, both input paths, and its modes |
| [The errors page](errors.md) | the violation model, the error codes, the annotation render |
| [The type design page](type-design.md) | the value domain: what each type denotes, why it has that shape, what it does not promise |
| [The tooling and CI page](tooling-ci.md) | the gates, the lanes, the exit codes, and each instrument's blind spots |
| [The testing page](testing.md) | the test layers, the three mutation sweeps, the ledgers, the completeness probe |
| [The releasing page](releasing.md) | `.github/workflows/release.yml` and the version surfaces — the order a release is cut in |
| [The theory page](theory.md) | the papers the design rests on, which code each touches, and the ledger of every result, obligation and deviation with the test that holds it |
| [The references page](references.md) | the reference implementations and the toolchain baseline |
| [The writing page](writing.md) | how to write a page here, and a comment in the source |
| [The glossary](glossary.md) | the words this set uses without stopping to define them |

## Docs are part of the change, not after it

Each page is a live claim about code someone is about to touch. Change a zone,
fix its page **in the same commit**; [the writing page](writing.md) says why.

```bash
python scripts/docs_lint.py
```

catches, among its rules, a dead link, a named path that does not exist, a
reference into the untracked working area, and a budget number copied into
prose. It **cannot** tell you a sentence has become false. That part is yours,
and [the writing page](writing.md) names the three classes it cannot reach.

## Two documentation surfaces

- **This set ships**, with the rest of the surface
  [the writing page](writing.md) names and owns: [The README](../../README.md),
  [the contributing guide](../../CONTRIBUTING.md),
  [the agent instructions](../../AGENTS.md), the user guide under [../](../index.md),
  and the comments in the source. A clone carries it.
- **A second surface is untracked**, so a clone does not. It holds the
  engineering contract, the operator prompt, the milestone backlog, and
  user-requested analyses.

[The writing page](writing.md) owns how the two are kept apart.
