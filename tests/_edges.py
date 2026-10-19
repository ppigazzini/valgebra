"""The values a drawn value reaches only by luck, written once.

Each is one some operation reads differently from its neighbors: `nan` equals
nothing including itself, the infinities bound every comparison, `-0.0` equals
`0.0` and hashes with it, the four integers sit at the ends of the carriers a
lowering uses, a newline is where a pattern's `.` and `$` part company, and a
lone surrogate is a `str` of one character with no text, which every kind
holds and no pattern matches.

A suite that asks a relation over values asks it of these beside its own spread
and its draws, so an edge is one every such suite reaches on every run rather
than one each remembers to write. The suites keep their own spreads, since the
shapes their schemas are built from differ; what they share here is what every
one of them would otherwise copy. A boundary nobody wrote here is one all of
them miss together.
"""

#: The edges of the scalar kinds.
EDGES: tuple[object, ...] = (
    float("nan"),
    float("inf"),
    float("-inf"),
    -0.0,
    2**53 + 1,
    -(2**63),
    2**63 - 1,
    2**70,
    "\n",
    "a\nb",
    "\ud800",
)
