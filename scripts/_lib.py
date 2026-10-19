"""What every script under `scripts/` says the same way.

The codes a gate answers in, the checkout it sits in, how it reads its command
line, how it asks git a question, and which Rust file is test code. Each is a
rule a script must not restate in a dialect of its own: a fourth code, a flag
answering to a prefix of its name, a git question asked of whichever directory
the caller happened to be in, a test file told apart by its name.

A script runs as `python scripts/<name>.py`, which puts this directory first on
the import path, so `import _lib` finds this module from any of them. Whatever
loads a script by path from a process of its own puts the directory there
first: `load_script` in `tests/_tree.py`, and the benchmarks' conftest. mkdocs
loads `docs_stubs.py` as a griffe extension with no loader of ours between, so
that script puts its own directory there. The leading underscore says it is no
script: the ledgers holding every script to a lane, a contract row and `--help`
read the files without one.

What a step or a measurement runs in is not here. `gate.py` hands a step the
caller's environment less what only a terminal or `uv run` put in it, and
`perf_gate.py` hands a workload the few names it needs and nothing else; the two
are different rules, and each is held by a test the other would fail.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

#: The three outcomes a gate reports, each with its own code. A gate that could
#: not run has proven nothing, so it shares a code with neither of the others;
#: `docs/dev/07-tooling-ci.md` gives the table.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CANNOT_RUN = 2

#: The root of the checkout the scripts sit in.
ROOT = Path(__file__).resolve().parent.parent


def arguments(description: str | None) -> argparse.ArgumentParser:
    """Start a script's command line, with `description` as its help text.

    The text keeps its own line breaks, and no flag answers to a prefix of its
    name: read by prefix, `--up` is `--update`, and a mistyped flag becomes one
    the script reads, for a measurement nobody asked for.
    """
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )


def git_output(*args: str, cwd: Path) -> str | None:
    """Give a git command's output, stripped, or `None` where it fails.

    `cwd` is required: a script reading its own tree names `ROOT`, and one asked
    about the checkout it was started in names that. `None` also answers a
    machine with no git on it.
    """
    try:
        done = subprocess.run(  # fixed argv, no shell
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


#: An attribute that makes the module below it test code: `#[cfg(test)]`, alone
#: or inside an `all(...)`, as an interpreter corpus is declared.
_TEST_ONLY = re.compile(r"#\[cfg\((?:all\()?test\b")


def declarer(path: Path) -> Path | None:
    """Name the file that declares the module `path` is, `lib.rs` for a top one.

    A module lives under a crate's `src/`; a file anywhere else is declared by
    nothing, and is no module.
    """
    if "src" not in path.parts[:-1]:
        return None
    parent = path.parent
    if parent.name == "src":
        root = parent / "lib.rs"
        return root if root.is_file() and root != path else None
    for candidate in (parent.with_suffix(".rs"), parent / "mod.rs"):
        if candidate.is_file() and candidate != path:
            return candidate
    return None


def is_test_module(path: Path) -> bool:
    """Whether the file that declares `path` declares it under `#[cfg(test)]`.

    Read from the declaration Rust reads rather than from the file's name:
    `tests.rs` is a name a shipped module could take, and a corpus could take
    any other. Every check that tells test code from shipped code asks this,
    and the coverage lanes, which can only name files by a pattern, are held
    to what it answers.
    """
    found_in = declarer(path)
    if found_in is None:
        return False
    found = re.search(
        r"((?:^[ \t]*(?:#\[[^\n]*\]|//[^\n]*)[ \t]*\n)*)"
        rf"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?mod\s+{re.escape(path.stem)}\s*;",
        found_in.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return found is not None and _TEST_ONLY.search(found.group(1)) is not None
