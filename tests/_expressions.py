"""Workflow expressions, evaluated as the runner evaluates them.

A ledger that reads a `${{ }}` expression as text answers a question about its
spelling: the version a `continue-on-error` names is in `!= '3.16'` as much as
in `== '3.16'`. The expressions are evaluated instead, over a context the
ledger builds -- a leg's matrix values, the results of the jobs a gate needs --
and a form this does not read fails rather than reading as true.
"""

from __future__ import annotations

import ast
import functools
import itertools
import re

#: A context's property as an expression spells it, read by name.
_CONTEXT_PROPERTY = re.compile(
    r"\b(needs|matrix|env|steps|inputs|github|vars)\.([\w-]+)"
)


def spelled_in_python(expression: str) -> str:
    """Spell a workflow expression in Python's grammar, without changing its sense.

    The operators become Python's, and a property of a context is read by
    index, since a name with a hyphen in it -- `needs.rust-lint`,
    `matrix.python-version` -- is a subtraction to Python.
    """
    text = expression.strip().removeprefix("${{").removesuffix("}}")
    text = re.sub(_CONTEXT_PROPERTY, r"\1['\2']", text)
    text = text.replace("&&", " and ").replace("||", " or ")
    return " ".join(re.sub(r"!(?!=)", " not ", text).split())


#: The expression language's named literals, which Python reads as names.
LITERALS = {"true": True, "false": False, "null": None}

#: The status functions, as they answer on a run nobody canceled.
STATUS = {"always": True, "cancelled": False}


def _leaf(node: ast.expr, context: dict, expression: str) -> object:
    """Read a value: a literal, a property of the context, or a status function."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in LITERALS:
        return LITERALS[node.id]
    if isinstance(node, ast.Name):
        return context.get(node.id, "")
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        owner = _leaf(node.value, context, expression)
        key = (
            node.attr
            if isinstance(node, ast.Attribute)
            else _leaf(node.slice, context, expression)
        )
        return owner.get(key, "") if isinstance(owner, dict) else ""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in STATUS
    ):
        return STATUS[node.func.id]
    message = f"the ledger cannot read {ast.unparse(node)!r} in {expression!r}"
    raise AssertionError(message)


def evaluate(expression: str, context: dict) -> object:
    """Evaluate a workflow expression as the runner does, on a run not canceled.

    A property nothing set reads as the empty string, which is what an output a
    step never wrote reads as, and a skipped job's outputs. A form this does not
    read fails here rather than reading as true.
    """

    def value(node: ast.expr) -> object:
        if isinstance(node, ast.BoolOp):
            conjunction = isinstance(node.op, ast.And)
            result: object = conjunction
            for operand in node.values:
                result = value(operand)
                if bool(result) != conjunction:
                    break
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not value(node.operand)
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
        ):
            same = value(node.left) == value(node.comparators[0])
            return same == isinstance(node.ops[0], ast.Eq)
        return _leaf(node, context, expression)

    return value(_parsed(expression))


@functools.cache
def _parsed(expression: str) -> ast.expr:
    """Parse an expression once: the gate's is read a few hundred times a row."""
    return ast.parse(spelled_in_python(expression), mode="eval").body


def legs(job: dict) -> list[dict[str, str]]:
    """Give each leg of a job as the `matrix.*` values its expressions read.

    Expanded as the runner expands a matrix: an `include` row widens every
    combination of the lists whose values it agrees with, and is a leg of its
    own where it agrees with none.
    """
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    axes = {key: value for key, value in matrix.items() if key != "include"}
    expanded = [
        {f"matrix.{key}": str(value) for key, value in zip(axes, values, strict=True)}
        for values in itertools.product(*axes.values())
    ]
    combinations = list(expanded)
    listed = {f"matrix.{key}" for key in axes}
    for row in matrix.get("include", []):
        named = {f"matrix.{key}": str(value) for key, value in row.items()}
        widened = [
            leg
            for leg in combinations
            if all(leg[key] == named[key] for key in listed & named.keys())
        ]
        for leg in widened:
            leg.update(named)
        if not widened:
            expanded.append(named)
    return expanded


def forgives(flag: object, leg: dict[str, str]) -> bool:
    """Whether a `continue-on-error` value forgives a failure on `leg`.

    `leg` is a leg as [`legs`] gives it, each `matrix.<name>` with its value.
    The value is evaluated there, as the runner evaluates it: `!= '3.16'`
    names the release `== '3.16'` names and forgives every other leg, and a
    clause that holds everywhere forgives everywhere, whatever it names.
    """
    if flag is None or isinstance(flag, bool):
        return bool(flag)
    text = str(flag).strip()
    if text.lower() in {"", "true", "false"}:
        return text.lower() == "true"
    matrix = {key.removeprefix("matrix."): value for key, value in leg.items()}
    return bool(evaluate(text, {"matrix": matrix}))
