"""A tiny, safe expression language for owner-defined policy rules.

    order_value_paise > 5000000 and payment_received_paise * 10 < order_value_paise

This is NOT Python evaluation. The text is parsed with `ast.parse` and then interpreted by a
tree walk that accepts only a whitelist of node types. There is no `eval`/`exec`/`compile`, no
function calls, attribute access, subscripts, strings, floats, lambdas or comprehensions.
Everything is integer arithmetic: + - * // % and comparisons, combined with and / or / not.

A rule's expression describes the VIOLATION: when it evaluates to True the order needs approval.
Anything wrong (unknown name, type error, division by zero, overflow) is an error, and the
policy engine treats an error as a violation, so a broken rule fails closed.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping

MAX_LENGTH = 300
MAX_NODES = 60
MAX_CONSTANT = 10**12
MAX_VALUE = 10**24  # intermediate results beyond this are an error, not a slow multiplication

# Facts an expression may read. All are integers (paise, milligrams or seconds).
NAMES: frozenset[str] = frozenset(
    {
        "order_value_paise",
        "payment_received_paise",
        "new_pending_paise",
        "list_value_paise",
        "quoted_rate_paise",
        "benchmark_paise",
        "benchmark_age_seconds",
        "outstanding_paise",
        "weight_mg",
    }
)


class ExpressionError(ValueError):
    pass


_COMPARE: dict[type[ast.cmpop], Callable[[int, int], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_ARITH: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def parse_expression(text: str) -> ast.Expression:
    """Parse and validate. Raises ExpressionError for anything outside the whitelist."""
    if not text or not text.strip():
        raise ExpressionError("expression is empty")
    if len(text) > MAX_LENGTH:
        raise ExpressionError(f"expression is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ExpressionError("not a valid expression") from exc
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > MAX_NODES:
            raise ExpressionError("expression is too complex")
        _check_node(node)
    return tree


def _check_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression | ast.Load | ast.And | ast.Or | ast.Not | ast.USub):
        return
    if isinstance(node, ast.BoolOp | ast.Compare):
        if isinstance(node, ast.Compare) and not all(type(op) in _COMPARE for op in node.ops):
            raise ExpressionError("only < <= > >= == != comparisons are allowed")
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.Not | ast.USub):
            raise ExpressionError("only 'not' and unary minus are allowed")
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ARITH:
            raise ExpressionError("only + - * // % are allowed (no '/', '**', bit operations)")
        return
    if isinstance(node, ast.Constant):
        if type(node.value) is not int:  # rejects float, str, bytes, None, bool
            raise ExpressionError("only whole-number constants are allowed")
        if abs(node.value) > MAX_CONSTANT:
            raise ExpressionError("constant is too large")
        return
    if isinstance(node, ast.Name):
        if node.id not in NAMES:
            raise ExpressionError(f"unknown name {node.id!r}")
        return
    if isinstance(node, ast.cmpop | ast.operator):
        return  # the operator nodes themselves were checked via their parents
    raise ExpressionError(f"{type(node).__name__} is not allowed")


def expression_names(text: str) -> frozenset[str]:
    """The facts an expression reads (validates it first)."""
    tree = parse_expression(text)
    return frozenset(n.id for n in ast.walk(tree) if isinstance(n, ast.Name))


def evaluate(text: str | ast.Expression, values: Mapping[str, int]) -> bool:
    """True if the VIOLATION condition holds. Raises ExpressionError on any problem."""
    tree = parse_expression(text) if isinstance(text, str) else text
    result = _eval(tree.body, values)
    if type(result) is not bool:
        raise ExpressionError("a rule must be a true/false condition")
    return result


def _int(value: int | bool) -> int:
    if type(value) is not int:
        raise ExpressionError("arithmetic needs whole numbers, not true/false")
    if abs(value) > MAX_VALUE:
        raise ExpressionError("number too large")
    return value


def _eval(node: ast.expr, values: Mapping[str, int]) -> int | bool:
    if isinstance(node, ast.Constant):
        return int(node.value)  # type: ignore[arg-type]  # validated as int
    if isinstance(node, ast.Name):
        value = values.get(node.id)
        if value is None or type(value) is not int:
            raise ExpressionError(f"{node.id} has no value")
        return value
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, values)
        if isinstance(node.op, ast.Not):
            if type(operand) is not bool:
                raise ExpressionError("'not' needs a true/false condition")
            return not operand
        return -_int(operand)
    if isinstance(node, ast.BinOp):
        left, right = _int(_eval(node.left, values)), _int(_eval(node.right, values))
        if isinstance(node.op, ast.FloorDiv | ast.Mod) and right == 0:
            raise ExpressionError("division by zero")
        return _int(_ARITH[type(node.op)](left, right))
    if isinstance(node, ast.Compare):
        left_v = _int(_eval(node.left, values))
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right_v = _int(_eval(comparator, values))
            if not _COMPARE[type(op)](left_v, right_v):
                return False
            left_v = right_v
        return True
    if isinstance(node, ast.BoolOp):
        results = [_eval(v, values) for v in node.values]
        if any(type(r) is not bool for r in results):
            raise ExpressionError("'and'/'or' need true/false conditions")
        return all(results) if isinstance(node.op, ast.And) else any(results)
    raise ExpressionError(f"{type(node).__name__} is not allowed")  # unreachable after validation
