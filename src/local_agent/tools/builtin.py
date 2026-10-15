import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool


@tool
def current_time(timezone: str = "Asia/Shanghai") -> str:
    """Return the current time in an IANA timezone."""
    return datetime.now(ZoneInfo(timezone)).isoformat()


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression using +, -, *, /, //, %, ** and parentheses."""
    value = _eval_arithmetic(ast.parse(expression, mode="eval").body)
    return str(value)


def _eval_arithmetic(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _eval_arithmetic(node.left)
        right = _eval_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("Exponent is too large")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_eval_arithmetic(node.operand))
    raise ValueError("Only numeric arithmetic expressions are allowed")


BUILTIN_TOOLS = [current_time, calculator]
