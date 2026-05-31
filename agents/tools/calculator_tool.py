"""
Safe mathematical expression calculator tool.

Uses Python's `ast` module to parse and evaluate expressions without
calling the dangerous built-in `eval()`. Supports basic arithmetic,
exponentiation, and common scientific math functions.
"""

import ast
import math
import operator
from typing import Union

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Allowed operators and functions for safe evaluation
# ---------------------------------------------------------------------------
_ALLOWED_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitXor: operator.pow,  # Allow ^ as power (common user expectation)
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = {
    # Basic math
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    # Powers and roots
    "sqrt": math.sqrt,
    "pow": math.pow,
    "exp": math.exp,
    # Logarithms
    "log": math.log,       # natural log (or log(x, base))
    "log2": math.log2,
    "log10": math.log10,
    # Trigonometry
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # Hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    # Rounding / misc
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
}

_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "tau": math.tau,
}


# ---------------------------------------------------------------------------
# Safe AST evaluator
# ---------------------------------------------------------------------------
class _SafeEvaluator(ast.NodeVisitor):
    """Walk an AST tree and evaluate mathematical expressions safely."""

    def visit(self, node):
        return super().visit(node)

    def generic_visit(self, node):
        raise ValueError(
            f"Unsupported expression element: {type(node).__name__}. "
            "Only arithmetic expressions, math functions, and numeric literals are allowed."
        )

    # --- Literals ---
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    # Python 3.7 compat (Num is deprecated but may still appear)
    visit_Num = visit_Constant

    # --- Named constants (pi, e, etc.) ---
    def visit_Name(self, node):
        name = node.id
        if name in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[name]
        raise ValueError(
            f"Unknown name '{name}'. Allowed constants: {', '.join(_ALLOWED_CONSTANTS)}"
        )

    # --- Binary operators ---
    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINARY_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _ALLOWED_BINARY_OPS[op_type](left, right)

    # --- Unary operators ---
    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = self.visit(node.operand)
        return _ALLOWED_UNARY_OPS[op_type](operand)

    # --- Function calls ---
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed (e.g. sqrt(x)).")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(
                f"Unknown function '{func_name}'. Allowed: {', '.join(sorted(_ALLOWED_FUNCTIONS))}"
            )
        args = [self.visit(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[func_name](*args)

    # --- Top-level expression wrapper ---
    def visit_Expression(self, node):
        return self.visit(node.body)


def _safe_eval(expression: str) -> Union[int, float, complex]:
    """
    Safely evaluate a mathematical expression string.

    Raises ValueError for unsupported or malicious expressions.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")

    evaluator = _SafeEvaluator()
    return evaluator.visit(tree)


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Use this tool when you need to perform calculations — arithmetic,
    exponentiation, logarithms, trigonometry, etc. The expression is
    evaluated safely (no arbitrary code execution).

    Supported operations:
        +, -, *, /, //, %, ** (power), ^ (also power)
    Supported functions:
        sqrt, pow, exp, log, log2, log10, sin, cos, tan, asin, acos,
        atan, atan2, sinh, cosh, tanh, ceil, floor, factorial, gcd,
        abs, round, min, max
    Supported constants:
        pi, e, inf, tau

    Args:
        expression: A mathematical expression string.
                    Examples: '2 ** 10', 'sqrt(144)', 'log2(1024)',
                              'sin(pi/4)', '3 * (4 + 5)'

    Returns:
        The computed result as a string, or an error message.
    """
    try:
        result = _safe_eval(expression)
        # Format nicely: drop trailing .0 for whole-number floats
        if isinstance(result, float) and result == int(result) and not math.isinf(result):
            result = int(result)
        return f"{expression} = {result}"
    except ValueError as e:
        return f"Calculator error: {e}"
    except ZeroDivisionError:
        return f"Calculator error: Division by zero in '{expression}'"
    except OverflowError:
        return f"Calculator error: Result too large for '{expression}'"
    except Exception as e:
        return f"Calculator error: {type(e).__name__}: {e}"
