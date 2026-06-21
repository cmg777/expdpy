"""Safe evaluation of user-defined variable expressions.

ExPanDaR evaluates ``var_def`` strings as ``dplyr::mutate`` code via ``eval(parse())``. We
instead walk a restricted Python AST with a strict allow-list — column references,
arithmetic/comparison/boolean operators, and the panel-aware functions
``isna``/``exp``/``log``/``lag``/``lead`` — and refuse everything else. No ``eval``/``exec``
is ever used, which is strictly safer than R's sandboxed environment.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["evaluate_var_def", "UDVError"]


class UDVError(ValueError):
    """Raised when a user-defined-variable expression is invalid or disallowed."""


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
    ast.BitAnd: lambda a, b: a & b,
    ast.BitOr: lambda a, b: a | b,
}
_ALLOWED_CMP = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}
_ALLOWED_FUNCS = {"isna", "exp", "log", "lag", "lead"}


class _Evaluator(ast.NodeVisitor):
    def __init__(self, df: pd.DataFrame, entities: Sequence[str]):
        self.df = df
        self.entities = list(entities)

    # --- functions -----------------------------------------------------------
    def _grouped_shift(self, series: pd.Series, n: int) -> pd.Series:
        if not self.entities:
            return series.shift(n)
        return series.groupby([self.df[c] for c in self.entities]).shift(n)

    def _call(self, name: str, args: list):
        if name == "isna":
            return (
                args[0].isna() if isinstance(args[0], pd.Series) else pd.isna(args[0])
            )
        if name == "exp":
            return np.exp(args[0])
        if name == "log":
            return np.log(args[0])
        if name in ("lag", "lead"):
            n = int(args[1]) if len(args) > 1 else 1
            series = args[0]
            if not isinstance(series, pd.Series):
                raise UDVError(f"{name}() expects a column expression")
            return self._grouped_shift(series, n if name == "lag" else -n)
        raise UDVError(f"function '{name}' is not allowed")

    # --- node visitors -------------------------------------------------------
    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        op = type(node.op)
        if op not in _ALLOWED_BINOPS:
            raise UDVError(f"operator {op.__name__} is not allowed")
        return _ALLOWED_BINOPS[op](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp):
        val = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.Not):
            return ~val if isinstance(val, pd.Series) else (not val)
        if isinstance(node.op, ast.Invert):
            return ~val
        raise UDVError("unary operator not allowed")

    def visit_BoolOp(self, node: ast.BoolOp):
        values = [self.visit(v) for v in node.values]
        result = values[0]
        for nxt in values[1:]:
            result = (result & nxt) if isinstance(node.op, ast.And) else (result | nxt)
        return result

    def visit_Compare(self, node: ast.Compare):
        if len(node.ops) != 1:
            raise UDVError("chained comparisons are not allowed")
        op = type(node.ops[0])
        if op not in _ALLOWED_CMP:
            raise UDVError("comparison operator not allowed")
        return _ALLOWED_CMP[op](self.visit(node.left), self.visit(node.comparators[0]))

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise UDVError("only isna/exp/log/lag/lead calls are allowed")
        if node.keywords:
            raise UDVError("keyword arguments are not allowed")
        return self._call(node.func.id, [self.visit(a) for a in node.args])

    def visit_Name(self, node: ast.Name):
        if node.id.startswith("__"):
            raise UDVError("dunder names are not allowed")
        if node.id not in self.df.columns:
            raise UDVError(f"unknown column '{node.id}'")
        return self.df[node.id]

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise UDVError("only numeric/boolean constants are allowed")

    def generic_visit(self, node: ast.AST):
        raise UDVError(f"syntax element {type(node).__name__} is not allowed")


def evaluate_var_def(
    expr: str,
    df: pd.DataFrame,
    entities: Sequence[str] | None = None,
    time: str | None = None,
) -> pd.Series:
    """Safely evaluate a user-defined-variable expression against ``df``.

    Parameters
    ----------
    expr
        The expression (Python syntax) referencing columns of ``df`` and the allowed
        functions ``isna``/``exp``/``log``/``lag``/``lead``.
    df
        The data frame providing the columns.
    entities, time
        Panel identifiers; ``lag``/``lead`` shift within ``entities`` groups ordered by
        ``time``.

    Returns
    -------
    pandas.Series
        The evaluated column, aligned to ``df.index``.

    Raises
    ------
    UDVError
        If the expression is syntactically invalid or uses a disallowed construct.
    """
    entities = list(entities) if entities else []
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - message passthrough
        raise UDVError(f"invalid expression: {exc}") from exc

    work = df
    if entities and time and time in df.columns:
        work = df.sort_values([*entities, time])

    result = _Evaluator(work, entities).visit(tree.body)
    if not isinstance(result, pd.Series):
        # A scalar expression broadcast across all rows.
        result = pd.Series(result, index=work.index)
    if len(result) != len(df):
        raise UDVError("expression did not produce one value per row")
    return result.reindex(df.index)
