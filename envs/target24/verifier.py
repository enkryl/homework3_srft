from __future__ import annotations

import ast
import re
from typing import List, Optional, Set, Tuple

from base.verifier import Verifier
from base.data import Data

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


class Target24Verifier(Verifier):
    """
    Integer arithmetic verifier.

    Expects:
      - data.answer: target integer as string, e.g. "24"
      - data.metadata["numbers"]: list[str], e.g. ["3","3","8","8"]
      - data.metadata["allowed_ops"]: string like "+ - * /"
      - data.metadata["allow_parentheses"]: bool
    """

    def extract_answer(self, test_solution: str) -> str:
        m = ANSWER_RE.search(test_solution or "")
        return m.group(1).strip() if m else ""

    def verify(self, data: Data, test_answer: str) -> bool:
        expr = self.extract_answer(test_answer)
        expr = self._preprocess_expr(expr)
        if not expr:
            return False

        meta = data.metadata or {}
        numbers: List[str] = list(meta.get("numbers", []))
        allowed_ops_str: str = str(meta.get("allowed_ops", "+ - * /"))
        allowed_ops: Set[str] = {op for op in allowed_ops_str.split() if op in {"+", "-", "*", "/"}}
        if not allowed_ops:
            allowed_ops = {"+", "-", "*", "/"}

        allow_parentheses = bool(meta.get("allow_parentheses", True))

        try:
            target = int(str(data.answer).strip())
        except Exception:
            return False

        val = self.try_eval_expression(
            expr=expr,
            numbers=numbers,
            allowed_ops=allowed_ops,
            allow_parentheses=allow_parentheses,
        )
        return val is not None and val == target

    # ---------------- public helper (used by env generation) ----------------

    def try_eval_expression(
        self,
        expr: str,
        numbers: List[str],
        allowed_ops: Set[str],
        allow_parentheses: bool,
    ) -> Optional[int]:
        expr = self._preprocess_expr(expr)

        if not expr:
            return None

        if not allow_parentheses and ("(" in expr or ")" in expr):
            return None

        try:
            node = ast.parse(expr, mode="eval")
        except SyntaxError:
            return None

        leaves: List[str] = []
        ok, binops = self._validate_and_collect(node.body, leaves, allowed_ops)
        if not ok:
            return None

        if sorted(leaves) != sorted(numbers):
            return None

        if binops != max(0, len(numbers) - 1):
            return None

        try:
            return self._eval_int(node.body, allowed_ops)
        except Exception:
            return None

    # ---------------- internal helpers ----------------

    @staticmethod
    def _preprocess_expr(expr: str) -> str:
        expr = (expr or "").strip()
        if not expr:
            return ""

        # Allow outputs like "2+3=5" or "2+3 == 5" (we take LHS)
        if "==" in expr:
            expr = expr.split("==", 1)[0].strip()
        elif "=" in expr:
            expr = expr.split("=", 1)[0].strip()

        expr = expr.rstrip(";").strip()
        return expr

    def _validate_and_collect(
        self,
        n: ast.AST,
        leaves: List[str],
        allowed_ops: Set[str],
    ) -> Tuple[bool, int]:
        """
        Validate AST and collect leaf constants as strings.
        Returns (ok, binop_count).
        """
        if isinstance(n, ast.Constant):
            if not isinstance(n.value, int):
                return False, 0
            leaves.append(str(n.value))
            return True, 0

        if isinstance(n, ast.UnaryOp):
            return False, 0

        if isinstance(n, ast.BinOp):
            op_sym = None
            if isinstance(n.op, ast.Add):
                op_sym = "+"
            elif isinstance(n.op, ast.Sub):
                op_sym = "-"
            elif isinstance(n.op, ast.Mult):
                op_sym = "*"
            elif isinstance(n.op, ast.Div):
                op_sym = "/"

            if op_sym is None or op_sym not in allowed_ops:
                return False, 0

            ok_l, c_l = self._validate_and_collect(n.left, leaves, allowed_ops)
            if not ok_l:
                return False, 0
            ok_r, c_r = self._validate_and_collect(n.right, leaves, allowed_ops)
            if not ok_r:
                return False, 0
            return True, 1 + c_l + c_r

        return False, 0

    def _eval_int(self, n: ast.AST, allowed_ops: Set[str]) -> int:
        if isinstance(n, ast.Constant):
            return int(n.value)

        if isinstance(n, ast.BinOp):
            a = self._eval_int(n.left, allowed_ops)
            b = self._eval_int(n.right, allowed_ops)

            if isinstance(n.op, ast.Add) and "+" in allowed_ops:
                return a + b
            if isinstance(n.op, ast.Sub) and "-" in allowed_ops:
                return a - b
            if isinstance(n.op, ast.Mult) and "*" in allowed_ops:
                return a * b
            if isinstance(n.op, ast.Div) and "/" in allowed_ops:
                if b == 0:
                    raise ZeroDivisionError()
                if a % b != 0:
                    raise ValueError("Non-exact division")
                return a // b

        raise ValueError("Invalid AST")