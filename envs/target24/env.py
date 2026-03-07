from __future__ import annotations

from typing import Optional, List, Any, Dict, Tuple
import random

from base.env import Env
from base.data import Data
from envs.target24.verifier import Target24Verifier


# ─────────────────────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────────────────────

def build_prompt(
    numbers: List[str],
    target: str,
    level: int,
    allowed_ops: str = "+ - * /",
    allow_parentheses: bool = True,
) -> str:
    """Формирует текст задачи для модели."""
    parens_rule = (
        "Parentheses are allowed."
        if allow_parentheses
        else "Parentheses are NOT allowed."
    )
    return (
        "You are solving an arithmetic puzzle.\n\n"
        "Rules:\n"
        "- Use each given number EXACTLY ONCE.\n"
        f"- Allowed operations: {allowed_ops}\n"
        "- Do NOT introduce any new numeric literals.\n"
        "- Division by zero is forbidden.\n"
        "- Division is allowed ONLY when it is exact (the result must be an integer).\n"
        f"- {parens_rule}\n\n"
        "Output format:\n"
        "<answer>\n"
        "YOUR_EXPRESSION_HERE\n"
        "</answer>\n\n"
        f"Level: {level}\n"
        f"Numbers: {', '.join(numbers)}\n"
        f"Target: {target}\n"
    )


# ─────────────────────────────────────────────────────────────
#  Level configuration
# ─────────────────────────────────────────────────────────────

# Каждый уровень описывается кортежем:
#   (num_numbers, allow_parentheses, min_number, max_number, max_target)
#
# Level  1: 2 числа, без скобок,  числа 1-9,   target ≤ 1000
# Level  2: 3 числа, без скобок,  числа 1-9,   target ≤ 1000
# Level  3: 3 числа, со скобками, числа 1-9,   target ≤ 1000
# Level  4: 4 числа, без скобок,  числа 1-9,   target ≤ 1000
# Level  5: 4 числа, со скобками, числа 1-9,   target ≤ 1000
# Level  6: 5 чисел, без скобок,  числа 1-99,  target ≤ 1000
# Level  7: 5 чисел, со скобками, числа 1-99,  target ≤ 1000
# Level  8: 6 чисел, без скобок,  числа 1-99,  target ≤ 1000
# Level  9: 6 чисел, со скобками, числа 1-99,  target ≤ 1000
# Level 10: 7 чисел, со скобками, числа 1-99,  target без ограничений

LEVEL_CONFIG: Dict[int, Tuple[int, bool, int, int, int]] = {
    1:  (2, False,  1,  9, 1000),
    2:  (3, False,  1,  9, 1000),
    3:  (3, True,   1,  9, 1000),
    4:  (4, False,  1,  9, 1000),
    5:  (4, True,   1,  9, 1000),
    6:  (5, False,  1, 99, 1000),
    7:  (5, True,   1, 99, 1000),
    8:  (6, False,  1, 99, 1000),
    9:  (6, True,   1, 99, 1000),
    10: (7, True,   1, 99, 999_999),   # без ограничений на target
}

MAX_LEVEL = max(LEVEL_CONFIG.keys())

# Уровни, на которых скобки обязательны (gold expression должен содержать
# скобки, которые реально влияют на результат).
LEVELS_REQUIRING_MEANINGFUL_PARENS = {3, 5, 7, 9, 10}


# ─────────────────────────────────────────────────────────────
#  Trivial-operation filter
# ─────────────────────────────────────────────────────────────

def _is_trivial_binop(a_val: int, op: str, b_val: int) -> bool:
    """
    Возвращает True, если бинарная операция тривиальна и не добавляет
    сложности задаче:  *1, /1, +0, -0.
    """
    if op == "*" and (a_val == 1 or b_val == 1):
        return True
    if op == "/" and b_val == 1:
        return True
    if op in ("+", "-") and b_val == 0:
        return True
    if op == "+" and a_val == 0:
        return True
    return False


# ─────────────────────────────────────────────────────────────
#  Random binary-tree expression builder
# ─────────────────────────────────────────────────────────────

def _random_binary_tree(rng: random.Random, leaves: List[str]) -> list:
    """
    Строит случайное бинарное дерево из списка листьев.

    Рекурсивно разбивает список на две непустые части в случайной точке,
    формируя полное бинарное дерево.  Каждый лист — строка (число).
    Внутренние узлы — списки [left, op_placeholder, right].

    Пример для 4 листьев ["2", "3", "5", "7"]:
      [["2", None, "3"], None, ["5", None, "7"]]
    Placeholder (None) позже заменяется на оператор.
    """
    if len(leaves) == 1:
        return leaves[0]  # type: ignore[return-value]

    # Выбираем случайную точку разбиения (от 1 до len-1)
    split = rng.randint(1, len(leaves) - 1)
    left = _random_binary_tree(rng, leaves[:split])
    right = _random_binary_tree(rng, leaves[split:])
    return [left, None, right]


def _fill_ops(rng: random.Random, tree, allowed_ops: List[str]) -> None:
    """Заменяет все None-плейсхолдеры в дереве на случайные операторы."""
    if isinstance(tree, str):
        return
    tree[1] = rng.choice(allowed_ops)
    _fill_ops(rng, tree[0], allowed_ops)
    _fill_ops(rng, tree[2], allowed_ops)


def _tree_to_expr(tree, parent_op: Optional[str] = None, is_right: bool = False) -> str:
    """
    Преобразует бинарное дерево в строковое выражение с минимально
    необходимыми скобками (скобки ставятся только там, где они
    меняют порядок вычислений из-за приоритета/ассоциативности).
    """
    if isinstance(tree, str):
        return tree

    left, op, right = tree
    left_str = _tree_to_expr(left, op, is_right=False)
    right_str = _tree_to_expr(right, op, is_right=True)
    expr = f"{left_str} {op} {right_str}"

    # Определяем, нужны ли скобки вокруг данного поддерева
    if parent_op is not None and _needs_parens(op, parent_op, is_right):
        return f"({expr})"
    return expr


def _op_precedence(op: str) -> int:
    """Приоритет оператора: * и / имеют приоритет 2, + и - — приоритет 1."""
    if op in ("*", "/"):
        return 2
    return 1


def _needs_parens(child_op: str, parent_op: str, is_right_child: bool) -> bool:
    """
    Определяет, нужны ли скобки вокруг поддерева с оператором child_op,
    если оно является дочерним узлом под parent_op.

    Скобки нужны когда:
    1. Приоритет child_op ниже, чем parent_op
    2. Приоритеты равны, но child — правый операнд и
       оператор не является чисто коммутативно-ассоциативным
       (вычитание и деление справа требуют скобок)
    """
    child_prec = _op_precedence(child_op)
    parent_prec = _op_precedence(parent_op)

    if child_prec < parent_prec:
        return True
    if child_prec == parent_prec and is_right_child:
        # a - (b + c) ≠ a - b + c;  a / (b * c) ≠ a / b * c
        if parent_op in ("-", "/"):
            return True
    return False


def _tree_to_forced_parens_expr(tree) -> str:
    """
    Преобразует дерево в выражение, где каждый внутренний узел
    обёрнут в скобки (кроме корня).  Это гарантирует наличие скобок
    для уровней, где скобки обязательны.
    """
    if isinstance(tree, str):
        return tree
    left, op, right = tree
    left_str = _tree_to_forced_parens_expr(left)
    right_str = _tree_to_forced_parens_expr(right)
    # Если поддерево — не лист, оборачиваем в скобки
    if isinstance(tree[0], list):
        left_str = f"({left_str})"
    if isinstance(tree[2], list):
        right_str = f"({right_str})"
    return f"{left_str} {op} {right_str}"


def _eval_tree(tree) -> Optional[int]:
    """
    Вычисляет значение дерева с целочисленной арифметикой.
    Возвращает None если деление не точное или деление на ноль.
    """
    if isinstance(tree, str):
        return int(tree)

    left_val = _eval_tree(tree[0])
    right_val = _eval_tree(tree[2])
    if left_val is None or right_val is None:
        return None

    op = tree[1]
    if op == "+":
        return left_val + right_val
    if op == "-":
        return left_val - right_val
    if op == "*":
        return left_val * right_val
    if op == "/":
        if right_val == 0 or left_val % right_val != 0:
            return None
        return left_val // right_val
    return None


def _has_trivial_ops(tree) -> bool:
    """Проверяет, содержит ли дерево тривиальные операции (*1, /1, +0, -0)."""
    if isinstance(tree, str):
        return False

    left_val = _eval_tree(tree[0])
    right_val = _eval_tree(tree[2])
    op = tree[1]

    if left_val is not None and right_val is not None:
        if _is_trivial_binop(left_val, op, right_val):
            return True

    return _has_trivial_ops(tree[0]) or _has_trivial_ops(tree[2])


# ─────────────────────────────────────────────────────────────
#  Main environment class
# ─────────────────────────────────────────────────────────────

class Target24Env(Env):
    """
    Target24 — среда для арифметических головоломок с 10 уровнями
    возрастающей сложности (curriculum learning).

    Уровни 1-5:  2-4 числа (1-9), без/со скобками
    Уровни 6-10: 5-7 чисел (1-99), без/со скобками, бо́льшие targets

    Все операции — целочисленные. Деление допускается только точное.
    На уровнях со скобками (3, 5, 7, 9, 10) генерируются только
    выражения, где скобки реально меняют результат (meaningful).
    """

    def __init__(self, name: str = "target24"):
        super().__init__(name=name, verifier=Target24Verifier)

    def extract_answer(self, test_solution: str) -> str:
        return self.verifier.extract_answer(test_solution)

    # ──────── public: генерация задач ────────

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 200,
        difficulty: Optional[int] = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Data]:
        """
        Генерирует решаемые задачи для заданного уровня сложности.

        Генератор сначала строит случайное выражение, затем вычисляет его
        значение и использует как target. Это гарантирует, что у каждой
        задачи есть решение (gold_expr).

        Args:
            num_of_questions: сколько задач сгенерировать
            max_attempts: макс. попыток на одну задачу
            difficulty: уровень сложности (1-10)
            seed: seed для воспроизводимости

        Optional kwargs:
            allowed_ops: str — разрешённые операции (default: "+ - * /")
            min_target: int — минимальное значение target (default: 0)
            filter_trivial: bool — фильтровать *1, /1, +0, -0 (default: True для level≥6)
        """
        rng = random.Random(seed)

        level = int(difficulty or 1)
        level = max(1, min(MAX_LEVEL, level))

        # Получаем конфигурацию уровня
        num_numbers, allow_parentheses, min_number, max_number, max_target_cfg = \
            LEVEL_CONFIG[level]

        # Пользовательские перезаписи (если нужны)
        allowed_ops_str = str(kwargs.get("allowed_ops", "+ - * /")).strip()
        allowed_ops = [op for op in allowed_ops_str.split() if op in {"+", "-", "*", "/"}]
        if not allowed_ops:
            allowed_ops = ["+", "-", "*", "/"]
            allowed_ops_str = "+ - * /"

        min_target = int(kwargs.get("min_target", 0))
        max_target = int(kwargs.get("max_target", max_target_cfg))

        # Фильтрация тривиальных операций: по умолчанию включена для level ≥ 6
        filter_trivial = bool(kwargs.get("filter_trivial", level >= 6))

        # Нужны ли meaningful скобки на этом уровне
        require_meaningful_parens = level in LEVELS_REQUIRING_MEANINGFUL_PARENS

        out: List[Data] = []
        hard_cap = max(50_000, num_of_questions * max_attempts)

        attempts = 0
        while len(out) < num_of_questions and attempts < hard_cap:
            attempts += 1
            inst = self._build_instance(
                rng=rng,
                level=level,
                num_numbers=num_numbers,
                allowed_ops=allowed_ops,
                allow_parentheses=allow_parentheses,
                min_number=min_number,
                max_number=max_number,
                min_target=min_target,
                max_target=max_target,
                require_meaningful_parens=require_meaningful_parens,
                filter_trivial=filter_trivial,
            )
            if inst is None:
                continue

            numbers, target, gold_expr = inst
            question = build_prompt(
                numbers=numbers,
                target=target,
                level=level,
                allowed_ops=allowed_ops_str,
                allow_parentheses=allow_parentheses,
            )

            metadata: Dict[str, Any] = {
                "numbers": numbers,
                "target": target,
                "allowed_ops": allowed_ops_str,
                "allow_parentheses": allow_parentheses,
                "level": level,
                "gold_expr": gold_expr,
            }

            out.append(Data(
                question=question,
                answer=target,
                difficulty=level,
                metadata=metadata,
            ))

        return out

    # ──────── internal: построение одной задачи ────────

    def _build_instance(
        self,
        rng: random.Random,
        level: int,
        num_numbers: int,
        allowed_ops: List[str],
        allow_parentheses: bool,
        min_number: int,
        max_number: int,
        min_target: int,
        max_target: int,
        require_meaningful_parens: bool,
        filter_trivial: bool,
    ) -> Optional[Tuple[List[str], str, str]]:
        """
        Пытается построить одну задачу.

        Генерирует случайные числа и выражение, проверяет все ограничения.
        Возвращает (numbers, target, gold_expr) или None.
        """
        nums = [rng.randint(min_number, max_number) for _ in range(num_numbers)]
        numbers = [str(n) for n in nums]

        if allow_parentheses:
            # Строим дерево и делаем из него выражение со скобками
            tree = _random_binary_tree(rng, list(numbers))
            _fill_ops(rng, tree, allowed_ops)

            # Проверка: дерево должно вычисляться (нет деления на 0 и т.д.)
            val = _eval_tree(tree)
            if val is None:
                return None

            # Фильтрация тривиальных операций
            if filter_trivial and _has_trivial_ops(tree):
                return None

            # Проверка диапазона target
            if not (min_target <= val <= max_target):
                return None

            if require_meaningful_parens:
                # Выражение со скобками (принудительные скобки на каждом узле)
                expr_with_parens = _tree_to_forced_parens_expr(tree)
                # Выражение без скобок — убираем ВСЕ скобки и оцениваем
                # по стандартному приоритету операций
                expr_flat = expr_with_parens.replace("(", "").replace(")", "")

                # Если выражение без скобок вычисляется так же,
                # значит скобки не нужны → reject
                val_without = self.verifier.try_eval_expression(
                    expr=expr_flat,
                    numbers=numbers,
                    allowed_ops=set(allowed_ops),
                    allow_parentheses=False,
                )
                if val_without is not None and val_without == val:
                    return None

                expr = expr_with_parens
            else:
                # Уровень со скобками, но без требования meaningful
                # (сейчас таких нет, но на всякий случай)
                expr = _tree_to_forced_parens_expr(tree)
        else:
            # Линейное выражение (без скобок)
            tree = None
            expr = self._build_linear_expr(rng, numbers, allowed_ops)

            # Для линейных выражений тоже проверяем тривиальность
            if filter_trivial:
                # Парсим в дерево для проверки
                # (линейное выражение можно проверить проще — через соседние пары)
                if self._has_trivial_linear_ops(numbers, expr):
                    return None

        # Финальная валидация через verifier
        val = self.verifier.try_eval_expression(
            expr=expr,
            numbers=numbers,
            allowed_ops=set(allowed_ops),
            allow_parentheses=allow_parentheses,
        )
        if val is None:
            return None

        if not (min_target <= val <= max_target):
            return None

        return numbers, str(val), expr

    # ──────── static helpers ────────

    @staticmethod
    def _build_linear_expr(
        rng: random.Random,
        numbers: List[str],
        allowed_ops: List[str],
    ) -> str:
        """Строит линейное выражение без скобок: a op1 b op2 c ..."""
        ops = [rng.choice(allowed_ops) for _ in range(len(numbers) - 1)]
        parts: List[str] = []
        for i, n in enumerate(numbers):
            parts.append(n)
            if i < len(ops):
                parts.append(ops[i])
        return " ".join(parts)

    @staticmethod
    def _has_trivial_linear_ops(numbers: List[str], expr: str) -> bool:
        """
        Проверяет наличие тривиальных операций в линейном выражении.
        Анализирует пары (число, оператор, число) в выражении.
        """
        tokens = expr.split()
        # tokens: [num, op, num, op, num, ...]
        for i in range(0, len(tokens) - 2, 2):
            try:
                a_val = int(tokens[i])
                op = tokens[i + 1]
                b_val = int(tokens[i + 2])
                if _is_trivial_binop(a_val, op, b_val):
                    return True
            except (ValueError, IndexError):
                continue
        return False