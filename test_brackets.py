"""
test_brackets.py — Проверка осмысленности скобок в сгенерированных выражениях.

Генерирует примеры для уровней со скобками (3, 5, 7, 9, 10) и проверяет,
что скобки во всех gold_expr действительно меняют результат вычислений.

Запуск:
    python test_brackets.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.target24.env import Target24Env, LEVELS_REQUIRING_MEANINGFUL_PARENS
from envs.target24.verifier import Target24Verifier


def test_meaningful_brackets(level: int, num_samples: int = 200, seed: int = 42):
    """
    Генерирует num_samples задач для данного уровня и проверяет,
    что все gold_expr со скобками реально в них нуждаются.
    """
    env = Target24Env()
    verifier = Target24Verifier()

    print(f"\n{'='*60}")
    print(f"Level {level}: генерируем {num_samples} примеров...")

    data = env.generate(
        num_of_questions=num_samples,
        max_attempts=500,
        difficulty=level,
        seed=seed,
    )

    if len(data) < num_samples:
        print(f"  ⚠️  Удалось сгенерировать только {len(data)} из {num_samples}")

    meaningless_count = 0
    total_with_parens = 0

    for d in data:
        gold_expr = d.metadata["gold_expr"]
        numbers = d.metadata["numbers"]
        ops_str = d.metadata["allowed_ops"]
        allowed_ops = {op for op in ops_str.split() if op in {"+", "-", "*", "/"}}

        # Есть ли скобки в gold_expr?
        if "(" not in gold_expr:
            continue

        total_with_parens += 1

        # Вычисляем значение со скобками
        val_with = verifier.try_eval_expression(
            expr=gold_expr,
            numbers=numbers,
            allowed_ops=allowed_ops,
            allow_parentheses=True,
        )

        # Убираем скобки и вычисляем
        stripped = gold_expr.replace("(", "").replace(")", "")
        val_without = verifier.try_eval_expression(
            expr=stripped,
            numbers=numbers,
            allowed_ops=allowed_ops,
            allow_parentheses=False,
        )

        if val_with is not None and val_without is not None and val_with == val_without:
            meaningless_count += 1
            print(f"  ❌ БЕССМЫСЛЕННЫЕ скобки: {gold_expr} = {val_with}")
            print(f"     Без скобок:           {stripped} = {val_without}")

    if total_with_parens == 0:
        print(f"  ⚠️  Нет выражений со скобками (level {level} может не требовать их)")
    elif meaningless_count == 0:
        print(f"  ✅ Все {total_with_parens} выражений со скобками — осмысленные!")
    else:
        print(f"  ❌ {meaningless_count}/{total_with_parens} выражений с бессмысленными скобками")

    return meaningless_count, total_with_parens


def test_trivial_ops(level: int, num_samples: int = 200, seed: int = 42):
    """Проверяет отсутствие тривиальных операций (*1, /1, +0, -0)."""
    env = Target24Env()

    data = env.generate(
        num_of_questions=num_samples,
        max_attempts=500,
        difficulty=level,
        seed=seed,
    )

    trivial_count = 0
    for d in data:
        expr = d.metadata["gold_expr"]
        # Простая проверка на типичные тривиальные паттерны
        for pattern in ["* 1 ", " 1 *", "/ 1 ", "/ 1)", "+ 0 ", " 0 +", "- 0 ", "- 0)"]:
            if pattern in expr or expr.endswith(pattern.rstrip()):
                trivial_count += 1
                print(f"  ⚠️  Тривиальная операция: {expr}")
                break

    if trivial_count == 0 and len(data) > 0:
        print(f"  ✅ Level {level}: нет тривиальных операций в {len(data)} примерах")
    elif len(data) > 0:
        print(f"  ❌ Level {level}: {trivial_count}/{len(data)} тривиальных операций")

    return trivial_count


def test_verifier_consistency(level: int, num_samples: int = 200, seed: int = 42):
    """Проверяет, что все gold_expr проходят верификацию."""
    env = Target24Env()

    data = env.generate(
        num_of_questions=num_samples,
        max_attempts=500,
        difficulty=level,
        seed=seed,
    )

    failures = 0
    for d in data:
        # Формируем ответ модели в нужном формате
        answer_text = f"<answer>{d.metadata['gold_expr']}</answer>"
        if not env.verify(d, answer_text):
            failures += 1
            print(f"  ❌ Верификация не прошла: {d.metadata['gold_expr']} ≠ {d.answer}")

    if failures == 0 and len(data) > 0:
        print(f"  ✅ Level {level}: все {len(data)} gold_expr проходят верификацию")
    elif len(data) > 0:
        print(f"  ❌ Level {level}: {failures}/{len(data)} не прошли верификацию")

    return failures


def main():
    print("="*60)
    print("ТЕСТ: Осмысленность скобок, тривиальные операции, верификация")
    print("="*60)

    all_ok = True

    # 1. Тест скобок (только уровни со скобками)
    print("\n>>> 1. Тест meaningful скобок")
    for level in sorted(LEVELS_REQUIRING_MEANINGFUL_PARENS):
        bad, total = test_meaningful_brackets(level, num_samples=200, seed=42 + level)
        if bad > 0:
            all_ok = False

    # 2. Тест тривиальных операций (уровни 6+)
    print(f"\n{'='*60}")
    print(">>> 2. Тест тривиальных операций (levels 6-10)")
    for level in range(6, 11):
        bad = test_trivial_ops(level, num_samples=100, seed=42 + level)
        if bad > 0:
            all_ok = False

    # 3. Тест верификации (все уровни)
    print(f"\n{'='*60}")
    print(">>> 3. Тест верификации gold_expr (все уровни)")
    for level in range(1, 11):
        bad = test_verifier_consistency(level, num_samples=100, seed=42 + level)
        if bad > 0:
            all_ok = False

    # 4. Примеры сгенерированных задач
    print(f"\n{'='*60}")
    print(">>> 4. Примеры выражений по уровням")
    env = Target24Env()
    for level in range(1, 11):
        data = env.generate(num_of_questions=5, max_attempts=500, difficulty=level, seed=123)
        print(f"\n  Level {level} ({len(data)} примеров):")
        for d in data:
            m = d.metadata
            print(f"    {m['gold_expr']} = {m['target']}  (numbers: {m['numbers']})")

    print(f"\n{'='*60}")
    if all_ok:
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        print("❌ ЕСТЬ ОШИБКИ — см. выше")
    print("="*60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
