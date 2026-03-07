Как воспроизвести результаты (инструкции)

Проверка + генерация данных: check.ipynb

создаёт data/train.jsonl и data/test_L{1..5}.jsonl

Обучение GRPO + curriculum: GRPO_curriculum.ipynb

сохраняет логи в logs/ и адаптеры в runs/

Анализ: Target24_GRPO_analysis.ipynb

строит графики accuracy, forgetting, reward/KL, компоненты наград, анализ ошибок