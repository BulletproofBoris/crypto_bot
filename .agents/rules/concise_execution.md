---
trigger: always_on
description: Prevent streaming timeouts by keeping reasoning concise and splitting multi-step operations into small atomic turns.
---

# Защита от обрыва стрима (Anti-Timeout Execution Rule)

1. **Короткие блоки размышлений (Thinking):** Не превышайте 2–4 предложения во внутреннем блоке рассуждений перед вызовом инструмента.
2. **Атомарность шагов:** Выполняйте не более 1–2 быстрых вызовов инструментов (`read_file`, `edit_file`, `run_command`) за один шаг.
3. **Разбиение больших задач:** Любую многоэтапную задачу разбивайте на маленькие подзадачи с быстрой промежуточной фиксацией результата.

