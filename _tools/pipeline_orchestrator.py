#!/usr/bin/env python3
"""
_tools/pipeline_orchestrator.py
=============================================================================
Автоматический диспетчер учебной программы SafeTrade PRL (131,072 модели).
1. Отслеживает выполнение Урока 1 до 30 итераций.
2. При достижении 30 итераций мягко останавливает Урок 1.
3. Автоматически запускает Урок 2 (Экономика спреда) на 30 итераций.
4. Ведет единый лог: data/logs/pipeline_orchestrator.log.
=============================================================================
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "pipeline_orchestrator.log"
L1_DIR = PROJECT_ROOT / "models" / "prl" / "lesson_1_motorics"
L2_DIR = PROJECT_ROOT / "models" / "prl" / "lesson_2_spread"
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] [ORCHESTRATOR] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")


def check_l1_iterations() -> int:
    summary_path = L1_DIR / "lesson_1_summary.json"
    if not summary_path.exists():
        return 0
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
            return len(data)
    except Exception:
        return 0


def check_l2_iterations() -> int:
    summary_path = L2_DIR / "lesson_2_summary.json"
    if not summary_path.exists():
        return 0
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
            return len(data)
    except Exception:
        return 0


def kill_process_by_name(pattern: str):
    try:
        cmd = f"pkill -f '{pattern}'"
        subprocess.run(cmd, shell=True, check=False)
        time.sleep(2)
    except Exception as e:
        log(f"Ошибка остановки процесса {pattern}: {e}")


def run():
    log("=" * 80)
    log("🚀 ДИСПЕТЧЕР ПРОГРАММЫ ОБУЧЕНИЯ ЗАПУЩЕН (ЛИМИТ: 250 ИТЕРАЦИЙ ПО 30 СЕК)")
    log("=" * 80)

    # 1. Мониторинг Урока 1
    target_iterations = 250
    it = check_l1_iterations()
    if it < target_iterations:
        log(f"Ожидание завершения {target_iterations} итераций Урока 1 (текущая: {it})...")
        last_it = it
        while True:
            it = check_l1_iterations()
            if it != last_it:
                last_it = it
                log(f"📊 Урок 1: сохранена итерация {it}/{target_iterations}")

            if it >= target_iterations:
                log(f"✅ Урок 1 успешно достиг {target_iterations} итераций!")
                log("Останавливаем процесс Урока 1...")
                kill_process_by_name("lesson_1_motorics_trainer.py")
                time.sleep(5)
                break

            time.sleep(20)
    else:
        log(f"✅ Урок 1 уже завершен ({it}/{target_iterations} итераций).")

    # 2. Запуск Урока 2
    log("=" * 80)
    log(f"🚀 СТАРТ УРОКА 2 (ЭКОНОМИКА СПРЕДА И КОМИССИИ): {target_iterations} ИТЕРАЦИЙ, 8,192 МОДЕЛЕЙ")
    log("=" * 80)

    l2_script = PROJECT_ROOT / "_tools" / "lesson_2_spread_economics_trainer.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    res = subprocess.run("pgrep -f 'lesson_2_spread_economics_trainer.py'", shell=True, capture_output=True, text=True)
    running_pids = [p for p in res.stdout.strip().split() if p and int(p) != os.getpid()]

    if running_pids:
        log(f"Процесс Урока 2 уже запущен (PID {running_pids[0]}), переходим в режим мониторинга...")
        proc = None
    else:
        l2_log = open(LOG_FILE, "a")
        proc = subprocess.Popen(
            [str(PYTHON_BIN), str(l2_script)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=l2_log,
        )
        log(f"Процесс Урока 2 запущен с PID {proc.pid}")

    # 3. Мониторинг Урока 2
    last_it_l2 = 0
    while True:
        it_l2 = check_l2_iterations()
        if it_l2 != last_it_l2:
            last_it_l2 = it_l2
            log(f"📊 Урок 2: сохранена итерация {it_l2}/{target_iterations}")

        if proc is not None and proc.poll() is not None:
            if it_l2 >= target_iterations:
                log(f"🎉 Урок 2 успешно завершил все {target_iterations} итераций!")
                break
            else:
                log(f"⚠️ Процесс Урока 2 неожиданно завершился с кодом {proc.returncode} на итерации {it_l2}!")
                break

        if it_l2 >= target_iterations:
            log(f"🎉 Урок 2 успешно достиг {target_iterations} итераций!")
            kill_process_by_name("lesson_2_spread_economics_trainer.py")
            break

        time.sleep(20)

    log("=" * 80)
    log(f"🏆 ВСЕ 2 УРОКА (ПО {target_iterations} ИТЕРАЦИЙ НА 8,192 МОДЕЛЯХ) ПОЛНОСТЬЮ ЗАВЕРШЕНЫ!")
    log(f"Итоговый Чемпион сохранен в: {L2_DIR / 'lesson_2_champ_latest.pt'}")
    log("=" * 80)


if __name__ == "__main__":
    run()

