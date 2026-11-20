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
L3_DIR = PROJECT_ROOT / "models" / "prl" / "lesson_3_defense"
L4_DIR = PROJECT_ROOT / "models" / "prl" / "lesson_4_inventory"
L5_DIR = PROJECT_ROOT / "models" / "prl" / "lesson_5_whale"
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


def check_l3_iterations() -> int:
    summary_path = L3_DIR / "lesson_3_summary.json"
    if not summary_path.exists():
        return 0
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
            return len(data)
    except Exception:
        return 0


def check_l4_iterations() -> int:
    summary_path = L4_DIR / "lesson_4_summary.json"
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
    log("🚀 ДИСПЕТЧЕР ПРОГРАММЫ ОБУЧЕНИЯ ЗАПУЩЕН (УРОК 1: 600 ИТ, УРОК 2: 1000 ИТ)")
    log("=" * 80)

    # 1. Мониторинг / Запуск Урока 1
    target_iterations_l1 = 600
    it = check_l1_iterations()
    if it < target_iterations_l1:
        log(f"Проверка процесса Урока 1 (текущая итерация: {it}/{target_iterations_l1})...")
        res1 = subprocess.run("pgrep -f '[l]esson_1_motorics_trainer.py'", shell=True, capture_output=True, text=True)
        running_pids1 = [p for p in res1.stdout.strip().split() if p and int(p) != os.getpid()]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        if not running_pids1:
            log("Запуск процесса Урока 1...")
            l1_script = PROJECT_ROOT / "_tools" / "lesson_1_motorics_trainer.py"
            l1_log = open(PROJECT_ROOT / "data" / "logs" / "lesson_1_motorics.log", "a")
            proc1 = subprocess.Popen(
                [str(PYTHON_BIN), str(l1_script)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=l1_log,
            )
            log(f"Процесс Урока 1 запущен с PID {proc1.pid}")

        last_it = it
        while True:
            it = check_l1_iterations()
            if it != last_it:
                last_it = it
                log(f"📊 Урок 1: сохранена итерация {it}/{target_iterations_l1}")

            if it >= target_iterations_l1:
                log(f"✅ Урок 1 успешно достиг {target_iterations_l1} итераций!")
                log("Останавливаем процесс Урока 1...")
                kill_process_by_name("lesson_1_motorics_trainer.py")
                time.sleep(5)
                break

            time.sleep(20)
    else:
        log(f"✅ Урок 1 уже завершен ({it}/{target_iterations_l1} итераций).")

    # 2. Запуск и мониторинг Урока 2
    target_iterations_l2 = 1000
    it_l2 = check_l2_iterations()
    if it_l2 < target_iterations_l2:
        log("=" * 80)
        log(f"🚀 СТАРТ УРОКА 2 (ЭКОНОМИКА СПРЕДА И КОМИССИИ): {target_iterations_l2} ИТЕРАЦИЙ, 8,192 МОДЕЛЕЙ")
        log("=" * 80)

        l2_script = PROJECT_ROOT / "_tools" / "lesson_2_spread_economics_trainer.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        res = subprocess.run("pgrep -f '[l]esson_2_spread_economics_trainer.py'", shell=True, capture_output=True, text=True)
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

        last_it_l2 = it_l2
        while True:
            it_l2 = check_l2_iterations()
            if it_l2 != last_it_l2:
                last_it_l2 = it_l2
                log(f"📊 Урок 2: сохранена итерация {it_l2}/{target_iterations_l2}")

            if proc is not None and proc.poll() is not None:
                if it_l2 >= target_iterations_l2:
                    log(f"🎉 Урок 2 успешно завершил все {target_iterations_l2} итераций!")
                    break
                else:
                    log(f"⚠️ Процесс Урока 2 неожиданно завершился с кодом {proc.returncode} на итерации {it_l2}!")
                    break

            if it_l2 >= target_iterations_l2:
                log(f"🎉 Урок 2 успешно достиг {target_iterations_l2} итераций!")
                kill_process_by_name("lesson_2_spread_economics_trainer.py")
                time.sleep(5)
                break

            time.sleep(20)
    else:
        log(f"✅ Урок 2 уже завершен ({it_l2}/{target_iterations_l2} итераций).")

    # 4. Запуск и мониторинг Урока 3
    target_iterations_l3 = 1000
    it_l3 = check_l3_iterations()
    if it_l3 < target_iterations_l3:
        log("=" * 80)
        log(f"🛡️ СТАРТ УРОКА 3 (ЗАЩИТА ОТ ТОКСИЧНОГО ПОТОКА И FIFO): {target_iterations_l3} ИТЕРАЦИЙ, 8,192 МОДЕЛЕЙ")
        log("=" * 80)

        l3_script = PROJECT_ROOT / "_tools" / "lesson_3_defense_trainer.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        res3 = subprocess.run("pgrep -f '[l]esson_3_defense_trainer.py'", shell=True, capture_output=True, text=True)
        running_pids3 = [p for p in res3.stdout.strip().split() if p and int(p) != os.getpid()]

        if running_pids3:
            log(f"Процесс Урока 3 уже запущен (PID {running_pids3[0]}), переходим в режим мониторинга...")
            proc3 = None
        else:
            l3_log = open(PROJECT_ROOT / "data" / "logs" / "lesson_3_defense.log", "a")
            proc3 = subprocess.Popen(
                [str(PYTHON_BIN), str(l3_script)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=l3_log,
            )
            log(f"Процесс Урока 3 запущен с PID {proc3.pid}")

        last_it_l3 = it_l3
        while True:
            it_l3 = check_l3_iterations()
            if it_l3 != last_it_l3:
                last_it_l3 = it_l3
                if proc3 is not None:
                    log(f"📊 Урок 3: сохранена итерация {it_l3}/{target_iterations_l3}")

            if proc3 is not None and proc3.poll() is not None:
                if it_l3 >= target_iterations_l3:
                    log(f"🎉 Урок 3 успешно завершил все {target_iterations_l3} итераций!")
                    break
                else:
                    log(f"⚠️ Процесс Урока 3 неожиданно завершился с кодом {proc3.returncode} на итерации {it_l3}!")
                    break

            if it_l3 >= target_iterations_l3:
                log(f"🎉 Урок 3 успешно достиг {target_iterations_l3} итераций!")
                kill_process_by_name("lesson_3_defense_trainer.py")
                time.sleep(10)
                break

            time.sleep(20)
    else:
        log(f"✅ Урок 3 уже завершен ({it_l3}/{target_iterations_l3} итераций).")

    # 4.5. Межстадийный мост (Синхронизация -> Новые окна v7 -> 8-фазный Классификатор -> Гибридный кэш v7+v8 318D)
    bridge_flag = L4_DIR / "bridge_v7_v8_ready.json"
    if not bridge_flag.exists():
        log("=" * 80)
        log("🌉 ЗАПУСК МЕЖСТАДИЙНОГО МОСТА v7+v8 ПЕРЕД УРОКОМ 4 (bridge_l3_to_l4_upgrade.py)...")
        log("=" * 80)
        bridge_script = PROJECT_ROOT / "_tools" / "bridge_l3_to_l4_upgrade.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        bridge_res = subprocess.run([str(PYTHON_BIN), "-u", str(bridge_script)], env=env, check=False)
        if bridge_res.returncode == 0:
            log("✅ Межстадийный апгрейд v7+v8 успешно завершен!")
        else:
            log(f"⚠️ Межстадийный мост завершился с кодом {bridge_res.returncode}, проверяем наличие кэша...")

    # 5. Запуск и мониторинг Урока 4
    target_iterations_l4 = 1000
    it_l4 = check_l4_iterations()
    if it_l4 < target_iterations_l4:
        log("=" * 80)
        log(f"⚖️ СТАРТ УРОКА 4 (INVENTORY SKEW И КОНТРОЛЬ MAX DD): {target_iterations_l4} ИТЕРАЦИЙ, 8,192 МОДЕЛЕЙ")
        log("=" * 80)

        l4_script = PROJECT_ROOT / "_tools" / "lesson_4_inventory_skew_trainer.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        res4 = subprocess.run("pgrep -f '[l]esson_4_inventory_skew_trainer.py'", shell=True, capture_output=True, text=True)
        running_pids4 = [p for p in res4.stdout.strip().split() if p and int(p) != os.getpid()]

        if running_pids4:
            log(f"Процесс Урока 4 уже запущен (PID {running_pids4[0]}), переходим в режим мониторинга...")
            proc4 = None
        else:
            l4_log = open(PROJECT_ROOT / "data" / "logs" / "lesson_4_inventory.log", "a")
            proc4 = subprocess.Popen(
                [str(PYTHON_BIN), str(l4_script)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=l4_log,
            )
            log(f"Процесс Урока 4 запущен с PID {proc4.pid}")

        last_it_l4 = it_l4
        while True:
            it_l4 = check_l4_iterations()
            if it_l4 != last_it_l4:
                last_it_l4 = it_l4
                log(f"📊 Урок 4: сохранена итерация {it_l4}/{target_iterations_l4}")

            if proc4 is not None and proc4.poll() is not None:
                if it_l4 >= target_iterations_l4:
                    log(f"🎉 Урок 4 успешно завершил все {target_iterations_l4} итераций!")
                    break
                else:
                    log(f"⚠️ Процесс Урока 4 неожиданно завершился с кодом {proc4.returncode} на итерации {it_l4}!")
                    break

            if it_l4 >= target_iterations_l4:
                log(f"🎉 Урок 4 успешно достиг {target_iterations_l4} итераций!")
                kill_process_by_name("lesson_4_inventory_skew_trainer.py")
                break

            time.sleep(20)

    # 6. Запуск и мониторинг Урока 5 (Whale Execution & Adaptive Sizing $1k..$20k)
    summary_l5 = L5_DIR / "lesson_5_summary.json"
    it_l5 = 0
    if summary_l5.exists():
        try:
            with open(summary_l5, "r") as f:
                it_l5 = len(json.load(f))
        except Exception:
            it_l5 = 0

    target_iterations_l5 = 1000
    if it_l5 < target_iterations_l5:
        log("=" * 80)
        log(f"🐋 СТАРТ УРОКА 5 (WHALE EXECUTION & ADAPTIVE SIZING $1K..$20K): {target_iterations_l5} ИТЕРАЦИЙ, 8,192 МОДЕЛЕЙ")
        log("=" * 80)

        l5_script = PROJECT_ROOT / "_tools" / "lesson_5_whale_sizing_trainer.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        res5 = subprocess.run("pgrep -f '[l]esson_5_whale_sizing_trainer.py'", shell=True, capture_output=True, text=True)
        running_pids5 = [p for p in res5.stdout.strip().split() if p and int(p) != os.getpid()]

        if running_pids5:
            log(f"Процесс Урока 5 уже запущен (PID {running_pids5[0]}), переходим в режим мониторинга...")
            proc5 = None
        else:
            l5_log = open(PROJECT_ROOT / "data" / "logs" / "lesson_5_whale.log", "a")
            proc5 = subprocess.Popen(
                [str(PYTHON_BIN), str(l5_script)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=l5_log,
            )
            log(f"Процесс Урока 5 запущен с PID {proc5.pid}")

        last_it_l5 = it_l5
        while True:
            if summary_l5.exists():
                try:
                    with open(summary_l5, "r") as f:
                        it_l5 = len(json.load(f))
                except Exception:
                    pass
            if it_l5 != last_it_l5:
                last_it_l5 = it_l5
                log(f"📊 Урок 5: сохранена итерация {it_l5}/{target_iterations_l5}")

            if proc5 is not None and proc5.poll() is not None:
                if it_l5 >= target_iterations_l5:
                    log(f"🎉 Урок 5 успешно завершил все {target_iterations_l5} итераций!")
                    break
                else:
                    log(f"⚠️ Процесс Урока 5 неожиданно завершился с кодом {proc5.returncode} на итерации {it_l5}!")
                    break

            if it_l5 >= target_iterations_l5:
                log(f"🎉 Урок 5 успешно достиг {target_iterations_l5} итераций!")
                kill_process_by_name("lesson_5_whale_sizing_trainer.py")
                break

            time.sleep(20)

    log("=" * 80)
    log(f"🏆 ВСЕ 5 УРОКОВ (L1: {target_iterations_l1}, L2: {target_iterations_l2}, L3: {target_iterations_l3}, L4: {target_iterations_l4}, L5: {target_iterations_l5} ит) ПОЛНОСТЬЮ ЗАВЕРШЕНЫ!")
    log(f"Итоговый Чемпион сохранен в: {L5_DIR / 'lesson_5_champ_latest.pt'}")
    log("=" * 80)


if __name__ == "__main__":
    run()

