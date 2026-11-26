#!/usr/bin/env python3
"""
_tools/update_data_pipeline_96d.py
=============================================================================
Единый мастер-пайплайн обновления рыночных данных, каузальных Walk-Forward
бустеров (Order-1, Order-2, Wave, Order-2.5) и пересборки матрицы Signal Stack v7 (96D)
для обоих миров (DEC + RAW) с автоматической очисткой промежуточных кэшей.

Использование:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python _tools/update_data_pipeline_96d.py --device cuda:0
=============================================================================
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parent.parent
PY = str(BASE_DIR / ".venv" / "bin" / "python")


def _run_step(title: str, cmd: list[str], env: dict[str, str]) -> None:
    t0 = time.time()
    print(f"\n{'=' * 84}\n▶ {title}\n  CMD: {' '.join(cmd)}\n{'=' * 84}", flush=True)
    rc = subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"Step failed (rc={rc}): {title}")
    print(f"✅ Завершено за {time.time() - t0:.1f}с: {title}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full causal data & 96D Signal Stack v7 rebuild pipeline")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device inside visible mask")
    parser.add_argument("--skip-sync", action="store_true", help="Skip VPS/GDrive raw data sync")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate 40d..88d npz caches")
    args = parser.parse_args()

    env = dict(os.environ)
    t_all = time.time()

    if not args.skip_sync:
        _run_step("1/9 Синхронизация свежих данных с VPS / GDrive", [PY, "_tools/sync_all_data.py"], env)
        _run_step("2/9 Выравнивание 4 площадок и паркетов", [PY, "_tools/add_gate_ourbit_to_parquet.py"], env)

    _run_step("3/9 Сборка 600ч LOB кэша и декантаминация собственных сделок (RAW -> DEC)", [PY, "_tools/build_600h_guard_dataset.py"], env)
    _run_step("3b/9 Очистка исторического стакана от self-impact", [PY, "_tools/decontaminate_historical_lob.py"], env)

    _run_step("4/9 Дообучение новых Walk-Forward окон Order-1 / Order-2 / Wave", [PY, "_tools/train_order1_wave_wf.py"], env)
    _run_step("5/9 Сборка каузальных матриц Order-1 (84D) и Order-2 (16D)", [PY, "_tools/build_order1_alpha_matrix.py"], env)

    _run_step("6/9 Сборка базовых слоёв 40D -> 60D -> 72D -> 84D", [PY, "_tools/build_hybrid_market_obs_40d.py"], env)
    _run_step("6b/9 Сборка 60D", [PY, "_tools/build_hybrid_market_obs_60d.py"], env)
    _run_step("6c/9 Сборка 72D Wave", [PY, "_tools/build_hybrid_market_obs_72d.py"], env)
    _run_step("6d/9 Сборка 84D O1+O2", [PY, "_tools/build_hybrid_market_obs_84d.py"], env)

    _run_step("7/9 Извлечение боевого self-impact и обучение Order-2.5 (84D -> 88D DEC+RAW)", [PY, "_tools/audit_and_extract_live_impact.py"], env)
    _run_step("8/9 Построение 88D кэша Order-2.5", [PY, "_tools/train_order25_impact_and_regret.py"], env)

    _run_step("9/9 Финальная сборка Signal Stack v7 (96D DEC + RAW)", [PY, "_tools/build_hybrid_market_obs_96d.py"], env)

    if not args.keep_intermediate:
        cdir = BASE_DIR / "data" / "cache"
        freed = 0
        for pat in (
            "rl_hybrid_market_obs_40d*.npz",
            "rl_hybrid_market_obs_60d*.npz",
            "rl_hybrid_market_obs_72d*.npz",
            "rl_hybrid_market_obs_84d*.npz",
            "rl_hybrid_market_obs_88d*.npz",
            "rl_hybrid_market_obs_94d*.npz",
        ):
            for f in cdir.glob(pat):
                freed += f.stat().st_size
                f.unlink(missing_ok=True)
        print(f"🧹 Удалены промежуточные кэши (освобождено {freed / 1e9:.2f} ГБ). Оставлены чистовые 96D DEC + RAW!")

    print(f"\n🏁 Полное обновление данных и кэшей 96D завершено за {(time.time() - t_all) / 60:.1f} мин!")


if __name__ == "__main__":
    main()
