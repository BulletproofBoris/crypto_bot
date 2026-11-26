#!/usr/bin/env python3
"""
_tools/update_data_pipeline_96d.py
=============================================================================
Единый мастер-пайплайн обновления рыночных данных и сборки матрицы
Signal Stack v7 (96D) для обоих миров (DEC + RAW) в СТРОГОМ ТОПОЛОГИЧЕСКОМ ПОРЯДКЕ:

  1. sync_all_data.py                (опц. --skip-sync)
  2. prepare_1s_dataset.py --coin prl (опц. --skip-prepare-1s)
  3. build_600h_guard_dataset.py     -> сырой rl_guard_lob_cache_600h.npz
  4. build_order1_alpha_matrix.py    -> order1_alpha_matrix_56d.npz + 84d.npz
  5. launch_dual_order2_wf.py        -> order2_risk_matrix_11d.npz + 16d.npz
  6. train_order1_wave_wf.py         -> order1_wave_matrix_6d.npz
  7. build_hybrid_market_obs_40d -> 60d -> 72d -> 84d (сырой 84D!)
  8. audit_and_extract_live_impact.py (опц.: пропускается при --eval-only)
  9. decontaminate_historical_lob.py -> резервирует RAW (LOB + 84D) и очищает DEC!
 10. train_order25_impact_and_regret.py -> строит 88D DEC + 88D RAW!
 11. build_hybrid_market_obs_96d.py  -> финальные 96D DEC + 96D RAW!

Флаг `--eval-only`:
  НЕ обучает и НЕ перезаписывает ни один боевой .ubj бустер в `models/prl/`,
  а собирает все кэши строго замороженными живыми моделями (для честной Sim/Live сверки).
=============================================================================
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
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
    parser.add_argument("--skip-prepare-1s", action="store_true", help="Skip step 2 & 3 if prl_1s_features & raw 600h LOB are already up to date")
    parser.add_argument("--eval-only", action="store_true", help="Compile all caches with frozen live .ubj boosters (0 model retraining/overwrites)")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate 40d..88d npz caches")
    args = parser.parse_args()

    env = dict(os.environ)
    t_all = time.time()

    if not args.skip_sync:
        _run_step("1/11 Синхронизация свежих данных с VPS / GDrive", [PY, "_tools/sync_all_data.py"], env)

    if not args.skip_prepare_1s:
        _run_step("2/11 Досчёт базового 1-секундного датасета PRL", [PY, "_tools/prepare_1s_dataset.py", "--coin", "prl"], env)
        _run_step("3/11 Сборка сырого 600ч LOB кэша", [PY, "_tools/build_600h_guard_dataset.py"], env)

    _run_step("4/11 Сборка каузальных матриц Order-1 (56D + 84D)", [PY, "_tools/build_order1_alpha_matrix.py"], env)

    o2_cmd = [PY, "_tools/launch_dual_order2_wf.py"] + (["--eval-only"] if args.eval_only else [])
    _run_step("5/11 Матрицы рисков Order-2 (11D + 16D)", o2_cmd, env)

    wave_cmd = [PY, "_tools/train_order1_wave_wf.py"] + (["--eval-only"] if args.eval_only else [])
    _run_step("6/11 Волновая матрица Order-1 Wave (6D)", wave_cmd, env)

    _run_step("7a/11 Сборка слоя 40D", [PY, "_tools/build_hybrid_market_obs_40d.py"], env)
    _run_step("7b/11 Сборка слоя 60D", [PY, "_tools/build_hybrid_market_obs_60d.py"], env)
    _run_step("7c/11 Сборка слоя 72D Wave", [PY, "_tools/build_hybrid_market_obs_72d.py"], env)
    _run_step("7d/11 Сборка сырого слоя 84D O1+O2", [PY, "_tools/build_hybrid_market_obs_84d.py"], env)

    if not args.eval_only:
        _run_step("8/11 Извлечение боевого self-impact из телеметрии", [PY, "_tools/audit_and_extract_live_impact.py"], env)

    _run_step("9/11 Очистка стакана и 84D от собственных исторических ордеров (RAW + DEC)", [PY, "_tools/decontaminate_historical_lob.py"], env)

    o25_cmd = [PY, "_tools/train_order25_impact_and_regret.py"] + (["--eval-only"] if args.eval_only else [])
    _run_step("10/11 Сборка слоя 88D Order-2.5 (DEC + RAW)", o25_cmd, env)

    _run_step("11/11 Финальная сборка Signal Stack v7 (96D DEC + RAW)", [PY, "_tools/build_hybrid_market_obs_96d.py"], env)

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
        print(f"🧹 Удалены промежуточные 40d..88d кэши (освобождено {freed / 1e9:.2f} ГБ). Базовые O1/O2/Wave и чистовые 96D DEC+RAW сохранены!")

    print(f"\n🏁 Сборка данных и кэшей 96D успешно завершена за {(time.time() - t_all) / 60:.1f} мин!")


if __name__ == "__main__":
    main()
