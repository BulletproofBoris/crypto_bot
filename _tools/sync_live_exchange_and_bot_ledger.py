#!/usr/bin/env python3
"""
_tools/sync_live_exchange_and_bot_ledger.py
================================================================================
Быстрый двухфазный контур обновления данных бирж и реального поведения робота (БЕЗ tmux):
  • `--net-download` (сетевая фаза, ~20-40с):
      1) Сжатым потоком gzip через SSH за 2 секунды скачивает свежие логи реальной торговли
         робота с VPS (`/opt/crypto_bot/data/logs/live_rl_trader.log*`), приводит локальный
         архив `full_live_rl_trader_history.log` к строго монотонному порядку и дописывает
         все новые события PLACE / CANCEL / FILL!
      2) Скачивает горячие L500 стаканы и сделки со всех бирж (`sync_all_data.py --download-only --coin prl`).
  • `--local-rebuild` (локальная вычислительная фаза в песочнице):
      1) Склеивает Parquet (`sync_all_data.py --build-only --coin prl`), досчитывает 223 колонки
         (`live_feature_updater.py`) и обновляет кэш стакана (`build_600h_guard_dataset.py`).
      2) Инкрементально дообучает `Write-Once` реестр (`build_l94_write_once_ledger.py`):
         новые 24ч эпохи и боевые окна `Order-1 (12H)`, `Wave (6D)`, `Order-2 (16D)` +
         посуточный каузальный `Order-2.5` на свежих интервенциях робота -> `112D` (DEC + RAW).
================================================================================
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.vps_config import create_ssh_client

HIST_LOG = BASE_DIR / "data" / "vps_live_history" / "full_live_rl_trader_history.log"
SYNC_META = BASE_DIR / "logs" / "l95_9h_marathon" / "last_sync_meta.json"
RE_TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) MSK")
MSK_TZ = timezone(timedelta(hours=3))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_ts(ts_s: str) -> float:
    return datetime.strptime(ts_s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=MSK_TZ).timestamp()


def sync_vps_live_trader_logs_fast() -> dict:
    """Молниеносная выгрузка новых строк боевого лога с VPS через gzip-поток SSH (~2с вместо 3 минут SFTP!)."""
    HIST_LOG.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if HIST_LOG.exists():
        lines = HIST_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()

    split_idx = None
    prev_t = 0.0
    for idx, ln in enumerate(lines):
        m = RE_TS.search(ln)
        if m:
            t_cur = _parse_ts(m.group(1))
            if prev_t > 0.0 and t_cur < prev_t - 86400.0 * 2:
                split_idx = idx
                break
            prev_t = t_cur
    if split_idx is not None:
        _log(f"🔧 Однократная хронологическая нормализация архива логов робота на строке {split_idx:,}...")
        lines = lines[split_idx:] + lines[:split_idx]

    max_local_t = 0.0
    for ln in lines[-60000:]:
        m = RE_TS.search(ln)
        if m:
            t_c = _parse_ts(m.group(1))
            if t_c > max_local_t:
                max_local_t = t_c

    since_date = datetime.fromtimestamp(max(0.0, max_local_t - 600.0), tz=MSK_TZ).strftime("%Y-%m-%d")
    _log(
        f"📡 Сжатая выгрузка логов реальной торговли робота с VPS "
        f"(после {datetime.fromtimestamp(max_local_t, tz=MSK_TZ) if max_local_t > 0 else 'начала'}, фильтр >= {since_date})..."
    )

    ssh = create_ssh_client(timeout=30.0)
    try:
        # Чистая POSIX-выгрузка последних 60 МБ логов (~4+ суток торговли) со сжатием gzip -1 на лету (~2-3 МБ за 2с!):
        remote_cmd = (
            "tail -q -c 35000000 /opt/crypto_bot/data/logs/live_rl_trader.log.1 "
            "/opt/crypto_bot/data/logs/live_rl_trader.log 2>/dev/null | gzip -1"
        )
        _, stdout, _ = ssh.exec_command(remote_cmd, timeout=90)
        gz_bytes = stdout.read()
    finally:
        ssh.close()

    raw_tail = gzip.decompress(gz_bytes).decode("utf-8", errors="ignore") if gz_bytes else ""
    remote_chunks = raw_tail.splitlines()

    appended_lines: list[str] = []
    buf_block: list[str] = []
    new_ticks = 0
    new_places = 0
    new_fills = 0
    new_max_t = max_local_t

    for ln in remote_chunks:
        buf_block.append(ln)
        m = RE_TS.search(ln)
        if m:
            t_blk = _parse_ts(m.group(1))
            if t_blk > max_local_t + 0.001:
                appended_lines.extend(buf_block)
                new_ticks += 1
                for b_ln in buf_block:
                    if "PLACE " in b_ln:
                        new_places += 1
                    elif "FILL " in b_ln:
                        new_fills += 1
                if t_blk > new_max_t:
                    new_max_t = t_blk
            buf_block.clear()

    if appended_lines or split_idx is not None:
        tmp_p = HIST_LOG.with_name(HIST_LOG.name + ".tmp")
        tmp_p.write_text("\n".join(lines + appended_lines) + "\n", encoding="utf-8")
        tmp_p.replace(HIST_LOG)

    res = {
        "compressed_kb": round(len(gz_bytes) / 1024.0, 1),
        "appended_lines": len(appended_lines),
        "new_ticks": new_ticks,
        "new_places": new_places,
        "new_fills": new_fills,
        "latest_msk": str(datetime.fromtimestamp(new_max_t, tz=MSK_TZ)),
    }
    _log(
        f"✅ Лог робота обновлён ({res['compressed_kb']} KB gz): +{len(appended_lines):,} строк "
        f"(+{new_ticks:,} тиков, +{new_places:,} PLACE, +{new_fills:,} FILL) -> до {res['latest_msk']}"
    )
    return res


def run_net_download() -> dict:
    t0 = time.time()
    bot_res = sync_vps_live_trader_logs_fast()
    py = str(BASE_DIR / ".venv" / "bin" / "python")
    _log("⏳ Скачивание горячих стаканов L500 и лент сделок с VPS/GDrive (--download-only --coin prl)...")
    rc = subprocess.run([py, "_tools/sync_all_data.py", "--download-only", "--coin", "prl"], cwd=str(BASE_DIR), check=False).returncode
    if rc != 0:
        raise RuntimeError(f"sync_all_data --download-only завершился с кодом {rc}")
    meta = {"net_elapsed_sec": round(time.time() - t0, 1), "bot_log_sync": bot_res, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    SYNC_META.parent.mkdir(parents=True, exist_ok=True)
    SYNC_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def run_local_rebuild(gpu_id_for_xgb: str = "1") -> dict:
    t0 = time.time()
    py = str(BASE_DIR / ".venv" / "bin" / "python")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BASE_DIR)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id_for_xgb)
    env["L94_GPU_ID"] = str(gpu_id_for_xgb)

    steps = [
        ([py, "_tools/sync_all_data.py", "--build-only", "--coin", "prl"], "Сборка локальных Parquet из скачанных горячих срезов бирж"),
        ([py, "_tools/prepare_1s_dataset.py", "--coin", "prl"], "Инкрементальный расчёт 223 колонок prl_1s_features.parquet"),
        ([py, "_tools/build_600h_guard_dataset.py"], "Обновление тензорного кэша стакана rl_guard_lob_cache_600h.npz"),
        ([py, "_tools/build_l94_write_once_ledger.py"], "Инкрементальное дообучение Write-Once реестра (Order-1 + Wave + Order-2 + Order-2.5 -> 112D)"),
    ]
    for cmd, desc in steps:
        _log(f"⏳ {desc}...")
        ts_s = time.time()
        rc = subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"Сбой на этапе '{desc}': rc={rc}")
        _log(f"  ✅ Завершено за {time.time() - ts_s:.1f}с")

    manifest_p = BASE_DIR / "data" / "cache" / "l94_ledger_manifest.json"
    man = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
    prev_meta = json.loads(SYNC_META.read_text(encoding="utf-8")) if SYNC_META.exists() else {}
    out = {
        **prev_meta,
        "rebuild_elapsed_min": round((time.time() - t0) / 60.0, 2),
        "total_rows": man.get("total_rows", 0),
        "o1_sealed_up_to": man.get("o1_sealed_up_to", 0),
        "o2_sealed_up_to": man.get("o2_sealed_up_to", 0),
        "sealed_112d_sha16": man.get("sealed_112d_sha16", ""),
    }
    SYNC_META.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_incremental_exchange_and_ledger_refresh(gpu_id_for_xgb: str = "1") -> dict:
    run_net_download()
    return run_local_rebuild(gpu_id_for_xgb=gpu_id_for_xgb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-download", action="store_true", help="Только сетевая выгрузка логов робота и стаканов бирж")
    ap.add_argument("--local-rebuild", action="store_true", help="Только локальная сборка Parquet и дообучение Order-1/2/2.5")
    ap.add_argument("--gpu-id", type=str, default="1")
    args = ap.parse_args()

    if args.net_download and not args.local_rebuild:
        res = run_net_download()
    elif args.local_rebuild and not args.net_download:
        res = run_local_rebuild(args.gpu_id)
    else:
        res = run_incremental_exchange_and_ledger_refresh(args.gpu_id)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
