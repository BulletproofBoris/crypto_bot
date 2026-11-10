#!/usr/bin/env python3
"""
_tools/deploy_vps.py
=============================================================================
Единый скрипт синхронизации и мониторинга боевого бота на VPS (81.19.137.18).
Запускается ВСЕГДА одной и той же командой без аргументов:
    /home/restorator/crypto_bot/.venv/bin/python _tools/deploy_vps.py

Автоматически определяет по локальному стейту (data/logs/.vps_sync_state.json):
1. Изменился ли код (_tools/live_rl_trader.py и др.) -> загружает и перезапускает сервис.
2. Изменился ли чемпион (models/prl/release_champion_v1.pt) -> загружает для Hot-Reload без рестарта.
3. Всегда выводит статус сервиса, сводку кошелька/слотов и последние 50 строк лога.
=============================================================================
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.vps_config import VPS_HOST, create_ssh_client

STATE_PATH = BASE_DIR / "data" / "logs" / ".vps_sync_state.json"
CODE_FILES = (
    "_tools/vps_config.py",
    "_tools/live_rl_trader.py",
    "_tools/live_feature_updater.py",
    "_tools/vectorized_free_multi_order_env.py",
    "_tools/swarm_router.py",
    "_tools/learned_regime_router.py",
    "_tools/multitask_lob_net.py",
    "_tools/fast_neural_engine.py",
)
CHAMP_FILE = "models/prl/release_champion_v1.pt"


def _file_sig(p: Path) -> str:
    if not p.exists():
        return ""
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _remote_atomic_mv(ssh, tmp_rem: str, fin_rem: str) -> None:
    cmd = f"mv -f {shlex.quote(tmp_rem)} {shlex.quote(fin_rem)}"
    _, out_ch, err_ch = ssh.exec_command(cmd)
    rc = out_ch.channel.recv_exit_status()
    if rc != 0:
        err_msg = err_ch.read().decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Remote mv failed (rc={rc}): {err_msg}")


def main():
    prev_state = {}
    if STATE_PATH.exists():
        try:
            prev_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            prev_state = {}

    cur_state = {}
    changed_code = []
    for rel_f in CODE_FILES:
        sig = _file_sig(BASE_DIR / rel_f)
        cur_state[rel_f] = sig
        if sig and prev_state.get(rel_f) != sig:
            changed_code.append(rel_f)

    champ_sig = _file_sig(BASE_DIR / CHAMP_FILE)
    cur_state[CHAMP_FILE] = champ_sig
    changed_champ = bool(champ_sig and prev_state.get(CHAMP_FILE) != champ_sig)

    print(f"🔌 Подключение к боевому VPS ({VPS_HOST or 'configured host'})...")
    ssh = create_ssh_client(timeout=15.0)
    try:
        if changed_code or changed_champ:
            sftp = ssh.open_sftp()
            try:
                for rel_f in changed_code:
                    loc = BASE_DIR / rel_f
                    tmp_rem = f"/opt/crypto_bot/{rel_f}.tmp"
                    fin_rem = f"/opt/crypto_bot/{rel_f}"
                    sftp.put(str(loc), tmp_rem)
                    _remote_atomic_mv(ssh, tmp_rem, fin_rem)
                    print(f"✅ [VPS SYNC CODE] {rel_f} -> {fin_rem}")

                if changed_champ:
                    loc_c = BASE_DIR / CHAMP_FILE
                    tmp_c = f"/opt/crypto_bot/{CHAMP_FILE}.tmp"
                    fin_c = f"/opt/crypto_bot/{CHAMP_FILE}"
                    sftp.put(str(loc_c), tmp_c)
                    _remote_atomic_mv(ssh, tmp_c, fin_c)
                    print(f"🔥 [VPS SYNC CHAMP] {loc_c.name} (sig={champ_sig}) -> {fin_c} (Hot-Reload)")
            finally:
                sftp.close()

        if changed_code:
            _, out_r, _ = ssh.exec_command(
                "systemctl restart live-rl-prl-l9.service && sleep 6 && systemctl is-active live-rl-prl-l9.service"
            )
            print(f"🔄 [VPS SERVICE RESTART]: {out_r.read().decode('utf-8', errors='ignore').strip()}")

        _atomic_write_json(STATE_PATH, cur_state)

        cmd = (
            "echo '=== SERVICE ==='; systemctl is-active live-rl-prl-l9.service; "
            "echo '=== FILLS & HOT-RELOADS ==='; grep -E '24-SLOT FILL|HOT-RELOAD CHAMPION' /opt/crypto_bot/data/logs/live_rl_trader.log | tail -n 20; "
            "echo '=== LOG (last 40 lines) ==='; tail -n 40 /opt/crypto_bot/data/logs/live_rl_trader.log"
        )
        _, out_t, _ = ssh.exec_command(cmd)
        print(out_t.read().decode("utf-8", errors="ignore").strip())
    finally:
        ssh.close()

    # Автоматическая фиксация и отправка (git commit & push) кода и актуального Чемпиона в Git
    try:
        subprocess.run(
            ["git", "add", "STATUS_CONTEXT.md", "models/prl/release_champion_v1.pt", "_tools"],
            cwd=str(BASE_DIR),
            check=False,
            capture_output=True,
            text=True,
        )
        diff_res = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(BASE_DIR),
            check=False,
        )
        if diff_res.returncode != 0:
            c_res = subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Auto-sync 24-slot engine & release_champion_v1.pt (sig={champ_sig})",
                ],
                cwd=str(BASE_DIR),
                check=False,
                capture_output=True,
                text=True,
            )
            p_res = subprocess.run(
                ["git", "push"],
                cwd=str(BASE_DIR),
                check=False,
                capture_output=True,
                text=True,
            )
            print(f"📦 [GIT PUSH]: rc={p_res.returncode} | {c_res.stdout.strip().splitlines()[-1] if c_res.stdout.strip() else ''} | {p_res.stderr.strip()}")
    except Exception as e_git:
        print(f"⚠️ [GIT PUSH WARN]: {e_git}")


if __name__ == "__main__":
    main()
