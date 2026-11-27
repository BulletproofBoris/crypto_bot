#!/usr/bin/env python3
"""
Безопасная in-place обрезка (copytruncate) лога live_rl_telemetry.jsonl.
Сохраняет тот же самый inode, чтобы живой процесс live_rl_trader.py (открытый с O_APPEND)
продолжал атомарно дописывать строки без утечки unlinked-дескрипторов на диске.
"""
import os

PATH = "/opt/crypto_bot/data/logs/live_rl_telemetry.jsonl"
MAX_BYTES = 25_000_000
KEEP_BYTES = 10_000_000


def main() -> None:
    if not os.path.exists(PATH):
        return
    sz = os.path.getsize(PATH)
    if sz <= MAX_BYTES:
        return
    with open(PATH, "rb") as rf:
        rf.seek(max(0, sz - KEEP_BYTES))
        tail = rf.read()
    nl = tail.find(b"\n")
    if nl >= 0:
        tail = tail[nl + 1 :]
    with open(PATH, "r+b") as wf:
        wf.seek(0)
        wf.write(tail)
        wf.truncate()


if __name__ == "__main__":
    main()
