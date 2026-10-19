#!/usr/bin/env python3
"""
Мониторинг ресурсов VPS и состояния всех служб сборщиков криптовалютного бота.
Поддерживает непрерывный режим и разовый запуск (--once).
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import paramiko

# Импорт конфигурации VPS
sys.path.append(os.path.dirname(__file__))
import vps_config

HOST = vps_config.VPS_HOST
PORT = vps_config.VPS_PORT
USER = vps_config.VPS_USER
PASS = vps_config.VPS_PASS
create_ssh_client = getattr(vps_config, "create_ssh_client", None)

# Пороги предупреждений
MEM_WARNING_PERCENT = 85.0
DISK_WARNING_PERCENT = 90.0

SERVICES = [
    ("safetrade-collector", "SafeTrade (PRL, Quantus)"),
    ("bigone-collector", "BigONE (PRL, QTC)"),
    ("rhea-collector", "Rhea Finance / Near Intents"),
    ("binance-collector", "Binance Macro (BTC + NEAR)"),
    ("onchain-collector", "On-chain Metrics & Rules"),
    ("crypto-paper-trader", "Online Triad Paper Trader (PRL, Quantus)"),
]

def get_server_stats(ssh):
    stats = {}
    
    # 1. Проверка ОЗУ
    stdin, stdout, stderr = ssh.exec_command("free -m | grep Mem")
    mem_line = stdout.read().decode().strip().split()
    if mem_line:
        total_mem = int(mem_line[1])
        available_mem = int(mem_line[6])
        used_mem = total_mem - available_mem
        mem_percent = (used_mem) / total_mem * 100
        stats['mem'] = {
            'total': total_mem,
            'used': used_mem,
            'available': available_mem,
            'percent': mem_percent
        }
        
    # 2. Проверка Диска
    stdin, stdout, stderr = ssh.exec_command("df -h / | tail -n 1")
    disk_line = stdout.read().decode().strip().split()
    if disk_line:
        stats['disk'] = {
            'size': disk_line[1],
            'used': disk_line[2],
            'avail': disk_line[3],
            'percent': float(disk_line[4].replace('%', ''))
        }
        
    # 3. Нагрузка CPU
    stdin, stdout, stderr = ssh.exec_command("uptime")
    uptime_line = stdout.read().decode().strip()
    load_avg = uptime_line.split("load average:")[1].strip() if "load average:" in uptime_line else "N/A"
    stats['load'] = load_avg
    
    # 4. Статусы служб
    stats['services'] = {}
    for s_name, desc in SERVICES:
        stdin, stdout, stderr = ssh.exec_command(f"systemctl is-active {s_name}")
        stats['services'][s_name] = stdout.read().decode().strip()

    # 5. Свежесть и размеры ключевых файлов
    cmd_files = (
        "ls -lh "
        "/opt/crypto_bot/data/realtime/prlusdt/depth.csv "
        "/opt/crypto_bot/data/realtime/prlusdt_bigone/depth_bigone.csv "
        "/opt/crypto_bot/data/realtime/quantususdt/depth.csv "
        "/opt/crypto_bot/data/realtime/quantus_bigone/depth_bigone.csv "
        "/opt/crypto_bot/data/realtime/macro_binance/binance_macro.csv "
        "/opt/crypto_bot/data/realtime/onchain/prl_onchain.csv "
        "/opt/crypto_bot/data/realtime/onchain/quantus_onchain.csv "
        "2>/dev/null | awk '{print $5, $9}'"
    )
    stdin, stdout, stderr = ssh.exec_command(cmd_files)
    stats['files'] = stdout.read().decode().strip()

    # 5. Проверка статуса Online Paper-Trading
    stdin, stdout, stderr = ssh.exec_command("cat /opt/crypto_bot/data/paper_trading_status.json 2>/dev/null")
    paper_raw = stdout.read().decode().strip()
    if paper_raw:
        try:
            stats['paper_trading'] = json.loads(paper_raw)
        except Exception:
            pass

    return stats

def format_stats(stats):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = [f"📊 === СТАТИСТИКА VPS ({HOST}) на {now} ==="]
    
    if 'mem' in stats:
        mem = stats['mem']
        mem_str = f"ОЗУ:  {mem['used']} MB / {mem['total']} MB ({mem['percent']:.1f}%)"
        if mem['percent'] >= MEM_WARNING_PERCENT:
            mem_str = f"⚠️ ВНИМАНИЕ! МАЛО ПАМЯТИ: {mem_str}"
        out.append(mem_str)
        
    if 'disk' in stats:
        disk = stats['disk']
        disk_str = f"Диск: {disk['used']} / {disk['size']} (Свободно: {disk['avail']}, занято {disk['percent']}%)"
        if disk['percent'] >= DISK_WARNING_PERCENT:
            disk_str = f"⚠️ ВНИМАНИЕ! ЗАКАНЧИВАЕТСЯ МЕСТО: {disk_str}"
        out.append(disk_str)
        
    out.append(f"CPU Load Average: {stats.get('load', 'N/A')}")
    out.append("\n🛠️ Статусы служб сбора данных:")
    
    all_ok = True
    for s_name, desc in SERVICES:
        status = stats.get('services', {}).get(s_name, 'unknown')
        if status == 'active':
            icon = "🟢 ACTIVE"
        else:
            icon = f"🔴 {status.upper()}"
            all_ok = False
        out.append(f"  - {s_name:<22} [{icon}] ({desc})")

    if 'paper_trading' in stats:
        pt = stats['paper_trading']
        out.append("\n🎯 ТОРГОВЫЙ СТАТУС ТРИАДЫ (ONLINE PAPER TRADING):")
        out.append(f"  Обновлено: {pt.get('updated_at', 'N/A')}")
        if 'prl' in pt:
            p = pt['prl']
            out.append(f"  • PRL / USDT:     ${p['mid_price']} | Q_PnL: {p['pnl_pct']:+5.2f}% | Order: {p['active_order']:<22} | Fills: {p['fills']} | Tokens: {p['tokens']:,.1f} | USDT: ${p['usdt']:,.2f}")
        if 'quantus' in pt:
            q = pt['quantus']
            out.append(f"  • QUANTUS / USDT: ${q['mid_price']} | Q_PnL: {q['pnl_pct']:+5.2f}% | Order: {q['active_order']:<22} | Fills: {q['fills']} | Tokens: {q['tokens']:,.1f} | USDT: ${q['usdt']:,.2f}")

    if stats.get('files'):
        out.append("\n📁 Размеры файлов данных на сервере:")
        for line in stats['files'].splitlines():
            parts = line.strip().split()
            if len(parts) == 2:
                size, path = parts
                fname = path.split("realtime/")[-1]
                out.append(f"  • {size:>7}  -> {fname}")

    out.append("\n" + ("✅ Все службы работают штатно!" if all_ok else "⚠️ Внимание: некоторые службы остановлены!"))
    return "\n".join(out)

def monitor(once=False, interval=10):
    ssh = None
    try:
        if create_ssh_client:
            ssh = create_ssh_client(timeout=10)
        else:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=10)
        
        if once:
            stats = get_server_stats(ssh)
            print(format_stats(stats))
            return
            
        print("✅ Соединение установлено. Режим непрерывного мониторинга (Ctrl+C для выхода):\n")
        while True:
            stats = get_server_stats(ssh)
            print(format_stats(stats))
            print("-" * 55)
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Мониторинг остановлен.")
    except Exception as e:
        print(f"❌ Ошибка подключения к серверу: {e}")
    finally:
        if ssh is not None:
            ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Мониторинг состояния VPS и сборщиков данных.")
    parser.add_argument("--once", action="store_true", help="Выполнить один замер и завершить работу.")
    parser.add_argument("--interval", type=int, default=10, help="Интервал обновления в секундах (по умолчанию 10).")
    args = parser.parse_args()

    # Если запускается без TTY (например в Jupyter через !), по умолчанию делаем 1 замер
    is_once = args.once or (not sys.stdin.isatty())
    monitor(once=is_once, interval=args.interval)
