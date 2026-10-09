import paramiko
import time
import sys
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(__file__))
import vps_config

HOST = vps_config.VPS_HOST
PORT = vps_config.VPS_PORT
USER = vps_config.VPS_USER
PASS = vps_config.VPS_PASS

# Thresholds for warnings
MEM_WARNING_PERCENT = 85.0
DISK_WARNING_PERCENT = 90.0

def get_server_stats(ssh):
    stats = {}
    
    # Check RAM
    stdin, stdout, stderr = ssh.exec_command("free -m | grep Mem")
    mem_line = stdout.read().decode().strip().split()
    if mem_line:
        total_mem = int(mem_line[1])
        used_mem = int(mem_line[2])
        free_mem = int(mem_line[3])
        buffers_cache = int(mem_line[5])
        available_mem = int(mem_line[6])
        
        mem_percent = (total_mem - available_mem) / total_mem * 100
        stats['mem'] = {
            'total': total_mem,
            'used': total_mem - available_mem,
            'available': available_mem,
            'percent': mem_percent
        }
        
    # Check Disk
    stdin, stdout, stderr = ssh.exec_command("df -h / | tail -n 1")
    disk_line = stdout.read().decode().strip().split()
    if disk_line:
        stats['disk'] = {
            'size': disk_line[1],
            'used': disk_line[2],
            'avail': disk_line[3],
            'percent': float(disk_line[4].replace('%', ''))
        }
        
    # Check CPU Load
    stdin, stdout, stderr = ssh.exec_command("uptime")
    uptime_line = stdout.read().decode().strip()
    load_avg = uptime_line.split("load average:")[1].strip() if "load average:" in uptime_line else "N/A"
    stats['load'] = load_avg
    
    # Check collectors status
    stdin, stdout, stderr = ssh.exec_command("systemctl is-active safetrade-collector")
    stats['st_status'] = stdout.read().decode().strip()

    stdin, stdout, stderr = ssh.exec_command("systemctl is-active bigone-collector")
    stats['bo_status'] = stdout.read().decode().strip()

    # Check hot files sizes
    stdin, stdout, stderr = ssh.exec_command("ls -lh /opt/crypto_bot/data/realtime/prlusdt/depth.csv /opt/crypto_bot/data/realtime/prlusdt_bigone/depth_bigone.csv 2>/dev/null | awk '{print $9, $5}'")
    stats['hot_files'] = stdout.read().decode().strip()
    
    return stats

def monitor():
    print(f"📡 Подключение к серверу {HOST} для мониторинга...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=10)
        print("✅ Соединение установлено. Нажмите Ctrl+C для выхода.\n")
        
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stats = get_server_stats(ssh)
            
            # Formating output
            out = [f"--- Статистика VPS от {now} ---"]
            
            if 'mem' in stats:
                mem = stats['mem']
                mem_str = f"ОЗУ: {mem['used']}MB / {mem['total']}MB ({mem['percent']:.1f}%)"
                if mem['percent'] >= MEM_WARNING_PERCENT:
                    mem_str = f"⚠️ ВНИМАНИЕ! МАЛО ПАМЯТИ! {mem_str}"
                out.append(mem_str)
                
            if 'disk' in stats:
                disk = stats['disk']
                disk_str = f"Диск: {disk['used']} / {disk['size']} ({disk['percent']}%)"
                if disk['percent'] >= DISK_WARNING_PERCENT:
                    disk_str = f"⚠️ ВНИМАНИЕ! ЗАКАНЧИВАЕТСЯ МЕСТО! {disk_str}"
                out.append(disk_str)
                
            out.append(f"Загрузка CPU: {stats.get('load', 'N/A')}")
            
            st_col = stats.get('st_status', 'unknown')
            bo_col = stats.get('bo_status', 'unknown')
            out.append(f"Сборщик SafeTrade: {'🟢 ACTIVE' if st_col == 'active' else '🔴 ' + st_col.upper()}")
            out.append(f"Сборщик BigONE:    {'🟢 ACTIVE' if bo_col == 'active' else '🔴 ' + bo_col.upper()}")
            
            print("\n".join(out))
            print("-" * 40)
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n🛑 Мониторинг остановлен.")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    monitor()
