import paramiko
import time
import sys
from datetime import datetime

HOST = "81.19.137.18"
USER = "root"
PASS = "cbqeoR6392Lr"

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
    
    # Check if collector is running
    stdin, stdout, stderr = ssh.exec_command("systemctl is-active safetrade-collector")
    collector_status = stdout.read().decode().strip()
    stats['collector_status'] = collector_status
    
    return stats

def monitor():
    print(f"📡 Подключение к серверу {HOST} для мониторинга...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(HOST, username=USER, password=PASS, timeout=10)
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
            
            collector = stats.get('collector_status', 'unknown')
            collector_str = f"Статус сборщика (safetrade-collector): {'🟢 ACTIVE' if collector == 'active' else '🔴 ' + collector.upper()}"
            out.append(collector_str)
            
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
