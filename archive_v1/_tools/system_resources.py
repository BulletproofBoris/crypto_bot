"""
System Resource Management & Multi-threading Configuration
==========================================================
Обеспечивает централизованное управление потоками и процессами:
- Выделяет процессам проекта до 80% ядер CPU.
- Гарантирует сохранение как минимум 1/5 (20%) ядер свободными для ОС, UI и системных служб.
- Автоматически настраивает переменные окружения BLAS, OpenMP, MKL, Polars и LightGBM.
"""

import os
import math


def get_safe_cpu_count(max_ratio: float = 0.8, min_reserved_cores: int = 1) -> int:
    """
    Возвращает безопасное число рабочих ядер CPU.
    Оставляет системе как минимум 1/5 (20%) ядер свободными.
    
    Например:
      16 ядер -> 12 рабочих ядер (4 ядра свободно для OS / WSL)
      8 ядер  -> 6 рабочих ядер (2 ядра свободно)
      4 ядра  -> 3 рабочих ядра (1 ядро свободно)
    """
    total = os.cpu_count() or 4
    # Резервируем минимум 20% (1/5) ядер
    reserved = max(min_reserved_cores, int(math.ceil(total * (1.0 - max_ratio))))
    safe = max(1, total - reserved)
    return safe


def configure_thread_limits():
    """
    Устанавливает лимиты потоков для C/C++ библиотек (OpenMP, MKL, OpenBLAS, Polars),
    чтобы фоновые вычислительные пакеты не забивали 100% ядер процессора.
    """
    safe_cores = get_safe_cpu_count()
    s = str(safe_cores)
    
    # OpenMP / BLAS / MKL
    os.environ["OMP_NUM_THREADS"] = s
    os.environ["MKL_NUM_THREADS"] = s
    os.environ["OPENBLAS_NUM_THREADS"] = s
    os.environ["VECLIB_MAXIMUM_THREADS"] = s
    os.environ["NUMEXPR_NUM_THREADS"] = s
    
    # Polars
    os.environ["POLARS_MAX_THREADS"] = s
    
    # TensorFlow intra/inter (если используется)
    os.environ["TF_NUM_INTRAOP_THREADS"] = s
    os.environ["TF_NUM_INTEROP_THREADS"] = str(max(1, safe_cores // 2))
    
    return safe_cores


# Автоматическая настройка при импорте модуля
SAFE_WORKERS = configure_thread_limits()

