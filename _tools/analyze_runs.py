import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os
import warnings

# Отключаем системный спам scipy
warnings.filterwarnings("ignore")

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def main():
    parser = argparse.ArgumentParser(description="Анализатор макро-тренда с A/B подсветкой последнего Swarm-запуска.")
    parser.add_argument("fold_dir", type=str, help="Путь к папке фолда")
    parser.add_argument("--arch", type=str, default="conv1d+gru", help="Какую архитектуру анализировать") 
    parser.add_argument("--runs", type=int, default=100, help="Выделенный бюджет пула")
    parser.add_argument("--max_x", type=int, default=None, help="Ограничение по оси X")
    parser.add_argument("--valid_min", type=float, default=0.0, help="Нижняя граница")
    parser.add_argument("--valid_max", type=float, default=None, help="Верхняя граница")
    
    args = parser.parse_args()
    
    db_path = os.path.join(args.fold_dir, f"trading_factory_{args.arch}.db")
    total_budget = args.runs

    if not os.path.exists(db_path):
        print(f"❌ БД не найдена: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        # Читаем колонку session_id напрямую из БД (она добавится туда автоматически оркестратором)
        df = pd.read_sql_query("SELECT val_loss, session_id FROM runs WHERE status='COMPLETED' AND val_loss IS NOT NULL ORDER BY rowid ASC", conn)
        conn.close()
    except Exception as e:
        print(f"❌ Ошибка БД: {e}")
        return

    if len(df) < 5:
        print("⚠️ Слишком мало данных.")
        return

    if args.valid_max is None:
        q1, q3 = df['val_loss'].quantile(0.25), df['val_loss'].quantile(0.75)
        valid_max = min(q3 + 1.5 * (q3 - q1), df['val_loss'].quantile(0.85))
    else:
        valid_max = args.valid_max

    x_data = np.arange(1, len(df) + 1)
    y_data_raw = df['val_loss'].values
    session_data = df['session_id'].values
    
    # --- ЛОГИКА ПОДСВЕТКИ ---
    latest_session = session_data[-1]
    
    if latest_session != 'legacy':
        latest_mask = (session_data == latest_session)
    else:
        latest_mask = np.zeros(len(df), dtype=bool)

    valid_mask = (y_data_raw <= valid_max) & (y_data_raw >= args.valid_min)
    x_fit = x_data[valid_mask]
    y_fit_raw = y_data_raw[valid_mask]

    if len(x_fit) < 5:
        print("❌ Недостаточно данных после фильтрации.")
        return

    obs_amp = max(1e-4, np.max(y_fit_raw) - np.min(y_fit_raw))
    amplitude_guess = obs_amp
    
    max_x = args.max_x if args.max_x else max(len(df)*1.5, total_budget)
    x_range = np.linspace(1, max_x, 500)
    
    plt.figure(figsize=(16, 9))

    chaotic_mask = y_data_raw > valid_max
    
    # 4 категории точек
    old_valid_mask = ~chaotic_mask & ~latest_mask
    new_valid_mask = ~chaotic_mask & latest_mask
    old_chaotic_mask = chaotic_mask & ~latest_mask
    new_chaotic_mask = chaotic_mask & latest_mask

    # 1. Старые запуски (бледные)
    if np.any(old_valid_mask):
        plt.scatter(x_data[old_valid_mask], y_data_raw[old_valid_mask], alpha=0.35, color='gray', label='Старая история')
    if np.any(old_chaotic_mask):
        plt.scatter(x_data[old_chaotic_mask], y_data_raw[old_chaotic_mask], alpha=0.2, color='indianred', marker='x', label='Старый хаос')

    # 2. Текущий Swarm (яркие)
    if np.any(new_valid_mask):
        plt.scatter(x_data[new_valid_mask], y_data_raw[new_valid_mask], alpha=0.95, color='dodgerblue', s=70, edgecolors='black', zorder=5, label=f'🔥 Swarm ({latest_session})')
    if np.any(new_chaotic_mask):
        plt.scatter(x_data[new_chaotic_mask], y_data_raw[new_chaotic_mask], alpha=0.9, color='darkorange', marker='X', s=80, zorder=5, label='Новый хаос')

    plt.step(x_data, df['val_loss'].cummin(), where='post', color='black', alpha=0.5, linewidth=2, label='Фактический Рекорд')

    # ==========================================================
    # ДЕТЕКТОР ЛЖИ: BOOTSTRAPPING
    # ==========================================================
    bootstrap_a, bootstrap_b, bootstrap_c = [], [], []
    min_c = max(0.0, np.min(y_fit_raw) - (obs_amp * 2.0))
    max_a = max(1e-3, obs_amp * 3.0)

    free_bounds = (
        [1e-5, 0.01, min_c], 
        [max_a, 2.0, np.max(y_fit_raw)]
    )
    
    successful_boots = 0
    N_BOOTSTRAPS = 100
    SAMPLE_FRACTION = 0.8

    for _ in range(N_BOOTSTRAPS):
        sample_size = max(5, int(len(x_fit) * SAMPLE_FRACTION))
        sample_indices = np.sort(np.random.choice(len(x_fit), size=sample_size, replace=False))
        
        x_boot = x_fit[sample_indices]
        y_raw_boot = y_fit_raw[sample_indices]
        
        y_cummin_boot = np.minimum.accumulate(y_raw_boot)
        p0 = [amplitude_guess, 0.05, max(0, np.min(y_cummin_boot) - 0.01)]
        
        try:
            popt, _ = curve_fit(exp_func, x_boot, y_cummin_boot, p0=p0, bounds=free_bounds, maxfev=2000)
            a_b, b_b, c_b = popt
            
            bootstrap_a.append(a_b)
            bootstrap_b.append(b_b)
            bootstrap_c.append(c_b)
            
            plt.plot(x_range, exp_func(x_range, a_b, b_b, c_b), color='red', alpha=0.04) 
            successful_boots += 1
        except:
            pass

    print(f"\n📊 АНАЛИЗ ФОЛДА: {os.path.basename(args.fold_dir)}")
    print(f"🥇 Текущий абсолютный рекорд (Loss): {np.min(y_fit_raw):.4f}")
    
    if successful_boots < 10:
        print("❌ Бутстрэппинг провалился (недостаточно стабильных данных).")
    else:
        true_c = np.median(bootstrap_c)
        true_a = np.median(bootstrap_a)
        true_b = np.median(bootstrap_b)
        
        q5_c = np.percentile(bootstrap_c, 5)
        q95_c = np.percentile(bootstrap_c, 95)
        uncertainty = q95_c - q5_c
        
        print(f"\n🎯 Истинная макро-асимптота (Медиана): {true_c:.4f}")
        print(f"📊 Доверительный интервал (90%): от {q5_c:.4f} до {q95_c:.4f}")
        print(f"⚖️ Ширина неопределенности: {uncertainty:.5f}")
        
        if uncertainty < 0.005:
            print("\n✅ ВЕРДИКТ: Оценка ПРАВДИВА. Веер сошелся, тренд железобетонный.")
            if np.min(y_fit_raw) <= q5_c:
                 print("🛑 ОРКЕСТРАТОР: Целесообразно закрыть фолд (Рекорд пробил 5% квантиль асимптоты).")
            else:
                 print("🟢 ОРКЕСТРАТОР: Можно продолжать (Рекорд еще не достиг медианной асимптоты).")
        else:
            print("\n⚠️ ВЕРДИКТ: Оценка СОМНИТЕЛЬНА. Широкий веер, математика не уверена в пределе. Нужны еще раны.")

        y_true_pred = exp_func(x_range, true_a, true_b, true_c)
        plt.plot(x_range, y_true_pred, color='darkred', linewidth=3, zorder=4, label=f'Истинный Тренд (Медиана)')
        
        plt.axhline(y=true_c, color='darkred', linestyle='--', linewidth=2, zorder=3, label=f'Истинная Асимптота ({true_c:.4f})')
        plt.fill_between(x_range, q5_c, q95_c, color='red', alpha=0.15, zorder=2, label=f'Доверительный коридор асимптоты (90%)')

    plt.axvline(x=total_budget, color='blue', linestyle='-.', alpha=0.4, label='Бюджет пула')
    
    plt.title("Детектор Лжи HPO (A/B Тестирование Swarm-запусков)")
    plt.xlabel("Раны (Эксперименты)"); plt.ylabel("Validation Loss")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.grid(alpha=0.3, linestyle='--')

    global_max = np.max(y_fit_raw)
    global_min = np.min(y_fit_raw)
    amplitude = global_max - global_min

    margin = amplitude * 0.05 if amplitude > 0 else 0.005

    y_lim_top = global_max + margin
    y_lim_bottom = global_min - margin

    # Если линия асимптоты или доверительного интервала (q5_c/q95_c) уходит 
    # за эти рамки, мы её обрезаем, чтобы масштаб не ломался
    plt.ylim(y_lim_bottom, y_lim_top)
    plt.show()
if __name__ == "__main__":
    main()