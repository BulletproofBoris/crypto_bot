import json
import os
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Умный поиск чекпоинтов Ray Tune")
    parser.add_argument('--target_phase', type=int, required=True, choices=[1, 2, 3], help="Целевая стадия (1, 2 или 3)")
    parser.add_argument('--position', type=str, required=True, choices=['start', 'end'], help="'start' или 'end'")
    parser.add_argument('--clean', action='store_true', help="Удалить все остальные чекпоинты")
    # 🌟 НОВЫЙ АРГУМЕНТ: Имя эксперимента
    parser.add_argument('--exp_name', type=str, default="pbt_trading_bot", help="Имя папки эксперимента")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent
    RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
    
    # 🌟 ИСПОЛЬЗУЕМ ИМЯ ЭКСПЕРИМЕНТА ИЗ АРГУМЕНТА
    EXPERIMENT_DIR = RL_DIR / "ray_results" / args.exp_name
    OUTPUT_JSON = RL_DIR / "healthy_checkpoints.json"

    if not EXPERIMENT_DIR.exists():
        print(f"❌ Директория {EXPERIMENT_DIR} не найдена!")
        return

    print(f"🔍 Поиск чекпоинтов: Стадия {args.target_phase} | Позиция: {args.position} | Эксперимент: {args.exp_name}")
    
    healthy_checkpoints = []

    for trial_dir in EXPERIMENT_DIR.glob("PPO_*"):
        if not trial_dir.is_dir():
            continue
            
        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue

        history = []
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    record = json.loads(line)
                    phase = record.get('env_runners', {}).get('custom_metrics', {}).get('task_phase_max', 1.0)
                    iteration = record.get('training_iteration', 0)
                    history.append({"iter": iteration, "phase": phase})
        except Exception as e:
            continue
        
        if not history: continue

        target_iter = -1

        if args.position == 'start':
            for entry in history:
                if entry["phase"] >= args.target_phase:
                    target_iter = entry["iter"]
                    break
        elif args.position == 'end':
            next_phase_iter = -1
            for entry in history:
                if entry["phase"] >= args.target_phase + 1:
                    next_phase_iter = entry["iter"]
                    break
            
            if next_phase_iter != -1:
                valid_iters = [e["iter"] for e in history if e["iter"] < next_phase_iter]
                target_iter = max(valid_iters) if valid_iters else -1
            else:
                valid_iters = [e["iter"] for e in history if e["phase"] == args.target_phase]
                target_iter = max(valid_iters) if valid_iters else -1

        if target_iter == -1:
            continue

        best_ckpt = None
        best_ckpt_iter = -1
        
        for ckpt_dir in trial_dir.glob("checkpoint_*"):
            if not ckpt_dir.is_dir(): continue
            try: ckpt_iter = int(ckpt_dir.name.split('_')[1])
            except ValueError: continue
            
            if best_ckpt_iter < ckpt_iter <= target_iter:
                best_ckpt_iter = ckpt_iter
                best_ckpt = ckpt_dir

        if best_ckpt:
            healthy_checkpoints.append(str(best_ckpt.absolute()))
            print(f"✅ {trial_dir.name[:15]}... | Целевая итер: {target_iter} | Взят: {best_ckpt.name}")

    if healthy_checkpoints:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(healthy_checkpoints, f, indent=4)
        print(f"\n💾 Сохранено {len(healthy_checkpoints)} чекпоинтов.")
        
        if args.clean:
            print("🧹 Очистка ненужных чекпоинтов...")
            clean_count = 0
            healthy_set = set(healthy_checkpoints)
            for trial_dir in EXPERIMENT_DIR.glob("PPO_*"):
                for ckpt_dir in trial_dir.glob("checkpoint_*"):
                    if ckpt_dir.is_dir() and str(ckpt_dir.absolute()) not in healthy_set:
                        try:
                            shutil.rmtree(ckpt_dir)
                            clean_count += 1
                        except: pass
            print(f"🗑️ Удалено {clean_count} старых чекпоинтов.")
    else:
        print("\n❌ Не найдено ни одного подходящего чекпоинта.")

if __name__ == "__main__":
    main()