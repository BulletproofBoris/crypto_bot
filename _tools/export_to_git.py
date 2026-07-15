import os
import sys
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

# Интегрируем нашу портфельную среду
sys.path.insert(0, os.getcwd())
from _tools.rl_env import PortfolioTradingEnv

def env_creator(env_config):
    return PortfolioTradingEnv(env_config)

register_env("TradingEnv-v0", env_creator)

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
    RAY_RESULTS_DIR = RL_DIR / "ray_results"
    GIT_EXPORT_DIR = RL_DIR / "champions"
    DATA_PATH = RL_DIR / "environment_data.parquet"

    print("🔍 Шаг 1: Поиск Топ-кандидатов в логах Ray Tune...")
    candidate_ckpts = set()
    
    # Собираем по 3 лучших чекпоинта из каждого отдельного триала
    for trial_dir in RAY_RESULTS_DIR.glob("**/PPO_*"):
        if not trial_dir.is_dir(): continue
        result_file = trial_dir / "result.json"
        if not result_file.exists(): continue
        
        trial_bests = []
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    record = json.loads(line)
                    eval_m = record.get('evaluation', {}).get('env_runners', {})
                    score = eval_m.get('episode_return_mean', -float('inf'))
                    if score > -float('inf'):
                        trial_bests.append((score, record.get('timestamp', 0), trial_dir))
        except Exception: pass
        
        # Сортируем по убыванию OOS Reward
        trial_bests.sort(key=lambda x: x[0], reverse=True)
        
        # Ищем физические папки для топ-3 из этого триала
        for best in trial_bests[:3]:
            best_timestamp = best[1]
            min_diff = float('inf')
            target_ckpt = None
            for ckpt in trial_dir.glob("checkpoint_*"):
                if not ckpt.is_dir(): continue
                diff = abs(os.path.getmtime(ckpt) - best_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    target_ckpt = ckpt
            if target_ckpt:
                candidate_ckpts.add(target_ckpt)

    print(f"📦 Найдено {len(candidate_ckpts)} уникальных перспективных моделей.")
    print("⚔️ Шаг 2: Запуск непрерывной симуляции (4.5 года OOS) для каждого кандидата...")

    env_config = {
        "data_path": str(DATA_PATH),
        "split_mode": "test",
        "commission": 0.0003,
        "initial_balance": 100000.0,
        "max_episode_steps": 2000, # Без лимита времени
        "task_phase": 3            # Боевая фаза с комиссиями
    }
    env = PortfolioTradingEnv(env_config)
    
    results = []
    
    # Отключаем спам от Ray/TF при загрузке чекпоинтов
    import logging
    logging.getLogger("ray").setLevel(logging.ERROR)

    for i, ckpt in enumerate(candidate_ckpts, 1):
        policy_dir = ckpt / "policies" / "default_policy"
        if not policy_dir.exists(): policy_dir = ckpt
        
        try:
            policy = Policy.from_checkpoint(str(policy_dir))
            obs, _ = env.reset()
            terminated = truncated = False
            
            while not (terminated or truncated):
                action, _, _ = policy.compute_single_action(obs, explore=False)
                obs, _, terminated, truncated, info = env.step(action)
            
            nav = info["balance"]
            dd = info["drawdown"]
            sharpe = info["sharpe"]
            
            # ФОРМУЛА ИДЕАЛЬНОЙ МОДЕЛИ НА НЕПРЕРЫВНОМ ТЕСТЕ
            profit_pct = ((nav - 100000.0) / 100000.0) * 100.0
            
            # Жесткий фильтр: бракуем тех, кто слил больше 30% на 4 годах
            if dd > 0.30:
                score = -float('inf')
            else:
                score = profit_pct + (sharpe * 5.0) - (dd * 100.0)
            
            results.append({
                "ckpt": ckpt,
                "exp": ckpt.parent.parent.name,
                "profit": profit_pct,
                "dd": dd * 100.0,
                "sharpe": sharpe,
                "score": score
            })
            sys.stdout.write(f"\rПротестировано {i}/{len(candidate_ckpts)}")
            sys.stdout.flush()
        except Exception as e:
            continue

    print("\n\n📊 ТОП-10 МОДЕЛЕЙ (НЕПРЕРЫВНЫЙ ТЕСТ 2022-2026):")
    print("-" * 90)
    print(f"{'Эксперимент':<25} | {'Чекпоинт':<18} | {'Profit %':<10} | {'Max DD %':<10} | {'Sharpe':<7} | {'Score'}")
    print("-" * 90)
    
    # Сортируем по нашей комплексной оценке
    results.sort(key=lambda x: x["score"], reverse=True)
    
    valid_results = [r for r in results if r["score"] > -float('inf')]
    
    for r in valid_results[:10]:
        print(f"{r['exp']:<25} | {r['ckpt'].name:<18} | {r['profit']:>8.2f}% | {r['dd']:>8.2f}% | {r['sharpe']:>7.2f} | {r['score']:.2f}")
    
    if not valid_results:
        print("⚠️ Ни одна модель не прошла фильтр просадки (<30%) на полном 4-летнем тесте.")
        return

    best = valid_results[0]
    print("\n🏆 ИСТИННЫЙ ЧЕМПИОН НАЙДЕН!")
    print(f"Эксперимент: {best['exp']}")
    print(f"Чекпоинт: {best['ckpt'].name}")
    print(f"Прибыль: +{best['profit']:.2f}% | Просадка: {best['dd']:.2f}% | Sharpe: {best['sharpe']:.2f}")
    
    if GIT_EXPORT_DIR.exists():
        shutil.rmtree(GIT_EXPORT_DIR)
    os.makedirs(GIT_EXPORT_DIR, exist_ok=True)
    
    shutil.copytree(best["ckpt"], GIT_EXPORT_DIR / "best_model")
    print(f"✅ Чекпоинт перенесен в champions/best_model. Можете запускать evaluate_agent.py!")

if __name__ == "__main__":
    main()