import os
import sys
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    parser = argparse.ArgumentParser(description="Инструмент оценки обученного агента.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Путь к папке чекпоинта')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Чекпоинт {checkpoint_path} не найден!")
        return

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env" / "environment_data.parquet"

    env_config = {
        "data_path": str(DATA_PATH),
        "split_mode": "test", # Строго тестовая выборка
        "commission": 0.0003, # Полная комиссия
        "initial_balance": 100000.0,
        "max_episode_steps": 2000, # Запас
        "task_phase": 3 # Суровый трейдинг (Phase 3)
    }

    # 🌟 Загружаем нейросеть локально (без оркестратора Ray)
    policy_dir = Path(checkpoint_path) / "policies" / "default_policy"
    if not policy_dir.exists():
        policy_dir = Path(checkpoint_path) # Фолбэк, если путь уже указывает на policy

    print(f"🔄 Восстановление агента из {policy_dir}...")
    policy = Policy.from_checkpoint(str(policy_dir))

    env = PortfolioTradingEnv(env_config)
    obs, info = env.reset()

    print(f"🚀 Запуск симуляции (Test Set: OOS)...")

    dates = [pd.to_datetime(env.unique_dates[env.current_step])]
    nav_history = [env.nav]
    market_nav_history = [env.nav]
    weights_history = [env.current_weights.copy()]

    market_nav = env_config["initial_balance"]

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, state_out, info_dict = policy.compute_single_action(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)

        dates.append(pd.to_datetime(info["date"]))
        nav_history.append(info["balance"])

        market_nav *= (1.0 + info.get("market_return", 0.0))
        market_nav_history.append(market_nav)

        w = env.current_weights
        weights_history.append(w.copy())

    print(f"✅ Симуляция завершена. Итоговый баланс (NAV): ${nav_history[-1]:,.2f}")
    print(f"   Рыночный бенчмарк: ${market_nav_history[-1]:,.2f}")
    print(f"   Sharpe Ratio: {info.get('sharpe', 0.0):.2f}")
    print(f"   Max Drawdown: {info.get('drawdown', 0.0)*100:.2f}%")

    generate_report(dates, nav_history, market_nav_history, weights_history, env.tickers)

def generate_report(dates, nav_history, market_nav_history, weights_history, tickers):
    print("📈 Генерация визуального HTML-отчета...")

    # Убрали shared_xaxes=True, чтобы 3-й график (гистограмма) мог корректно подписать тикеры
    fig = make_subplots(
        rows=3, cols=1,
        vertical_spacing=0.08,
        subplot_titles=(
            "Динамика Портфеля (NAV) vs Рынок",
            "Динамика долей портфеля (Asset Allocation - ВСЕ АКТИВЫ)",
            "Среднее распределение активов за период (Гистограмма)"
        ),
        row_heights=[0.4, 0.4, 0.2]
    )

    # 1. Графики NAV
    fig.add_trace(go.Scatter(x=dates, y=nav_history, name="RL Agent", line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=market_nav_history, name="Market Benchmark", line=dict(color='gray', dash='dash')), row=1, col=1)

    # 2. График Asset Allocation (Все активы)
    weights_matrix = np.array(weights_history)

    cash_w = weights_matrix[:, -1]
    fig.add_trace(go.Scatter(x=dates, y=cash_w, name="Cash", stackgroup='one', fillcolor='rgba(200, 200, 200, 0.5)', line=dict(width=0)), row=2, col=1)

    for i, ticker in enumerate(tickers):
        ticker_w = weights_matrix[:, i]
        # УБРАН ФИЛЬТР > 0.05, теперь выводятся абсолютно все активы
        fig.add_trace(go.Scatter(x=dates, y=ticker_w, name=ticker, stackgroup='one', line=dict(width=0)), row=2, col=1)

    # 3. Гистограмма (Bar Chart) среднего распределения
    avg_weights = np.mean(weights_matrix, axis=0) * 100 # Переводим в проценты
    all_labels = tickers + ["Cash"]

    # Сортируем для красивого отображения (по убыванию веса)
    sorted_indices = np.argsort(avg_weights)[::-1]
    sorted_labels = [all_labels[i] for i in sorted_indices]
    sorted_weights = avg_weights[sorted_indices]

    # Чтобы гистограмма не была замусорена пылинками, отсечем активы, чей средний вес < 0.1%
    mask = sorted_weights > 0.1
    final_labels = np.array(sorted_labels)[mask]
    final_weights = sorted_weights[mask]

    fig.add_trace(go.Bar(
        x=final_labels,
        y=final_weights,
        marker_color='indigo',
        name="Средний вес",
        text=[f"{w:.1f}%" for w in final_weights],
        textposition='auto'
    ), row=3, col=1)

    fig.update_layout(height=1100, title_text="Отчет об оценке RL Агента (Полный обзор)", hovermode="x unified")
    fig.update_yaxes(title_text="Баланс ($)", row=1, col=1)
    fig.update_yaxes(title_text="Доля (0-1)", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Вес (%)", row=3, col=1)

    output_path = "evaluation_report.html"
    fig.write_html(output_path)
    print(f"📊 Отчет сохранен в: {Path(output_path).resolve()}")

if __name__ == "__main__":
    main()