import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import gc

class PortfolioTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config):
        super(PortfolioTradingEnv, self).__init__()
        
        data_path = env_config.get("data_path", "/home/restorator/trader_test/data/processed/2000_2026_1d/rl_env/environment_data.parquet")
        
        # 1. ЧИТАЕМ ДАННЫЕ ОПТИМИЗИРОВАННО
        print("🧬 [ENV] Чтение данных с диска...")
        sample_df = pd.read_parquet(data_path).head(1)
        all_cols = sample_df.columns.tolist()
        
        # Выцепляем вероятности ансамблей (p0=SL, p1=Hold, p2=TP)
        self.prob_cols = [c for c in all_cols if c.endswith('_p0') or c.endswith('_p1') or c.endswith('_p2')]
        
        # Выцепляем глобальный макро-контекст (если он есть в данных)
        potential_macro = ['usdrub_close', 'brent_close', 'sp500_close', 'imoex_close', 'vix_close']
        self.macro_cols = [c for c in potential_macro if c in all_cols]
        
        # Безопасный выбор базовых колонок
        core_cols = ['datetime', 'ticker']
        if 'close' in all_cols: 
            core_cols.append('close')
        if 'close_y' in all_cols: 
            core_cols.append('close_y')
        
        # Грузим только то, что нужно агенту (без мусорных сырых фичей)
        columns_to_load = list(set(core_cols + self.prob_cols + self.macro_cols))
        df = pd.read_parquet(data_path, columns=columns_to_load)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Глобальные размерности ДО сплита (защита от mismatch весов нейросети)
        self.tickers = sorted(df['ticker'].unique().tolist())
        self.num_tickers = len(self.tickers)
        self.num_probs = len(self.prob_cols)
        self.num_macro = len(self.macro_cols)
        
        # --- TRAIN / TEST SPLIT ---
        split_mode = env_config.get("split_mode", "train")
        cutoff_date = pd.to_datetime("2022-01-01") 
        
        if split_mode == "train":
            df_filtered = df[df['datetime'] < cutoff_date]
        elif split_mode == "test":
            df_filtered = df[df['datetime'] >= cutoff_date]
        else:
            df_filtered = df

        print(f"🧬 [{split_mode.upper()}] Сборка State Space для {self.num_tickers} тикеров...")
        
        # А) Строим матрицу Цен
        price_col = 'close_y' if 'close_y' in df_filtered.columns else 'close'
        self.price_pivot = df_filtered.pivot(index='datetime', columns='ticker', values=price_col)
        self.price_pivot = self.price_pivot.reindex(columns=self.tickers).ffill().bfill().fillna(0.0)
        self.unique_dates = self.price_pivot.index.tolist()
        self.prices_matrix = self.price_pivot.values.astype(np.float32)
        
        # Б) Строим тензор Вероятностей (Мнения ансамблей)
        self.probs_tensor = np.zeros((len(self.unique_dates), self.num_tickers, self.num_probs), dtype=np.float16)
        for t_idx, ticker in enumerate(self.tickers):
            if ticker in df_filtered['ticker'].values:
                ticker_df = df_filtered[df_filtered['ticker'] == ticker].set_index('datetime')
                ticker_df = ticker_df.reindex(self.unique_dates).fillna(0.0)
                ticker_df = ticker_df.reindex(columns=self.prob_cols).fillna(0.0)
                self.probs_tensor[:, t_idx, :] = ticker_df[self.prob_cols].values.astype(np.float16)
                
        # В) Строим матрицу Макро-контекста
        if self.num_macro > 0:
            macro_df = df_filtered.groupby('datetime')[self.macro_cols].first()
            macro_df = macro_df.reindex(self.unique_dates).fillna(0.0)
            self.macro_matrix = macro_df.values.astype(np.float32)
        else:
            self.macro_matrix = np.zeros((len(self.unique_dates), 0), dtype=np.float32)
        
        # ОЧИСТКА ПАМЯТИ
        del df, df_filtered, self.price_pivot
        gc.collect()

        # --- НАСТРОЙКА RL-ПРОСТРАНСТВ ---
        
        ### ИЗМЕНЕНО: Action Space теперь от -1.0 до 1.0. Даем агенту зону "строгого отказа" от актива.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_tickers + 1,), dtype=np.float32
        )
        
        ### ИЗМЕНЕНО: Добавлены 2 доп. слота для account_context (drawdown и task_phase)
        self.obs_dim = (self.num_tickers * self.num_probs) + self.num_macro + (self.num_tickers + 1) + 3
        
        self.observation_space = spaces.Box(
            low=-1.0e5, high=1.0e5, shape=(self.obs_dim,), dtype=np.float32 # Расширил границы для безопасности
        )
        
        self.initial_balance = env_config.get("initial_balance", 100000.0)
        self.task_phase = env_config.get("task_phase", 1)
        
        # Настройка фаз обучения (Curriculum Learning)
        if self.task_phase == 1:
            self.commission = 0.0
            self.max_episode_steps = min(60, max(1, len(self.unique_dates) - 2))
        elif self.task_phase == 2:
            self.commission = 0.00015
            self.max_episode_steps = min(126, max(1, len(self.unique_dates) - 2))
        else:
            self.commission = env_config.get("commission", 0.0003)
            self.max_episode_steps = min(env_config.get("max_episode_steps", 252), max(1, len(self.unique_dates) - 2))
        
        self.current_step = 0
        self.episode_step = 0
        self.nav = self.initial_balance
        self.prev_nav = self.initial_balance
        self.peak_nav = self.initial_balance
        self.current_weights = np.zeros(self.num_tickers + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0  # На старте сидим 100% в кэше

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random, _ = gym.utils.seeding.np_random()
            
        max_start = max(0, len(self.unique_dates) - self.max_episode_steps - 2)
        self.current_step = self.np_random.integers(0, max_start) if max_start > 0 else 0
        
        self.episode_step = 0
        self.nav = self.initial_balance
        self.prev_nav = self.initial_balance
        self.peak_nav = self.initial_balance
        self.returns_history = []
        
        if self.num_macro > 0:
            self.macro_start = self.macro_matrix[self.current_step]
        
        self.current_weights = np.zeros(self.num_tickers + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        
        return self._get_observation(), {}

    def _get_observation(self):
        # 1. Прогнозы ансамблей
        probs_features = self.probs_tensor[self.current_step].flatten().astype(np.float32)
        
        # 2. Макро-индикаторы
        if self.num_macro > 0:
            macro_features = (self.macro_matrix[self.current_step] - self.macro_start) / (np.abs(self.macro_start) + 1e-9)
        else:
            macro_features = np.array([], dtype=np.float32)
        
        ### НОВОЕ: Расчет текущей просадки для передачи агенту
        current_drawdown = (self.peak_nav - self.nav) / self.peak_nav if self.peak_nav > 0 else 0.0
        
        ### ИЗМЕНЕНО: Обогащенный внутренний контекст
        account_context = np.array([
            self.episode_step / self.max_episode_steps,
            current_drawdown,             # Агент "чувствует" просадку
            float(self.task_phase) / 3.0  # Агент понимает суровость текущей фазы
        ], dtype=np.float32)
        
        obs = np.concatenate([probs_features, macro_features, self.current_weights, account_context])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.clip(obs, -10.0, 10.0)
        return obs.astype(np.float32)

    def step(self, action):
        ### НОВОЕ: ЕСТЕСТВЕННАЯ КОМПОНОВКА ПОРТФЕЛЯ (Без Softmax)
        # Всё, что меньше 0 — агент осознанно обнуляет.
        raw_weights = np.clip(action, 0.0, 1.0)
        
        weight_sum = np.sum(raw_weights)
        if weight_sum > 0:
            desired_weights = raw_weights / weight_sum
        else:
            # Если всё в минусе (паника) -> уходим в 100% кэш
            desired_weights = np.zeros_like(raw_weights)
            desired_weights[-1] = 1.0
        
        # 2. ИНЕРЦИЯ ПОРТФЕЛЯ (Смягчение шоковых транзакций)
        target_weights = (self.current_weights * 0.8) + (desired_weights * 0.2)
        target_weights = target_weights / np.sum(target_weights)
        
        ### НОВОЕ: Считаем оборот портфеля (Turnover) для штрафов за суету
        turnover = np.sum(np.abs(desired_weights - self.current_weights))
        
        prices_today = self.prices_matrix[self.current_step]
        prices_tomorrow = self.prices_matrix[self.current_step + 1]
        
        valid_price_mask = (prices_today > 0) & (prices_tomorrow > 0)
        asset_returns = np.zeros(self.num_tickers, dtype=np.float32)
        asset_returns[valid_price_mask] = (prices_tomorrow[valid_price_mask] - prices_today[valid_price_mask]) / prices_today[valid_price_mask]
        asset_returns = np.clip(asset_returns, -0.99, 10.0) 
        
        portfolio_returns = np.append(asset_returns, 0.0) # Кэш не генерирует доходность
        
        # 3. РАСЧЕТ БЕНЧМАРКА (Среднее по рынку за сегодня)
        market_return = np.mean(asset_returns[valid_price_mask]) if np.any(valid_price_mask) else 0.0
        
        # 4. Комиссия
        weight_changes = np.sum(np.abs(target_weights - self.current_weights))
        transaction_cost = weight_changes * (self.commission * 0.5) 
        
        # 5. ДОХОДНОСТЬ И НАГРАДА
        agent_return = np.sum(self.current_weights * portfolio_returns) - transaction_cost
        
        ### НОВОЕ: Асимметричный Reward Shaping
        alpha = agent_return - market_return
        
        if market_return < 0 and agent_return < 0:
            # Двойной штраф за падение вместе с рынком (учим защищать капитал)
            step_reward = alpha * 150.0  
        else:
            step_reward = alpha * 100.0
        
        # Штраф за сидение в кэше на фазе 1 (прогрев)
        if self.task_phase == 1 and self.current_weights[-1] > 0.3:
            step_reward -= (self.current_weights[-1] * 0.5) 
            
        ### ИЗМЕНЕНО: Поощряем сидение в кэше, если рынок падает (пропорционально падению)
        if self.task_phase >= 3 and market_return < 0:
            step_reward += (self.current_weights[-1] * abs(market_return) * 200.0)
            
        ### НОВОЕ: Штраф за "суету" (излишний оборот)
        if self.task_phase >= 3:
            step_reward -= (turnover * 0.05)
            
        self.returns_history.append(agent_return)
        
        # Переоценка реального NAV
        self.nav *= (1.0 + agent_return)
        self.nav = max(self.nav, 1.0) 
        
        self.prev_nav = self.nav
        next_weights_raw = target_weights * (1.0 + portfolio_returns)
        
        weight_sum = np.sum(next_weights_raw)
        if weight_sum > 0:
            self.current_weights = next_weights_raw / weight_sum
        else:
            self.current_weights = np.zeros_like(next_weights_raw)
            self.current_weights[-1] = 1.0 
        
        self.current_step += 1
        self.episode_step += 1
        
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        
        if self.nav < self.initial_balance * 0.5:
            terminated = True
            step_reward -= 50.0 # Штраф за Margin Call
            
        self.peak_nav = max(self.peak_nav, self.nav)
        drawdown = (self.peak_nav - self.nav) / self.peak_nav if self.peak_nav > 0 else 0.0
        
        if self.task_phase >= 3 and drawdown > 0.1:
            step_reward -= (drawdown * 2.0) # Штраф за просадку

        # Финальный клиппинг аномальных наград
        step_reward = float(np.clip(step_reward, -100.0, 100.0))

        obs = self._get_observation()
        
        returns_arr = np.array(self.returns_history)
        sharpe = 0.0
        if len(returns_arr) > 5 and np.std(returns_arr) > 0:
            sharpe = float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252))
        
        safe_date_idx = min(self.current_step, len(self.unique_dates) - 1)
        
        info = {
            "balance": float(self.nav),
            "date": str(self.unique_dates[safe_date_idx]),
            "cash_weight": float(self.current_weights[-1]),
            "agent_return": float(agent_return),
            "market_return": float(market_return),
            "drawdown": float(drawdown),
            "sharpe": sharpe
        }
        
        if np.isnan(step_reward) or np.isinf(step_reward):
            step_reward = -1.0
            
        return obs, float(step_reward), bool(terminated), bool(truncated), info