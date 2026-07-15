# _core/feature_generator.py
import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
import json
from tqdm import tqdm

from .paths import RESULTS_DIR, ensure_dirs

def hurst_exponent(ts):
    lags = range(2, 100)
    if len(ts) < 100: return np.nan
    ts = np.asarray(ts)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    tau = [t for t in tau if t > 0]
    if len(tau) < 2: return np.nan
    poly = np.polyfit(np.log(range(2, 2 + len(tau))), np.log(tau), 1)
    return poly[0] * 2.0

def create_cross_sectional_features(all_data: dict) -> pd.DataFrame:
    print("Этап 1: Расчет кросс-секционных признаков...")
    frames = []
    for ticker, df in all_data.items():
        df_temp = df.reset_index()
        df_copy = df_temp[['datetime', 'close', 'volume']].copy()
        df_copy['ticker'] = ticker
        frames.append(df_copy)
    
    full_df = pd.concat(frames, ignore_index=True).sort_values(["datetime", "ticker"])
    full_df['datetime'] = pd.to_datetime(full_df['datetime'])

    full_df['return_60d'] = full_df.groupby('ticker')['close'].pct_change(periods=60, fill_method=None)
    full_df['volatility_20d'] = full_df.groupby('ticker')['close'].pct_change(fill_method=None).rolling(20).std()
    full_df['dollar_volume_20d'] = (full_df['close'] * full_df['volume']).rolling(20).mean()

    grouped_by_date = full_df.groupby('datetime')
    full_df['cs_rank_mom_60'] = grouped_by_date['return_60d'].rank(pct=True)
    full_df['cs_rank_vol_20'] = grouped_by_date['volatility_20d'].rank(pct=True)
    mom_mean = grouped_by_date['return_60d'].transform('mean')
    mom_std = grouped_by_date['return_60d'].transform('std')
    full_df['cs_zscore_mom_60'] = (full_df['return_60d'] - mom_mean) / mom_std
    full_df['cs_rank_liquidity_20'] = grouped_by_date['dollar_volume_20d'].rank(pct=True)
    
    cs_cols = ['datetime', 'ticker', 'cs_rank_mom_60', 'cs_rank_vol_20', 'cs_zscore_mom_60', 'cs_rank_liquidity_20']
    cs_features = full_df[cs_cols]
    return cs_features

def create_individual_features(ticker_df_tuple: tuple, external_data: pd.DataFrame, cs_features: pd.DataFrame) -> tuple:
    warnings.simplefilter('ignore', FutureWarning)
    warnings.simplefilter('ignore', RuntimeWarning)
    
    ticker, df = ticker_df_tuple
    
    df_with_features = df.copy()

    if 'datetime' in df_with_features.columns:
        df_with_features['datetime'] = pd.to_datetime(df_with_features['datetime'], errors='coerce')
        df_with_features.dropna(subset=['datetime'], inplace=True)
        df_with_features.sort_values('datetime', inplace=True)
        df_with_features.set_index('datetime', inplace=True)
    elif not isinstance(df_with_features.index, pd.DatetimeIndex):
        try:
            df_with_features.index = pd.to_datetime(df_with_features.index, errors='coerce')
            df_with_features = df_with_features[~df_with_features.index.isna()].sort_index()
        except Exception:
            pass

    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    initial_ohlcv = df_with_features[ohlcv_cols].copy()

    if isinstance(external_data, pd.DataFrame) and not external_data.empty:
        ext = external_data.copy()
        if 'datetime' in ext.columns:
            ext['datetime'] = pd.to_datetime(ext['datetime'], errors='coerce')
            ext.dropna(subset=['datetime'], inplace=True)
            ext.sort_values('datetime', inplace=True)
            ext.set_index('datetime', inplace=True)
        ext = ext[~ext.index.duplicated(keep='first')]
        df_with_features = df_with_features.join(ext, how='left')
    
    df_with_features['spread_hl'] = (df_with_features['high'] - df_with_features['low']) / df_with_features['close']
    df_with_features['spread_oc'] = (df_with_features['open'] - df_with_features['close']) / df_with_features['close']
    df_with_features['return'] = df_with_features['close'].pct_change()

    # --- Генерация TA признаков ---
    df_with_features.ta.rsi(length=14, append=True); df_with_features.ta.rsi(length=28, append=True)
    df_with_features.ta.macd(append=True); df_with_features.ta.bbands(append=True); df_with_features.ta.atr(append=True)
    df_with_features.ta.adx(append=True); df_with_features.ta.cmf(length=20, append=True)
    df_with_features.ta.roc(length=20, append=True); df_with_features.ta.roc(length=60, append=True); df_with_features.ta.roc(length=120, append=True)
    df_with_features.ta.stoch(append=True); df_with_features.ta.obv(append=True); df_with_features.ta.mfi(append=True)
    df_with_features.ta.willr(append=True); df_with_features.ta.cci(append=True); df_with_features.ta.aroon(append=True)
    df_with_features.ta.chop(append=True); df_with_features.ta.vortex(append=True); df_with_features.ta.ao(append=True)
    df_with_features.ta.kst(append=True); df_with_features.ta.psar(append=True); df_with_features.ta.tsi(append=True)
    df_with_features.ta.uo(append=True); df_with_features.ta.massi(append=True)
    
    try:
        df_with_features.ta.ichimoku(append=True)
    except Exception as e:
        warnings.warn(f"ichimoku пропущен: {e}")
    try:
        df_with_features.ta.vwap(append=True)
    except Exception as e:
        warnings.warn(f"vwap пропущен: {e}")
    try:
        df_with_features.ta.amat(append=True)
    except Exception as e:
        warnings.warn(f"amat пропущен: {e}")

    df_with_features.ta.bop(append=True); df_with_features.ta.cksp(append=True)
    df_with_features.ta.ppo(append=True); df_with_features.ta.mom(append=True); df_with_features.ta.thermo(append=True)
    df_with_features.ta.kc(append=True)

    # --- Генерация кастомных признаков ---
    for n in [20, 50, 100, 200]:
        df_with_features[f'SMA_{n}'] = ta.sma(df_with_features['close'], length=n)
        df_with_features[f'EMA_{n}'] = ta.ema(df_with_features['close'], length=n)
        df_with_features[f'close_vs_sma{n}'] = (df_with_features['close'] - df_with_features[f'SMA_{n}']) / df_with_features[f'SMA_{n}']
        df_with_features[f'close_vs_ema{n}'] = (df_with_features['close'] - df_with_features[f'EMA_{n}']) / df_with_features[f'EMA_{n}']
    df_with_features['sma50_vs_sma200'] = (df_with_features['SMA_50'] - df_with_features['SMA_200']) / df_with_features['SMA_200']
    df_with_features['ema50_vs_ema200'] = (df_with_features['EMA_50'] - df_with_features['EMA_200']) / df_with_features['EMA_200']

    df_with_features['day_of_week'] = df_with_features.index.dayofweek
    df_with_features['month'] = df_with_features.index.month
    
    # 👇👇👇 ДОБАВЛЯЕМ МЕТРИКУ ВРЕМЕННОГО РАЗРЫВА 👇👇👇
    # Вычисляем разницу в днях между текущим и предыдущим индексом (datetime)
    df_with_features['days_since_last_trade'] = df_with_features.index.to_series().diff().dt.days
    # Для самого первого дня ставим 1.0, чтобы избежать NaN
    df_with_features['days_since_last_trade'] = df_with_features['days_since_last_trade'].fillna(1.0)
    # 👆👆👆 КОНЕЦ ИЗМЕНЕНИЯ 👆👆👆

    for lag in [1, 2, 3, 5, 10, 21]: df_with_features[f'return_lag_{lag}'] = df_with_features['return'].shift(lag)
    df_with_features['upper_shadow'] = (df_with_features['high'] - np.maximum(df_with_features['open'], df_with_features['close']))
    df_with_features['lower_shadow'] = (np.minimum(df_with_features['open'], df_with_features['close']) - df_with_features['low'])
    log_hl = np.log(df_with_features['high'] / df_with_features['low'])
    log_co = np.log(df_with_features['close'] / df_with_features['open'])
    df_with_features['volatility_gk'] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    for n in [20, 50]:
         df_with_features[f'volume_quantile_{n}d'] = df_with_features['volume'].rolling(window=n).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    for n in [20, 50, 200]:
        rolling_max = df_with_features['close'].rolling(window=n, min_periods=1).max()
        is_new_max = df_with_features['close'] >= rolling_max
        new_max_counter = is_new_max.cumsum()
        df_with_features[f'days_since_{n}d_high'] = df_with_features.groupby(new_max_counter).cumcount()
    
    df_with_features['hurst_100d'] = df_with_features['close'].rolling(window=100).apply(hurst_exponent, raw=True)
    df_with_features['volume_zscore_252d'] = (df_with_features['volume'] - df_with_features['volume'].rolling(252).mean()) / df_with_features['volume'].rolling(252).std()

    external_cols = ['usdrub_close', 'imoex_close', 'sp500_close', 'vix_close', 'brent_close', 'cbr_rate_close']
    for col in external_cols:
        if col not in df_with_features.columns:
            df_with_features[col] = np.nan
    df_with_features['usdrub_return_1d'] = df_with_features['usdrub_close'].pct_change()
    df_with_features['brent_return_1d'] = df_with_features['brent_close'].pct_change()
    df_with_features['sp500_return_1d'] = df_with_features['sp500_close'].pct_change()
    df_with_features['imoex_return_1d'] = df_with_features['imoex_close'].pct_change()
    df_with_features['usdrub_return_5d'] = df_with_features['usdrub_close'].pct_change(5)
    df_with_features['usdrub_volatility_20d'] = df_with_features['usdrub_return_1d'].rolling(20).std()
    df_with_features['corr_usdrub_60d'] = df_with_features['return'].rolling(60).corr(df_with_features['usdrub_return_1d'])
    df_with_features['cbr_rate'] = df_with_features['cbr_rate_close'] 
    df_with_features['is_rate_high'] = (df_with_features['cbr_rate_close'] > 10).astype(int)
    df_with_features['imoex_volatility_20d'] = df_with_features['imoex_return_1d'].rolling(20).std()
    rolling_cov = df_with_features['return'].rolling(60).cov(df_with_features['imoex_return_1d'])
    rolling_var = df_with_features['imoex_return_1d'].rolling(60).var()
    df_with_features['beta_vs_imoex_60d'] = rolling_cov / rolling_var
    df_with_features['sp500_return_lag_1d'] = df_with_features['sp500_return_1d'].shift(1)
    df_with_features['vix_roc_5d'] = ta.roc(df_with_features['vix_close'], length=5)
    df_with_features['brent_return_5d'] = df_with_features['brent_return_1d'].shift(1)
    for name in ['brent', 'sp500']:
        if f'{name}_return_1d' in df_with_features.columns:
            df_with_features[f'corr_{name}_20d'] = df_with_features['return'].rolling(20).corr(df_with_features[f'{name}_return_1d'])
    if 'imoex_close' in df_with_features.columns:
        df_with_features['market_bull_trend'] = (df_with_features['imoex_close'] > ta.sma(df_with_features['imoex_close'], 200)).astype(int)
        if 'RSI_14' in df_with_features.columns:
            df_with_features['rsi_14_x_bull_trend'] = df_with_features['RSI_14'] * df_with_features['market_bull_trend']
    if 'vix_close' in df_with_features.columns:
        df_with_features['vix_high_fear_regime'] = (df_with_features['vix_close'] > 25).astype(int)
    
    if 'ATRr_14' in df_with_features.columns:
        df_with_features['vol_of_vol_20d'] = df_with_features['ATRr_14'].rolling(20).std()
    if 'RSI_14' in df_with_features.columns:
        df_with_features['rsi_vol_20d'] = df_with_features['RSI_14'].rolling(20).std()
    low_50d = df_with_features['low'].rolling(50).min()
    high_50d = df_with_features['high'].rolling(50).max()
    df_with_features['price_to_range_50d'] = (df_with_features['close'] - low_50d) / (high_50d - low_50d + 1e-9)
    abs_return_sum_60d = df_with_features['return'].abs().rolling(60).sum()
    total_return_60d = df_with_features['close'].pct_change(60)
    df_with_features['trend_efficiency_60d'] = (total_return_60d / (abs_return_sum_60d + 1e-9)).abs()
    df_with_features['return_skew_60d'] = df_with_features['return'].rolling(60).skew()
    
    if 'ROC_20' in df_with_features.columns and 'ATRr_14' in df_with_features.columns:
        raw_mom_x_vol = df_with_features['ROC_20'] / (df_with_features['ATRr_14'] + 1e-9)
        df_with_features['mom_x_vol_norm'] = np.sign(raw_mom_x_vol) * np.log1p(np.abs(raw_mom_x_vol))
    
    if 'vix_high_fear_regime' in df_with_features.columns:
        df_with_features['return_x_vix_high'] = df_with_features['return'] * df_with_features['vix_high_fear_regime']
    if 'SMA_50' in df_with_features.columns and 'SMA_200' in df_with_features.columns:
        sma50 = df_with_features['SMA_50']
        sma200 = df_with_features['SMA_200']
        sma_cross = (sma50 > sma200) != (sma50.shift(1) > sma200.shift(1))
        sma_cross_counter = sma_cross.cumsum()
        df_with_features['days_since_sma_cross'] = df_with_features.groupby(sma_cross_counter).cumcount()
    up_volume = df_with_features['volume'].where(df_with_features['return'] > 0, 0)
    down_volume = df_with_features['volume'].where(df_with_features['return'] < 0, 0)
    sum_up_vol_20d = up_volume.rolling(20).sum()
    sum_down_vol_20d = down_volume.rolling(20).sum()
    df_with_features['ud_volume_ratio_20d'] = sum_up_vol_20d / (sum_down_vol_20d + 1e-9)
    if 'CMF_20' in df_with_features.columns:
        df_with_features['cmf_roc_20d'] = df_with_features['CMF_20'].pct_change(20)
    df_with_features['chaikin_vol_20d'] = ta.vhf(df_with_features['close'], length=20)
    df_with_features['return_kurt_60d'] = df_with_features['return'].rolling(60).kurt()
    return_60d = df_with_features['close'].pct_change(60)
    vol_60d = df_with_features['return'].rolling(60).std()
    df_with_features['sharpe_60d'] = return_60d / (vol_60d + 1e-9)
    mean_252d = df_with_features['close'].rolling(252).mean()
    std_252d = df_with_features['close'].rolling(252).std()
    df_with_features['price_zscore_252d'] = (df_with_features['close'] - mean_252d) / (std_252d + 1e-9)
    if 'VWAP_D' in df_with_features.columns:
        df_with_features['close_vs_vwap_d'] = (df_with_features['close'] - df_with_features['VWAP_D']) / (df_with_features['VWAP_D'] + 1e-9)
    df_with_features['autocorr_1d_roll_20d'] = df_with_features['return'].rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
    if 'imoex_return_1d' in df_with_features.columns:
        df_with_features['relative_return_imoex'] = df_with_features['return'] - df_with_features['imoex_return_1d']
    df_with_features['month_sin'] = np.sin(2 * np.pi * df_with_features['month']/12)
    df_with_features['month_cos'] = np.cos(2 * np.pi * df_with_features['month']/12)
    
    # --- БЛОК КОМПЛЕКСНОЙ ОЧИСТКИ ---
    features_to_clip = {
        "ROC": (-99.9, 1000), "AO": (-5000, 5000),
        "KST": (-500, 500), "MOM": (-5000, 5000),
        "cmf_roc_20d": (-1000, 1000), "ud_volume_ratio_20d": (0, 5000)
    }
    
    for col in df_with_features.columns:
        if col == 'mom_x_vol_norm': continue
        for key, (lower, upper) in features_to_clip.items():
            if key in col:
                df_with_features[col] = df_with_features[col].clip(lower=lower, upper=upper)
    
    df_final = df_with_features.drop(columns=ohlcv_cols, errors='ignore')
    helper_cols = ['usdrub_return_1d', 'brent_return_1d', 'sp500_return_1d', 'imoex_return_1d']
    df_final.drop(columns=helper_cols, inplace=True, errors='ignore')
    df_final = pd.concat([initial_ohlcv, df_final], axis=1)
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    ticker_cs = cs_features[cs_features['ticker'] == ticker]
    if not ticker_cs.empty:
        df_final.reset_index(inplace=True)
        df_final = pd.merge(df_final, ticker_cs.drop('ticker', axis=1), on='datetime', how='left')
    
    df_final.fillna(method='ffill', inplace=True)
    df_final.fillna(method='bfill', inplace=True)
    df_final.fillna(0, inplace=True)
    
    return ticker, df_final

if __name__ == '__main__':
    ensure_dirs()
    print("--- 🔬 Запуск генератора признаков в тестовом режиме ---")
    
    sample_data_stock = {
        'datetime': pd.to_datetime(pd.date_range(start='2020-01-01', periods=300)),
        'open': np.random.uniform(95, 105, 300), 'high': np.random.uniform(100, 110, 300),
        'low': np.random.uniform(90, 100, 300), 'close': np.random.uniform(98, 108, 300),
        'volume': np.random.uniform(10000, 50000, 300)
    }
    sample_df1 = pd.DataFrame(sample_data_stock)
    sample_df1.set_index('datetime', inplace=True)

    sample_df2 = sample_df1.copy()
    all_sample_data = {'TICKER1': sample_df1, 'TICKER2': sample_df2}

    ext_dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=300))
    external_df = pd.DataFrame(index=ext_dates)
    external_df['usdrub_close'] = np.random.uniform(70, 80, 300)
    external_df['imoex_close'] = np.random.uniform(3000, 4000, 300)
    external_df['sp500_close'] = np.random.uniform(4000, 5000, 300)
    external_df['vix_close'] = np.random.uniform(15, 30, 300)
    external_df['brent_close'] = np.random.uniform(60, 80, 300)
    external_df['cbr_rate_close'] = np.random.choice([7.5, 8.5, 9.0], 300)
    
    print("Этап 1: Расчет кросс-секционных признаков (на тестовых данных)...")
    cs_features_sample = create_cross_sectional_features(all_sample_data)
    
    print("Этап 2: Расчет индивидуальных признаков (на тестовых данных)...")
    _, first_ticker_features = create_individual_features(('TICKER1', sample_df1), external_df, cs_features_sample)

    all_test_cols = first_ticker_features.columns.tolist()
    
    base_structural_cols = [
        'datetime', 'open', 'high', 'low', 'close', 'volume', 
        'cs_rank_mom_60', 'cs_rank_vol_20', 'cs_zscore_mom_60', 'cs_rank_liquidity_20'
    ]
    
    pure_indicator_cols = [col for col in all_test_cols if col not in base_structural_cols]
    num_pure_indicators = len(pure_indicator_cols)
    
    NUM_OHLCV_DT = 6
    NUM_CS_FEATURES = 4
    
    projected_total = num_pure_indicators + NUM_OHLCV_DT + NUM_CS_FEATURES
    
    print("\n--- 📝 Результаты подсчета признаков ---")
    print(f"Количество 'чистых' индикаторов (TA, макро и др.): {num_pure_indicators}")
    print(f"Количество колонок OHLCV + datetime: {NUM_OHLCV_DT}")
    print(f"Количество кросс-секционных признаков: {NUM_CS_FEATURES}")
    print("--------------------------------------------------")
    print(f"✅ Прогнозируемое реальное количество признаков: {projected_total}")
    print("   (Может отличаться на 1-2 из-за особенностей реальных данных)")

    final_feature_list_for_json = sorted(pure_indicator_cols + base_structural_cols)
    output_path = RESULTS_DIR / "base_features.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_feature_list_for_json, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Обновленный список из {len(final_feature_list_for_json)} признаков сохранен в: {output_path}")