import os
import pandas as pd
import numpy as np

def run_simulation():
    # We can load directly from depth parquets or features parquet
    st_path = "data/processed/prl/safetrade_depth.parquet"
    bo_path = "data/processed/prl/bigone_depth.parquet"
    
    print("1. Loading SafeTrade and BigONE orderbook data...")
    # Read only top of book and volumes to be super fast and light
    st_cols = ['timestamp', 'bid_1_price', 'bid_1_qty', 'ask_1_price', 'ask_1_qty']
    bo_cols = ['timestamp', 'bid_1_price', 'bid_1_qty', 'ask_1_price', 'ask_1_qty']
    
    df_st = pd.read_parquet(st_path, columns=st_cols)
    df_bo = pd.read_parquet(bo_path, columns=bo_cols)
    
    print(f"   SafeTrade raw rows: {len(df_st)}")
    print(f"   BigONE raw rows: {len(df_bo)}")
    
    # Standardize timestamps to 1s
    df_st['timestamp'] = pd.to_datetime(df_st['timestamp']).dt.floor('s')
    df_bo['timestamp'] = pd.to_datetime(df_bo['timestamp']).dt.floor('s')
    
    df_st = df_st.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    df_bo = df_bo.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    
    # Resample to 1s continuous grid and ffill
    idx = df_st.index.union(df_bo.index)
    start_time = max(df_st.index.min(), df_bo.index.min())
    end_time = min(df_st.index.max(), df_bo.index.max())
    
    common_idx = pd.date_range(start_time, end_time, freq='1s')
    print(f"   Common overlapping timeline: {start_time} to {end_time} ({len(common_idx)} seconds)")
    
    df_st_1s = df_st.reindex(common_idx).ffill()
    df_bo_1s = df_bo.reindex(common_idx).ffill()
    
    # Prefix columns
    df_st_1s.columns = [f"ST_{c}" for c in df_st_1s.columns]
    df_bo_1s.columns = [f"BO_{c}" for c in df_bo_1s.columns]
    
    m = pd.concat([df_st_1s, df_bo_1s], axis=1).dropna()
    print(f"   Synchronized paired ticks: {len(m)}")
    
    # 2. Spread Analysis
    # Direction 1: Buy SafeTrade, Sell BigONE
    # Profit per unit = BO_bid_1 - ST_ask_1
    m['spread_buy_st_sell_bo'] = m['BO_bid_1_price'] - m['ST_ask_1_price']
    m['spread_pct_buy_st_sell_bo'] = m['spread_buy_st_sell_bo'] / m['ST_ask_1_price']
    
    # Direction 2: Buy BigONE, Sell SafeTrade
    # Profit per unit = ST_bid_1 - BO_ask_1
    m['spread_buy_bo_sell_st'] = m['ST_bid_1_price'] - m['BO_ask_1_price']
    m['spread_pct_buy_bo_sell_st'] = m['spread_buy_bo_sell_st'] / m['BO_ask_1_price']
    
    # Mid-price difference
    st_mid = (m['ST_bid_1_price'] + m['ST_ask_1_price']) / 2.0
    bo_mid = (m['BO_bid_1_price'] + m['BO_ask_1_price']) / 2.0
    m['mid_diff_pct'] = (bo_mid - st_mid) / st_mid
    
    # Fee threshold:
    # SafeTrade taker fee: 0.2%, BigONE taker fee: 0.2% -> Total roundtrip fee: 0.4%
    fee_threshold = 0.004
    
    opp_st_to_bo = (m['spread_pct_buy_st_sell_bo'] > fee_threshold)
    opp_bo_to_st = (m['spread_pct_buy_bo_sell_st'] > fee_threshold)
    
    print("\n" + "="*60)
    print("📊 1. СТАТИСТИКА МЕЖБИРЖЕВОГО СПРЕДА (SafeTrade vs BigONE)")
    print("="*60)
    print(f"Период анализа: {m.index.min()} — {m.index.max()} (Всего: {len(m):,} секунд)")
    print(f"Средняя цена SafeTrade mid: ${st_mid.mean():.4f}")
    print(f"Средняя цена BigONE mid:   ${bo_mid.mean():.4f}")
    print(f"Среднее расхождение Mid:   {m['mid_diff_pct'].mean() * 100:+.2f}% (Std: {m['mid_diff_pct'].std() * 100:.2f}%)")
    
    print(f"\nВозможности арбитража с учетом комиссий 0.4% (Taker/Taker):")
    print(f"  Купить SafeTrade -> Продать BigONE (спред > +0.4%): {opp_st_to_bo.sum():,} сек ({opp_st_to_bo.mean()*100:.2f}%)")
    print(f"  Купить BigONE -> Продать SafeTrade (спред > +0.4%): {opp_bo_to_st.sum():,} сек ({opp_bo_to_st.mean()*100:.2f}%)")
    print(f"  Суммарно окон арбитража: {(opp_st_to_bo | opp_bo_to_st).sum():,} сек ({(opp_st_to_bo | opp_bo_to_st).mean()*100:.2f}%)")
    
    # 3. Simulation of Strategy A: Naive sequential transfer (Buy -> Withdraw -> Wait 5m -> Sell)
    print("\n" + "="*60)
    print("⚠️ 2. СИМУЛЯЦИЯ НАИВНОГО ПОСЛЕДОВАТЕЛЬНОГО АРБИТРАЖА (Ончейн-перевод)")
    print("="*60)
    # Suppose we buy on ST when spread > 0.4%, transfer on-chain (wait 5 minutes = 300s), then sell on BO
    transfer_delay_s = 300
    st_to_bo_signals = m[opp_st_to_bo].copy()
    
    naive_returns = []
    for idx_time in st_to_bo_signals.index:
        idx_sell = idx_time + pd.Timedelta(seconds=transfer_delay_s)
        if idx_sell in m.index:
            buy_price = m.loc[idx_time, 'ST_ask_1_price'] * 1.002 # +0.2% fee
            sell_price = m.loc[idx_sell, 'BO_bid_1_price'] * 0.998 # -0.2% fee
            ret = (sell_price - buy_price) / buy_price
            naive_returns.append(ret)
            
    if naive_returns:
        naive_returns = np.array(naive_returns)
        print(f"Тест: Задержка ончейн-перевода = {transfer_delay_s} секунд (5 минут)")
        print(f"Количество попыток: {len(naive_returns)}")
        print(f"Винрейт (прибыльных сделок после задержки): {(naive_returns > 0).mean() * 100:.1f}%")
        print(f"Средняя доходность на попытку: {naive_returns.mean() * 100:+.2f}%")
        print(f"Максимальная прибыль: {naive_returns.max() * 100:+.2f}%")
        print(f"Худший убыток (из-за пролива цены за 5 мин): {naive_returns.min() * 100:+.2f}%")
        print("ВЫВОД: При ончейн-переводе задержка полностью уничтожает арбитраж и превращает его в лотерею!")

    # 4. Simulation of Strategy B: Simultaneous Pre-funded Arbitrage
    print("\n" + "="*60)
    print("🚀 3. СИМУЛЯЦИЯ ОДНОВРЕМЕННОГО АРБИТРАЖА (Предзагруженные балансы)")
    print("="*60)
    
    # Initial Balances
    INIT_USDT = 5000.0
    INIT_PRL = 5000.0
    
    TRADE_SIZE_USD = 50.0 # Trade size per arbitrage execution
    MIN_PROFIT_SPREAD = 0.008 # 0.8% threshold (0.4% fees + 0.4% net profit)
    COOLDOWN_S = 10 # 10 seconds cooldown between trades to avoid eating non-existent depth
    
    # We test two modes:
    # Mode 1: No rebalancing (Pure Simultaneous)
    # Mode 2: Inventory-Skewed (Threshold shifts when balance gets skewed)
    
    def simulate_arb(use_skew=False):
        st_usdt = INIT_USDT
        st_prl = INIT_PRL
        bo_usdt = INIT_USDT
        bo_prl = INIT_PRL
        
        trades = []
        last_trade_time = -999999
        
        timestamps = m.index
        st_ask = m['ST_ask_1_price'].values
        st_bid = m['ST_bid_1_price'].values
        bo_ask = m['BO_ask_1_price'].values
        bo_bid = m['BO_bid_1_price'].values
        
        n = len(m)
        
        history = []
        
        for i in range(n):
            if i - last_trade_time < COOLDOWN_S:
                continue
                
            p_st_ask = st_ask[i]
            p_st_bid = st_bid[i]
            p_bo_ask = bo_ask[i]
            p_bo_bid = bo_bid[i]
            
            # Opportunity 1: Buy on SafeTrade, Sell on BigONE
            spread1 = (p_bo_bid - p_st_ask) / p_st_ask
            
            # Opportunity 2: Buy on BigONE, Sell on SafeTrade
            spread2 = (p_st_bid - p_bo_ask) / p_bo_ask
            
            # Inventory Skew adjustment
            # If SafeTrade is running low on USDT or BigONE running low on PRL, require higher spread1
            thresh1 = MIN_PROFIT_SPREAD
            thresh2 = MIN_PROFIT_SPREAD
            
            if use_skew:
                # Skew ratio: ideal is 50%
                st_usdt_ratio = st_usdt / (st_usdt + bo_usdt + 1e-6)
                bo_prl_ratio = bo_prl / (st_prl + bo_prl + 1e-6)
                
                # If ST USDT < 30%, increase thresh1
                if st_usdt_ratio < 0.3:
                    thresh1 += (0.3 - st_usdt_ratio) * 0.05
                if bo_prl_ratio < 0.3:
                    thresh1 += (0.3 - bo_prl_ratio) * 0.05
                    
                # Reverse for thresh2
                bo_usdt_ratio = bo_usdt / (st_usdt + bo_usdt + 1e-6)
                st_prl_ratio = st_prl / (st_prl + bo_prl + 1e-6)
                if bo_usdt_ratio < 0.3:
                    thresh2 += (0.3 - bo_usdt_ratio) * 0.05
                if st_prl_ratio < 0.3:
                    thresh2 += (0.3 - st_prl_ratio) * 0.05
                    
            executed = False
            
            # Check Trade 1: Buy ST, Sell BO
            if spread1 > thresh1:
                # Need USDT on ST and PRL on BO
                q_to_buy = TRADE_SIZE_USD / p_st_ask
                cost_st = TRADE_SIZE_USD * 1.002 # with 0.2% taker fee
                revenue_bo = (q_to_buy * p_bo_bid) * 0.998 # with 0.2% taker fee
                
                if st_usdt >= cost_st and bo_prl >= q_to_buy:
                    st_usdt -= cost_st
                    st_prl += q_to_buy
                    
                    bo_prl -= q_to_buy
                    bo_usdt += revenue_bo
                    
                    executed = True
                    trades.append({
                        'time': timestamps[i],
                        'type': 'BUY_ST_SELL_BO',
                        'spread': spread1,
                        'profit_usd': revenue_bo - cost_st,
                        'q_prl': q_to_buy
                    })
                    
            # Check Trade 2: Buy BO, Sell ST
            elif spread2 > thresh2:
                q_to_buy = TRADE_SIZE_USD / p_bo_ask
                cost_bo = TRADE_SIZE_USD * 1.002
                revenue_st = (q_to_buy * p_st_bid) * 0.998
                
                if bo_usdt >= cost_bo and st_prl >= q_to_buy:
                    bo_usdt -= cost_bo
                    bo_prl += q_to_buy
                    
                    st_prl -= q_to_buy
                    st_usdt += revenue_st
                    
                    executed = True
                    trades.append({
                        'time': timestamps[i],
                        'type': 'BUY_BO_SELL_ST',
                        'spread': spread2,
                        'profit_usd': revenue_st - cost_bo,
                        'q_prl': q_to_buy
                    })
                    
            if executed:
                last_trade_time = i
                
            # Log periodic stats
            if i % 1000 == 0 or i == n - 1:
                history.append({
                    'time': timestamps[i],
                    'total_prl': st_prl + bo_prl,
                    'total_usdt': st_usdt + bo_usdt,
                    'st_prl': st_prl,
                    'bo_prl': bo_prl,
                    'st_usdt': st_usdt,
                    'bo_usdt': bo_usdt,
                })
                
        return pd.DataFrame(trades), pd.DataFrame(history)

    print("\n--- ТЕСТ 1: Классический арбитраж без управления инвентарем ---")
    trades_raw, hist_raw = simulate_arb(use_skew=False)
    print(f"Сделок исполнено: {len(trades_raw)}")
    if len(trades_raw) > 0:
        total_profit = trades_raw['profit_usd'].sum()
        end_st_usdt = hist_raw['st_usdt'].iloc[-1]
        end_bo_usdt = hist_raw['bo_usdt'].iloc[-1]
        end_st_prl = hist_raw['st_prl'].iloc[-1]
        end_bo_prl = hist_raw['bo_prl'].iloc[-1]
        
        print(f"Суммарный профит в USDT: ${total_profit:+.2f}")
        print(f"Финальные балансы:")
        print(f"  SafeTrade: {end_st_prl:.1f} PRL | ${end_st_usdt:.1f} USDT")
        print(f"  BigONE:    {end_bo_prl:.1f} PRL | ${end_bo_usdt:.1f} USDT")
        print(f"  Всего PRL:  {end_st_prl + end_bo_prl:.1f} (было {INIT_PRL*2:.1f})")
        print(f"  Всего USDT: ${end_st_usdt + end_bo_usdt:.1f} (было ${INIT_USDT*2:.1f})")
        
    print("\n--- ТЕСТ 2: Адаптивный арбитраж с управлением инвентарем (Inventory Skew) ---")
    trades_skew, hist_skew = simulate_arb(use_skew=True)
    print(f"Сделок исполнено: {len(trades_skew)}")
    if len(trades_skew) > 0:
        total_profit_skew = trades_skew['profit_usd'].sum()
        end_st_usdt = hist_skew['st_usdt'].iloc[-1]
        end_bo_usdt = hist_skew['bo_usdt'].iloc[-1]
        end_st_prl = hist_skew['st_prl'].iloc[-1]
        end_bo_prl = hist_skew['bo_prl'].iloc[-1]
        
        print(f"Суммарный профит в USDT: ${total_profit_skew:+.2f}")
        print(f"Финальные балансы:")
        print(f"  SafeTrade: {end_st_prl:.1f} PRL | ${end_st_usdt:.1f} USDT")
        print(f"  BigONE:    {end_bo_prl:.1f} PRL | ${end_bo_usdt:.1f} USDT")
        print(f"  Всего PRL:  {end_st_prl + end_bo_prl:.1f} (было {INIT_PRL*2:.1f})")
        print(f"  Всего USDT: ${end_st_usdt + end_bo_usdt:.1f} (было ${INIT_USDT*2:.1f})")

if __name__ == '__main__':
    run_simulation()
