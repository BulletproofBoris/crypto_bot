#!/usr/bin/env python3
"""
_tools/extract_trade_kpi_analysis.py
=============================================================================
Качественный и количественный расчет КПД каждой принятой операции
для лучших моделей Урока 1 и Урока 2 на протяжении всех 600 часов истории.
Запускается на CPU без вмешательства в текущее GPU-обучение.
=============================================================================
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _tools.multitask_lob_net import MultiTaskLOBNet
from _tools.vectorized_cuda_l2_env import VectorizedCudaL2Env
from _tools.pretrain_lob_backbone import build_lob_10_and_flow_tensors

CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "rl_guard_lob_cache_600h.npz"
L1_CHAMP_PATH = PROJECT_ROOT / "models" / "prl" / "archive_run1" / "lesson_1_motorics" / "iteration_500_champ.pt"
L2_CHAMP_PATH = PROJECT_ROOT / "models" / "prl" / "archive_run1" / "lesson_2_spread" / "iteration_500_champ.pt"
OUT_JSON = PROJECT_ROOT / "data" / "reports" / "kpi_comparison_l1_l2.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def simulate_model_trade_by_trade(champ_path: Path, model_name: str, raw, lob_tensors, flow_market, targets_tensor, sim_steps: int = 3000):
    device = torch.device("cpu")
    total_ticks = len(raw["bid_p"])
    step_stride = max(1, (total_ticks - 100) // sim_steps)

    # Загружаем чекпоинт
    c_data = torch.load(champ_path, map_location=device)
    backbone_model = MultiTaskLOBNet(flow_dim=16, n_actions=7, latent_dim=256).to(device)
    backbone_model.load_state_dict(c_data["backbone_state"], strict=False)
    backbone_model.eval()

    w1 = c_data["w1"].float().to(device) # (260, H)
    b1 = c_data["b1"].float().to(device) # (1, H)
    w2 = c_data["w2"].float().to(device) # (H, 7)
    b2 = c_data["b2"].float().to(device) # (1, 7)

    vec_env = VectorizedCudaL2Env(
        bids=torch.from_numpy(raw["bid_p"]).float(),
        asks=torch.from_numpy(raw["ask_p"]).float(),
        bids_q=torch.from_numpy(raw["bid_q"]).float(),
        asks_q=torch.from_numpy(raw["ask_q"]).float(),
        buy_vol=torch.from_numpy(raw["buy_vol"]).float(),
        sell_vol=torch.from_numpy(raw["sell_vol"]).float(),
        lob_tensors=torch.from_numpy(lob_tensors).float(),
        flow_market=torch.from_numpy(flow_market).float(),
        targets=targets_tensor,
        num_envs=1,
        device=device,
    )

    vec_env.reset_interval(start_idx=0, end_idx=total_ticks, episode_len=total_ticks)
    h_state = backbone_model.init_hidden(batch_size=1, device=device)

    trades = []
    current_trade = None
    step_records = []

    cum_pnl = 0.0
    cum_cycles = 0
    cum_quotes = 0
    cum_locks = 0

    with torch.no_grad():
        for s in range(sim_steps):
            cur_t = s * step_stride
            if cur_t >= total_ticks - 1:
                break
            vec_env.t[0] = cur_t
            hour = cur_t / 3600.0

            b1_px = float(vec_env.bids[cur_t, 0].item())
            a1_px = float(vec_env.asks[cur_t, 0].item())
            spread_ticks = (a1_px - b1_px) / max(1e-4, vec_env.tick_size)

            lob, flow, _ = vec_env.get_observations(cur_t_scalar=cur_t)
            masks = vec_env.get_action_masks(cur_t_scalar=cur_t)

            lob_repr = backbone_model.encode_lob_batch(lob)
            z, h_state = backbone_model.backbone.forward_step_fast(lob_repr, flow, h_state)

            p_sweep = torch.sigmoid(backbone_model.head_sweep_12s(z))
            p_bleed = torch.sigmoid(backbone_model.head_bleed_60s(z))
            p_crash = torch.sigmoid(backbone_model.head_crash_300s(z))
            p_exit = torch.sigmoid(backbone_model.head_exit_30s(z))
            z_aug = torch.cat([z, p_sweep, p_bleed, p_crash, p_exit], dim=-1).unsqueeze(1)

            h1 = torch.relu(torch.bmm(z_aug, w1.unsqueeze(0)) + b1)
            logits = (torch.bmm(h1, w2.unsqueeze(0)) + b2).squeeze(1)
            logits = logits.masked_fill(~masks, -1e9)
            action = int(logits.argmax(dim=-1).item())

            prev_pos = float(vec_env.pos_qty[0].item())
            prev_cash = float(vec_env.cash[0].item())

            vec_env.step(torch.tensor([action], device=device))

            cur_pos = float(vec_env.pos_qty[0].item())
            cur_cash = float(vec_env.cash[0].item())
            cur_mtm = float(vec_env.prev_mtm[0].item())

            if 1 <= action <= 4:
                cum_quotes += 1

            # Фиксация входа в сделку
            if prev_pos < 1e-6 and cur_pos > 1e-6:
                current_trade = {
                    "entry_step": s,
                    "entry_hour": round(hour, 2),
                    "entry_px": float(vec_env.avg_entry[0].item()),
                    "entry_spread_ticks": round(spread_ticks, 1),
                    "order_action": action,
                    "pos_qty": cur_pos,
                }

            # Фиксация выхода из сделки (завершение цикла)
            if prev_pos > 1e-6 and cur_pos < 1e-6 and current_trade is not None:
                cum_cycles += 1
                exit_px = b1_px # выход по биду или аску
                hold_steps = s - current_trade["entry_step"]
                hold_hours = round(hold_steps * (step_stride / 3600.0), 2)
                
                # Долларовый PnL сделки (с вычетом комиссий)
                trade_cash_delta = cur_cash - prev_cash
                # Чистый PnL этой сделки
                net_trade_pnl = (exit_px - current_trade["entry_px"]) * current_trade["pos_qty"] - (
                    (current_trade["pos_qty"] * current_trade["entry_px"] * 0.001) + (current_trade["pos_qty"] * exit_px * 0.001)
                )

                # КПД операции: 
                # 1. Доходность на единицу риска/спреда (Efficiency Ratio)
                gross_spread_usd = current_trade["pos_qty"] * (current_trade["entry_spread_ticks"] * vec_env.tick_size)
                kpd_spread = round((net_trade_pnl / max(0.01, gross_spread_usd)) * 100.0, 1)

                # 2. Доходность на капитал в базисных пунктах (bps)
                notional = current_trade["pos_qty"] * current_trade["entry_px"]
                kpd_bps = round((net_trade_pnl / max(1.0, notional)) * 10000.0, 1)

                trades.append({
                    "trade_id": len(trades) + 1,
                    "entry_hour": current_trade["entry_hour"],
                    "exit_hour": round(hour, 2),
                    "hold_hours": hold_hours,
                    "spread_ticks": current_trade["entry_spread_ticks"],
                    "entry_px": round(current_trade["entry_px"], 2),
                    "exit_px": round(exit_px, 2),
                    "net_pnl": round(net_trade_pnl, 2),
                    "kpd_bps": kpd_bps,
                    "kpd_spread_pct": kpd_spread,
                    "cum_pnl": round(cur_mtm, 2),
                })
                current_trade = None

            if s % 20 == 0 or s == sim_steps - 1:
                step_records.append({
                    "step": s,
                    "hour": round(hour, 1),
                    "mtm_pnl": round(cur_mtm, 2),
                    "cycles": cum_cycles,
                    "quotes": cum_quotes,
                })

    # Сводные метрики КПД
    n_trades = len(trades)
    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = round(len(wins) / max(1, n_trades) * 100.0, 1)
    profit_factor = round(sum(wins) / max(0.01, abs(sum(losses))), 2) if losses else 999.0

    avg_win = round(float(np.mean(wins)), 2) if wins else 0.0
    avg_loss = round(float(np.mean(losses)), 2) if losses else 0.0
    avg_pnl = round(float(np.mean(pnls)), 2) if pnls else 0.0
    avg_hold = round(float(np.mean([t["hold_hours"] for t in trades])), 2) if trades else 0.0

    wide_spread_trades = sum(1 for t in trades if t["spread_ticks"] >= 3.0)
    wide_spread_pct = round(wide_spread_trades / max(1, n_trades) * 100.0, 1)

    summary = {
        "model_name": model_name,
        "total_trades": n_trades,
        "final_pnl": round(cur_mtm, 2),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_hold_hours": avg_hold,
        "wide_spread_trades": wide_spread_trades,
        "wide_spread_pct": wide_spread_pct,
        "trades": trades,
        "trajectory": step_records,
    }
    return summary


def main():
    print("Загрузка данных для сравнительного анализа КПД Урока 1 и Урока 2...")
    raw = np.load(CACHE_PATH)
    lob_tensors, flow_market = build_lob_10_and_flow_tensors(raw)
    targets_tensor = torch.zeros(len(raw["bid_p"]), 4)

    print("Симуляция Чемпиона Урока 1 (Моторика)...")
    res_l1 = simulate_model_trade_by_trade(L1_CHAMP_PATH, "Чемпион Урока 1 (Школа Моторики)", raw, lob_tensors, flow_market, targets_tensor)
    print(f"Урок 1: Сделок {res_l1['total_trades']}, PnL ${res_l1['final_pnl']}, WinRate {res_l1['win_rate_pct']}%, Спреды >=3t: {res_l1['wide_spread_pct']}%")

    print("Симуляция Чемпиона Урока 2 (Экономика Спреда)...")
    res_l2 = simulate_model_trade_by_trade(L2_CHAMP_PATH, "Чемпион Урока 2 (Экономика Спреда)", raw, lob_tensors, flow_market, targets_tensor)
    print(f"Урок 2: Сделок {res_l2['total_trades']}, PnL ${res_l2['final_pnl']}, WinRate {res_l2['win_rate_pct']}%, Спреды >=3t: {res_l2['wide_spread_pct']}%")

    output_payload = {
        "lesson_1": res_l1,
        "lesson_2": res_l2,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Анализ успешно сохранен в: {OUT_JSON}")


if __name__ == "__main__":
    main()

