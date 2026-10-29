#!/usr/bin/env python3
"""
Конвейер Безоткатной Эволюции (Non-Regression Gate) для RL-трейдера PRL/USDT.
Автоматически прогоняет текущую боевую политику по 5 эталонным рыночным режимам
(Golden Benchmark Suite) и сверяет паспорт метрик с зафиксированным эталоном
(models/prl/rl_execution/regression_baseline_passport.json).
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.train_rl_execution import SafeTradeL2ExecutionEnv, MaskableActorCritic

PASSPORT_PATH = BASE_DIR / "models" / "prl" / "rl_execution" / "regression_baseline_passport.json"


def _load_best_cache(coin: str = "prl") -> dict:
    cache_dir = BASE_DIR / "data" / "cache"
    cands = sorted(cache_dir.glob(f"rl_triad_lob_cache_{coin}_*h.npz"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"Кэш RL-полигона не найден в {cache_dir}")
    c_path = cands[-1]
    data = np.load(c_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def select_golden_regimes(data: dict, win_sec: int = 14400) -> list[dict]:
    """Выбирает 5 детерминированных 4-часовых эталонных режимов из активной истории L2 стакана."""
    b1 = np.maximum(data["bid_p"][:, 0], float(data["tick_size"]))
    a1 = np.maximum(data["ask_p"][:, 0], b1 + float(data["tick_size"]))
    tick = float(data["tick_size"])
    sp = (a1 - b1) / tick
    b_vol = data["buy_vol"]
    s_vol = data["sell_vol"]
    b1_s = pd.Series(b1)
    dd_300 = (b1_s.rolling(300, min_periods=1).max().values - b1) / tick
    n = len(b1)
    step = 3600
    starts = [
        s for s in range(14400, n - win_sec - 600, step)
        if np.sum(b_vol[s:s + win_sec] > 0) >= 40
        and np.sum(s_vol[s:s + win_sec] > 0) >= 40
        and np.sum(dd_300[s:s + win_sec] >= 1.0) >= 300
    ]
    if not starts:
        starts = list(range(14400, n - win_sec - 600, step))

    best_pump = max(starts, key=lambda s: (b1[s + win_sec] - b1[s]) / tick + 0.5 * (np.max(b1[s:s + win_sec]) - b1[s]) / tick)
    best_dump = min(starts, key=lambda s: (b1[s + win_sec] - b1[s]) / tick - 0.5 * (b1[s] - np.min(b1[s:s + win_sec])) / tick)
    best_wide = max(
        starts,
        key=lambda s: float(np.mean(sp[s:s + win_sec] >= 1.8)) * 10.0 + float(np.mean((b_vol[s:s + win_sec] + s_vol[s:s + win_sec]) > 0)) * 8.0 - abs((b1[s + win_sec] - b1[s]) / tick) * 0.25,
    )
    best_tight = max(
        starts,
        key=lambda s: float(np.mean(sp[s:s + win_sec] <= 1.2)) * 10.0 + float(np.mean((b_vol[s:s + win_sec] + s_vol[s:s + win_sec]) > 0)) * 8.0 - abs((b1[s + win_sec] - b1[s]) / tick) * 0.25,
    )
    recov_cands = [s for s in starts if s >= n // 2 and np.sum(b_vol[s:s + 1800] > 0) >= 15]
    best_recov = recov_cands[-1] if recov_cands else starts[-1]

    return [
        {"id": "R1_PUMP_BREAKOUT", "title": "🚀 Памп и пробой вверх", "start": int(best_pump), "end": int(best_pump + win_sec), "force_underwater": False},
        {"id": "R2_TOXIC_DUMP", "title": "🩸 Токсичный пролив вниз", "start": int(best_dump), "end": int(best_dump + win_sec), "force_underwater": False},
        {"id": "R3_WIDE_SPREAD_SCALP", "title": "🦀 Живой коридор (спред >= 2т)", "start": int(best_wide), "end": int(best_wide + win_sec), "force_underwater": False},
        {"id": "R4_TIGHT_1TICK_CHOP", "title": "🧱 Вязкий флэт (спред 1т)", "start": int(best_tight), "end": int(best_tight + win_sec), "force_underwater": False},
        {"id": "R5_UNDERWATER_RECOVERY", "title": "🔄 Свежий рынок + выход из -6т", "start": int(best_recov), "end": int(best_recov + win_sec), "force_underwater": True},
    ]


def _slice_data(data: dict, start: int, end: int, pad: int = 14400) -> tuple[dict, int, int]:
    s0 = max(0, start - pad)
    e0 = min(len(data["bid_p"]), end + 10)
    sub = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(data["bid_p"]):
            sub[k] = v[s0:e0]
        else:
            sub[k] = v
    return sub, start - s0, end - s0


def evaluate_regime(data: dict, agent: MaskableActorCritic, reg: dict, order_usd: float = 50.0, enable_unfreeze: bool = True) -> dict:
    sub_data, rel_s, rel_e = _slice_data(data, reg["start"], reg["end"])
    env = SafeTradeL2ExecutionEnv(sub_data, rel_s, rel_e, episode_len=rel_e - rel_s - 2, order_usd=order_usd)
    env.enable_unfreeze = bool(enable_unfreeze)
    obs = env.reset(eval_mode=True)

    if reg.get("force_underwater", False):
        b1_0 = max(env.bid_p[env.t, 0], env.tick_size)
        env.avg_entry = round((b1_0 + 6.0 * env.tick_size) / env.tick_size) * env.tick_size
        env.pos_qty = round(order_usd / env.avg_entry, 4)
        env.entry_t = env.t - 305
        env.order_side = "SELL"
        env.order_price = round((env.avg_entry + env.tick_size) / env.tick_size) * env.tick_size
        env.order_qty = env.pos_qty
        env.q_ahead, _ = env._get_lob_queue(env.t, "SELL", env.order_price)
        env.active_after = env.t
        env.prev_mtm = env.pos_qty * (b1_0 - env.avg_entry)
        obs = env._get_obs()

    baseline_mtm = float(env.prev_mtm)
    max_mtm = baseline_mtm
    max_dd = 0.0
    max_hold_sec = 0
    requotes = 0
    done = False

    with torch.no_grad():
        while not done:
            mask = env.get_action_mask()
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            logits, _ = agent(obs_t, mask_t)
            act = int(torch.argmax(logits, dim=-1).item())
            prev_px = env.order_price if env.order_side else 0.0
            obs, _, done = env.step(act)
            if env.order_side and abs(env.order_price - prev_px) > 1e-5:
                requotes += 1
            if env.pos_qty > 1e-8:
                max_hold_sec = max(max_hold_sec, env.t - env.entry_t)
            rel_mtm = float(env.prev_mtm) - baseline_mtm
            max_mtm = max(max_mtm, rel_mtm)
            max_dd = max(max_dd, max_mtm - rel_mtm)

    net_pnl = float(env.prev_mtm) - baseline_mtm
    return {
        "id": reg["id"],
        "title": reg["title"],
        "net_pnl_usd": round(net_pnl, 4),
        "net_pnl_pct": round(net_pnl / order_usd * 100.0, 2),
        "n_trades": int(env.n_trades),
        "max_freeze_min": round(max_hold_sec / 60.0, 1),
        "max_dd_usd": round(max_dd, 4),
        "requotes": int(requotes),
    }


def run_regression_gate(save_as_baseline: bool = False, order_usd: float = 50.0) -> pd.DataFrame:
    t0 = time.perf_counter()
    data = _load_best_cache("prl")
    obs_dim = data["X_market"].shape[1] + 10
    ckpt_path = BASE_DIR / "models" / "prl" / "rl_execution" / "best_maskable_ppo.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    hidden_dim = state_dict["backbone.0.weight"].shape[0]
    agent = MaskableActorCritic(obs_dim=obs_dim, n_actions=7, hidden_dim=hidden_dim)
    agent.load_state_dict(state_dict)
    agent.eval()

    regimes = select_golden_regimes(data)
    cand_results = [evaluate_regime(data, agent, r, order_usd=order_usd, enable_unfreeze=True) for r in regimes]

    if save_as_baseline or not PASSPORT_PATH.exists():
        frozen_results = [evaluate_regime(data, agent, r, order_usd=order_usd, enable_unfreeze=False) for r in regimes]
        PASSPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        PASSPORT_PATH.write_text(
            json.dumps({"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "regimes": cand_results, "prev_frozen": frozen_results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        base_map = {r["id"]: r for r in frozen_results}
        comp_label = "Старый (заморозка 75%)"
    else:
        saved = json.loads(PASSPORT_PATH.read_text(encoding="utf-8"))
        base_map = {r["id"]: r for r in saved["regimes"]}
        comp_label = f"Эталон ({saved.get('updated_at', 'Baseline')})"

    rows = []
    all_passed = True
    for c in cand_results:
        b = base_map.get(c["id"], c)
        d_pnl = round(c["net_pnl_pct"] - b["net_pnl_pct"], 2)
        max_allowed_freeze = max(45.0, b["max_freeze_min"] + 1.0)
        passed = (d_pnl >= -0.20) and (c["max_freeze_min"] <= max_allowed_freeze)
        if not passed:
            all_passed = False
        rows.append({
            "Режим": c["title"],
            f"PnL % ({comp_label})": b["net_pnl_pct"],
            "PnL % (Текущий)": c["net_pnl_pct"],
            "Δ PnL %": f"{d_pnl:+.2f}%",
            "Сделок (было -> стало)": f"{b['n_trades']} -> {c['n_trades']}",
            "Макс. зависание (мин)": f"{b['max_freeze_min']} -> {c['max_freeze_min']}",
            "Статус": "✅ PASS" if passed else "❌ REGRESSION",
        })

    df_res = pd.DataFrame(rows)
    tot_base = sum(base_map[c["id"]]["net_pnl_pct"] for c in cand_results)
    tot_cand = sum(c["net_pnl_pct"] for c in cand_results)
    verdict = "✅ ДОПУЩЕНО К ДЕПЛОЮ (Non-Regression Gate PASSED)" if all_passed and tot_cand >= tot_base - 0.10 else "❌ ОТКЛОНЕНО: ОБНАРУЖЕНА РЕГРЕССИЯ"
    print(f"🛡️ Проверка Безоткатной Эволюции (5 эталонных режимов x 4ч = 20ч L2 стакана) за {time.perf_counter() - t0:.1f} сек:")
    print(f"📊 Суммарный PnL по 5 режимам: {tot_base:+.2f}% ➔ {tot_cand:+.2f}% (Δ={tot_cand - tot_base:+.2f}%) | Вердикт: {verdict}\n")
    print(df_res.to_string(index=False))
    return df_res


def finetune_ppo_with_gate(epochs: int = 8, steps_per_epoch: int = 16384, order_usd: float = 50.0):
    """H4: Дообучение Maskable PPO внутри обновлённой среды (H1+H2+H3) под строгим контролем шлюза 5 режимов."""
    import shutil
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical

    data = _load_best_cache("prl")
    regimes = select_golden_regimes(data)
    obs_dim = data["X_market"].shape[1] + 10
    ckpt_path = BASE_DIR / "models" / "prl" / "rl_execution" / "best_maskable_ppo.pt"
    backup_path = BASE_DIR / "models" / "prl" / "rl_execution" / "best_maskable_ppo_pre_h4.pt"
    shutil.copy2(ckpt_path, backup_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    hidden_dim = state_dict["backbone.0.weight"].shape[0]

    model = MaskableActorCritic(obs_dim=obs_dim, n_actions=7, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    best_ref = MaskableActorCritic(obs_dim=obs_dim, n_actions=7, hidden_dim=hidden_dim)
    best_ref.load_state_dict(state_dict)
    best_ref.eval()

    base_eval = [evaluate_regime(data, best_ref, r, order_usd=order_usd, enable_unfreeze=True) for r in regimes]
    best_map = {r["id"]: r for r in base_eval}
    best_tot_pnl = sum(r["net_pnl_pct"] for r in base_eval)
    print(f"🎯 H4 Старт дообучения PPO: текущий эталон по 5 режимам = {best_tot_pnl:+.2f}%", flush=True)

    n_total = len(data["X_market"])
    train_env = SafeTradeL2ExecutionEnv(data, start_idx=max(14400, n_total - 260 * 3600), end_idx=n_total - 20 * 3600, episode_len=7200, order_usd=order_usd)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-4)
    ce_w = torch.tensor([1.0, 15.0, 15.0, 4.0, 4.0, 6.0, 8.0], dtype=torch.float32)

    for ep in range(1, epochs + 1):
        model.train()
        b_obs, b_masks, b_acts, b_tacts, b_logp, b_rews, b_vals, b_dones = [], [], [], [], [], [], [], []
        obs = train_env.reset()
        for _ in range(steps_per_epoch):
            mask = train_env.get_action_mask()
            with torch.no_grad():
                o_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                m_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
                logits, val = model(o_t, m_t)
                ref_logits, _ = best_ref(o_t, m_t)
                t_act = int(torch.argmax(ref_logits, dim=-1).item())
                dist = Categorical(logits=logits)
                act_t = dist.sample()
                logp_t = dist.log_prob(act_t)
            act = int(act_t.item())
            next_obs, rew, done = train_env.step(act)
            b_obs.append(obs); b_masks.append(mask); b_acts.append(act); b_tacts.append(t_act)
            b_logp.append(float(logp_t.item())); b_rews.append(rew); b_vals.append(float(val.item())); b_dones.append(done)
            obs = train_env.reset() if done else next_obs

        advs = np.zeros(steps_per_epoch, dtype=np.float32)
        last_gae = 0.0
        for t_i in reversed(range(steps_per_epoch)):
            next_val = 0.0 if (t_i == steps_per_epoch - 1 or b_dones[t_i]) else b_vals[t_i + 1]
            delta = b_rews[t_i] + 0.995 * next_val - b_vals[t_i]
            last_gae = delta + 0.995 * 0.95 * (0.0 if b_dones[t_i] else last_gae)
            advs[t_i] = last_gae
        rets = advs + np.array(b_vals, dtype=np.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        T_obs = torch.as_tensor(np.array(b_obs), dtype=torch.float32)
        T_masks = torch.as_tensor(np.array(b_masks), dtype=torch.bool)
        T_acts = torch.as_tensor(np.array(b_acts), dtype=torch.long)
        T_tacts = torch.as_tensor(np.array(b_tacts), dtype=torch.long)
        T_old_logp = torch.as_tensor(np.array(b_logp), dtype=torch.float32)
        T_advs = torch.as_tensor(advs, dtype=torch.float32)
        T_rets = torch.as_tensor(rets, dtype=torch.float32)

        for _ in range(2):
            perm = torch.randperm(steps_per_epoch)
            for i in range(0, steps_per_epoch, 1024):
                idx = perm[i:i + 1024]
                logits, vals = model(T_obs[idx], T_masks[idx])
                dist = Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(T_acts[idx]) - T_old_logp[idx])
                surr = torch.min(ratio * T_advs[idx], torch.clamp(ratio, 0.88, 1.12) * T_advs[idx])
                bc_loss = F.cross_entropy(logits, T_tacts[idx], weight=ce_w)
                loss = -surr.mean() + 0.25 * F.mse_loss(vals, T_rets[idx]) + 0.25 * bc_loss - 0.003 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()

        model.eval()
        cand_eval = [evaluate_regime(data, model, r, order_usd=order_usd, enable_unfreeze=True) for r in regimes]
        cand_tot = sum(r["net_pnl_pct"] for r in cand_eval)
        gate_ok = all(
            (c["net_pnl_pct"] >= best_map[c["id"]]["net_pnl_pct"] - 0.15)
            and (c["max_freeze_min"] <= max(45.0, best_map[c["id"]]["max_freeze_min"] + 1.0))
            for c in cand_eval
        )
        if gate_ok and cand_tot > best_tot_pnl + 0.01:
            best_tot_pnl = cand_tot
            best_map = {c["id"]: c for c in cand_eval}
            best_ref.load_state_dict(model.state_dict())
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [H4 Ep {ep}/{epochs}] 🏆 НОВЫЙ РЕКОРД ШЛЮЗА: {cand_tot:+.2f}% -> чекпоинт сохранён!", flush=True)
        else:
            print(f"  [H4 Ep {ep}/{epochs}] ↩️ Кандидат {cand_tot:+.2f}% (gate_ok={gate_ok}) не превзошёл {best_tot_pnl:+.2f}% -> откат к эталону", flush=True)
            with torch.no_grad():
                for p_c, p_b in zip(model.parameters(), best_ref.parameters()):
                    p_c.data.mul_(0.25).add_(p_b.data, alpha=0.75)


if __name__ == "__main__":
    if "--finetune-ppo" in sys.argv:
        finetune_ppo_with_gate()
    save_flag = "--save-baseline" in sys.argv
    run_regression_gate(save_as_baseline=save_flag)
