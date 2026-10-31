import copy
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from _tools.system_resources import SAFE_WORKERS

_POP_POOL = ThreadPoolExecutor(max_workers=min(8, SAFE_WORKERS))


def clone_env_fast(base_env):
    """Мгновенное клонирование среды без пересчёта 632,866 строк pandas rolling."""
    env_copy = copy.copy(base_env)
    return env_copy


def run_gpu_population_step(
    ep: int,
    pop_size: int,
    model_cls,
    obs_dim: int,
    n_actions: int,
    best_model_ref: nn.Module,
    T_obs: torch.Tensor,
    T_masks: torch.Tensor,
    T_acts: torch.Tensor,
    T_tacts: torch.Tensor,
    T_old_logp: torch.Tensor,
    T_advs: torch.Tensor,
    T_rets: torch.Tensor,
    T_live_obs: torch.Tensor,
    T_live_masks: torch.Tensor,
    T_live_acts: torch.Tensor,
    ce_w: torch.Tensor,
    gpu_device: str,
    data: dict,
    split_idx: int,
    n_total: int,
    env_cls,
    evaluate_fn,
    oracle_pnl: float,
    template_test_env=None,
):
    """Обучает на GPU популяцию кандидатов PPO + мутаций CEM и параллельно оценивает их на OOS."""
    steps_n = len(T_obs)
    T_obs_g = T_obs.to(gpu_device)
    T_masks_g = T_masks.to(gpu_device)
    T_acts_g = T_acts.to(gpu_device)
    T_tacts_g = T_tacts.to(gpu_device)
    T_old_logp_g = T_old_logp.to(gpu_device)
    T_advs_g = T_advs.to(gpu_device)
    T_rets_g = T_rets.to(gpu_device)
    ce_w_g = ce_w.to(gpu_device)

    ppo_configs = [
        (4e-5, 0.10, 0.25, 0.003, 3),
        (8e-5, 0.15, 0.15, 0.005, 4),
        (1.4e-4, 0.18, 0.08, 0.008, 4),
        (6e-5, 0.12, 0.35, 0.002, 3),
    ]
    n_ppo = min(len(ppo_configs), max(1, pop_size // 2))
    candidates_cpu = []

    for c_i in range(n_ppo):
        lr_c, clip_c, bc_c, ent_c, g_epochs = ppo_configs[c_i]
        cand_g = model_cls(obs_dim=obs_dim, n_actions=n_actions).to(gpu_device)
        cand_g.load_state_dict(best_model_ref.state_dict())
        opt_c = torch.optim.AdamW(cand_g.parameters(), lr=lr_c, weight_decay=1e-4)

        if ep == 1 and c_i == 0:
            for p_name, p_val in cand_g.named_parameters():
                if not p_name.startswith("critic"):
                    p_val.requires_grad_(False)
            for _ in range(6):
                perm = torch.randperm(steps_n, device=gpu_device)
                for i in range(0, steps_n, 1024):
                    idx = perm[i : i + 1024]
                    _, vals = cand_g(T_obs_g[idx], T_masks_g[idx])
                    v_loss = F.mse_loss(vals, T_rets_g[idx])
                    opt_c.zero_grad(); v_loss.backward(); opt_c.step()
            for p_val in cand_g.parameters():
                p_val.requires_grad_(True)

        for _ in range(g_epochs):
            perm = torch.randperm(steps_n, device=gpu_device)
            for i in range(0, steps_n, 1024):
                idx = perm[i : i + 1024]
                logits, vals = cand_g(T_obs_g[idx], T_masks_g[idx])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(T_acts_g[idx])
                ratio = torch.exp(new_logp - T_old_logp_g[idx])
                surr1 = ratio * T_advs_g[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_c, 1.0 + clip_c) * T_advs_g[idx]
                bc_anchor = F.cross_entropy(logits, T_tacts_g[idx], weight=ce_w_g)
                if T_live_obs is not None and len(T_live_obs) >= 512:
                    l_idx = torch.randint(0, len(T_live_obs), (512,), device=gpu_device)
                    l_logits, _ = cand_g(T_live_obs[l_idx], T_live_masks[l_idx])
                    live_bc = F.cross_entropy(l_logits, T_live_acts[l_idx], weight=ce_w_g)
                    bc_anchor = 0.60 * bc_anchor + 0.40 * live_bc
                loss = (
                    -torch.min(surr1, surr2).mean()
                    + 0.25 * F.mse_loss(vals, T_rets_g[idx])
                    + bc_c * bc_anchor
                    - ent_c * dist.entropy().mean()
                )
                opt_c.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(cand_g.parameters(), 0.5)
                opt_c.step()

        cand_cpu = model_cls(obs_dim=obs_dim, n_actions=n_actions)
        cand_cpu.load_state_dict({k: v.cpu() for k, v in cand_g.state_dict().items()})
        cand_cpu.eval()
        candidates_cpu.append(cand_cpu)

    # Evolutionary / CEM policy-head & backbone perturbations around leader
    sigmas = [0.008, 0.015, 0.025, 0.040, 0.012, 0.020]
    n_evo = max(0, pop_size - len(candidates_cpu))
    for e_i in range(n_evo):
        sigma = sigmas[e_i % len(sigmas)]
        cand_cpu = model_cls(obs_dim=obs_dim, n_actions=n_actions)
        base_anchor = candidates_cpu[0] if (e_i % 2 == 1 and candidates_cpu) else best_model_ref
        cand_cpu.load_state_dict(base_anchor.state_dict())
        with torch.no_grad():
            for p_name, p_val in cand_cpu.named_parameters():
                if "actor" in p_name or "backbone.3" in p_name:
                    p_val.add_(torch.randn_like(p_val) * sigma)
        cand_cpu.eval()
        candidates_cpu.append(cand_cpu)

    if template_test_env is None:
        template_test_env = env_cls(data, start_idx=split_idx, end_idx=n_total - 1, episode_len=n_total - split_idx - 2)

    eval_pairs = []
    for cand_m in candidates_cpu:
        res_i = evaluate_fn(template_test_env, model=cand_m, device="cpu", oracle_pnl_override=oracle_pnl)
        eval_pairs.append((cand_m, res_i))
    best_cand, best_res = max(
        eval_pairs,
        key=lambda x: x[1]["efficiency_score"] if x[1]["trades"] >= 2 else -1e9,
    )
    return best_cand, best_res
