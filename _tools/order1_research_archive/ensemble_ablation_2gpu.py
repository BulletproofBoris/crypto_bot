#!/usr/bin/env python3
"""
Ablation v4: Proper seq_len + 20 epochs + 2× GPU parallel.
seq_len = min(H*K, 30) — каузальное окно вместо seq_len=1.
On-the-fly sequence generation — экономия RAM.
2 горизонта параллельно на 2 GPU (ThreadPoolExecutor).
"""
import os, sys, json, time, warnings, traceback, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from order1_research_lab import (
    NEURAL_FACTORY, NUM_CLASSES,
    compute_evaluation_metrics, compute_class_weights, FocalLoss,
)
from data_labeler import compute_single_horizon_labels

# ─── Config ───
HORIZONS = [1, 2, 5, 10, 20, 40, 80, 160]
K_STAR = {1:3.0, 2:3.0, 5:3.0, 10:3.0, 20:3.0, 40:5.0, 80:2.5, 160:2.0}
MAX_SL = 30  # cap seq_len
ARCHS = ["mlp_bottleneck", "conv1d_gru", "causal_cnn",
         "attention", "wavenet_tcn", "cross_mixer"]
MAX_ENS = 4
EPOCHS = 20
BS = 2048
WIN_TR = 172800; WIN_VA = 43200; TICK = 0.0001

BASE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(BASE, "..", "data", "features", "prl_1s_features.parquet")
OUT = os.path.join(BASE, "..", "scratch", "ensemble_ablation_v4")

def get_sl(H):
    return min(max(1, int(round(H * K_STAR[H]))), MAX_SL)


# ─── Data Loading ───
def load_data():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET)
    total = pf.metadata.num_rows
    need = WIN_TR + WIN_VA
    start = total - need

    schema = pf.schema_arrow
    all_cols = [f.name for f in schema]
    label_cols = [c for c in all_cols if any(c.startswith(p) for p in
                  ['ST_mid_price','ST_spread','ST_bid_1','ST_ask_1',
                   'ST_bid_2','ST_ask_2','ST_bids_vol','ST_asks_vol',
                   'ST_trade_vol','ST_sec_since_tick','BO_sec_since_tick'])]
    skip_pref = ("class_","label_","target_","z_","ret_")
    drop = {'ST_sec_since_tick','BO_sec_since_tick','is_data_gap','is_st_gap_60s'}
    feat_cols = [c for c in all_cols if c not in drop
                 and not any(c.startswith(p) for p in skip_pref)]
    need_cols = list(set(feat_cols + label_cols +
                         ['ST_sec_since_tick','BO_sec_since_tick']))

    rg_off = sum(pf.metadata.row_group(i).num_rows
                 for i in range(pf.metadata.num_row_groups - 1))
    table = pf.read_row_group(pf.metadata.num_row_groups - 1, columns=need_cols)
    table = table.slice(start - rg_off, need)
    df = table.to_pandas(); del table; gc.collect()

    print(f"  Loaded: {df.shape[0]:,} × {df.shape[1]}, "
          f"{df.memory_usage(deep=True).sum()/1e9:.2f} GB")

    gap = (df['ST_sec_since_tick'] > 60) | (df['BO_sec_since_tick'] > 60)
    tr_mask = ~gap.iloc[:WIN_TR].values
    va_mask = ~gap.iloc[WIN_TR:].values
    print(f"  Valid: {tr_mask.sum():,}/{WIN_TR} tr, {va_mask.sum():,}/{WIN_VA} val")

    df_tr = df.iloc[:WIN_TR].reset_index(drop=True)
    df_va = df.iloc[WIN_TR:].reset_index(drop=True)

    num_f = [c for c in feat_cols if c in df.columns
             and df[c].dtype.kind in ('f','i','u') and c not in drop]
    X_tr = df.iloc[:WIN_TR][num_f].values.astype(np.float32)[tr_mask]
    X_va = df.iloc[WIN_TR:][num_f].values.astype(np.float32)[va_mask]
    X_tr = np.nan_to_num(X_tr, nan=0., posinf=0., neginf=0.)
    X_va = np.nan_to_num(X_va, nan=0., posinf=0., neginf=0.)
    print(f"  Features: X_tr={X_tr.shape} X_va={X_va.shape}")

    del df; gc.collect()
    return X_tr, X_va, df_tr, df_va, tr_mask, va_mask


# ─── LightGBM ───
def train_lgb(X, y, Xv, yv):
    import lightgbm as lgb
    dt = lgb.Dataset(X, label=y, free_raw_data=True)
    dv = lgb.Dataset(Xv, label=yv, reference=dt, free_raw_data=True)
    p = {'objective':'multiclass','num_class':NUM_CLASSES,
         'metric':'multi_logloss','boosting_type':'gbdt',
         'num_leaves':63,'learning_rate':0.05,
         'feature_fraction':0.8,'bagging_fraction':0.8,
         'bagging_freq':5,'verbose':-1,'seed':42,'n_jobs':-1}
    m = lgb.train(p, dt, num_boost_round=70, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(10, verbose=False)])
    pr = m.predict(Xv)
    met = compute_evaluation_metrics(yv, pr)
    del m, dt, dv
    return pr, met


# ─── NN with proper seq_len ───
def train_nn(X_tr_pad, y_tr, X_va_pad, y_va, seq_len, arch, device):
    """Train NN with on-the-fly causal sequence generation."""
    N_tr, N_va = len(y_tr), len(y_va)
    n_f = X_tr_pad.shape[1]
    offsets = np.arange(seq_len)

    def make_batch(Xp, idx):
        seqs = Xp[idx[:, None] + offsets[None, :]]  # (B, seq_len, F)
        return torch.tensor(seqs, dtype=torch.bfloat16, device=device)

    net = NEURAL_FACTORY[arch](seq_len=seq_len, n_features=n_f,
                                n_classes=NUM_CLASSES)
    net = net.to(torch.bfloat16).to(device)
    cw = compute_class_weights(y_tr, strategy="sqrt")
    crit = FocalLoss(alpha=cw, gamma=2.0)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS,
                                                      eta_min=1e-5)
    y_t = torch.tensor(y_tr, dtype=torch.long, device=device)

    best_auc, best_w = -1.0, None
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(N_tr)
        for s in range(0, N_tr - BS + 1, BS):
            idx = perm[s:s + BS]
            xb = make_batch(X_tr_pad, idx)
            opt.zero_grad()
            crit(net(xb).float(), y_t[idx]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            del xb
        sch.step()

        # Validate every 5 epochs
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            net.eval()
            ch = []
            with torch.no_grad():
                for s in range(0, N_va, BS):
                    idx = np.arange(s, min(s + BS, N_va))
                    xb = make_batch(X_va_pad, idx)
                    ch.append(F.softmax(net(xb).float(), dim=-1).cpu().numpy())
                    del xb
            vp = np.concatenate(ch)
            a = float(compute_evaluation_metrics(y_va, vp)["auc_macro"])
            if a > best_auc:
                best_auc = a
                best_w = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    # Final eval with best weights
    if best_w:
        net.load_state_dict(best_w)
    net.eval()
    ch = []
    with torch.no_grad():
        for s in range(0, N_va, BS):
            idx = np.arange(s, min(s + BS, N_va))
            xb = make_batch(X_va_pad, idx)
            ch.append(F.softmax(net(xb).float(), dim=-1).cpu().numpy())
            del xb
    probs = np.concatenate(ch)
    met = compute_evaluation_metrics(y_va, probs)

    del net, y_t, best_w
    torch.cuda.empty_cache(); gc.collect()
    return probs, met


# ─── Run all NNs on one GPU ───
def run_nns_gpu(X_tr_norm, y_tr, X_va_norm, y_va, seq_len, gpu_id, H):
    n_f = X_tr_norm.shape[1]
    # Causal padding: seq_len-1 zero rows prepended
    X_tr_pad = np.vstack([np.zeros((seq_len - 1, n_f), dtype=np.float32),
                          X_tr_norm])
    X_va_pad = np.vstack([np.zeros((seq_len - 1, n_f), dtype=np.float32),
                          X_va_norm])

    results = {}
    for arch in ARCHS:
        t0 = time.time()
        try:
            p, m = train_nn(X_tr_pad, y_tr, X_va_pad, y_va,
                            seq_len, arch, torch.device(f"cuda:{gpu_id}"))
            results[arch] = {"probs": p, "metrics": m}
            print(f"    [GPU{gpu_id}] H={H:3d}s {arch:18s} | "
                  f"AUC={m['auc_macro']:.4f} | Dir={m['conf_dir_acc']:.1f}% | "
                  f"IC={m['ic']:+.4f} | {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"    [GPU{gpu_id}] H={H:3d}s {arch:18s} | FAIL: {e}",
                  flush=True)
            traceback.print_exc()
        gc.collect()

    del X_tr_pad, X_va_pad; gc.collect()
    return results


# ─── Greedy Forward Selection ───
def greedy_select(preds, y_val, max_k=MAX_ENS):
    avail = list(preds.keys())
    sel, best = [], -1.0
    hist = []
    for step in range(max_k):
        bc, ba = None, best
        for n in avail:
            if n in sel:
                continue
            avg = np.mean([preds[x] for x in sel + [n]], axis=0)
            a = float(compute_evaluation_metrics(y_val, avg)["auc_macro"])
            if a > ba:
                ba, bc = a, n
        if bc is None:
            break
        sel.append(bc); best = ba
        avg = np.mean([preds[x] for x in sel], axis=0)
        m = compute_evaluation_metrics(y_val, avg)
        hist.append({"step": step+1, "added": bc, "ensemble": list(sel),
                      "auc_macro": round(float(m["auc_macro"]), 5),
                      "conf_dir_acc": round(float(m["conf_dir_acc"]), 2),
                      "ic": round(float(m["ic"]), 5)})
    return sel, hist


# ─── Main ───
def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 60)
    print("  Ablation v4 — proper seq_len + 20 epochs + 2× GPU")
    print("=" * 60)
    t_all = time.time()

    # Show seq_len table
    print("\n  Seq lengths:")
    for H in HORIZONS:
        raw = int(round(H * K_STAR[H]))
        capped = get_sl(H)
        tag = " (capped)" if raw > MAX_SL else ""
        print(f"    H={H:3d}s → raw={raw:3d} → seq_len={capped:2d}{tag}")

    # Resume support
    partial_path = os.path.join(OUT, "ablation_partial.json")
    done = {}
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            done = {r["H"]: r for r in json.load(f)}
        print(f"\n  Resume: {sorted(done.keys())} done")

    print("\n[1/2] Loading data...")
    X_tr, X_va, df_tr, df_va, tr_mask, va_mask = load_data()

    # Global normalization (same for all horizons)
    mu = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-8
    X_tr_n = np.nan_to_num((X_tr - mu) / std, nan=0., posinf=0.,
                            neginf=0.).clip(-5, 5).astype(np.float32)
    X_va_n = np.nan_to_num((X_va - mu) / std, nan=0., posinf=0.,
                            neginf=0.).clip(-5, 5).astype(np.float32)

    print(f"\n[2/2] Running {len(HORIZONS)} horizons in pairs on 2 GPUs...")
    results = list(done.values())

    pairs = [(HORIZONS[i], HORIZONS[i+1]) for i in range(0, len(HORIZONS), 2)]

    for H_a, H_b in pairs:
        if H_a in done and H_b in done:
            print(f"\n  ⏭ H={H_a}s + H={H_b}s — both done"); continue

        print(f"\n{'='*60}")
        print(f"  PAIR: H={H_a}s (sl={get_sl(H_a)}) + H={H_b}s (sl={get_sl(H_b)})")
        print(f"{'='*60}")

        # Labels for horizons that need running
        to_run = [H for H in [H_a, H_b] if H not in done]
        y_data = {}
        for H in to_run:
            t0 = time.time()
            K = K_STAR[H]
            l_tr = compute_single_horizon_labels(df_tr, "prl", H, K,
                                                  min_vol=0, tick_size=TICK, venue="ST")
            l_va = compute_single_horizon_labels(df_va, "prl", H, K,
                                                  min_vol=0, tick_size=TICK, venue="ST")
            y_tr = (l_tr['class_7'] + 3).clip(0, 6).astype(np.int64)[tr_mask]
            y_va = (l_va['class_7'] + 3).clip(0, 6).astype(np.int64)[va_mask]
            y_data[H] = (y_tr, y_va)
            print(f"  Labels H={H}s: {len(y_tr):,} tr, {len(y_va):,} val "
                  f"({time.time()-t0:.1f}s)")

        # LightGBM — sequential (CPU-bound)
        lgb_data = {}
        for H in to_run:
            t0 = time.time()
            y_tr, y_va = y_data[H]
            p, m = train_lgb(X_tr, y_tr, X_va, y_va)
            lgb_data[H] = (p, m)
            print(f"  LightGBM H={H:3d}s | AUC={m['auc_macro']:.4f} | "
                  f"Dir={m['conf_dir_acc']:.1f}% | IC={m['ic']:+.4f} | "
                  f"{time.time()-t0:.1f}s", flush=True)

        # NNs — parallel on 2 GPUs
        nn_data = {}
        if len(to_run) == 2:
            H0, H1 = to_run
            print(f"\n  NNs: GPU0 → H={H0}s (sl={get_sl(H0)}), "
                  f"GPU1 → H={H1}s (sl={get_sl(H1)})", flush=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                f0 = pool.submit(run_nns_gpu, X_tr_n, y_data[H0][0],
                                 X_va_n, y_data[H0][1], get_sl(H0), 0, H0)
                f1 = pool.submit(run_nns_gpu, X_tr_n, y_data[H1][0],
                                 X_va_n, y_data[H1][1], get_sl(H1), 1, H1)
                nn_data[H0] = f0.result()
                nn_data[H1] = f1.result()
        elif len(to_run) == 1:
            H0 = to_run[0]
            print(f"\n  NNs: GPU0 → H={H0}s (sl={get_sl(H0)})", flush=True)
            nn_data[H0] = run_nns_gpu(X_tr_n, y_data[H0][0],
                                       X_va_n, y_data[H0][1],
                                       get_sl(H0), 0, H0)

        # Greedy selection for each horizon
        for H in to_run:
            y_tr, y_va = y_data[H]
            all_preds = {"LightGBM": lgb_data[H][0]}
            all_mets = {"LightGBM": lgb_data[H][1]}
            for arch, r in nn_data.get(H, {}).items():
                all_preds[arch] = r["probs"]
                all_mets[arch] = r["metrics"]

            sel, hist = greedy_select(all_preds, y_va)
            print(f"\n  ── Greedy H={H}s (sl={get_sl(H)}, max {MAX_ENS}) ──")
            for h in hist:
                print(f"    Step {h['step']}: +{h['added']:18s} → "
                      f"AUC={h['auc_macro']:.4f} | Dir={h['conf_dir_acc']:.1f}% | "
                      f"IC={h['ic']:+.4f}")

            results.append({
                "H": H, "K": K_STAR[H],
                "L": max(45, int(H * K_STAR[H])),
                "seq_len": get_sl(H),
                "individual": {n: {"auc_macro": round(float(m["auc_macro"]), 5),
                                    "conf_dir_acc": round(float(m["conf_dir_acc"]), 2),
                                    "ic": round(float(m["ic"]), 5)}
                               for n, m in all_mets.items()},
                "greedy_history": hist,
                "selected_ensemble": sel,
            })

        # Save partial
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 Saved ({len(results)}/{len(HORIZONS)})", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    # Final
    results.sort(key=lambda r: r["H"])
    out_path = os.path.join(OUT, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  ИТОГО — v4 (seq_len + 20 epochs)")
    print("=" * 70)
    total = 0
    for r in results:
        e = r["selected_ensemble"]
        h = r["greedy_history"][-1] if r["greedy_history"] else {}
        total += len(e)
        print(f"  H={r['H']:3d}s (sl={r['seq_len']:2d}): "
              f"{' + '.join(e):55s} | AUC={h.get('auc_macro',0):.4f} | "
              f"Dir={h.get('conf_dir_acc',0):.1f}% | IC={h.get('ic',0):+.4f}")
    print(f"\n  Моделей: {total}")
    print(f"  Время: {time.time()-t_all:.0f}s ({(time.time()-t_all)/60:.1f} мин)")
    print(f"  Файл: {out_path}")


if __name__ == "__main__":
    main()
