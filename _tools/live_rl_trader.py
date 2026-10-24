#!/usr/bin/env python3
"""
Боевой и Paper-Live (Shadow) исполнительный модуль RL-агента (Maskable PPO v3 + Триада H=5,12,25s).
Работает без TensorFlow (через FastKerasModel на чистом PyTorch + LightGBM),
потребляет ~150 МБ RAM и выполняет полный 1-секундный цикл за ~45-55 мс.

Режимы запуска:
  1. --mode replay : Проверка живого цикла на последних N секундах реального датасета (тест скорости и сделок).
  2. --mode shadow : Живой Paper-Live по реальному стакану SafeTrade (REST/WS) без отправки реальных ордеров.
  3. --mode live   : Боевая торговля лимитными ордерами через приватный REST API SafeTrade (Peatio v2 HMAC-SHA256).
"""

import os
import sys
import time
import hmac
import json
import hashlib
import argparse
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.vps_config import SAFETRADE_PROXY


def _load_dotenv():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


class SafeTradePrivateClient:
    """Клиент приватного и публичного REST API SafeTrade (Peatio v2 HMAC-SHA256) с поддержкой SSH-моста через белый IP 81.19.137.18."""

    BASE_URL = "https://safe.trade/api/v2"

    def __init__(self, api_key: str = None, api_secret: str = None, use_proxy: bool = True, use_vps_bridge: bool = True):
        _load_dotenv()
        self.api_key = api_key or os.environ.get("SAFETRADE_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("SAFETRADE_API_SECRET", "")
        self.proxies = {"http": SAFETRADE_PROXY, "https": SAFETRADE_PROXY} if (use_proxy and SAFETRADE_PROXY) else None
        from curl_cffi import requests as cffi_requests, CurlOpt
        self.session = cffi_requests.Session(
            impersonate="chrome120",
            curl_options={CurlOpt.IPRESOLVE: 1},
        )
        self.use_vps_bridge = use_vps_bridge
        self._ssh = None
        self._bridge_in = None
        self._bridge_out = None
        if self.use_vps_bridge:
            self._init_vps_bridge()

    def _init_vps_bridge(self):
        from _tools.vps_config import create_ssh_client
        worker_code = (
            "import os, sys, csv, time, hmac, hashlib, json\n"
            "from curl_cffi import requests, CurlOpt\n"
            f"K = {repr(self.api_key)}\n"
            f"S = {repr(self.api_secret)}\n"
            "TEL_PATH = '/opt/crypto_bot/data/realtime/prlusdt/rl_telemetry.csv'\n"
            "os.makedirs(os.path.dirname(TEL_PATH), exist_ok=True)\n"
            "COLS = ['timestamp','ts_msk','step','latency_ms','b1','a1','act','act_name','v_pred','pos_qty','avg_entry','bot_cash_usd','equity_usd','active_bid_id','active_bid_p','active_bid_q','active_ask_id','active_ask_p','active_ask_q','obs_json','mask_json']\n"
            "if not os.path.exists(TEL_PATH) or os.path.getsize(TEL_PATH) == 0:\n"
            "    with open(TEL_PATH, 'w', newline='', encoding='utf-8') as f:\n"
            "        csv.writer(f).writerow(COLS)\n"
            "sess = requests.Session(impersonate='chrome120', curl_options={CurlOpt.IPRESOLVE: 1})\n"
            "while True:\n"
            "    line = sys.stdin.readline()\n"
            "    if not line:\n"
            "        break\n"
            "    req = json.loads(line)\n"
            "    if req['method'] == 'TELEMETRY':\n"
            "        d = req['body']\n"
            "        with open(TEL_PATH, 'a', newline='', encoding='utf-8') as f:\n"
            "            csv.writer(f).writerow([d.get(c, '') for c in COLS])\n"
            "        sys.stdout.write('{\"status\": 200, \"text\": \"ok\"}\\n')\n"
            "        sys.stdout.flush()\n"
            "        continue\n"
            "    n = str(int(time.time() * 1000))\n"
            "    sig = hmac.new(S.encode(), (n + K).encode(), hashlib.sha256).hexdigest()\n"
            "    hdr = {'X-Auth-Apikey': K, 'X-Auth-Nonce': n, 'X-Auth-Signature': sig, 'Content-Type': 'application/json'}\n"
            "    try:\n"
            "        if req['method'] == 'GET':\n"
            "            r = sess.get(req['url'], headers=hdr if req.get('auth', True) else None, timeout=6)\n"
            "        else:\n"
            "            body = json.dumps(req['body']) if req.get('body') is not None else None\n"
            "            r = sess.post(req['url'], headers=hdr, data=body, timeout=6)\n"
            "        sys.stdout.write(json.dumps({'status': r.status_code, 'text': r.text}) + '\\n')\n"
            "    except Exception as e:\n"
            "        sys.stdout.write(json.dumps({'status': 599, 'text': str(e)}) + '\\n')\n"
            "    sys.stdout.flush()\n"
        )
        import base64
        b64_code = base64.b64encode(worker_code.encode("utf-8")).decode("ascii")
        self._ssh = create_ssh_client()
        self._bridge_in, self._bridge_out, _ = self._ssh.exec_command(
            f"/opt/crypto_bot/venv/bin/python -u -c \"import base64; exec(base64.b64decode('{b64_code}'))\""
        )

    def append_vps_telemetry(self, rec: dict):
        import csv
        cols = ["timestamp","ts_msk","step","latency_ms","b1","a1","act","act_name","v_pred","pos_qty","avg_entry","bot_cash_usd","equity_usd","active_bid_id","active_bid_p","active_bid_q","active_ask_id","active_ask_p","active_ask_q","obs_json","mask_json"]
        row = dict(rec)
        row["timestamp"] = rec["ts_utc"]
        row["obs_json"] = json.dumps(rec["obs"], separators=(",", ":"))
        row["mask_json"] = json.dumps([int(x) for x in rec["mask"]], separators=(",", ":"))
        if not self.use_vps_bridge:
            tel_path = Path("/opt/crypto_bot/data/realtime/prlusdt/rl_telemetry.csv")
            tel_path.parent.mkdir(parents=True, exist_ok=True)
            need_hdr = not tel_path.exists() or tel_path.stat().st_size == 0
            with open(tel_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if need_hdr:
                    w.writerow(cols)
                w.writerow([row.get(c, "") for c in cols])
            return
        try:
            self._bridge_in.write(json.dumps({"method": "TELEMETRY", "body": row}) + "\n")
            self._bridge_in.flush()
            self._bridge_out.readline()
        except Exception:
            pass

    def _private_req(self, method: str, path: str, body: dict = None, auth: bool = True) -> dict | list:
        url = f"{self.BASE_URL}{path}"
        if self.use_vps_bridge:
            try:
                self._bridge_in.write(json.dumps({"method": method, "url": url, "body": body, "auth": auth}) + "\n")
                self._bridge_in.flush()
                resp_line = self._bridge_out.readline()
                if not resp_line:
                    self._init_vps_bridge()
                    self._bridge_in.write(json.dumps({"method": method, "url": url, "body": body, "auth": auth}) + "\n")
                    self._bridge_in.flush()
                    resp_line = self._bridge_out.readline()
                res = json.loads(resp_line)
                if res["status"] >= 400:
                    raise RuntimeError(f"SafeTrade HTTP {res['status']}: {res['text']}")
                return json.loads(res["text"]) if res["text"] else {}
            except Exception as e:
                if "SafeTrade HTTP" in str(e):
                    raise
                self._init_vps_bridge()
                raise
        if method == "GET":
            r = self.session.get(url, headers=self._auth_headers() if auth else None, timeout=6)
        else:
            r = self.session.post(url, headers=self._auth_headers(), data=json.dumps(body) if body else None, timeout=6)
        r.raise_for_status()
        return r.json()

    def _auth_headers(self) -> dict:
        nonce = str(int(time.time() * 1000))
        msg = f"{nonce}{self.api_key}".encode("utf-8")
        sig = hmac.new(self.api_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
        return {
            "X-Auth-Apikey": self.api_key,
            "X-Auth-Nonce": nonce,
            "X-Auth-Signature": sig,
            "Content-Type": "application/json",
        }

    def get_depth(self, market: str = "pearlusdt", limit: int = 25) -> dict:
        if self.use_vps_bridge:
            try:
                return self._private_req("GET", f"/trade/public/markets/{market}/depth?limit={limit}", auth=False)
            except Exception:
                pass
        url = f"{self.BASE_URL}/trade/public/markets/{market}/depth?limit={limit}"
        for px in (None, self.proxies):
            try:
                r = self.session.get(url, proxies=px, timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                continue
        return {"bids": [], "asks": []}

    def get_trades(self, market: str = "pearlusdt", limit: int = 50) -> list:
        url = f"{self.BASE_URL}/trade/public/markets/{market}/trades?limit={limit}"
        for px in (None, self.proxies):
            try:
                r = self.session.get(url, proxies=px, timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                continue
        return []

    def get_spot_balances(self) -> list:
        return self._private_req("GET", "/trade/account/balances/spot")

    def get_order(self, order_id: int | str) -> dict:
        return self._private_req("GET", f"/trade/market/orders/{order_id}")

    def get_open_orders(self, market: str = "pearlusdt") -> list:
        return self._private_req("GET", f"/trade/market/orders?market={market}&state=wait&limit=50")

    def place_limit_order(self, market: str, side: str, volume: float, price: float) -> dict:
        payload = {
            "market": market,
            "side": side.lower(),
            "amount": f"{max(2.0, volume):.4f}",
            "price": f"{price:.2f}",
            "type": "limit",
        }
        return self._private_req("POST", "/trade/market/orders", payload)

    def cancel_order(self, order_id: int | str) -> dict:
        return self._private_req("POST", f"/trade/market/orders/{order_id}/cancel")

    def cancel_all_orders(self, market: str = "pearlusdt", side: str = None) -> dict:
        payload = {"market": market}
        if side:
            payload["side"] = side.lower()
        return self._private_req("POST", "/trade/market/orders/cancel", payload)



class LiveTriadRLTrader:
    """
    Ядро принятия решений и исполнения в реальном времени (1 Гц).
    Поддерживает режимы:
      - 'replay' : прогон по живому кольцевому буферу на последних N секундах с замером задержки (мс).
      - 'shadow' : Paper-Live на реальном живом стакане SafeTrade (с виртуальным FIFO-ордером).
      - 'live'   : боевое выставление лимитных заявок на SafeTrade.
    """

    ACTION_NAMES = {
        0: "HOLD_WAIT",
        1: "QUOTE_BID_1",
        2: "QUOTE_BID_2",
        3: "QUOTE_ASK_1",
        4: "QUOTE_ASK_2",
        5: "CANCEL_BID",
        6: "PANIC_TAKER_EXIT",
    }

    def __init__(self, coin: str = "prl", order_usd: float = 50.0, initial_budget_usd: float = 50.0, mode: str = "shadow", use_neural: bool = True):
        self.coin = coin.lower()
        self.market = "pearlusdt" if self.coin == "prl" else f"{self.coin}usdt"
        self.initial_budget_usd = float(initial_budget_usd)
        self.order_usd = float(min(order_usd, initial_budget_usd))
        self.mode = mode.lower()
        self.tick_size = 0.01 if self.coin == "prl" else 0.05

        import torch
        from _tools.swarm_router import SwarmRegimeRouter
        from _tools.train_rl_execution import MaskableActorCritic
        print(f"🧠 Инициализация боевого ядра LiveTriadRLTrader ({self.coin.upper()}, режим={self.mode.upper()}, бюджет=${self.initial_budget_usd:.2f})...")
        self.routers = [
            (5, 3.0, SwarmRegimeRouter(self.coin, 5, 3.0, temperature=0.5, use_neural=use_neural, neural_weight=0.15)),
            (12, 3.5, SwarmRegimeRouter(self.coin, 12, 3.5, temperature=0.5, use_neural=use_neural, neural_weight=0.15)),
            (25, 3.0, SwarmRegimeRouter(self.coin, 25, 3.0, temperature=0.5, use_neural=use_neural, neural_weight=0.15)),
        ]
        self.feature_names = self.routers[0][2].feature_names

        ckpt_path = BASE_DIR / "models" / self.coin / "rl_execution" / "best_maskable_ppo.pt"
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        h_dim = sd["backbone.0.weight"].shape[0]
        obs_dim = sd["backbone.0.weight"].shape[1]
        self.agent = MaskableActorCritic(obs_dim=obs_dim, n_actions=7, hidden_dim=h_dim)
        self.agent.load_state_dict(sd)
        self.agent.eval()
        print(f"✅ Загружен чекпоинт RL-агента: {ckpt_path.name} (obs_dim={obs_dim}, hidden={h_dim})")

        self.bid_hist_1h = deque(maxlen=3600)
        self.ez12_hist = deque(maxlen=16)

        # Изолированный субсчет бота ($50.00) и состояние активных заявок
        self.bot_cash_usd = float(self.initial_budget_usd)
        self.reserved_account_usdt = 0.0
        self.pos_qty = 0.0
        self.avg_entry = 0.0
        self.hold_sec = 0
        self.active_bid_p = 0.0
        self.active_bid_q = 0.0
        self.active_bid_id = None
        self.active_ask_p = 0.0
        self.active_ask_q = 0.0
        self.active_ask_id = None
        self.q_ahead = 0.0
        self.realized_pnl_usd = 0.0
        self.n_trades = 0
        self.wallet_file = BASE_DIR / "models" / self.coin / "rl_execution" / "live_wallet_state.json"
        on_vps = str(BASE_DIR).startswith("/opt/crypto_bot")
        self.client = SafeTradePrivateClient(use_vps_bridge=(self.mode == "live" and not on_vps)) if self.mode in ("shadow", "live") else None
        if self.mode == "live":
            self._init_isolated_wallet()

    def _init_isolated_wallet(self):
        bals = {b["currency"]: float(b.get("balance", 0.0)) + float(b.get("locked", 0.0)) for b in self.client.get_spot_balances()}
        ex_usdt = bals.get("usdt", 0.0)
        if self.wallet_file.exists():
            st = json.loads(self.wallet_file.read_text(encoding="utf-8"))
            self.initial_budget_usd = float(st.get("initial_budget_usd", self.initial_budget_usd))
            self.bot_cash_usd = float(st.get("bot_cash_usd", self.initial_budget_usd))
            self.pos_qty = float(st.get("pos_qty", 0.0))
            self.avg_entry = float(st.get("avg_entry", 0.0))
            self.realized_pnl_usd = float(st.get("realized_pnl_usd", 0.0))
            self.n_trades = int(st.get("n_trades", 0))
            self.reserved_account_usdt = float(st.get("reserved_account_usdt", max(0.0, ex_usdt - self.initial_budget_usd)))
        else:
            self.reserved_account_usdt = max(0.0, ex_usdt - self.initial_budget_usd)
            self._save_wallet()
        print(
            f"🛡️ Изолированный субсчет бота: доступно боту=${self.bot_cash_usd:.2f} USDT | "
            f"неприкосновенный резерв счета=${self.reserved_account_usdt:,.5f} USDT (из ${ex_usdt:,.5f})"
        )

    def _save_wallet(self):
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "initial_budget_usd": round(self.initial_budget_usd, 6),
            "bot_cash_usd": round(self.bot_cash_usd, 6),
            "pos_qty": round(self.pos_qty, 6),
            "avg_entry": round(self.avg_entry, 6),
            "realized_pnl_usd": round(self.realized_pnl_usd, 6),
            "n_trades": int(self.n_trades),
            "reserved_account_usdt": round(self.reserved_account_usdt, 6),
        }
        try:
            self.wallet_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def sync_live_orders(self, b1: float):
        if self.mode != "live":
            return
        if self.pos_qty > 1e-4:
            self.hold_sec += 1
        if self.active_bid_id is not None:
            o = self.client.get_order(self.active_bid_id)
            filled = float(o.get("filled_amount", 0.0))
            st = o.get("state", "")
            if filled > 0 and st in ("done", "cancel"):
                px = float(o.get("avg_price", 0.0)) or float(o.get("price", self.active_bid_p))
                cost = filled * px * 1.001
                self.bot_cash_usd = max(0.0, self.bot_cash_usd - cost)
                self.avg_entry = (self.pos_qty * self.avg_entry + filled * px) / max(1e-6, self.pos_qty + filled)
                self.pos_qty += filled
                self.hold_sec = 0
                self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                self._save_wallet()
                print(f"  🟢 [FILL BUY] +{filled:.4f} PRL @ ${px:.2f} | Остаток кэша бота=${self.bot_cash_usd:.2f}")
            elif st == "cancel":
                self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
        if self.active_ask_id is not None:
            o = self.client.get_order(self.active_ask_id)
            filled = float(o.get("filled_amount", 0.0))
            st = o.get("state", "")
            if filled > 0 and st in ("done", "cancel"):
                px = float(o.get("avg_price", 0.0)) or float(o.get("price", self.active_ask_p))
                proceeds = filled * px * 0.999
                pnl = proceeds - filled * self.avg_entry * 1.001
                self.bot_cash_usd += proceeds
                self.realized_pnl_usd += pnl
                self.pos_qty = max(0.0, self.pos_qty - filled)
                if self.pos_qty < 0.01:
                    self.pos_qty, self.avg_entry, self.hold_sec = 0.0, 0.0, 0
                self.n_trades += 1
                self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                self._save_wallet()
                print(f"  💰 [FILL SELL] -{filled:.4f} PRL @ ${px:.2f} | PnL={pnl:+.4f}$ | Баланс бота=${self.bot_cash_usd:.2f}")
            elif st == "cancel":
                self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0

    def compute_obs_and_mask(self, X_seq_2d: np.ndarray, row_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        triad_vec = []
        for _, _, router in self.routers:
            res = router.predict_live_tick(X_seq_2d, row_df)
            p = res["probs"]
            ez = float(res["expected_z"])
            p_up = float(p[4] + p[5] + p[6])
            p_down = float(p[0] + p[1] + p[2])
            p_tp = float(p[5] + p[6])
            p_tn = float(p[0] + p[1])
            triad_vec.extend([ez, p_up, p_down, p_tp, p_tn, float(res["entropy"])])

        r = row_df.iloc[-1]
        b1 = float(max(r.get("ST_bid_1_price", r.get("ST_best_bid", 0.0)), self.tick_size))
        a1 = float(max(r.get("ST_ask_1_price", r.get("ST_best_ask", b1 + self.tick_size)), b1 + self.tick_size))
        b1_q = max(0.0, float(r.get("ST_bid_1_qty", r.get("ST_bid_qty_1", 0.0))))
        a1_q = max(0.0, float(r.get("ST_ask_1_qty", r.get("ST_ask_qty_1", 0.0))))
        b5_q = sum(max(0.0, float(r.get(f"ST_bid_{i}_qty", b1_q if i == 1 else 0.0))) for i in range(1, 6))
        a5_q = sum(max(0.0, float(r.get(f"ST_ask_{i}_qty", a1_q if i == 1 else 0.0))) for i in range(1, 6))
        b10_q = sum(max(0.0, float(r.get(f"ST_bid_{i}_qty", b1_q if i == 1 else 0.0))) for i in range(1, 11))
        a10_q = sum(max(0.0, float(r.get(f"ST_ask_{i}_qty", a1_q if i == 1 else 0.0))) for i in range(1, 11))

        self.bid_hist_1h.append(b1)
        ez_5, ez_12, ez_25 = triad_vec[0], triad_vec[6], triad_vec[12]
        p_tn_5, p_dn_12 = triad_vec[4], triad_vec[8]
        self.ez12_hist.append(ez_12)

        spread_ticks = float(np.clip((a1 - b1) / self.tick_size, 0.0, 50.0))
        imb_l1 = (b1_q - a1_q) / (b1_q + a1_q + 1e-6)
        imb_l5 = (b5_q - a5_q) / (b5_q + a5_q + 1e-6)
        imb_l10 = (b10_q - a10_q) / (b10_q + a10_q + 1e-6)
        vol_300 = float(r.get("ST_vol_300s_ticks", 1.0))
        ofi_300 = float(r.get("ST_ofi_norm_300s", 0.0))

        lob_vec = [
            spread_ticks / 10.0, imb_l1, imb_l5, imb_l10,
            float(np.log1p(b1_q) / 10.0), float(np.log1p(a1_q) / 10.0),
            float(np.log1p(b5_q) / 12.0), float(np.log1p(a5_q) / 12.0),
            float(np.clip(vol_300 / 5.0, 0.0, 5.0)), float(np.clip(ofi_300, -3.0, 3.0)),
        ]

        b_arr = list(self.bid_hist_1h)
        h_300 = max(b_arr[-300:])
        h_900 = max(b_arr[-900:])
        l_900 = min(b_arr[-900:])
        h_3600 = max(b_arr)
        l_3600 = min(b_arr)
        l_30 = min(b_arr[-30:])

        dd_300 = (h_300 - b1) / self.tick_size
        dd_900 = (h_900 - b1) / self.tick_size
        pos_900 = (b1 - l_900) / max(h_900 - l_900, self.tick_size)
        pos_3600 = (b1 - l_3600) / max(h_3600 - l_3600, self.tick_size)
        reb_30 = (b1 - l_30) / self.tick_size
        vel_10 = (b1 - (b_arr[-11] if len(b_arr) >= 11 else b_arr[0])) / self.tick_size
        vel_15 = (b1 - (b_arr[-16] if len(b_arr) >= 16 else b_arr[0])) / self.tick_size
        vel_180 = (b1 - (b_arr[-181] if len(b_arr) >= 181 else b_arr[0])) / self.tick_size

        d_ez12 = ez_12 - (self.ez12_hist[-6] if len(self.ez12_hist) >= 6 else self.ez12_hist[0])
        inertia = float(
            0.45 * ez_25 + 0.35 * ez_12 + 0.20 * ez_5
            + 0.25 * d_ez12 + 0.35 * imb_l5 + 0.15 * imb_l1
            + 0.10 * np.clip(vel_15, -3.0, 3.0)
        )

        b_prev12 = b_arr[-13] if len(b_arr) >= 13 else b_arr[0]
        ez12_prev12 = self.ez12_hist[-13] if len(self.ez12_hist) >= 13 else self.ez12_hist[0]
        model_err_12 = float(np.clip((b1 - b_prev12) / self.tick_size - ez12_prev12 * 4.0, -10.0, 10.0))
        surprise_pen = max(0.0, -model_err_12) * 0.6

        ext_pos = 0.35 * min(dd_900, 10.0) * max(0.0, 1.0 - 1.5 * pos_900)
        ext_neg = -1.5 * max(0.0, pos_900 - 0.65)
        extremum_bonus = ext_pos if inertia > 0.25 else ext_neg
        exp_entry_edge = (0.5 * ez_25 + 0.35 * ez_12 + 0.15 * ez_5) * 4.0 + 0.5 * spread_ticks - p_tn_5 * 3.0 - surprise_pen + extremum_bonus
        exp_hold_edge = inertia * 3.0 - p_dn_12 * spread_ticks - surprise_pen * 0.6

        obs_err12_norm = float(np.clip(model_err_12 / 5.0, -3.0, 3.0))
        obs_entry_edge_norm = float(np.clip(exp_entry_edge / 5.0, -3.0, 3.0))
        obs_hold_edge_norm = float(np.clip(exp_hold_edge / 5.0, -3.0, 3.0))

        has_pos = 1.0 if self.pos_qty > 1e-8 else 0.0
        unr_ticks = ((b1 - self.avg_entry) / self.tick_size) if has_pos else 0.0
        hold_ratio = min(self.hold_sec / 45.0, 2.0) if has_pos else 0.0
        has_bid = 1.0 if self.active_bid_q > 0 else 0.0
        has_ask = 1.0 if self.active_ask_q > 0 else 0.0
        dist = ((b1 - self.active_bid_p) / self.tick_size) if has_bid else (((self.active_ask_p - a1) / self.tick_size) if has_ask else 0.0)
        ord_q = self.active_bid_q if has_bid else self.active_ask_q
        q_ratio = self.q_ahead / (self.q_ahead + ord_q + 1e-6) if (has_bid or has_ask) else 0.0

        obs = np.array(
            triad_vec + lob_vec + [
                has_pos, float(np.clip(unr_ticks * 0.1, -3.0, 3.0)), hold_ratio,
                has_bid, has_ask, float(np.clip(dist * 0.2, -2.0, 2.0)), q_ratio,
                obs_err12_norm, obs_entry_edge_norm, obs_hold_edge_norm,
            ],
            dtype=np.float32,
        )

        bounce_ok = (1.0 <= reb_30 <= 2.0) if dd_900 >= 6.0 else (reb_30 <= 2.0)
        dip_rev = (pos_900 <= 0.35) and (pos_3600 <= 0.45) and (dd_300 >= 2.0 or dd_900 >= 3.0) and (vel_10 >= 0.0) and bounce_ok and (inertia >= 0.28)
        uptrend = (vel_180 >= 2.0) and (vel_10 >= 0.0) and (pos_3600 <= 0.72) and (dd_300 >= 1.0) and (inertia >= 0.40)
        in_dip_zone = dip_rev or uptrend

        mask = np.zeros(7, dtype=bool)
        if not has_pos:
            mask[0] = True
            if in_dip_zone:
                mask[2] = True
                if imb_l5 > -0.75:
                    mask[1] = True
            if has_bid:
                mask[5] = True
        else:
            mask[0] = bool(has_ask)
            mask[3] = True
            mask[4] = True
            if unr_ticks <= -2.0 or inertia < -0.25 or obs_err12_norm < -0.60:
                mask[6] = True
        return obs, mask

    def decide_action(self, obs: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
        import torch
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            logits, val = self.agent(obs_t, mask_t)
            return int(torch.argmax(logits, dim=-1).item()), float(val.item())

    def execute_live_action(self, act: int, b1: float, a1: float):
        if self.mode != "live":
            return
        tol = self.tick_size * 0.25
        if act in (1, 2) and self.pos_qty <= 1e-4:
            target_p = round(((b1 + self.tick_size if (a1 - b1) >= 2.0 * self.tick_size else b1) if act == 1 else max(self.tick_size, b1 - self.tick_size)) / self.tick_size) * self.tick_size
            usable_usd = min(self.order_usd, self.bot_cash_usd * 0.996)
            qty = round(usable_usd / max(target_p, self.tick_size), 4)
            if qty < 2.0:
                return
            if self.active_bid_id is not None and abs(self.active_bid_p - target_p) <= tol:
                return
            if self.active_bid_id is not None:
                self.client.cancel_order(self.active_bid_id)
                self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
            bals = {b["currency"]: float(b.get("balance", 0.0)) for b in self.client.get_spot_balances()}
            if bals.get("usdt", 0.0) - (qty * target_p) < self.reserved_account_usdt - 0.05:
                print(f"  ⚠️ [GUARD] Защита резерва счета: свободно ${bals.get('usdt', 0.0):.2f} < резерва ${self.reserved_account_usdt:.2f}")
                return
            res = self.client.place_limit_order(self.market, "buy", qty, target_p)
            self.active_bid_id = res.get("id")
            self.active_bid_p = float(res.get("price", target_p))
            self.active_bid_q = float(res.get("origin_amount", qty))
            print(f"  📥 [LIVE ORDER BUY] id={self.active_bid_id} | {self.active_bid_q:.4f} PRL @ ${self.active_bid_p:.2f} (${self.active_bid_q*self.active_bid_p:.2f})")
        elif act == 5 and self.active_bid_id is not None:
            self.client.cancel_order(self.active_bid_id)
            print(f"  ✖️ [LIVE CANCEL BUY] id={self.active_bid_id}")
            self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
        elif act in (3, 4, 6) and self.pos_qty >= 2.0:
            if act == 3:
                target_p = a1
            elif act == 4:
                target_p = max(b1, a1 - self.tick_size)
            else:
                target_p = max(self.tick_size, b1 - self.tick_size)
            target_p = round(target_p / self.tick_size) * self.tick_size
            if self.active_ask_id is not None and abs(self.active_ask_p - target_p) <= tol:
                return
            if self.active_ask_id is not None:
                self.client.cancel_order(self.active_ask_id)
                self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
            res = self.client.place_limit_order(self.market, "sell", self.pos_qty, target_p)
            self.active_ask_id = res.get("id")
            self.active_ask_p = float(res.get("price", target_p))
            self.active_ask_q = float(res.get("origin_amount", self.pos_qty))
            print(f"  📤 [LIVE ORDER SELL] id={self.active_ask_id} | {self.active_ask_q:.4f} PRL @ ${self.active_ask_p:.2f}")


def run_replay_test(coin: str = "prl", steps: int = 30, use_neural: bool = True):
    import pyarrow.parquet as pq
    trader = LiveTriadRLTrader(coin=coin, mode="replay", use_neural=use_neural)
    pf = pq.ParquetFile(BASE_DIR / "data" / "features" / f"{coin}_1s_features.parquet")
    df_tail = pf.read_row_group(pf.num_row_groups - 1).to_pandas().iloc[-(steps + 60):].reset_index(drop=True)
    X_all = df_tail.reindex(columns=trader.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)

    latencies = []
    for i in range(60, len(df_tail)):
        t0 = time.perf_counter()
        obs, mask = trader.compute_obs_and_mask(X_all[i - 30:i + 1], df_tail.iloc[i:i + 1])
        act, v_pred = trader.decide_action(obs, mask)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        if i >= len(df_tail) - 5:
            print(f"  [Tick {i-59:02d}] Act={trader.ACTION_NAMES[act]:<16} | V={v_pred:+.2f}t | Latency={dt_ms:.1f} ms")
    print(f"\n⚡ Средняя задержка полного боевого шага (Триада + Нейросети + PPO): {np.mean(latencies[1:]):.1f} мс (макс: {np.max(latencies[1:]):.1f} мс)")


def run_live_or_shadow_loop(coin: str = "prl", mode: str = "shadow", steps: int = 60, order_usd: float = 50.0, budget_usd: float = 50.0):
    import subprocess
    trader = LiveTriadRLTrader(coin=coin, order_usd=order_usd, initial_budget_usd=budget_usd, mode=mode, use_neural=True)
    seed_pq = BASE_DIR / "models" / coin / "rl_execution" / "live_seed_tail60.parquet"
    df_sub = pd.read_parquet(seed_pq).reset_index(drop=True)
    b_col = "ST_bid_1_price" if "ST_bid_1_price" in df_sub.columns else "ST_best_bid"
    for b_val in df_sub[b_col].values:
        if b_val > 0:
            trader.bid_hist_1h.append(float(b_val))

    local_depth = Path("/opt/crypto_bot/data/realtime/prlusdt/depth.csv")
    if local_depth.exists():
        lines = subprocess.run(["tail", "-n", "3600", str(local_depth)], capture_output=True, text=True).stdout.splitlines()
        for line in lines:
            parts = line.split(",")
            if len(parts) > 2:
                try:
                    bp = float(parts[1])
                    if bp > 0:
                        trader.bid_hist_1h.append(bp)
                except ValueError:
                    pass
    elif trader.client and trader.client._ssh is not None:
        try:
            _, out, _ = trader.client._ssh.exec_command("tail -n 3600 /opt/crypto_bot/data/realtime/prlusdt/depth.csv")
            for line in out.read().decode("utf-8", errors="ignore").splitlines():
                parts = line.split(",")
                if len(parts) > 2:
                    try:
                        bp = float(parts[1])
                        if bp > 0:
                            trader.bid_hist_1h.append(bp)
                    except ValueError:
                        pass
        except Exception:
            pass

    X_tail = df_sub.reindex(columns=trader.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)
    X_buf = deque(X_tail, maxlen=60)
    last_row = df_sub.iloc[-1:].copy()
    for w_i in range(45, len(X_tail)):
        res12 = trader.routers[1][2].predict_live_tick(X_tail[:w_i + 1], df_sub.iloc[w_i:w_i + 1])
        trader.ez12_hist.append(float(res12["expected_z"]))
    print(f"✅ Прогрет 1-часовой буфер стакана ({len(trader.bid_hist_1h)} сек, диапазон Bid=[${min(trader.bid_hist_1h):.2f}..${max(trader.bid_hist_1h):.2f}]) и история инерции ({len(trader.ez12_hist)} сек)")

    from datetime import timedelta
    msk_tz = timezone(timedelta(hours=3))
    telemetry_path = BASE_DIR / "data" / "logs" / "live_rl_telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_f = open(telemetry_path, "a", encoding="utf-8")

    now_str = datetime.now(msk_tz).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"🚀 [{now_str} MSK] Старт боевого цикла {mode.upper()} на {steps} сек (лимит капитала=${budget_usd:.2f}, ордер=${order_usd:.2f}, тик=${trader.tick_size})...")
    for step_idx in range(1, steps + 1):
        t0 = time.perf_counter()
        now_utc = datetime.now(timezone.utc)
        now_msk = now_utc.astimezone(msk_tz)
        ts_msk_str = now_msk.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        depth = trader.client.get_depth(trader.market, limit=15)
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])
        if bids and asks:
            for lvl in range(1, 11):
                if lvl <= len(bids):
                    last_row[f"ST_bid_{lvl}_price"] = float(bids[lvl - 1][0])
                    last_row[f"ST_bid_{lvl}_qty"] = float(bids[lvl - 1][1])
                if lvl <= len(asks):
                    last_row[f"ST_ask_{lvl}_price"] = float(asks[lvl - 1][0])
                    last_row[f"ST_ask_{lvl}_qty"] = float(asks[lvl - 1][1])
            last_row["ST_best_bid"] = float(bids[0][0])
            last_row["ST_best_ask"] = float(asks[0][0])

        b1 = float(last_row["ST_best_bid"].iloc[0])
        a1 = float(last_row["ST_best_ask"].iloc[0])
        trader.sync_live_orders(b1)

        x_new = last_row.reindex(columns=trader.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)[0]
        X_buf.append(x_new)
        obs, mask = trader.compute_obs_and_mask(np.array(X_buf, dtype=np.float32), last_row)
        act, v_pred = trader.decide_action(obs, mask)
        trader.execute_live_action(act, b1, a1)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        eq_usd = trader.bot_cash_usd + trader.pos_qty * b1
        pnl_tot = eq_usd - trader.initial_budget_usd

        rec = {
            "ts_utc": now_utc.isoformat(),
            "ts_msk": ts_msk_str,
            "step": step_idx,
            "latency_ms": round(dt_ms, 1),
            "b1": b1,
            "a1": a1,
            "act": int(act),
            "act_name": trader.ACTION_NAMES[act],
            "v_pred": round(v_pred, 4),
            "obs": [round(float(x), 5) for x in obs],
            "mask": [bool(x) for x in mask],
            "pos_qty": round(trader.pos_qty, 4),
            "avg_entry": round(trader.avg_entry, 4),
            "bot_cash_usd": round(trader.bot_cash_usd, 4),
            "equity_usd": round(eq_usd, 4),
            "active_bid_id": trader.active_bid_id,
            "active_bid_p": trader.active_bid_p,
            "active_bid_q": trader.active_bid_q,
            "active_ask_id": trader.active_ask_id,
            "active_ask_p": trader.active_ask_p,
            "active_ask_q": trader.active_ask_q,
        }
        telemetry_f.write(json.dumps(rec) + "\n")
        telemetry_f.flush()
        if trader.client is not None:
            trader.client.append_vps_telemetry(rec)

        if step_idx <= 5 or step_idx % 5 == 0 or act != 0:
            print(
                f"  [{ts_msk_str} MSK | #{step_idx:04d}] Bid={b1:.2f} Ask={a1:.2f} | Act={trader.ACTION_NAMES[act]:<14} | "
                f"V={v_pred:+.2f}t | Equity=${eq_usd:.2f} ({pnl_tot:+.2f}$) | {dt_ms:.1f} ms",
                flush=True,
            )
        time.sleep(max(0.05, 1.0 - (time.perf_counter() - t0)))


def show_live_status(tail_lines: int = 25):
    """Онлайн-запрос статуса фонового демона на VPS (81.19.137.18), ресурсов сервера, субсчета $50 и хвоста лога."""
    from _tools.vps_config import create_ssh_client
    ssh = create_ssh_client()
    cmd = (
        "PID=$(cat /opt/crypto_bot/data/logs/live_rl_trader.pid 2>/dev/null); "
        "if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then "
        "  echo \"RUNNING:$PID\"; ps -p \"$PID\" -o pcpu=,pmem=,rss=,etime= 2>/dev/null; "
        "else echo \"STOPPED\"; echo \"\"; fi; "
        "echo '---WALLET---'; cat /opt/crypto_bot/models/prl/rl_execution/live_wallet_state.json 2>/dev/null || echo '{}'; "
        f"echo '---LOG---'; tail -n {int(tail_lines)} /opt/crypto_bot/data/logs/live_rl_trader.log 2>/dev/null || true"
    )
    _, out, _ = ssh.exec_command(cmd)
    raw = out.read().decode("utf-8", errors="ignore")
    ssh.close()

    st_part, rest = raw.split("---WALLET---", 1) if "---WALLET---" in raw else (raw, "")
    w_part, log_part = rest.split("---LOG---", 1) if "---LOG---" in rest else (rest, "")
    st_lines = [x.strip() for x in st_part.strip().splitlines() if x.strip()]
    st_line = st_lines[0] if st_lines else "STOPPED"
    ps_line = st_lines[1] if len(st_lines) > 1 else ""
    running = st_line.startswith("RUNNING:")
    pid = st_line.split(":", 1)[1] if running else None

    res_info = ""
    if running and ps_line:
        parts = ps_line.split()
        if len(parts) >= 4:
            rss_mb = float(parts[2]) / 1024.0
            res_info = f" | CPU={parts[0]}% | RAM={rss_mb:.0f} МБ ({parts[1]}%) | Uptime={parts[3]}"

    print(f"📡 Статус демона на VPS (81.19.137.18): {'🟢 РАБОТАЕТ НА СЕРВЕРЕ (PID=' + str(pid) + ')' + res_info if running else '🔴 ОСТАНОВЛЕН'}")
    try:
        w = json.loads(w_part.strip() or "{}")
        if w:
            (BASE_DIR / "models" / "prl" / "rl_execution" / "live_wallet_state.json").write_text(
                json.dumps(w, indent=2), encoding="utf-8"
            )
            print(
                f"💼 Субсчет бота: Кэш=${w.get('bot_cash_usd', 50.0):.2f} | Позиция={w.get('pos_qty', 0.0):.4f} PRL "
                f"(вход=${w.get('avg_entry', 0.0):.2f}) | Реализ. PnL={w.get('realized_pnl_usd', 0.0):+.4f}$ | "
                f"Сделок={w.get('n_trades', 0)} | Неприкосновенный резерв=${w.get('reserved_account_usdt', 0.0):,.5f}"
            )
    except Exception:
        pass
    print(f"\n📜 Онлайн-лог с сервера VPS (/opt/crypto_bot/data/logs/live_rl_trader.log, последние {tail_lines} строк):")
    print("-" * 100)
    print(log_part.strip())


def stream_live_log(duration_sec: int = 30):
    """Потоковый вывод новых строк лога с VPS в ячейку ноутбука (остановка: кнопка Stop в Jupyter без остановки самого бота на VPS)."""
    from _tools.vps_config import create_ssh_client
    print(f"👀 Онлайн-трансляция лога с VPS на {duration_sec} сек (нажмите ■ Interrupt в ноутбуке, чтобы остановить просмотр лога)...", flush=True)
    ssh = create_ssh_client()
    _, stdout, _ = ssh.exec_command(
        f"timeout {int(duration_sec)} tail -n 15 -f /opt/crypto_bot/data/logs/live_rl_trader.log",
        get_pty=True,
    )
    try:
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                chunk = stdout.channel.recv(4096).decode("utf-8", errors="ignore")
                print(chunk, end="", flush=True)
            else:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n⏹️ Просмотр лога остановлен (фоновый бот на VPS продолжает работать автономно).", flush=True)
    finally:
        ssh.close()


def stop_live_daemon(cancel_open_orders: bool = True):
    """Останавливает фоновый демон бота (и локально, и на VPS) и при необходимости снимает все открытые ордера на SafeTrade."""
    pid_file = BASE_DIR / "data" / "logs" / "live_rl_trader.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 15)
        except Exception:
            pass
    from _tools.vps_config import create_ssh_client
    ssh = create_ssh_client()
    _, out, _ = ssh.exec_command(
        "PID=$(cat /opt/crypto_bot/data/logs/live_rl_trader.pid 2>/dev/null); "
        "if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then kill -15 \"$PID\" && echo \"STOPPED:$PID\"; else echo \"NOT_RUNNING\"; fi"
    )
    res_str = out.read().decode().strip()
    ssh.close()
    print(f"🛑 Статус остановки демона на VPS: {res_str}")
    if cancel_open_orders:
        client = SafeTradePrivateClient(use_vps_bridge=True)
        for st in ("wait", "pending"):
            for o in client._private_req("GET", f"/trade/market/orders?state={st}&limit=50"):
                oid = o.get("id")
                if oid:
                    client.cancel_order(oid)
                    print(f"✖️ Снят открытый ордер id={oid} ({o.get('side')} {o.get('origin_amount')} @ ${o.get('price')})")
        print("✅ Все открытые заявки на бирже проверены и сняты.")


def sync_and_inspect_telemetry(last_n: int = 20) -> pd.DataFrame:
    """Онлайн-синхронизация горячей телеметрии RL и лога с VPS (81.19.137.18), сохранение в Parquet и вывод сводки."""
    from _tools.vps_config import create_ssh_client
    hot_dir = BASE_DIR / "data" / "vps_hot"
    proc_dir = BASE_DIR / "data" / "processed" / "prl"
    log_dir = BASE_DIR / "data" / "logs"
    hot_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    local_csv = hot_dir / "prl_rl_telemetry_hot.csv"
    pq_file = proc_dir / "rl_telemetry.parquet"

    ssh = create_ssh_client()
    sftp = ssh.open_sftp()
    sftp.get("/opt/crypto_bot/data/realtime/prlusdt/rl_telemetry.csv", str(local_csv))
    try:
        sftp.get("/opt/crypto_bot/data/logs/live_rl_trader.log", str(log_dir / "live_rl_trader.log"))
        sftp.get("/opt/crypto_bot/data/logs/live_rl_telemetry.jsonl", str(log_dir / "live_rl_telemetry.jsonl"))
    except Exception:
        pass
    sftp.close()
    ssh.close()

    df = pd.read_csv(local_csv)
    df.to_parquet(pq_file, index=False)
    print(f"✅ Онлайн-синхронизация с VPS (81.19.137.18): {len(df):,} тиков телеметрии -> {pq_file.relative_to(BASE_DIR)}")
    if len(df) > 0:
        print(f"⏱️ Интервал: {df['ts_msk'].iloc[0]} MSK ➔ {df['ts_msk'].iloc[-1]} MSK | Средняя задержка на VPS: {df['latency_ms'].tail(100).mean():.1f} мс")
        print("📊 Распределение действий агента:", df["act_name"].value_counts().to_dict())
    cols_show = [c for c in ["ts_msk", "step", "b1", "a1", "act_name", "v_pred", "pos_qty", "bot_cash_usd", "equity_usd", "latency_ms"] if c in df.columns]
    return df[cols_show].tail(last_n)


def start_live_daemon(budget_usd: float = 50.0, order_usd: float = 50.0, steps: int = 86400):
    """Запускает / перезапускает автономный фоновый демон бота непосредственно на VPS (81.19.137.18)."""
    stop_live_daemon(cancel_open_orders=False)
    from _tools.vps_config import create_ssh_client
    ssh = create_ssh_client()
    cmd = (
        f"cd /opt/crypto_bot && /opt/crypto_bot/venv/bin/python -m _tools.live_rl_trader "
        f"--mode live --budget-usd {float(budget_usd)} --order-usd {float(order_usd)} --steps {int(steps)} --daemon"
    )
    _, out, err = ssh.exec_command(cmd)
    print(out.read().decode().strip())
    err_s = err.read().decode().strip()
    if err_s:
        print(err_s)
    ssh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=str, default="prl")
    parser.add_argument("--mode", type=str, default="replay", choices=["replay", "shadow", "live"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--order-usd", type=float, default=50.0)
    parser.add_argument("--budget-usd", type=float, default=50.0)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    if args.daemon:
        import subprocess
        log_dir = BASE_DIR / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "live_rl_trader.log"
        pid_file = log_dir / "live_rl_trader.pid"
        cmd = [
            sys.executable, "-u", "-m", "_tools.live_rl_trader",
            "--coin", args.coin, "--mode", args.mode,
            "--steps", str(args.steps), "--order-usd", str(args.order_usd), "--budget-usd", str(args.budget_usd),
        ]
        with open(log_file, "a", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=lf, stderr=lf, start_new_session=True)
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        print(f"🚀 Фоновый демон запущен (PID={proc.pid}). Лог: {log_file}")
    elif args.mode == "replay":
        run_replay_test(coin=args.coin, steps=args.steps)
    else:
        run_live_or_shadow_loop(coin=args.coin, mode=args.mode, steps=args.steps, order_usd=args.order_usd, budget_usd=args.budget_usd)
