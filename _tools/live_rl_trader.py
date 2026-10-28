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
import math
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
        for attempt in range(2):
            if method == "GET":
                r = self.session.get(url, headers=self._auth_headers() if auth else None, timeout=6)
            else:
                r = self.session.post(url, headers=self._auth_headers(), data=json.dumps(body) if body else None, timeout=6)
            if r.status_code == 403 and attempt == 0:
                time.sleep(0.35)
                continue
            break
        if r.status_code >= 400:
            raise RuntimeError(f"SafeTrade HTTP {r.status_code}: {r.text}")
        return r.json() if r.text else {}

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
        if use_neural:
            for _, _, rtr in self.routers:
                rtr.preload_neural_models()
            print("⚡ Все 84 нейросетевые модели (FastKerasModel) предзагружены в RAM (0 обращений к диску на тиках).")
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

        self.bid_hist_1h = deque(maxlen=14400)
        self.ez12_hist = deque(maxlen=16)

        # Изолированный субсчет бота ($50.00) и динамический учет резерва пользователя
        self.base_currency = "pearl" if self.coin == "prl" else self.coin
        self.bot_cash_usd = float(self.initial_budget_usd)
        self.reserved_account_usdt = 0.0
        self.user_reserved_coin = 0.0
        self.ex_usdt_free = 0.0
        self.ex_coin_free = 0.0
        self.pos_qty = 0.0
        self.avg_entry = 0.0
        self.hold_sec = 0
        self.peak_bid = 0.0
        self.ez25_entry = 0.0
        self.last_ez25 = 0.0
        self.last_inertia = 0.0
        self.last_pos3600 = 0.0
        self.last_vel10 = 0.0
        self.last_imb5 = 0.0
        self.active_bid_p = 0.0
        self.active_bid_q = 0.0
        self.active_bid_id = None
        self.active_ask_p = 0.0
        self.active_ask_q = 0.0
        self.active_ask_id = None
        self.slot2_ask_p = 0.0
        self.slot2_ask_q = 0.0
        self.slot2_ask_id = None
        self.split_done_for_pos = False
        self.last_dd300 = 0.0
        self.last_dd900 = 0.0
        self.last_sp_ticks = 1.0
        self.last_ez12 = 0.0
        self.last_hold_edge = 0.0
        self.last_q_a1 = 0.0
        self.last_q_a2 = 0.0
        self.q_ahead = 0.0
        self.realized_pnl_usd = 0.0
        self.net_deposits_usd = 0.0
        self.deposits_log = []
        self.n_trades = 0
        self.wallet_file = BASE_DIR / "models" / self.coin / "rl_execution" / "live_wallet_state.json"
        on_vps = str(BASE_DIR).startswith("/opt/crypto_bot")
        self.client = SafeTradePrivateClient(use_vps_bridge=(self.mode == "live" and not on_vps)) if self.mode in ("shadow", "live") else None
        if self.mode == "live":
            self._init_isolated_wallet()

    def _init_isolated_wallet(self):
        if self.wallet_file.exists():
            st = json.loads(self.wallet_file.read_text(encoding="utf-8"))
            self.initial_budget_usd = float(st.get("initial_budget_usd", self.initial_budget_usd))
            self.bot_cash_usd = float(st.get("bot_cash_usd", self.initial_budget_usd))
            self.pos_qty = float(st.get("pos_qty", 0.0))
            self.avg_entry = float(st.get("avg_entry", 0.0))
            self.realized_pnl_usd = float(st.get("realized_pnl_usd", 0.0))
            self.net_deposits_usd = float(st.get("net_deposits_usd", 0.0))
            self.deposits_log = list(st.get("deposits_log", []))
            self.n_trades = int(st.get("n_trades", 0))
            self.reserved_account_usdt = float(st.get("reserved_account_usdt", 0.0))
            self.user_reserved_coin = float(st.get("user_reserved_coin", 0.0))
            self.active_bid_id = st.get("active_bid_id")
            self.active_bid_p = float(st.get("active_bid_p", 0.0))
            self.active_bid_q = float(st.get("active_bid_q", 0.0))
            self.active_ask_id = st.get("active_ask_id")
            self.active_ask_p = float(st.get("active_ask_p", 0.0))
            self.active_ask_q = float(st.get("active_ask_q", 0.0))
            self.slot2_ask_id = st.get("slot2_ask_id")
            self.slot2_ask_p = float(st.get("slot2_ask_p", 0.0))
            self.slot2_ask_q = float(st.get("slot2_ask_q", 0.0))
            self.split_done_for_pos = bool(st.get("split_done_for_pos", False))
            self.car_state = int(st.get("car_state", 0))
            self.car_sell_px = float(st.get("car_sell_px", 0.0))
            self.car_cycles = int(st.get("car_cycles", 0))
        self.bid_age_sec = 0
        self.bid_partial_filled = 0.0
        b1_init = 0.0
        try:
            d_init = self.client.get_depth(self.market, limit=5)
            bids_init = d_init.get("bids", [])
            if bids_init:
                b1_init = float(bids_init[0][0])
        except Exception:
            pass
        self.sync_live_orders(b1_init)
        self.sync_account_balances(b1_init, init_mode=True)
        print(
            f"🛡️ Изолированный субсчет бота: кэш=${self.bot_cash_usd:.2f} USDT, поз={self.pos_qty:.4f} {self.coin.upper()} "
            f"(база капитала=${self.initial_budget_usd:.2f}, пополнения=${self.net_deposits_usd:+.2f}) | "
            f"личный резерв пользователя=${self.reserved_account_usdt:,.2f} USDT + {self.user_reserved_coin:,.4f} {self.coin.upper()}"
        )

    def _save_wallet(self):
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "initial_budget_usd": round(self.initial_budget_usd, 6),
            "bot_cash_usd": round(self.bot_cash_usd, 6),
            "pos_qty": round(self.pos_qty, 6),
            "avg_entry": round(self.avg_entry, 6),
            "realized_pnl_usd": round(self.realized_pnl_usd, 6),
            "net_deposits_usd": round(self.net_deposits_usd, 6),
            "deposits_log": self.deposits_log[-50:],
            "n_trades": int(self.n_trades),
            "reserved_account_usdt": round(self.reserved_account_usdt, 6),
            "user_reserved_coin": round(self.user_reserved_coin, 6),
            "active_bid_id": self.active_bid_id,
            "active_bid_p": round(self.active_bid_p, 4),
            "active_bid_q": round(self.active_bid_q, 4),
            "active_ask_id": self.active_ask_id,
            "active_ask_p": round(self.active_ask_p, 4),
            "active_ask_q": round(self.active_ask_q, 4),
            "slot2_ask_id": self.slot2_ask_id,
            "slot2_ask_p": round(self.slot2_ask_p, 4),
            "slot2_ask_q": round(self.slot2_ask_q, 4),
            "split_done_for_pos": bool(self.split_done_for_pos),
            "car_state": int(getattr(self, "car_state", 0)),
            "car_sell_px": round(float(getattr(self, "car_sell_px", 0.0)), 4),
            "car_cycles": int(getattr(self, "car_cycles", 0)),
        }
        try:
            self.wallet_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def sync_live_orders(self, b1: float, count_hold: bool = True):
        if self.mode != "live":
            return
        if count_hold and self.pos_qty > 1e-4:
            self.hold_sec += 1
            if b1 > 0:
                self.peak_bid = max(self.peak_bid, self.avg_entry, b1)
        if self.active_bid_id is not None:
            if count_hold:
                self.bid_age_sec = getattr(self, "bid_age_sec", 0) + 1
            try:
                o = self.client.get_order(self.active_bid_id)
                filled = float(o.get("filled_amount", 0.0))
                st = o.get("state", "")
                self.bid_partial_filled = filled if st == "wait" else 0.0
                if filled > 0 and st in ("done", "cancel"):
                    px = float(o.get("avg_price", 0.0)) or float(o.get("price", self.active_bid_p))
                    fee_rate = float(o.get("maker_fee", 0.001) or 0.001)
                    cost = filled * px
                    net_filled = math.floor(filled * (1.0 - fee_rate) * 10000.0) / 10000.0
                    self.bot_cash_usd = max(0.0, self.bot_cash_usd - cost)
                    if getattr(self, "car_state", 0) == 2 and self.pos_qty >= 2.0:
                        net_car_pnl = filled * (self.car_sell_px * (1.0 - fee_rate) - px / max(1e-6, 1.0 - fee_rate))
                        self.pos_qty = math.floor((self.pos_qty + net_filled) * 10000.0) / 10000.0
                        self.avg_entry = round(max(self.tick_size, self.avg_entry - max(0.0, net_car_pnl) / max(1e-6, self.pos_qty)), 4)
                        self.car_cycles = getattr(self, "car_cycles", 0) + 1
                        self.car_state = 0
                        print(f"  🎠 [CAROUSEL BUYBACK #{self.car_cycles}] +{net_filled:.4f} PRL @ ${px:.2f} (после продажи @ ${self.car_sell_px:.2f}) | Новый avg_entry=${self.avg_entry:.4f}")
                    else:
                        self.avg_entry = (self.pos_qty * self.avg_entry + net_filled * px) / max(1e-6, self.pos_qty + net_filled)
                        self.pos_qty = math.floor((self.pos_qty + net_filled) * 10000.0) / 10000.0
                        self.hold_sec = 0
                        self.peak_bid = max(px, b1 if b1 > 0 else px)
                        self.ez25_entry = self.last_ez25
                        self.split_done_for_pos = False
                    self.q_ahead = 0.0
                    self.bid_age_sec = 0
                    self.bid_partial_filled = 0.0
                    self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                    self.sync_account_balances(b1, from_trade_fill=True)
                    self._save_wallet()
                    print(f"  🟢 [FILL BUY] +{net_filled:.4f} PRL (брутто {filled:.4f}) @ ${px:.2f} | Остаток кэша бота=${self.bot_cash_usd:.2f}")
                elif st == "cancel":
                    self.q_ahead = 0.0
                    self.bid_age_sec = 0
                    self.bid_partial_filled = 0.0
                    self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                    self._save_wallet()
            except Exception as e:
                print(f"  ⚠️ [SYNC BID WARN] id={self.active_bid_id}: {e}")
        if self.active_ask_id is not None:
            try:
                o = self.client.get_order(self.active_ask_id)
                filled = float(o.get("filled_amount", 0.0))
                st = o.get("state", "")
                if filled > 0 and st in ("done", "cancel"):
                    px = float(o.get("avg_price", 0.0)) or float(o.get("price", self.active_ask_p))
                    fee_rate = float(o.get("maker_fee", 0.001) or 0.001)
                    proceeds = filled * px * (1.0 - fee_rate)
                    cost_basis = (filled / max(1e-6, 1.0 - fee_rate)) * self.avg_entry
                    pnl = proceeds - cost_basis
                    self.bot_cash_usd += proceeds
                    self.realized_pnl_usd += pnl
                    self.pos_qty = max(0.0, math.floor((self.pos_qty - filled) * 10000.0) / 10000.0)
                    if self.pos_qty < 2.0:
                        self.pos_qty, self.avg_entry, self.hold_sec, self.peak_bid = 0.0, 0.0, 0, 0.0
                        self.split_done_for_pos = False
                    self.n_trades += 1
                    self.q_ahead = 0.0
                    if self.slot2_ask_id is not None and self.pos_qty >= 2.0:
                        self.active_ask_id = self.slot2_ask_id
                        self.active_ask_p = self.slot2_ask_p
                        self.active_ask_q = self.slot2_ask_q
                        self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                        self.q_ahead = -1.0
                    else:
                        self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                    self.sync_account_balances(b1, from_trade_fill=True)
                    self._save_wallet()
                    print(f"  💰 [FILL SELL SLOT-1] -{filled:.4f} PRL @ ${px:.2f} | PnL={pnl:+.4f}$ | Остаток поз={self.pos_qty:.4f} PRL | Кэш=${self.bot_cash_usd:.2f}")
                elif st == "cancel":
                    self.q_ahead = 0.0
                    self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                    self._save_wallet()
            except Exception as e:
                print(f"  ⚠️ [SYNC ASK WARN] id={self.active_ask_id}: {e}")
        if self.slot2_ask_id is not None:
            try:
                o2 = self.client.get_order(self.slot2_ask_id)
                filled2 = float(o2.get("filled_amount", 0.0))
                st2 = o2.get("state", "")
                if filled2 > 0 and st2 in ("done", "cancel"):
                    px2 = float(o2.get("avg_price", 0.0)) or float(o2.get("price", self.slot2_ask_p))
                    fee_rate2 = float(o2.get("maker_fee", 0.001) or 0.001)
                    proceeds2 = filled2 * px2 * (1.0 - fee_rate2)
                    cost_basis2 = (filled2 / max(1e-6, 1.0 - fee_rate2)) * self.avg_entry
                    pnl2 = proceeds2 - cost_basis2
                    self.bot_cash_usd += proceeds2
                    self.realized_pnl_usd += pnl2
                    self.pos_qty = max(0.0, math.floor((self.pos_qty - filled2) * 10000.0) / 10000.0)
                    if self.pos_qty < 2.0:
                        self.pos_qty, self.avg_entry, self.hold_sec, self.peak_bid = 0.0, 0.0, 0, 0.0
                        self.split_done_for_pos = False
                        self.car_state = 0
                    elif getattr(self, "car_state", 0) == 1:
                        self.car_sell_px = px2
                        self.car_state = 2
                        buy_p = round((px2 - 2.0 * self.tick_size) / self.tick_size) * self.tick_size
                        buy_q = math.floor((proceeds2 * 0.998 / max(buy_p, self.tick_size)) * 10000.0) / 10000.0
                        if buy_q >= 2.0 and self.active_bid_id is None:
                            try:
                                rb = self.client.place_limit_order(self.market, "buy", buy_q, buy_p)
                                self.active_bid_id = rb.get("id")
                                self.active_bid_p = float(rb.get("price", buy_p))
                                self.active_bid_q = float(rb.get("origin_amount", buy_q))
                                print(f"  🎠 [CAROUSEL LEG-2 BUY] id={self.active_bid_id} | Откуп {self.active_bid_q:.4f} PRL @ ${self.active_bid_p:.2f} (-2 тика от ${px2:.2f})")
                            except Exception as eb:
                                print(f"  ⚠️ [CAROUSEL BUY ERROR]: {eb}")
                    self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                    self.sync_account_balances(b1, from_trade_fill=True)
                    self._save_wallet()
                    print(f"  💰 [FILL SELL SLOT-2] -{filled2:.4f} PRL @ ${px2:.2f} | PnL={pnl2:+.4f}$ | Остаток поз={self.pos_qty:.4f} PRL | Кэш=${self.bot_cash_usd:.2f}")
                elif st2 == "cancel":
                    self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                    self._save_wallet()
            except Exception as e:
                print(f"  ⚠️ [SYNC ASK-2 WARN] id={self.slot2_ask_id}: {e}")

    def sync_account_balances(self, b1: float = 0.0, init_mode: bool = False, from_trade_fill: bool = False):
        if self.mode != "live" or self.client is None:
            return
        try:
            raw_bals = self.client.get_spot_balances()
            b_map = {b["currency"]: (float(b.get("balance", 0.0)), float(b.get("locked", 0.0))) for b in raw_bals}
            u_free, u_lock = b_map.get("usdt", (0.0, 0.0))
            c_free, c_lock = b_map.get(self.base_currency, (0.0, 0.0))
            u_tot, c_tot = u_free + u_lock, c_free + c_lock
            self.ex_usdt_free, self.ex_coin_free = u_free, c_free

            # На выделенном аккаунте (reserved_account_usdt == 0 и user_reserved_coin == 0)
            # 100% спотового баланса биржи (USDT + PRL) является прямым источником истины для бота!
            if self.reserved_account_usdt <= 1e-4 and self.user_reserved_coin <= 1e-4:
                prev_pos = self.pos_qty
                prev_cash_tot = self.bot_cash_usd
                real_pos = math.floor(c_tot * 10000.0) / 10000.0 if c_tot >= 2.0 else 0.0
                px_ref = b1 if b1 > 0 else (self.avg_entry if self.avg_entry > 0 else (self.active_bid_p if self.active_bid_p > 0 else 1.77))

                # Автоматический учет внешних пополнений монетами (PRL) или кэшем (USDT) без искажения PnL
                if not from_trade_fill and self.initial_budget_usd > 0.0:
                    dc_ext = real_pos - prev_pos
                    du_ext = u_tot - prev_cash_tot
                    if dc_ext >= 1.0 and du_ext >= -1.0:
                        dep_usd = round(dc_ext * px_ref, 6)
                        self.net_deposits_usd = round(self.net_deposits_usd + dep_usd, 6)
                        self.initial_budget_usd = round(self.initial_budget_usd + dep_usd, 6)
                        if prev_pos < 2.0:
                            self.avg_entry = px_ref
                            self.hold_sec = 0
                            self.peak_bid = px_ref
                        else:
                            self.avg_entry = round((prev_pos * self.avg_entry + dc_ext * px_ref) / real_pos, 6)
                            self.peak_bid = max(self.peak_bid, self.avg_entry, px_ref)
                        self.deposits_log.append({
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "asset": self.coin.upper(),
                            "qty": round(dc_ext, 6),
                            "price_usd": round(px_ref, 4),
                            "usd_equiv": round(dep_usd, 4),
                            "new_budget_usd": round(self.initial_budget_usd, 4),
                        })
                        print(
                            f"  🏦 [DEPOSIT SYNC] Внешнее пополнение: +{dc_ext:.4f} {self.coin.upper()} (~${dep_usd:.2f} по Bid=${px_ref:.2f}) | "
                            f"База капитала бота: ${self.initial_budget_usd:.2f} (чистый торговый PnL сохранен)"
                        )
                    if du_ext >= 1.0 and dc_ext >= -0.5:
                        self.net_deposits_usd = round(self.net_deposits_usd + du_ext, 6)
                        self.initial_budget_usd = round(self.initial_budget_usd + du_ext, 6)
                        self.deposits_log.append({
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "asset": "USDT",
                            "qty": round(du_ext, 4),
                            "price_usd": 1.0,
                            "usd_equiv": round(du_ext, 4),
                            "new_budget_usd": round(self.initial_budget_usd, 4),
                        })
                        print(
                            f"  🏦 [DEPOSIT SYNC] Внешнее пополнение: +${du_ext:.2f} USDT | "
                            f"База капитала бота: ${self.initial_budget_usd:.2f} (чистый торговый PnL сохранен)"
                        )

                self.bot_cash_usd = u_tot
                if real_pos >= 2.0 and prev_pos < 2.0:
                    self.avg_entry = self.avg_entry if self.avg_entry > 0 else px_ref
                    self.peak_bid = max(self.peak_bid, self.avg_entry, px_ref)
                    if not init_mode:
                        print(f"  🔄 [SPOT SYNC] Позиция на споте активна: {real_pos:.4f} PRL (вход=${self.avg_entry:.2f}) | Свободный USDT=${u_free:.2f}")
                elif real_pos < 2.0 and prev_pos >= 2.0:
                    self.avg_entry, self.hold_sec, self.peak_bid = 0.0, 0, 0.0
                    self.n_trades += 1
                    self.realized_pnl_usd = u_tot - self.initial_budget_usd
                    if not init_mode:
                        print(f"  🔄 [SPOT SYNC] Позиция полностью продана в USDT: баланс=${u_tot:.2f} (PnL={self.realized_pnl_usd:+.4f}$)")
                self.pos_qty = real_pos
                if init_mode and self.initial_budget_usd <= 0.0:
                    self.initial_budget_usd = round(u_tot + real_pos * px_ref, 4)
                self._save_wallet()
                return

            if self.pos_qty >= 2.0 and self.active_ask_id is None and c_tot < 2.0:
                px_ref = b1 if b1 > 0 else (self.avg_entry if self.avg_entry > 0 else 1.82)
                if self.bot_cash_usd < self.initial_budget_usd * 0.5:
                    self.bot_cash_usd += self.pos_qty * px_ref * 0.999
                    self.realized_pnl_usd = self.bot_cash_usd - self.initial_budget_usd
                print(f"  🔄 [BALANCE SYNC] Позиция бота ({self.pos_qty:.4f} PRL) закрыта извне. Кэш бота=${self.bot_cash_usd:.2f}")
                self.pos_qty, self.avg_entry, self.hold_sec = 0.0, 0.0, 0
            elif self.pos_qty >= 2.0 and self.active_ask_id is None and c_free < self.pos_qty:
                clamped = math.floor(c_free * 10000.0) / 10000.0
                if clamped != self.pos_qty:
                    print(f"  🔧 [POS SYNC] Уточнен объем позиции бота по споту: {self.pos_qty:.4f} -> {clamped:.4f} PRL")
                    self.pos_qty = clamped if clamped >= 2.0 else 0.0

            new_res_u = max(0.0, u_tot - self.bot_cash_usd)
            new_res_c = max(0.0, c_tot - self.pos_qty)
            if not init_mode and (abs(new_res_u - self.reserved_account_usdt) >= 0.50 or abs(new_res_c - self.user_reserved_coin) >= 0.50):
                du = new_res_u - self.reserved_account_usdt
                dc = new_res_c - self.user_reserved_coin
                print(
                    f"  🔄 [USER TRADE SYNC] Движение личного резерва счета: USDT=${new_res_u:,.2f} ({du:+.2f}$), "
                    f"личные монеты={new_res_c:,.4f} {self.coin.upper()} ({dc:+.4f}) | Бот изолирован: кэш=${self.bot_cash_usd:.2f}, поз={self.pos_qty:.4f}"
                )
            self.reserved_account_usdt = new_res_u
            self.user_reserved_coin = new_res_c
            self._save_wallet()
        except Exception as e:
            print(f"  ⚠️ [BALANCE SYNC WARN] {e}")

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
        tol = self.tick_size * 0.25
        b1 = float(max(r.get("ST_bid_1_price", r.get("ST_best_bid", 0.0)), self.tick_size))
        a1 = float(max(r.get("ST_ask_1_price", r.get("ST_best_ask", b1 + self.tick_size)), b1 + self.tick_size))

        # Вычитаем собственный импакт (наши активные лимитные заявки Slot 1 / Slot 2) из 10 уровней стакана L2
        bid_qs_clean, ask_qs_clean, ask_ps = [], [], []
        for i in range(1, 11):
            bp = float(r.get(f"ST_bid_{i}_price", b1 if i == 1 else 0.0))
            bq = max(0.0, float(r.get(f"ST_bid_{i}_qty", r.get("ST_bid_qty_1", 0.0) if i == 1 else 0.0)))
            if self.active_bid_q > 0 and abs(bp - self.active_bid_p) <= tol:
                bq = max(0.0, bq - self.active_bid_q)
            bid_qs_clean.append(bq)

            ap = float(r.get(f"ST_ask_{i}_price", a1 if i == 1 else 0.0))
            aq = max(0.0, float(r.get(f"ST_ask_{i}_qty", r.get("ST_ask_qty_1", 0.0) if i == 1 else 0.0)))
            if self.active_ask_q > 0 and abs(ap - self.active_ask_p) <= tol:
                aq = max(0.0, aq - self.active_ask_q)
            if self.slot2_ask_q > 0 and abs(ap - self.slot2_ask_p) <= tol:
                aq = max(0.0, aq - self.slot2_ask_q)
            ask_qs_clean.append(aq)
            ask_ps.append(ap)

        b1_q, a1_q = bid_qs_clean[0], ask_qs_clean[0]
        b5_q, a5_q = sum(bid_qs_clean[:5]), sum(ask_qs_clean[:5])
        b10_q, a10_q = sum(bid_qs_clean), sum(ask_qs_clean)

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
        h_3600 = max(b_arr[-3600:])
        l_3600 = min(b_arr[-3600:])
        h_14400 = max(b_arr)
        l_14400 = min(b_arr)
        l_30 = min(b_arr[-30:])

        dd_300 = round((h_300 - b1) / self.tick_size, 4)
        dd_900 = round((h_900 - b1) / self.tick_size, 4)
        pos_900 = ((b1 - l_900) / max(h_900 - l_900, self.tick_size)) if h_900 > l_900 + 1e-6 else 1.0
        pos_3600 = ((b1 - l_3600) / max(h_3600 - l_3600, self.tick_size)) if h_3600 > l_3600 + 1e-6 else 1.0
        rng_14400 = round((h_14400 - l_14400) / self.tick_size, 4)
        pos_14400 = (b1 - l_14400) / max(h_14400 - l_14400, self.tick_size)
        reb_30 = round((b1 - l_30) / self.tick_size, 4)
        vel_10 = round((b1 - (b_arr[-11] if len(b_arr) >= 11 else b_arr[0])) / self.tick_size, 4)
        vel_15 = round((b1 - (b_arr[-16] if len(b_arr) >= 16 else b_arr[0])) / self.tick_size, 4)
        vel_180 = round((b1 - (b_arr[-181] if len(b_arr) >= 181 else b_arr[0])) / self.tick_size, 4)

        d_ez12 = ez_12 - (self.ez12_hist[-6] if len(self.ez12_hist) >= 6 else self.ez12_hist[0])
        inertia = float(
            0.45 * ez_25 + 0.35 * ez_12 + 0.20 * ez_5
            + 0.25 * d_ez12 + 0.35 * imb_l5 + 0.15 * imb_l1
            + 0.10 * np.clip(vel_15, -3.0, 3.0)
        )
        self.last_ez25 = ez_25
        self.last_ez12 = ez_12
        self.last_inertia = inertia
        self.last_pos3600 = pos_3600
        self.last_rng14400 = rng_14400
        self.last_pos14400 = pos_14400
        self.last_b1_q = b1_q
        self.last_a1_q = a1_q
        self.last_vel10 = vel_10
        self.last_imb5 = imb_l5
        self.last_dd300 = dd_300
        self.last_dd900 = dd_900
        self.last_sp_ticks = spread_ticks

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
        self.last_hold_edge = float(obs_hold_edge_norm * 5.0)
        if self.pos_qty > 1e-8 and self.avg_entry > 0:
            fl_p = round((self.avg_entry + self.tick_size) / self.tick_size) * self.tick_size
            wv_p = round((self.avg_entry + 2.0 * self.tick_size) / self.tick_size) * self.tick_size
            self.last_q_a1 = sum(aq for ap, aq in zip(ask_ps, ask_qs_clean) if abs(ap - fl_p) <= tol)
            self.last_q_a2 = sum(aq for ap, aq in zip(ask_ps, ask_qs_clean) if abs(ap - wv_p) <= tol)
        else:
            self.last_q_a1, self.last_q_a2 = a1_q, (ask_qs_clean[1] if len(ask_qs_clean) > 1 else 0.0)

        has_pos = 1.0 if self.pos_qty > 1e-8 else 0.0
        unr_ticks = ((b1 - self.avg_entry) / self.tick_size) if has_pos else 0.0
        hold_ratio = min(self.hold_sec / 45.0, 2.0) if has_pos else 0.0
        has_bid = 1.0 if self.active_bid_q > 0 else 0.0
        has_ask = 1.0 if self.active_ask_q > 0 else 0.0
        tol = self.tick_size * 0.25
        if has_bid:
            q_at, q_better = 0.0, 0.0
            for k in range(1, 11):
                lp = float(r.get(f"ST_bid_{k}_price", 0.0))
                lq = float(r.get(f"ST_bid_{k}_qty", 0.0))
                if abs(lp - self.active_bid_p) <= tol:
                    q_at += max(0.0, lq - self.active_bid_q)
                elif lp > self.active_bid_p + tol:
                    q_better += lq
            if self.q_ahead < 0.0:
                self.q_ahead = q_at
            else:
                q_better_eff = max(q_better, getattr(self, "_prev_q_better", q_better))
                q_at_prev = getattr(self, "_prev_q_at", q_at)
                vol_hit = float(r.get("ST_sell_vol", 0.0)) if abs(b1 - self.active_bid_p) <= tol else 0.0
                vol_at_px = max(0.0, vol_hit - q_better_eff)
                if vol_at_px > 0.0:
                    self.q_ahead = max(0.0, self.q_ahead - vol_at_px)
                if q_at < q_at_prev - vol_at_px and q_at_prev > 1e-6:
                    net_cancel = (q_at_prev - vol_at_px) - q_at
                    self.q_ahead = max(0.0, self.q_ahead - net_cancel * min(1.0, 0.35 * self.q_ahead / q_at_prev))
                self.q_ahead = min(self.q_ahead, q_at)
            self._prev_q_at, self._prev_q_better = q_at, q_better
        elif has_ask:
            q_at, q_better = 0.0, 0.0
            for k in range(1, 11):
                lp = float(r.get(f"ST_ask_{k}_price", 0.0))
                lq = float(r.get(f"ST_ask_{k}_qty", 0.0))
                if abs(lp - self.active_ask_p) <= tol:
                    q_at += max(0.0, lq - self.active_ask_q)
                elif 0.0 < lp < self.active_ask_p - tol:
                    q_better += lq
            if self.q_ahead < 0.0:
                self.q_ahead = q_at
            else:
                q_better_eff = max(q_better, getattr(self, "_prev_q_better", q_better))
                q_at_prev = getattr(self, "_prev_q_at", q_at)
                vol_hit = float(r.get("ST_buy_vol", 0.0)) if abs(a1 - self.active_ask_p) <= tol else 0.0
                vol_at_px = max(0.0, vol_hit - q_better_eff)
                if vol_at_px > 0.0:
                    self.q_ahead = max(0.0, self.q_ahead - vol_at_px)
                if q_at < q_at_prev - vol_at_px and q_at_prev > 1e-6:
                    net_cancel = (q_at_prev - vol_at_px) - q_at
                    self.q_ahead = max(0.0, self.q_ahead - net_cancel * min(1.0, 0.35 * self.q_ahead / q_at_prev))
                self.q_ahead = min(self.q_ahead, q_at)
            self._prev_q_at, self._prev_q_better = q_at, q_better
        else:
            self.q_ahead = 0.0
            self._prev_q_at, self._prev_q_better = 0.0, 0.0
        dist = ((b1 - self.active_bid_p) / self.tick_size) if has_bid else (((self.active_ask_p - a1) / self.tick_size) if has_ask else 0.0)
        ord_q = self.active_bid_q if has_bid else self.active_ask_q
        q_val = max(0.0, self.q_ahead)
        q_ratio = q_val / (q_val + ord_q + 1e-6) if (has_bid or has_ask) else 0.0

        obs = np.array(
            triad_vec + lob_vec + [
                has_pos, float(np.clip(unr_ticks * 0.1, -3.0, 3.0)), hold_ratio,
                has_bid, has_ask, float(np.clip(dist * 0.2, -2.0, 2.0)), q_ratio,
                obs_err12_norm, obs_entry_edge_norm, obs_hold_edge_norm,
            ],
            dtype=np.float32,
        )

        vel_3 = (b1 - (b_arr[-4] if len(b_arr) >= 4 else b_arr[0])) / self.tick_size
        self.last_vel3 = vel_3
        safe_macro_4h = (rng_14400 < 6.0) or (pos_14400 <= 0.72)
        in_dip_zone = (
            (pos_3600 <= 0.80)
            and safe_macro_4h
            and (vel_10 >= -0.5)
            and (inertia >= 0.15)
            and (exp_entry_edge >= 0.60)
            and ((dd_300 >= 1.0) or (spread_ticks >= 2.0))
        )

        mask = np.zeros(7, dtype=bool)
        if not has_pos:
            mask[0] = True
            if has_bid:
                drift_limit = self.tick_size * 1.25 if getattr(self, "active_bid_act", 1) == 2 else tol
                need_cancel = (inertia < -0.10) or (pos_3600 > 0.82) or (imb_l5 < -0.55) or (b1 > self.active_bid_p + drift_limit)
                if need_cancel:
                    mask[5] = True
                    if in_dip_zone:
                        mask[2] = True
                        if imb_l5 > -0.65:
                            mask[1] = True
                elif getattr(self, "active_bid_act", 1) == 2 and in_dip_zone and imb_l5 > -0.65:
                    mask[1] = True
            elif in_dip_zone:
                mask[2] = True
                if imb_l5 > -0.65:
                    mask[1] = True
        else:
            mask[0] = bool(has_ask)
            mask[3] = True
            mask[4] = True
            if unr_ticks <= -2.0 or inertia < -0.25 or obs_err12_norm < -0.60:
                mask[6] = True
        return obs, mask

    def _compute_adaptive_wave_weight(self) -> float:
        ez25 = float(self.last_ez25)
        ez12 = float(self.last_ez12)
        hold_edge = float(self.last_hold_edge)
        b1_cur = float(self.bid_hist_1h[-1]) if len(self.bid_hist_1h) > 0 else self.avg_entry
        if self.avg_entry > 0 and b1_cur < self.avg_entry - self.tick_size * 0.25:
            return 0.0
        q_a1 = float(self.last_q_a1)
        q_a2 = float(self.last_q_a2)
        own_ask_share = self.pos_qty / (q_a1 + self.pos_qty + 1e-6)
        if self.last_pos3600 >= 0.65 and q_a2 > 3000.0:
            return 0.0
        if q_a2 > 15000.0 and q_a1 < 500.0:
            return 0.50
        if (hold_edge >= 0.90 and ez25 >= 0.15 and ez12 >= 0.08) or (hold_edge >= 0.80 and q_a2 < 250.0 and q_a1 > 500.0):
            return 1.0
        if (hold_edge >= 0.45 and ez25 >= 0.08) or (ez25 >= 0.14 and hold_edge >= 0.20) or (own_ask_share > 0.15 and ez25 >= 0.08 and hold_edge >= 0.10):
            q_adv = q_a1 / (q_a1 + q_a2 + 200.0)
            return float(np.clip(0.35 + 0.12 * max(0.0, hold_edge) + 0.25 * q_adv, 0.30, 0.65))
        return 0.0

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
        if self.pos_qty <= 1e-4 and self.active_bid_id is not None:
            at_b1 = abs(self.active_bid_p - b1) <= tol
            compressed_4h_floor = (4.8 <= getattr(self, "last_rng14400", 10.0) <= 5.5 and getattr(self, "last_pos14400", 0.5) <= 0.05 and self.last_dd900 <= 1.0)
            pre_dump = (self.last_sp_ticks < 1.5 and self.last_ez25 < -0.15 and self.last_inertia < 0.08 and getattr(self, "last_a1_q", self.last_q_a1) > 4000.0) and not compressed_4h_floor
            front_q_hold = at_b1 and ((0.0 <= self.q_ahead < 600.0) or getattr(self, "bid_partial_filled", 0.0) > 0.0) and not pre_dump and self.last_pos3600 <= 0.65 and self.last_inertia >= -0.10
            v14_hold_q = at_b1 and ((self.last_pos3600 <= 0.25 and not pre_dump and getattr(self, "last_b1_q", 0.0) >= 5000.0) or compressed_4h_floor or front_q_hold)
            bid_age = getattr(self, "bid_age_sec", 0)
            if self.active_bid_p > b1 + tol and self.last_vel10 <= -1.0 and self.last_hold_edge < 0.0 and getattr(self, "last_a1_q", self.last_q_a1) > 500.0:
                act = 5
            elif v14_hold_q:
                if act == 5:
                    act = 0
            elif self.last_pos3600 > 0.85 or self.last_inertia < -0.20:
                act = 5
            elif self.last_dd300 < 0.5 and self.last_dd900 <= 1.0 and self.last_sp_ticks < 1.5 and self.last_ez12 < 0.30:
                if bid_age >= 15 and self.last_ez12 < 0.12 and self.last_inertia < 0.08 and getattr(self, "bid_partial_filled", 0.0) <= 0.0:
                    act = 5
                elif act == 5:
                    act = 0
            elif pre_dump:
                act = 5

        if act in (1, 2) and self.pos_qty <= 1e-4:
            step_in = (a1 - b1) >= 2.0 * self.tick_size and self.last_inertia >= 0.35
            target_p = round(((b1 + self.tick_size if step_in else b1) if act == 1 else max(self.tick_size, b1 - self.tick_size)) / self.tick_size) * self.tick_size
            if self.active_bid_id is not None and abs(self.active_bid_p - target_p) <= tol:
                return
            if self.active_bid_id is not None:
                try:
                    self.client.cancel_order(self.active_bid_id)
                except Exception as e:
                    print(f"  ⚠️ [CANCEL BUY WARN] id={self.active_bid_id}: {e}")
                self.sync_live_orders(b1)
                self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                self.sync_account_balances(b1)
                if self.pos_qty > 1e-4:
                    return
            self.sync_account_balances(b1)
            max_cap = self.ex_usdt_free if (self.reserved_account_usdt <= 1e-4 and self.user_reserved_coin <= 1e-4) else self.order_usd
            usable_usd = min(max_cap, self.bot_cash_usd * 0.998, self.ex_usdt_free * 0.998)
            qty = math.floor((usable_usd / max(target_p, self.tick_size)) * 10000.0) / 10000.0
            if qty < 2.0:
                return
            try:
                res = self.client.place_limit_order(self.market, "buy", qty, target_p)
                self.active_bid_id = res.get("id")
                self.active_bid_p = float(res.get("price", target_p))
                self.active_bid_q = float(res.get("origin_amount", qty))
                self.active_bid_act = int(act)
                self.q_ahead = -1.0
                self.bid_age_sec = 0
                self.bid_partial_filled = 0.0
                self._save_wallet()
                print(f"  📥 [LIVE ORDER BUY] id={self.active_bid_id} | {self.active_bid_q:.4f} PRL @ ${self.active_bid_p:.2f} (${self.active_bid_q*self.active_bid_p:.2f})")
            except Exception as e:
                print(f"  ⚠️ [ORDER BUY ERROR] qty={qty:.4f} @ ${target_p:.2f}: {e}")
        elif act == 5 and self.active_bid_id is not None:
            old_id = self.active_bid_id
            try:
                self.client.cancel_order(old_id)
                print(f"  ✖️ [LIVE CANCEL BUY] id={old_id}")
            except Exception as e:
                print(f"  ⚠️ [CANCEL BUY WARN] id={old_id}: {e}")
            self.sync_live_orders(b1)
            self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
            self.q_ahead = 0.0
            self.sync_account_balances(b1)
            self._save_wallet()
        elif (act in (3, 4, 6) or (act == 0 and self.active_ask_id is not None)) and self.pos_qty >= 2.0:
            floor_p = round((self.avg_entry + self.tick_size) / self.tick_size) * self.tick_size
            wave_p = round((self.avg_entry + 2.0 * self.tick_size) / self.tick_size) * self.tick_size
            super_p = round((self.avg_entry + 3.0 * self.tick_size) / self.tick_size) * self.tick_size
            if act == 0:
                if self.active_ask_p >= wave_p - tol and b1 < self.avg_entry - tol and self.last_hold_edge < -0.20 and a1 <= floor_p + tol:
                    self.split_done_for_pos = False
                    act = 4
                elif getattr(self, "car_state", 0) == 0 and (self.avg_entry - b1) / self.tick_size >= 3.0 and self.hold_sec >= 120 and self.slot2_ask_id is None:
                    act = 4
                else:
                    return
            if act in (3, 4):
                w2 = self._compute_adaptive_wave_weight()
                q_a1 = float(self.last_q_a1)
                q_a2 = float(self.last_q_a2)
                triad_super = (
                    self.last_sp_ticks < 1.5
                    and self.last_ez12 >= 0.70
                    and self.last_ez25 >= 0.70
                    and self.last_hold_edge >= 3.10
                    and (self.last_ez25 >= 0.85 or self.last_hold_edge >= 3.80)
                    and self.last_inertia >= 1.25
                    and q_a1 < 1000.0
                    and q_a2 < 4500.0
                    and self.last_dd900 >= 2.0
                )
                is_super = (
                    (self.last_dd300 >= 4.0 and self.last_sp_ticks < 1.5 and self.last_ez25 >= 0.20 and self.last_hold_edge >= 1.50 and q_a2 < 800.0)
                    or (self.last_dd900 >= 3.0 and self.last_pos3600 <= 0.05 and self.last_ez25 >= 0.65 and self.last_hold_edge >= 4.0 and q_a2 < 300.0)
                    or triad_super
                )
                if self.active_ask_id is not None and self.active_ask_p >= wave_p - tol and self.split_done_for_pos:
                    return
                elif self.active_ask_id is not None and self.slot2_ask_id is not None:
                    if w2 <= 0.05 and a1 <= floor_p + tol and q_a2 <= 15000.0:
                        try:
                            self.client.cancel_order(self.slot2_ask_id)
                        except Exception as e:
                            print(f"  ⚠️ [CANCEL SLOT-2 WARN] id={self.slot2_ask_id}: {e}")
                        self.sync_live_orders(b1)
                        self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                        self.sync_account_balances(b1)
                        rem_free = math.floor(self.ex_coin_free * 10000.0) / 10000.0
                        if rem_free >= 2.0:
                            try:
                                r2 = self.client.place_limit_order(self.market, "sell", rem_free, max(a1, floor_p))
                                self.slot2_ask_id = r2.get("id")
                                self.slot2_ask_p = float(r2.get("price", max(a1, floor_p)))
                                self.slot2_ask_q = float(r2.get("origin_amount", rem_free))
                                self._save_wallet()
                                print(f"  🔄 [REBALANCE SLOT-2 -> +1t] id={self.slot2_ask_id} | {self.slot2_ask_q:.4f} PRL @ ${self.slot2_ask_p:.2f}")
                            except Exception as e:
                                print(f"  ⚠️ [REBALANCE SLOT-2 ERROR]: {e}")
                    return
                elif self.active_ask_id is not None and self.slot2_ask_id is None:
                    underwater_ticks = (self.avg_entry - b1) / self.tick_size
                    if getattr(self, "car_state", 0) == 0 and underwater_ticks >= 3.0 and self.hold_sec >= 120 and self.pos_qty >= 8.0:
                        if (a1 - b1) >= 1.5 * self.tick_size or (self.last_pos3600 >= 0.30 and self.last_inertia > -0.10):
                            try:
                                self.client.cancel_order(self.active_ask_id)
                            except Exception:
                                pass
                            self.sync_live_orders(b1)
                            self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                            self.sync_account_balances(b1)
                            tot_c = math.floor(min(self.pos_qty, self.ex_coin_free) * 10000.0) / 10000.0
                            q_car = math.floor((tot_c * 0.25) * 10000.0) / 10000.0
                            q_core = math.floor((tot_c - q_car) * 10000.0) / 10000.0
                            if q_core >= 2.0 and q_car >= 2.0:
                                try:
                                    r1 = self.client.place_limit_order(self.market, "sell", q_core, max(a1, floor_p))
                                    self.active_ask_id = r1.get("id")
                                    self.active_ask_p = float(r1.get("price", max(a1, floor_p)))
                                    self.active_ask_q = float(r1.get("origin_amount", q_core))
                                    r2 = self.client.place_limit_order(self.market, "sell", q_car, a1)
                                    self.slot2_ask_id = r2.get("id")
                                    self.slot2_ask_p = float(r2.get("price", a1))
                                    self.slot2_ask_q = float(r2.get("origin_amount", q_car))
                                    self.car_state = 1
                                    self._save_wallet()
                                    print(f"  🎠 [CAROUSEL LEG-1 SELL] Core={self.active_ask_q:.4f} PRL @ ${self.active_ask_p:.2f} | Carousel(25%)={self.slot2_ask_q:.4f} PRL @ ${self.slot2_ask_p:.2f}")
                                    return
                                except Exception as ec:
                                    print(f"  ⚠️ [CAROUSEL LEG-1 ERROR]: {ec}")
                    if abs(self.active_ask_p - floor_p) <= tol:
                        return
                    elif (w2 >= 0.30 or q_a2 > 15000.0) and self.active_ask_p >= wave_p - tol:
                        return
                    else:
                        act = 4
                elif self.active_ask_id is None and (is_super or 0.15 <= w2 <= 0.85) and not self.split_done_for_pos:
                    self.sync_account_balances(b1)
                    tot_sell = math.floor(min(self.pos_qty, self.ex_coin_free) * 10000.0) / 10000.0
                    eff_w2 = 1.00 if is_super else w2
                    if is_super and eff_w2 >= 0.99 and tot_sell >= 2.0:
                        p_sup = round(max(a1 + 2.0 * self.tick_size, super_p) / self.tick_size) * self.tick_size
                        try:
                            res_s = self.client.place_limit_order(self.market, "sell", tot_sell, p_sup)
                            self.active_ask_id = res_s.get("id")
                            self.active_ask_p = float(res_s.get("price", p_sup))
                            self.active_ask_q = float(res_s.get("origin_amount", tot_sell))
                            self.q_ahead = -1.0
                            self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                            self.split_done_for_pos = True
                            self._save_wallet()
                            print(f"  🚀 [LIVE SUPER-WAVE +3t] id={self.active_ask_id} | {self.active_ask_q:.4f} PRL @ ${self.active_ask_p:.2f}")
                            return
                        except Exception as e:
                            print(f"  ⚠️ [SUPER-WAVE SELL ERROR]: {e}")
                            self.sync_account_balances(b1)
                    q2 = math.floor((tot_sell * eff_w2) * 10000.0) / 10000.0
                    q1 = math.floor((tot_sell - q2) * 10000.0) / 10000.0
                    if q1 >= 2.0 and q2 >= 2.0:
                        p1 = round((max(a1 + self.tick_size, wave_p) if is_super else max(a1, floor_p)) / self.tick_size) * self.tick_size
                        p2 = round((max(a1 + 2.0 * self.tick_size, super_p) if is_super else max(a1 + self.tick_size, wave_p)) / self.tick_size) * self.tick_size
                        try:
                            res1 = self.client.place_limit_order(self.market, "sell", q1, p1)
                            self.active_ask_id = res1.get("id")
                            self.active_ask_p = float(res1.get("price", p1))
                            self.active_ask_q = float(res1.get("origin_amount", q1))
                            self.q_ahead = -1.0
                            res2 = self.client.place_limit_order(self.market, "sell", q2, p2)
                            self.slot2_ask_id = res2.get("id")
                            self.slot2_ask_p = float(res2.get("price", p2))
                            self.slot2_ask_q = float(res2.get("origin_amount", q2))
                            self.split_done_for_pos = True
                            self._save_wallet()
                            print(
                                f"  📤 [LIVE 2-SLOT SELL] Slot1(id={self.active_ask_id}): {self.active_ask_q:.4f} PRL @ ${self.active_ask_p:.2f} | "
                                f"Slot2(id={self.slot2_ask_id}, w2={eff_w2:.0%}): {self.slot2_ask_q:.4f} PRL @ ${self.slot2_ask_p:.2f}"
                            )
                            return
                        except Exception as e:
                            print(f"  ⚠️ [2-SLOT SELL ERROR]: {e}")
                            self.sync_account_balances(b1)
                elif self.active_ask_id is None and w2 < 0.15:
                    act = 4

            if act == 6:
                target_p = max(self.tick_size, b1 - self.tick_size)
            elif act == 3:
                target_p = max(a1 + self.tick_size, wave_p)
            else:
                if self.active_ask_id is not None and abs(self.active_ask_p - floor_p) <= tol and b1 < floor_p - tol:
                    return
                target_p = max(a1, floor_p)
            target_p = round(target_p / self.tick_size) * self.tick_size
            if self.active_ask_id is not None and self.slot2_ask_id is None and abs(self.active_ask_p - target_p) <= tol:
                return
            for oid_attr in ("active_ask_id", "slot2_ask_id"):
                oid = getattr(self, oid_attr)
                if oid is not None:
                    try:
                        self.client.cancel_order(oid)
                    except Exception as e:
                        print(f"  ⚠️ [CANCEL SELL WARN] id={oid}: {e}")
            self.sync_live_orders(b1)
            self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
            self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
            if self.pos_qty < 2.0:
                return
            self.sync_account_balances(b1)
            sell_qty = math.floor(min(self.pos_qty, self.ex_coin_free) * 10000.0) / 10000.0
            if sell_qty < 2.0:
                return
            try:
                res = self.client.place_limit_order(self.market, "sell", sell_qty, target_p)
                self.active_ask_id = res.get("id")
                self.active_ask_p = float(res.get("price", target_p))
                self.active_ask_q = float(res.get("origin_amount", sell_qty))
                self.q_ahead = -1.0
                self._save_wallet()
                print(f"  📤 [LIVE ORDER SELL] id={self.active_ask_id} | {self.active_ask_q:.4f} PRL @ ${self.active_ask_p:.2f}")
            except Exception as e:
                print(f"  ⚠️ [ORDER SELL ERROR] qty={sell_qty:.4f} @ ${target_p:.2f}: {e}")
                self.sync_account_balances(b1)


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
    raw_d_lines = []
    if local_depth.exists():
        raw_d_lines = subprocess.run(["tail", "-n", "14400", str(local_depth)], capture_output=True, text=True).stdout.splitlines()
    elif trader.client and trader.client._ssh is not None:
        try:
            _, out, _ = trader.client._ssh.exec_command("tail -n 14400 /opt/crypto_bot/data/realtime/prlusdt/depth.csv")
            raw_d_lines = out.read().decode("utf-8", errors="ignore").splitlines()
        except Exception:
            pass
    sec_bp = []
    for line in raw_d_lines:
        parts = line.split(",")
        if len(parts) > 2:
            try:
                ts_v = float(parts[0])
                sec_i = int(ts_v // 1000) if ts_v > 1e11 else int(ts_v)
                bp = float(parts[1])
                if sec_i > 1000 and bp > 0:
                    sec_bp.append((sec_i, bp))
            except ValueError:
                pass
    if sec_bp:
        trader.bid_hist_1h.clear()
        end_s = sec_bp[-1][0]
        cur_bp = sec_bp[0][1]
        idx_p = 0
        for s in range(end_s - 14399, end_s + 1):
            while idx_p < len(sec_bp) and sec_bp[idx_p][0] <= s:
                cur_bp = sec_bp[idx_p][1]
                idx_p += 1
            trader.bid_hist_1h.append(cur_bp)

    from _tools.live_feature_updater import LiveFeatureStreamer
    streamer = LiveFeatureStreamer(df_sub, trader.feature_names, coin=coin)
    df_init = streamer.get_recent_window(60)
    X_tail = df_init.reindex(columns=trader.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)
    X_buf = deque(X_tail, maxlen=60)
    last_row = df_init.iloc[-1:].copy()
    for w_i in range(max(0, len(X_tail) - 15), len(X_tail)):
        res12 = trader.routers[1][2].predict_live_tick(X_tail[:w_i + 1], df_init.iloc[w_i:w_i + 1])
        trader.ez12_hist.append(float(res12["expected_z"]))
    print(f"✅ Прогрет 1-часовой буфер стакана ({len(trader.bid_hist_1h)} сек, диапазон Bid=[${min(trader.bid_hist_1h):.2f}..${max(trader.bid_hist_1h):.2f}]), потоковый банк 299 признаков и история инерции ({len(trader.ez12_hist)} сек)")

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

        if local_depth.exists() and (time.time() - local_depth.stat().st_mtime) < 2.5:
            depth = None
        else:
            depth = trader.client.get_depth(trader.market, limit=50)
        last_row = streamer.step(depth, now_utc)

        b1 = float(last_row["ST_best_bid"].iloc[0])
        a1 = float(last_row["ST_best_ask"].iloc[0])
        if trader.pos_qty > 1e-4:
            trader.hold_sec += 1
            if b1 > 0:
                trader.peak_bid = max(trader.peak_bid, trader.avg_entry, b1)
        if step_idx % 3 == 1:
            trader.sync_live_orders(b1, count_hold=False)
        if step_idx % 15 == 1:
            trader.sync_account_balances(b1)

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
    """Онлайн-запрос статуса фонового демона на VPS (81.19.137.18), ресурсов сервера, субсчета и хвоста лога."""
    from _tools.vps_config import create_ssh_client
    print("⏳ Запрос статуса демона на VPS (81.19.137.18)...", flush=True)
    ssh = create_ssh_client(timeout=12.0)
    cmd = (
        "PID=$(cat /opt/crypto_bot/data/logs/live_rl_trader.pid 2>/dev/null); "
        "if [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null; then "
        "  echo \"RUNNING:$PID\"; ps -p \"$PID\" -o pcpu=,pmem=,rss=,etime= 2>/dev/null; "
        "else echo \"STOPPED\"; echo \"\"; fi; "
        "echo '---WALLET---'; cat /opt/crypto_bot/models/prl/rl_execution/live_wallet_state.json 2>/dev/null || echo '{}'; "
        f"echo '---LOG---'; tail -n {int(tail_lines)} /opt/crypto_bot/data/logs/live_rl_trader.log 2>/dev/null || true"
    )
    _, out, _ = ssh.exec_command(cmd, timeout=12.0)
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

    print(f"📡 Статус демона на VPS (81.19.137.18): {'🟢 РАБОТАЕТ НА СЕРВЕРЕ (PID=' + str(pid) + ')' + res_info if running else '🔴 ОСТАНОВЛЕН'}", flush=True)
    try:
        w = json.loads(w_part.strip() or "{}")
        if w:
            (BASE_DIR / "models" / "prl" / "rl_execution" / "live_wallet_state.json").write_text(
                json.dumps(w, indent=2), encoding="utf-8"
            )
            print(
                f"💼 Счет бота: Кэш=${w.get('bot_cash_usd', 0.0):.2f} | Позиция={w.get('pos_qty', 0.0):.4f} PRL "
                f"(вход=${w.get('avg_entry', 0.0):.2f}) | Реализ. PnL={w.get('realized_pnl_usd', 0.0):+.4f}$ | "
                f"Сделок={w.get('n_trades', 0)}",
                flush=True,
            )
    except Exception:
        pass
    print(f"\n📜 Онлайн-лог с сервера VPS (/opt/crypto_bot/data/logs/live_rl_trader.log, последние {tail_lines} строк):", flush=True)
    print("-" * 100, flush=True)
    print(log_part.strip(), flush=True)


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
    """Останавливает фоновый демон бота (и локально, и на VPS) и при необходимости снимает активные ордера бота на SafeTrade."""
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
        w_file = BASE_DIR / "models" / "prl" / "rl_execution" / "live_wallet_state.json"
        bot_oids = set()
        if w_file.exists():
            try:
                w_st = json.loads(w_file.read_text(encoding="utf-8"))
                for k in ("active_bid_id", "active_ask_id"):
                    if w_st.get(k):
                        bot_oids.add(w_st[k])
            except Exception:
                pass
        for oid in bot_oids:
            try:
                client.cancel_order(oid)
                print(f"✖️ Снят активный ордер бота id={oid}")
            except Exception:
                pass
        print("✅ Открытые заявки бота проверены и сняты (ручные ордера пользователя не затронуты).")


def sync_and_inspect_telemetry(last_n: int = 20) -> pd.DataFrame:
    """Молниеносная онлайн-синхронизация горячей телеметрии RL с VPS (81.19.137.18) через gzip-поток (~1 сек вместо 85 МБ SFTP)."""
    import io
    import gzip
    from _tools.vps_config import create_ssh_client
    hot_dir = BASE_DIR / "data" / "vps_hot"
    proc_dir = BASE_DIR / "data" / "processed" / "prl"
    hot_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    local_csv = hot_dir / "prl_rl_telemetry_hot.csv"
    pq_file = proc_dir / "rl_telemetry.parquet"

    print("⏳ Сжатая выгрузка горячей телеметрии с VPS (81.19.137.18)...", flush=True)
    ssh = create_ssh_client(timeout=12.0)
    _, out, _ = ssh.exec_command("gzip -1 -c /opt/crypto_bot/data/realtime/prlusdt/rl_telemetry.csv", timeout=20.0)
    gz_bytes = out.read()
    ssh.close()

    raw_bytes = gzip.decompress(gz_bytes)
    local_csv.write_bytes(raw_bytes)
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.to_parquet(pq_file, index=False)
    print(f"✅ Онлайн-синхронизация с VPS (81.19.137.18): {len(df):,} тиков ({len(gz_bytes)/1024:.0f} КБ gzip) -> {pq_file.relative_to(BASE_DIR)}", flush=True)
    if len(df) > 0:
        print(f"⏱️ Интервал: {df['ts_msk'].iloc[0]} MSK ➔ {df['ts_msk'].iloc[-1]} MSK | Средняя задержка на VPS: {df['latency_ms'].tail(100).mean():.1f} мс", flush=True)
        print("📊 Распределение действий агента:", df["act_name"].value_counts().to_dict(), flush=True)
    cols_show = [c for c in ["ts_msk", "step", "b1", "a1", "act_name", "v_pred", "pos_qty", "bot_cash_usd", "equity_usd", "latency_ms"] if c in df.columns]
    return df[cols_show].tail(last_n)


def start_live_daemon(budget_usd: float = 54.0, order_usd: float = 54.0, steps: int = 86400, push_wallet: bool = False):
    """Запускает / перезапускает автономный фоновый демон бота непосредственно на VPS (81.19.137.18) с авто-синхронизацией кода и .env."""
    stop_live_daemon(cancel_open_orders=False)
    from _tools.vps_config import create_ssh_client
    ssh = create_ssh_client(timeout=12.0)
    try:
        sftp = ssh.open_sftp()
        env_local = BASE_DIR / ".env"
        if env_local.exists():
            sftp.put(str(env_local), "/opt/crypto_bot/.env")
        for mod_name in ("live_rl_trader.py", "live_feature_updater.py", "microstructure_features.py", "swarm_router.py", "fast_neural_engine.py", "train_rl_execution.py"):
            loc_mod = BASE_DIR / "_tools" / mod_name
            if loc_mod.exists():
                sftp.put(str(loc_mod), f"/opt/crypto_bot/_tools/{mod_name}")
        ppo_local = BASE_DIR / "models" / "prl" / "rl_execution" / "best_maskable_ppo.pt"
        if ppo_local.exists():
            sftp.put(str(ppo_local), "/opt/crypto_bot/models/prl/rl_execution/best_maskable_ppo.pt")
        for pat in ("h*s_k*", "*_h*s"):
            for h_dir in (BASE_DIR / "models" / "prl").glob(pat):
                if not h_dir.is_dir():
                    continue
                for root, _, files in os.walk(h_dir):
                    rel_r = Path(root).relative_to(BASE_DIR / "models" / "prl")
                    rem_d = f"/opt/crypto_bot/models/prl/{rel_r}"
                    try:
                        rem_attrs = {a.filename: a.st_size for a in sftp.listdir_attr(rem_d)}
                    except Exception:
                        ssh.exec_command(f"mkdir -p {rem_d}")[1].read()
                        rem_attrs = {}
                    for fn in files:
                        lf = Path(root) / fn
                        if rem_attrs.get(fn) == lf.stat().st_size and not fn.endswith(".json"):
                            continue
                        sftp.put(str(lf), f"{rem_d}/{fn}")
        w_local = BASE_DIR / "models" / "prl" / "rl_execution" / "live_wallet_state.json"
        if push_wallet and w_local.exists():
            sftp.put(str(w_local), "/opt/crypto_bot/models/prl/rl_execution/live_wallet_state.json")
        sftp.close()
    except Exception as e:
        print(f"⚠️ SFTP sync warning: {e}")
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
