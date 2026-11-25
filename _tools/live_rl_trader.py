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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from datetime import timedelta as _td_msk
from pathlib import Path
import numpy as np
import pandas as pd


def _msk_ms() -> str:
    """Точная метка времени MSK с миллисекундами для строк исполнения/отмен (сверка live с симулятором по секундам)."""
    return datetime.now(timezone(_td_msk(hours=3))).strftime("%H:%M:%S.%f")[:-3]
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from _tools.system_resources import SAFE_WORKERS
from _tools.vps_config import SAFETRADE_PROXY


def _load_dotenv():
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        parsed = {}
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                parsed[k.strip()] = v.strip().strip('"').strip("'")
        for k, v in parsed.items():
            os.environ.setdefault(k, v)


class SafeTradePrivateClient:
    """Клиент приватного и публичного REST API SafeTrade (Peatio v2 HMAC-SHA256) с поддержкой SSH-моста через белый IP 81.19.137.18."""

    BASE_URL = "https://safe.trade/api/v2"

    def __init__(self, api_key: str = None, api_secret: str = None, use_proxy: bool = True, use_vps_bridge: bool = True):
        import threading
        _load_dotenv()
        self.api_key = api_key or os.environ.get("SAFETRADE_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("SAFETRADE_API_SECRET", "")
        self.proxies = {"http": SAFETRADE_PROXY, "https": SAFETRADE_PROXY} if (use_proxy and SAFETRADE_PROXY) else None
        from curl_cffi import requests as cffi_requests, CurlOpt
        import queue
        self._cffi_requests = cffi_requests
        self._CurlOpt = CurlOpt
        self.session = cffi_requests.Session(
            impersonate="chrome120",
            curl_options={CurlOpt.IPRESOLVE: 1},
        )
        self._req_lock = threading.Lock()
        self._last_nonce = int(time.time() * 1000)
        self._last_dispatch_ts = 0.0
        self.use_vps_bridge = use_vps_bridge
        self._ssh = None
        self._bridge_in = None
        self._bridge_out = None
        self._sess_pool = queue.Queue()
        if self.use_vps_bridge:
            self._init_vps_bridge()
        else:
            def _make_warm_sess(_):
                s = cffi_requests.Session(
                    impersonate="chrome120",
                    curl_options={CurlOpt.IPRESOLVE: 1},
                )
                try:
                    s.get(f"{self.BASE_URL}/trade/public/markets/pearlusdt/depth?limit=1", timeout=4)
                except Exception:
                    pass
                return s
            with ThreadPoolExecutor(max_workers=6) as wp:
                for ws in wp.map(_make_warm_sess, range(6)):
                    self._sess_pool.put(ws)

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
            "last_n = int(time.time() * 1000)\n"
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
            "    try:\n"
            "        for att in range(3):\n"
            "            last_n = max(int(time.time() * 1000), last_n + 1)\n"
            "            n = str(last_n)\n"
            "            sig = hmac.new(S.encode(), (n + K).encode(), hashlib.sha256).hexdigest()\n"
            "            hdr = {'X-Auth-Apikey': K, 'X-Auth-Nonce': n, 'X-Auth-Signature': sig, 'Content-Type': 'application/json'}\n"
            "            if req['method'] == 'GET':\n"
            "                r = sess.get(req['url'], headers=hdr if req.get('auth', True) else None, timeout=6)\n"
            "            else:\n"
            "                body = json.dumps(req['body']) if req.get('body') is not None else None\n"
            "                r = sess.post(req['url'], headers=hdr, data=body, timeout=6)\n"
            "            if r.status_code in (401, 403, 429) and att < 2:\n"
            "                time.sleep(0.25 * (att + 1))\n"
            "                last_n = max(int(time.time() * 1000), last_n + 5)\n"
            "                continue\n"
            "            break\n"
            "        sys.stdout.write(json.dumps({'status': r.status_code, 'text': r.text}) + '\\n')\n"
            "    except Exception as e:\n"
            "        sys.stdout.write(json.dumps({'status': 599, 'text': str(e)}) + '\\n')\n"
            "    sys.stdout.flush()\n"
        )
        import base64
        if self._ssh is not None:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None
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
            try:
                tel_path = Path("/opt/crypto_bot/data/realtime/prlusdt/rl_telemetry.csv")
                tel_path.parent.mkdir(parents=True, exist_ok=True)
                need_hdr = not tel_path.exists() or tel_path.stat().st_size == 0
                with open(tel_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if need_hdr:
                        w.writerow(cols)
                    w.writerow([row.get(c, "") for c in cols])
            except OSError:
                pass
            return
        with self._req_lock:
            try:
                self._bridge_in.write(json.dumps({"method": "TELEMETRY", "body": row}) + "\n")
                self._bridge_in.flush()
                self._bridge_out.readline()
            except Exception:
                pass

    def _private_req(self, method: str, path: str, body: dict = None, auth: bool = True) -> dict | list:
        url = f"{self.BASE_URL}{path}"
        if self.use_vps_bridge:
            with self._req_lock:
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
        data_str = json.dumps(body) if body else None
        sess = self._sess_pool.get()
        try:
            for attempt in range(3):
                with self._req_lock:
                    dt_disp = time.perf_counter() - self._last_dispatch_ts
                    if dt_disp < 0.010:
                        time.sleep(0.010 - dt_disp)
                    hdrs = self._auth_headers() if auth else None
                    self._last_dispatch_ts = time.perf_counter()
                if method == "GET":
                    r = sess.get(url, headers=hdrs, timeout=6)
                else:
                    r = sess.post(url, headers=hdrs, data=data_str, timeout=6)
                if r.status_code in (401, 403, 429) and attempt < 2:
                    time.sleep(0.12 * (attempt + 1))
                    with self._req_lock:
                        self._last_nonce = max(int(time.time() * 1000), self._last_nonce + 5)
                    continue
                break
        finally:
            self._sess_pool.put(sess)
        if r.status_code >= 400:
            raise RuntimeError(f"SafeTrade HTTP {r.status_code}: {r.text}")
        return r.json() if r.text else {}

    def _auth_headers(self) -> dict:
        now_ms = int(time.time() * 1000)
        self._last_nonce = max(now_ms, self._last_nonce + 1)
        nonce = str(self._last_nonce)
        msg = f"{nonce}{self.api_key}".encode("utf-8")
        sig = hmac.new(self.api_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
        return {
            "X-Auth-Apikey": self.api_key,
            "X-Auth-Nonce": nonce,
            "X-Auth-Signature": sig,
            "Content-Type": "application/json",
        }

    def get_depth(self, market: str = "pearlusdt", limit: int = 100) -> dict:
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

    def place_limit_order(
        self,
        market: str,
        side: str,
        volume: float,
        price: float,
        price_precision: int | None = None,
        amount_precision: int = 4,
        min_amount: float = 2.0,
    ) -> dict:
        if price_precision is None:
            price_precision = 4 if market.lower() == "pearlusdt" else 2
        prec = max(0, min(8, int(price_precision)))
        aprec = max(0, min(8, int(amount_precision)))
        payload = {
            "market": market,
            "side": side.lower(),
            "amount": f"{max(float(min_amount), float(volume)):.{aprec}f}",
            "price": f"{float(price):.{prec}f}",
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
        # Эталонный шаг центовых признаков (1 unit = $0.01 для PRL, 100% инвариантность входов MultiTaskLOBNet / Swarm / 480D)
        self.tick_size = 0.01 if self.coin == "prl" else 0.05
        # Реальный биржевой шаг дискретизации цены (например, $0.0001 / price_precision=4 на SafeTrade pearlusdt)
        self.price_precision = 4 if self.coin == "prl" else 2
        self.amount_precision = 4
        self.min_order_amount = 2.0
        self.exec_tick_size = round(10.0 ** (-self.price_precision), 8) if self.coin == "prl" else self.tick_size
        self._refresh_market_precision(verbose=True)

        import torch
        from _tools.swarm_router import SwarmRegimeRouter
        from _tools.train_rl_execution import MaskableActorCritic
        print(f"🧠 Инициализация боевого ядра LiveTriadRLTrader ({self.coin.upper()}, режим={self.mode.upper()}, бюджет=${self.initial_budget_usd:.2f})...")
        self._triad_pool = ThreadPoolExecutor(max_workers=6)
        self._io_pool = ThreadPoolExecutor(max_workers=6)
        self.routers = [
            (5, 3.0, SwarmRegimeRouter(self.coin, 5, 3.0, temperature=0.25, use_neural=use_neural, neural_weight=0.15)),
            (12, 3.5, SwarmRegimeRouter(self.coin, 12, 3.5, temperature=0.25, use_neural=use_neural, neural_weight=0.15)),
            (25, 3.0, SwarmRegimeRouter(self.coin, 25, 3.0, temperature=0.25, use_neural=use_neural, neural_weight=0.15)),
        ]
        try:
            self.router_60 = SwarmRegimeRouter(self.coin, 60, 3.0, temperature=0.25, use_neural=use_neural, neural_weight=0.15)
        except Exception as e:
            print(f"ℹ️ Router 60s пропущен: {e}")
            self.router_60 = None
        try:
            self.router_90 = SwarmRegimeRouter(self.coin, 90, 3.0, temperature=0.25, use_neural=use_neural, neural_weight=0.15)
        except Exception:
            self.router_90 = None
        try:
            self.router_120 = SwarmRegimeRouter(self.coin, 120, 3.0, temperature=0.25, use_neural=use_neural, neural_weight=0.15)
        except Exception:
            self.router_120 = None
        if use_neural:
            list(self._triad_pool.map(lambda r_item: r_item[2].preload_neural_models(), self.routers))
            if self.router_60 is not None:
                self.router_60.preload_neural_models()
            if self.router_90 is not None:
                self.router_90.preload_neural_models()
            if self.router_120 is not None:
                self.router_120.preload_neural_models()
            torch.set_num_threads(2)
            print(f"⚡ Все активные нейросетевые модели (FastKerasModel, пул={SAFE_WORKERS} потоков) предзагружены в RAM (0 обращений к диску на тиках).")
        self.feature_names = self.routers[0][2].feature_names

        ckpt_path = BASE_DIR / "models" / self.coin / "rl_execution" / "best_maskable_ppo.pt"
        if ckpt_path.exists():
            try:
                sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
                h_dim = sd["backbone.0.weight"].shape[0]
                obs_dim = sd["backbone.0.weight"].shape[1]
                self.agent = MaskableActorCritic(obs_dim=obs_dim, n_actions=7, hidden_dim=h_dim)
                self.agent.load_state_dict(sd)
                self.agent.eval()
                print(f"✅ Загружен чекпоинт RL-агента: {ckpt_path.name} (obs_dim={obs_dim}, hidden={h_dim})")
            except Exception as e_ppo:
                print(f"⚠️ [LEGACY PPO LOAD WARN]: {e_ppo}")
                self.agent = None
        else:
            self.agent = None

        # Подключаем 24-слотового боевого Чемпиона Урока 7 (release_champion_v1.pt: MultiTaskLOBNet + 324D->256->26)
        self.l7_backbone = None
        self.l7_h_state = None
        self.l7_w1 = None
        self.l7_b1 = None
        self.l7_wg = None
        self.l7_bg = None
        self.l7_w2 = None
        self.l7_b2 = None
        self.v7_hist_5s = deque(maxlen=6)
        self.st_trade_5s = deque(maxlen=5)
        self.bo_trade_5s = deque(maxlen=5)
        self.cv_hist_state = {}
        self.cv_first_order_net = None
        self.l7_use_cross_venue = False
        self.last_l7_out = None
        self.last_l7_target_exp_frac = 1.0
        self.rel_ckpt_path = BASE_DIR / "models" / self.coin / "release_champion_v1.pt"
        self.rel_ckpt_mtime = 0.0
        cv_fo_path = BASE_DIR / "models" / self.coin / "cross_venue_first_order" / "cross_venue_first_order_v1.pt"
        if cv_fo_path.exists():
            try:
                from _tools.cross_venue_first_order_net import CrossVenueFirstOrderNet
                cv_ck = torch.load(cv_fo_path, map_location="cpu", weights_only=False)
                self.cv_first_order_net = CrossVenueFirstOrderNet(
                    in_dim=int(cv_ck.get("in_dim", 28)),
                    h_dim=int(cv_ck.get("h_dim", 128)),
                    out_dim=int(cv_ck.get("out_dim", 6)),
                    isolated_ln=bool(cv_ck.get("isolated_ln", False)),
                ).eval()
                self.cv_first_order_net.load_state_dict(cv_ck["state_dict"], strict=True)
                cv_auc = float(cv_ck.get("blind_7h_4venue_auc", cv_ck.get("oos_mean_auc", 0.0)))
                print(
                    f"🌐 Подключена кросс-биржевая модель 1-го порядка {cv_fo_path.name} "
                    f"(in_dim={cv_ck.get('in_dim', 28)}, isolated_ln={bool(cv_ck.get('isolated_ln', False))}, AUC={cv_auc:.4f})!"
                )
            except Exception as e_cv:
                print(f"⚠️ Не удалось загрузить {cv_fo_path.name}: {e_cv}")
        self.bo_swarm_routers = []
        self.bo_feature_names = []
        self.l7_w_reg1 = None
        self.l7_b_reg1 = None
        self.l7_b_reg1_b = None
        self.l7_b_reg1_a = None
        self.l7_w_reg2 = None
        self.l7_b_reg2 = None
        self.l7_b_reg2_b = None
        self.l7_b_reg2_a = None
        self.l7_w2_moe = None
        self.l7_b2_moe = None
        self.l7_w_gate1 = None
        self.l7_b_gate1 = None
        self.l7_w_gate2 = None
        self.l7_b_gate2 = None
        self.l25_regulator_mode = False
        self.l26_endogenous_mode = False
        self.l27_multimodal_mode = False
        self.l28_hierarchical_mode = False
        self.last_l25_state = None
        self.last_l26_state = None
        self.last_l27_state = None
        self.last_l28_state = None
        self.l26_api_tokens = 12.0
        self.l27_api_tokens = 10.0
        self.l28_api_tokens = 10.0
        self.st_tree_router = None
        self.bo_tree_router = None
        self.st_x_ring = deque(maxlen=360)
        self.bo_x_ring = deque(maxlen=360)
        self.l28_vel8_hist = deque(maxlen=6)
        try:
            from _tools.build_bo_swarm_36d_cache import CausalTop2BigONERouter, HORIZONS as BO_HORIZONS
            if (BASE_DIR / "models" / "prl_bo").exists():
                self.bo_swarm_routers = [CausalTop2BigONERouter(H, K, live_only_top2=True) for H, K in BO_HORIZONS]
                if self.bo_swarm_routers and self.bo_swarm_routers[0].feature_names:
                    self.bo_feature_names = list(self.bo_swarm_routers[0].feature_names)
                n_bo_b = sum(len(r_bo.boosters) for r_bo in self.bo_swarm_routers)
                if n_bo_b > 0:
                    print(f"🐝 Подключен каузальный рой 1-го порядка BigONE (6 горизонтов, {n_bo_b} Top-2 LightGBM бустеров -> +36D)!")
        except Exception as e_bo:
            print(f"⚠️ Рой BigONE prl_bo не подключен: {e_bo}")
        if self.rel_ckpt_path.exists():
            self.maybe_hot_reload_champion(force=True)

        self.carousel_net = None
        self.hazard_net = None
        self.realism_net = None
        self.regime_gate = None
        self.guard_predictor = None
        if not bool(getattr(self, "order3_mode", False)):
            from _tools.carousel_net import CarouselPredictor
            from _tools.hazard_net import HazardPredictor
            from _tools.execution_realism_net import ExecutionRealismPredictor
            from _tools.microstructure_regime_gate import RegimeGatePredictor, REGIME_GATE_MODEL_PATH
            from _tools.multi_horizon_guard_net import MultiHorizonGuardPredictor, GUARD_ENSEMBLE_DIR
            try:
                self.carousel_net = CarouselPredictor(BASE_DIR / "models" / self.coin / "rl_execution" / "carousel_net.pt")
                self.hazard_net = HazardPredictor(BASE_DIR / "models" / self.coin / "rl_execution" / "hazard_net.pt")
                self.realism_net = ExecutionRealismPredictor(BASE_DIR / "models" / self.coin / "rl_execution" / "execution_realism_net.pt")
                self.regime_gate = RegimeGatePredictor(REGIME_GATE_MODEL_PATH)
                self.guard_predictor = MultiHorizonGuardPredictor(GUARD_ENSEMBLE_DIR)
            except Exception as e_leg:
                print(f"⚠️ Legacy auxiliary predictors skipped: {e_leg}")
        self.last_guard_probs = np.array([0.05, 0.05, 0.02, 0.05], dtype=np.float32)
        self.last_regime_probs = np.array([0.9, 0.05, 0.0, 0.05], dtype=np.float32)
        self.last_car_probs = np.full(5, 0.5, dtype=np.float32)
        self.last_car_cap_prl = 120.0
        self.last_p_adverse = 0.0
        self.last_p_exhaust = 0.0
        self.last_p_fill_real = 0.5
        self.last_p_frontrun = 0.2
        self.last_ez90 = 0.0
        self.last_ez120 = 0.0

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
        self.max_slots = 24
        self.extra_slots = []
        self.bid_slots = [{"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0} for s in range(12)]
        self.ask_slots = [{"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0} for s in range(12)]
        self.tr_low_px = 0.0
        self.tr_low_qty = 0.0
        self.tr_mid_px = 0.0
        self.tr_mid_qty = 0.0
        self.tr_high_px = 0.0
        self.tr_high_qty = 0.0
        self.last_buy_fill_ts = 0.0
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
        st = {}
        if self.wallet_file.exists():
            try:
                st = json.loads(self.wallet_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"⚠️ Ошибка чтения {self.wallet_file.name}: {e}. Пробуем .bak резервную копию...")
                bak_file = self.wallet_file.with_suffix(".json.bak")
                if bak_file.exists():
                    try:
                        st = json.loads(bak_file.read_text(encoding="utf-8"))
                        print("✅ Успешно восстановлено состояние из резервной копии .bak")
                    except Exception as e2:
                        print(f"❌ Резервная копия .bak также повреждена: {e2}")
        if st:
            self.initial_budget_usd = float(st.get("initial_budget_usd", self.initial_budget_usd))
            self.bot_cash_usd = float(st.get("bot_cash_usd", self.initial_budget_usd))
            self.pos_qty = float(st.get("pos_qty", 0.0))
            self.avg_entry = float(st.get("avg_entry", 0.0))
            self.realized_pnl_usd = float(st.get("realized_pnl_usd", 0.0))
            self.net_deposits_usd = float(st.get("net_deposits_usd", 0.0))
            if not bool(st.get("base_1023_applied", False)):
                self.initial_budget_usd = 1023.0
                self.net_deposits_usd = 1023.0
            self.base_1023_applied = True
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
            self.extra_slots = list(st.get("extra_slots", []))
            self.tr_low_px = float(st.get("tr_low_px", 0.0))
            self.tr_low_qty = float(st.get("tr_low_qty", 0.0))
            self.tr_mid_px = float(st.get("tr_mid_px", 0.0))
            self.tr_mid_qty = float(st.get("tr_mid_qty", 0.0))
            self.tr_high_px = float(st.get("tr_high_px", self.avg_entry if self.pos_qty >= 1.95 else 0.0))
            self.tr_high_qty = float(st.get("tr_high_qty", self.pos_qty if self.pos_qty >= 1.95 else 0.0))
            self._reconcile_lifo_tranches()
            raw_b_slots = st.get("bid_slots", [])
            if isinstance(raw_b_slots, list) and len(raw_b_slots) == 12:
                for s_i in range(12):
                    item = raw_b_slots[s_i] if isinstance(raw_b_slots[s_i], dict) else {}
                    self.bid_slots[s_i] = {
                        "slot": s_i,
                        "id": item.get("id"),
                        "px": float(item.get("px", 0.0)),
                        "qty": float(item.get("qty", 0.0)),
                        "act_ts": float(item.get("act_ts", 0.0)),
                        "acc_fill": float(item.get("acc_fill", 0.0)),
                    }
            elif self.active_bid_id is not None:
                self.bid_slots[2] = {
                    "slot": 2,
                    "id": self.active_bid_id,
                    "px": float(self.active_bid_p),
                    "qty": float(self.active_bid_q),
                    "act_ts": time.time(),
                    "acc_fill": 0.0,
                }
            raw_a_slots = st.get("ask_slots", [])
            if isinstance(raw_a_slots, list) and len(raw_a_slots) == 12:
                for s_i in range(12):
                    item = raw_a_slots[s_i] if isinstance(raw_a_slots[s_i], dict) else {}
                    self.ask_slots[s_i] = {
                        "slot": s_i,
                        "id": item.get("id"),
                        "px": float(item.get("px", 0.0)),
                        "qty": float(item.get("qty", 0.0)),
                        "act_ts": float(item.get("act_ts", 0.0)),
                        "acc_fill": float(item.get("acc_fill", 0.0)),
                    }
            else:
                if self.active_ask_id is not None:
                    self.ask_slots[2] = {
                        "slot": 2,
                        "id": self.active_ask_id,
                        "px": float(self.active_ask_p),
                        "qty": float(self.active_ask_q),
                        "act_ts": time.time(),
                        "acc_fill": 0.0,
                    }
                if self.slot2_ask_id is not None:
                    self.ask_slots[4] = {
                        "slot": 4,
                        "id": self.slot2_ask_id,
                        "px": float(self.slot2_ask_p),
                        "qty": float(self.slot2_ask_q),
                        "act_ts": time.time(),
                        "acc_fill": 0.0,
                    }
            self.last_buy_fill_ts = float(st.get("last_buy_fill_ts", 0.0))
            self.split_done_for_pos = bool(st.get("split_done_for_pos", False))
            self.hold_sec = int(st.get("hold_sec", 120 if self.pos_qty >= 2.0 else 0))
            self.car_state = int(st.get("car_state", 0))
            self.car_sell_px = float(st.get("car_sell_px", 0.0))
            self.car_cycles = int(st.get("car_cycles", 0))
            self.cooldown_until_ts = float(st.get("cooldown_until_ts", 0.0))
            self.last_exit_ts = float(st.get("last_exit_ts", 0.0))
            self.last_exit_price = float(st.get("last_exit_price", 0.0))
            if self.pos_qty < 2.0:
                self.car_state = 0
            elif self.car_state == 1 and self.slot2_ask_id is None:
                self.car_state = 0
            elif self.car_state == 2 and self.active_bid_id is None and self.bot_cash_usd < 3.5:
                self.car_state = 0
        cur_key_fp = (
            hashlib.sha256(self.client.api_key.encode("utf-8")).hexdigest()[:12]
            if (self.client is not None and getattr(self.client, "api_key", ""))
            else ""
        )
        saved_key_fp = str(st.get("api_key_fp", "")).strip() if st else ""
        if saved_key_fp and cur_key_fp and saved_key_fp != cur_key_fp:
            raise RuntimeError(
                f"🚨 ACCOUNT MISMATCH GUARD: Текущий SAFETRADE_API_KEY (fp={cur_key_fp}) "
                f"не совпадает с привязанным к кошельку бота (fp={saved_key_fp})! Старт заблокирован."
            )
        self.api_key_fp = cur_key_fp or saved_key_fp
        self.bid_age_sec = 0
        self.bid_partial_filled = 0.0
        self.adverse_streak = 0
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

    def _refresh_market_precision(self, verbose: bool = False) -> None:
        """Автоопределение биржевого шага цены (price_precision / exec_tick_size) из exchange_rules.json без искажения признакового tick_size."""
        rules_path = BASE_DIR / "data" / "metadata" / "exchange_rules.json"
        if not rules_path.exists():
            return
        try:
            mtime = float(rules_path.stat().st_mtime)
            if not verbose and mtime <= float(getattr(self, "_rules_mtime", 0.0)) + 0.5:
                return
            with open(rules_path, "r", encoding="utf-8") as rf:
                rules_doc = json.load(rf)
            self._rules_mtime = mtime
            st_mkt = (rules_doc.get("safetrade") or {}).get(self.market) or {}
            if st_mkt and str(st_mkt.get("state", "enabled")).lower() == "enabled":
                new_pp = int(st_mkt.get("price_precision", self.price_precision))
                new_ap = int(st_mkt.get("amount_precision", getattr(self, "amount_precision", 4)))
                new_min = float(st_mkt.get("min_amount", getattr(self, "min_order_amount", 2.0)) or 2.0)
                if 1 <= new_pp <= 8:
                    old_exec = getattr(self, "exec_tick_size", self.tick_size)
                    self.price_precision = new_pp
                    self.amount_precision = max(0, min(8, new_ap))
                    self.min_order_amount = max(0.01, new_min)
                    self.exec_tick_size = round(10.0 ** (-new_pp), 8)
                    if verbose or abs(self.exec_tick_size - old_exec) > 1e-9:
                        print(
                            f"📐 [MARKET RULES] {self.market}: price_precision={self.price_precision} "
                            f"(exec_tick=${self.exec_tick_size:.{self.price_precision}f}, feat_tick=${self.tick_size:.2f}, "
                            f"amount_prec={self.amount_precision}, min_amount={self.min_order_amount})"
                        )
        except Exception:
            pass

    def maybe_hot_reload_champion(self, force: bool = False):
        """Бесшовная горячая подгрузка обновленного чекпоинта release_champion_v1.pt на лету без рестарта и без снятия 24 слотов."""
        self._refresh_market_precision(verbose=False)
        ckpt_p = getattr(self, "rel_ckpt_path", None)
        if ckpt_p is None or not ckpt_p.exists():
            return
        try:
            cur_mtime = float(ckpt_p.stat().st_mtime)
            if not force and cur_mtime <= float(getattr(self, "rel_ckpt_mtime", 0.0)) + 0.5:
                return
            import torch
            rel_ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            if str(rel_ck.get("arch_version", "")).startswith("order3"):
                from _tools.order3_live_bridge import Order3LiveInferenceEngine
                if not getattr(self, "bo_swarm_routers", None) and (BASE_DIR / "models" / "prl_bo").exists():
                    try:
                        from _tools.build_bo_swarm_36d_cache import CausalTop2BigONERouter, HORIZONS as BO_HORIZONS
                        self.bo_swarm_routers = [CausalTop2BigONERouter(H, K, live_only_top2=True) for H, K in BO_HORIZONS]
                        if self.bo_swarm_routers and self.bo_swarm_routers[0].feature_names:
                            self.bo_feature_names = list(self.bo_swarm_routers[0].feature_names)
                    except Exception as e_bo_init:
                        print(f"  ⚠️ [ORDER-3 BO ROUTERS WARN]: {e_bo_init}")
                self.order3_mode = True
                self.order3_engine = Order3LiveInferenceEngine(checkpoint_path=ckpt_p, device="cpu")
                self.rel_ckpt_mtime = cur_mtime
                in_d = int(rel_ck["w1"].shape[0]) if "w1" in rel_ck else int(rel_ck.get("in_dim", 194))
                print(f"🔥 [HOT-RELOAD ORDER-3 CHAMPION] Загружен непрерывный {in_d}D Order-3 Чемпион ({rel_ck.get('arch_version')}, tag={rel_ck.get('champ_tag', 'v1')}) из {ckpt_p.name} (Geo={rel_ck.get('geo_score_bp', 0):+.2f} bp)!")
                return
            if "w1" in rel_ck and ("w2" in rel_ck or "w2_moe" in rel_ck):
                self.l7_w1 = rel_ck["w1"].float()
                self.l7_b1 = rel_ck["b1"].float()
                self.l7_wg = rel_ck["wg"].float() if "wg" in rel_ck else None
                self.l7_bg = rel_ck["bg"].float() if "bg" in rel_ck else None
                self.l7_w_res1 = rel_ck["w_res1"].float() if "w_res1" in rel_ck else None
                self.l7_b_res1 = rel_ck["b_res1"].float() if "b_res1" in rel_ck else None
                self.l7_w_res2 = rel_ck["w_res2"].float() if "w_res2" in rel_ck else None
                self.l7_w_rec = rel_ck["w_rec"].float() if "w_rec" in rel_ck and rel_ck["w_rec"] is not None else None
                self.l7_b_rec = rel_ck["b_rec"].float() if "b_rec" in rel_ck and rel_ck["b_rec"] is not None else None
                if not hasattr(self, "l7_mem_h") or self.l7_mem_h is None:
                    self.l7_mem_h = torch.zeros((1, 16), dtype=torch.float32)
                else:
                    self.l7_mem_h.zero_()
                self.l7_w2_moe = rel_ck["w2_moe"].float() if "w2_moe" in rel_ck and rel_ck["w2_moe"] is not None else None
                self.l7_b2_moe = rel_ck["b2_moe"].float() if "b2_moe" in rel_ck and rel_ck["b2_moe"] is not None else None
                self.l7_w_gate1 = rel_ck["w_gate1"].float() if "w_gate1" in rel_ck and rel_ck["w_gate1"] is not None else None
                self.l7_b_gate1 = rel_ck["b_gate1"].float() if "b_gate1" in rel_ck and rel_ck["b_gate1"] is not None else None
                self.l7_w_gate2 = rel_ck["w_gate2"].float() if "w_gate2" in rel_ck and rel_ck["w_gate2"] is not None else None
                self.l7_b_gate2 = rel_ck["b_gate2"].float() if "b_gate2" in rel_ck and rel_ck["b_gate2"] is not None else None
                self.l7_w2 = rel_ck["w2"].float() if "w2" in rel_ck and rel_ck["w2"] is not None else self.l7_w2_moe
                self.l7_b2 = rel_ck["b2"].float() if "b2" in rel_ck and rel_ck["b2"] is not None else self.l7_b2_moe
                self.l7_w_reg1 = rel_ck["w_reg1"].float() if "w_reg1" in rel_ck and rel_ck["w_reg1"] is not None else None
                self.l7_b_reg1 = rel_ck["b_reg1"].float() if "b_reg1" in rel_ck and rel_ck["b_reg1"] is not None else None
                self.l7_b_reg1_b = rel_ck["b_reg1_b"].float() if "b_reg1_b" in rel_ck and rel_ck["b_reg1_b"] is not None else None
                self.l7_b_reg1_a = rel_ck["b_reg1_a"].float() if "b_reg1_a" in rel_ck and rel_ck["b_reg1_a"] is not None else None
                self.l7_w_reg2 = rel_ck["w_reg2"].float() if "w_reg2" in rel_ck and rel_ck["w_reg2"] is not None else None
                self.l7_b_reg2 = rel_ck["b_reg2"].float() if "b_reg2" in rel_ck and rel_ck["b_reg2"] is not None else None
                self.l7_b_reg2_b = rel_ck["b_reg2_b"].float() if "b_reg2_b" in rel_ck and rel_ck["b_reg2_b"] is not None else None
                self.l7_b_reg2_a = rel_ck["b_reg2_a"].float() if "b_reg2_a" in rel_ck and rel_ck["b_reg2_a"] is not None else None
                self.l7_allow_scratch = bool(rel_ck.get("allow_scratch_stop_asks", False))
                self.l7_use_cross_venue = bool(rel_ck.get("use_cross_venue_v1", False))
                self.l7_decoupled_backbone = bool(rel_ck.get("decoupled_market_backbone", int(self.l7_w1.shape[0]) >= 420))
                self.l7_selective_front_cancel = bool(rel_ck.get("selective_front_cancel", int(rel_ck.get("lesson", 0)) >= 22))
                if self.l7_use_cross_venue:
                    cv_fo_path = BASE_DIR / "models" / self.coin / "cross_venue_first_order" / "cross_venue_first_order_v1.pt"
                    if cv_fo_path.exists():
                        from _tools.cross_venue_first_order_net import CrossVenueFirstOrderNet
                        cv_ck = torch.load(cv_fo_path, map_location="cpu", weights_only=False)
                        self.cv_first_order_net = CrossVenueFirstOrderNet(
                            in_dim=int(cv_ck.get("in_dim", 28)),
                            h_dim=int(cv_ck.get("h_dim", 128)),
                            out_dim=int(cv_ck.get("out_dim", 6)),
                            isolated_ln=bool(cv_ck.get("isolated_ln", False)),
                        ).eval()
                        self.cv_first_order_net.load_state_dict(cv_ck["state_dict"], strict=True)
                if int(self.l7_w1.shape[0]) >= 480 and not getattr(self, "bo_swarm_routers", None):
                    from _tools.build_bo_swarm_36d_cache import CausalTop2BigONERouter, HORIZONS as BO_HORIZONS
                    if (BASE_DIR / "models" / "prl_bo").exists():
                        self.bo_swarm_routers = [CausalTop2BigONERouter(H, K, live_only_top2=True) for H, K in BO_HORIZONS]
                        if self.bo_swarm_routers and self.bo_swarm_routers[0].feature_names:
                            self.bo_feature_names = list(self.bo_swarm_routers[0].feature_names)
                self.rel_ckpt_mtime = cur_mtime
                rel_name = rel_ck.get("release_name", f"iter_{rel_ck.get('iteration', '?')}")
                val_ev = float(rel_ck.get("val_ev_pct", 0.0))
                in_d = int(self.l7_w1.shape[0])
                h_d = int(self.l7_w1.shape[1])
                out_d = int(self.l7_w2_moe.shape[-1]) if self.l7_w2_moe is not None else int(self.l7_w2.shape[-1])
                self.l28_hierarchical_mode = bool(in_d == 508 and self.l7_w2_moe is not None and out_d == 180 and self.l7_w_gate1 is not None)
                self.l27_multimodal_mode = bool((not self.l28_hierarchical_mode) and in_d == 480 and self.l7_w2_moe is not None and out_d == 180 and self.l7_w_gate1 is not None)
                self.l26_endogenous_mode = bool((not self.l28_hierarchical_mode) and (not self.l27_multimodal_mode) and in_d >= 480 and out_d == 44 and self.l7_w_reg1 is not None and self.l7_b_reg1_b is not None)
                self.l25_regulator_mode = bool(in_d >= 480 and out_d == 35 and self.l7_w_reg1 is not None and self.l7_w_reg2 is not None)
                if self.l28_hierarchical_mode:
                    self._ensure_l28_tree_routers()
                arch_s = (
                    f"{in_d}D-HierarchicalSubCentMoE(Tree7H+4x45D+Reg26->48->2)"
                    if self.l28_hierarchical_mode
                    else (
                        f"{in_d}D-MultimodalMoE(4x45D+Router288->4+Reg26->48->2)"
                        if self.l27_multimodal_mode
                        else (
                            f"{in_d}D-EndogenousNonlinear(44D+22->48->2)"
                            if self.l26_endogenous_mode
                            else (
                                f"{in_d}D-DualSwarm+PerSlotRegulator(35D+16->32->2)"
                                if self.l25_regulator_mode
                                else (
                                    f"{in_d}D-2StageResSwiGLU({h_d})"
                                    if self.l7_w_res1 is not None
                                    else (f"{in_d}D-SwiGLU({h_d})" if (self.l7_wg is not None or h_d == 384) else f"{in_d}D")
                                )
                            )
                        )
                    )
                )
                print(
                    f"  🔥 [HOT-RELOAD CHAMPION] На лету подключен новый 24-слотовый Чемпион {ckpt_p.name} "
                    f"({rel_name}, {arch_s}, ValEV={val_ev:+.3f}%, scratch={self.l7_allow_scratch}) без остановки торгов!"
                )
        except Exception as e_hr:
            print(f"  ⚠️ [HOT-RELOAD WARN]: {e_hr}")

    def _ensure_l28_tree_routers(self):
        if getattr(self, "st_tree_router", None) is None:
            from _tools.hierarchical_regime_tree import HierarchicalRegimeTreeRouter
            self.st_tree_router = HierarchicalRegimeTreeRouter("prl", use_hybrid_nn=True)
            self.bo_tree_router = HierarchicalRegimeTreeRouter("prl_bo", use_hybrid_nn=True)
            if not hasattr(self, "st_x_ring") or self.st_x_ring is None:
                self.st_x_ring = deque(maxlen=360)
            if not hasattr(self, "bo_x_ring") or self.bo_x_ring is None:
                self.bo_x_ring = deque(maxlen=360)
            if not hasattr(self, "l28_vel8_hist") or self.l28_vel8_hist is None:
                self.l28_vel8_hist = deque(maxlen=6)
            print("🌲 [L28 TREE] Подключены 7-горизонтные иерархические деревья SafeTrade + BigONE (1s..120s, без 90s)!")

    def _refresh_legacy_slot_mirrors(self):
        if getattr(self, "l7_w1", None) is None:
            return
        act_b = [s for s in self.bid_slots if s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-4]
        if act_b:
            act_b.sort(key=lambda x: (-float(x.get("px", 0.0)), int(x.get("slot", 0))))
            top_b = act_b[0]
            self.active_bid_id = top_b["id"]
            self.active_bid_p = float(top_b["px"])
            self.active_bid_q = float(sum(float(x.get("qty", 0.0)) for x in act_b))
        else:
            self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0

        act_a = [s for s in self.ask_slots if s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-4]
        if act_a:
            act_a.sort(key=lambda x: (float(x.get("px", 0.0)), int(x.get("slot", 0))))
            top_a = act_a[0]
            self.active_ask_id = top_a["id"]
            self.active_ask_p = float(top_a["px"])
            self.active_ask_q = float(top_a["qty"])
            if len(act_a) >= 2:
                sec_a = act_a[1]
                self.slot2_ask_id = sec_a["id"]
                self.slot2_ask_p = float(sec_a["px"])
                self.slot2_ask_q = float(sec_a["qty"])
            else:
                self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
        else:
            self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
            self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0

    def _reconcile_lifo_tranches(self):
        """
        Поддерживает строгий инвариант 3-уровневого LIFO-стека [tr_low, tr_mid, tr_high]:
        1) При пустой позиции (< 1.95 PRL) все транши обнуляются.
        2) Сумма траншей никогда не превышает pos_qty (исключает фантомный low_tr > pos_qty после внешней сверки).
        3) При распродаже нижнего транша (< 1.95 PRL) и наличии mid_tranche >= 1.95 PRL выполняется всплытие mid -> low.
        """
        if self.pos_qty < 1.95:
            self.tr_low_px, self.tr_low_qty = 0.0, 0.0
            self.tr_mid_px, self.tr_mid_qty = 0.0, 0.0
            self.tr_high_px, self.tr_high_qty = 0.0, 0.0
            return

        ref_px = self.avg_entry if self.avg_entry > 0.0 else max(self.tr_high_px, self.tr_mid_px, self.tr_low_px, self.tick_size)
        self.tr_low_qty = max(0.0, float(getattr(self, "tr_low_qty", 0.0)))
        self.tr_mid_qty = max(0.0, float(getattr(self, "tr_mid_qty", 0.0)))
        self.tr_high_qty = max(0.0, float(getattr(self, "tr_high_qty", 0.0)))
        if self.tr_low_qty <= 1e-4 or self.tr_low_px <= 0.0:
            self.tr_low_qty, self.tr_low_px = 0.0, 0.0
        if self.tr_mid_qty <= 1e-4 or self.tr_mid_px <= 0.0:
            self.tr_mid_qty, self.tr_mid_px = 0.0, 0.0
        if self.tr_high_qty <= 1e-4 or self.tr_high_px <= 0.0:
            self.tr_high_qty, self.tr_high_px = 0.0, 0.0

        # Сверху вниз подрезаем избыток или доливаем недостачу в core (tr_high), чтобы сумма точно равнялась pos_qty
        rem_cap = float(self.pos_qty)
        if self.tr_low_qty > rem_cap:
            self.tr_low_qty = rem_cap
        rem_cap = max(0.0, rem_cap - self.tr_low_qty)
        if self.tr_mid_qty > rem_cap:
            self.tr_mid_qty = rem_cap
        rem_cap = max(0.0, rem_cap - self.tr_mid_qty)
        if abs(self.tr_high_qty - rem_cap) > 1e-4:
            if rem_cap > 1e-4 and self.tr_high_px <= 0.0:
                used_cost = self.tr_low_qty * self.tr_low_px + self.tr_mid_qty * self.tr_mid_px
                tot_cost = self.pos_qty * ref_px
                self.tr_high_px = max(self.tick_size, (tot_cost - used_cost) / rem_cap) if tot_cost > used_cost else ref_px
            self.tr_high_qty = rem_cap
            if self.tr_high_qty <= 1e-4:
                self.tr_high_px = 0.0

        # Автоматическое всплытие mid_tranche -> low_tranche, когда нижний транш < 1.95 PRL
        if self.tr_low_qty < 1.95 and self.tr_mid_qty >= 1.95:
            prom_q = self.tr_low_qty + self.tr_mid_qty
            self.tr_low_px = (
                (self.tr_low_qty * self.tr_low_px + self.tr_mid_qty * self.tr_mid_px) / prom_q
                if self.tr_low_qty > 1e-4 else self.tr_mid_px
            )
            self.tr_low_qty = prom_q
            self.tr_mid_qty, self.tr_mid_px = 0.0, 0.0
        elif 0.0 < self.tr_low_qty < 1.95:
            # Склеиваем микро-пыль нижнего транша в верхний/средний
            if self.tr_high_qty > 1e-4:
                tot_h = self.tr_high_qty + self.tr_low_qty
                self.tr_high_px = (self.tr_high_qty * self.tr_high_px + self.tr_low_qty * self.tr_low_px) / tot_h
                self.tr_high_qty = tot_h
            else:
                self.tr_high_qty = self.tr_low_qty
                self.tr_high_px = self.tr_low_px
            self.tr_low_qty, self.tr_low_px = 0.0, 0.0

        # Если tr_low_px оказался не дешевле avg_entry (размылся выше средней), переносим его в core (tr_high)
        if self.tr_low_qty > 1e-4 and self.avg_entry > 0.0 and self.tr_low_px >= self.avg_entry - 0.05 * self.tick_size:
            if self.tr_mid_qty >= 1.95 and self.tr_mid_px < self.avg_entry - 0.1 * self.tick_size:
                # Меняем местами с более дешевым mid_tranche
                self.tr_low_px, self.tr_mid_px = self.tr_mid_px, self.tr_low_px
                self.tr_low_qty, self.tr_mid_qty = self.tr_mid_qty, self.tr_low_qty
            else:
                tot_h = self.tr_high_qty + self.tr_low_qty
                self.tr_high_px = (
                    (self.tr_high_qty * (self.tr_high_px or ref_px) + self.tr_low_qty * self.tr_low_px) / tot_h
                    if tot_h > 1e-4 else ref_px
                )
                self.tr_high_qty = tot_h
                self.tr_low_qty, self.tr_low_px = 0.0, 0.0

    def _save_wallet(self):
        self._refresh_legacy_slot_mirrors()
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "api_key_fp": str(getattr(self, "api_key_fp", "")),
            "initial_budget_usd": round(self.initial_budget_usd, 6),
            "base_1023_applied": bool(getattr(self, "base_1023_applied", True)),
            "bot_cash_usd": round(self.bot_cash_usd, 6),
            "pos_qty": round(self.pos_qty, 6),
            "avg_entry": round(self.avg_entry, 6),
            "hold_sec": int(self.hold_sec),
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
            "extra_slots": getattr(self, "extra_slots", []),
            "bid_slots": getattr(self, "bid_slots", []),
            "ask_slots": getattr(self, "ask_slots", []),
            "tr_low_px": round(float(getattr(self, "tr_low_px", 0.0)), 6),
            "tr_low_qty": round(float(getattr(self, "tr_low_qty", 0.0)), 6),
            "tr_mid_px": round(float(getattr(self, "tr_mid_px", 0.0)), 6),
            "tr_mid_qty": round(float(getattr(self, "tr_mid_qty", 0.0)), 6),
            "tr_high_px": round(float(getattr(self, "tr_high_px", 0.0)), 6),
            "tr_high_qty": round(float(getattr(self, "tr_high_qty", 0.0)), 6),
            "last_buy_fill_ts": round(float(getattr(self, "last_buy_fill_ts", 0.0)), 3),
            "split_done_for_pos": bool(self.split_done_for_pos),
            "car_state": int(getattr(self, "car_state", 0)),
            "car_sell_px": round(float(getattr(self, "car_sell_px", 0.0)), 4),
            "car_cycles": int(getattr(self, "car_cycles", 0)),
            "cooldown_until_ts": round(float(getattr(self, "cooldown_until_ts", 0.0)), 3),
            "last_exit_ts": round(float(getattr(self, "last_exit_ts", 0.0)), 3),
            "last_exit_price": round(float(getattr(self, "last_exit_price", 0.0)), 4),
        }
        try:
            self.wallet_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.wallet_file.with_name(f"{self.wallet_file.name}.tmp.{os.getpid()}")
            bak_file = self.wallet_file.with_suffix(".json.bak")
            content = json.dumps(payload, indent=2)
            tmp_file.write_text(content, encoding="utf-8")
            if self.wallet_file.exists():
                try:
                    os.replace(self.wallet_file, bak_file)
                except Exception:
                    pass
            os.replace(tmp_file, self.wallet_file)
        except Exception as e:
            print(f"❌ Ошибка атомарной записи кошелька {self.wallet_file.name}: {e}")

    def _sync_24slot_orders(self, b1: float, count_hold: bool = False, pre_fetched_orders: dict = None):
        if count_hold and self.pos_qty > 1e-4:
            self.hold_sec += 1
            if b1 > 0:
                self.peak_bid = max(self.peak_bid, self.avg_entry, b1)

        active_oids = [
            s["id"] for s in (self.bid_slots + self.ask_slots)
            if s.get("id") is not None
        ]
        if not active_oids:
            self._refresh_legacy_slot_mirrors()
            return

        oid_map = dict(pre_fetched_orders) if pre_fetched_orders else {}
        missing_for_open = [oid for oid in active_oids if oid not in oid_map]
        if len(missing_for_open) >= 1:
            try:
                open_list = self.client.get_open_orders(self.market)
                for o_item in open_list:
                    oid_k = o_item.get("id")
                    if oid_k is not None:
                        oid_map[oid_k] = o_item
            except Exception:
                pass

        still_missing = [oid for oid in active_oids if oid not in oid_map]
        if len(still_missing) > 1:
            def _fetch_one(m_oid):
                try:
                    return (m_oid, self.client.get_order(m_oid))
                except Exception:
                    return (m_oid, None)
            for m_oid, m_ord in self._io_pool.map(_fetch_one, still_missing):
                if m_ord is not None:
                    oid_map[m_oid] = m_ord

        had_any_fill = False
        had_any_change = False

        # 1. Синхронизация 12 Bid-слотов (0..11)
        for s_idx, slot in enumerate(self.bid_slots):
            oid = slot.get("id")
            if oid is None:
                continue
            try:
                o = oid_map.get(oid)
                if o is None:
                    o = self.client.get_order(oid)
                filled = float(o.get("filled_amount", 0.0))
                orig_q = float(o.get("origin_amount", slot.get("qty", 0.0)))
                st = str(o.get("state", "")).lower()
                prev_acc = float(slot.get("acc_fill", 0.0))
                d_fill = max(0.0, filled - prev_acc)

                if ((st == "wait" and (d_fill >= 2.0 or (bool(getattr(self, "order3_mode", False)) and d_fill > 1e-4))) or (st in ("done", "cancel") and d_fill > 1e-4)):
                    px = float(o.get("avg_price", 0.0)) or float(o.get("price", slot.get("px", b1)))
                    fee_rate = float(o.get("maker_fee", 0.001) or 0.001)
                    cost = d_fill * px
                    net_d = math.floor(d_fill * (1.0 - fee_rate) * 10000.0) / 10000.0
                    if net_d > 0.0:
                        self._reconcile_lifo_tranches()
                        was_flat = self.pos_qty < 1.95
                        old_q = self.pos_qty
                        old_avg = self.avg_entry
                        if was_flat:
                            self.tr_high_px = px
                            self.tr_high_qty = net_d
                            self.tr_mid_px, self.tr_mid_qty = 0.0, 0.0
                            self.tr_low_px, self.tr_low_qty = 0.0, 0.0
                            self.hold_sec = 0
                        elif self.tr_low_qty <= 1e-4:
                            if px <= old_avg - 0.35 * self.tick_size:
                                self.tr_low_px = px
                                self.tr_low_qty = net_d
                            elif old_q >= 1.95 and px >= old_avg + 0.35 * self.tick_size:
                                self.tr_low_px = old_avg
                                self.tr_low_qty = old_q
                                self.tr_high_px = px
                                self.tr_high_qty = net_d
                            else:
                                new_h_q = self.tr_high_qty + net_d
                                self.tr_high_px = (self.tr_high_qty * (self.tr_high_px or old_avg) + net_d * px) / max(1e-6, new_h_q)
                                self.tr_high_qty = new_h_q
                        else:
                            # Уже есть нижний транш tr_low: проверяем более глубокое дно (deeper_dip), тот же уровень (same_low) или средний эшелон (mid_dip)
                            if px < self.tr_low_px - 0.35 * self.tick_size:
                                pushed_m_q = self.tr_mid_qty + self.tr_low_qty
                                self.tr_mid_px = (self.tr_mid_qty * self.tr_mid_px + self.tr_low_qty * self.tr_low_px) / max(1e-6, pushed_m_q)
                                self.tr_mid_qty = pushed_m_q
                                self.tr_low_px = px
                                self.tr_low_qty = net_d
                            elif abs(px - self.tr_low_px) <= 0.35 * self.tick_size:
                                new_l_q = self.tr_low_qty + net_d
                                self.tr_low_px = (self.tr_low_qty * self.tr_low_px + net_d * px) / max(1e-6, new_l_q)
                                self.tr_low_qty = new_l_q
                            elif px <= old_avg - 0.35 * self.tick_size:
                                new_m_q = self.tr_mid_qty + net_d
                                self.tr_mid_px = (self.tr_mid_qty * self.tr_mid_px + net_d * px) / max(1e-6, new_m_q)
                                self.tr_mid_qty = new_m_q
                            else:
                                new_h_q = self.tr_high_qty + net_d
                                self.tr_high_px = (self.tr_high_qty * (self.tr_high_px or old_avg) + net_d * px) / max(1e-6, new_h_q)
                                self.tr_high_qty = new_h_q

                        self.bot_cash_usd = max(0.0, self.bot_cash_usd - cost)
                        self.avg_entry = (self.pos_qty * self.avg_entry + net_d * px) / max(1e-6, self.pos_qty + net_d)
                        self.pos_qty = math.floor((self.pos_qty + net_d) * 10000.0) / 10000.0
                        self._reconcile_lifo_tranches()
                        self.ex_coin_free = max(0.0, self.ex_coin_free + net_d)
                        self.peak_bid = max(self.peak_bid, px, b1 if b1 > 0 else px)
                        self.last_buy_fill_ts = time.time()
                        slot["acc_fill"] = filled
                        slot["qty"] = max(0.0, orig_q - filled)
                        had_any_fill = True
                        had_any_change = True
                        print(
                            f"  🟢 [24-SLOT FILL BID #{s_idx:02d}] +{net_d:.4f} PRL @ ${px:.2f} | "
                            f"Поз={self.pos_qty:.4f} PRL (avg=${self.avg_entry:.4f}, low_tr={self.tr_low_qty:.1f}@${self.tr_low_px:.2f}, mid_tr={self.tr_mid_qty:.1f}@${self.tr_mid_px:.2f}) | "
                            f"Кэш=${self.bot_cash_usd:.2f} | px4={px:.4f} id={oid} t={_msk_ms()}"
                        )
                if st in ("done", "cancel"):
                    if st == "cancel" and d_fill <= 1e-4:
                        print(f"  ✖️ [24-SLOT SYNC-CLEARED BID #{s_idx:02d}] id={oid} state=cancel filled={filled:.4f} | t={_msk_ms()}")
                    slot["id"] = None
                    slot["px"] = 0.0
                    slot["qty"] = 0.0
                    slot["act_ts"] = 0.0
                    slot["acc_fill"] = 0.0
                    had_any_change = True
                elif st == "wait":
                    slot["qty"] = max(0.0, orig_q - filled)
            except Exception as e_b:
                print(f"  ⚠️ [24-SLOT SYNC BID #{s_idx:02d} WARN] id={oid}: {e_b}")

        # 2. Синхронизация 12 Ask-слотов (0..11)
        for s_idx, slot in enumerate(self.ask_slots):
            oid = slot.get("id")
            if oid is None:
                continue
            try:
                o = oid_map.get(oid)
                if o is None:
                    o = self.client.get_order(oid)
                filled = float(o.get("filled_amount", 0.0))
                orig_q = float(o.get("origin_amount", slot.get("qty", 0.0)))
                st = str(o.get("state", "")).lower()
                prev_acc = float(slot.get("acc_fill", 0.0))
                d_fill = max(0.0, filled - prev_acc)

                if ((st == "wait" and (d_fill >= 2.0 or (bool(getattr(self, "order3_mode", False)) and d_fill > 1e-4))) or (st in ("done", "cancel") and d_fill > 1e-4)):
                    self._reconcile_lifo_tranches()
                    exec_vol = float(o.get("executed_volume", o.get("funds_received", 0.0)) or 0.0)
                    lim_px = float(o.get("price", slot.get("px", b1)) or b1)
                    px = float(o.get("avg_price", 0.0) or 0.0) or (
                        (exec_vol / filled) if (exec_vol > 0.0 and filled > 1e-6) else (max(lim_px, b1) if (b1 > 0.0 and lim_px < 0.96 * b1) else lim_px)
                    )
                    fee_rate = float(o.get("maker_fee", 0.001) or 0.001)
                    proceeds = d_fill * px * (1.0 - fee_rate)
                    # 3-уровневый LIFO учет списания себестоимости (сначала tr_low -> tr_mid для LIFO-слотов, остальное из core tr_high)
                    is_lifo_slot = (2 <= s_idx < 6) if getattr(self, "l7_allow_scratch", False) else (s_idx < 6)
                    from_low = min(d_fill, self.tr_low_qty) if is_lifo_slot else 0.0
                    rem_lifo = max(0.0, d_fill - from_low) if is_lifo_slot else 0.0
                    from_mid = min(rem_lifo, self.tr_mid_qty)
                    from_high = max(0.0, d_fill - from_low - from_mid)
                    if from_high > self.tr_high_qty + 1e-4:
                        spill = from_high - self.tr_high_qty
                        from_high = self.tr_high_qty
                        extra_mid = min(spill, max(0.0, self.tr_mid_qty - from_mid))
                        from_mid += extra_mid
                        spill -= extra_mid
                        from_low += min(spill, max(0.0, self.tr_low_qty - from_low))

                    fallback_ref = self.avg_entry if self.avg_entry > 0.0 else px
                    low_ref = self.tr_low_px if self.tr_low_px > 0.0 else fallback_ref
                    mid_ref = self.tr_mid_px if self.tr_mid_px > 0.0 else fallback_ref
                    high_ref = self.tr_high_px if self.tr_high_px > 0.0 else fallback_ref
                    cost_basis = ((from_low * low_ref) + (from_mid * mid_ref) + (from_high * high_ref)) / max(1e-6, 1.0 - fee_rate)
                    pnl = proceeds - cost_basis

                    self.tr_low_qty = max(0.0, self.tr_low_qty - from_low)
                    self.tr_mid_qty = max(0.0, self.tr_mid_qty - from_mid)
                    self.tr_high_qty = max(0.0, self.tr_high_qty - from_high)
                    self.bot_cash_usd += proceeds
                    self.realized_pnl_usd += pnl
                    self.pos_qty = max(0.0, math.floor((self.pos_qty - d_fill) * 10000.0) / 10000.0)
                    if self.pos_qty < 1.95:
                        self.pos_qty, self.avg_entry, self.hold_sec, self.peak_bid = 0.0, 0.0, 0, 0.0
                        self._reconcile_lifo_tranches()
                    else:
                        rem_tr_q = self.tr_low_qty + self.tr_mid_qty + self.tr_high_qty
                        if rem_tr_q > 1e-4:
                            self.avg_entry = (
                                self.tr_low_qty * low_ref
                                + self.tr_mid_qty * mid_ref
                                + self.tr_high_qty * high_ref
                            ) / rem_tr_q
                        self._reconcile_lifo_tranches()
                    self.n_trades += 1
                    self.last_exit_ts = time.time()
                    self.last_exit_price = float(px)
                    slot["acc_fill"] = filled
                    slot["qty"] = max(0.0, orig_q - filled)
                    had_any_fill = True
                    had_any_change = True
                    print(
                        f"  💰 [24-SLOT FILL ASK #{s_idx:02d}] -{d_fill:.4f} PRL @ ${px:.2f} | "
                        f"PnL={pnl:+.4f}$ | Остаток поз={self.pos_qty:.4f} PRL (avg=${self.avg_entry:.4f}, low_tr={self.tr_low_qty:.1f}@${self.tr_low_px:.2f}) | "
                        f"Кэш=${self.bot_cash_usd:.2f} | px4={px:.4f} id={oid} t={_msk_ms()}"
                    )
                if st in ("done", "cancel"):
                    if st == "cancel" and d_fill <= 1e-4:
                        print(f"  ✖️ [24-SLOT SYNC-CLEARED ASK #{s_idx:02d}] id={oid} state=cancel filled={filled:.4f} | t={_msk_ms()}")
                    slot["id"] = None
                    slot["px"] = 0.0
                    slot["qty"] = 0.0
                    slot["act_ts"] = 0.0
                    slot["acc_fill"] = 0.0
                    had_any_change = True
                elif st == "wait":
                    slot["qty"] = max(0.0, orig_q - filled)
            except Exception as e_a:
                print(f"  ⚠️ [24-SLOT SYNC ASK #{s_idx:02d} WARN] id={oid}: {e_a}")

        self._refresh_legacy_slot_mirrors()
        if had_any_fill:
            self.sync_account_balances(b1, from_trade_fill=True)
        if had_any_change or had_any_fill:
            self._save_wallet()

    @property
    def is_24slot_active(self) -> bool:
        return (getattr(self, "l7_w1", None) is not None) or bool(getattr(self, "order3_mode", False))

    def sync_live_orders(self, b1: float, count_hold: bool = False):
        if self.mode != "live":
            return
        if self.is_24slot_active:
            self._sync_24slot_orders(b1, count_hold=count_hold)
            return
        if count_hold and self.pos_qty > 1e-4:
            self.hold_sec += 1
            if b1 > 0:
                self.peak_bid = max(self.peak_bid, self.avg_entry, b1)
        if getattr(self, "car_state", 0) == 1 and self.slot2_ask_id is None:
            self.car_state = 2 if (self.active_bid_id is not None or self.bot_cash_usd >= 3.5) else 0
        elif getattr(self, "car_state", 0) == 2 and self.active_bid_id is None and self.bot_cash_usd < 3.5:
            self.car_state = 0
        oid_map = {}
        extra_oids = [s.get("id") for s in getattr(self, "extra_slots", []) if s.get("id") is not None]
        active_oids = [oid for oid in (self.active_bid_id, self.active_ask_id, self.slot2_ask_id, *extra_oids) if oid is not None]
        if len(active_oids) > 1:
            try:
                open_list = self.client.get_open_orders(self.market)
                for o_item in open_list:
                    oid_k = o_item.get("id")
                    if oid_k in active_oids:
                        oid_map[oid_k] = o_item
            except Exception:
                pass
        if self.active_bid_id is not None:
            if count_hold:
                self.bid_age_sec = getattr(self, "bid_age_sec", 0) + 1
            try:
                o = oid_map.get(self.active_bid_id)
                if isinstance(o, Exception):
                    raise o
                if o is None:
                    o = self.client.get_order(self.active_bid_id)
                filled = float(o.get("filled_amount", 0.0))
                st = o.get("state", "")
                prev_acc = float(getattr(self, "bid_accounted_filled", 0.0))
                self.bid_partial_filled = filled if st == "wait" else 0.0
                if st == "wait" and (filled - prev_acc) >= 2.0:
                    d_fill = filled - prev_acc
                    px = float(o.get("avg_price", 0.0)) or float(o.get("price", self.active_bid_p))
                    fee_rate = float(o.get("maker_fee", 0.001) or 0.001)
                    cost = d_fill * px
                    net_d = math.floor(d_fill * (1.0 - fee_rate) * 10000.0) / 10000.0
                    self.bot_cash_usd = max(0.0, self.bot_cash_usd - cost)
                    self.avg_entry = (self.pos_qty * self.avg_entry + net_d * px) / max(1e-6, self.pos_qty + net_d)
                    self.pos_qty = math.floor((self.pos_qty + net_d) * 10000.0) / 10000.0
                    self.bid_accounted_filled = filled
                    self.ex_coin_free = max(0.0, self.ex_coin_free + net_d)
                    if self.active_ask_id is None and self.pos_qty >= 2.0:
                        fl_p = round(math.ceil((self.avg_entry + 0.75 * self.tick_size) / self.tick_size - 1e-4) * self.tick_size, 2)
                        ask_px = max(b1 + self.tick_size, fl_p) if b1 > 0 else fl_p
                        try:
                            ra = self.client.place_limit_order(self.market, "sell", self.pos_qty, ask_px)
                            self.active_ask_id = ra.get("id")
                            self.active_ask_p = float(ra.get("price", ask_px))
                            self.active_ask_q = float(ra.get("origin_amount", self.pos_qty))
                            self.ex_coin_free = max(0.0, self.ex_coin_free - self.active_ask_q)
                            print(f"  ⚡ [PARTIAL BID -> INSTANT ASK] +{net_d:.4f} PRL @ ${px:.2f} сразу выставлены в Ask @ ${self.active_ask_p:.2f} (бид продолжает наливаться)")
                        except Exception:
                            pass
                    self._save_wallet()
                elif filled > 0 and st in ("done", "cancel"):
                    rem_fill = max(0.0, filled - prev_acc)
                    px = float(o.get("avg_price", 0.0)) or float(o.get("price", self.active_bid_p))
                    fee_rate = float(o.get("maker_fee", 0.001) or 0.001)
                    cost = rem_fill * px
                    net_filled = math.floor(rem_fill * (1.0 - fee_rate) * 10000.0) / 10000.0
                    tot_net_filled = math.floor(filled * (1.0 - fee_rate) * 10000.0) / 10000.0
                    self.bot_cash_usd = max(0.0, self.bot_cash_usd - cost)
                    if getattr(self, "car_state", 0) in (1, 2) and self.pos_qty >= 2.0:
                        if net_filled > 0:
                            self.avg_entry = round((self.pos_qty * self.avg_entry + net_filled * px) / max(1e-6, self.pos_qty + net_filled), 4)
                            self.pos_qty = math.floor((self.pos_qty + net_filled) * 10000.0) / 10000.0
                        self.car_cycles = getattr(self, "car_cycles", 0) + 1
                        self.car_state = 0
                        self.split_done_for_pos = False
                        print(f"  🎠 [CAROUSEL BUYBACK #{self.car_cycles}] +{tot_net_filled:.4f} PRL @ ${px:.2f} (после продажи @ ${self.car_sell_px:.2f}) | Поз={self.pos_qty:.4f} PRL | Новый avg_entry=${self.avg_entry:.4f}")
                        for oid_attr in ("active_ask_id", "slot2_ask_id"):
                            oid = getattr(self, oid_attr)
                            if oid is not None:
                                try:
                                    self.client.cancel_order(oid)
                                except Exception:
                                    pass
                        self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                        self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                        self.sync_account_balances(b1, from_trade_fill=True)
                        new_floor_p = round((math.floor(self.avg_entry / self.tick_size + 1e-9) + 1) * self.tick_size, 2)
                        tot_q = math.floor(min(self.pos_qty, self.ex_coin_free) * 10000.0) / 10000.0
                        if tot_q >= 2.0:
                            try:
                                ra = self.client.place_limit_order(self.market, "sell", tot_q, max(b1 + self.tick_size, new_floor_p))
                                self.active_ask_id = ra.get("id")
                                self.active_ask_p = float(ra.get("price", new_floor_p))
                                self.active_ask_q = float(ra.get("origin_amount", tot_q))
                            except Exception:
                                pass
                    else:
                        is_dip_scale = (self.pos_qty >= 2.0 and self.avg_entry > 0.0 and px <= self.avg_entry - 0.75 * self.tick_size and len(getattr(self, "extra_slots", [])) < 8)
                        if net_filled > 0:
                            if is_dip_scale and net_filled >= 4.0:
                                q_ind = math.floor((net_filled * 0.50) * 10000.0) / 10000.0
                                q_merge = math.floor((net_filled - q_ind) * 10000.0) / 10000.0
                                self.avg_entry = (self.pos_qty * self.avg_entry + q_merge * px) / max(1e-6, self.pos_qty + q_merge)
                                self.pos_qty = math.floor((self.pos_qty + net_filled) * 10000.0) / 10000.0
                                dyn_target_ticks = 0.75
                                if float(getattr(self, "last_ez90", 0.0)) > 1.10 and float(getattr(self, "last_ez120", 0.0)) > 1.10 and float(getattr(self, "last_p_exhaust", 0.5)) < 0.35:
                                    dyn_target_ticks = 2.75 if (float(getattr(self, "last_ez90", 0.0)) > 1.65 and float(getattr(self, "last_ez120", 0.0)) > 1.55) else 1.75
                                ind_ask_p = round(math.ceil((px + dyn_target_ticks * self.tick_size) / self.tick_size - 1e-4) * self.tick_size, 2)
                                ind_ask_p = max(b1 + self.tick_size if b1 > 0 else ind_ask_p, ind_ask_p)
                                try:
                                    r_ind = self.client.place_limit_order(self.market, "sell", q_ind, ind_ask_p)
                                    self.extra_slots.append({
                                        "id": r_ind.get("id"),
                                        "p": float(r_ind.get("price", ind_ask_p)),
                                        "q": float(r_ind.get("origin_amount", q_ind)),
                                        "entry_px": px,
                                        "entry_ts": time.time(),
                                    })
                                    print(f"  ⚡ [10-SLOT INDEPENDENT ASK #{len(self.extra_slots)+2}] +{q_ind:.4f} PRL (вход @ ${px:.2f}) выставлен автономно @ ${ind_ask_p:.2f} | Остальные +{q_merge:.4f} PRL снизили avg_entry до ${self.avg_entry:.4f}")
                                except Exception as e_ind:
                                    print(f"  ⚠️ [10-SLOT ASK WARN]: {e_ind}")
                            else:
                                self.avg_entry = (self.pos_qty * self.avg_entry + net_filled * px) / max(1e-6, self.pos_qty + net_filled)
                                self.pos_qty = math.floor((self.pos_qty + net_filled) * 10000.0) / 10000.0
                        if not is_dip_scale:
                            self.hold_sec = 0
                        self.peak_bid = max(px, b1 if b1 > 0 else px)
                        self.ez25_entry = self.last_ez25
                        self.ez12_entry = self.last_ez12
                        self.split_done_for_pos = False
                        self.car_state = 0
                    self.last_buy_fill_ts = time.time()
                    self.q_ahead = 0.0
                    self.bid_age_sec = 0
                    self.bid_partial_filled = 0.0
                    self.bid_accounted_filled = 0.0
                    self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                    self.sync_account_balances(b1, from_trade_fill=True)
                    ind_locked_q = sum(float(s.get("q", 0.0)) for s in getattr(self, "extra_slots", []))
                    if self.active_ask_id is not None and (self.active_ask_q + self.slot2_ask_q + ind_locked_q) < self.pos_qty - 1.0:
                        try:
                            self.client.cancel_order(self.active_ask_id)
                        except Exception:
                            pass
                        self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                        self.ex_coin_free = max(0.0, self.pos_qty - self.slot2_ask_q - ind_locked_q)
                    actual_net = max(tot_net_filled, self.pos_qty if self.pos_qty > tot_net_filled else tot_net_filled)
                    self._save_wallet()
                    print(f"  🟢 [FILL BUY] +{actual_net:.4f} PRL (брутто {max(filled, actual_net / max(1e-6, 1.0 - fee_rate)):.4f}) @ ${px:.2f} | Остаток кэша бота=${self.bot_cash_usd:.2f}")
                elif st == "cancel":
                    self.q_ahead = 0.0
                    self.bid_age_sec = 0
                    self.bid_partial_filled = 0.0
                    self.bid_accounted_filled = 0.0
                    self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                    self._save_wallet()
                elif st == "wait" and b1 > 0 and self.active_bid_p <= b1 - 2.0 * self.tick_size + 1e-6:
                    old_b_id = self.active_bid_id
                    old_b_p = self.active_bid_p
                    old_b_q = self.active_bid_q
                    try:
                        self.client.cancel_order(old_b_id)
                        print(f"  🧹 [AUTO-CANCEL STALE BID] Снят отставший от рынка бид id={old_b_id}: {old_b_q:.4f} PRL @ ${old_b_p:.2f} (текущий Bid=${b1:.2f})")
                    except Exception:
                        pass
                    self.q_ahead = 0.0
                    self.bid_age_sec = 0
                    self.bid_partial_filled = 0.0
                    self.bid_accounted_filled = 0.0
                    self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                    self.sync_account_balances(b1)
                    self._save_wallet()
            except Exception as e:
                print(f"  ⚠️ [SYNC BID WARN] id={self.active_bid_id}: {e}")
        if self.active_ask_id is not None:
            try:
                o = oid_map.get(self.active_ask_id)
                if isinstance(o, Exception):
                    raise o
                if o is None:
                    o = self.client.get_order(self.active_ask_id)
                filled = float(o.get("filled_amount", 0.0))
                st = o.get("state", "")
                if filled > 0 and st in ("done", "cancel"):
                    px = float(o.get("avg_price", 0.0)) or float(o.get("price", self.active_ask_p))
                    fee_rate = float(o.get("maker_fee", 0.001) or 0.001)
                    proceeds = filled * px * (1.0 - fee_rate)
                    eff_entry = self.avg_entry if self.avg_entry > 0.0 else px
                    cost_basis = (filled / max(1e-6, 1.0 - fee_rate)) * eff_entry
                    pnl = proceeds - cost_basis
                    self.bot_cash_usd += proceeds
                    self.realized_pnl_usd += pnl
                    self.pos_qty = max(0.0, math.floor((self.pos_qty - filled) * 10000.0) / 10000.0)
                    if self.pos_qty < 2.0:
                        if eff_entry > 0.0 and (getattr(self, "last_dd900", 0.0) >= 4.0 or (self.hold_sec >= 600 and getattr(self, "last_dd900", 0.0) >= 3.0) or (b1 > 0.0 and (eff_entry - b1) / self.tick_size >= 2.8)):
                            real_t = (px - eff_entry) / self.tick_size
                            if real_t <= -2.0:
                                self.cooldown_until_ts = time.time() + 600.0
                                print(f"  🛡️ [DUMP COOLDOWN] 10-мин пауза на вход после разморозки на проливе (realized={real_t:.1f}t, dd900={self.last_dd900:.1f}t)")
                            elif real_t <= -0.8 and b1 > 0.0 and (eff_entry - b1) / self.tick_size >= 2.8:
                                self.cooldown_until_ts = time.time() + 900.0
                                print(f"  🛡️ [DUMP COOLDOWN] 15-мин пауза на вход после ранней разморозки (realized={real_t:.1f}t, dd900={self.last_dd900:.1f}t)")
                        self.last_exit_ts = time.time()
                        self.last_exit_price = float(px)
                        self.pos_qty, self.avg_entry, self.hold_sec, self.peak_bid = 0.0, 0.0, 0, 0.0
                        self.split_done_for_pos = False
                        self.user_manual_ask = False
                        self.car_state = 0
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
                        self.user_manual_ask = False
                    self.sync_account_balances(b1, from_trade_fill=True)
                    self._save_wallet()
                    print(f"  💰 [FILL SELL SLOT-1] -{filled:.4f} PRL @ ${px:.2f} | PnL={pnl:+.4f}$ | Остаток поз={self.pos_qty:.4f} PRL | Кэш=${self.bot_cash_usd:.2f}")
                elif st == "cancel":
                    self.q_ahead = 0.0
                    self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                    self.user_manual_ask = False
                    self._save_wallet()
            except Exception as e:
                print(f"  ⚠️ [SYNC ASK WARN] id={self.active_ask_id}: {e}")
        if self.slot2_ask_id is not None:
            try:
                o2 = oid_map.get(self.slot2_ask_id)
                if isinstance(o2, Exception):
                    raise o2
                if o2 is None:
                    o2 = self.client.get_order(self.slot2_ask_id)
                filled2 = float(o2.get("filled_amount", 0.0))
                st2 = o2.get("state", "")
                if filled2 > 0 and st2 in ("done", "cancel"):
                    px2 = float(o2.get("avg_price", 0.0)) or float(o2.get("price", self.slot2_ask_p))
                    fee_rate2 = float(o2.get("maker_fee", 0.001) or 0.001)
                    proceeds2 = filled2 * px2 * (1.0 - fee_rate2)
                    eff_entry2 = self.avg_entry if self.avg_entry > 0.0 else px2
                    cost_basis2 = (filled2 / max(1e-6, 1.0 - fee_rate2)) * eff_entry2
                    pnl2 = proceeds2 - cost_basis2
                    self.bot_cash_usd += proceeds2
                    self.realized_pnl_usd += pnl2
                    self.pos_qty = max(0.0, math.floor((self.pos_qty - filled2) * 10000.0) / 10000.0)
                    if self.pos_qty < 2.0:
                        self.pos_qty, self.avg_entry, self.hold_sec, self.peak_bid = 0.0, 0.0, 0, 0.0
                        self.split_done_for_pos = False
                        self.car_state = 0
                    elif getattr(self, "car_state", 0) in (1, 2):
                        if self.active_bid_id is not None:
                            try:
                                self.client.cancel_order(self.active_bid_id)
                            except Exception:
                                pass
                            self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                        self.car_sell_px = px2
                        self.car_state = 2
                        self.sync_account_balances(b1, from_trade_fill=True)
                        buy_p = round(min(b1, px2 - 2.0 * self.tick_size) / self.tick_size) * self.tick_size
                        usable_u = min(self.bot_cash_usd * 0.998, self.ex_usdt_free * 0.998)
                        buy_q = math.floor((usable_u / max(buy_p, self.tick_size)) * 10000.0) / 10000.0
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
        if getattr(self, "extra_slots", None):
            rem_extra = []
            for idx_s, s_item in enumerate(self.extra_slots):
                sid = s_item.get("id")
                if sid is None:
                    continue
                try:
                    os_ord = oid_map.get(sid)
                    if os_ord is None:
                        os_ord = self.client.get_order(sid)
                    s_filled = float(os_ord.get("filled_amount", 0.0))
                    s_st = os_ord.get("state", "")
                    if s_filled > 0 and s_st in ("done", "cancel"):
                        s_px = float(os_ord.get("avg_price", 0.0)) or float(os_ord.get("price", s_item.get("p", 0.0)))
                        s_fee = float(os_ord.get("maker_fee", 0.001) or 0.001)
                        s_proc = s_filled * s_px * (1.0 - s_fee)
                        s_ent = float(s_item.get("entry_px", self.avg_entry or s_px))
                        s_cost = (s_filled / max(1e-6, 1.0 - s_fee)) * s_ent
                        s_pnl = s_proc - s_cost
                        self.bot_cash_usd += s_proc
                        self.realized_pnl_usd += s_pnl
                        self.pos_qty = max(0.0, math.floor((self.pos_qty - s_filled) * 10000.0) / 10000.0)
                        if s_pnl > 0 and self.pos_qty >= 2.0 and b1 > 0 and (self.avg_entry - b1) / self.tick_size >= 1.5:
                            self.avg_entry = round(max(b1, self.avg_entry - (s_pnl * 0.45) / self.pos_qty), 4)
                        self.n_trades += 1
                        self.sync_account_balances(b1, from_trade_fill=True)
                        print(f"  💰 [FILL SELL SLOT-{idx_s+3}] -{s_filled:.4f} PRL @ ${s_px:.2f} (вход=${s_ent:.2f}) | PnL={s_pnl:+.4f}$ | Остаток поз={self.pos_qty:.4f} PRL | Кэш=${self.bot_cash_usd:.2f}")
                    elif s_st != "cancel":
                        s_ent = float(s_item.get("entry_px", 0.0))
                        s_cur_p = float(s_item.get("p", 0.0))
                        s_rem_q = float(os_ord.get("remaining_volume", s_item.get("q", 0.0)))
                        s_age = time.time() - float(s_item.get("entry_t", time.time()))
                        tot_slots_active = (6 if self.pos_qty >= 2.0 else 0) + len(getattr(self, "extra_slots", []))
                        decay_l1 = 260.0 if tot_slots_active >= 7 else (360.0 if tot_slots_active >= 4 else 450.0)
                        decay_l2 = 600.0 if tot_slots_active >= 7 else 900.0
                        plus1_p = round(math.ceil((s_ent + 0.75 * self.tick_size) / self.tick_size - 1e-4) * self.tick_size, 2)
                        target_decay_p = None
                        if s_age >= decay_l2:
                            target_decay_p = max(b1 + self.tick_size, float(getattr(self, "last_a1", b1 + self.tick_size)))
                        elif s_age >= decay_l1 and s_cur_p > plus1_p + 0.25 * self.tick_size:
                            target_decay_p = plus1_p
                        elif (
                            s_ent > 0.0
                            and s_cur_p > plus1_p + 0.25 * self.tick_size
                            and b1 >= s_ent
                            and (float(getattr(self, "last_p_exhaust", 0.0)) >= 0.45 or float(getattr(self, "last_p_adverse", 0.0)) >= 0.52)
                        ):
                            target_decay_p = plus1_p

                        if target_decay_p is not None and abs(s_cur_p - target_decay_p) > 0.25 * self.tick_size and s_rem_q >= 2.0:
                            try:
                                self.client.cancel_order(sid)
                                r_t = self.client.place_limit_order(self.market, "sell", s_rem_q, target_decay_p)
                                s_item["id"] = r_t.get("id")
                                s_item["p"] = float(r_t.get("price", target_decay_p))
                                s_item["q"] = float(r_t.get("origin_amount", s_rem_q))
                                decay_tag = "TIME-DECAY 15m" if s_age >= 900.0 else ("TIME-DECAY 7.5m" if s_age >= 450.0 else "DYNAMIC TARGET")
                                print(f"  🎯 [{decay_tag} -> ${s_item['p']:.2f}] Слот #{idx_s+3} (age={s_age:.0f}s, p_exh={getattr(self, 'last_p_exhaust', 0):.0%})")
                            except Exception:
                                pass
                        rem_extra.append(s_item)
                except Exception as es:
                    print(f"  ⚠️ [SYNC EXTRA SLOT WARN] id={sid}: {es}")
                    rem_extra.append(s_item)
            self.extra_slots = rem_extra
            self._save_wallet()
        if self.pos_qty >= 2.0 and self.active_ask_id is None and self.slot2_ask_id is None:
            try:
                open_ords = self.client.get_open_orders(self.market)
                sells = [o for o in open_ords if str(o.get("side", "")).lower() == "sell"]
                if sells:
                    sells.sort(key=lambda x: float(x.get("price", 0.0)), reverse=True)
                    top_s = sells[0]
                    self.active_ask_id = top_s.get("id")
                    self.active_ask_p = float(top_s.get("price", 0.0))
                    self.active_ask_q = float(top_s.get("remaining_volume", top_s.get("origin_amount", self.pos_qty)))
                    is_dedicated = (self.reserved_account_usdt <= 1e-4 and self.user_reserved_coin <= 1e-4)
                    self.user_manual_ask = not is_dedicated
                    self._save_wallet()
                    tag = "RECOVER BOT SELL" if is_dedicated else "ADOPT USER SELL"
                    print(f"  🤝 [{tag}] Подхвачен активный ордер продажи id={self.active_ask_id}: {self.active_ask_q:.4f} PRL @ ${self.active_ask_p:.2f}")
            except Exception:
                pass

    def sync_account_balances(self, b1: float = 0.0, init_mode: bool = False, from_trade_fill: bool = False):
        if self.mode != "live" or self.client is None:
            return
        try:
            raw_bals = self.client.get_spot_balances()
            b_map = {b["currency"]: (float(b.get("balance", 0.0)), float(b.get("locked", 0.0))) for b in raw_bals}
            u_free, u_lock = b_map.get("usdt", (0.0, 0.0))
            c_free, c_lock = b_map.get(self.base_currency, (0.0, 0.0))
            if self.reserved_account_usdt <= 1e-4 and self.user_reserved_coin <= 1e-4:
                is_24slot = self.is_24slot_active
                if is_24slot:
                    tracked_bid_oids = {s["id"] for s in self.bid_slots if s.get("id") is not None}
                    tracked_bid_usd = sum(float(s.get("qty", 0.0)) * float(s.get("px", 0.0)) for s in self.bid_slots if s.get("id") is not None)
                    max_bid_dist = max(7.0 * self.tick_size, 0.30 * b1) if b1 > 0 else 7.0 * self.tick_size
                    stale_tracked_bid = any(
                        s.get("id") is not None and b1 > 0 and float(s.get("px", 0.0)) <= b1 - max_bid_dist + 1e-6
                        for s in self.bid_slots
                    )
                else:
                    tracked_bid_oids = {self.active_bid_id} if self.active_bid_id is not None else set()
                    tracked_bid_usd = (self.active_bid_q * self.active_bid_p) if self.active_bid_id is not None else 0.0
                    stale_tracked_bid = (
                        self.active_bid_id is not None
                        and b1 > 0
                        and self.active_bid_p <= b1 - 2.0 * self.tick_size + 1e-6
                    )
                    max_bid_dist = 2.0 * self.tick_size
                if u_lock > tracked_bid_usd + 0.50 or stale_tracked_bid:
                    try:
                        open_ords = self.client.get_open_orders(self.market)
                        swept_any = False
                        for o_b in open_ords:
                            if str(o_b.get("side", "")).lower() != "buy":
                                continue
                            oid_b = o_b.get("id")
                            o_px = float(o_b.get("price", 0.0))
                            o_q = float(o_b.get("remaining_volume", o_b.get("origin_amount", 0.0)))
                            is_orphan = (oid_b not in tracked_bid_oids)
                            is_far_below = (not is_24slot and b1 > 0 and o_px <= b1 - max_bid_dist + 1e-6)
                            if is_orphan or is_far_below:
                                try:
                                    self.client.cancel_order(oid_b)
                                    swept_any = True
                                    reason = "осиротевший" if is_orphan else f"далеко от рынка (Bid=${b1:.2f})"
                                    print(f"  🧹 [SWEEP STALE BUY] Снят неактуальный ({reason}) ордер покупки id={oid_b}: {o_q:.4f} PRL @ ${o_px:.2f}")
                                    if is_24slot:
                                        for s in self.bid_slots:
                                            if s.get("id") == oid_b:
                                                s["id"], s["px"], s["qty"], s["act_ts"], s["acc_fill"] = None, 0.0, 0.0, 0.0, 0.0
                                        self._refresh_legacy_slot_mirrors()
                                    elif oid_b == self.active_bid_id:
                                        self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                                        self.bid_age_sec = 0
                                        self.bid_partial_filled = 0.0
                                except Exception:
                                    pass
                        if swept_any:
                            time.sleep(0.15)
                            raw_bals = self.client.get_spot_balances()
                            b_map = {b["currency"]: (float(b.get("balance", 0.0)), float(b.get("locked", 0.0))) for b in raw_bals}
                            u_free, u_lock = b_map.get("usdt", (0.0, 0.0))
                            c_free, c_lock = b_map.get(self.base_currency, (0.0, 0.0))
                    except Exception:
                        pass
            u_tot, c_tot = u_free + u_lock, c_free + c_lock
            self.ex_usdt_free, self.ex_coin_free = u_free, c_free

            # На выделенном аккаунте (reserved_account_usdt == 0 и user_reserved_coin == 0)
            # 100% спотового баланса биржи (USDT + PRL) является прямым источником истины для бота!
            if self.reserved_account_usdt <= 1e-4 and self.user_reserved_coin <= 1e-4:
                prev_pos = self.pos_qty
                prev_cash_tot = self.bot_cash_usd
                real_pos = math.floor(c_tot * 10000.0) / 10000.0 if c_tot >= 1.95 else 0.0
                px_ref = b1 if b1 > 0 else (self.avg_entry if self.avg_entry > 0 else (self.active_bid_p if self.active_bid_p > 0 else 1.77))
                prev_eq_est = prev_cash_tot + prev_pos * px_ref
                cur_eq_est = u_tot + real_pos * px_ref
                if not init_mode and prev_eq_est >= 5.0 and cur_eq_est < prev_eq_est * 0.5:
                    time.sleep(0.15)
                    raw_bals = self.client.get_spot_balances()
                    b_map = {b["currency"]: (float(b.get("balance", 0.0)), float(b.get("locked", 0.0))) for b in raw_bals}
                    u_free, u_lock = b_map.get("usdt", (0.0, 0.0))
                    c_free, c_lock = b_map.get(self.base_currency, (0.0, 0.0))
                    u_tot, c_tot = u_free + u_lock, c_free + c_lock
                    self.ex_usdt_free, self.ex_coin_free = u_free, c_free
                    real_pos = math.floor(c_tot * 10000.0) / 10000.0 if c_tot >= 1.95 else 0.0
                    if (u_tot + real_pos * px_ref) < prev_eq_est * 0.5:
                        return

                # Автоматический учет внешних пополнений монетами (PRL) или кэшем (USDT) без искажения PnL
                if not from_trade_fill and self.initial_budget_usd > 0.0:
                    dc_ext = real_pos - prev_pos
                    du_ext = u_tot - prev_cash_tot
                    if dc_ext >= 1.0 and du_ext >= -1.0:
                        dep_usd = round(dc_ext * px_ref, 6)
                        self.net_deposits_usd = round(self.net_deposits_usd + dep_usd, 6)
                        self.initial_budget_usd = round(self.initial_budget_usd + dep_usd, 6)
                        if prev_pos < 1.95:
                            self.avg_entry = px_ref
                            self.hold_sec = 0
                            self.peak_bid = px_ref
                        else:
                            self.avg_entry = round((prev_pos * self.avg_entry + dc_ext * px_ref) / real_pos, 6)
                            self.peak_bid = max(self.peak_bid, self.avg_entry, px_ref)
                            if self.active_ask_id is not None or self.slot2_ask_id is not None:
                                for oid_attr in ("active_ask_id", "slot2_ask_id"):
                                    oid = getattr(self, oid_attr)
                                    if oid is not None:
                                        try:
                                            self.client.cancel_order(oid)
                                        except Exception:
                                            pass
                                self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                                self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                                self.split_done_for_pos = False
                                self.ex_coin_free = real_pos
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
                if real_pos >= 1.95 and prev_pos < 1.95:
                    self.avg_entry = self.avg_entry if self.avg_entry > 0 else px_ref
                    self.peak_bid = max(self.peak_bid, self.avg_entry, px_ref)
                    if (self.tr_low_qty + self.tr_high_qty) <= 1e-4:
                        self.tr_high_px = self.avg_entry
                        self.tr_high_qty = real_pos
                    if not init_mode:
                        print(f"  🔄 [SPOT SYNC] Позиция на споте активна: {real_pos:.4f} PRL (вход=${self.avg_entry:.2f}) | Свободный USDT=${u_free:.2f}")
                elif real_pos < 1.95 and prev_pos >= 1.95:
                    self.avg_entry, self.hold_sec, self.peak_bid = 0.0, 0, 0.0
                    self.tr_low_px, self.tr_low_qty = 0.0, 0.0
                    self.tr_mid_px, self.tr_mid_qty = 0.0, 0.0
                    self.tr_high_px, self.tr_high_qty = 0.0, 0.0
                    self.n_trades += 1
                    self.realized_pnl_usd = u_tot - self.initial_budget_usd
                    if not init_mode:
                        print(f"  🔄 [SPOT SYNC] Позиция полностью продана в USDT: баланс=${u_tot:.2f} (PnL={self.realized_pnl_usd:+.4f}$)")
                self.pos_qty = real_pos
                self._reconcile_lifo_tranches()
                ind_locked_q = sum(float(s.get("q", 0.0)) for s in getattr(self, "extra_slots", []))
                if (
                    not is_24slot
                    and not init_mode
                    and real_pos >= 2.0
                    and c_free >= 2.0
                    and self.active_ask_id is not None
                    and self.slot2_ask_id is None
                    and (self.active_ask_q + ind_locked_q) < real_pos - 1.5
                ):
                    old_ask_id = self.active_ask_id
                    try:
                        self.client.cancel_order(old_ask_id)
                        print(f"  🔄 [MERGE ORPHAN COINS] Снят частичный Ask id={old_ask_id} ({self.active_ask_q:.4f} PRL) для объединения со свободными {c_free:.4f} PRL (всего {real_pos:.4f} PRL)")
                    except Exception:
                        pass
                    self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                    self.split_done_for_pos = False
                    self.ex_coin_free = max(0.0, real_pos - ind_locked_q)
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

    def _compute_order3_fast_state(self, row_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Сверхбыстрый (100% побитово эквивалентный) контур потоковых признаков стакана и EMA для Order-3 (~1.5 мс вместо ~95 мс):
        обновляет чистый L2 стакан без собственного импакта, 4-часовой диапазон и каузальные EMA без вызова устаревших сетей.
        """
        r = row_df.iloc[-1]
        exec_tick = float(getattr(self, "exec_tick_size", self.tick_size))
        tol = exec_tick * 0.25
        already_cleaned = float(r.get("_l2_cleaned", 0.0)) > 0.5
        b1_raw = float(max(r.get("ST_bid_1_price", r.get("ST_best_bid", 0.0)), exec_tick))
        a1_raw = float(max(r.get("ST_ask_1_price", r.get("ST_best_ask", b1_raw + exec_tick)), b1_raw + exec_tick))

        max_raw_lvl = 50 if "ST_bid_50_price" in r.index else 11
        bp_exec_list, bq_exec_list, ap_exec_list, aq_exec_list = [], [], [], []
        act_bids = [(float(s.get("px", 0.0)), float(s.get("qty", 0.0))) for s in self.bid_slots if s.get("id") is not None and float(s.get("qty", 0.0)) > 0.0]
        act_asks = [(float(s.get("px", 0.0)), float(s.get("qty", 0.0))) for s in self.ask_slots if s.get("id") is not None and float(s.get("qty", 0.0)) > 0.0]

        for i in range(1, max_raw_lvl + 1):
            bp = float(r.get(f"ST_bid_{i}_price", b1_raw if i == 1 else 0.0) or 0.0)
            bq = max(0.0, float(r.get(f"ST_bid_{i}_qty", r.get("ST_bid_qty_1", 0.0) if i == 1 else 0.0) or 0.0))
            if not already_cleaned and bp > 0.0 and act_bids:
                for spx, sq in act_bids:
                    if abs(bp - spx) <= tol:
                        bq = max(0.0, bq - sq)
            if bp > 0.0 and (bq > 1e-4 or i == max_raw_lvl):
                bp_exec_list.append(bp)
                bq_exec_list.append(bq)

            ap = float(r.get(f"ST_ask_{i}_price", a1_raw if i == 1 else 0.0) or 0.0)
            aq = max(0.0, float(r.get(f"ST_ask_{i}_qty", r.get("ST_ask_qty_1", 0.0) if i == 1 else 0.0) or 0.0))
            if not already_cleaned and ap > 0.0 and act_asks:
                for spx, sq in act_asks:
                    if abs(ap - spx) <= tol:
                        aq = max(0.0, aq - sq)
            if ap > 0.0 and (aq > 1e-4 or i == max_raw_lvl):
                ap_exec_list.append(ap)
                aq_exec_list.append(aq)

        if not bp_exec_list:
            bp_exec_list, bq_exec_list = [b1_raw], [0.0]
        if not ap_exec_list:
            ap_exec_list, aq_exec_list = [a1_raw], [0.0]

        b1 = float(max(bp_exec_list[0], exec_tick))
        a1 = float(max(ap_exec_list[0], b1 + exec_tick))
        self.last_clean_bp_10 = (bp_exec_list + [b1] * 10)[:10]
        self.last_clean_ap_10 = (ap_exec_list + [a1] * 10)[:10]
        self.last_clean_bq_10 = np.asarray((bq_exec_list + [0.0] * 10)[:10], dtype=np.float32)
        self.last_clean_aq_10 = np.asarray((aq_exec_list + [0.0] * 10)[:10], dtype=np.float32)

        self.last_b1_q = float(self.last_clean_bq_10[0])
        self.last_a1_q = float(self.last_clean_aq_10[0])
        self.last_bids_top3_q = float(np.sum(self.last_clean_bq_10[:3]))
        self.last_asks_top3_q = float(np.sum(self.last_clean_aq_10[:3]))
        b5_ex = float(np.sum(self.last_clean_bq_10[:5]))
        a5_ex = float(np.sum(self.last_clean_aq_10[:5]))
        b10_ex = float(np.sum(self.last_clean_bq_10[:10]))
        a10_ex = float(np.sum(self.last_clean_aq_10[:10]))
        self.last_imb3 = float(np.clip((self.last_bids_top3_q - self.last_asks_top3_q) / (self.last_bids_top3_q + self.last_asks_top3_q + 1e-5), -1.0, 1.0))
        self.last_imb5_exec = float(np.clip((b5_ex - a5_ex) / (b5_ex + a5_ex + 1e-5), -1.0, 1.0))
        self.last_imb5 = self.last_imb5_exec
        self.last_imb10 = float(np.clip((b10_ex - a10_ex) / (b10_ex + a10_ex + 1e-5), -1.0, 1.0))

        self.bid_hist_1h.append(b1)
        h_14400 = max(self.bid_hist_1h)
        l_14400 = min(self.bid_hist_1h)
        pos_14400 = (b1 - l_14400) / max(h_14400 - l_14400, self.tick_size)
        self.last_pos14400 = pos_14400
        self.last_range_pos_4h = float(np.clip(2.0 * pos_14400 - 1.0, -1.0, 1.0))
        self.last_range_amp_4h = float(np.clip(((h_14400 - l_14400) / max(l_14400, 1e-4)) * 4.0, 0.0, 2.5))

        if not hasattr(self, "flow_hist_10s"):
            self.flow_hist_10s = deque(maxlen=11)
        if not hasattr(self, "flow_ts_hist_30s"):
            self.flow_ts_hist_30s = deque(maxlen=45)
        buy_v1 = float(r.get("ST_buy_vol", r.get("ST_buy_vol_1s", 0.0)) or 0.0)
        sell_v1 = float(r.get("ST_sell_vol", r.get("ST_sell_vol_1s", 0.0)) or 0.0)
        self.flow_hist_10s.append((self.last_b1_q, self.last_a1_q, buy_v1, sell_v1))
        self.flow_ts_hist_30s.append((time.time(), buy_v1, sell_v1))
        self.last_buy_vol_10s = float(sum(x[2] for x in list(self.flow_hist_10s)[-10:]))

        st_mid_now = max(1e-4, 0.5 * (b1 + a1))
        if not hasattr(self, "_o3_ema_state") or self._o3_ema_state is None:
            bo_b1_init = float(r.get("BO_bid_1_price", r.get("BO_best_bid", 0.0)) or 0.0)
            bo_a1_init = float(r.get("BO_ask_1_price", r.get("BO_best_ask", 0.0)) or 0.0)
            bo_m_init = 0.5 * (bo_b1_init + bo_a1_init) if (bo_b1_init > 1e-4 and bo_a1_init > 1e-4) else st_mid_now
            self._o3_ema_state = {
                "b15": buy_v1, "s15": sell_v1,
                "b30": buy_v1, "s30": sell_v1,
                "b60": buy_v1, "s60": sell_v1,
                "bo_b30": 0.0, "bo_s30": 0.0,
                "bo_m30": bo_m_init,
                "m5": st_mid_now, "m15": st_mid_now, "m30": st_mid_now, "m60": st_mid_now, "m120": st_mid_now,
                "m900": st_mid_now, "m3600": st_mid_now,
            }
        es = self._o3_ema_state
        es.setdefault("m900", st_mid_now)
        es.setdefault("m3600", st_mid_now)
        for k_w, a_w in (("15", 2.0 / 16.0), ("30", 2.0 / 31.0), ("60", 2.0 / 61.0)):
            es[f"b{k_w}"] = (1.0 - a_w) * es[f"b{k_w}"] + a_w * buy_v1
            es[f"s{k_w}"] = (1.0 - a_w) * es[f"s{k_w}"] + a_w * sell_v1
        self.last_flow_15s = float(np.clip((es["b15"] - es["s15"]) / (es["b15"] + es["s15"] + 1e-5), -1.0, 1.0))
        self.last_flow_30s = float(np.clip((es["b30"] - es["s30"]) / (es["b30"] + es["s30"] + 1e-5), -1.0, 1.0))
        self.last_flow_60s = float(np.clip((es["b60"] - es["s60"]) / (es["b60"] + es["s60"] + 1e-5), -1.0, 1.0))

        for k_m, a_m in (("m5", 2.0 / 6.0), ("m15", 2.0 / 16.0), ("m30", 2.0 / 31.0), ("m60", 2.0 / 61.0), ("m120", 2.0 / 121.0), ("m900", 2.0 / 901.0), ("m3600", 2.0 / 3601.0)):
            es[k_m] = (1.0 - a_m) * es[k_m] + a_m * st_mid_now
        self.last_ret_5s = float(np.clip(((st_mid_now - es["m5"]) / st_mid_now) * 50.0, -2.0, 2.0))
        self.last_ret_15s = float(np.clip(((st_mid_now - es["m15"]) / st_mid_now) * 40.0, -2.0, 2.0))
        self.last_ret_30s = float(np.clip(((st_mid_now - es["m30"]) / st_mid_now) * 30.0, -2.0, 2.0))
        self.last_ret_60s = float(np.clip(((st_mid_now - es["m60"]) / st_mid_now) * 20.0, -2.0, 2.0))
        self.last_ret_120s = float(np.clip(((st_mid_now - es["m120"]) / st_mid_now) * 15.0, -2.0, 2.0))
        self.last_ret_15m = float(np.clip(((st_mid_now - es["m900"]) / st_mid_now) * 10.0, -2.5, 2.5))
        self.last_ret_1h = float(np.clip(((st_mid_now - es["m3600"]) / st_mid_now) * 6.0, -2.5, 2.5))
        self.last_vol_10s = float(np.clip(abs(self.last_ret_5s) * 2.0, 0.0, 2.0))

        bo_b1_v = float(r.get("BO_bid_1_price", r.get("BO_best_bid", 0.0)) or 0.0)
        bo_a1_v = float(r.get("BO_ask_1_price", r.get("BO_best_ask", 0.0)) or 0.0)
        self.last_bo_b1 = bo_b1_v
        self.last_bo_a1 = bo_a1_v
        self.last_bo_age_sec = float(r.get("BO_sec_since_tick", 0.0) or 0.0)
        if bo_b1_v > 1e-4 and bo_a1_v > 1e-4:
            bo_bq1 = max(0.0, float(r.get("BO_bid_1_qty", 0.0) or 0.0))
            bo_aq1 = max(0.0, float(r.get("BO_ask_1_qty", 0.0) or 0.0))
            bo_bq3 = bo_bq1 + max(0.0, float(r.get("BO_bid_2_qty", 0.0) or 0.0)) + max(0.0, float(r.get("BO_bid_3_qty", 0.0) or 0.0))
            bo_aq3 = bo_aq1 + max(0.0, float(r.get("BO_ask_2_qty", 0.0) or 0.0)) + max(0.0, float(r.get("BO_ask_3_qty", 0.0) or 0.0))
            bo_bv1 = max(0.0, float(r.get("BO_buy_vol", r.get("BO_buy_vol_1s", 0.0)) or 0.0))
            bo_sv1 = max(0.0, float(r.get("BO_sell_vol", r.get("BO_sell_vol_1s", 0.0)) or 0.0))
            a_bo30 = 2.0 / 31.0
            es["bo_b30"] = (1.0 - a_bo30) * es["bo_b30"] + a_bo30 * bo_bv1
            es["bo_s30"] = (1.0 - a_bo30) * es["bo_s30"] + a_bo30 * bo_sv1
            bo_mid_v = 0.5 * (bo_b1_v + bo_a1_v)
            es["bo_m30"] = (1.0 - a_bo30) * es["bo_m30"] + a_bo30 * bo_mid_v
            self.last_bo_b1_q = bo_bq1
            self.last_bo_a1_q = bo_aq1
            self.last_bo_imb1 = float(np.clip((bo_bq1 - bo_aq1) / (bo_bq1 + bo_aq1 + 1e-5), -1.0, 1.0))
            self.last_bo_imb3 = float(np.clip((bo_bq3 - bo_aq3) / (bo_bq3 + bo_aq3 + 1e-5), -1.0, 1.0))
            self.last_bo_flow_30s = float(np.clip((es["bo_b30"] - es["bo_s30"]) / (es["bo_b30"] + es["bo_s30"] + 1e-5), -1.0, 1.0))
            self.last_bo_ret_30s = float(np.clip(((bo_mid_v - es["bo_m30"]) / bo_mid_v) * 50.0, -2.0, 2.0))
        self.last_features = r.to_dict()

        obs_stub = getattr(self, "_order3_obs_stub", None)
        if obs_stub is None:
            in_d_stub = int(getattr(self.agent.backbone[0], "in_features", 49))
            obs_stub = np.zeros(in_d_stub, dtype=np.float32)
            self._order3_obs_stub = obs_stub
        mask_stub = np.array([True, False, False, False, False, False, False], dtype=bool)
        return obs_stub, mask_stub

    def compute_obs_and_mask(self, X_seq_2d: np.ndarray, row_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if bool(getattr(self, "order3_mode", False)) and getattr(self, "order3_engine", None) is not None:
            return self._compute_order3_fast_state(row_df)
        r_last = row_df.iloc[-1]
        b1_chk = float(r_last.get("ST_bid_1_price", r_last.get("ST_best_bid", 0.0)))
        a1_chk = float(r_last.get("ST_ask_1_price", r_last.get("ST_best_ask", 0.0)))

        run_slow = True  # default, overridden below

        if bool(getattr(self, "l28_hierarchical_mode", False)):
            self._ensure_l28_tree_routers()
            idx_last = row_df.index[-1]
            now_ts_tree = int(idx_last.timestamp()) if hasattr(idx_last, "timestamp") else int(time.time())
            x_st_row = np.array(
                [float(r_last.get(fn, 0.0) or 0.0) for fn in self.st_tree_router.feature_names],
                dtype=np.float32,
            )
            x_bo_row = np.array(
                [float(r_last.get(fn, 0.0) or 0.0) for fn in self.bo_tree_router.feature_names],
                dtype=np.float32,
            )
            self.st_x_ring.append(x_st_row)
            self.bo_x_ring.append(x_bo_row)
            st_ring_2d = np.asarray(self.st_x_ring, dtype=np.float32)
            bo_ring_2d = np.asarray(self.bo_x_ring, dtype=np.float32)
            bo_age_chk = float(r_last.get("BO_sec_since_tick", 0.0) or 0.0)
            st_t42, st_mat7, st_coh4 = self.st_tree_router.predict_live_1row(
                st_ring_2d, r_last, now_ts_tree, is_stale=False
            )
            try:
                bo_t42, bo_mat7, bo_coh4 = self.bo_tree_router.predict_live_1row(
                    bo_ring_2d, r_last, now_ts_tree, is_stale=(bo_age_chk > 120.0)
                )
            except Exception as _bo_err:
                import logging
                logging.getLogger(__name__).warning("BO tree predict error (fallback to zeros): %s", _bo_err)
                bo_t42 = np.zeros(42, dtype=np.float32)
                bo_mat7 = np.zeros(7, dtype=np.float32)
                bo_coh4 = np.zeros(4, dtype=np.float32)
            self.last_l28_st_tree_42 = st_t42
            self.last_l28_st_mat_7 = st_mat7
            self.last_l28_st_coh_4 = st_coh4
            self.last_l28_bo_tree_42 = bo_t42
            self.last_l28_bo_mat_7 = bo_mat7
            self.last_l28_bo_coh_4 = bo_coh4
            triad_vec = [float(v) for v in st_t42[12:30]]
            self.last_triad_probs = [np.full(7, 1.0 / 7.0, dtype=np.float32) for _ in range(3)]
            self.last_ez60 = float(st_t42[30])
            self.last_p60 = np.full(7, 1.0 / 7.0, dtype=np.float32)
            self.last_vec60 = [float(v) for v in st_t42[30:36]]
            self.last_ez120 = float(st_t42[36])
            self.last_vec120 = [float(v) for v in st_t42[36:42]]
            self.last_ez90 = 0.5 * (self.last_ez60 + self.last_ez120)
            self.last_vec90 = [self.last_ez90, 0.33, 0.33, 0.10, 0.10, 0.90]
            run_slow = True  # L28 tree mode: always refresh regime_gate
        else:
            self._slow_router_step = getattr(self, "_slow_router_step", 0) + 1
            prev_ba = getattr(self, "_slow_router_ba", None)
            run_slow = (
                (self._slow_router_step % 3 == 1)
                or (prev_ba != (b1_chk, a1_chk))
                or (getattr(self, "_cached_slow_res", None) is None)
            )

            # Быстрая триада (H=5, 12, 25) считается каждую секунду; макро-горизонты (H=60, 90, 120) — при любом сдвиге цены или раз в 3с
            fast_routers = [self.routers[0][2], self.routers[1][2], self.routers[2][2]]
            slow_routers = [r for r in [self.router_60, getattr(self, "router_90", None), getattr(self, "router_120", None)] if r is not None]

            shared_reg_probs = None
            lr_ref = getattr(fast_routers[0], "learned_router", None)
            if lr_ref is not None and hasattr(lr_ref, "predict_live_1row"):
                live_vels = None
                bh_len = len(self.bid_hist_1h)
                if bh_len >= 2 and b1_chk > 0:
                    b_10 = self.bid_hist_1h[-min(bh_len, 10)]
                    b_60 = self.bid_hist_1h[-min(bh_len, 60)]
                    b_300 = self.bid_hist_1h[-min(bh_len, 300)]
                    live_vels = (
                        (b1_chk - b_10) / self.tick_size,
                        (b1_chk - b_60) / self.tick_size,
                        (b1_chk - b_300) / self.tick_size,
                    )
                shared_reg_probs = lr_ref.predict_live_1row(row_df, live_vels=live_vels)

            if run_slow:
                all_routers = fast_routers + slow_routers
                futs = [self._triad_pool.submit(rt.predict_live_tick, X_seq_2d, row_df, shared_reg_probs) for rt in all_routers]
                res_all = [f.result() for f in futs]
                res_map = res_all[:3]
                self._cached_slow_res = res_all[3:]
                self._slow_router_ba = (b1_chk, a1_chk)
            else:
                futs = [self._triad_pool.submit(rt.predict_live_tick, X_seq_2d, row_df, shared_reg_probs) for rt in fast_routers]
                res_map = [f.result() for f in futs]
            res_map = res_map + list(self._cached_slow_res)

            triad_vec = []
            triad_probs = []
            for i in range(3):
                res = res_map[i]
                p = res["probs"]
                triad_probs.append(np.asarray(p, dtype=np.float32))
                ez = float(res["expected_z"])
                p_up = float(p[4] + p[5] + p[6])
                p_down = float(p[0] + p[1] + p[2])
                p_tp = float(p[5] + p[6])
                p_tn = float(p[0] + p[1])
                triad_vec.extend([ez, p_up, p_down, p_tp, p_tn, float(res["entropy"])])
            self.last_triad_probs = triad_probs

            if getattr(self, "router_60", None) is not None and len(res_map) > 3:
                res60 = res_map[3]
                self.last_ez60 = float(res60["expected_z"])
                self.last_p60 = np.asarray(res60["probs"], dtype=np.float32)
                p60 = self.last_p60
                self.last_vec60 = [
                    self.last_ez60,
                    float(p60[4] + p60[5] + p60[6]),
                    float(p60[0] + p60[1] + p60[2]),
                    float(p60[5] + p60[6]),
                    float(p60[0] + p60[1]),
                    float(res60["entropy"]),
                ]
            else:
                self.last_ez60 = float(getattr(self, "last_ez25", 0.0))
                p60 = np.asarray(self.last_triad_probs[-1] if self.last_triad_probs else np.zeros(7, dtype=np.float32), dtype=np.float32)
                self.last_p60 = p60
                self.last_vec60 = [
                    self.last_ez60,
                    float(p60[4] + p60[5] + p60[6]),
                    float(p60[0] + p60[1] + p60[2]),
                    float(p60[5] + p60[6]),
                    float(p60[0] + p60[1]),
                    0.5,
                ]
            curr_idx = 4
            self.last_vec90 = [0.0, 0.33, 0.33, 0.10, 0.10, 0.90]
            self.last_vec120 = [0.0, 0.33, 0.33, 0.10, 0.10, 0.90]
            if getattr(self, "router_90", None) is not None:
                r90 = res_map[curr_idx]
                p90 = np.asarray(r90["probs"], dtype=np.float32)
                self.last_ez90 = float(r90["expected_z"])
                self.last_vec90 = [
                    self.last_ez90,
                    float(p90[4] + p90[5] + p90[6]),
                    float(p90[0] + p90[1] + p90[2]),
                    float(p90[5] + p90[6]),
                    float(p90[0] + p90[1]),
                    float(r90["entropy"]),
                ]
                curr_idx += 1
            if getattr(self, "router_120", None) is not None:
                r120 = res_map[curr_idx]
                p120 = np.asarray(r120["probs"], dtype=np.float32)
                self.last_ez120 = float(r120["expected_z"])
                self.last_vec120 = [
                    self.last_ez120,
                    float(p120[4] + p120[5] + p120[6]),
                    float(p120[0] + p120[1] + p120[2]),
                    float(p120[5] + p120[6]),
                    float(p120[0] + p120[1]),
                    float(r120["entropy"]),
                ]

        r = row_df.iloc[-1]
        exec_tick = float(getattr(self, "exec_tick_size", self.tick_size))
        tol = exec_tick * 0.25
        already_cleaned = float(r.get("_l2_cleaned", 0.0)) > 0.5
        b1_raw = float(max(r.get("ST_bid_1_price", r.get("ST_best_bid", 0.0)), exec_tick))
        a1_raw = float(max(r.get("ST_ask_1_price", r.get("ST_best_ask", b1_raw + exec_tick)), b1_raw + exec_tick))

        # Вычитаем собственный импакт (наши активные лимитные заявки 24 слотов / Slot 1..2) из до 50 уровней стакана L2 со сдвигом пустых уровней
        is_24slot = self.is_24slot_active
        max_raw_lvl = 50 if "ST_bid_50_price" in r.index else 11
        bp_exec_list, bq_exec_list, ap_exec_list, aq_exec_list = [], [], [], []
        for i in range(1, max_raw_lvl + 1):
            bp = float(r.get(f"ST_bid_{i}_price", b1_raw if i == 1 else 0.0) or 0.0)
            bq = max(0.0, float(r.get(f"ST_bid_{i}_qty", r.get("ST_bid_qty_1", 0.0) if i == 1 else 0.0) or 0.0))
            if not already_cleaned and bp > 0.0:
                if is_24slot:
                    for s_b in self.bid_slots:
                        if s_b.get("id") is not None and float(s_b.get("qty", 0.0)) > 0 and abs(bp - float(s_b.get("px", 0.0))) <= tol:
                            bq = max(0.0, bq - float(s_b.get("qty", 0.0)))
                elif self.active_bid_q > 0 and abs(bp - self.active_bid_p) <= tol:
                    bq = max(0.0, bq - self.active_bid_q)
            if bp > 0 and (bq > 1e-4 or i == max_raw_lvl):
                bp_exec_list.append(bp)
                bq_exec_list.append(bq)

            ap = float(r.get(f"ST_ask_{i}_price", a1_raw if i == 1 else 0.0) or 0.0)
            aq = max(0.0, float(r.get(f"ST_ask_{i}_qty", r.get("ST_ask_qty_1", 0.0) if i == 1 else 0.0) or 0.0))
            if not already_cleaned and ap > 0.0:
                if is_24slot:
                    for s_a in self.ask_slots:
                        if s_a.get("id") is not None and float(s_a.get("qty", 0.0)) > 0 and abs(ap - float(s_a.get("px", 0.0))) <= tol:
                            aq = max(0.0, aq - float(s_a.get("qty", 0.0)))
                else:
                    if self.active_ask_q > 0 and abs(ap - self.active_ask_p) <= tol:
                        aq = max(0.0, aq - self.active_ask_q)
                    if self.slot2_ask_q > 0 and abs(ap - self.slot2_ask_p) <= tol:
                        aq = max(0.0, aq - self.slot2_ask_q)
                    if getattr(self, "extra_slots", None):
                        for s_ex in self.extra_slots:
                            if float(s_ex.get("q", 0.0)) > 0 and abs(ap - float(s_ex.get("p", 0.0))) <= tol:
                                aq = max(0.0, aq - float(s_ex.get("q", 0.0)))
            if ap > 0 and (aq > 1e-4 or i == max_raw_lvl):
                ap_exec_list.append(ap)
                aq_exec_list.append(aq)

        if not bp_exec_list:
            bp_exec_list, bq_exec_list = [b1_raw], [0.0]
        if not ap_exec_list:
            ap_exec_list, aq_exec_list = [a1_raw], [0.0]
        b1 = float(max(bp_exec_list[0], exec_tick))
        a1 = float(max(ap_exec_list[0], b1 + exec_tick))
        bp_exec_np = np.asarray(bp_exec_list, dtype=np.float32)
        bq_exec_np = np.maximum(0.0, np.asarray(bq_exec_list, dtype=np.float32))
        ap_exec_np = np.asarray(ap_exec_list, dtype=np.float32)
        aq_exec_np = np.maximum(0.0, np.asarray(aq_exec_list, dtype=np.float32))

        # Масштабно-инвариантная агрегация стакана в 10 эталонных корзин feat_tick ($0.01) для центовых стволов MultiTaskLOBNet / CrossVenueNet,
        # при сохранении сырой 4-десятичной книги bp_exec_np / ap_exec_np для Sparse-LOB очередей и исполнения $0.0001
        feat_tick = float(self.tick_size)
        if exec_tick < feat_tick * 0.5:
            b_cent_top = math.floor((b1 + 1e-7) / feat_tick) * feat_tick
            a_cent_top = max(b_cent_top + feat_tick, math.ceil((a1 - 1e-7) / feat_tick) * feat_tick)
            bid_ps = [round(max(feat_tick, b_cent_top - k * feat_tick), 6) for k in range(10)]
            ask_ps = [round(a_cent_top + k * feat_tick, 6) for k in range(10)]
            bid_qs_clean = [0.0] * 10
            ask_qs_clean = [0.0] * 10
            for p_lv, q_lv in zip(bp_exec_list, bq_exec_list):
                if p_lv > 0.0 and q_lv > 0.0:
                    k_b = int(math.floor((b_cent_top - p_lv + 1e-7) / feat_tick))
                    if 0 <= k_b < 10:
                        bid_qs_clean[k_b] += q_lv
            for p_lv, q_lv in zip(ap_exec_list, aq_exec_list):
                if p_lv > 0.0 and q_lv > 0.0:
                    k_a = int(math.floor((p_lv - a_cent_top + 1e-7) / feat_tick))
                    if 0 <= k_a < 10:
                        ask_qs_clean[k_a] += q_lv
            spread_ticks = float(np.clip((a_cent_top - b_cent_top) / feat_tick, 1.0, 50.0))
        else:
            bid_ps = (bp_exec_list + [b1_raw] * 10)[:10]
            bid_qs_clean = (bq_exec_list + [0.0] * 10)[:10]
            ask_ps = (ap_exec_list + [a1_raw] * 10)[:10]
            ask_qs_clean = (aq_exec_list + [0.0] * 10)[:10]
            spread_ticks = float(np.clip((a1 - b1) / feat_tick, 0.0, 50.0))

        b1_q, a1_q = bid_qs_clean[0], ask_qs_clean[0]
        b5_q, a5_q = sum(bid_qs_clean[:5]), sum(ask_qs_clean[:5])
        b10_q, a10_q = sum(bid_qs_clean), sum(ask_qs_clean)

        self.bid_hist_1h.append(b1)
        ez_5, ez_12, ez_25 = triad_vec[0], triad_vec[6], triad_vec[12]
        p_tn_5, p_dn_12 = triad_vec[4], triad_vec[8]
        self.ez12_hist.append(ez_12)
        imb_l1 = (b1_q - a1_q) / (b1_q + a1_q + 1e-6)
        imb_l5 = (b5_q - a5_q) / (b5_q + a5_q + 1e-6)
        imb_l10 = (b10_q - a10_q) / (b10_q + a10_q + 1e-6)
        vol_300 = float(r.get("ST_vol_300s_ticks", 1.0))
        ofi_300 = float(r.get("ST_ofi_norm_300s", 0.0))
        if not hasattr(self, "flow_hist_10s"):
            self.flow_hist_10s = deque(maxlen=11)
        buy_v1 = float(r.get("ST_buy_vol", r.get("ST_buy_vol_1s", 0.0)))
        sell_v1 = float(r.get("ST_sell_vol", r.get("ST_sell_vol_1s", 0.0)))
        self.flow_hist_10s.append((b1_q, a1_q, buy_v1, sell_v1))
        net_tr_10 = sum(x[2] - x[3] for x in list(self.flow_hist_10s)[-10:])
        dq_10 = (b1_q - self.flow_hist_10s[0][0]) - (a1_q - self.flow_hist_10s[0][1])
        fast_flow_10s = float(np.clip((net_tr_10 + 0.5 * dq_10) / (b1_q + a1_q + 500.0) * 3.0, -3.0, 3.0))

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
        ez_60 = float(getattr(self, "last_ez60", ez_25))
        inertia = float(
            0.15 * ez_60 + 0.35 * ez_25 + 0.30 * ez_12 + 0.20 * ez_5
            + 0.25 * d_ez12 + 0.35 * imb_l5 + 0.15 * imb_l1
            + 0.10 * np.clip(vel_15, -3.0, 3.0) + 0.08 * max(0.0, fast_flow_10s)
        )
        fh = list(self.flow_hist_10s)
        b1_5s_ago = b_arr[-6] if len(b_arr) >= 6 else b_arr[0]
        same_b1_5s = abs(b1 - b1_5s_ago) < (self.tick_size * 0.25)
        b1q_5s_ago = fh[-6][0] if len(fh) >= 6 else fh[0][0]
        a1q_5s_ago = fh[-6][1] if len(fh) >= 6 else fh[0][1]
        self.last_dq_bid_5s = float(b1_q - b1q_5s_ago) if same_b1_5s else 0.0
        self.last_buy_vol_10s = float(sum(x[2] for x in fh[-10:]))
        sell_vol_10s = float(sum(x[3] for x in fh[-10:]))
        buy_vol_30s = float(sum(x[2] for x in fh[-30:]))
        sell_vol_30s = float(sum(x[3] for x in fh[-30:]))
        ez_90 = float(getattr(self, "last_ez90", ez_60))
        ez_120 = float(getattr(self, "last_ez120", ez_90))
        if getattr(self, "carousel_net", None) is not None and self.carousel_net.ready:
            two_sided = min(buy_vol_30s, sell_vol_30s) / (max(b1_q, a1_q) + 50.0)
            feat47 = np.array(
                triad_vec + lob_vec + self.last_vec60 + [
                    math.log1p(max(0.0, b1_q)),
                    math.log1p(max(0.0, a1_q)),
                    math.log1p(max(0.0, float(sum(bid_qs_clean[:5])))),
                    math.log1p(max(0.0, float(sum(ask_qs_clean[:5])))),
                    math.log1p(max(0.0, self.last_buy_vol_10s)),
                    math.log1p(max(0.0, sell_vol_10s)),
                    math.log1p(max(0.0, buy_vol_30s)),
                    math.log1p(max(0.0, sell_vol_30s)),
                    float(np.clip(spread_ticks, 1.0, 20.0)),
                    float(np.clip(two_sided, 0.0, 10.0)),
                    ez_90,
                    ez_120,
                    ez_12 - ez_120,
                ],
                dtype=np.float32,
            )
            self.last_car_probs, self.last_car_cap_prl = self.carousel_net.predict_single(feat47)
            if getattr(self, "hazard_net", None) is not None and self.hazard_net.ready:
                pa, pe, ps = self.hazard_net.predict_batch_v2(feat47[None, :])
                self.last_p_adverse, self.last_p_exhaust, self.last_p_ask_sweep = float(pa[0]), float(pe[0]), float(ps[0])
        if getattr(self, "realism_net", None) is not None and self.realism_net.ready:
            buy_15s = float(sum(x[2] for x in fh[-15:]))
            sell_15s = float(sum(x[3] for x in fh[-15:]))
            est_ord_q = max(20.0, (self.order_usd * 0.25) / max(b1, self.tick_size))
            feat16 = np.array([
                float(np.clip(spread_ticks, 1.0, 20.0)),
                imb_l1,
                imb_l5,
                float(np.log1p(b1_q) / 10.0),
                float(np.log1p(a1_q) / 10.0),
                float(np.log1p(b5_q) / 12.0),
                float(np.log1p(a5_q) / 12.0),
                float(np.log1p(max(0.0, buy_15s)) / 8.0),
                float(np.log1p(max(0.0, sell_15s)) / 8.0),
                float(est_ord_q / (b1_q + est_ord_q + 10.0)),
                float(est_ord_q / (a1_q + est_ord_q + 10.0)),
                ez_5,
                ez_12,
                ez_25,
                ez_90,
                ez_120,
            ], dtype=np.float32)
            self.last_p_fill_real, self.last_p_frontrun = self.realism_net.predict_single(feat16)
        self.last_ofi_5s = float(self.last_dq_bid_5s - (a1_q - a1q_5s_ago) + sum(x[2] - x[3] for x in fh[-5:]))
        self.last_ez5 = ez_5
        self.last_ez25 = ez_25
        self.last_ez12 = ez_12
        self.last_inertia = inertia
        self.last_pos3600 = pos_3600
        self.last_rng14400 = rng_14400
        self.last_pos14400 = pos_14400
        self.last_b1_q = float(bq_exec_np[0]) if len(bq_exec_np) > 0 else b1_q
        self.last_b2_q = float(bid_qs_clean[1]) if len(bid_qs_clean) > 1 else 0.0
        self.last_a1_q = float(aq_exec_np[0]) if len(aq_exec_np) > 0 else a1_q
        self.last_vel10 = vel_10
        self.last_imb5 = imb_l5
        self.last_dd300 = dd_300
        self.last_dd900 = dd_900
        self.last_sp_ticks = spread_ticks

        # Полная потоковая синхронизация всех 40D признаков для Order-3 (1-в-1 с build_hybrid_market_obs_40d.py):
        self.last_clean_bp_10 = (bp_exec_list + [b1] * 10)[:10]
        self.last_clean_ap_10 = (ap_exec_list + [a1] * 10)[:10]
        self.last_clean_bq_10 = np.asarray((bq_exec_list + [0.0] * 10)[:10], dtype=np.float32)
        self.last_clean_aq_10 = np.asarray((aq_exec_list + [0.0] * 10)[:10], dtype=np.float32)
        self.last_bids_top3_q = float(np.sum(self.last_clean_bq_10[:3]))
        self.last_asks_top3_q = float(np.sum(self.last_clean_aq_10[:3]))
        b5_ex = float(np.sum(self.last_clean_bq_10[:5]))
        a5_ex = float(np.sum(self.last_clean_aq_10[:5]))
        b10_ex = float(np.sum(self.last_clean_bq_10[:10]))
        a10_ex = float(np.sum(self.last_clean_aq_10[:10]))
        self.last_imb3 = float(np.clip((self.last_bids_top3_q - self.last_asks_top3_q) / (self.last_bids_top3_q + self.last_asks_top3_q + 1e-5), -1.0, 1.0))
        self.last_imb5_exec = float(np.clip((b5_ex - a5_ex) / (b5_ex + a5_ex + 1e-5), -1.0, 1.0))
        self.last_imb10 = float(np.clip((b10_ex - a10_ex) / (b10_ex + a10_ex + 1e-5), -1.0, 1.0))

        st_mid_now = max(1e-4, 0.5 * (b1 + a1))
        if not hasattr(self, "_o3_ema_state") or self._o3_ema_state is None:
            bo_b1_init = float(r.get("BO_bid_1_price", r.get("BO_best_bid", 0.0)) or 0.0)
            bo_a1_init = float(r.get("BO_ask_1_price", r.get("BO_best_ask", 0.0)) or 0.0)
            bo_m_init = 0.5 * (bo_b1_init + bo_a1_init) if (bo_b1_init > 1e-4 and bo_a1_init > 1e-4) else st_mid_now
            self._o3_ema_state = {
                "b15": buy_v1, "s15": sell_v1,
                "b30": buy_v1, "s30": sell_v1,
                "b60": buy_v1, "s60": sell_v1,
                "bo_b30": 0.0, "bo_s30": 0.0,
                "bo_m30": bo_m_init,
                "m5": st_mid_now, "m15": st_mid_now, "m30": st_mid_now, "m60": st_mid_now, "m120": st_mid_now,
                "m900": st_mid_now, "m3600": st_mid_now,
            }
        es = self._o3_ema_state
        if "m900" not in es:
            es["m900"] = st_mid_now
            es["m3600"] = st_mid_now
        for k_w, a_w in (("15", 2.0 / 16.0), ("30", 2.0 / 31.0), ("60", 2.0 / 61.0)):
            es[f"b{k_w}"] = (1.0 - a_w) * es[f"b{k_w}"] + a_w * buy_v1
            es[f"s{k_w}"] = (1.0 - a_w) * es[f"s{k_w}"] + a_w * sell_v1
        self.last_flow_15s = float(np.clip((es["b15"] - es["s15"]) / (es["b15"] + es["s15"] + 1e-5), -1.0, 1.0))
        self.last_flow_30s = float(np.clip((es["b30"] - es["s30"]) / (es["b30"] + es["s30"] + 1e-5), -1.0, 1.0))
        self.last_flow_60s = float(np.clip((es["b60"] - es["s60"]) / (es["b60"] + es["s60"] + 1e-5), -1.0, 1.0))

        for k_m, a_m in (("m5", 2.0 / 6.0), ("m15", 2.0 / 16.0), ("m30", 2.0 / 31.0), ("m60", 2.0 / 61.0), ("m120", 2.0 / 121.0), ("m900", 2.0 / 901.0), ("m3600", 2.0 / 3601.0)):
            es[k_m] = (1.0 - a_m) * es[k_m] + a_m * st_mid_now
        self.last_ret_5s = float(np.clip(((st_mid_now - es["m5"]) / st_mid_now) * 50.0, -2.0, 2.0))
        self.last_ret_15s = float(np.clip(((st_mid_now - es["m15"]) / st_mid_now) * 40.0, -2.0, 2.0))
        self.last_ret_30s = float(np.clip(((st_mid_now - es["m30"]) / st_mid_now) * 30.0, -2.0, 2.0))
        self.last_ret_60s = float(np.clip(((st_mid_now - es["m60"]) / st_mid_now) * 20.0, -2.0, 2.0))
        self.last_ret_120s = float(np.clip(((st_mid_now - es["m120"]) / st_mid_now) * 15.0, -2.0, 2.0))
        self.last_ret_15m = float(np.clip(((st_mid_now - es["m900"]) / st_mid_now) * 10.0, -2.5, 2.5))
        self.last_ret_1h = float(np.clip(((st_mid_now - es["m3600"]) / st_mid_now) * 6.0, -2.5, 2.5))
        self.last_range_pos_4h = float(np.clip(2.0 * pos_14400 - 1.0, -1.0, 1.0))
        self.last_range_amp_4h = float(np.clip(((h_14400 - l_14400) / max(l_14400, 1e-4)) * 4.0, 0.0, 2.5))
        self.last_vol_10s = float(np.clip(abs(self.last_ret_5s) * 2.0, 0.0, 2.0))

        bo_b1_v = float(r.get("BO_bid_1_price", r.get("BO_best_bid", 0.0)) or 0.0)
        bo_a1_v = float(r.get("BO_ask_1_price", r.get("BO_best_ask", 0.0)) or 0.0)
        self.last_bo_b1 = bo_b1_v
        self.last_bo_a1 = bo_a1_v
        self.last_bo_age_sec = float(r.get("BO_sec_since_tick", 0.0) or 0.0)
        if bo_b1_v > 1e-4 and bo_a1_v > 1e-4:
            bo_bq1 = max(0.0, float(r.get("BO_bid_1_qty", 0.0) or 0.0))
            bo_aq1 = max(0.0, float(r.get("BO_ask_1_qty", 0.0) or 0.0))
            bo_bq3 = bo_bq1 + max(0.0, float(r.get("BO_bid_2_qty", 0.0) or 0.0)) + max(0.0, float(r.get("BO_bid_3_qty", 0.0) or 0.0))
            bo_aq3 = bo_aq1 + max(0.0, float(r.get("BO_ask_2_qty", 0.0) or 0.0)) + max(0.0, float(r.get("BO_ask_3_qty", 0.0) or 0.0))
            bo_bv1 = max(0.0, float(r.get("BO_buy_vol", r.get("BO_buy_vol_1s", 0.0)) or 0.0))
            bo_sv1 = max(0.0, float(r.get("BO_sell_vol", r.get("BO_sell_vol_1s", 0.0)) or 0.0))
            a_bo30 = 2.0 / 31.0
            es["bo_b30"] = (1.0 - a_bo30) * es["bo_b30"] + a_bo30 * bo_bv1
            es["bo_s30"] = (1.0 - a_bo30) * es["bo_s30"] + a_bo30 * bo_sv1
            bo_mid_v = 0.5 * (bo_b1_v + bo_a1_v)
            es["bo_m30"] = (1.0 - a_bo30) * es["bo_m30"] + a_bo30 * bo_mid_v
            self.last_bo_b1_q = bo_bq1
            self.last_bo_a1_q = bo_aq1
            self.last_bo_imb1 = float(np.clip((bo_bq1 - bo_aq1) / (bo_bq1 + bo_aq1 + 1e-5), -1.0, 1.0))
            self.last_bo_imb3 = float(np.clip((bo_bq3 - bo_aq3) / (bo_bq3 + bo_aq3 + 1e-5), -1.0, 1.0))
            self.last_bo_flow_30s = float(np.clip((es["bo_b30"] - es["bo_s30"]) / (es["bo_b30"] + es["bo_s30"] + 1e-5), -1.0, 1.0))
            self.last_bo_ret_30s = float(np.clip(((bo_mid_v - es["bo_m30"]) / bo_mid_v) * 50.0, -2.0, 2.0))
        self.last_features = r.to_dict()

        b_prev12 = b_arr[-13] if len(b_arr) >= 13 else b_arr[0]
        ez12_prev12 = self.ez12_hist[-13] if len(self.ez12_hist) >= 13 else self.ez12_hist[0]
        model_err_12 = float(np.clip((b1 - b_prev12) / self.tick_size - ez12_prev12 * 4.0, -10.0, 10.0))
        surprise_pen = max(0.0, -model_err_12) * 0.6

        ext_pos = 0.35 * min(dd_900, 10.0) * max(0.0, 1.0 - 1.5 * pos_900)
        ext_neg = -1.5 * max(0.0, pos_900 - 0.65)
        extremum_bonus = ext_pos if inertia > 0.25 else ext_neg
        exp_entry_edge = (0.15 * ez_60 + 0.40 * ez_25 + 0.30 * ez_12 + 0.15 * ez_5) * 4.0 + 0.5 * spread_ticks - p_tn_5 * 3.0 - surprise_pen + extremum_bonus
        exp_hold_edge = inertia * 3.0 - p_dn_12 * spread_ticks - surprise_pen * 0.6

        obs_err12_norm = float(np.clip(model_err_12 / 5.0, -3.0, 3.0))
        obs_entry_edge_norm = float(np.clip(exp_entry_edge / 5.0, -3.0, 3.0))
        obs_hold_edge_norm = float(np.clip(exp_hold_edge / 5.0, -3.0, 3.0))
        self.last_hold_edge = float(obs_hold_edge_norm * 5.0)
        if self.pos_qty > 1e-8 and self.avg_entry > 0:
            raw_fl = (self.avg_entry * 1.0022) if self.hold_sec >= 900 else (self.avg_entry + 0.75 * self.tick_size)
            fl_p = round(math.ceil(raw_fl / self.tick_size - 1e-4) * self.tick_size, 2)
            wv_p = round(fl_p + self.tick_size, 2)
            self.last_q_a1 = sum(aq for ap, aq in zip(ask_ps, ask_qs_clean) if abs(ap - fl_p) <= tol)
            self.last_q_a2 = sum(aq for ap, aq in zip(ask_ps, ask_qs_clean) if abs(ap - wv_p) <= tol)
        else:
            self.last_q_a1, self.last_q_a2 = a1_q, (ask_qs_clean[1] if len(ask_qs_clean) > 1 else 0.0)

        tot_eq_now = max(1.0, self.bot_cash_usd + self.pos_qty * b1)
        underwater_now = ((self.avg_entry - b1) / self.tick_size) if (self.pos_qty >= 1.95 and self.avg_entry > 0.0) else 0.0
        min_open_px = min([self.avg_entry] + [float(s.get("entry_px", self.avg_entry)) for s in getattr(self, "extra_slots", []) if float(s.get("entry_px", 0.0)) > 0]) if (self.pos_qty >= 1.95 and self.avg_entry > 0.0) else 0.0
        spread_t_now = (a1 - b1) / max(self.tick_size, 1e-6)
        dyn_dip_step = max(self.tick_size, round(min(3.0, 0.5 * spread_t_now)) * self.tick_size)
        dt_since_buy = time.time() - float(getattr(self, "last_buy_fill_ts", 0.0))
        slot_spacing_ok = bool(
            self.active_bid_id is not None
            or (
                dt_since_buy >= 25.0
                and float(getattr(self, "last_p_adverse", 0.0)) < 0.42
                and len(getattr(self, "extra_slots", [])) < 8
                and (
                    b1 <= min_open_px - dyn_dip_step + tol
                    or (dt_since_buy >= 150.0 and pos_3600 <= 0.48 and b1_q >= 3500.0 and float(getattr(self, "last_p_adverse", 0.0)) < 0.35)
                )
            )
        )
        ambivalent_scale_in = bool(
            self.pos_qty >= 1.95
            and (self.active_ask_id is not None or len(getattr(self, "extra_slots", [])) > 0)
            and self.bot_cash_usd >= max(5.0, tot_eq_now * 0.08)
            and slot_spacing_ok
            and underwater_now >= 1.0
            and inertia >= 0.15
            and exp_entry_edge >= 0.65
            and not getattr(self, "last_waterfall_trap", False)
        )
        half_pos_ready_to_buy = bool(
            self.pos_qty >= 1.95
            and (self.active_ask_id is not None or len(getattr(self, "extra_slots", [])) > 0)
            and self.bot_cash_usd >= max(15.0, tot_eq_now * 0.10)
            and (self.pos_qty * b1) < 0.65 * tot_eq_now
            and slot_spacing_ok
            and (
                self.active_bid_id is not None
                or (inertia >= 0.18 and exp_entry_edge >= 0.65 and not getattr(self, "last_waterfall_trap", False))
            )
        )
        self.micro_pos_parked = bool(
            (
                self.pos_qty >= 1.95
                and (self.active_ask_id is not None or len(getattr(self, "extra_slots", [])) > 0)
                and self.bot_cash_usd >= max(15.0, tot_eq_now * 0.10)
                and (self.pos_qty * b1) < 0.35 * tot_eq_now
                and slot_spacing_ok
            )
            or half_pos_ready_to_buy
            or ambivalent_scale_in
        )
        has_pos = 1.0 if (self.pos_qty > 1e-8 and not self.micro_pos_parked) else 0.0
        unr_ticks = ((b1 - self.avg_entry) / self.tick_size) if has_pos else 0.0
        hold_ratio = min(self.hold_sec / 45.0, 2.0) if has_pos else 0.0
        has_bid = 1.0 if self.active_bid_q > 0 else 0.0
        has_ask = 1.0 if (self.active_ask_q > 0 and not self.micro_pos_parked) else 0.0
        tol = self.tick_size * 0.25
        if has_bid:
            q_at, q_better = 0.0, 0.0
            for lp, lq in zip(bid_ps, bid_qs_clean):
                if abs(lp - self.active_bid_p) <= tol:
                    q_at += max(0.0, lq)
                elif lp > self.active_bid_p + tol:
                    q_better += lq
            if self.q_ahead < 0.0:
                self.q_ahead = q_at
            else:
                q_better_eff = max(q_better, getattr(self, "_prev_q_better", q_better))
                q_at_prev = getattr(self, "_prev_q_at", q_at)
                vol_hit = float(r.get("ST_sell_vol", 0.0)) if self.active_bid_p >= b1 - tol else 0.0
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
            for lp, lq in zip(ask_ps, ask_qs_clean):
                if abs(lp - self.active_ask_p) <= tol:
                    q_at += max(0.0, lq)
                elif 0.0 < lp < self.active_ask_p - tol:
                    q_better += lq
            if self.q_ahead < 0.0:
                self.q_ahead = q_at
            else:
                q_better_eff = max(q_better, getattr(self, "_prev_q_better", q_better))
                q_at_prev = getattr(self, "_prev_q_at", q_at)
                vol_hit = float(r.get("ST_buy_vol", 0.0)) if self.active_ask_p <= a1 + tol else 0.0
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

        v52_vec = []
        in_feat_dim = getattr(self.agent.backbone[0], "in_features", 38)
        if in_feat_dim >= 42:
            v52_vec = [
                float(self.last_car_probs[1]),
                float(math.log1p(max(0.0, self.last_car_cap_prl)) / 7.0),
                float(self.last_p_adverse),
                float(self.last_p_exhaust),
            ]
        v70_vec = []
        if in_feat_dim >= 49:
            slot_entries = ([self.avg_entry] if (self.pos_qty >= 1.95 and self.avg_entry > 0.0) else []) + [
                float(s.get("entry_px", self.avg_entry)) for s in getattr(self, "extra_slots", []) if float(s.get("entry_px", 0.0)) > 0
            ]
            n_open_slots = min(10, (1 if self.pos_qty >= 1.95 else 0) + (1 if self.slot2_ask_id is not None else 0) + len(getattr(self, "extra_slots", [])))
            slot_fill_ratio = float(min(1.0, max(n_open_slots / 10.0, (self.pos_qty * b1) / max(1.0, tot_eq_now))))
            lowest_slot_edge_ticks = float(np.clip(((b1 - min(slot_entries)) / self.tick_size) * 0.2, -3.0, 3.0)) if slot_entries else 0.0
            slot_dispersion_ticks = float(np.clip(((max(slot_entries) - min(slot_entries)) / self.tick_size) * 0.2, 0.0, 3.0)) if len(slot_entries) >= 2 else 0.0
            free_cash_ratio = float(np.clip(self.bot_cash_usd / max(1.0, tot_eq_now), 0.0, 1.0))
            v70_vec = [
                slot_fill_ratio,
                lowest_slot_edge_ticks,
                slot_dispersion_ticks,
                free_cash_ratio,
                float(np.clip(ez_90 * 0.5, -3.0, 3.0)),
                float(np.clip(ez_120 * 0.5, -3.0, 3.0)),
                float(getattr(self, "last_p_frontrun", 0.2)),
            ]
        obs = np.array(
            triad_vec + lob_vec + v52_vec + v70_vec + [
                has_pos, float(np.clip(unr_ticks * 0.1, -3.0, 3.0)), hold_ratio,
                has_bid, has_ask, float(np.clip(dist * 0.2, -2.0, 2.0)), q_ratio,
                obs_err12_norm, obs_entry_edge_norm, obs_hold_edge_norm,
            ],
            dtype=np.float32,
        )

        # Вычисляем 26 выходов 24-слотового Чемпиона Урока 7 (release_champion_v1.pt) на текущем живом тике
        if self.l7_backbone is not None and self.l7_w1 is not None:
            try:
                import torch
                with torch.no_grad():
                    mid_p = 0.5 * (b1 + a1)
                    bp_np = np.asarray(bid_ps, dtype=np.float32)
                    bq_np = np.maximum(0.0, np.asarray(bid_qs_clean, dtype=np.float32))
                    ap_np = np.asarray(ask_ps, dtype=np.float32)
                    aq_np = np.maximum(0.0, np.asarray(ask_qs_clean, dtype=np.float32))
                    lob_10_4 = np.empty((1, 10, 4), dtype=np.float32)
                    lob_10_4[0, :, 0] = np.clip((mid_p - bp_np) / self.tick_size, 0.0, 50.0) * 0.1
                    lob_10_4[0, :, 1] = np.log1p(bq_np) * 0.1
                    lob_10_4[0, :, 2] = np.clip((ap_np - mid_p) / self.tick_size, 0.0, 50.0) * 0.1
                    lob_10_4[0, :, 3] = np.log1p(aq_np) * 0.1

                    vb_cur = float(r.get("ST_buy_vol", r.get("ST_buy_vol_1s", 0.0)))
                    vs_cur = float(r.get("ST_sell_vol", r.get("ST_sell_vol_1s", 0.0)))
                    tot_bud = max(25.0, float(self.initial_budget_usd), float(self.bot_cash_usd + self.pos_qty * b1))
                    res_cash_now = sum(float(s.get("qty", 0.0)) * float(s.get("px", 0.0)) * 1.002 for s in self.bid_slots if s.get("id") is not None)
                    unres_cash_now = max(0.0, float(self.bot_cash_usd) - res_cash_now)
                    pos_rat = float(np.clip((self.pos_qty * b1) / tot_bud, 0.0, 1.0))
                    free_c_rat = float(np.clip(unres_cash_now / tot_bud, 0.0, 1.0))
                    n_b_cnt = sum(1 for s in self.bid_slots if s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-4)
                    n_a_cnt = sum(1 for s in self.ask_slots if s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-4)
                    has_bid_sim = 1.0 if n_b_cnt > 0 else 0.0
                    has_ask_sim = 1.0 if n_a_cnt > 0 else 0.0
                    n_b_sl = float(n_b_cnt) / 12.0
                    n_a_sl = float(n_a_cnt) / 12.0
                    has_pos_sim = 1.0 if self.pos_qty > 1e-4 else 0.0
                    eff_pos_scale = float(np.clip(pos_rat / 0.25, 0.15, 1.0))
                    is_l28 = bool(getattr(self, "l28_hierarchical_mode", False))
                    is_l27 = bool(getattr(self, "l27_multimodal_mode", False))
                    is_l26 = bool(getattr(self, "l26_endogenous_mode", False)) or is_l27 or is_l28
                    if is_l26:
                        unr_val = float(np.clip(((b1 - self.avg_entry) / max(self.avg_entry, self.tick_size)) * 25.0 * eff_pos_scale, -2.5, 2.5)) if (has_pos_sim > 0.5 and self.avg_entry > 0.0) else 0.0
                    else:
                        unr_sim = (((b1 - self.avg_entry) / self.tick_size) * eff_pos_scale) if (has_pos_sim > 0.5 and self.avg_entry > 0.0) else 0.0
                        unr_val = float(np.clip(unr_sim / 5.0, -2.0, 2.0))
                    hold_rat_sim = (min(self.hold_sec / 300.0, 2.0) * eff_pos_scale) if has_pos_sim > 0.5 else 0.0

                    acct_8d_list = [
                        float(has_pos_sim),
                        pos_rat,
                        free_c_rat,
                        unr_val,
                        float(np.clip(hold_rat_sim, 0.0, 2.0)),
                        has_bid_sim,
                        has_ask_sim,
                        float(n_b_sl - n_a_sl),
                    ]
                    dec_bb = bool(getattr(self, "l7_decoupled_backbone", False))
                    flow_16 = np.array([[
                        float(np.clip(spread_ticks, 1.0, 10.0) / 5.0),
                        float(imb_l1),
                        float(np.log1p(max(0.0, vb_cur)) / 8.0),
                        float(np.log1p(max(0.0, vs_cur)) / 8.0),
                        float((vb_cur - vs_cur) / max(1.0, vb_cur + vs_cur)),
                        float(np.clip(dd_300, 0.0, 20.0) / 10.0),
                        float(np.log1p(max(0.0, b10_q)) / 12.0),
                        float(np.log1p(max(0.0, a10_q)) / 12.0),
                        *( [0.0] * 8 if dec_bb else acct_8d_list ),
                    ]], dtype=np.float32)

                    l1_b_usd = max(100.0, b1 * b1_q)
                    l1_a_usd = max(100.0, a1 * a1_q)
                    roll_b30 = max(10.0, sum(x[2] for x in list(self.flow_hist_10s)) * 3.0)
                    low_tr_rat = float(np.clip(self.tr_low_qty / max(self.pos_qty, 1e-6), 0.0, 1.0)) if has_pos_sim > 0.5 else 0.0
                    lifo_ref = self.tr_low_px if (self.tr_low_qty > 1e-4 and self.tr_low_px > 0.0) else self.avg_entry
                    if is_l26:
                        lifo_edge_t = float(np.clip(((b1 - lifo_ref) / max(lifo_ref, self.tick_size)) * 30.0, -2.5, 2.5)) if (has_pos_sim > 0.5 and lifo_ref > 0.0) else 0.0
                    else:
                        lifo_edge_t = float(np.clip(((b1 - lifo_ref) / self.tick_size) / 3.0, -2.0, 2.0)) if (has_pos_sim > 0.5 and lifo_ref > 0.0) else 0.0
                    multi_6d = np.array([[
                        float(np.log1p(tot_bud / l1_b_usd) / 3.0),
                        float(np.log1p(unres_cash_now / l1_b_usd) / 3.0),
                        float(np.clip((self.pos_qty * a1) / l1_a_usd, 0.0, 4.0) / 2.0),
                        float(np.clip(self.pos_qty / roll_b30, 0.0, 10.0) / 5.0),
                        low_tr_rat,
                        lifo_edge_t,
                    ]], dtype=np.float32)

                    lob_t = torch.from_numpy(lob_10_4)
                    flow_t = torch.from_numpy(flow_16)
                    z_t, self.l7_h_state = self.l7_backbone.backbone.forward_step(lob_t, flow_t, self.l7_h_state)
                    ps_t = torch.sigmoid(self.l7_backbone.head_sweep_12s(z_t))
                    pb_t = torch.sigmoid(self.l7_backbone.head_bleed_60s(z_t))
                    pc_t = torch.sigmoid(self.l7_backbone.head_crash_300s(z_t))
                    pe_t = torch.sigmoid(self.l7_backbone.head_exit_30s(z_t))
                    self.last_l7_ps = float(ps_t.item())
                    self.last_l7_pb = float(pb_t.item())
                    self.last_l7_pc = float(pc_t.item())
                    self.last_l7_pe = float(pe_t.item())
                    v52_4d = [
                        float(self.last_car_probs[1]),
                        float(math.log1p(max(0.0, self.last_car_cap_prl)) / 7.0),
                        float(self.last_p_adverse),
                        float(self.last_p_exhaust),
                    ]
                    real_2d = [
                        float(getattr(self, "last_p_fill_real", 0.5)),
                        float(getattr(self, "last_p_frontrun", 0.2)),
                    ]
                    mat_6d = [
                        float(math.log1p(getattr(self.routers[0][2], "num_models", 65)) / 4.0),
                        float(math.log1p(getattr(self.routers[1][2], "num_models", 22)) / 4.0),
                        float(math.log1p(getattr(self.routers[2][2], "num_models", 12)) / 4.0),
                        float(math.log1p(getattr(self.router_60, "num_models", 5)) / 4.0),
                        float(math.log1p(getattr(self.router_90, "num_models", 3)) / 4.0),
                        float(math.log1p(getattr(self.router_120, "num_models", 2)) / 4.0),
                    ]
                    bo_vb_cur = float(r.get("BO_buy_vol", 0.0) or 0.0)
                    bo_vs_cur = float(r.get("BO_sell_vol", 0.0) or 0.0)
                    self.st_trade_5s.append((max(0.0, vb_cur), max(0.0, vs_cur)))
                    self.bo_trade_5s.append((max(0.0, bo_vb_cur), max(0.0, bo_vs_cur)))
                    st_b5s = sum(x[0] for x in self.st_trade_5s)
                    st_s5s = sum(x[1] for x in self.st_trade_5s)
                    bo_b5s = sum(x[0] for x in self.bo_trade_5s)
                    bo_s5s = sum(x[1] for x in self.bo_trade_5s)
                    bo_bp_5 = np.array([float(r.get(f"BO_bid_{i}_price", 0.0) or 0.0) for i in range(1, 6)], dtype=np.float32)
                    bo_bq_5 = np.array([float(r.get(f"BO_bid_{i}_qty", 0.0) or 0.0) for i in range(1, 6)], dtype=np.float32)
                    bo_ap_5 = np.array([float(r.get(f"BO_ask_{i}_price", 0.0) or 0.0) for i in range(1, 6)], dtype=np.float32)
                    bo_aq_5 = np.array([float(r.get(f"BO_ask_{i}_qty", 0.0) or 0.0) for i in range(1, 6)], dtype=np.float32)
                    bo_sec_age = float(r.get("BO_sec_since_tick", 0.0) or 0.0)
                    from _tools.cross_venue_first_order_net import compute_cross_venue_single_tick, peek_local_venue_snap
                    cv_net_obj = getattr(self, "cv_first_order_net", None)
                    cv_in_dim = int(getattr(cv_net_obj, "in_dim", 28)) if cv_net_obj is not None else 28
                    gt_sn = ob_sn = mac_sn = None
                    if cv_in_dim > 28:
                        rt_root = BASE_DIR / "data" / "realtime"
                        gt_sn = peek_local_venue_snap(rt_root / "prlusdt_gate", "depth_gate.csv", "trades_gate.csv", self.cv_hist_state)
                        ob_sn = peek_local_venue_snap(rt_root / "prlusdt_ourbit", "depth_ourbit.csv", "trades_ourbit.csv", self.cv_hist_state)
                        mac_sn = {
                            "btc_mid": float(r.get("macro_btc_mid", 0.0) or 0.0),
                            "near_mid": float(r.get("macro_near_mid", 0.0) or 0.0),
                        }
                    cv_10_np, cv_vec_np = compute_cross_venue_single_tick(
                        st_bp=bp_np,
                        st_bq=bq_np,
                        st_ap=ap_np,
                        st_aq=aq_np,
                        st_buy_5s=st_b5s,
                        st_sell_5s=st_s5s,
                        bo_bp=bo_bp_5,
                        bo_bq=bo_bq_5,
                        bo_ap=bo_ap_5,
                        bo_aq=bo_aq_5,
                        bo_buy_5s=bo_b5s,
                        bo_sell_5s=bo_s5s,
                        bo_sec_since=bo_sec_age,
                        hist_state=self.cv_hist_state,
                        tick=self.tick_size,
                        gt_snap=gt_sn,
                        ob_snap=ob_sn,
                        macro_snap=mac_sn,
                        out_dim_x=cv_in_dim,
                    )
                    if getattr(self, "l7_use_cross_venue", False) and cv_net_obj is not None:
                        cv_probs_6 = cv_net_obj.predict_probs(torch.from_numpy(cv_vec_np).unsqueeze(0)).squeeze(0).numpy()
                        slot_18_28 = cv_10_np.tolist()
                        slot_52_58 = ((cv_probs_6 - 0.5) * 2.0).astype(np.float32).tolist()
                    else:
                        slot_18_28 = [0.0] * 10
                        slot_52_58 = [0.0] * 6
                    v7_58_arr = np.array(
                        triad_vec + slot_18_28 + self.last_vec60 + self.last_vec90 + self.last_vec120 + v52_4d + real_2d + slot_52_58,
                        dtype=np.float32,
                    )
                    v7_cur_58 = v7_58_arr[:58]
                    v7_prev5 = self.v7_hist_5s[0] if len(self.v7_hist_5s) > 0 else v7_cur_58
                    self.v7_hist_5s.append(v7_cur_58.copy())
                    v7_t = torch.from_numpy(v7_cur_58).unsqueeze(0)
                    m6_t = torch.from_numpy(multi_6d)
                    z_aug = torch.cat([z_t, ps_t, pb_t, pc_t, pe_t, v7_t, m6_t], dim=-1)
                    if int(self.l7_w1.shape[0]) >= 356:
                        b_slot_12 = [
                            float(np.clip((float(self.bid_slots[si].get("qty", 0.0)) * b1) / max(tot_bud, 25.0), 0.0, 1.0))
                            if self.bid_slots[si].get("id") is not None else 0.0
                            for si in range(12)
                        ]
                        a_slot_12 = [
                            float(np.clip(float(self.ask_slots[si].get("qty", 0.0)) / max(self.pos_qty, 2.0), 0.0, 1.0))
                            if self.ask_slots[si].get("id") is not None else 0.0
                            for si in range(12)
                        ]
                        vel_8d = [float(np.clip(v7_cur_58[vi] - v7_prev5[vi], -3.0, 3.0)) for vi in (0, 2, 4, 6, 8, 12, 34, 42)]
                        slot_32_t = torch.tensor([b_slot_12 + a_slot_12 + vel_8d], dtype=torch.float32)
                        z_aug = torch.cat([z_aug, slot_32_t], dim=-1)
                    if int(self.l7_w1.shape[0]) >= 388:
                        if is_l26:
                            b_dist_12 = [
                                float(np.clip(((b1 - float(self.bid_slots[si].get("px", b1))) / max(b1, self.tick_size)) * 12.0, -0.5, 3.0))
                                if (self.bid_slots[si].get("id") is not None and float(self.bid_slots[si].get("qty", 0.0)) > 1e-6)
                                else 0.0
                                for si in range(12)
                            ]
                            a_dist_12 = [
                                float(np.clip(((float(self.ask_slots[si].get("px", a1)) - a1) / max(a1, self.tick_size)) * 12.0, -0.5, 3.0))
                                if (self.ask_slots[si].get("id") is not None and float(self.ask_slots[si].get("qty", 0.0)) > 1e-6)
                                else 0.0
                                for si in range(12)
                            ]
                        else:
                            b_dist_12 = [
                                float(np.clip((b1 - float(self.bid_slots[si].get("px", b1))) / (5.0 * self.tick_size), -0.5, 1.5))
                                if (self.bid_slots[si].get("id") is not None and float(self.bid_slots[si].get("qty", 0.0)) > 1e-6)
                                else 0.0
                                for si in range(12)
                            ]
                            a_dist_12 = [
                                float(np.clip((float(self.ask_slots[si].get("px", a1)) - a1) / (5.0 * self.tick_size), -0.5, 1.5))
                                if (self.ask_slots[si].get("id") is not None and float(self.ask_slots[si].get("qty", 0.0)) > 1e-6)
                                else 0.0
                                for si in range(12)
                            ]
                        a1_5s_ago = float(getattr(self, "ask_hist_5s", [a1])[0]) if hasattr(self, "ask_hist_5s") else a1
                        if not hasattr(self, "ask_hist_5s"):
                            self.ask_hist_5s = deque(maxlen=6)
                        self.ask_hist_5s.append(a1)
                        db1_5s = float(np.clip((b1 - b1_5s_ago) / (3.0 * self.tick_size), -2.0, 2.0))
                        da1_5s = float(np.clip((a1 - a1_5s_ago) / (3.0 * self.tick_size), -2.0, 2.0))
                        dbq_5s = float(np.clip(math.log1p(max(0.0, b1_q)) - math.log1p(max(0.0, b1q_5s_ago)), -2.0, 2.0))
                        daq_5s = float(np.clip(math.log1p(max(0.0, a1_q)) - math.log1p(max(0.0, a1q_5s_ago)), -2.0, 2.0))
                        b_deep_q = max(0.0, b5_q - b1_q)
                        a_deep_q = max(0.0, a5_q - a1_q)
                        imb_deep = float(np.clip((b_deep_q - a_deep_q) / (b_deep_q + a_deep_q + 50.0), -1.0, 1.0))
                        lvel_8d = [
                            db1_5s,
                            da1_5s,
                            dbq_5s,
                            daq_5s,
                            float(np.clip((b1_q - a1_q) / (b1_q + a1_q + 10.0), -1.0, 1.0)),
                            imb_deep,
                            float(np.clip(math.log1p(max(0.0, vb_cur)) / 8.0, 0.0, 2.0)),
                            float(np.clip(math.log1p(max(0.0, vs_cur)) / 8.0, 0.0, 2.0)),
                        ]
                        geom_32_t = torch.tensor([b_dist_12 + a_dist_12 + lvel_8d], dtype=torch.float32)
                        z_aug = torch.cat([z_aug, geom_32_t], dim=-1)
                    if int(self.l7_w1.shape[0]) >= 420:
                        now_q_ts = time.time()
                        self.last_clean_bp_10 = bp_exec_np.copy()
                        self.last_clean_ap_10 = ap_exec_np.copy()
                        self.last_clean_bq_10 = bq_exec_np.copy()
                        self.last_clean_aq_10 = aq_exec_np.copy()
                        if is_l26:
                            from _tools.l26_live_bridge import lookup_sparse_lob_qty_1d
                        roll_s30 = max(10.0, sum(x[3] for x in list(self.flow_hist_10s)) * 3.0)
                        for si, sl_b in enumerate(self.bid_slots):
                            oid_b = sl_b.get("id")
                            sq_b = float(sl_b.get("qty", 0.0))
                            spx_b = float(sl_b.get("px", 0.0))
                            if oid_b is not None and sq_b > 1e-6:
                                if is_l26:
                                    vis_q = lookup_sparse_lob_qty_1d(bp_exec_np, bq_exec_np, spx_b, exec_tick) if spx_b <= b1 + 0.1 * exec_tick else 0.0
                                    base_qa_raw_b = vis_q
                                else:
                                    lvl_i = int(np.clip(round((b1 - spx_b) / self.tick_size), 0, 9))
                                    vis_q = max(0.0, float(bq_np[lvl_i])) if spx_b <= b1 + 0.1 * self.tick_size else 0.0
                                    base_qa_raw_b = max(10.0, float(bq_np[lvl_i]))
                                if sl_b.get("qa_oid") != oid_b:
                                    base_qa_b = 0.0 if spx_b > b1 + 0.1 * exec_tick else base_qa_raw_b
                                    b_over50 = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (sq_b * spx_b - 50.0) / 18.0))))
                                    init_qa_b = base_qa_b * (1.0 + 0.75 * b_over50) + b_over50 * (0.45 * sq_b)
                                    sl_b["qa_oid"] = oid_b
                                    sl_b["qa0"] = init_qa_b
                                    sl_b["qa"] = min(init_qa_b, vis_q) if spx_b <= b1 + 0.1 * exec_tick else init_qa_b
                                else:
                                    cur_qa = float(sl_b.get("qa", vis_q))
                                    if b1 <= spx_b + 1e-6 and vs_cur > 0.0:
                                        cur_qa = max(0.0, cur_qa - max(0.0, vs_cur))
                                    if spx_b <= b1 + 0.1 * exec_tick:
                                        cur_qa = min(cur_qa, vis_q)
                                    sl_b["qa"] = cur_qa
                        for si, sl_a in enumerate(self.ask_slots):
                            oid_a = sl_a.get("id")
                            sq_a = float(sl_a.get("qty", 0.0))
                            spx_a = float(sl_a.get("px", 0.0))
                            if oid_a is not None and sq_a > 1e-6:
                                if is_l26:
                                    vis_q = lookup_sparse_lob_qty_1d(ap_exec_np, aq_exec_np, spx_a, exec_tick) if spx_a >= a1 - 0.1 * exec_tick else 0.0
                                    base_qa_raw_a = vis_q
                                else:
                                    lvl_i = int(np.clip(round((spx_a - a1) / self.tick_size), 0, 9))
                                    vis_q = max(0.0, float(aq_np[lvl_i])) if spx_a >= a1 - 0.1 * self.tick_size else 0.0
                                    base_qa_raw_a = max(10.0, float(aq_np[lvl_i]))
                                if sl_a.get("qa_oid") != oid_a:
                                    base_qa_a = 0.0 if spx_a < a1 - 0.1 * exec_tick else base_qa_raw_a
                                    a_over50 = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (sq_a * spx_a - 50.0) / 18.0))))
                                    init_qa_a = base_qa_a * (1.0 + 0.75 * a_over50) + a_over50 * (0.45 * sq_a)
                                    sl_a["qa_oid"] = oid_a
                                    sl_a["qa0"] = init_qa_a
                                    sl_a["qa"] = min(init_qa_a, vis_q) if spx_a >= a1 - 0.1 * exec_tick else init_qa_a
                                else:
                                    cur_qa = float(sl_a.get("qa", vis_q))
                                    if a1 >= spx_a - 1e-6 and vb_cur > 0.0:
                                        cur_qa = max(0.0, cur_qa - max(0.0, vb_cur))
                                    if spx_a >= a1 - 0.1 * exec_tick:
                                        cur_qa = min(cur_qa, vis_q)
                                    sl_a["qa"] = cur_qa

                        act_bs = [s for s in self.bid_slots if s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-6]
                        if act_bs:
                            best_bs = min(act_bs, key=lambda x: (b1 - float(x.get("px", b1)), int(x.get("slot", 0))))
                            bb_qa = float(best_bs.get("qa", 0.0))
                            bb_qa0 = float(best_bs.get("qa0", bb_qa))
                            best_b_rat = float(np.clip(bb_qa / (bb_qa0 + 10.0), 0.0, 1.5))
                            best_b_tape = float(np.clip(math.log1p(bb_qa / roll_s30) / 3.0, 0.0, 2.0))
                            best_b_age = float(np.clip((now_q_ts - float(best_bs.get("act_ts", now_q_ts))) / 120.0, 0.0, 2.0))
                        else:
                            best_b_rat, best_b_tape, best_b_age = 1.0, 0.0, 0.0

                        deep_bs = [s for s in self.bid_slots[4:] if s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-6]
                        if deep_bs:
                            deep_b_rat = float(np.mean([np.clip(float(s.get("qa", 0.0)) / (float(s.get("qa0", 0.0)) + 10.0), 0.0, 1.5) for s in deep_bs]))
                        else:
                            deep_b_rat = 1.0

                        act_as = [s for s in self.ask_slots if s.get("id") is not None and float(s.get("qty", 0.0)) > 1e-6]
                        if act_as:
                            best_as = min(act_as, key=lambda x: (float(x.get("px", a1)) - a1, int(x.get("slot", 0))))
                            ba_qa = float(best_as.get("qa", 0.0))
                            ba_qa0 = float(best_as.get("qa0", ba_qa))
                            best_a_rat = float(np.clip(ba_qa / (ba_qa0 + 10.0), 0.0, 1.5))
                            best_a_tape = float(np.clip(math.log1p(ba_qa / roll_b30) / 3.0, 0.0, 2.0))
                            best_a_age = float(np.clip((now_q_ts - float(best_as.get("act_ts", now_q_ts))) / 120.0, 0.0, 2.0))
                        else:
                            best_a_rat, best_a_tape, best_a_age = 1.0, 0.0, 0.0

                        q_imb_adv = (1.0 - min(1.0, max(0.0, best_a_rat))) * has_ask_sim - (1.0 - min(1.0, max(0.0, best_b_rat))) * has_bid_sim
                        queue_8d_list = [best_b_rat, best_b_tape, best_b_age, deep_b_rat, best_a_rat, best_a_tape, best_a_age, q_imb_adv]
                        if not hasattr(self, "l7_mem_h") or self.l7_mem_h is None:
                            self.l7_mem_h = torch.zeros((1, 16), dtype=torch.float32)
                        ext_32_t = torch.cat([torch.tensor([acct_8d_list + queue_8d_list], dtype=torch.float32), self.l7_mem_h.float()], dim=-1)
                        z_aug = torch.cat([z_aug, ext_32_t], dim=-1)
                    if int(self.l7_w1.shape[0]) >= 444:
                        slot_qprog_24 = []
                        for sl_q in self.bid_slots:
                            if sl_q.get("id") is not None and float(sl_q.get("qty", 0.0)) > 1e-6:
                                q_rat_s = float(np.clip(float(sl_q.get("qa", 0.0)) / (float(sl_q.get("qa0", 0.0)) + 10.0), 0.0, 1.5))
                                slot_qprog_24.append(float(np.clip(1.0 - q_rat_s, -0.5, 1.0)))
                            else:
                                slot_qprog_24.append(-1.0)
                        for sl_q in self.ask_slots:
                            if sl_q.get("id") is not None and float(sl_q.get("qty", 0.0)) > 1e-6:
                                q_rat_s = float(np.clip(float(sl_q.get("qa", 0.0)) / (float(sl_q.get("qa0", 0.0)) + 10.0), 0.0, 1.5))
                                slot_qprog_24.append(float(np.clip(1.0 - q_rat_s, -0.5, 1.0)))
                            else:
                                slot_qprog_24.append(-1.0)
                        z_aug = torch.cat([z_aug, torch.tensor([slot_qprog_24], dtype=torch.float32)], dim=-1)
                    if int(self.l7_w1.shape[0]) >= 480 and not is_l28:
                        bo_36_parts = []
                        bo_r_list = getattr(self, "bo_swarm_routers", None) or []
                        if len(bo_r_list) == 6 and getattr(self, "bo_feature_names", None):
                            x_bo_1d = np.array(
                                [float(r.get(fn, 0.0) or 0.0) for fn in self.bo_feature_names],
                                dtype=np.float32,
                            )
                            for r_bo in bo_r_list:
                                bo_36_parts.append(r_bo.predict_live_1row(x_bo_1d, bo_sec_age))
                            bo_36_vec = np.concatenate(bo_36_parts, axis=0).astype(np.float32)
                        else:
                            bo_36_vec = np.zeros(36, dtype=np.float32)
                        z_aug = torch.cat([z_aug, torch.from_numpy(bo_36_vec).unsqueeze(0)], dim=-1)
                    if is_l28 and self.l7_w2_moe is not None and int(self.l7_w2_moe.shape[-1]) == 180:
                        from _tools.l28_live_bridge import (
                            assemble_l28_508d_live,
                            build_l28_agent_features_from_trader,
                            forward_l28_hierarchical_live,
                        )
                        ag_l28 = build_l28_agent_features_from_trader(
                            self, b1, a1, b1_q, a1_q, roll_b30, roll_s30, now_q_ts, maker_fee=0.001
                        )
                        z_508_t, adv_r_l28 = assemble_l28_508d_live(
                            z_t_256=z_t,
                            ps_t=ps_t,
                            pb_t=pb_t,
                            pc_t=pc_t,
                            pe_t=pe_t,
                            st_tree_42=getattr(self, "last_l28_st_tree_42", np.zeros(42, dtype=np.float32)),
                            st_coh_4=getattr(self, "last_l28_st_coh_4", np.zeros(4, dtype=np.float32)),
                            bo_tree_42=getattr(self, "last_l28_bo_tree_42", np.zeros(42, dtype=np.float32)),
                            bo_coh_4=getattr(self, "last_l28_bo_coh_4", np.zeros(4, dtype=np.float32)),
                            slot_18_28=slot_18_28,
                            v52_4d=v52_4d,
                            real_2d=real_2d,
                            slot_52_58=slot_52_58,
                            lvel_8d=lvel_8d,
                            bp_exec_np=bp_exec_np,
                            bq_exec_np=bq_exec_np,
                            ap_exec_np=ap_exec_np,
                            aq_exec_np=aq_exec_np,
                            vb_cur=vb_cur,
                            vs_cur=vs_cur,
                            exec_tick=exec_tick,
                            vel8_hist_deque=self.l28_vel8_hist,
                            multi_6d_np=ag_l28["multi_6d"],
                            b_slot_12=ag_l28["b_slot_12"],
                            a_slot_12=ag_l28["a_slot_12"],
                            b_dist_12=ag_l28["b_dist_12"],
                            a_dist_12=ag_l28["a_dist_12"],
                            acct_8d_list=ag_l28["acct_8d"],
                            queue_8d_list=ag_l28["queue_8d"],
                            mem_h_t=self.l7_mem_h,
                            slot_qprog_24=ag_l28["slot_qprog_24"],
                        )
                        forward_l28_hierarchical_live(self, z_508_t, ag_l28["acct_8d"], now_q_ts, adv_r_l28)
                    elif is_l27 and self.l7_w2_moe is not None and int(self.l7_w2_moe.shape[-1]) == 180:
                        from _tools.l27_live_bridge import forward_l27_multimodal_live
                        forward_l27_multimodal_live(self, z_aug, acct_8d_list, now_q_ts)
                    elif is_l26 and int(self.l7_w2.shape[-1]) == 44:
                        from _tools.l26_live_bridge import forward_l26_strategist_live
                        forward_l26_strategist_live(self, z_aug, acct_8d_list, now_q_ts)
                    else:
                        self.last_l26_state = None
                        self.last_l27_state = None
                        pre_all = z_aug @ self.l7_w1 + self.l7_b1.squeeze(0)
                        if int(self.l7_w1.shape[1]) == 384:
                            h_base = torch.relu(pre_all[..., :256])
                            h_gate = (2.0 * torch.sigmoid(pre_all[..., 256:384])).repeat_interleave(2, dim=-1)
                            h1_t = h_base * h_gate
                        else:
                            h1_t = torch.relu(pre_all)
                            if getattr(self, "l7_wg", None) is not None and getattr(self, "l7_bg", None) is not None:
                                gate_t = 2.0 * torch.sigmoid(z_aug @ self.l7_wg + self.l7_bg.squeeze(0))
                                h1_t = h1_t * gate_t
                        if getattr(self, "l7_w_res1", None) is not None and getattr(self, "l7_w_res2", None) is not None:
                            res_pre = h1_t @ self.l7_w_res1 + self.l7_b_res1.squeeze(0)
                            res_mid = torch.relu(res_pre[..., :64]) * (2.0 * torch.sigmoid(res_pre[..., 64:128]))
                            h1_t = h1_t + (res_mid @ self.l7_w_res2)
                        if getattr(self, "l7_w_rec", None) is not None and getattr(self, "l7_b_rec", None) is not None:
                            rec_raw = h1_t @ self.l7_w_rec + self.l7_b_rec.squeeze(0)
                            rec_u = torch.tanh(rec_raw[..., :16])
                            rec_gate = torch.sigmoid(rec_raw[..., 16:32] + 1.5)
                            self.l7_mem_h = (rec_gate * self.l7_mem_h + (1.0 - rec_gate) * rec_u).detach()
                        if getattr(self, "l25_regulator_mode", False) and int(self.l7_w2.shape[-1]) == 35:
                            from _tools.fused_l25_regulator_tick import _sparse_slot_softmax
                            strat_raw = (h1_t @ self.l7_w2 + self.l7_b2.squeeze(0)).float()  # [1, 35]
                            exp_bid_f = float(torch.sigmoid(strat_raw[0, 0]).item())
                            tau_s = 0.30 + 1.70 * torch.sigmoid(strat_raw[:, 34:35])
                            pi_b_np = _sparse_slot_softmax(strat_raw[:, 1:13], tau_s, floor=0.03).squeeze(0).numpy().astype(np.float32)
                            pi_a_np = _sparse_slot_softmax(strat_raw[:, 13:25], tau_s, floor=0.03).squeeze(0).numpy().astype(np.float32)
                            panic_p = float(torch.sigmoid(strat_raw[0, 25]).item())
                            ctx_b_np = torch.tanh(strat_raw[0, 26:30]).numpy().astype(np.float32)
                            ctx_a_np = torch.tanh(strat_raw[0, 30:34]).numpy().astype(np.float32)
                            self.last_l25_state = {
                                "exp_bid_frac": exp_bid_f,
                                "tau_slot": float(tau_s.item()),
                                "pi_bid": pi_b_np,
                                "pi_ask": pi_a_np,
                                "panic_prob": panic_p,
                                "ctx_4d_b": ctx_b_np,
                                "ctx_4d_a": ctx_a_np,
                                "free_cash_ratio": float(acct_8d_list[2]) if int(self.l7_w1.shape[0]) >= 420 else 1.0,
                                "now_q_ts": float(now_q_ts) if int(self.l7_w1.shape[0]) >= 420 else time.time(),
                            }
                            # Совместимый 26D вектор для телеметрии и старых масок:
                            compat_26 = np.zeros(26, dtype=np.float32)
                            compat_26[:12] = pi_b_np * exp_bid_f
                            compat_26[12:24] = pi_a_np
                            compat_26[24] = 1.0 if exp_bid_f < 0.03 else 0.0
                            compat_26[25] = panic_p
                            self.last_l7_out = compat_26
                            self.last_l7_target_exp_frac = exp_bid_f
                        else:
                            self.last_l25_state = None
                            raw_26 = torch.sigmoid(h1_t @ self.l7_w2 + self.l7_b2.squeeze(0)).squeeze(0).numpy()
                            self.last_l7_out = raw_26
                            if raw_26[24] <= 0.75:
                                b_w_12 = raw_26[:12]
                            elif getattr(self, "l7_selective_front_cancel", False):
                                b_w_12 = raw_26[:12].copy()
                                b_w_12[:6] = 0.0
                            else:
                                b_w_12 = np.zeros(12, dtype=np.float32)
                            b_tot_s = float(np.sum(np.where(b_w_12 >= 0.08, b_w_12, 0.0)))
                            self.last_l7_target_exp_frac = float(np.clip(b_tot_s, 0.0, 1.0))
            except Exception as e_l7_step:
                print(f"⚠️ [L7/L9 STEP WARN]: {e_l7_step}")
                self.last_l7_out = None

        vel_3 = (b1 - (b_arr[-4] if len(b_arr) >= 4 else b_arr[0])) / self.tick_size
        self.last_vel3 = vel_3
        self.last_ez5 = ez_5
        self.last_vel180 = vel_180
        self.last_err12 = float(obs_err12_norm)
        self.last_entry_edge = float(exp_entry_edge)
        safe_macro_4h = (rng_14400 < 6.0) or (pos_14400 <= 0.72)
        waterfall_trap = ((dd_900 >= 8.0) and (pos_14400 >= 0.45) and (ez_25 < 0.0)) or ((dd_900 >= 10.0) and (b1_q < 2500.0) and (a1_q > 800.0) and (ez_12 < 0.0))
        staircase_bleed = (
            ((pos_3600 <= 0.05) and (pos_14400 <= 0.05) and (vel_180 <= -2.0) and (ez_25 < 0.0))
            or ((pos_3600 <= 0.05) and (pos_14400 <= 0.05) and (rng_14400 >= 10.0) and (dd_900 <= 3.5) and (a1_q > 1000.0) and (b1_q < 20000.0))
            or ((ez_5 < 0.15) and (ez_12 <= 0.0) and (ez_25 <= 0.0))
            or ((pos_3600 <= 0.05) and (ez_12 <= 0.0) and (ez_25 <= 0.0) and not ((b1_q >= 9000.0) and (imb_l5 >= 0.60) and (ez_5 >= 0.15)))
            or ((rng_14400 >= 22.0) and (ez_12 <= 0.05) and (ez_25 <= 0.02) and (a1_q > 400.0))
            or ((ez_60 <= -0.10) and (ez_25 <= 0.05) and (dd_900 >= 2.0) and (b1_q < 8000.0))
        )
        top_rollover = (spread_ticks < 1.5) and (rng_14400 >= 16.0) and (pos_14400 >= 0.50) and (pos_3600 <= 0.15) and (dd_300 <= 1.0)
        self.last_waterfall_trap = waterfall_trap
        in_dip_zone = (
            (pos_3600 <= 0.80)
            and safe_macro_4h
            and not waterfall_trap
            and not staircase_bleed
            and not top_rollover
            and (vel_10 >= -0.5)
            and (inertia >= 0.15)
            and (exp_entry_edge >= 0.60)
            and ((dd_300 >= 1.0) or (spread_ticks >= 2.0))
        )
        a1_q_brk = float(getattr(self, "last_a1_q", self.last_q_a1))
        brk_wide = (
            (spread_ticks >= 1.8)
            and (float(getattr(self, "last_b1_q", 0.0)) >= 5500.0)
            and (a1_q_brk <= 2500.0)
            and (a1_q_brk <= 1000.0 or pos_3600 >= 0.35)
            and (inertia >= 0.60)
            and (ez_25 >= 0.35)
            and (dd_300 >= 1.0)
        )
        in_breakout_zone = (
            (
                (ez_5 >= 0.40)
                and (ez_12 >= 0.80)
                and (ez_25 >= 0.85)
                and (inertia >= 0.70)
                and (exp_entry_edge >= 3.20)
                and (imb_l5 >= 0.25)
                and (vel_10 >= 1.0)
                and (rng_14400 <= 11.0)
            )
            or brk_wide
        )
        self.last_in_breakout = in_breakout_zone

        # v8.0 Neural Regime Gate: адаптивное расширение шлюза входа без жестких барьеров
        b1_dom = b1_q / max(1.0, b1_q + a1_q)
        feat24 = np.array([
            float(b1_dom - 0.5),
            float(imb_l5 * 0.5),
            float(imb_l5 * 0.5),
            0.0,
            0.0,
            float((spread_ticks - 1.0) * 0.5),
            float(vel_10 * 0.2),
            float(vel_180 * 0.1),
            float(inertia * 0.33),
            float(getattr(self, "last_cvd_norm", 0.0)),
            float(getattr(self, "last_log_vol", 0.0)),
            float(self.last_car_probs[1] - 0.2),
            float(self.last_p_adverse),
            float(self.last_p_exhaust),
            0.0,
            float(np.clip(ez_90 * 0.5, -3.0, 3.0)),
            float(np.clip(ez_120 * 0.5, -3.0, 3.0)),
            float(np.log1p(b1_q) / 8.0),
            float(np.log1p(a1_q) / 8.0),
            float(np.log1p(b1_q * 4.0) / 10.0),
            float(np.log1p(a1_q * 4.0) / 10.0),
            0.0,
            float(vel_3 * 0.3),
            0.0,
        ], dtype=np.float32)

        if run_slow or getattr(self, "last_regime_probs", None) is None:
            reg_probs = self.regime_gate.predict_single(feat24)
            self.last_regime_probs = reg_probs
        else:
            reg_probs = self.last_regime_probs
        p_flat, p_acc, p_brk, p_tox = float(reg_probs[0]), float(reg_probs[1]), float(reg_probs[2]), float(reg_probs[3])
        is_accum_zone = (p_acc >= 0.35) and (p_tox < 0.45) and (self.last_p_adverse < 0.45)
        is_break_zone = (p_brk >= 0.42) and (p_tox < 0.35)
        is_toxic_block = (p_tox >= 0.50) or (self.last_p_adverse >= 0.65)

        # Вычисление роя нейро-ограничителей (Guard Ensemble: sweep12, bleed60, crash300, exit30)
        b1_p = float(b1)
        a1_p = float(a1)
        if self.l7_w1 is not None and getattr(self, "last_l7_out", None) is not None:
            guard_p = np.array([
                float(getattr(self, "last_l7_ps", 0.05)),
                float(getattr(self, "last_l7_pb", 0.05)),
                float(getattr(self, "last_l7_pc", 0.02)),
                float(getattr(self, "last_l7_pe", 0.05)),
            ], dtype=np.float32)
        else:
            sp_t = max(1.0, (a1_p - b1_p) / self.tick_size)
            sp_pct = (a1_p - b1_p) / max(b1_p, 1e-4) * 100.0
            v_buy = float(getattr(self, "last_buy_vol", 0.0))
            v_sell = float(getattr(self, "last_sell_vol", 0.0))
            f20 = np.array([
                sp_t, sp_pct,
                np.log1p(b1_q), np.log1p(a1_q),
                (b1_q - a1_q) / max(1.0, b1_q + a1_q),
                float(vel_10), float(vel_3), float(vel_180),
                float(dd_300), float(pos_3600), float(rng_14400),
                float(inertia),
                np.log1p(v_buy), np.log1p(v_sell),
                (v_buy - v_sell) / max(1.0, v_buy + v_sell),
                ez_5, ez_12, ez_25, ez_60, ez_90,
            ], dtype=np.float32)
            guard_p = self.guard_predictor.predict_single(f20)
        self.last_guard_probs = guard_p
        p_sw12, p_bl60, p_cr300, p_ex30 = float(guard_p[0]), float(guard_p[1]), float(guard_p[2]), float(guard_p[3])
        self.last_ps, self.last_pb, self.last_pc, self.last_pe = p_sw12, p_bl60, p_cr300, p_ex30

        # Адаптивное снятие дискретного барьера:
        # Ослабляем ложные срабатывания табличного bleed60 на растущем рынке (требуем p_bl60 >= 0.78 либо связку с crash300 >= 0.52),
        # а для 24-слотового Чемпиона (l7_bid_active) опираемся на его собственные нейросетевые головы риска (l7_safe_haz).
        danger_crash = (p_cr300 >= 0.65) or (p_bl60 >= 0.78) or (p_bl60 >= 0.68 and p_cr300 >= 0.52)
        safe_maker_flow = (not danger_crash) and (not is_toxic_block) and (self.last_p_adverse < 0.50)

        # Прямое полномочие 24-слотового Чемпиона (release_champion_v1.pt) на открытие и удержание шлюза покупки:
        l7_out_cur = getattr(self, "last_l7_out", None)
        l7_bid_active = False
        l7_cancel_active = False
        if l7_out_cur is not None and len(l7_out_cur) == 26:
            l7_cancel_active = float(l7_out_cur[24]) > 0.75
            l7_max_b = float(np.max(l7_out_cur[:12]))
            l7_safe_haz = (float(getattr(self, "last_l7_pb", 0.0)) < 0.58) and (float(getattr(self, "last_l7_pc", 0.0)) < 0.58)
            tot_bud_c = max(25.0, float(self.initial_budget_usd), float(self.bot_cash_usd + self.pos_qty * b1_p))
            pos_usd_c = float(self.pos_qty * max(self.avg_entry, b1_p))
            targ_exp_c = tot_bud_c * float(getattr(self, "last_l7_target_exp_frac", 1.0))
            min_bw_thresh = 0.08 if self.active_bid_id is not None else 0.10
            l7_bid_active = (not l7_cancel_active) and (l7_max_b >= min_bw_thresh) and l7_safe_haz and (pos_usd_c < targ_exp_c - 5.0) and (p_cr300 < 0.68)
        self.last_l7_bid_active = bool(l7_bid_active)

        in_dip_zone = ((in_dip_zone or in_breakout_zone or is_accum_zone or is_break_zone or safe_maker_flow) and not is_toxic_block and not danger_crash) or l7_bid_active
        self.last_can_buy_best = bool(in_dip_zone and ((b1_dom >= 0.38 and imb_l5 > -0.65) or l7_bid_active))

        mask = np.zeros(7, dtype=bool)
        in_cooldown = time.time() < getattr(self, "cooldown_until_ts", 0.0)
        if not has_pos:
            mask[0] = True
            if has_bid:
                drift_limit = self.tick_size * 1.25 if getattr(self, "active_bid_act", 1) == 2 else tol
                brk_hold = in_breakout_zone or l7_bid_active
                bid_age = getattr(self, "bid_age_sec", 0)
                need_cancel = l7_cancel_active or (not l7_bid_active and bid_age >= 15 and ((inertia < -0.15) or (imb_l5 < -0.65))) or (b1 > self.active_bid_p + drift_limit)
                if need_cancel:
                    mask[5] = True
                    if in_dip_zone and not in_cooldown:
                        mask[2] = True
                        if imb_l5 > -0.65 or l7_bid_active:
                            mask[1] = True
                elif (getattr(self, "active_bid_act", 1) == 2 or l7_bid_active) and in_dip_zone and (imb_l5 > -0.65 or l7_bid_active) and not in_cooldown:
                    mask[1] = True
                    mask[2] = True
            elif in_dip_zone and not in_cooldown:
                mask[2] = True
                if imb_l5 > -0.65 or l7_bid_active:
                    mask[1] = True
        else:
            mask[0] = bool(has_ask)
            mask[3] = True
            mask[4] = True
            threat_exit = (p_ex30 >= 0.65) or (p_cr300 >= 0.70)
            if unr_ticks <= -2.0 or inertia < -0.25 or obs_err12_norm < -0.60 or threat_exit:
                mask[6] = True
        return obs, mask

    def _compute_adaptive_wave_weight(self) -> float:
        ez60 = float(getattr(self, "last_ez60", 0.0))
        ez25 = float(self.last_ez25)
        ez12 = float(self.last_ez12)
        hold_edge = float(self.last_hold_edge)
        b1_cur = float(self.bid_hist_1h[-1]) if len(self.bid_hist_1h) > 0 else self.avg_entry
        a1_cur = b1_cur + float(getattr(self, "last_sp_ticks", 1.0)) * self.tick_size
        vel10 = float(getattr(self, "last_vel10", 0.0))
        vel3 = float(getattr(self, "last_vel3", 0.0))
        imb5 = float(getattr(self, "last_imb5", 0.0))
        inertia = float(getattr(self, "last_inertia", 0.0))
        step_in_supported = (
            self.avg_entry > 0
            and b1_cur >= self.avg_entry - self.tick_size * 1.25
            and (
                a1_cur >= self.avg_entry + 1.8 * self.tick_size
                or (
                    (self.last_pos3600 >= 0.30 or self.last_dd300 >= 3.0)
                    and float(getattr(self, "last_b1_q", 0.0)) >= 4000.0
                    and (imb5 >= 0.20 or (ez12 >= 0.55 and ez25 >= 0.35 and float(getattr(self, "last_a1_q", self.last_q_a1)) <= 2500.0))
                )
            )
        )
        if self.avg_entry > 0 and b1_cur < self.avg_entry - self.tick_size * 0.25 and not step_in_supported:
            return 0.0
        if float(getattr(self, "last_pos14400", 0.5)) <= 0.15 and float(getattr(self, "ez12_entry", 0.0)) < -0.25 and float(getattr(self, "ez25_entry", 0.0)) < -0.25:
            return 0.0
        if (
            vel10 >= 1.0
            or getattr(self, "last_in_breakout", False)
            or (self.avg_entry > 0 and b1_cur >= self.avg_entry + self.tick_size * 0.75 and inertia >= 0.0)
            or (imb5 >= 0.22 and inertia >= 0.25 and ez12 >= 0.15)
            or (ez60 >= 0.28 and ez25 >= 0.15 and inertia >= 0.20)
            or (
                imb5 >= 0.35
                and float(getattr(self, "last_b1_q", 0.0)) >= 5000.0
                and float(getattr(self, "last_a1_q", self.last_q_a1)) <= 1500.0
                and (inertia >= 0.05 or (a1_cur >= self.avg_entry + 1.8 * self.tick_size and imb5 >= 0.50))
            )
        ):
            return 1.0
        q_a1 = float(self.last_q_a1)
        q_a2 = float(self.last_q_a2)
        own_ask_share = self.pos_qty / (q_a1 + self.pos_qty + 1e-6)
        if self.last_pos3600 >= 0.65 and q_a2 > 3000.0 and vel10 <= 0.0 and inertia < 0.25 and ez60 < 0.20:
            return 0.0
        if q_a2 > 15000.0 and q_a1 < 500.0:
            return 0.50
        if (hold_edge >= 0.90 and ez25 >= 0.15 and ez12 >= 0.08) or (hold_edge >= 0.80 and q_a2 < 250.0 and q_a1 > 500.0):
            return 1.0
        if (hold_edge >= 0.45 and (ez25 >= 0.08 or ez60 >= 0.15)) or (ez25 >= 0.14 and hold_edge >= 0.20) or (own_ask_share > 0.15 and ez25 >= 0.08 and hold_edge >= 0.10):
            q_adv = q_a1 / (q_a1 + q_a2 + 200.0)
            return float(np.clip(0.35 + 0.12 * max(0.0, hold_edge) + 0.25 * q_adv, 0.30, 0.65))
        return 0.0

    def _mc_lookahead_refine(self, obs: np.ndarray, mask: np.ndarray, act: int) -> int:
        """
        15-секундный векторный Lookahead-фильтр стакана по распределениям Триады probs[-3..+3] (H=5, 12, 25, 60с):
        оценивает математическое ожидание E[PnL] и вероятность исполнения P(fill) для QUOTE_BID_1 vs QUOTE_BID_2
        перед выставлением ордера в стакан.
        """
        can_buy_more = (self.pos_qty <= 1e-4) or bool(getattr(self, "micro_pos_parked", False))
        if act not in (1, 2) or not can_buy_more:
            return act
        probs_list = getattr(self, "last_triad_probs", None)
        if not probs_list or len(probs_list) < 3:
            return act
        p5, p12, p25 = probs_list[0], probs_list[1], probs_list[2]
        p60 = getattr(self, "last_p60", p25)
        k_vals = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        b1_q = float(getattr(self, "last_b1_q", 500.0))
        a1_q = float(getattr(self, "last_a1_q", 500.0))
        sp = float(getattr(self, "last_sp_ticks", 1.0))
        fee_ticks = 0.32

        # Вероятность проторговки очереди на пассивном биде b1 (QUOTE_BID_2) за 12 секунд
        q_pass_factor = float(np.clip(350.0 / (b1_q + 350.0), 0.08, 0.90))
        p_fill_b2 = float(np.clip(p5[0] + p5[1] + p5[2] + p5[3] * q_pass_factor, 0.05, 0.95))
        # Вероятность исполнения при перешагивании в спред (QUOTE_BID_1, впереди очереди q_ahead=0)
        p_fill_b1 = float(np.clip(p5[0] + p5[1] + p5[2] + p5[3] * 0.82, 0.15, 0.98)) if sp >= 2.0 else p_fill_b2

        # Ожидаемый профит после входа (на горизонте 12-60с) за вычетом риска токсичного пробоя (-2..-3 тика)
        ez_post = 0.35 * float(np.dot(p12, k_vals)) + 0.40 * float(np.dot(p25, k_vals)) + 0.25 * float(np.dot(p60, k_vals))
        tail_crash_prob = 0.45 * float(p5[0] + p5[1]) + 0.35 * float(p12[0] + p12[1]) + 0.20 * float(p60[0] + p60[1])
        adverse_pen = tail_crash_prob * 2.4

        # E[PnL] в тиках для пассивного бида b1 (захват полного спреда) и агрессивного b1+1 (минус 1 тик спреда)
        ev_b2 = p_fill_b2 * (max(1.0, sp * 0.75) + ez_post - fee_ticks) - adverse_pen
        ev_b1 = p_fill_b1 * (max(1.0, (sp - 1.0) * 0.85) + ez_post - fee_ticks) - adverse_pen * 1.15

        # 1. Защита от токсичного пролива: если оба варианта дают отрицательное МО при высоком хвосте падения
        l7_bid_act = bool(getattr(self, "last_l7_bid_active", False))
        if ev_b1 < -0.35 and ev_b2 < -0.35 and tail_crash_prob > (0.55 if l7_bid_act else 0.36) and mask[0] and self.active_bid_id is None:
            return 0
        if l7_bid_act:
            return act
        # 2. При спреде >= 2 тика: если пассивный вход дает заметно выше МО, чем шаг +1 тик внутрь спреда -> выбираем act=2
        if act == 1 and mask[2] and sp >= 2.0 and ev_b2 > ev_b1 + 0.12:
            return 2
        # 3. Если выбран пассивный act=2, но перед нами огромная стена b1_q > 3500 и спред >= 2 тика, а шаг +1 тик высокопрофитный
        if act == 2 and mask[1] and sp >= 2.0 and b1_q > 3500.0 and ev_b1 >= 0.45 and ev_b1 > ev_b2:
            return 1
        return act

    def decide_action(self, obs: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
        if self.mode == "live" and self.is_24slot_active:
            # В боевом режиме 24-слотовой матрицы (execute_live_24slot_grid) не тратим CPU на устаревший 1-ордерный PPO + MC-Lookahead
            return 0, 0.0

        import torch
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
            logits, val = self.agent(obs_t, mask_t)
            act = int(torch.argmax(logits, dim=-1).item())
            can_buy_more = (self.pos_qty <= 1e-4) or bool(getattr(self, "micro_pos_parked", False))

            # Интеграция решения 24-слотового Чемпиона Урока 7 (release_champion_v1.pt):
            l7_out = getattr(self, "last_l7_out", None)
            if l7_out is not None and len(l7_out) == 26:
                cancel_sig = float(l7_out[24]) > 0.75
                panic_sig = float(l7_out[25]) > 0.85
                b_w = l7_out[:12]
                a_w = l7_out[12:24]
                if not can_buy_more:
                    # Управление выходом из позиции по весам Ask-слотов Чемпиона L7_0150:
                    if panic_sig and mask[6]:
                        act = 6
                    elif float(np.max(a_w[4:12])) >= 0.16 and float(np.max(a_w[4:12])) > float(np.max(a_w[:4])) and mask[4]:
                        act = 4
                    elif float(np.max(a_w[:12])) >= 0.12 and mask[3]:
                        act = 3
                else:
                    # Управление входом и удержанием Bid по весам Bid-слотов и Target Exposure Cap Чемпиона L7_0150:
                    tot_bud = max(25.0, float(self.initial_budget_usd), float(self.bot_cash_usd))
                    pos_usd_only = float(self.pos_qty * max(self.avg_entry, 0.01))
                    target_exp_usd = tot_bud * float(getattr(self, "last_l7_target_exp_frac", 1.0))
                    min_bw = 0.08 if self.active_bid_id is not None else 0.10
                    if cancel_sig or float(np.max(b_w)) < min_bw:
                        if mask[5] and self.active_bid_id is not None:
                            act = 5
                        elif mask[0]:
                            act = 0
                    elif pos_usd_only < target_exp_usd - 5.0:
                        if self.active_bid_id is not None:
                            # Ордер уже стоит в стакане, и Чемпион L7_0150 продолжает держать слот (b_w >= 0.08, cancel=False):
                            # запрещаем старому 49D PPO дергать CANCEL_BID (act=5)!
                            if act == 5 and mask[0]:
                                act = 0
                        else:
                            if float(np.max(b_w[:2])) >= 0.10 and mask[1]:
                                act = 1
                            elif float(np.max(b_w[2:12])) >= 0.10 and mask[2]:
                                act = 2

            if act == 0 and can_buy_more and self.active_bid_id is None and getattr(self, "last_in_breakout", False):
                if mask[1]:
                    act = 1
                elif mask[2]:
                    act = 2
            act = self._mc_lookahead_refine(obs, mask, act)
            return act, float(val.item())

    def compute_l25_regulator_plan(self, b1: float, a1: float, now_ts: float = None, maker_fee: float = 0.001) -> dict:
        """
        Чистый математический расчёт плана Послотового Микро-Регулятора Урока 25 (1-в-1 с `fused_l25_regulator_tick.py`):
        - Свободные целевые объёмы `q_target_b`, `q_target_a` (не зависящие от соседних стоящих слотов!)
        - Матрица состояния 24 слотов `[24, 16] -> SiLU(32) -> [24, 2]` (`g_place`, `g_cancel`)
        - Никаких ручных таймеров (`30s`, `180s`), `upsize_b` или лимитов `top-3`.
        """
        import torch
        import torch.nn.functional as F

        st = self.last_l25_state
        if now_ts is None:
            now_ts = float(st.get("now_q_ts", time.time()))
        tick = float(self.tick_size)
        b1_c = max(tick, float(b1))
        a1_c = max(b1_c + tick, float(a1))

        exp_bid_frac = float(st["exp_bid_frac"])
        pi_bid = np.asarray(st["pi_bid"], dtype=np.float32)
        pi_ask = np.asarray(st["pi_ask"], dtype=np.float32)
        panic_prob = float(st["panic_prob"])
        ctx_4d_b = np.asarray(st["ctx_4d_b"], dtype=np.float32)
        ctx_4d_a = np.asarray(st["ctx_4d_a"], dtype=np.float32)

        slot_offsets = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0], dtype=np.float32) * tick
        is_penny = np.array([True, True, False, False, False, False, False, False, False, False, False, False], dtype=bool)
        is_ask_penny = np.array([True, True, True, True, False, False, False, False, False, False, False, False], dtype=bool)
        wide_sp = (a1_c - b1_c) >= (2.0 * tick - 1e-6)
        penny_add = tick if wide_sp else 0.0

        t_bid_px = np.where(is_penny, b1_c + penny_add, np.maximum(tick, b1_c - slot_offsets)).astype(np.float32)
        t_bid_px = np.round(t_bid_px / tick).astype(np.float32) * tick

        tot_bud_eff = max(float(self.initial_budget_usd), float(self.bot_cash_usd + self.pos_qty * b1_c))
        target_bid_pool_usd = max(0.0, tot_bud_eff * 0.996 * exp_bid_frac - float(self.pos_qty) * b1_c)
        q_target_b = (target_bid_pool_usd * pi_bid) / (t_bid_px * (1.0 + maker_fee))
        for k in range(6):
            s0, s1 = 2 * k, 2 * k + 1
            if (0.05 < q_target_b[s0] < 2.0) or (0.05 < q_target_b[s1] < 2.0):
                comb_b = q_target_b[s0] + q_target_b[s1]
                if pi_bid[s0] >= pi_bid[s1]:
                    q_target_b[s0], q_target_b[s1] = comb_b, 0.0
                else:
                    q_target_b[s1], q_target_b[s0] = comb_b, 0.0
        q_target_b = np.where(q_target_b >= 2.0, q_target_b, 0.0).astype(np.float32)

        self._reconcile_lifo_tranches()
        hp_now = float(self.pos_qty) >= 1.95
        has_low_tr = hp_now and (float(self.tr_low_qty) >= 1.95) and (float(self.tr_low_px) > 0.0) and (float(self.tr_low_px) < float(self.avg_entry) - 0.1 * tick)
        avg_tp_tick = math.ceil((float(self.avg_entry) + 0.55 * tick) / tick - 1e-5) * tick if (hp_now and self.avg_entry > 0.0) else a1_c
        lifo_tp_tick = math.ceil((float(self.tr_low_px) + 0.55 * tick) / tick - 1e-5) * tick if has_low_tr else avg_tp_tick
        base_avg_tp = avg_tp_tick if hp_now else a1_c
        base_lifo_tp = lifo_tp_tick if has_low_tr else base_avg_tp

        allow_scr = bool(getattr(self, "l7_allow_scratch", False))
        slot_base_tp = np.zeros(12, dtype=np.float32)
        if allow_scr:
            avg_round_tick = max(tick, round(float(self.avg_entry) / tick) * tick) if hp_now else base_avg_tp
            slot_base_tp[0] = max(tick, avg_round_tick - tick) if hp_now else base_avg_tp
            slot_base_tp[1] = avg_round_tick if hp_now else base_avg_tp
            slot_base_tp[2:6] = base_lifo_tp
            slot_base_tp[6:12] = base_avg_tp
        else:
            slot_base_tp[:6] = base_lifo_tp
            slot_base_tp[6:12] = base_avg_tp

        raw_slot_a_px = np.where(is_ask_penny, a1_c - penny_add, a1_c + slot_offsets).astype(np.float32)
        t_ask_px = np.maximum(raw_slot_a_px, slot_base_tp + slot_offsets).astype(np.float32)
        t_ask_px = np.round(t_ask_px / tick).astype(np.float32) * tick

        q_target_a = (float(self.pos_qty) * pi_ask).astype(np.float32) if hp_now else np.zeros(12, dtype=np.float32)
        need_lifo_cap = has_low_tr and (base_lifo_tp < base_avg_tp - 0.1 * tick)
        lifo_s0 = 2 if allow_scr else 0
        if need_lifo_cap:
            want_low_sum = max(float(np.sum(q_target_a[lifo_s0:6])), 1e-6)
            if want_low_sum > float(self.tr_low_qty):
                low_cap_sc = float(self.tr_low_qty) / want_low_sum
                spill_q = float(np.sum(q_target_a[lifo_s0:6] * (1.0 - low_cap_sc)))
                q_target_a[lifo_s0:6] *= low_cap_sc
                core_pi_s = max(float(np.sum(pi_ask[6:])), 1e-6)
                q_target_a[6:] += spill_q * (pi_ask[6:] / core_pi_s)

        dust_mask_a = (q_target_a > 0.0) & (q_target_a < 2.0)
        dust_sum_a = float(np.sum(q_target_a[dust_mask_a]))
        q_target_a[dust_mask_a] = 0.0
        if hp_now and dust_sum_a >= 0.01:
            best_a_t = int(np.argmax(pi_ask))
            q_target_a[best_a_t] = min(float(self.pos_qty), q_target_a[best_a_t] + dust_sum_a)
        q_target_a = np.where(q_target_a >= 2.0, q_target_a, 0.0).astype(np.float32)

        # Собираем [24, 16] тензор микросостояния слотов:
        bid_px_cur = np.array([float(s.get("px", 0.0)) if s.get("id") is not None else 0.0 for s in self.bid_slots], dtype=np.float32)
        bid_qty_cur = np.array([float(s.get("qty", 0.0)) if s.get("id") is not None else 0.0 for s in self.bid_slots], dtype=np.float32)
        bid_qa_cur = np.array([float(s.get("qa", 0.0)) if s.get("id") is not None else 0.0 for s in self.bid_slots], dtype=np.float32)
        bid_qa0_cur = np.array([float(s.get("qa0", 0.0)) if s.get("id") is not None else 0.0 for s in self.bid_slots], dtype=np.float32)
        bid_act_cur = np.array([float(s.get("act_ts", now_ts)) if s.get("id") is not None else now_ts for s in self.bid_slots], dtype=np.float64)

        ask_px_cur = np.array([float(s.get("px", 0.0)) if s.get("id") is not None else 0.0 for s in self.ask_slots], dtype=np.float32)
        ask_qty_cur = np.array([float(s.get("qty", 0.0)) if s.get("id") is not None else 0.0 for s in self.ask_slots], dtype=np.float32)
        ask_qa_cur = np.array([float(s.get("qa", 0.0)) if s.get("id") is not None else 0.0 for s in self.ask_slots], dtype=np.float32)
        ask_qa0_cur = np.array([float(s.get("qa0", 0.0)) if s.get("id") is not None else 0.0 for s in self.ask_slots], dtype=np.float32)
        ask_act_cur = np.array([float(s.get("act_ts", now_ts)) if s.get("id") is not None else now_ts for s in self.ask_slots], dtype=np.float64)

        had_b_pre = bid_qty_cur > 1e-6
        had_a_pre = ask_qty_cur > 1e-6

        res_cash_pre = float(np.sum(bid_qty_cur * bid_px_cur * (1.0 + maker_fee)))
        tot_eq_cur = max(1.0, float(self.bot_cash_usd) + float(self.pos_qty) * b1_c)
        free_cash_ratio = float(np.clip(max(0.0, float(self.bot_cash_usd) - res_cash_pre) / tot_eq_cur, 0.0, 1.0))
        free_pos_ratio = float(np.clip((float(self.pos_qty) - float(np.sum(ask_qty_cur))) / max(float(self.pos_qty), 2.0), 0.0, 1.0))

        bud_den = max(tot_bud_eff, 25.0)
        pos_den = max(float(self.pos_qty), 2.0)
        cur_val_b = np.clip((bid_qty_cur * bid_px_cur) / bud_den, 0.0, 1.5)
        targ_val_b = np.clip((q_target_b * t_bid_px) / bud_den, 0.0, 1.5)
        cur_val_a = np.clip(ask_qty_cur / pos_den, 0.0, 1.5)
        targ_val_a = np.clip(q_target_a / pos_den, 0.0, 1.5)

        rel_delta_b = np.clip((q_target_b - bid_qty_cur) / np.maximum(np.maximum(bid_qty_cur, q_target_b), 2.0), -2.0, 2.0)
        rel_delta_a = np.clip((q_target_a - ask_qty_cur) / np.maximum(np.maximum(ask_qty_cur, q_target_a), 2.0), -2.0, 2.0)

        drift_b = np.where(had_b_pre, np.clip((t_bid_px - bid_px_cur) / tick, -5.0, 5.0) / 5.0, 0.0)
        drift_a = np.where(had_a_pre, np.clip((ask_px_cur - t_ask_px) / tick, -5.0, 5.0) / 5.0, 0.0)

        b_qa_rat = np.where(had_b_pre, np.clip(bid_qa_cur / (bid_qa0_cur + 10.0), 0.0, 1.5), 2.0)
        a_qa_rat = np.where(had_a_pre, np.clip(ask_qa_cur / (ask_qa0_cur + 10.0), 0.0, 1.5), 2.0)
        qprog_b_01 = np.where(had_b_pre, np.clip(1.0 - b_qa_rat, 0.0, 1.0), 0.0)
        qprog_a_01 = np.where(had_a_pre, np.clip(1.0 - a_qa_rat, 0.0, 1.0), 0.0)
        log_qa_b = np.where(had_b_pre, np.log1p(np.maximum(bid_qa_cur, 0.0)) / 10.0, 0.0)
        log_qa_a = np.where(had_a_pre, np.log1p(np.maximum(ask_qa_cur, 0.0)) / 10.0, 0.0)
        age_b_norm = np.where(had_b_pre, np.log1p(np.maximum(now_ts - bid_act_cur, 0.0)) / 6.0, 0.0)
        age_a_norm = np.where(had_a_pre, np.log1p(np.maximum(now_ts - ask_act_cur, 0.0)) / 6.0, 0.0)
        depth_norm = slot_offsets / (6.0 * tick)
        front_1s = (np.arange(12) < 6).astype(np.float32)

        feat_b = np.column_stack([
            np.zeros(12, dtype=np.float32),
            had_b_pre.astype(np.float32),
            cur_val_b, targ_val_b, rel_delta_b, drift_b,
            qprog_b_01, log_qa_b, age_b_norm, depth_norm, front_1s,
            np.full(12, free_cash_ratio, dtype=np.float32),
            np.broadcast_to(ctx_4d_b, (12, 4)),
        ]).astype(np.float32)
        feat_a = np.column_stack([
            np.ones(12, dtype=np.float32),
            had_a_pre.astype(np.float32),
            cur_val_a, targ_val_a, rel_delta_a, drift_a,
            qprog_a_01, log_qa_a, age_a_norm, depth_norm, front_1s,
            np.full(12, free_pos_ratio, dtype=np.float32),
            np.broadcast_to(ctx_4d_a, (12, 4)),
        ]).astype(np.float32)

        feat_24 = torch.from_numpy(np.concatenate([feat_b, feat_a], axis=0))  # [24, 16]
        with torch.no_grad():
            reg_h = F.silu(feat_24 @ self.l7_w_reg1 + self.l7_b_reg1.view(1, -1))
            reg_out = (reg_h @ self.l7_w_reg2 + self.l7_b_reg2.view(1, -1)).numpy()  # [24, 2]

        g_plc_b = reg_out[:12, 0] > 0.0
        g_cnl_b = reg_out[:12, 1] > 0.0
        g_plc_a = reg_out[12:, 0] > 0.0
        g_cnl_a = reg_out[12:, 1] > 0.0

        clear_b = had_b_pre & g_cnl_b
        rem_bid_qty = np.where(clear_b, 0.0, bid_qty_cur)
        rem_bid_px = np.where(clear_b, 0.0, bid_px_cur)
        res_cash_after_cnl = float(np.sum(rem_bid_qty * rem_bid_px * (1.0 + maker_fee)))
        free_cash_for_b = max(0.0, float(self.bot_cash_usd) * 0.998 - res_cash_after_cnl)
        rem_exp_for_b = max(0.0, tot_bud_eff * 0.996 * exp_bid_frac - float(self.pos_qty) * b1_c - res_cash_after_cnl)
        avail_b_usd = min(free_cash_for_b, rem_exp_for_b)

        empty_b_now = rem_bid_qty <= 1e-6
        want_place_b = empty_b_now & g_plc_b & (q_target_b >= 2.0)
        want_b_usd = np.where(want_place_b, q_target_b * t_bid_px * (1.0 + maker_fee), 0.0)
        tot_want_b_usd = float(np.sum(want_b_usd))
        scale_cap_b = (avail_b_usd / max(tot_want_b_usd, 1e-6)) if tot_want_b_usd > avail_b_usd else 1.0
        plc_b_qty = (want_b_usd * scale_cap_b) / (t_bid_px * (1.0 + maker_fee))
        valid_b = want_place_b & (plc_b_qty >= 2.0)
        plc_b_qty = np.where(valid_b, plc_b_qty, 0.0).astype(np.float32)

        under_tp_a = had_a_pre & (ask_px_cur < slot_base_tp - 0.4 * tick)
        clear_a = had_a_pre & ((not hp_now) | under_tp_a | g_cnl_a)
        rem_ask_qty = np.where(clear_a, 0.0, ask_qty_cur)
        unreserved_pos = max(0.0, float(self.pos_qty) - float(np.sum(rem_ask_qty)))
        empty_a_now = rem_ask_qty <= 1e-6
        want_place_a = empty_a_now & hp_now & g_plc_a & (q_target_a >= 2.0)
        want_a_q = np.where(want_place_a, q_target_a, 0.0).astype(np.float32)

        if need_lifo_cap:
            lifo_placed = float(np.sum(rem_ask_qty[lifo_s0:6]))
            rem_low_tr = max(0.0, float(self.tr_low_qty) - lifo_placed)
            w_low_s = max(float(np.sum(want_a_q[lifo_s0:6])), 1e-6)
            if w_low_s > rem_low_tr:
                want_a_q[lifo_s0:6] *= (rem_low_tr / w_low_s)

        tot_want_a_q = float(np.sum(want_a_q))
        scale_cap_a = (unreserved_pos / max(tot_want_a_q, 1e-6)) if tot_want_a_q > unreserved_pos else 1.0
        plc_a_qty = want_a_q * scale_cap_a
        valid_a = want_place_a & (plc_a_qty >= 2.0)
        plc_a_qty = np.where(valid_a, plc_a_qty, 0.0).astype(np.float32)

        return {
            "panic_sell": (panic_prob > 0.85) and (float(self.pos_qty) >= 1.95),
            "panic_frac": float(np.clip((panic_prob - 0.80) * 5.0, 0.25, 1.0)) if panic_prob > 0.85 else 0.0,
            "t_bid_px": t_bid_px,
            "t_ask_px": t_ask_px,
            "q_target_b": q_target_b,
            "q_target_a": q_target_a,
            "reg_out": reg_out,
            "g_plc_b": g_plc_b,
            "g_cnl_b": g_cnl_b,
            "g_plc_a": g_plc_a,
            "g_cnl_a": g_cnl_a,
            "clear_b": clear_b,
            "clear_a": clear_a,
            "under_tp_a": under_tp_a,
            "plc_b_qty": plc_b_qty,
            "plc_a_qty": plc_a_qty,
            "pi_bid": pi_bid,
            "pi_ask": pi_ask,
            "exp_bid_frac": exp_bid_frac,
        }

    def _execute_live_24slot_l25_regulator(self, b1: float, a1: float):
        """Боевое исполнение плана Послотового Микро-Регулятора Урока 25 на SafeTrade (без таймеров и эвристик)."""
        now_ts = time.time()
        plan = self.compute_l25_regulator_plan(b1, a1, now_ts=now_ts)

        if plan["panic_sell"]:
            panic_frac = plan["panic_frac"]
            ask_oids = [s["id"] for s in self.ask_slots if s.get("id") is not None]
            if ask_oids:
                list(self._io_pool.map(lambda oid: self._safe_cancel_oid(oid), ask_oids))
                for s in self.ask_slots:
                    s["id"], s["px"], s["qty"], s["act_ts"], s["acc_fill"] = None, 0.0, 0.0, 0.0, 0.0
                self._sync_24slot_orders(b1, count_hold=False)
                self.sync_account_balances(b1)
            avail_pos_q = min(self.pos_qty, max(self.ex_coin_free, self.pos_qty))
            sell_q = math.floor(avail_pos_q * panic_frac * 10000.0) / 10000.0
            if (avail_pos_q - sell_q) < 2.0 and avail_pos_q >= 2.0:
                sell_q = math.floor(avail_pos_q * 10000.0) / 10000.0
            if sell_q >= 2.0:
                p_panic = round(max(self.tick_size, b1 - self.tick_size) / self.tick_size) * self.tick_size
                try:
                    res_p = self.client.place_limit_order(self.market, "sell", sell_q, p_panic)
                    self.ask_slots[0] = {
                        "slot": 0,
                        "id": res_p.get("id"),
                        "px": float(res_p.get("price", p_panic)),
                        "qty": float(res_p.get("origin_amount", sell_q)),
                        "act_ts": now_ts,
                        "acc_fill": 0.0,
                    }
                    print(f"  🚨 [L25 PANIC SELL ({panic_frac:.0%})] id={self.ask_slots[0]['id']} | {sell_q:.4f} PRL @ ${p_panic:.2f}")
                except Exception as ep:
                    print(f"  ⚠️ [L25 PANIC SELL WARN]: {ep}")
                self._sync_24slot_orders(b1, count_hold=False)
            self.last_exec_act = 6
            self._save_wallet()
            return

        to_cancel_b = []
        for s in range(12):
            if plan["clear_b"][s] and self.bid_slots[s].get("id") is not None:
                reason = "reg_requeue" if (plan["g_plc_b"][s] and plan["q_target_b"][s] >= 2.0) else "reg_cancel"
                to_cancel_b.append((s, self.bid_slots[s]["id"], reason))

        to_cancel_a = []
        for s in range(12):
            if plan["clear_a"][s] and self.ask_slots[s].get("id") is not None:
                reason = "tp_guard" if plan["under_tp_a"][s] else ("reg_requeue" if (plan["g_plc_a"][s] and plan["q_target_a"][s] >= 2.0) else "reg_cancel")
                to_cancel_a.append((s, self.ask_slots[s]["id"], reason))

        if to_cancel_b or to_cancel_a:
            canceled_map = {}
            need_fill_sync = False
            all_c_oids = [x[1] for x in (to_cancel_b + to_cancel_a)]
            for oid, res_c in zip(all_c_oids, self._io_pool.map(self._safe_cancel_oid, all_c_oids)):
                if isinstance(res_c, dict):
                    canceled_map[oid] = res_c

            for s, oid, reason in to_cancel_b:
                if self.bid_slots[s].get("id") == oid:
                    c_info = canceled_map.get(oid)
                    prev_acc = float(self.bid_slots[s].get("acc_fill", 0.0))
                    s_px = float(self.bid_slots[s].get("px", 0.0))
                    no_new_fill = isinstance(c_info, dict) and float(c_info.get("filled_amount", 0.0)) - prev_acc <= 1e-4
                    if no_new_fill:
                        self.ex_usdt_free += float(self.bid_slots[s].get("qty", 0.0)) * s_px
                        self.bid_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
                    else:
                        need_fill_sync = True
                print(f"  ✖️ [L25-REG CANCEL BID #{s:02d} ({reason})] id={oid}")

            for s, oid, reason in to_cancel_a:
                if self.ask_slots[s].get("id") == oid:
                    c_info = canceled_map.get(oid)
                    prev_acc = float(self.ask_slots[s].get("acc_fill", 0.0))
                    no_new_fill = isinstance(c_info, dict) and float(c_info.get("filled_amount", 0.0)) - prev_acc <= 1e-4
                    if no_new_fill:
                        self.ex_coin_free += float(self.ask_slots[s].get("qty", 0.0))
                        self.ask_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
                    else:
                        need_fill_sync = True
                print(f"  ✖️ [L25-REG CANCEL ASK #{s:02d} ({reason})] id={oid}")

            if need_fill_sync:
                self._sync_24slot_orders(b1, count_hold=False, pre_fetched_orders=canceled_map)
                for s, oid, _ in to_cancel_b:
                    if self.bid_slots[s].get("id") == oid:
                        self.ex_usdt_free += float(self.bid_slots[s].get("qty", 0.0)) * float(self.bid_slots[s].get("px", 0.0))
                        self.bid_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
                for s, oid, _ in to_cancel_a:
                    if self.ask_slots[s].get("id") == oid:
                        self.ex_coin_free += float(self.ask_slots[s].get("qty", 0.0))
                        self.ask_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
            else:
                time.sleep(0.14)
            # Пересчитываем доступные объёмы после фактических отмен/частичных наливов:
            plan = self.compute_l25_regulator_plan(b1, a1, now_ts=now_ts)

        placed_b = 0
        b_tasks = []
        avail_ex_usdt = self.ex_usdt_free if self.ex_usdt_free > 0.0 else 1e9
        for s in sorted(range(12), key=lambda idx: float(plan["pi_bid"][idx]), reverse=True):
            if self.bid_slots[s].get("id") is None and plan["plc_b_qty"][s] >= 2.0:
                px_s = float(plan["t_bid_px"][s])
                q_s = math.floor(float(plan["plc_b_qty"][s]) * 10000.0) / 10000.0
                if avail_ex_usdt < q_s * px_s * 1.0015:
                    q_s = math.floor(((avail_ex_usdt / (px_s * 1.0015)) * 0.999) * 10000.0) / 10000.0
                if q_s >= 2.0:
                    b_tasks.append((s, q_s, px_s))
                    avail_ex_usdt = max(0.0, avail_ex_usdt - q_s * px_s * 1.0015)

        if b_tasks:
            b_results = list(self._io_pool.map(lambda item: self._place_slot_order("buy", item[0], item[1], item[2]), b_tasks))
            bq_10_ref = getattr(self, "last_clean_bq_10", np.zeros(10, dtype=np.float32))
            for s, res_o, q_req, p_req in b_results:
                if res_o and res_o.get("id") is not None:
                    oid_new = res_o.get("id")
                    q_act = float(res_o.get("origin_amount", q_req))
                    p_act = float(res_o.get("price", p_req))
                    lvl_b_i = int(np.clip(round((b1 - p_act) / self.tick_size), 0, 9))
                    base_qa_b = 0.0 if p_act > b1 + 0.1 * self.tick_size else max(10.0, float(bq_10_ref[lvl_b_i]))
                    b_over50 = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (q_act * p_act - 50.0) / 18.0))))
                    init_qa_b = base_qa_b * (1.0 + 0.75 * b_over50) + b_over50 * (0.45 * q_act)
                    self.ex_usdt_free = max(0.0, self.ex_usdt_free - q_act * p_act * 1.001)
                    self.bid_slots[s] = {
                        "slot": s,
                        "id": oid_new,
                        "px": p_act,
                        "qty": q_act,
                        "act_ts": now_ts,
                        "acc_fill": 0.0,
                        "qa_oid": oid_new,
                        "qa0": init_qa_b,
                        "qa": init_qa_b,
                    }
                    placed_b += 1
                    print(f"  📥 [L25-REG PLACE BID #{s:02d}] id={oid_new} | {q_act:.4f} PRL @ ${p_act:.2f} (π={plan['pi_bid'][s]:.2f}, E={plan['exp_bid_frac']:.2f})")

        placed_a = 0
        a_tasks = []
        avail_ex_coin = max(0.0, self.ex_coin_free - 0.0005) if self.ex_coin_free > 0.0 else float(self.pos_qty)
        for s in sorted(range(12), key=lambda idx: float(plan["pi_ask"][idx]), reverse=True):
            if self.ask_slots[s].get("id") is None and plan["plc_a_qty"][s] >= 2.0:
                px_s = float(plan["t_ask_px"][s])
                q_s = math.floor(min(float(plan["plc_a_qty"][s]), avail_ex_coin) * 10000.0) / 10000.0
                if q_s >= 2.0:
                    a_tasks.append((s, q_s, px_s))
                    avail_ex_coin = max(0.0, avail_ex_coin - q_s)

        if a_tasks:
            a_results = list(self._io_pool.map(lambda item: self._place_slot_order("sell", item[0], item[1], item[2]), a_tasks))
            aq_10_ref = getattr(self, "last_clean_aq_10", np.zeros(10, dtype=np.float32))
            for s, res_o, q_req, p_req in a_results:
                if res_o and res_o.get("id") is not None:
                    oid_new = res_o.get("id")
                    q_act = float(res_o.get("origin_amount", q_req))
                    p_act = float(res_o.get("price", p_req))
                    lvl_a_i = int(np.clip(round((p_act - a1) / self.tick_size), 0, 9))
                    base_qa_a = 0.0 if p_act < a1 - 0.1 * self.tick_size else max(10.0, float(aq_10_ref[lvl_a_i]))
                    a_over50 = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (q_act * p_act - 50.0) / 18.0))))
                    init_qa_a = base_qa_a * (1.0 + 0.75 * a_over50) + a_over50 * (0.45 * q_act)
                    self.ex_coin_free = max(0.0, self.ex_coin_free - q_act)
                    self.ask_slots[s] = {
                        "slot": s,
                        "id": oid_new,
                        "px": p_act,
                        "qty": q_act,
                        "act_ts": now_ts,
                        "acc_fill": 0.0,
                        "qa_oid": oid_new,
                        "qa0": init_qa_a,
                        "qa": init_qa_a,
                    }
                    placed_a += 1
                    print(f"  📤 [L25-REG PLACE ASK #{s:02d}] id={oid_new} | {q_act:.4f} PRL @ ${p_act:.2f} (π={plan['pi_ask'][s]:.2f})")

        self._refresh_legacy_slot_mirrors()
        if to_cancel_b and placed_b == 0:
            self.last_exec_act = 5
        elif placed_a > 0:
            self.last_exec_act = 3 if any(self.ask_slots[s].get("id") is not None and s < 4 for s in range(12)) else 4
        elif placed_b > 0:
            self.last_exec_act = 1 if any(self.bid_slots[s].get("id") is not None and s < 2 for s in range(12)) else 2
        else:
            self.last_exec_act = 0

        if to_cancel_b or to_cancel_a or placed_b > 0 or placed_a > 0:
            self._save_wallet()

    def execute_live_24slot_grid(self, b1: float, a1: float):
        """
        Прямое исполнение 24 независимых слотов (12 Bid + 12 Ask + CancelBids + PanicSell)
        в точном соответствии с математикой векторного тренажера (vectorized_free_multi_order_env.py)
        без устаревшей 1-ордерной обертки и эмпирических if/else эвристик.
        """
        if self.mode != "live" or self.client is None:
            return
        if getattr(self, "order3_mode", False) and getattr(self, "order3_engine", None) is not None and b1 > 0 and a1 > 0:
            from _tools.order3_live_bridge import execute_live_24slot_order3
            execute_live_24slot_order3(self, b1, a1)
            return
        if getattr(self, "l28_hierarchical_mode", False) and getattr(self, "last_l28_state", None) is not None and b1 > 0 and a1 > 0:
            from _tools.l28_live_bridge import execute_live_24slot_l28
            execute_live_24slot_l28(self, b1, a1)
            return
        if getattr(self, "l27_multimodal_mode", False) and getattr(self, "last_l27_state", None) is not None and b1 > 0 and a1 > 0:
            from _tools.l27_live_bridge import execute_live_24slot_l27
            execute_live_24slot_l27(self, b1, a1)
            return
        if getattr(self, "l26_endogenous_mode", False) and getattr(self, "last_l26_state", None) is not None and b1 > 0 and a1 > 0:
            from _tools.l26_live_bridge import execute_live_24slot_l26
            execute_live_24slot_l26(self, b1, a1)
            return
        if getattr(self, "l25_regulator_mode", False) and getattr(self, "last_l25_state", None) is not None and b1 > 0 and a1 > 0:
            self._execute_live_24slot_l25_regulator(b1, a1)
            return
        l7_out = getattr(self, "last_l7_out", None)
        if l7_out is None or len(l7_out) != 26 or b1 <= 0 or a1 <= 0:
            return

        now_ts = time.time()
        tol = self.tick_size * 0.25
        cancel_bids = float(l7_out[24]) > 0.75
        sel_front = bool(getattr(self, "l7_selective_front_cancel", False))
        panic_sell = (float(l7_out[25]) > 0.85) and (self.pos_qty >= 1.95)
        panic_frac = float(np.clip((float(l7_out[25]) - 0.80) * 5.0, 0.25, 1.0)) if panic_sell else 0.0
        b_w = np.asarray(l7_out[:12], dtype=np.float64).copy()
        if cancel_bids:
            if sel_front:
                b_w[:6] = 0.0
            else:
                b_w[:] = 0.0
        a_w = np.asarray(l7_out[12:24], dtype=np.float64)

        # 0. Пропорциональный экстренный выход PanicSell (raw_outputs[25] > 0.85 ➔ 25%..100% позиции)
        if panic_sell:
            ask_oids = [s["id"] for s in self.ask_slots if s.get("id") is not None]
            if ask_oids:
                list(self._io_pool.map(lambda oid: self._safe_cancel_oid(oid), ask_oids))
                for s in self.ask_slots:
                    s["id"], s["px"], s["qty"], s["act_ts"], s["acc_fill"] = None, 0.0, 0.0, 0.0, 0.0
                self._sync_24slot_orders(b1, count_hold=False)
                self.sync_account_balances(b1)
            avail_pos_q = min(self.pos_qty, max(self.ex_coin_free, self.pos_qty))
            sell_q = math.floor(avail_pos_q * panic_frac * 10000.0) / 10000.0
            if sell_q < 2.0 and avail_pos_q >= 2.0:
                sell_q = math.floor(avail_pos_q * 10000.0) / 10000.0
            if sell_q >= 2.0:
                p_panic = round(max(self.tick_size, b1 - self.tick_size) / self.tick_size) * self.tick_size
                try:
                    res_p = self.client.place_limit_order(self.market, "sell", sell_q, p_panic)
                    self.ask_slots[0] = {
                        "slot": 0,
                        "id": res_p.get("id"),
                        "px": float(res_p.get("price", p_panic)),
                        "qty": float(res_p.get("origin_amount", sell_q)),
                        "act_ts": now_ts,
                        "acc_fill": 0.0,
                    }
                    print(f"  🚨 [24-SLOT PANIC SELL ({panic_frac:.0%})] id={self.ask_slots[0]['id']} | {sell_q:.4f} PRL @ ${p_panic:.2f}")
                except Exception as ep:
                    print(f"  ⚠️ [24-SLOT PANIC SELL WARN]: {ep}")
                self._sync_24slot_orders(b1, count_hold=False)
            self.last_exec_act = 6
            self._save_wallet()
            return

        # 1. Целевая сетка цен 12 Bid-слотов и 12 Ask-слотов (1-в-1 с vectorized_free_multi_order_env.py)
        slot_offsets = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0]
        is_penny = [True, True, False, False, False, False, False, False, False, False, False, False]
        is_ask_penny = [True, True, True, True, False, False, False, False, False, False, False, False]
        wide_sp = (a1 - b1) >= 1.9 * self.tick_size

        t_bid_px = np.zeros(12, dtype=np.float64)
        for s in range(12):
            raw_bp = (b1 + self.tick_size if wide_sp else b1) if is_penny[s] else max(self.tick_size, b1 - slot_offsets[s] * self.tick_size)
            t_bid_px[s] = round(raw_bp / self.tick_size) * self.tick_size

        t_ask_px = np.zeros(12, dtype=np.float64)
        base_tp = np.zeros(12, dtype=np.float64)
        self._reconcile_lifo_tranches()
        has_low_tr = (self.pos_qty >= 1.95) and (self.tr_low_qty >= 1.95) and (self.tr_low_px > 0.0) and (self.tr_low_px < self.avg_entry - 0.1 * self.tick_size)
        if self.pos_qty >= 1.95 and self.avg_entry > 0.0:
            base_avg_tp = math.ceil((self.avg_entry + 0.55 * self.tick_size) / self.tick_size - 1e-5) * self.tick_size
            lifo_tp = math.ceil((self.tr_low_px + 0.55 * self.tick_size) / self.tick_size - 1e-5) * self.tick_size if has_low_tr else base_avg_tp
            base_lifo_tp = lifo_tp if has_low_tr else base_avg_tp
            if getattr(self, "l7_allow_scratch", False):
                avg_round_tick = round(self.avg_entry / self.tick_size) * self.tick_size
                base_tp[0] = max(self.tick_size, avg_round_tick - 1.0 * self.tick_size)
                base_tp[1] = avg_round_tick
                for s in range(2, 12):
                    slot_base = base_lifo_tp if s < 6 else base_avg_tp
                    base_tp[s] = slot_base + slot_offsets[s] * self.tick_size
                for s in range(12):
                    mkt_ap = (a1 - self.tick_size if wide_sp else a1) if is_ask_penny[s] else (a1 + slot_offsets[s] * self.tick_size)
                    t_ask_px[s] = round(max(mkt_ap, base_tp[s]) / self.tick_size) * self.tick_size
            else:
                for s in range(12):
                    slot_base = base_lifo_tp if s < 6 else base_avg_tp
                    base_tp[s] = slot_base + slot_offsets[s] * self.tick_size
                    mkt_ap = (a1 - self.tick_size if wide_sp else a1) if is_ask_penny[s] else (a1 + slot_offsets[s] * self.tick_size)
                    t_ask_px[s] = round(max(mkt_ap, base_tp[s]) / self.tick_size) * self.tick_size

        # 2. Снятие слотов с защитой приоритета очереди (Queue-Priority Preservation):
        #    - При тревоге (cancel_bids == True, голова #25 > 0.75) все Bid-слоты снимаются МГНОВЕННО (0с).
        #    - В спокойном рынке (cancel_bids == False), пока цена слота актуальна (|t_bid_px - px| < 1.9t),
        #      ордер удерживает свою позицию в очереди стакана минимум 180с (снимается по (off) только при b_w < 0.05 и age >= 180с).
        #    - Для Ask-слотов при наличии позиции (pos_qty >= 1.95) тейк-профит удерживает очередь минимум 180с до сдвига сетки,
        #      НО если после распродажи дешевого транша (tr_low) старый ордер (s >= 2) оказался НИЖЕ нового минимального тейк-профита base_tp[s],
        #      он снимается МГНОВЕННО (0с), чтобы не продать дорогой транш (tr_high) в ноль!
        to_cancel_b = []
        adv_now_b = float(getattr(self, "last_l7_adv", 0.0))
        pb_now_b = float(getattr(self, "last_l7_pb", 0.0))
        is_danger_b = cancel_bids if sel_front else (cancel_bids or (adv_now_b >= 0.12) or (pb_now_b >= 0.45))
        tot_bud_chk = max(25.0, float(self.initial_budget_usd), float(self.bot_cash_usd + self.pos_qty * b1))
        res_cash_chk = sum(float(s.get("qty", 0.0)) * float(s.get("px", 0.0)) * 1.002 for s in self.bid_slots if s.get("id") is not None)
        unres_cash_chk = max(0.0, self.bot_cash_usd - res_cash_chk)
        for s in range(12):
            slot = self.bid_slots[s]
            if slot.get("id") is None:
                continue
            age_s = now_ts - float(slot.get("act_ts", 0.0))
            if sel_front:
                slot_off_b = (cancel_bids and s < 6) or (b_w[s] < 0.06)
            else:
                min_hold_b = 30.0 if s < 4 else 120.0
                off_thresh_b = 0.06 if is_danger_b else (0.06 if s < 4 else 0.04)
                slot_off_b = cancel_bids or (b_w[s] < off_thresh_b and (is_danger_b or age_s >= min_hold_b))
            grid_shift_b = (age_s >= 10.0) and (abs(t_bid_px[s] - float(slot.get("px", 0.0))) >= 1.9 * self.tick_size)
            slot_usd = float(slot.get("qty", 0.0)) * float(slot.get("px", 0.0))
            upsize_b = (not is_danger_b) and (age_s >= 30.0) and (b_w[s] >= 0.25) and (slot_usd < 0.25 * tot_bud_chk * b_w[s]) and (unres_cash_chk > 0.25 * tot_bud_chk)
            if slot_off_b or grid_shift_b or upsize_b:
                to_cancel_b.append((s, slot["id"], "off" if slot_off_b else ("shift" if grid_shift_b else "upsize")))

        if not is_danger_b and not to_cancel_b:
            b_tot_chk = float(np.sum(np.where(b_w >= 0.08, b_w, 0.0)))
            targ_usd_chk = tot_bud_chk * float(np.clip(b_tot_chk, 0.0, 1.0))
            free_act_slots = [s for s in range(12) if self.bid_slots[s].get("id") is None and b_w[s] >= 0.12]
            if not free_act_slots and (res_cash_chk + self.pos_qty * b1) < 0.65 * targ_usd_chk and unres_cash_chk >= 50.0:
                occ_cands = [
                    s for s in range(12)
                    if self.bid_slots[s].get("id") is not None and b_w[s] >= 0.12 and (now_ts - float(self.bid_slots[s].get("act_ts", 0.0))) >= 30.0
                ]
                if occ_cands:
                    smallest_s = min(occ_cands, key=lambda idx: float(self.bid_slots[idx].get("qty", 0.0)) * float(self.bid_slots[idx].get("px", 0.0)))
                    to_cancel_b.append((smallest_s, self.bid_slots[smallest_s]["id"], "upsize"))

        to_cancel_a = []
        for s in range(12):
            slot = self.ask_slots[s]
            if slot.get("id") is None:
                continue
            age_s = now_ts - float(slot.get("act_ts", 0.0))
            cur_apx = float(slot.get("px", 0.0))
            if sel_front:
                slot_off_a = (self.pos_qty < 1.95) or (a_w[s] < 0.04)
            else:
                slot_off_a = (self.pos_qty < 1.95) or (a_w[s] < 0.04 and age_s >= 180.0) or (a_w[s] < 0.015 and age_s >= 25.0)
            under_tp_a = (self.pos_qty >= 1.95) and (cur_apx < base_tp[s] - 0.4 * self.tick_size)
            grid_shift_a = under_tp_a or ((self.pos_qty >= 1.95) and (age_s >= 15.0) and (abs(t_ask_px[s] - cur_apx) >= 1.9 * self.tick_size))
            if slot_off_a or grid_shift_a:
                to_cancel_a.append((s, slot["id"], "off" if slot_off_a else "shift"))

        if to_cancel_b or to_cancel_a:
            canceled_map = {}
            need_fill_sync = False
            all_c_oids = [x[1] for x in (to_cancel_b + to_cancel_a)]
            for oid, res_c in zip(all_c_oids, self._io_pool.map(self._safe_cancel_oid, all_c_oids)):
                if isinstance(res_c, dict):
                    canceled_map[oid] = res_c

            for s, oid, reason in to_cancel_b:
                if self.bid_slots[s].get("id") == oid:
                    c_info = canceled_map.get(oid)
                    prev_acc = float(self.bid_slots[s].get("acc_fill", 0.0))
                    s_px = float(self.bid_slots[s].get("px", 0.0))
                    no_new_fill = (
                        isinstance(c_info, dict) and float(c_info.get("filled_amount", 0.0)) - prev_acc <= 1e-4
                    )
                    if no_new_fill:
                        rel_usd = float(self.bid_slots[s].get("qty", 0.0)) * s_px
                        self.ex_usdt_free += rel_usd
                        self.bid_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
                    else:
                        need_fill_sync = True
                print(f"  ✖️ [24-SLOT CANCEL BID #{s:02d} ({reason})] id={oid}")

            for s, oid, reason in to_cancel_a:
                if self.ask_slots[s].get("id") == oid:
                    c_info = canceled_map.get(oid)
                    prev_acc = float(self.ask_slots[s].get("acc_fill", 0.0))
                    s_px = float(self.ask_slots[s].get("px", 0.0))
                    no_new_fill = (
                        isinstance(c_info, dict) and float(c_info.get("filled_amount", 0.0)) - prev_acc <= 1e-4
                    )
                    if no_new_fill:
                        rel_q = float(self.ask_slots[s].get("qty", 0.0))
                        self.ex_coin_free += rel_q
                        self.ask_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
                    else:
                        need_fill_sync = True
                print(f"  ✖️ [24-SLOT CANCEL ASK #{s:02d} ({reason})] id={oid}")

            if need_fill_sync:
                self._sync_24slot_orders(b1, count_hold=False, pre_fetched_orders=canceled_map)
                for s, oid, _ in to_cancel_b:
                    if self.bid_slots[s].get("id") == oid:
                        rel_usd = float(self.bid_slots[s].get("qty", 0.0)) * float(self.bid_slots[s].get("px", 0.0))
                        self.ex_usdt_free += rel_usd
                        self.bid_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
                for s, oid, _ in to_cancel_a:
                    if self.ask_slots[s].get("id") == oid:
                        rel_q = float(self.ask_slots[s].get("qty", 0.0))
                        self.ex_coin_free += rel_q
                        self.ask_slots[s] = {"slot": s, "id": None, "px": 0.0, "qty": 0.0, "act_ts": 0.0, "acc_fill": 0.0}
            else:
                # Даем движку SafeTrade (Peatio v2) ~140 мс освободить locked-баланс после отмены слотов перед перестановкой
                time.sleep(0.14)

        # 3. Выставление новых Bid-слотов в свободные ячейки (b_w >= 0.12) строго по формуле VectorizedFreeMultiOrderEnv:
        #    scale_b = (1.0 / b_sum) if b_sum > 4.0 else 0.25; b_alloc = b_w * scale_b
        want_b = [((not cancel_bids) and (self.bid_slots[s].get("id") is None) and (b_w[s] >= 0.12)) for s in range(12)]
        placed_b = 0
        if any(want_b):
            tot_bud = max(25.0, float(self.initial_budget_usd), float(self.bot_cash_usd + self.pos_qty * b1))
            res_cash = sum(float(s.get("qty", 0.0)) * float(s.get("px", 0.0)) * 1.002 for s in self.bid_slots if s.get("id") is not None)
            unreserved_cash = max(0.0, self.bot_cash_usd - res_cash)
            if self.ex_usdt_free > 0.0:
                unreserved_cash = min(unreserved_cash, self.ex_usdt_free)
            b_tot_sig = float(np.sum(np.where(b_w >= 0.08, b_w, 0.0)))
            target_exp_frac = float(np.clip(b_tot_sig, 0.0, 1.0))
            self.last_l7_target_exp_frac = target_exp_frac
            target_exp_usd = tot_bud * target_exp_frac
            cur_exp_usd = res_cash + self.pos_qty * b1
            avail_for_new_bids = max(0.0, min(unreserved_cash, target_exp_usd - cur_exp_usd))

            if avail_for_new_bids >= 2.0 * b1 * 1.003:
                b_sum = sum(b_w[s] for s in range(12) if want_b[s])
                scale_b = 1.0 / max(b_sum, 1e-6)
                t_bid_qty = np.zeros(12, dtype=np.float64)
                for s in range(12):
                    if want_b[s]:
                        usd_s = avail_for_new_bids * b_w[s] * scale_b
                        t_bid_qty[s] = math.floor(((usd_s * 0.998) / max(t_bid_px[s], self.tick_size)) * 10000.0) / 10000.0

                # Если отдельные транши внутри пары одного ценового эшелона (2k, 2k+1) оказались < 2.0 PRL (биржевой минимум SafeTrade),
                # укрупняем их внутри эшелона или в пользу слота с максимальным весом b_w
                for k in range(6):
                    s0, s1 = 2 * k, 2 * k + 1
                    if want_b[s0] and want_b[s1] and (0.0 < t_bid_qty[s0] < 2.0 or 0.0 < t_bid_qty[s1] < 2.0):
                        comb_q = math.floor((t_bid_qty[s0] + t_bid_qty[s1]) * 10000.0) / 10000.0
                        if b_w[s0] >= b_w[s1]:
                            t_bid_qty[s0], t_bid_qty[s1] = comb_q, 0.0
                        else:
                            t_bid_qty[s1], t_bid_qty[s0] = comb_q, 0.0
                if not any(t_bid_qty >= 2.0) and np.sum(t_bid_qty) >= 2.0:
                    best_s = int(np.argmax([b_w[s] if want_b[s] else -1.0 for s in range(12)]))
                    t_bid_qty[:] = 0.0
                    t_bid_qty[best_s] = math.floor(min(np.sum(t_bid_qty), (avail_for_new_bids * 0.998) / t_bid_px[best_s]) * 10000.0) / 10000.0

                b_tasks = sorted(
                    [(s, float(t_bid_qty[s]), float(t_bid_px[s])) for s in range(12) if t_bid_qty[s] >= 2.0],
                    key=lambda x: b_w[x[0]],
                    reverse=True,
                )[:3]
                if b_tasks:
                    b_results = list(self._io_pool.map(lambda item: self._place_slot_order("buy", item[0], item[1], item[2]), b_tasks))
                    for s, res_o, q_req, p_req in b_results:
                        if res_o and res_o.get("id") is not None:
                            q_act = float(res_o.get("origin_amount", q_req))
                            p_act = float(res_o.get("price", p_req))
                            self.ex_usdt_free = max(0.0, self.ex_usdt_free - q_act * p_act * 1.001)
                            self.bid_slots[s] = {
                                "slot": s,
                                "id": res_o.get("id"),
                                "px": p_act,
                                "qty": q_act,
                                "act_ts": now_ts,
                                "acc_fill": 0.0,
                            }
                            placed_b += 1
                            print(f"  📥 [24-SLOT PLACE BID #{s:02d}] id={self.bid_slots[s]['id']} | {self.bid_slots[s]['qty']:.4f} PRL @ ${self.bid_slots[s]['px']:.2f} (w={b_w[s]:.2f})")

        # 4. Выставление новых Ask-слотов на свободную часть позиции (a_w >= 0.12) с 100% покрытием неразмещенного остатка:
        #    scale_a = 1.0 / max(a_sum, 1e-6), чтобы весь купленный объем сразу работал в очереди тейк-профитов (+1t..+5t)
        placed_a = 0
        if self.pos_qty >= 1.95 and self.avg_entry > 0.0:
            res_pos = sum(float(s.get("qty", 0.0)) for s in self.ask_slots if s.get("id") is not None)
            unreserved_pos = max(0.0, self.pos_qty - res_pos)
            if self.ex_coin_free > 0.0:
                unreserved_pos = max(0.0, min(unreserved_pos, self.ex_coin_free - 0.0005))
            want_a = [(self.ask_slots[s].get("id") is None and a_w[s] >= 0.12) for s in range(12)]
            if unreserved_pos >= 2.0 and not any(want_a) and res_pos < 1.95:
                # Гарантируем, что позиция не висит вообще без тейк-профита, выбирая лучший свободный слот Чемпиона
                free_s = [s for s in range(12) if self.ask_slots[s].get("id") is None]
                if free_s:
                    top_s = max(free_s, key=lambda idx: a_w[idx])
                    want_a[top_s] = True

            if unreserved_pos >= 2.0 and any(want_a):
                a_sum = sum(a_w[s] for s in range(12) if want_a[s])
                scale_a = 1.0 / max(a_sum, 1e-6)
                t_ask_qty = np.zeros(12, dtype=np.float64)
                for s in range(12):
                    if want_a[s]:
                        t_ask_qty[s] = math.floor((unreserved_pos * a_w[s] * scale_a) * 10000.0) / 10000.0

                if has_low_tr and self.pos_qty >= 1.95 and self.avg_entry > 0.0 and base_lifo_tp < base_avg_tp - 0.1 * self.tick_size:
                    lifo_s0 = 2 if getattr(self, "l7_allow_scratch", False) else 0
                    lifo_placed = sum(
                        float(self.ask_slots[s].get("qty", 0.0))
                        for s in range(lifo_s0, 6)
                        if self.ask_slots[s].get("id") is not None
                    )
                    rem_low_tr = max(0.0, self.tr_low_qty - lifo_placed)
                    want_low_sum = float(np.sum(t_ask_qty[lifo_s0:6]))
                    if want_low_sum > rem_low_tr and want_low_sum > 1e-6:
                        low_cap_scale = rem_low_tr / want_low_sum
                        for s in range(lifo_s0, 6):
                            t_ask_qty[s] = math.floor((t_ask_qty[s] * low_cap_scale) * 10000.0) / 10000.0

                # Укрупнение микро-остатков < 2.0 PRL (биржевой минимум SafeTrade) в пользу самых приоритетных Ask-слотов
                dust_q = 0.0
                for s in range(12):
                    if 0.0 < t_ask_qty[s] < 2.0:
                        dust_q += t_ask_qty[s]
                        t_ask_qty[s] = 0.0
                if dust_q > 0.0:
                    ranked_a = sorted([s for s in range(12) if want_a[s]], key=lambda idx: a_w[idx], reverse=True)
                    for s in ranked_a:
                        if t_ask_qty[s] >= 2.0:
                            t_ask_qty[s] = math.floor(min(unreserved_pos, t_ask_qty[s] + dust_q) * 10000.0) / 10000.0
                            dust_q = 0.0
                            break
                    if dust_q >= 2.0 and ranked_a:
                        t_ask_qty[ranked_a[0]] = math.floor(min(unreserved_pos, dust_q) * 10000.0) / 10000.0

                a_tasks = sorted(
                    [(s, float(t_ask_qty[s]), float(t_ask_px[s])) for s in range(12) if t_ask_qty[s] >= 2.0],
                    key=lambda x: a_w[x[0]],
                    reverse=True,
                )[:4]
                if a_tasks:
                    a_results = list(self._io_pool.map(lambda item: self._place_slot_order("sell", item[0], item[1], item[2]), a_tasks))
                    for s, res_o, q_req, p_req in a_results:
                        if res_o and res_o.get("id") is not None:
                            q_act = float(res_o.get("origin_amount", q_req))
                            p_act = float(res_o.get("price", p_req))
                            self.ex_coin_free = max(0.0, self.ex_coin_free - q_act)
                            self.ask_slots[s] = {
                                "slot": s,
                                "id": res_o.get("id"),
                                "px": p_act,
                                "qty": q_act,
                                "act_ts": now_ts,
                                "acc_fill": 0.0,
                            }
                            placed_a += 1
                            print(f"  📤 [24-SLOT PLACE ASK #{s:02d}] id={self.ask_slots[s]['id']} | {self.ask_slots[s]['qty']:.4f} PRL @ ${self.ask_slots[s]['px']:.2f} (w={a_w[s]:.2f})")

        self._refresh_legacy_slot_mirrors()
        if cancel_bids and to_cancel_b:
            self.last_exec_act = 5
        elif placed_a > 0:
            self.last_exec_act = 3 if any(self.ask_slots[s].get("id") is not None and s < 4 for s in range(12)) else 4
        elif placed_b > 0:
            self.last_exec_act = 1 if any(self.bid_slots[s].get("id") is not None and s < 2 for s in range(12)) else 2
        else:
            self.last_exec_act = 0

        if to_cancel_b or to_cancel_a or placed_b > 0 or placed_a > 0:
            self._save_wallet()

    def _safe_cancel_oid(self, oid):
        for attempt in range(3):
            try:
                if attempt > 0:
                    time.sleep(0.16 * attempt)
                return self.client.cancel_order(oid)
            except Exception as e_c:
                if "429" in str(e_c) or "limit_exceeded" in str(e_c):
                    time.sleep(0.22 * (attempt + 1))
                    continue
                return None
        return None

    def _place_slot_order(self, side: str, slot_idx: int, qty: float, price: float):
        if not bool(getattr(self, "order3_mode", False)):
            time.sleep(0.11)
        p_prec = int(getattr(self, "price_precision", 4 if self.coin == "prl" else 2))
        a_prec = int(getattr(self, "amount_precision", 4))
        min_amt = float(getattr(self, "min_order_amount", 2.0))
        try:
            res = self.client.place_limit_order(
                self.market, side, qty, price,
                price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt,
            )
            return (slot_idx, res, qty, price)
        except Exception as e:
            err_s = str(e)
            if "429" in err_s or "limit_exceeded" in err_s:
                for attempt in range(1, 3):
                    time.sleep(0.25 * attempt)
                    try:
                        res_r429 = self.client.place_limit_order(
                            self.market, side, qty, price,
                            price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt,
                        )
                        return (slot_idx, res_r429, qty, price)
                    except Exception as e429:
                        err_s = str(e429)
                        if "429" not in err_s and "limit_exceeded" not in err_s:
                            break
            if "insufficient_balance" in err_s:
                time.sleep(0.18)
                retry_qty = math.floor((qty * 0.999) * 10000.0) / 10000.0
                if retry_qty >= min_amt:
                    try:
                        res_r = self.client.place_limit_order(
                            self.market, side, retry_qty, price,
                            price_precision=p_prec, amount_precision=a_prec, min_amount=min_amt,
                        )
                        return (slot_idx, res_r, retry_qty, price)
                    except Exception as e2:
                        err_s = str(e2)
            print(f"  ⚠️ [24-SLOT ORDER {side.upper()} #{slot_idx:02d} WARN] qty={qty:.4f} @ ${price:.2f}: {err_s}")
            return (slot_idx, None, qty, price)

    def execute_live_action(self, act: int, b1: float, a1: float):
        if self.is_24slot_active:
            self.execute_live_24slot_grid(b1, a1)
            return
        self.last_exec_act = int(act)

        # Orderflow Imbalance Micro-Stop: отсечение хвостовых просадок при устойчивом прессинге
        underwater_ticks = ((self.avg_entry - b1) / self.tick_size) if (self.pos_qty >= 2.0 and self.avg_entry > 0) else 0.0
        if self.pos_qty >= 2.0 and underwater_ticks >= 2.5:
            b3_vol = float(getattr(self, "last_b1_q", 0.0)) + float(getattr(self, "last_b2_q", 0.0)) + float(getattr(self, "last_b3_q", 0.0))
            a3_vol = float(getattr(self, "last_a1_q", 0.0)) + float(getattr(self, "last_a2_q", 0.0)) + float(getattr(self, "last_a3_q", 0.0))
            imb3 = (b3_vol - a3_vol) / max(1.0, b3_vol + a3_vol)
            p_adv_curr = float(getattr(self, "last_p_adverse", 0.0))
            if imb3 <= -0.45 and p_adv_curr >= 0.52:
                self.adverse_streak = getattr(self, "adverse_streak", 0) + 1
            else:
                self.adverse_streak = max(0, getattr(self, "adverse_streak", 0) - 1)
            if self.adverse_streak >= 35:
                cut_q = math.floor((self.pos_qty * 0.40) * 10000.0) / 10000.0
                if cut_q >= 2.0:
                    try:
                        print(f"  🚨 [ORDERFLOW MICRO-STOP CUT] Сброс {cut_q:.4f} PRL по ${b1:.2f} (imb3={imb3:.2f}, p_adv={p_adv_curr:.0%}, streak={self.adverse_streak}s)")
                        self.client.place_limit_order(self.market, "sell", cut_q, b1)
                        self.adverse_streak = 0
                        self.sync_account_balances(b1)
                    except Exception as e_cut:
                        print(f"  ⚠️ [ORDERFLOW CUT ERROR]: {e_cut}")
        else:
            self.adverse_streak = 0
        if self.mode != "live":
            return
        tol = self.tick_size * 0.25
        now_ts = time.time()
        in_cooldown = now_ts < getattr(self, "cooldown_until_ts", 0.0)
        can_buy_more = (self.pos_qty <= 1e-4) or bool(getattr(self, "micro_pos_parked", False))
        b1_q_w = float(getattr(self, "last_b1_q", 0.0))
        a1_q_w = float(getattr(self, "last_a1_q", self.last_q_a1))
        rng144_w = float(getattr(self, "last_rng14400", 10.0))
        pos144_w = float(getattr(self, "last_pos14400", 0.5))
        ez5_w = float(getattr(self, "last_ez5", 0.0))
        wide_scalp_ok = bool(
            (a1 - b1) >= 1.8 * self.tick_size
            and (
                (b1_q_w >= 2500.0 and a1_q_w <= 2500.0 and self.last_pos3600 <= 0.75 and self.last_inertia >= 0.05 and (self.last_inertia >= 0.20 or self.last_imb5 >= 0.50 or self.last_ez25 > -0.25))
                or (b1_q_w >= 18000.0 and a1_q_w <= 7500.0 and self.last_pos3600 <= 0.50 and rng144_w <= 9.0 and self.last_ez12 >= 0.80 and self.last_ez25 >= 0.65 and self.last_inertia >= 0.75 and self.last_imb5 >= 0.65)
                or (b1_q_w >= 1700.0 and a1_q_w <= 1250.0 and 0.45 <= self.last_pos3600 <= 0.55 and pos144_w <= 0.10 and self.last_dd900 == 1.0 and ez5_w >= 0.20 and self.last_imb5 >= 0.55)
            )
            and not (ez5_w < 0.15 and self.last_ez12 <= 0.0 and self.last_ez25 <= 0.0)
            and not (self.last_pos3600 <= 0.05 and self.last_ez12 <= 0.0 and self.last_ez25 <= 0.0 and not (b1_q_w >= 9000.0 and self.last_imb5 >= 0.60 and ez5_w >= 0.15))
            and not (self.last_pos3600 <= 0.05 and pos144_w <= 0.05 and rng144_w >= 10.0 and self.last_dd900 <= 3.5 and a1_q_w > 1000.0 and b1_q_w < 20000.0)
            and not (self.last_pos3600 <= 0.05 and pos144_w <= 0.05 and float(getattr(self, "last_vel180", 0.0)) <= -2.0 and self.last_ez25 < 0.0)
            and not (self.last_dd900 >= 8.0 and pos144_w >= 0.45 and self.last_ez25 < 0.0)
            and not (self.last_dd900 >= 10.0 and b1_q_w < 2500.0 and a1_q_w > 800.0 and self.last_ez12 < 0.0)
            and not (rng144_w >= 22.0 and self.last_ez12 <= 0.05 and self.last_ez25 <= 0.02 and a1_q_w > 400.0)
            and not (float(getattr(self, "last_entry_edge", 0.0)) < -1.50 and self.last_hold_edge < -1.50 and float(getattr(self, "last_err12", 0.0)) < -1.0 and self.last_ez25 <= 0.0)
            and not (float(getattr(self, "last_exit_ts", 0.0)) > 0.0 and (time.time() - float(getattr(self, "last_exit_ts", 0.0))) <= 15.0 and (b1 + self.tick_size) >= float(getattr(self, "last_exit_price", 0.0)) - tol and ez5_w < 0.0 and float(getattr(self, "last_err12", 0.0)) < -0.30)
            and not (ez5_w < 0.10 and self.last_ez12 < 0.10 and self.last_ez25 < -0.05 and self.last_hold_edge < 0.0 and self.last_imb5 < 0.35)
            and self.last_vel10 >= -0.5
        )
        if can_buy_more and self.active_bid_id is None and act == 0 and wide_scalp_ok and not in_cooldown:
            act = 1
        if can_buy_more and self.active_bid_id is not None:
            b2_q_w = float(getattr(self, "last_b2_q", 0.0))
            dust_b1_flicker = (
                abs(self.active_bid_p - (b1 - self.tick_size)) <= tol
                and b2_q_w >= 11000.0
                and rng144_w <= 6.0
                and pos144_w <= 0.20
            )
            at_b1 = ((self.active_bid_p >= b1 - tol) and (self.active_bid_p <= b1 + self.tick_size + tol)) or dust_b1_flicker
            compressed_4h_floor = (
                (4.8 <= rng144_w <= 5.5 and pos144_w <= 0.05 and self.last_dd900 <= 1.0)
                or (self.last_pos3600 <= 0.05 and pos144_w <= 0.05 and rng144_w <= 9.0 and b1_q_w >= 12000.0 and b1_q_w >= 1.8 * a1_q_w and self.last_imb5 >= 0.65)
                or dust_b1_flicker
            )
            l1_bid_erosion = (
                self.last_sp_ticks < 1.5
                and float(getattr(self, "last_dq_bid_5s", 0.0)) <= -1000.0
                and b1_q_w < 1800.0
                and float(getattr(self, "last_buy_vol_10s", 100.0)) < 100.0
                and not (
                    (a1_q_w <= 10.0 and self.last_inertia >= 0.15 and rng144_w >= 16.5)
                    or ((self.last_inertia >= 0.25 or a1_q_w <= 100.0) and (rng144_w <= 12.5 or 21.5 <= rng144_w <= 22.5))
                )
            )
            bid_age = getattr(self, "bid_age_sec", 0)
            stale_top_rollover = (
                bid_age >= 180
                and self.last_sp_ticks < 1.5
                and self.last_pos3600 <= 0.15
                and pos144_w >= 0.50
                and getattr(self, "bid_partial_filled", 0.0) <= 0.0
            )
            staircase_ask_wall = (
                self.last_pos3600 <= 0.05
                and pos144_w <= 0.05
                and rng144_w >= 10.0
                and self.last_dd900 <= 3.5
                and a1_q_w > 1000.0
                and b1_q_w < 20000.0
                and getattr(self, "bid_partial_filled", 0.0) <= 0.0
            )
            l1_crush = (self.last_sp_ticks < 1.5 and a1_q_w > 5000.0 and b1_q_w < 800.0 and self.last_inertia < 0.0) or l1_bid_erosion or stale_top_rollover or staircase_ask_wall
            pre_dump = ((self.last_sp_ticks < 1.5 and self.last_ez25 < -0.15 and self.last_inertia < 0.08 and a1_q_w > 4000.0) or l1_crush) and not compressed_4h_floor
            front_q_hold = (
                (
                    at_b1
                    and not pre_dump
                    and (
                        (((0.0 <= self.q_ahead < 600.0) or getattr(self, "bid_partial_filled", 0.0) > 0.0) and self.last_pos3600 <= 0.65 and self.last_inertia >= -0.10)
                        or (0.0 <= self.q_ahead < 50.0 and (b1_q_w >= 1200.0 or b2_q_w >= 1700.0) and a1_q_w <= 1250.0 and ((self.last_pos3600 <= 0.55 and pos144_w >= 0.80) or (pos144_w <= 0.20 and (rng144_w <= 12.5 or 14.5 <= rng144_w <= 16.5 or 21.5 <= rng144_w <= 22.5))) and self.last_dd900 <= 2.0)
                        or (abs(self.active_bid_p - b1) <= tol and 0.0 <= self.q_ahead < 950.0 and self.last_sp_ticks < 1.5 and a1_q_w <= 30.0 and self.last_pos3600 <= 0.05 and pos144_w <= 0.15 and self.last_dd300 >= 1.0 and self.last_dd900 >= 2.0 and self.last_imb5 >= 0.17 and (rng144_w <= 9.0 or (15.5 <= rng144_w <= 16.5 and pos144_w <= 0.02 and self.last_dd900 == 3.0)))
                        or (abs(self.active_bid_p - b1) <= tol and (a1 - b1) >= 1.8 * self.tick_size and b1_q_w >= 3000.0 and a1_q_w <= 3600.0 and 0.65 <= self.last_pos3600 <= 0.75 and pos144_w <= 0.25 and self.last_dd900 == 1.0 and 21.5 <= rng144_w <= 22.5)
                        or dust_b1_flicker
                    )
                )
                or (
                    abs(self.active_bid_p - (b1 - self.tick_size)) <= tol
                    and 0.0 <= self.q_ahead < 10.0
                    and self.last_pos3600 <= 0.50
                    and pos144_w <= 0.20
                    and 21.5 <= rng144_w <= 22.5
                    and not pre_dump
                )
            )
            brk_hold = (getattr(self, "last_in_breakout", False) or getattr(self, "last_l7_bid_active", False) or (at_b1 and (a1 - b1) >= 1.8 * self.tick_size and b1_q_w >= 1500.0 and self.last_inertia >= -0.05)) and not pre_dump
            v14_hold_q = (at_b1 or front_q_hold) and ((self.last_pos3600 <= 0.25 and not pre_dump and b1_q_w >= 5000.0) or compressed_4h_floor or front_q_hold or brk_hold)
            if self.active_bid_p > b1 + tol and self.last_vel10 <= -1.0 and ((self.last_hold_edge < 0.0 and getattr(self, "last_a1_q", self.last_q_a1) > 500.0) or (self.last_ez25 < 0.0 and self.last_ez12 < 0.10)):
                act = 5
            elif v14_hold_q:
                if act == 5 and not (getattr(self, "last_l7_out", None) is not None and float(self.last_l7_out[24]) > 0.75):
                    act = 0
            elif not getattr(self, "last_l7_bid_active", False) and ((self.last_pos3600 > 0.85 and not brk_hold) or self.last_inertia < -0.20):
                act = 5
            elif not getattr(self, "last_l7_bid_active", False) and self.last_dd300 < 0.5 and self.last_dd900 <= 1.0 and self.last_sp_ticks < 1.5 and self.last_ez12 < 0.30:
                if bid_age >= 15 and self.last_ez12 < 0.12 and self.last_inertia < 0.08 and getattr(self, "bid_partial_filled", 0.0) <= 0.0:
                    act = 5
                elif act == 5:
                    act = 0
            elif pre_dump:
                act = 5
        self.last_exec_act = int(act)

        if act in (1, 2) and can_buy_more:
            a1_q_cur = float(getattr(self, "last_a1_q", self.last_q_a1))
            b1_q_cur = float(getattr(self, "last_b1_q", 0.0))
            b1_dom = b1_q_cur / max(1.0, b1_q_cur + a1_q_cur)
            ez60_cur = float(getattr(self, "last_ez60", self.last_ez25))
            was_loss_exit = float(getattr(self, "cooldown_until_ts", 0.0)) >= float(getattr(self, "last_exit_ts", 0.0)) + 25.0
            in_post_loss_window = was_loss_exit and float(getattr(self, "last_exit_ts", 0.0)) > 0.0 and (now_ts - float(self.last_exit_ts)) <= 2700.0
            macro_step_in_ok = not in_post_loss_window and not (ez60_cur < 0.0 and self.last_dd900 >= 2.0 and b1_dom < 0.60)
            step_in_l5_ok = (
                self.last_imb5 >= 0.0
                or b1_dom >= 0.60
                or self.last_ez12 >= 0.90
                or (23.5 <= rng144_w <= 25.5 and self.last_dd900 == 1.0 and self.last_imb5 >= -0.10)
                or (18.5 <= rng144_w <= 19.5 and self.last_dd900 >= 11.0 and self.last_pos3600 <= 0.05)
            )
            top_step_risk = bool(
                (a1 - b1) >= 1.8 * self.tick_size
                and self.last_pos3600 > 0.62
                and self.last_dd300 < 1.0
                and b1_dom < 0.42
                and not getattr(self, "last_in_breakout", False)
            )
            step_in = (
                (a1 - b1) >= 1.8 * self.tick_size
                and self.pos_qty < 1.95
                and not top_step_risk
                and float(getattr(self, "last_p_adverse", 0.0)) < 0.45
                and macro_step_in_ok
                and step_in_l5_ok
                and ((b1_dom >= 0.40 and self.last_inertia >= 0.35) or wide_scalp_ok or float(getattr(self, "last_p_ask_sweep", 0.0)) >= 0.72)
            )
            t_sz = float(getattr(self, "exec_tick_size", self.tick_size))
            target_p = round(((b1 + t_sz if step_in else b1) if act == 1 else max(t_sz, b1 - t_sz)) / t_sz) * t_sz
            if self.pos_qty >= 1.95 and self.avg_entry > 0.0:
                min_open_px = min([self.avg_entry] + [float(s.get("entry_px", self.avg_entry)) for s in getattr(self, "extra_slots", []) if float(s.get("entry_px", 0.0)) > 0])
                if (time.time() - float(getattr(self, "last_buy_fill_ts", 0.0))) < 150.0 or self.last_pos3600 > 0.48:
                    target_p = min(target_p, round((min_open_px - t_sz) / t_sz) * t_sz)
            if in_post_loss_window and self.last_exit_price > 0.0 and target_p > self.last_exit_price + tol and not (getattr(self, "last_in_breakout", False) or (b1_dom >= 0.60 and ez60_cur >= 0.25 and self.last_ez25 >= 0.45)):
                target_p = min(target_p, b1)
                if target_p > self.last_exit_price + tol:
                    target_p = round(max(t_sz, b1 - t_sz) / t_sz) * t_sz
            if self.active_bid_id is not None and float(getattr(self, "last_p_frontrun", 0.0)) >= 0.70 and float(getattr(self, "last_p_adverse", 0.0)) >= 0.45:
                try:
                    self.client.cancel_order(self.active_bid_id)
                    print(f"  🛡️ [PREVENTIVE FRONT-RUN CANCEL] id={self.active_bid_id} (p_fr={self.last_p_frontrun:.0%}, p_adv={self.last_p_adverse:.0%})")
                except Exception:
                    pass
                self.sync_live_orders(b1)
                self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                return
            if self.active_bid_id is not None and abs(self.active_bid_p - target_p) <= tol:
                self.last_exec_act = 0
                return
            if self.active_bid_id is not None:
                old_bid_id = self.active_bid_id
                had_part = float(getattr(self, "bid_partial_filled", 0.0)) > 0.0
                old_bid_usd = self.active_bid_q * self.active_bid_p
                try:
                    c_res = self.client.cancel_order(old_bid_id)
                    if isinstance(c_res, dict) and float(c_res.get("filled_amount", 0.0)) > float(getattr(self, "bid_accounted_filled", 0.0)):
                        had_part = True
                    print(f"  ✖️ [LIVE CANCEL BUY] id={old_bid_id}")
                except Exception as e:
                    print(f"  ⚠️ [CANCEL BUY WARN] id={old_bid_id}: {e}")
                    had_part = True
                time.sleep(0.15)
                if had_part:
                    self.sync_live_orders(b1)
                    self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                    self.sync_account_balances(b1)
                    can_buy_more = (self.pos_qty <= 1e-4) or (
                        self.pos_qty >= 1.95
                        and self.active_ask_id is not None
                        and self.bot_cash_usd >= 25.0
                        and (self.pos_qty * b1) < 0.55 * (self.bot_cash_usd + self.pos_qty * b1)
                    )
                    if not can_buy_more:
                        return
                else:
                    self.ex_usdt_free = max(self.ex_usdt_free, min(self.bot_cash_usd, self.ex_usdt_free + old_bid_usd))
                    self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                    self.q_ahead = 0.0
                    self.bid_age_sec = 0
                    self.bid_partial_filled = 0.0
                    self.bid_accounted_filled = 0.0
            elif self.ex_usdt_free < self.bot_cash_usd * 0.90:
                self.sync_account_balances(b1)
            max_cap = self.ex_usdt_free if (self.reserved_account_usdt <= 1e-4 and self.user_reserved_coin <= 1e-4) else self.order_usd
            l7_exp_cap_usd = max(25.0, float(self.initial_budget_usd), float(self.bot_cash_usd)) * float(getattr(self, "last_l7_target_exp_frac", 1.0)) - float(self.pos_qty * max(self.avg_entry, b1))
            avail_usd = min(max_cap, max(0.0, l7_exp_cap_usd), self.bot_cash_usd * 0.998, max(self.ex_usdt_free, self.bot_cash_usd if self.active_bid_id is None else 0.0) * 0.998)
            # Адаптивное квантование бюджета с шагом 0.1% (0.001..1.000) с участием CarouselNet и HazardNet
            p10_car = float(self.last_car_probs[0]) if hasattr(self, "last_car_probs") else 0.5
            p50_car = float(self.last_car_probs[1]) if hasattr(self, "last_car_probs") else 0.5
            p150_car = float(self.last_car_probs[2]) if hasattr(self, "last_car_probs") else 0.5
            q_cap_prl = float(getattr(self, "last_car_cap_prl", 120.0))
            p_adv_now = float(getattr(self, "last_p_adverse", 0.0))
            raw_conv = (
                0.25 * max(0.0, float(getattr(self, "last_entry_edge", 0.0)))
                + 0.35 * max(0.0, self.last_inertia)
                + 0.20 * max(0.0, self.last_ez25 + self.last_ez12)
                + 0.20 * p50_car
            )
            if getattr(self, "last_in_breakout", False) or (self.last_ez12 >= 0.70 and self.last_ez25 >= 0.60 and self.last_inertia >= 0.60) or p150_car >= 0.65:
                quant_w = 1.000
            else:
                quant_w = float(np.clip(0.28 + 0.65 * raw_conv, 0.001, 0.950))
            if p_adv_now >= 0.85 and float(getattr(self, "last_dd900", 0.0)) >= 3.0:
                quant_w *= 0.55
            a1_q_usd = a1_q_cur * max(a1, self.tick_size)
            b1_q_usd = b1_q_cur * max(b1, self.tick_size)
            flow_cap_full = max(35.0, (q_cap_prl * 2.6 + 85.0 * p10_car + 220.0 * p50_car) * b1 + 0.25 * min(b1_q_usd, 600.0))
            if avail_usd <= 150.0 and flow_cap_full >= avail_usd * 1.15 and p_adv_now < 0.75:
                eff_w = 1.000
            elif avail_usd <= 600.0:
                flow_cap_usd = max(25.0, (q_cap_prl * 2.2 + 65.0 * p10_car + 180.0 * p50_car) * b1 + 0.25 * min(b1_q_usd, 600.0))
                if a1_q_usd > 120.0 and self.last_inertia < 0.35:
                    flow_cap_usd *= 0.55
                cap_w = float(np.clip(flow_cap_usd / max(avail_usd, 1.0), 0.380, 1.000))
                eff_w = 0.65 * quant_w + 0.35 * min(quant_w, cap_w)
            else:
                flow_cap_usd = flow_cap_full
                if a1_q_usd > 120.0 and self.last_inertia < 0.35:
                    flow_cap_usd *= 0.55
                cap_w = float(np.clip(flow_cap_usd / max(avail_usd, 1.0), 0.420, 1.000))
                eff_w = 0.72 * quant_w + 0.28 * min(quant_w, cap_w)
            if top_step_risk and p_adv_now >= 0.35:
                eff_w = min(eff_w, 0.500)
            if self.pos_qty <= 1e-4 and self.last_pos3600 >= 0.55 and not getattr(self, "last_in_breakout", False):
                # Progressive Tranche Scaling: запрет входа всей котлетой в верхней половине часового диапазона
                eff_w = min(eff_w, 0.400)
            if self.pos_qty >= 1.95:
                # 10-Slot Scale-In Allocation: 2..4 slots (20%..40% of total equity) per price level
                tot_eq_s = max(1.0, self.bot_cash_usd + self.pos_qty * b1)
                slot_scale_usd = tot_eq_s * min(0.420, max(0.200, 0.22 + 0.18 * p50_car))
                usable_usd = min(avail_usd, max(5.0, slot_scale_usd))
            else:
                quant_step_w = float(min(1.0, max(0.001, round(eff_w * 1000.0) / 1000.0)))
                usable_usd = max(min(avail_usd, 5.0), avail_usd * quant_step_w)
            if float(getattr(self, "last_p_frontrun", 0.0)) > 0.60 and usable_usd >= 12.0:
                stealth_mult = max(0.45, 1.0 - 0.65 * (float(self.last_p_frontrun) - 0.60))
                usable_usd = max(5.0, usable_usd * stealth_mult)
            qty = math.floor((usable_usd / max(target_p, self.tick_size)) * 10000.0) / 10000.0
            if qty < 2.0:
                return
            try:
                res = self.client.place_limit_order(self.market, "buy", qty, target_p)
            except Exception as e:
                if "insufficient_balance" in str(e):
                    time.sleep(0.20)
                    self.sync_account_balances(b1)
                    avail_u2 = min(self.order_usd, self.bot_cash_usd * 0.998, self.ex_usdt_free * 0.998)
                    qty = math.floor(min(qty, avail_u2 / max(target_p, self.tick_size)) * 10000.0) / 10000.0
                    if qty < 2.0:
                        return
                    try:
                        res = self.client.place_limit_order(self.market, "buy", qty, target_p)
                    except Exception as e2:
                        print(f"  ⚠️ [ORDER BUY ERROR] qty={qty:.4f} @ ${target_p:.2f}: {e2}")
                        return
                else:
                    print(f"  ⚠️ [ORDER BUY ERROR] qty={qty:.4f} @ ${target_p:.2f}: {e}")
                    self.sync_account_balances(b1)
                    return
            self.active_bid_id = res.get("id")
            self.active_bid_p = float(res.get("price", target_p))
            self.active_bid_q = float(res.get("origin_amount", qty))
            self.active_bid_act = int(act)
            self.ex_usdt_free = max(0.0, self.ex_usdt_free - self.active_bid_q * self.active_bid_p)
            self.q_ahead = -1.0
            self.bid_age_sec = 0
            self.bid_partial_filled = 0.0
            self.bid_accounted_filled = 0.0
            self._save_wallet()
            print(f"  📥 [LIVE ORDER BUY] id={self.active_bid_id} | {self.active_bid_q:.4f} PRL @ ${self.active_bid_p:.2f} (${self.active_bid_q*self.active_bid_p:.2f})")
        elif act == 5 and self.active_bid_id is not None:
            old_id = self.active_bid_id
            old_bid_usd = self.active_bid_q * self.active_bid_p
            had_part = float(getattr(self, "bid_partial_filled", 0.0)) > 0.0
            try:
                c_res = self.client.cancel_order(old_id)
                if isinstance(c_res, dict) and float(c_res.get("filled_amount", 0.0)) > float(getattr(self, "bid_accounted_filled", 0.0)):
                    had_part = True
                print(f"  ✖️ [LIVE CANCEL BUY] id={old_id}")
            except Exception as e:
                print(f"  ⚠️ [CANCEL BUY WARN] id={old_id}: {e}")
                had_part = True
            if had_part:
                self.sync_live_orders(b1)
                self.sync_account_balances(b1)
            else:
                self.ex_usdt_free = max(self.ex_usdt_free, min(self.bot_cash_usd, self.ex_usdt_free + old_bid_usd))
            self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
            self.q_ahead = 0.0
            self.bid_age_sec = 0
            self.bid_partial_filled = 0.0
            self.bid_accounted_filled = 0.0
            self._save_wallet()
        elif (act in (3, 4, 6) or (act == 0 and self.active_ask_id is not None)) and self.pos_qty >= 1.95:
            early_wide_unfreeze = (
                (self.avg_entry - b1) / self.tick_size >= 3.0
                and a1 >= self.avg_entry - tol
                and float(getattr(self, "last_a1_q", self.last_q_a1)) <= 250.0
                and self.last_dd900 >= 4.0
            )
            bullish_triad_hold = (
                (self.avg_entry - b1) / self.tick_size <= 4.0
                and self.last_ez12 >= 0.45
                and self.last_ez25 >= 0.45
                and float(getattr(self, "last_ez5", 0.0)) >= 0.20
            )
            has_rebound_support = (
                len(getattr(self, "extra_slots", [])) > 0
                or (
                    float(self.last_car_probs[1]) >= 0.58
                    and float(getattr(self, "last_p_adverse", 0.0)) < 0.42
                    and float(getattr(self, "last_b1_q", 0.0)) >= 3500.0
                )
            )
            unfreeze_min_sec = 900 if (float(getattr(self, "last_rng14400", 10.0)) >= 25.0 and float(getattr(self, "last_pos14400", 0.5)) <= 0.25) else (540 if has_rebound_support else 360)
            l9_out_vec = getattr(self, "last_l7_out", None)
            l9_haz_threat = (
                float(getattr(self, "last_l7_pb", 0.0)) >= 0.62
                or float(getattr(self, "last_l7_ps", 0.0)) >= 0.55
                or float(getattr(self, "last_l7_pc", 0.0)) >= 0.60
                or self.last_imb5 <= -0.38
            )
            can_stop_1 = (b1 <= self.avg_entry - 0.9 * self.tick_size) and (self.hold_sec >= 30) and l9_haz_threat
            can_scratch_0 = (
                ((b1 <= self.avg_entry - 0.5 * self.tick_size) and (self.hold_sec >= 25))
                or ((self.hold_sec >= 65) and l9_haz_threat)
            )
            l9_stop_1 = bool(can_stop_1 and getattr(self, "l7_allow_scratch", False) and l9_out_vec is not None and float(l9_out_vec[12]) >= 0.25 and float(l9_out_vec[12]) >= float(np.max(l9_out_vec[14:24])))
            l9_scratch_0 = bool(can_scratch_0 and getattr(self, "l7_allow_scratch", False) and l9_out_vec is not None and float(l9_out_vec[13]) >= 0.25 and float(l9_out_vec[13]) >= float(np.max(l9_out_vec[14:24])) + 0.12)
            unfreeze_mode = l9_stop_1 or l9_scratch_0 or (
                not bullish_triad_hold
                and (
                    ((self.avg_entry - b1) / self.tick_size >= 3.0 and self.hold_sec >= unfreeze_min_sec)
                    or early_wide_unfreeze
                )
            )
            if getattr(self, "user_manual_ask", False) and self.active_ask_id is not None and act != 6 and not unfreeze_mode:
                return
            if l9_stop_1:
                raw_floor_p = max(b1 + self.tick_size, self.avg_entry - self.tick_size)
            elif l9_scratch_0:
                raw_floor_p = max(b1 + self.tick_size, self.avg_entry)
            else:
                raw_floor_p = (self.avg_entry * 1.0022) if self.hold_sec >= 900 else (self.avg_entry + 0.75 * self.tick_size)
            floor_p = round(math.ceil(raw_floor_p / self.tick_size - 1e-4) * self.tick_size, 2)
            wave_p = round(floor_p + self.tick_size, 2)
            super_p = round(floor_p + 2.0 * self.tick_size, 2)
            car_min_ticks = 1.5 if (self.hold_sec >= 1800 and not bullish_triad_hold) else 3.0
            pump_up = (
                self.last_vel10 >= 1.0
                or getattr(self, "last_in_breakout", False)
                or (self.last_imb5 >= 0.22 and self.last_inertia >= 0.25)
                or (float(getattr(self, "last_p_ask_sweep", 0.0)) >= 0.78 and self.last_inertia >= 0.20)
            )
            car_step_up_ask = (self.last_sp_ticks < 1.5) and (
                self.last_imb5 > 0.25 and self.last_vel10 >= 1.0
            )
            opt_car_sell_p = round(min(floor_p, (a1 + self.tick_size) if car_step_up_ask else max(a1, b1 + self.tick_size)) / self.tick_size) * self.tick_size
            if unfreeze_mode and (self.active_ask_id is None or self.slot2_ask_id is not None or self.active_ask_p > opt_car_sell_p + tol or (self.active_ask_p < opt_car_sell_p - tol and car_step_up_ask)):
                for oid_attr in ("active_ask_id", "slot2_ask_id"):
                    oid = getattr(self, oid_attr)
                    if oid is not None:
                        try:
                            self.client.cancel_order(oid)
                        except Exception:
                            pass
                self.sync_live_orders(b1)
                self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                if getattr(self, "car_state", 0) != 2:
                    self.car_state = 0
                self.split_done_for_pos = False
                self.user_manual_ask = False
                self.sync_account_balances(b1)
                sell_q = math.floor(min(self.pos_qty, self.ex_coin_free) * 10000.0) / 10000.0
                if sell_q >= 1.95:
                    try:
                        ru = self.client.place_limit_order(self.market, "sell", sell_q, opt_car_sell_p)
                        self.active_ask_id = ru.get("id")
                        self.active_ask_p = float(ru.get("price", opt_car_sell_p))
                        self.active_ask_q = float(ru.get("origin_amount", sell_q))
                        self.q_ahead = -1.0
                        self._save_wallet()
                        print(f"  🔄 [UNFREEZE TO TURNOVER] id={self.active_ask_id} | Разморозка 100% поз ({self.active_ask_q:.4f} PRL) в оборот @ ${self.active_ask_p:.2f}")
                    except Exception as eu:
                        print(f"  ⚠️ [UNFREEZE SELL ERROR]: {eu}")
                return
            if unfreeze_mode and self.active_ask_id is not None:
                return
            if getattr(self, "car_state", 0) in (1, 2) and (self.active_bid_id is not None or self.bot_cash_usd >= 3.5) and self.car_sell_px > 0:
                step_back = self.tick_size if b1 >= self.car_sell_px + tol else 2.0 * self.tick_size
                max_buy_p = round((self.car_sell_px - step_back) / self.tick_size) * self.tick_size
                target_buy_p = round(min(b1, max_buy_p) / self.tick_size) * self.tick_size
                dump_danger = (self.last_inertia < -0.25 or self.last_imb5 < -0.45)
                if dump_danger and b1 <= target_buy_p + tol:
                    target_buy_p = round(max(self.tick_size, b1 - self.tick_size) / self.tick_size) * self.tick_size
                if self.active_bid_id is None and self.bot_cash_usd >= 3.5 and not dump_danger:
                    self.sync_account_balances(b1)
                    usable_u = min(self.bot_cash_usd * 0.998, self.ex_usdt_free * 0.998)
                    buy_q = math.floor((usable_u / max(target_buy_p, self.tick_size)) * 10000.0) / 10000.0
                    if buy_q >= 2.0:
                        try:
                            rb = self.client.place_limit_order(self.market, "buy", buy_q, target_buy_p)
                            self.active_bid_id = rb.get("id")
                            self.active_bid_p = float(rb.get("price", target_buy_p))
                            self.active_bid_q = float(rb.get("origin_amount", buy_q))
                            self._save_wallet()
                            print(f"  🎠 [CAROUSEL LEG-2 BUY] id={self.active_bid_id} | Откуп {self.active_bid_q:.4f} PRL @ ${self.active_bid_p:.2f} (после продажи @ ${self.car_sell_px:.2f})")
                        except Exception as eb:
                            print(f"  ⚠️ [CAROUSEL BUY ERROR]: {eb}")
                    elif self.slot2_ask_id is None:
                        self.car_state = 0
                elif self.active_bid_id is not None and (
                    (dump_danger and self.active_bid_p >= b1 - tol and target_buy_p < self.active_bid_p - tol)
                    or (not dump_danger and target_buy_p > self.active_bid_p + tol and self.last_inertia >= -0.15)
                ):
                    try:
                        self.client.cancel_order(self.active_bid_id)
                    except Exception:
                        pass
                    self.sync_live_orders(b1)
                    self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                    self.sync_account_balances(b1)
                    usable_u = min(self.bot_cash_usd * 0.998, self.ex_usdt_free * 0.998)
                    buy_q = math.floor((usable_u / max(target_buy_p, self.tick_size)) * 10000.0) / 10000.0
                    if buy_q >= 2.0:
                        try:
                            rb = self.client.place_limit_order(self.market, "buy", buy_q, target_buy_p)
                            self.active_bid_id = rb.get("id")
                            self.active_bid_p = float(rb.get("price", target_buy_p))
                            self.active_bid_q = float(rb.get("origin_amount", buy_q))
                            self._save_wallet()
                            tag = "CAROUSEL DODGE DUMP" if dump_danger else "CAROUSEL TRAIL BUY"
                            print(f"  🎠 [{tag}] id={self.active_bid_id} | Переставлен откуп {self.active_bid_q:.4f} PRL @ ${self.active_bid_p:.2f}")
                        except Exception as eb:
                            print(f"  ⚠️ [CAROUSEL TRAIL BUY ERROR]: {eb}")
            if getattr(self, "car_state", 0) == 0 and not can_buy_more and self.active_bid_id is not None and b1 > self.active_bid_p + self.tick_size + tol:
                old_b_id = self.active_bid_id
                old_b_usd = self.active_bid_q * self.active_bid_p
                try:
                    self.client.cancel_order(old_b_id)
                    print(f"  ✖️ [CANCEL STALE SCALE-IN BID] id={old_b_id} @ ${self.active_bid_p:.2f} (рынок ушел вверх Bid=${b1:.2f})")
                except Exception as e:
                    print(f"  ⚠️ [CANCEL STALE BID WARN] id={old_b_id}: {e}")
                self.ex_usdt_free = max(self.ex_usdt_free, min(self.bot_cash_usd, self.ex_usdt_free + old_b_usd))
                self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                self.q_ahead = 0.0
                self.bid_age_sec = 0
                self.bid_partial_filled = 0.0
                self.bid_accounted_filled = 0.0
                self._save_wallet()
            can_step_up_car = (getattr(self, "car_state", 0) == 2 and self.slot2_ask_id is None and a1 >= self.car_sell_px + 2.0 * self.tick_size - tol and self.pos_qty >= 20.0)
            empty_spread_ask_p = round((b1 + self.tick_size) / self.tick_size) * self.tick_size
            empty_spread_step_ask = (
                (a1 - b1) >= 1.8 * self.tick_size
                and empty_spread_ask_p >= floor_p - tol
                and not pump_up
                and self.last_hold_edge < 0.20
            )
            overhigh_ask = (
                self.active_ask_id is not None
                and self.slot2_ask_id is None
                and (
                    (self.active_ask_p > max(super_p, a1) + tol and not pump_up)
                    or (empty_spread_step_ask and self.active_ask_p > empty_spread_ask_p + tol)
                )
            )
            if act == 0:
                wave_hold_ok = b1 >= self.avg_entry - tol or (
                    b1 >= self.avg_entry - 1.25 * self.tick_size
                    and (
                        a1 >= self.avg_entry + 1.8 * self.tick_size
                        or (
                            (self.last_pos3600 >= 0.30 or self.last_dd300 >= 3.0)
                            and float(getattr(self, "last_b1_q", 0.0)) >= 4000.0
                            and (self.last_imb5 >= 0.20 or (self.last_ez12 >= 0.55 and self.last_ez25 >= 0.35 and float(getattr(self, "last_a1_q", self.last_q_a1)) <= 2500.0))
                        )
                    )
                )
                if pump_up and self.active_ask_id is not None and a1 >= self.active_ask_p - tol and self.last_imb5 >= 0.20 and b1 >= self.avg_entry:
                    self.split_done_for_pos = False
                    act = 3
                elif overhigh_ask:
                    self.split_done_for_pos = False
                    act = 4 if empty_spread_step_ask else 3
                elif self.active_ask_p >= wave_p - tol and not wave_hold_ok and self.last_hold_edge < -0.20 and a1 <= floor_p + tol:
                    act = 4
                elif (getattr(self, "car_state", 0) == 0 or can_step_up_car) and (self.avg_entry - b1) / self.tick_size >= car_min_ticks and self.hold_sec >= 60 and self.slot2_ask_id is None:
                    self.split_done_for_pos = False
                    act = 4
                elif getattr(self, "car_state", 0) == 1 and self.slot2_ask_id is not None and (
                    self.slot2_ask_p > opt_car_sell_p + 1.5 * self.tick_size
                    or (self.slot2_ask_p < opt_car_sell_p - tol and car_step_up_ask)
                ):
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
                    and (
                        (q_a2 < 4500.0 and self.last_dd900 >= 2.0)
                        or (float(getattr(self, "last_rng14400", 10.0)) >= 19.0 and self.last_ez12 >= 1.25 and self.last_ez25 >= 0.90 and self.last_hold_edge >= 3.80 and self.last_inertia >= 1.50 and self.last_dd900 >= 1.0)
                    )
                )
                is_super = (
                    (self.last_dd300 >= 4.0 and self.last_sp_ticks < 1.5 and self.last_ez25 >= 0.20 and self.last_hold_edge >= 1.50 and q_a2 < 800.0)
                    or (self.last_dd900 >= 3.0 and self.last_pos3600 <= 0.05 and self.last_ez25 >= 0.65 and self.last_hold_edge >= 4.0 and q_a2 < 300.0)
                    or triad_super
                    or (pump_up and (self.last_vel10 >= 2.0 or getattr(self, "last_vel3", 0.0) >= 2.0 or getattr(self, "last_in_breakout", False)))
                )
                car_ready_now = (getattr(self, "car_state", 0) == 0 or can_step_up_car) and (self.avg_entry - b1) / self.tick_size >= car_min_ticks and self.hold_sec >= 60 and self.slot2_ask_id is None
                if self.active_ask_id is not None and self.active_ask_p >= wave_p - tol and self.split_done_for_pos and not overhigh_ask and not car_ready_now:
                    if pump_up and a1 >= self.active_ask_p - tol and self.last_imb5 >= 0.20 and b1 >= self.avg_entry:
                        self.split_done_for_pos = False
                        act = 3
                    elif b1 < self.avg_entry - tol and self.last_hold_edge < -0.20 and a1 <= floor_p + tol:
                        act = 4
                    else:
                        return
                elif self.active_ask_id is not None and self.slot2_ask_id is not None:
                    if getattr(self, "car_state", 0) == 1:
                        need_repeg = (
                            self.slot2_ask_p > opt_car_sell_p + 1.5 * self.tick_size
                            or (self.slot2_ask_p < opt_car_sell_p - tol and car_step_up_ask)
                        )
                        if need_repeg and b1 <= self.avg_entry - min(2.0, car_min_ticks) * self.tick_size:
                            try:
                                self.client.cancel_order(self.slot2_ask_id)
                            except Exception:
                                pass
                            self.sync_live_orders(b1)
                            self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                            self.sync_account_balances(b1)
                            rem_free = math.floor(self.ex_coin_free * 10000.0) / 10000.0
                            if rem_free >= 2.0:
                                try:
                                    r2 = self.client.place_limit_order(self.market, "sell", rem_free, opt_car_sell_p)
                                    self.slot2_ask_id = r2.get("id")
                                    self.slot2_ask_p = float(r2.get("price", opt_car_sell_p))
                                    self.slot2_ask_q = float(r2.get("origin_amount", rem_free))
                                    self._save_wallet()
                                    print(f"  🎠 [CAROUSEL RE-PEG SELL] id={self.slot2_ask_id} | {self.slot2_ask_q:.4f} PRL @ ${self.slot2_ask_p:.2f}")
                                except Exception as e:
                                    print(f"  ⚠️ [CAROUSEL RE-PEG ERROR]: {e}")
                        return
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
                    if (getattr(self, "car_state", 0) == 0 or can_step_up_car) and underwater_ticks >= car_min_ticks and self.hold_sec >= 60 and self.pos_qty >= 8.0:
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
                                r2 = self.client.place_limit_order(self.market, "sell", q_car, opt_car_sell_p)
                                self.slot2_ask_id = r2.get("id")
                                self.slot2_ask_p = float(r2.get("price", opt_car_sell_p))
                                self.slot2_ask_q = float(r2.get("origin_amount", q_car))
                                self.car_state = 1
                                self._save_wallet()
                                leg_tag = "CAROUSEL STEP-UP SELL" if can_step_up_car else "CAROUSEL LEG-1 SELL"
                                print(f"  🎠 [{leg_tag}] Core={self.active_ask_q:.4f} PRL @ ${self.active_ask_p:.2f} | Carousel(25%)={self.slot2_ask_q:.4f} PRL @ ${self.slot2_ask_p:.2f}")
                                return
                            except Exception as ec:
                                print(f"  ⚠️ [CAROUSEL LEG-1 ERROR]: {ec}")
                    if pump_up and a1 >= self.active_ask_p - tol and self.last_imb5 >= 0.20 and b1 >= self.avg_entry:
                        act = 3
                    elif overhigh_ask:
                        act = 4 if empty_spread_step_ask else 3
                    elif self.active_ask_p >= wave_p - tol and b1 >= self.avg_entry - tol:
                        return
                    elif abs(self.active_ask_p - floor_p) <= tol:
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
                elif self.active_ask_id is None and w2 > 0.85:
                    self.split_done_for_pos = True
                    act = 3
                elif self.active_ask_id is None and w2 < 0.15 and not (self.last_vel10 >= 1.0 or getattr(self, "last_in_breakout", False)):
                    act = 4

            self.last_exec_act = int(act)
            if act == 6:
                target_p = max(self.tick_size, b1 - self.tick_size)
            elif act == 3:
                is_sup_step = (self.last_vel10 >= 2.0 or getattr(self, "last_in_breakout", False))
                target_p = max(a1 + (2.0 * self.tick_size if is_sup_step else 0.0), b1 + 2.0 * self.tick_size, super_p if is_sup_step else wave_p)
            else:
                if self.active_ask_id is not None and abs(self.active_ask_p - floor_p) <= tol and b1 < floor_p - tol:
                    self.last_exec_act = 0
                    return
                target_p = max(empty_spread_ask_p if empty_spread_step_ask else a1, floor_p)
            target_p = round(target_p / self.tick_size) * self.tick_size
            if self.active_ask_id is not None and self.slot2_ask_id is None and abs(self.active_ask_p - target_p) <= tol:
                self.last_exec_act = 0
                return
            had_sell_fill = False
            freed_q = 0.0
            for oid_attr, q_attr in (("active_ask_id", "active_ask_q"), ("slot2_ask_id", "slot2_ask_q")):
                oid = getattr(self, oid_attr)
                if oid is not None:
                    freed_q += float(getattr(self, q_attr, 0.0))
                    try:
                        c_res = self.client.cancel_order(oid)
                        if isinstance(c_res, dict) and float(c_res.get("filled_amount", 0.0)) > 0.0:
                            had_sell_fill = True
                    except Exception as e:
                        print(f"  ⚠️ [CANCEL SELL WARN] id={oid}: {e}")
                        had_sell_fill = True
            if had_sell_fill or (self.ex_coin_free + freed_q) < self.pos_qty * 0.95:
                self.sync_live_orders(b1)
                self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
                if self.pos_qty < 1.95:
                    return
                self.sync_account_balances(b1)
            else:
                self.ex_coin_free = max(self.ex_coin_free, min(self.pos_qty, self.ex_coin_free + freed_q))
                self.active_ask_id, self.active_ask_p, self.active_ask_q = None, 0.0, 0.0
                self.slot2_ask_id, self.slot2_ask_p, self.slot2_ask_q = None, 0.0, 0.0
            if self.pos_qty < 1.95:
                return
            sell_qty = math.floor(min(self.pos_qty, self.ex_coin_free) * 10000.0) / 10000.0
            if sell_qty < 1.95:
                return
            try:
                res = self.client.place_limit_order(self.market, "sell", sell_qty, target_p)
                self.active_ask_id = res.get("id")
                self.active_ask_p = float(res.get("price", target_p))
                self.active_ask_q = float(res.get("origin_amount", sell_qty))
                self.ex_coin_free = max(0.0, self.ex_coin_free - self.active_ask_q)
                self.q_ahead = -1.0
                self._save_wallet()
                print(f"  📤 [LIVE ORDER SELL] id={self.active_ask_id} | {self.active_ask_q:.4f} PRL @ ${self.active_ask_p:.2f}")
            except Exception as e:
                print(f"  ⚠️ [ORDER SELL ERROR] qty={sell_qty:.4f} @ ${target_p:.2f}: {e}")
                self.sync_account_balances(b1)

        # Автономная Idle-Cash микрокарусель на свободном кэше в режиме HOLD_WAIT
        if self.pos_qty < 1.95 and self.active_bid_id is None and self.active_ask_id is None and act in (0, 2):
            if self.bot_cash_usd >= 15.0 and (now_ts - float(getattr(self, "_last_idle_car_ts", 0.0))) >= 45.0:
                fh = getattr(self, "_flow_history", [])
                vol_flow_now = float(sum((x[2] + x[3]) for x in fh[-5:])) if fh else 0.0
                p50_c = float(self.last_car_probs[1]) if hasattr(self, "last_car_probs") else 0.5
                p_fill_c = float(getattr(self, "last_p_fill_real", 0.5))
                p_fr_c = float(getattr(self, "last_p_frontrun", 0.2))
                q_cap_c = float(getattr(self, "last_car_cap_prl", 120.0))
                dd900_c = float(getattr(self, "last_dd900", 0.0))
                if p50_c >= 0.58 and p_fill_c >= 0.22 and p_fr_c < 0.65 and q_cap_c >= 25.0 and dd900_c <= 1.8 and vol_flow_now >= 10.0:
                    c_usd = min(35.0, self.bot_cash_usd * 0.10)
                    c_qty = math.floor((c_usd / b1) * 10000.0) / 10000.0
                    if c_qty >= 2.0:
                        try:
                            r_c = self.client.place_limit_order(self.market, "buy", c_qty, b1)
                            self.active_bid_id = r_c.get("id")
                            self.active_bid_p = float(r_c.get("price", b1))
                            self.active_bid_q = float(r_c.get("origin_amount", c_qty))
                            self._last_idle_car_ts = now_ts
                            self._save_wallet()
                            print(f"  🎠 [IDLE-CASH CAROUSEL BUY] id={self.active_bid_id} | {self.active_bid_q:.4f} PRL @ ${self.active_bid_p:.2f} (flow={vol_flow_now:.0f}, p_fill={p_fill_c:.0%})")
                        except Exception as e_car:
                            pass

    def subsecond_guardian_step(self, streamer, rest_depth: dict = None):
        """
        Внутрисекундный Сторож Очереди и Контур Исполнения ~2.8 Гц (+350 мс и +680 мс между тиками):
        1) Экстренно снимает лимитный бид при внезапном обвале подпорки b1_q или пробое на BigONE.
        2) Мгновенно обнаруживает полный/частичный налив нашего бида (даже без пробоя уровня b1 вниз!)
           по уменьшению b1_q и выставляет тейк-профит (Ask) за <350 мс.
        """
        if self.mode != "live" or self.client is None:
            return
        if self.is_24slot_active:
            has_b_slots = any(s.get("id") is not None for s in self.bid_slots)
            has_a_slots = any(s.get("id") is not None for s in self.ask_slots)
            if not has_b_slots and not has_a_slots and self.pos_qty < 2.005:
                return
            snap = streamer.peek_fast_bbo(rest_depth)
            b1, a1 = snap["b1"], snap["a1"]
            b1_q, a1_q = snap["b1_q"], snap.get("a1_q", 0.0)
            if b1 <= 0 or a1 <= 0:
                return
            exec_t = float(getattr(self, "exec_tick_size", self.tick_size))
            tol = exec_t * 0.25
            prev_b1_q = float(getattr(self, "_last_subsec_b1_q", b1_q))
            prev_a1_q = float(getattr(self, "_last_subsec_a1_q", a1_q))
            self._last_subsec_b1_q = b1_q
            self._last_subsec_a1_q = a1_q
            if bool(getattr(self, "order3_mode", False)):
                # Строго физический событийный сторож (без внетактового шага рекуррентной памяти mem_h 1.0 Гц!):
                need_sync = False
                crossed_oids = []
                if has_b_slots:
                    top_b_px = max(float(s.get("px", 0.0)) for s in self.bid_slots if s.get("id") is not None)
                    if b1 < top_b_px - tol or ((abs(top_b_px - b1) <= tol) and ((prev_b1_q - b1_q) >= 2.0)):
                        need_sync = True
                    for s in self.bid_slots:
                        if s.get("id") is not None and float(s.get("px", 0.0)) >= (a1 - 0.1 * exec_t):
                            crossed_oids.append(("b", s["slot"], s["id"]))
                if has_a_slots:
                    low_a_px = min(float(s.get("px", 1e9)) for s in self.ask_slots if s.get("id") is not None)
                    if a1 > low_a_px + tol or ((abs(low_a_px - a1) <= tol) and ((prev_a1_q - a1_q) >= 2.0)):
                        need_sync = True
                    for s in self.ask_slots:
                        if s.get("id") is not None and float(s.get("px", 0.0)) <= (b1 + 0.1 * exec_t):
                            crossed_oids.append(("a", s["slot"], s["id"]))
                if crossed_oids and getattr(self, "order3_engine", None) is not None and self.order3_engine.api_tokens >= 1.0:
                    c_oids = [x[2] for x in crossed_oids]
                    raw_c = list(self._io_pool.map(self._safe_cancel_oid, c_oids))
                    c_m = {oid: rc for oid, rc in zip(c_oids, raw_c) if isinstance(rc, dict)}
                    self.order3_engine.api_tokens = max(0.0, self.order3_engine.api_tokens - float(len(c_oids)))
                    self._sync_24slot_orders(b1, count_hold=False, pre_fetched_orders=c_m)
                    need_sync = False
                if need_sync and (time.time() - float(getattr(self, "_last_subsec_sync_ts", 0.0))) >= 1.5:
                    self._last_subsec_sync_ts = time.time()
                    self._sync_24slot_orders(b1, count_hold=False)
                return
            if has_b_slots:
                top_b_px = max(float(s.get("px", 0.0)) for s in self.bid_slots if s.get("id") is not None)
                bid_q_drained = (abs(top_b_px - b1) <= tol) and ((prev_b1_q - b1_q) >= 2.0)
                if b1 < top_b_px - tol or bid_q_drained:
                    self._sync_24slot_orders(b1, count_hold=False)
            res_pos = sum(float(s.get("qty", 0.0)) for s in self.ask_slots if s.get("id") is not None)
            if self.pos_qty >= 2.005 and (self.pos_qty - res_pos) >= 2.005:
                self.execute_live_24slot_grid(b1, a1)
            return
        if self.active_bid_id is None and self.pos_qty < 2.005:
            return
        snap = streamer.peek_fast_bbo(rest_depth)
        b1, a1 = snap["b1"], snap["a1"]
        b1_q, a1_q, bo_b1 = snap["b1_q"], snap["a1_q"], snap["bo_b1"]
        if b1 <= 0 or a1 <= 0:
            return
        tol = self.tick_size * 0.25

        # Если есть открытая позиция >= 2.005 PRL без активного Ask (или с неполным Ask при свободном остатке >= 2.005 PRL) — ставим/объединяем Ask
        if (
            self.pos_qty >= 2.005
            and self.ex_coin_free >= 2.005
            and self.active_ask_id is not None
            and self.slot2_ask_id is None
            and getattr(self, "car_state", 0) == 0
            and not getattr(self, "user_manual_ask", False)
            and self.active_ask_q < self.pos_qty - 1.5
        ):
            self.sync_account_balances(b1)
        if self.pos_qty >= 2.005 and min(self.pos_qty, self.ex_coin_free) >= 2.001 and self.active_ask_id is None and not getattr(self, "micro_pos_parked", False):
            act_exit = 3 if (self.last_vel10 >= 1.0 or getattr(self, "last_in_breakout", False) or self.last_inertia >= 0.25) else 4
            print(f"  ⚡ [SUB-SEC 2.8Hz] Мгновенная постановка Ask (act={act_exit}) на свободную позицию {self.pos_qty:.4f} PRL")
            self.execute_live_action(act_exit, b1, a1)

        if self.active_bid_id is not None:
            prev_b1_q = float(getattr(self, "_last_subsec_b1_q", b1_q))
            self._last_subsec_b1_q = b1_q
            # Случай А: Цена пробила наш бид вниз ИЛИ объем на нашем уровне b1 уменьшился >= 2.0 PRL (вероятный налив!)
            bid_q_drained = (abs(self.active_bid_p - b1) <= tol) and ((prev_b1_q - b1_q) >= 2.0 or b1_q < self.active_bid_q - 0.5)
            if b1 < self.active_bid_p - tol or bid_q_drained:
                self.sync_live_orders(b1, count_hold=False)
                if self.pos_qty >= 2.005 and min(self.pos_qty, self.ex_coin_free) >= 2.001 and self.active_ask_id is None:
                    act_exit = 3 if (self.last_vel10 >= 1.0 or getattr(self, "last_in_breakout", False) or self.last_inertia >= 0.25) else 4
                    print(f"  ⚡ [SUB-SEC GUARDIAN] Мгновенный выход на продажу (act={act_exit}) после налива бида @ ${self.avg_entry:.2f} (<350 мс)")
                    self.execute_live_action(act_exit, b1, a1)
                if self.active_bid_id is None:
                    return

            # Случай Б: Мы стоим на лучшем биде b1, но внешняя защита очереди рухнула под лавиной аска или пробоем BigONE
            if abs(self.active_bid_p - b1) <= tol:
                ext_b1_q = max(0.0, b1_q - self.active_bid_q)
                b2 = float(snap.get("b2", 0.0))
                b2_q = float(snap.get("b2_q", 0.0))
                ext_sup_q = b2_q if (ext_b1_q <= 1e-4 and b2 >= self.active_bid_p - self.tick_size - tol) else ext_b1_q
                bo_crash = (bo_b1 > 0.0) and (bo_b1 < self.active_bid_p - 2.5 * self.tick_size)
                book_collapse = (ext_sup_q < 35.0) and (a1_q > 5.0 * max(10.0, ext_sup_q))
                if bo_crash or book_collapse:
                    oid = self.active_bid_id
                    try:
                        self.client.cancel_order(oid)
                        print(
                            f"  🛡️ [SUB-SEC GUARDIAN] Экстренное снятие бида id={oid} @ ${self.active_bid_p:.2f} за <350 мс "
                            f"(ext_sup_q={ext_sup_q:.1f}, a1_q={a1_q:.1f}, BO_bid=${bo_b1:.3f})"
                        )
                    except Exception:
                        pass
                    self.sync_live_orders(b1, count_hold=False)
                    self.active_bid_id, self.active_bid_p, self.active_bid_q = None, 0.0, 0.0
                    self.q_ahead = -1.0
                    self._save_wallet()


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
    df_init = streamer.get_recent_window(180)
    # Полный прогрев медленных каналов Order-3 при старте демона (0 холодного старта после деплоя):
    # mid по секундам из depth.csv (60m окна [66..71, 88..90], EMA m5..m3600) и лента taker-объёмов из trades.csv (30m CVD [91..93], EMA b15..s60).
    if getattr(trader, "order3_mode", False):
        try:
            from collections import defaultdict as _dd
            from _tools.order3_live_bridge import _LIVE_WF_CACHE as _o3_cache
            sec_mid = {}
            for line in raw_d_lines:
                p_d = line.split(",")
                if len(p_d) > 101:
                    try:
                        tv_d = float(p_d[0])
                        si_d = int(tv_d // 1000) if tv_d > 1e11 else int(tv_d)
                        bp_d, ap_d = float(p_d[1]), float(p_d[101])
                        if bp_d > 0 and ap_d > bp_d:
                            sec_mid[si_d] = 0.5 * (bp_d + ap_d)
                    except ValueError:
                        pass
            tr_path = Path("/opt/crypto_bot/data/realtime/prlusdt/trades.csv")
            tr_lines = subprocess.run(["tail", "-n", "40000", str(tr_path)], capture_output=True, text=True).stdout.splitlines() if tr_path.exists() else []
            sec_bv, sec_sv = _dd(float), _dd(float)
            for line in tr_lines:
                p_t = line.split(",")
                if len(p_t) >= 5:
                    try:
                        amt_t, side_t, tv_t = float(p_t[2]), p_t[3].strip().lower(), float(p_t[4])
                    except ValueError:
                        continue
                    si_t = int(tv_t // 1000) if tv_t > 1e11 else int(tv_t)
                    if side_t == "buy":
                        sec_bv[si_t] += max(0.0, amt_t)
                    elif side_t == "sell":
                        sec_sv[si_t] += max(0.0, amt_t)
            if len(sec_mid) >= 600:
                end_s = max(sec_mid)
                start_s = end_s - 3600
                cur_m = next(sec_mid[s] for s in sorted(sec_mid) if s >= start_s)
                es_w = {"b15": 0.0, "s15": 0.0, "b30": 0.0, "s30": 0.0, "b60": 0.0, "s60": 0.0, "bo_b30": 0.0, "bo_s30": 0.0, "bo_m30": cur_m,
                        "m5": cur_m, "m15": cur_m, "m30": cur_m, "m60": cur_m, "m120": cur_m, "m900": cur_m, "m3600": cur_m}
                mh_w, fh_w = [], []
                for s_w in range(start_s, end_s + 1):
                    cur_m = sec_mid.get(s_w, cur_m)
                    bv_w, sv_w = sec_bv.get(s_w, 0.0), sec_sv.get(s_w, 0.0)
                    for k_w, a_w in (("15", 2.0 / 16.0), ("30", 2.0 / 31.0), ("60", 2.0 / 61.0)):
                        es_w[f"b{k_w}"] = (1.0 - a_w) * es_w[f"b{k_w}"] + a_w * bv_w
                        es_w[f"s{k_w}"] = (1.0 - a_w) * es_w[f"s{k_w}"] + a_w * sv_w
                    for k_m, a_m in (("m5", 2.0 / 6.0), ("m15", 2.0 / 16.0), ("m30", 2.0 / 31.0), ("m60", 2.0 / 61.0), ("m120", 2.0 / 121.0), ("m900", 2.0 / 901.0), ("m3600", 2.0 / 3601.0)):
                        es_w[k_m] = (1.0 - a_m) * es_w[k_m] + a_m * cur_m
                    mh_w.append((float(s_w), cur_m))
                    if s_w > end_s - 1800 and (bv_w > 0.0 or sv_w > 0.0):
                        fh_w.append((float(s_w), bv_w, sv_w))
                es_w["bo_m30"] = cur_m
                trader._o3_ema_state = es_w
                _o3_cache["mid_hist_3600s"] = mh_w
                _o3_cache["flow_hist_1800s"] = fh_w
                _o3_cache["last_fh1800_ts"] = float(end_s)
                print(f"✅ [ORDER-3 WARMUP] Прогреты 60m mid ({len(mh_w)} сек) и 30m лента ({len(fh_w)} сек с объёмом, "
                      f"buy={sum(x[1] for x in fh_w):.0f} / sell={sum(x[2] for x in fh_w):.0f} PRL) + EMA m5..m3600", flush=True)
            else:
                print(f"⚠️ [ORDER-3 WARMUP] Недостаточно истории depth.csv ({len(sec_mid)} сек) — холодный старт медленных каналов", flush=True)
        except Exception as e_w:
            print(f"⚠️ [ORDER-3 WARMUP WARN]: {e_w}", flush=True)
    X_tail = df_init.reindex(columns=trader.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)
    X_buf = deque(X_tail, maxlen=180)
    last_row = df_init.iloc[-1:].copy()
    if getattr(trader, "l28_hierarchical_mode", False):
        trader._ensure_l28_tree_routers()
        st_init = df_init.reindex(columns=trader.st_tree_router.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)
        bo_init = df_init.reindex(columns=trader.bo_tree_router.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)
        for w_i in range(len(df_init)):
            trader.st_x_ring.append(st_init[w_i])
            trader.bo_x_ring.append(bo_init[w_i])
            if w_i >= max(0, len(df_init) - 25):
                idx_w = df_init.index[w_i]
                ts_w = int(idx_w.timestamp()) if hasattr(idx_w, "timestamp") else int(time.time())
                r_w = df_init.iloc[w_i]
                st_42_w, _, _ = trader.st_tree_router.predict_live_1row(
                    np.asarray(trader.st_x_ring, dtype=np.float32), r_w, ts_w, is_stale=False
                )
                bo_age_w = float(r_w.get("BO_sec_since_tick", 0.0) or 0.0)
                try:
                    trader.bo_tree_router.predict_live_1row(
                        np.asarray(trader.bo_x_ring, dtype=np.float32), r_w, ts_w, is_stale=(bo_age_w > 120.0)
                    )
                except Exception:
                    pass
                trader.ez12_hist.append(float(st_42_w[18]))
    elif not getattr(trader, "order3_mode", False):
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
    p_prec_log = int(getattr(trader, "price_precision", 4 if coin.lower() == "prl" else 2))
    exec_t_log = float(getattr(trader, "exec_tick_size", trader.tick_size))
    print(f"🚀 [{now_str} MSK] Старт боевого цикла {mode.upper()} на {steps} сек (лимит капитала=${budget_usd:.2f}, ордер=${order_usd:.2f}, exec_tick=${exec_t_log:.{p_prec_log}f}, feat_tick=${trader.tick_size:.2f})...")

    import threading
    shared_depth = {"data": None, "ts": 0.0, "stop": False}

    def _async_depth_poller():
        last_keepalive = 0.0
        while not shared_depth["stop"]:
            try:
                now_t = time.time()
                need_poll = not (local_depth.exists() and (now_t - local_depth.stat().st_mtime) < 1.2)
                if need_poll or (trader.client is not None and (now_t - last_keepalive) >= 25.0):
                    d_res = trader.client.get_depth(trader.market, limit=5 if not need_poll else 100)
                    last_keepalive = now_t
                    if need_poll and d_res and d_res.get("bids"):
                        shared_depth["data"] = d_res
                        shared_depth["ts"] = now_t
            except Exception:
                pass
            time.sleep(0.35)

    poller_th = threading.Thread(target=_async_depth_poller, daemon=True)
    poller_th.start()

    for step_idx in range(1, steps + 1):
        t0 = time.perf_counter()
        now_utc = datetime.now(timezone.utc)
        now_msk = now_utc.astimezone(msk_tz)
        ts_msk_str = now_msk.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        depth = shared_depth["data"] if (time.time() - shared_depth["ts"]) < 3.0 else None
        is_24s = trader.is_24slot_active
        own_orders = {
            "bid_p": trader.active_bid_p,
            "bid_q": trader.active_bid_q,
            "bids": (
                [(float(s.get("px", 0.0)), float(s.get("qty", 0.0))) for s in trader.bid_slots if s.get("id") is not None]
                if is_24s else ([(trader.active_bid_p, trader.active_bid_q)] if trader.active_bid_q > 0 else [])
            ),
            "asks": (
                [(float(s.get("px", 0.0)), float(s.get("qty", 0.0))) for s in trader.ask_slots if s.get("id") is not None]
                if is_24s else [
                    (trader.active_ask_p, trader.active_ask_q),
                    (trader.slot2_ask_p, trader.slot2_ask_q),
                    *[(float(s.get("p", 0.0)), float(s.get("q", 0.0))) for s in getattr(trader, "extra_slots", [])],
                ]
            ),
            "tick_size": float(getattr(trader, "exec_tick_size", trader.tick_size)),
        }
        last_row = streamer.step(depth, now_utc, own_orders=own_orders)

        b1_col = "ST_best_bid" if "ST_best_bid" in last_row.columns else ("best_bid" if "best_bid" in last_row.columns else last_row.columns[0])
        a1_col = "ST_best_ask" if "ST_best_ask" in last_row.columns else ("best_ask" if "best_ask" in last_row.columns else last_row.columns[1])
        b1 = float(last_row[b1_col].iloc[0])
        a1 = float(last_row[a1_col].iloc[0])
        if trader.pos_qty > 1e-4:
            trader.hold_sec += 1
            if b1 > 0:
                trader.peak_bid = max(trader.peak_bid, trader.avg_entry, b1)

        # Пред-инференсная синхронизация исполнений при касании спреда (убирает 2–5с слепоту нейросети после реального филла!):
        if getattr(trader, "order3_mode", False) and is_24s and b1 > 0:
            exec_t_pre = float(getattr(trader, "exec_tick_size", trader.tick_size))
            touched_b = any(s.get("id") is not None and float(s.get("px", 0.0)) >= b1 - 0.5 * exec_t_pre for s in trader.bid_slots)
            touched_a = any(s.get("id") is not None and float(s.get("px", 0.0)) <= a1 + 0.5 * exec_t_pre for s in trader.ask_slots)
            since_sync = time.time() - float(getattr(trader, "_last_pre_inf_sync_ts", 0.0))
            if (touched_b or touched_a) and since_sync >= 0.95:
                trader._last_pre_inf_sync_ts = time.time()
                trader.sync_live_orders(b1, count_hold=False)

        if getattr(trader, "order3_mode", False):
            obs, mask = trader.compute_obs_and_mask(None, last_row)
        else:
            x_new = last_row.reindex(columns=trader.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)[0]
            X_buf.append(x_new)
            obs, mask = trader.compute_obs_and_mask(np.array(X_buf, dtype=np.float32), last_row)
        act, v_pred = trader.decide_action(obs, mask)
        trader.execute_live_action(act, b1, a1)
        act = int(getattr(trader, "last_exec_act", act))
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
            "p_car50": round(float(trader.last_car_probs[1]), 4),
            "cap_prl": round(float(trader.last_car_cap_prl), 1),
            "p_adv": round(float(trader.last_p_adverse), 4),
            "p_exh": round(float(trader.last_p_exhaust), 4),
        }
        telemetry_f.write(json.dumps(rec) + "\n")
        telemetry_f.flush()
        if trader.client is not None:
            trader.client.append_vps_telemetry(rec)

        if step_idx <= 5 or step_idx % 5 == 0 or act != 0:
            reg_idx = int(np.argmax(getattr(trader, "last_regime_probs", [1, 0, 0, 0])))
            reg_tag = ["FLAT", "ACC", "BRK", "TOX"][reg_idx]
            gp = getattr(trader, "last_guard_probs", [0, 0, 0, 0])
            g_tag = f"G=[Sw{gp[0]*100:.0f}%|Bl{gp[1]*100:.0f}%|Cr{gp[2]*100:.0f}%|Ex{gp[3]*100:.0f}%]"
            nb_act = sum(1 for s in trader.bid_slots if s.get("id") is not None) if is_24s else (1 if trader.active_bid_id is not None else 0)
            na_act = sum(1 for s in trader.ask_slots if s.get("id") is not None) if is_24s else (1 if trader.active_ask_id is not None else 0)
            sl_tag = f"Slots=[B:{nb_act}/12 A:{na_act}/12]"
            p_prec = int(getattr(trader, "price_precision", 4 if coin.lower() == "prl" else 2))
            m_tag = ""
            if getattr(trader, "order3_mode", False):
                m_tag = " M#v1(Order3)"
            elif (getattr(trader, "l28_hierarchical_mode", False) or getattr(trader, "l27_multimodal_mode", False)) and getattr(trader, "last_l27_state", None):
                gm = trader.last_l27_state.get("g_mode", [1.0, 0.0, 0.0, 0.0])
                m_tag = f" M#{int(np.argmax(gm))}({float(np.max(gm)):.0%})"
            pnl_pct = (pnl_tot / max(1.0, trader.initial_budget_usd)) * 100.0
            print(
                f"  [{ts_msk_str} MSK | #{step_idx:04d}] Bid={b1:.{p_prec}f} Ask={a1:.{p_prec}f} | Act={trader.ACTION_NAMES[act]:<14} | "
                f"{sl_tag}{m_tag} | Reg={reg_tag} {g_tag} Adv={trader.last_p_adverse:.0%} | "
                f"Equity=${eq_usd:.2f} ({pnl_tot:+.2f}$ / {pnl_pct:+.2f}%) | {dt_ms:.1f} ms",
                flush=True,
            )

        # Используем свободные ~800 мс между 1-секундными тиками для фоновой сверки ордеров/кошелька
        # и 2 внутрисекундных проверок Контура Исполнения ~2.8 Гц (на +350 мс и +680 мс)
        has_open_orders = (
            any(s.get("id") is not None for s in (trader.bid_slots + trader.ask_slots))
            if is_24s else (
                trader.active_bid_id is not None
                or trader.active_ask_id is not None
                or trader.slot2_ask_id is not None
                or len(getattr(trader, "extra_slots", [])) > 0
            )
        )
        near_touch = False
        if has_open_orders and is_24s and b1 > 0:
            near_b = any(s.get("id") is not None and abs(b1 - float(s.get("px", 0.0))) <= b1 * 0.005 for s in trader.bid_slots)
            near_a = any(s.get("id") is not None and abs(float(s.get("px", 0.0)) - a1) <= a1 * 0.005 for s in trader.ask_slots)
            near_touch = near_b or near_a
        if (near_touch and step_idx % 2 == 0) or (has_open_orders and step_idx % 5 == 0) or (step_idx % 10 == 0):
            trader.sync_live_orders(b1, count_hold=False)
        if step_idx % 25 == 0:
            trader.sync_account_balances(b1)
        if step_idx % 30 == 0:
            trader.maybe_hot_reload_champion()

        if getattr(trader, "order3_mode", False):
            last_depth_ns = local_depth.stat().st_mtime_ns if local_depth.exists() else 0
            last_rest_ts = float(shared_depth["ts"])
            while True:
                rem_sec = 1.0 - (time.perf_counter() - t0)
                if rem_sec <= 0.015:
                    break
                time.sleep(min(0.04, rem_sec - 0.005))
                if has_open_orders or trader.pos_qty >= 2.005:
                    cur_ns = local_depth.stat().st_mtime_ns if local_depth.exists() else 0
                    cur_rest_ts = float(shared_depth["ts"])
                    if cur_ns != last_depth_ns or cur_rest_ts != last_rest_ts:
                        last_depth_ns = cur_ns
                        last_rest_ts = cur_rest_ts
                        trader.subsecond_guardian_step(streamer, shared_depth["data"])
        else:
            for sub_target in (0.35, 0.68):
                rem_sub = sub_target - (time.perf_counter() - t0)
                if rem_sub > 0.02:
                    time.sleep(rem_sub)
                    trader.subsecond_guardian_step(streamer, shared_depth["data"])

        time.sleep(max(0.005, 1.0 - (time.perf_counter() - t0)))
    shared_depth["stop"] = True


def show_live_status(tail_lines: int = 25):
    """Онлайн-запрос статуса фонового демона на VPS (81.19.137.18), ресурсов сервера, субсчета и хвоста лога."""
    from _tools.vps_config import create_ssh_client
    print("⏳ Запрос статуса демона на VPS (81.19.137.18)...", flush=True)
    ssh = create_ssh_client(timeout=12.0)
    cmd = (
        "PID=$(cat /opt/crypto_bot/data/logs/live_rl_trader.pid 2>/dev/null); "
        "if [ -z \"$PID\" ] || ! kill -0 \"$PID\" 2>/dev/null; then PID=$(pgrep -f '[_]tools[./]live_rl_trader' 2>/dev/null | head -n 1); fi; "
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
        "PIDS=$(pgrep -f '[_]tools[./]live_rl_trader' 2>/dev/null | tr '\\n' ' '); "
        "if [ -n \"$PIDS\" ]; then kill -15 $PIDS 2>/dev/null; sleep 1; kill -9 $PIDS 2>/dev/null; echo \"STOPPED:$PIDS\"; "
        "else echo \"NOT_RUNNING\"; fi; rm -f /opt/crypto_bot/data/logs/live_rl_trader.pid"
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
                for k in ("active_bid_id", "active_ask_id", "slot2_ask_id"):
                    if w_st.get(k):
                        bot_oids.add(w_st[k])
                for arr_k in ("bid_slots", "ask_slots", "extra_slots"):
                    for item in w_st.get(arr_k, []) or []:
                        if isinstance(item, dict) and item.get("id") is not None:
                            bot_oids.add(item["id"])
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


def start_live_daemon(budget_usd: float = 54.0, order_usd: float = 54.0, steps: int = 86400, push_wallet: bool = False, sync_models: bool = False):
    """Запускает / перезапускает автономный фоновый демон бота непосредственно на VPS (81.19.137.18) с авто-синхронизацией кода и .env."""
    from _tools.vps_config import create_ssh_client
    ssh = create_ssh_client(timeout=12.0)
    try:
        sftp = ssh.open_sftp()
        if sftp.get_channel() is not None:
            sftp.get_channel().settimeout(20.0)
        env_local = BASE_DIR / ".env"
        if env_local.exists():
            sftp.put(str(env_local), "/opt/crypto_bot/.env")
        for mod_name in (
            "system_resources.py", "live_rl_trader.py", "live_feature_updater.py",
            "microstructure_features.py", "swarm_router.py", "fast_neural_engine.py",
            "train_rl_execution.py", "rl_gpu_population.py", "carousel_net.py",
            "hazard_net.py", "execution_realism_net.py", "multi_slot_benchmark.py",
            "multitask_lob_net.py", "multi_horizon_guard_net.py", "microstructure_regime_gate.py",
            "cross_venue_first_order_net.py", "build_bo_swarm_36d_cache.py", "fused_l25_regulator_tick.py",
        ):
            loc_mod = BASE_DIR / "_tools" / mod_name
            if loc_mod.exists():
                sftp.put(str(loc_mod), f"/opt/crypto_bot/_tools/{mod_name}")
        rel_local = BASE_DIR / "models" / "prl" / "release_champion_v1.pt"
        if rel_local.exists():
            sftp.put(str(rel_local), "/opt/crypto_bot/models/prl/release_champion_v1.pt")
        for pt_name in ("best_maskable_ppo.pt", "carousel_net.pt", "hazard_net.pt", "execution_realism_net.pt"):
            pt_local = BASE_DIR / "models" / "prl" / "rl_execution" / pt_name
            if pt_local.exists():
                sftp.put(str(pt_local), f"/opt/crypto_bot/models/prl/rl_execution/{pt_name}")
        for af in (BASE_DIR / "models" / "prl").glob("ensemble_alliance_h*.json"):
            sftp.put(str(af), f"/opt/crypto_bot/models/prl/{af.name}")
        to_sync = []
        if sync_models:
            for pat in ("h*s_k*", "*_h*s"):
                for h_dir in (BASE_DIR / "models" / "prl").glob(pat):
                    if not h_dir.is_dir():
                        continue
                    for root, _, files in os.walk(h_dir):
                        rel_r = Path(root).relative_to(BASE_DIR / "models" / "prl")
                        rem_d = f"/opt/crypto_bot/models/prl/{rel_r}"
                        rem_attrs = {}
                        rem_mtimes = {}
                        try:
                            for a in sftp.listdir_attr(rem_d):
                                rem_attrs[a.filename] = a.st_size
                                rem_mtimes[a.filename] = getattr(a, "st_mtime", 0) or 0
                        except Exception:
                            ssh.exec_command(f"mkdir -p {rem_d}")[1].read()
                        for fn in files:
                            lf = Path(root) / fn
                            st = lf.stat()
                            if rem_attrs.get(fn) == st.st_size and abs(rem_mtimes.get(fn, 0) - st.st_mtime) < 2 and not fn.endswith(".json") and "wavenet_tcn" not in str(rel_r) and "cross_mixer" not in str(rel_r) and "causal_cnn" not in str(rel_r) and "attention" not in str(rel_r):
                                continue
                            to_sync.append((lf, f"{rel_r}/{fn}", f"{rem_d}/{fn}"))
        if len(to_sync) <= 15:
            for lf, _, rem_f in to_sync:
                sftp.put(str(lf), rem_f)
        else:
            import io
            import tarfile
            print(f"📦 Пакетная потоковая синхронизация {len(to_sync)} обновленных файлов моделей через tar.gz...", flush=True)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tf:
                for lf, arc_rel, _ in to_sync:
                    tf.add(str(lf), arcname=arc_rel)
            buf.seek(0)
            sftp.putfo(buf, "/tmp/prl_models_sync.tar.gz")
            ssh.exec_command("tar -xzf /tmp/prl_models_sync.tar.gz -C /opt/crypto_bot/models/prl && rm -f /tmp/prl_models_sync.tar.gz")[1].read()
        w_local = BASE_DIR / "models" / "prl" / "rl_execution" / "live_wallet_state.json"
        if push_wallet and w_local.exists():
            sftp.put(str(w_local), "/opt/crypto_bot/models/prl/rl_execution/live_wallet_state.json")
        sftp.close()
    except Exception as e:
        print(f"⚠️ SFTP sync warning: {e}")
    stop_live_daemon(cancel_open_orders=False)
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
            proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdin=subprocess.DEVNULL, stdout=lf, stderr=lf, start_new_session=True, close_fds=True)
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        print(f"🚀 Фоновый демон запущен (PID={proc.pid}). Лог: {log_file}")
    elif args.mode == "replay":
        run_replay_test(coin=args.coin, steps=args.steps)
    else:
        run_live_or_shadow_loop(coin=args.coin, mode=args.mode, steps=args.steps, order_usd=args.order_usd, budget_usd=args.budget_usd)
