#!/usr/bin/env python3
"""
Автоматический модуль перевода накопленных монет Pearl (PRL) с майнинг-кошелька на SafeTrade.
Периодичность: раз в 3 часа (или через cron/systemd).
"""

import os
import sys
import time
import json
import logging
import hashlib
import re
from pathlib import Path
from curl_cffi import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("pearl_topup")

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def point_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    (x1, y1), (x2, y2) = p1, p2
    if x1 == x2 and y1 != y2: return None
    lam = ((3 * x1 * x1) * pow(2 * y1, P - 2, P) if x1 == x2 else (y2 - y1) * pow(x2 - x1, P - 2, P)) % P
    x3 = (lam * lam - x1 - x2) % P
    return (x3, (lam * (x1 - x3) - y1) % P)

def point_mul(k, pt):
    res, add = None, pt
    while k:
        if k & 1: res = point_add(res, add)
        add = point_add(add, add)
        k >>= 1
    return res

def tagged_hash(tag: str, msg: bytes) -> bytes:
    th = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(th + th + msg).digest()

def schnorr_sign(msg32: bytes, seckey_int: int) -> tuple[str, str]:
    Q_pt = point_mul(seckey_int, (Gx, Gy))
    d0 = seckey_int if Q_pt[1] % 2 == 0 else N - seckey_int
    pk_bytes = Q_pt[0].to_bytes(32, "big")
    aux = bytes(32)
    t_bytes = (d0 ^ int.from_bytes(tagged_hash("BIP0340/aux", aux), "big")).to_bytes(32, "big")
    k0 = int.from_bytes(tagged_hash("BIP0340/nonce", t_bytes + pk_bytes + msg32), "big") % N
    R = point_mul(k0, (Gx, Gy))
    k = k0 if R[1] % 2 == 0 else N - k0
    r_bytes = R[0].to_bytes(32, "big")
    e = int.from_bytes(tagged_hash("BIP0340/challenge", r_bytes + pk_bytes + msg32), "big") % N
    s = (k + e * d0) % N
    return r_bytes.hex(), s.to_bytes(32, "big").hex()

class PearlAutoTopup:
    PROJ_ID = "P30rTUyDempsUkhVeeuyZhIehE5D"
    ACT_BUILD_TX = "7fecea36e1aaab66fda670fd2a5f34001ea4ddd93b"
    ACT_SEND_TX = "7f29b39adece436db22533ed2a274bb1e71477da8b"
    ACT_MAX_SPENDABLE = "7f49048cc5b5afc424ea1e405dbc8affd03b564e48"

    def __init__(self):
        self._load_config()
        self.sess = requests.Session(impersonate="chrome120")
        self._derive_keys()
        self._actions_discovered = False

    def _load_config(self):
        env_file = BASE_DIR / ".env"
        cfg = {}
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")

        self.wif = cfg.get("pearl_wallet_recovery_key")
        self.dsr = cfg.get("PEARL_DESCOPE_DSR")
        self.ds = cfg.get("PEARL_DESCOPE_DS")
        self.src_addr = cfg.get("PEARL_MINING_ADDRESS", "prl1pkxttscqu5rker42j8qfvt2806y4eqyu83pg8zpgf76xmg74en66qp5d9x8")
        self.dst_addr = cfg.get("SAFETRADE_PRL_DEPOSIT_ADDRESS", "prl1peuk44egx0ggc82wdl0pc9e3jfnnknfaskvjh7yre6lqhmxtxanwsfuhd7a")

        if not self.wif or not self.dsr:
            raise ValueError("Не найден pearl_wallet_recovery_key или PEARL_DESCOPE_DSR в .env")

    def _derive_keys(self):
        ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        n = 0
        for c in self.wif:
            n = n * 58 + ALPHABET.index(c)
        raw = n.to_bytes(38, "big")
        self.priv_int = int.from_bytes(raw[1:-5], "big")
        pub = point_mul(self.priv_int, (Gx, Gy))
        self.pub_hex = ("02" if pub[1] % 2 == 0 else "03") + pub[0].to_bytes(32, "big").hex()

        d_internal = self.priv_int if pub[1] % 2 == 0 else N - self.priv_int
        tweak = int.from_bytes(tagged_hash("TapTweak", pub[0].to_bytes(32, "big")), "big")
        self.d_tweaked = (d_internal + tweak) % N

    def _refresh_ds(self) -> str:
        try:
            r = self.sess.post(
                "https://auth.compute.pearlresearch.ai/v1/auth/refresh",
                headers={"Authorization": f"Bearer {self.PROJ_ID}:{self.dsr}", "Cookie": f"DSR={self.dsr}", "Content-Type": "application/json"},
                data="{}", timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                new_ds = data.get("sessionJwt")
                new_dsr = data.get("refreshJwt")
                if new_ds:
                    self.ds = new_ds
                if new_dsr:
                    self.dsr = new_dsr
                    self._save_env_tokens(new_ds, new_dsr)
                return self.ds
        except Exception as e:
            logger.warning(f"Не удалось освежить токен через /v1/auth/refresh: {e}")
        return self.ds

    def _save_env_tokens(self, new_ds: str, new_dsr: str):
        try:
            env_file = BASE_DIR / ".env"
            lines = env_file.read_text(encoding="utf-8").splitlines()
            out = []
            for l in lines:
                if l.startswith("PEARL_DESCOPE_DS="):
                    out.append(f"PEARL_DESCOPE_DS={new_ds}")
                elif l.startswith("PEARL_DESCOPE_DSR="):
                    out.append(f"PEARL_DESCOPE_DSR={new_dsr}")
                else:
                    out.append(l)
            env_file.write_text("\n".join(out) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _discover_actions(self, ds: str):
        if self._actions_discovered:
            return
        self._actions_discovered = True
        try:
            r = self.sess.get(
                "https://compute.pearlresearch.ai/wallet",
                headers={"Cookie": f"DS={ds}; DSR={self.dsr}"},
                timeout=15,
            )
            scripts = re.findall(r'src="(/_next/static/[^"]+\.js)"', r.text)
            for sc in scripts:
                jr = self.sess.get("https://compute.pearlresearch.ai" + sc, timeout=15)
                for m in re.finditer(
                    r'createServerReference\)\("([0-9a-f]+)"[^\)]*"(fetchWalletMaxSpendable|buildTransaction|sendSignedTransaction)"',
                    jr.text,
                ):
                    act_id, act_name = m.group(1), m.group(2)
                    if act_name == "fetchWalletMaxSpendable":
                        self.ACT_MAX_SPENDABLE = act_id
                    elif act_name == "buildTransaction":
                        self.ACT_BUILD_TX = act_id
                    elif act_name == "sendSignedTransaction":
                        self.ACT_SEND_TX = act_id
            logger.info(
                f"Автообнаружение Next-Action IDs: maxSpendable={self.ACT_MAX_SPENDABLE[:10]}..., "
                f"buildTx={self.ACT_BUILD_TX[:10]}..., sendTx={self.ACT_SEND_TX[:10]}..."
            )
        except Exception as e:
            logger.warning(f"Ошибка автообнаружения Next-Action IDs: {e}")

    def _call_action(self, action_attr: str, payload: list):
        ds = self._refresh_ds()
        action_id = getattr(self, action_attr, action_attr)
        r = self.sess.post(
            "https://compute.pearlresearch.ai/wallet",
            headers={
                "Accept": "text/x-component",
                "Content-Type": "text/plain;charset=UTF-8",
                "Next-Action": action_id,
                "Cookie": f"DS={ds}; DSR={self.dsr}",
            },
            data=json.dumps(payload),
            timeout=20,
        )
        if r.status_code == 404 and not self._actions_discovered:
            logger.info("Next-Action ID устарел после редеплоя сайта, запускаю автообнаружение...")
            self._discover_actions(ds)
            action_id = getattr(self, action_attr, action_attr)
            r = self.sess.post(
                "https://compute.pearlresearch.ai/wallet",
                headers={
                    "Accept": "text/x-component",
                    "Content-Type": "text/plain;charset=UTF-8",
                    "Next-Action": action_id,
                    "Cookie": f"DS={ds}; DSR={self.dsr}",
                },
                data=json.dumps(payload),
                timeout=20,
            )
        if r.status_code != 200:
            raise RuntimeError(f"Action HTTP {r.status_code}: {r.text[:200]}")
        
        text = r.text
        json_obj = {}
        # 1. Поиск основного JSON словаря
        m_json = re.search(r"1:(\{.*?\})\s*$", text, re.DOTALL)
        if not m_json:
            m_json = re.search(r"1:(\{.*?\})", text)
        if m_json:
            try:
                json_obj = json.loads(m_json.group(1))
            except Exception:
                pass

        # 2. Если psbtBase64 указывает на $2, извлекаем chunk 2 (формат "2:T<hex_len>,<payload>1:{...")
        ref = str(json_obj.get("psbtBase64", ""))
        if ref.startswith("$"):
            chunk_num = ref.lstrip("$")
            patt = f"{chunk_num}:T"
            if patt in text:
                s_idx = text.find(patt)
                comma_idx = text.find(",", s_idx)
                # конец чанка перед следующим 1:{
                end_idx = text.find("1:{", comma_idx)
                if end_idx == -1:
                    end_idx = len(text)
                psbt_str = text[comma_idx+1:end_idx].strip()
                json_obj["psbtBase64"] = psbt_str

        return json_obj

    def get_max_spendable(self) -> float:
        res = self._call_action("ACT_MAX_SPENDABLE", [self.src_addr])
        return float(res.get("maxSpendable", 0.0))

    def transfer(self, amount: float = None) -> dict:
        max_sp = self.get_max_spendable()
        logger.info(f"Доступно к отправке: {max_sp:.8f} PRL")

        if max_sp < 1.0:
            logger.info("Баланс меньше 1.0 PRL. Пропуск перевода.")
            return {"status": "skipped", "reason": "low_balance", "balance": max_sp}

        send_amt = max_sp if amount is None else min(amount, max_sp)
        logger.info(f"Подготовка транзакции: {send_amt:.8f} PRL -> {self.dst_addr}")

        build_res = self._call_action("ACT_BUILD_TX", [{
            "senderAddress": self.src_addr,
            "senderPublicKey": self.pub_hex,
            "recipientAddress": self.dst_addr,
            "amount": send_amt,
        }])

        psbt_b64 = build_res.get("psbtBase64")
        sighashes = build_res.get("sighashes", [])
        if not psbt_b64 or not sighashes:
            raise RuntimeError(f"Ошибка построения PSBT: {build_res}")

        logger.info(f"Подписание {len(sighashes)} входов (BIP-340 Schnorr)...")
        signatures = []
        for sh in sighashes:
            msg_bytes = bytes.fromhex(sh)
            r_h, s_h = schnorr_sign(msg_bytes, self.d_tweaked)
            signatures.append({"r": r_h, "s": s_h})

        logger.info("Отправка подписанной транзакции в сеть...")
        send_res = self._call_action("ACT_SEND_TX", [{
            "psbtBase64": psbt_b64,
            "signatures": signatures,
        }])

        txid = send_res.get("txid")
        logger.info(f"Успешно отправлено! TXID: {txid}")
        return {"status": "success", "txid": txid, "amount": send_amt}

def main():
    bot = PearlAutoTopup()
    if "--keepalive" in sys.argv:
        bot._refresh_ds()
        return
    if "--loop" in sys.argv:
        logger.info("Запуск демона автопополнения (цикл раз в 3 часа)...")
        while True:
            try:
                bot.transfer()
            except Exception as e:
                logger.error(f"Ошибка при автопополнении: {e}")
            time.sleep(3 * 3600)
    else:
        res = bot.transfer()
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
