import json
import os
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = Path(CURRENT_DIR).parent
SECRETS_FILE = os.path.join(CURRENT_DIR, "project_secrets.json")


def _load_dotenv_into_dict() -> dict:
    env_file = BASE_DIR / ".env"
    out = {}
    if env_file.exists():
        try:
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
        except OSError:
            pass
    return out


def _load_secrets() -> dict:
    merged = _load_dotenv_into_dict()
    if not os.path.exists(SECRETS_FILE):
        for mod_path in ("_tools.sync_secrets", "sync_secrets"):
            try:
                mod = __import__(mod_path, fromlist=["download_secrets_from_gdrive"])
                mod.download_secrets_from_gdrive()
                break
            except Exception:
                continue
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged.update(data)
        except Exception:
            pass
    return merged


def _cfg(key: str, default: str = "") -> str:
    val = os.environ.get(key)
    if val is not None and val != "":
        return val
    sec_val = _sec.get(key)
    if sec_val is not None and str(sec_val) != "":
        return str(sec_val)
    return default


_sec = _load_secrets()

VPS_HOST = _cfg("VPS_HOST", "")
VPS_PORT = int(_cfg("VPS_PORT", "2222"))
VPS_USER = _cfg("VPS_USER", "root")
VPS_PASS = _cfg("VPS_PASS", "")
DEFAULT_KEY = os.path.expanduser("~/.ssh/id_ed25519")
VPS_KEY_PATH = _cfg("VPS_KEY_PATH", DEFAULT_KEY if os.path.exists(DEFAULT_KEY) else "") or None
SAFETRADE_PROXY = _cfg("SAFETRADE_PROXY", "")
FALLBACK_PROXY = SAFETRADE_PROXY


def create_ssh_client(timeout: float = 30.0):
    """Создает SSHClient в рамках единого соединения (с защитой от MaxStartups/banner timeout)."""
    import time
    import paramiko

    if not VPS_HOST:
        raise RuntimeError(
            "VPS_HOST не задан! Укажите VPS_HOST в .env, переменных окружения или _tools/project_secrets.json."
        )
    last_err = None
    for attempt in range(3):
        client = paramiko.SSHClient()
        known_hosts = os.path.expanduser("~/.ssh/known_hosts")
        if os.path.exists(known_hosts):
            try:
                client.load_system_host_keys()
                client.load_host_keys(known_hosts)
            except Exception:
                pass
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            key_file = VPS_KEY_PATH if (VPS_KEY_PATH and os.path.exists(VPS_KEY_PATH)) else None
            client.connect(
                VPS_HOST,
                port=VPS_PORT,
                username=VPS_USER,
                password=VPS_PASS or None,
                key_filename=key_file,
                timeout=timeout,
                banner_timeout=45.0,
                auth_timeout=30.0,
                look_for_keys=False,
                allow_agent=False,
            )
            return client
        except Exception as e:
            last_err = e
            try:
                client.close()
            except Exception:
                pass
            time.sleep(2.5 * (attempt + 1))
    raise last_err
