import os
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_FILE = os.path.join(CURRENT_DIR, "project_secrets.json")

def _load_secrets():
    if not os.path.exists(SECRETS_FILE):
        try:
            from .sync_secrets import download_secrets_from_gdrive
            download_secrets_from_gdrive()
        except Exception:
            try:
                from sync_secrets import download_secrets_from_gdrive
                download_secrets_from_gdrive()
            except Exception:
                pass
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

_sec = _load_secrets()

VPS_HOST = os.environ.get("VPS_HOST", _sec.get("VPS_HOST", "81.19.137.18"))
VPS_PORT = int(os.environ.get("VPS_PORT", _sec.get("VPS_PORT", 2222)))
VPS_USER = os.environ.get("VPS_USER", _sec.get("VPS_USER", "root"))
VPS_PASS = os.environ.get("VPS_PASS", _sec.get("VPS_PASS", ""))
DEFAULT_KEY = os.path.expanduser("~/.ssh/id_ed25519")
VPS_KEY_PATH = os.environ.get("VPS_KEY_PATH", _sec.get("VPS_KEY_PATH", DEFAULT_KEY if os.path.exists(DEFAULT_KEY) else None))
SAFETRADE_PROXY = os.environ.get("SAFETRADE_PROXY", _sec.get("SAFETRADE_PROXY", "http://edtilhmt:iblj7uuixsmy@31.59.20.176:6754"))
FALLBACK_PROXY = SAFETRADE_PROXY

def create_ssh_client(timeout=30.0):
    """Создает SSHClient в рамках единого соединения (с защитой от MaxStartups/banner timeout)."""
    import time
    import paramiko
    last_err = None
    for attempt in range(3):
        client = paramiko.SSHClient()
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
