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
SAFETRADE_PROXY = os.environ.get("SAFETRADE_PROXY", _sec.get("SAFETRADE_PROXY", "http://edtilhmt:iblj7uuixsmy@31.59.20.176:6754"))
FALLBACK_PROXY = SAFETRADE_PROXY
