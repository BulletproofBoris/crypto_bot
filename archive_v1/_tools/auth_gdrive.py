import os
import sys
import webbrowser
import subprocess
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Нормализуем прокси: 127.0.0.1:2080 работает как socks5h, а не http
for _var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    _val = os.environ.get(_var, '')
    if 'http://127.0.0.1:2080' in _val:
        os.environ[_var] = 'socks5h://127.0.0.1:2080'
if not os.environ.get('ALL_PROXY'):
    os.environ['ALL_PROXY'] = 'socks5h://127.0.0.1:2080'

# В среде WSL регистрируем браузер Windows для автоматического открытия ссылки
if os.path.exists("/mnt/c/WINDOWS/system32/cmd.exe"):
    class WindowsDefaultBrowser(webbrowser.BaseBrowser):
        def open(self, url, new=0, autoraise=True):
            try:
                subprocess.Popen(["/mnt/c/WINDOWS/system32/cmd.exe", "/c", "start", "", url.replace("&", "^&")])
                return True
            except Exception:
                return False
    webbrowser.register("windows-default", WindowsDefaultBrowser, preferred=True)

SCOPES = ['https://www.googleapis.com/auth/drive.file']
CLIENT_SECRET_FILE = os.path.join(os.path.dirname(__file__), 'client_secret.json')
TOKEN_FILE = os.path.join(os.path.dirname(__file__), 'token.json')

def authenticate(force_new=False):
    creds = None
    if os.path.exists(TOKEN_FILE) and not force_new:
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            creds = None
    
    if not creds or not creds.valid:
        refreshed = False
        if creds and creds.expired and creds.refresh_token and not force_new:
            try:
                print("🔄 Попытка обновления существующего токена...")
                creds.refresh(Request())
                refreshed = True
            except Exception as e:
                print(f"⚠️ Токен просрочен или отозван ({e}).")
                print("🌐 Запускаем повторный вход через браузер...")
        
        if not refreshed:
            print("🚀 Запускаем процесс авторизации через браузер...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
            
    print("✅ Успешная авторизация! Файл token.json сохранен и готов к работе на сервере.")

if __name__ == '__main__':
    import sys
    force = "--force" in sys.argv
    authenticate(force_new=force)
