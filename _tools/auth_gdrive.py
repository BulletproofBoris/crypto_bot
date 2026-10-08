import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/drive.file']
CLIENT_SECRET_FILE = os.path.join(os.path.dirname(__file__), 'client_secret.json')
TOKEN_FILE = os.path.join(os.path.dirname(__file__), 'token.json')

def authenticate():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("🚀 Запускаем процесс авторизации...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
            
    print("✅ Успешная авторизация! Файл token.json сохранен и готов к работе на сервере.")

if __name__ == '__main__':
    authenticate()
