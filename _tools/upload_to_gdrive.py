import os
import sys
import argparse
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("HTTPLIB2_CA_CERTS", certifi.where())
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

FOLDER_ID = '1-qWy2q2nYoZtSCiMDZ1DiKZ5roBqXx-c'
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_drive_service(token_path):
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    return build('drive', 'v3', credentials=creds, cache_discovery=False)

def upload_file(service, file_path, folder_id):
    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    
    print(f"Загрузка {file_name} в Google Drive...")
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
    print(f"✅ Файл успешно загружен (доступ по ссылке открыт). File ID: {file_id}")
    return file_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Путь к файлу для загрузки')
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"❌ Ошибка: Файл {args.file_path} не найден.")
        sys.exit(1)

    token_path = os.path.join(os.path.dirname(__file__), 'token.json')
    if not os.path.exists(token_path):
        print(f"❌ Ошибка: Файл токена {token_path} не найден. Запустите auth_gdrive.py")
        sys.exit(1)

    try:
        service = get_drive_service(token_path)
        upload_file(service, args.file_path, FOLDER_ID)
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        sys.exit(1)
