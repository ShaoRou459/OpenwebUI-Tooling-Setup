import os
import requests
from pathlib import Path

TOKEN = "openwebui api token here"
BASE_URL = "openwebui base url"

def upload_file(file_path, token=TOKEN, base_url=BASE_URL):
    """
    Upload a file to the OpenWebUI /api/v1/files endpoint.
    Returns the public URL you can embed or share.
    """
    url = f"{base_url.rstrip('/')}/api/v1/files/"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    with open(file_path, "rb") as f:
        files = {"file": (Path(file_path).name, f)}
        r = requests.post(url, headers=headers, files=files, timeout=30)
    r.raise_for_status()
    data = r.json()
    return f"{base_url}/api/v1/files/{data['id']}/content"
