# config.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

SQLITE_DB_PATH = str((BASE_DIR / "app" / "db" / "rag.db").resolve())
CHROMA_DB_DIR  = str((BASE_DIR / "vector_store").resolve())

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Confluence ----
DOMAIN_ENV = os.getenv("CONF_DOMAIN", "https://atlassian.net").rstrip("/")

CONFLUENCE = {
    # нормализованный базовый URL для ссылок
    "base_url": os.getenv("CONF_BASE_URL", f"{DOMAIN_ENV}/wiki").rstrip("/"),

    # дублируем под разные регистры/старые вызовы (обратная совместимость)
    "domain": DOMAIN_ENV,
    "DOMAIN": DOMAIN_ENV,

    "email": os.getenv("CONF_EMAIL", "gmail.com"),
    "EMAIL": os.getenv("CONF_EMAIL", "gmail.com"),

    "api_token": os.getenv("CONF_TOKEN", ""),
    "API_TOKEN": os.getenv("CONF_TOKEN", ""),
}
