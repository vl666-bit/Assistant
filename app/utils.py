# app/utils.py
import hashlib
import time
import uuid
import traceback
from pathlib import Path
from typing import Any, Dict

def now_ts() -> int:
    return int(time.time())

def uuid4_str() -> str:
    return str(uuid.uuid4())

def sha256_text(s: str) -> str:
    if s is None:
        s = ""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def format_exc(e: Exception) -> Dict[str, Any]:
    return {"error": str(e), "trace": traceback.format_exc(limit=6)}

def save_file(filename: str, content: bytes) -> Path:
    """
    Сохраняет бинарный файл в data/uploads и возвращает путь.
    """
    root = Path("data") / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / filename
    with open(out_path, "wb") as f:
        f.write(content)
    return out_path
