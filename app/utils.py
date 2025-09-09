from pathlib import Path

DATA_DIR = Path("data/raw")

def save_file(filename: str, content: bytes) -> Path:
    path = DATA_DIR / filename
    with open(path, "wb") as f:
        f.write(content)
    return path
