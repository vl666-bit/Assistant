# app/db/chunks_db.py
import sqlite3
from typing import List, Dict, Optional, Iterable, Tuple, Any, Union
from config import SQLITE_DB_PATH
import numpy as np

DB_PATH = SQLITE_DB_PATH


# ===== Low-level =====

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 134217728")  # ~128MB
    return conn


# ===== Schema =====

def init_chunks_db():
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB,                 -- bytes(float32[])
            UNIQUE(page_id, chunk_index)
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page_idx ON chunks(page_id, chunk_index)")
        conn.commit()


# ===== Insert / Upsert =====

def _to_bytes32(vec: Union[bytes, bytearray, memoryview, np.ndarray, List[float], Tuple[float, ...], None]) -> bytes:
    if vec is None:
        return b""
    if isinstance(vec, (bytes, bytearray, memoryview)):
        return bytes(vec)
    if isinstance(vec, np.ndarray):
        return np.asarray(vec, dtype=np.float32).tobytes(order="C")
    # list/tuple
    return np.asarray(vec, dtype=np.float32).tobytes(order="C")


def insert_chunk(page_id: str, chunk_index: int, text: str, embedding: Optional[bytes]):
    """Upsert одного чанка."""
    emb_bytes = _to_bytes32(embedding)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO chunks (page_id, chunk_index, text, embedding)
            VALUES (?, ?, ?, ?)
        """, (page_id, int(chunk_index), text, emb_bytes))
        conn.commit()


def store_chunks(page_id: str, chunks: List[str], embeddings: Optional[Iterable[Any]] = None):
    """
    Батч-вставка: все чанки страницы за один транзакционный проход.
    embeddings — iterable той же длины (может быть None → пустые эмбеды).
    """
    if embeddings is None:
        embeddings = [None] * len(chunks)
    # безопасно запакуем заранее
    packed: List[Tuple[str, int, str, bytes]] = []
    for i, (txt, emb) in enumerate(zip(chunks, embeddings)):
        packed.append((page_id, i, txt, _to_bytes32(emb)))

    with _connect() as conn:
        cur = conn.cursor()
        cur.executemany("""
            INSERT OR REPLACE INTO chunks (page_id, chunk_index, text, embedding)
            VALUES (?, ?, ?, ?)
        """, packed)
        conn.commit()


# ===== Delete / Cleanup =====

def delete_chunks(page_id: str):
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM chunks WHERE page_id = ?", (page_id,))
        conn.commit()


def delete_chunks_many(page_ids: List[str]):
    if not page_ids:
        return
    q = ",".join(["?"] * len(page_ids))
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM chunks WHERE page_id IN ({q})", (*page_ids,))
        conn.commit()


def vacuum():
    with _connect() as conn:
        conn.execute("VACUUM")


# ===== Presence / Counts =====

def has_chunks(page_id: str) -> bool:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM chunks WHERE page_id = ? LIMIT 1", (page_id,))
        row = cur.fetchone()
        return row is not None


def get_page_ids_with_chunks(page_ids: List[str]) -> List[str]:
    if not page_ids:
        return []
    q_marks = ",".join(["?"] * len(page_ids))
    with _connect() as conn:
        cur = conn.cursor()
        sql = f"SELECT DISTINCT page_id FROM chunks WHERE page_id IN ({q_marks})"
        cur.execute(sql, (*page_ids,))
        rows = cur.fetchall()
        return [r["page_id"] for r in rows]


def count_chunks() -> int:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM chunks")
        return int(cur.fetchone()["c"])


# ===== Reads (text / embeddings) =====

def get_chunks_by_page(page_id: str) -> List[Dict]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT chunk_index, text
            FROM chunks
            WHERE page_id = ?
            ORDER BY chunk_index
        """, (page_id,))
        rows = cur.fetchall()
        return [{"chunk_index": r["chunk_index"], "text": r["text"]} for r in rows]


def get_chunks_with_embeddings(page_id: str) -> List[Dict]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT chunk_index, text, embedding
            FROM chunks
            WHERE page_id = ?
            ORDER BY chunk_index
        """, (page_id,))
        rows = cur.fetchall()
        return [{
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "embedding": r["embedding"],
        } for r in rows]


def get_all_chunks(limit: int = 100) -> List[Dict]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT page_id, chunk_index, text FROM chunks LIMIT ?", (int(limit),))
        rows = cur.fetchall()
        return [{
            "page_id": r["page_id"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
        } for r in rows]


def get_chunks_for_pages_with_embeddings(page_ids: List[str], limit_per_page: Optional[int] = None) -> List[Dict]:
    if not page_ids:
        return []
    q_marks = ",".join(["?"] * len(page_ids))
    with _connect() as conn:
        cur = conn.cursor()
        if limit_per_page and limit_per_page > 0:
            sql = f"""
                SELECT page_id, chunk_index, text, embedding
                FROM (
                    SELECT page_id, chunk_index, text, embedding,
                           ROW_NUMBER() OVER (PARTITION BY page_id ORDER BY chunk_index) AS rn
                    FROM chunks
                    WHERE page_id IN ({q_marks})
                )
                WHERE rn <= ?
                ORDER BY page_id, chunk_index
            """
            cur.execute(sql, (*page_ids, int(limit_per_page)))
        else:
            sql = f"""
                SELECT page_id, chunk_index, text, embedding
                FROM chunks
                WHERE page_id IN ({q_marks})
                ORDER BY page_id, chunk_index
            """
            cur.execute(sql, (*page_ids,))
        rows = cur.fetchall()
        return [{
            "page_id": r["page_id"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "embedding": r["embedding"],
        } for r in rows]


# ===== Helpers for in-memory retrieval (optional) =====

def rows_to_numpy(rows: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """
    Преобразует embedding BLOB → np.float32 массив; возвращает матрицу [N, D] и
    параллельно список метаданных (без преобразования текста).
    Пустые эмбеды (b"") будут отброшены.
    """
    vecs: List[np.ndarray] = []
    metas: List[Dict] = []
    for r in rows:
        emb: bytes = r.get("embedding") or b""
        if not emb:
            continue
        arr = np.frombuffer(emb, dtype=np.float32)
        if arr.size == 0:
            continue
        vecs.append(arr)
        metas.append(r)
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32), []
    # Проверим одинаковую размерность; если нет — отфильтруем «битые»
    dim = vecs[0].shape[0]
    good_vecs, good_metas = [], []
    for v, m in zip(vecs, metas):
        if v.shape[0] == dim:
            good_vecs.append(v)
            good_metas.append(m)
    if not good_vecs:
        return np.zeros((0, 0), dtype=np.float32), []
    mat = np.vstack(good_vecs).astype(np.float32, copy=False)
    return mat, good_metas


def health() -> Dict[str, str]:
    with _connect() as conn:
        conn.execute("SELECT 1")
    return {"db_path": DB_PATH, "status": "ok"}
