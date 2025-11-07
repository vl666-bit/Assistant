# app/db/pages_db.py
import sqlite3
from typing import List, Optional, Dict, Any
import re
import pickle
import pymorphy2
import numpy as np
from config import SQLITE_DB_PATH

DB_PATH = SQLITE_DB_PATH
morph = pymorphy2.MorphAnalyzer()


# ===================== Low-level =====================

def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 134217728")  # ~128MB
    return conn


# ===================== Schema init =====================

def init_pages_db():
    print(">>> init_pages_db вызван!")
    conn = _connect()
    cur = conn.cursor()

    # === проекты
    cur.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL
    )
    """)

    # === страницы (мета + контент по требованию)
    # добавляем колонку url для прямой ссылки на Confluence-страницу
    cur.execute("""
    CREATE TABLE IF NOT EXISTS pages (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        parent_id TEXT,
        title TEXT NOT NULL,
        last_modified TEXT,
        content TEXT,
        url TEXT,
        FOREIGN KEY (project_id) REFERENCES projects (id),
        FOREIGN KEY (parent_id) REFERENCES pages (id)
    )
    """)

    # миграция: если БД создана ранее без колонки url — добавим
    cur.execute("PRAGMA table_info(pages)")
    cols = [r["name"] for r in cur.fetchall()]
    if "url" not in cols:
        cur.execute("ALTER TABLE pages ADD COLUMN url TEXT")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_project ON pages(project_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_parent ON pages(parent_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_title ON pages(title)")

    # === Оглавление: узлы (page level=0 + h1..h6)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS outline_nodes (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        page_id TEXT NOT NULL,
        heading_id TEXT,
        title TEXT NOT NULL,
        level INTEGER NOT NULL,      -- 0 = страница, 1..6 = hN
        parent_id TEXT,
        path TEXT,
        updated_at INTEGER,
        FOREIGN KEY (project_id) REFERENCES projects(id),
        FOREIGN KEY (page_id) REFERENCES pages(id),
        FOREIGN KEY (parent_id) REFERENCES outline_nodes(id)
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_outline_project ON outline_nodes(project_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_outline_page ON outline_nodes(page_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_outline_parent ON outline_nodes(parent_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_outline_level ON outline_nodes(level)")

    # === Кеш эмбеддингов (универсальный)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embedding_cache (
        object_type TEXT NOT NULL,    -- 'outline_title' | 'page_chunk' | ...
        object_id TEXT NOT NULL,      -- outline_nodes.id или "<page_id>#<chunk_idx>"
        model_name TEXT NOT NULL,
        content_sha256 TEXT NOT NULL,
        vector BLOB NOT NULL,         -- pickle.dumps(np.ndarray | list[float])
        created_at INTEGER,
        PRIMARY KEY (object_type, object_id, model_name)
    )
    """)

    conn.commit()
    conn.close()


# ===================== Projects =====================

def insert_project(project_id: str, name: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO projects (id, name)
        VALUES (?, ?)
    """, (project_id, name))
    conn.commit()
    conn.close()


def get_project(project_id: str) -> Optional[Dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM projects WHERE id = ?", (project_id,))
    row = cur.fetchone()
    conn.close()
    return {"id": row["id"], "name": row["name"]} if row else None


def get_all_projects() -> List[Dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM projects")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r["id"], "name": r["name"]} for r in rows]


# ===================== Pages (CRUD) =====================

def update_parent_id(page_id: str, parent_id: Optional[str]) -> None:
    """Безопасно обновляет parent_id у уже вставленной страницы."""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE pages SET parent_id = ? WHERE id = ?", (parent_id, page_id))
        conn.commit()


def insert_page(page_id: str, project_id: str, title: str,
                parent_id: Optional[str] = None,
                last_modified: Optional[str] = None,
                content: Optional[str] = None,
                url: Optional[str] = None):
    """
    Базовая вставка страницы. Поддерживает url.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO pages (id, project_id, parent_id, title, last_modified, content, url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (page_id, project_id, parent_id, title, last_modified, content, url))
    conn.commit()
    conn.close()


def upsert_page_meta(page_id: str, title: str,
                     parent_id: Optional[str], space_id: str,
                     last_modified: Optional[str] = None,
                     url: Optional[str] = None):
    """Апсерт только меты (без контента), но с возможным url."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO pages (id, project_id, parent_id, title, last_modified, content, url)
        VALUES (?, ?, ?, ?, ?, NULL, ?)
        ON CONFLICT(id) DO UPDATE SET
            project_id=excluded.project_id,
            parent_id=excluded.parent_id,
            title=excluded.title,
            last_modified=excluded.last_modified,
            -- url обновляем только если пришёл новый (COALESCE)
            url=COALESCE(excluded.url, pages.url)
    """, (page_id, space_id, parent_id, title, last_modified, url))
    conn.commit()
    conn.close()


def upsert_page_content(page_id: str, html: Optional[str], text: Optional[str]):
    """
    Совместимость с пайпом: html приходит, но схема не хранит html.
    Сохраняем только text в поле content.
    """
    update_page_content(page_id, text or None)


def get_page(page_id: str) -> Optional[Dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, project_id, parent_id, title, last_modified, content, url
        FROM pages WHERE id = ?
    """, (page_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "parent_id": row["parent_id"],
        "title": row["title"],
        "last_modified": row["last_modified"],
        "content": row["content"],
        "url": row["url"]
    }


# ---- Удобные геттеры url ----

def get_page_by_id(page_id: str) -> Optional[Dict]:
    """Унифицированный вид под pack_sources: {'page_id','title','url'}"""
    row = get_page(page_id)
    if not row:
        return None
    return {"page_id": str(row["id"]), "title": row["title"], "url": row["url"]}


def get_pages_by_ids(page_ids: List[str]) -> Dict[str, Dict]:
    """
    Батч-достать мету по списку id.
    Возвращает словарь {page_id: {'page_id','title','url'}}.
    """
    if not page_ids:
        return {}
    ids = [str(x) for x in page_ids]
    placeholders = ",".join("?" for _ in ids)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, title, url FROM pages WHERE id IN ({placeholders})",
            ids
        )
        out: Dict[str, Dict] = {}
        for r in cur.fetchall():
            pid = str(r["id"])
            out[pid] = {"page_id": pid, "title": r["title"], "url": r["url"]}
        return out


def get_page_url(page_id: str) -> Optional[str]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT url FROM pages WHERE id = ? LIMIT 1", (page_id,))
        row = cur.fetchone()
        return (row["url"] or None) if row else None


# ---- остальной существующий код ----

def get_pages_by_project(project_id: str) -> List[Dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, title, parent_id FROM pages WHERE project_id = ?", (project_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r["id"], "title": r["title"], "parent_id": r["parent_id"]} for r in rows]


def list_pages_in_space(project_id: str) -> List[Dict]:
    """Синоним get_pages_by_project, но с явным именем под Confluence-«space»."""
    return get_pages_by_project(project_id)


def get_child_pages(parent_id: str) -> List[Dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM pages WHERE parent_id = ?", (parent_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r["id"], "title": r["title"]} for r in rows]


def list_children_ids(parent_id: str) -> List[str]:
    """Вспомогательно для BFS/веток."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id FROM pages WHERE parent_id = ?", (parent_id,))
    rows = cur.fetchall()
    conn.close()
    return [r["id"] for r in rows]


def get_all_pages() -> List[Dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id, project_id, parent_id, title FROM pages")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r["id"], "project_id": r["project_id"], "parent_id": r["parent_id"], "title": r["title"]} for r in rows]


def list_all_pages() -> List[Dict]:
    """То же, но с ожидаемыми ключами page_id/title (для совместимости некоторых модулей)."""
    pages = get_all_pages()
    return [{"page_id": p["id"], "title": p["title"], "parent_id": p["parent_id"], "space_id": p["project_id"]} for p in pages]


def has_content(page_id: str) -> bool:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT content FROM pages WHERE id = ?", (page_id,))
    row = cur.fetchone()
    conn.close()
    return bool(row and row["content"])


def has_page_content(page_id: str) -> bool:
    """Алиас под ожидаемое название в пайпе."""
    return has_content(page_id)


def list_pages_with_content(limit: int = 0) -> List[Dict]:
    conn = _connect()
    cur = conn.cursor()
    sql = "SELECT id, project_id, title, content FROM pages WHERE content IS NOT NULL"
    if limit and limit > 0:
        sql += " LIMIT ?"
        cur.execute(sql, (int(limit),))
    else:
        cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return [{"id": r["id"], "project_id": r["project_id"], "title": r["title"], "content": r["content"]} for r in rows]


def update_page_content(page_id: str, content: Optional[str]):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("UPDATE pages SET content = ? WHERE id = ?", (content, page_id))
    conn.commit()
    conn.close()


def count() -> int:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM pages")
        row = cur.fetchone()
        return int(row["c"]) if row else 0


def all_page_ids() -> List[str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT id FROM pages")
    rows = cur.fetchall()
    conn.close()
    return [str(r["id"]) for r in rows]


# ===================== Outline (tree) =====================

def upsert_outline_node(
    id: str,
    project_id: str,
    page_id: str,
    heading_id: Optional[str],
    title: str,
    level: int,
    parent_id: Optional[str],
    path: Optional[str],
    updated_at: Optional[int]
):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO outline_nodes (id, project_id, page_id, heading_id, title, level, parent_id, path, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            project_id=excluded.project_id,
            page_id=excluded.page_id,
            heading_id=excluded.heading_id,
            title=excluded.title,
            level=excluded.level,
            parent_id=excluded.parent_id,
            path=excluded.path,
            updated_at=excluded.updated_at
    """, (id, project_id, page_id, heading_id, title, level, parent_id, path, updated_at))
    conn.commit()
    conn.close()


def get_all_outline_nodes() -> List[Dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, project_id, page_id, heading_id, title, level, parent_id, path, updated_at
        FROM outline_nodes
    """)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_outline_path(node_id: str) -> Optional[str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT path FROM outline_nodes WHERE id = ?", (node_id,))
    row = cur.fetchone()
    conn.close()
    return row["path"] if row else None


def count_outline_nodes() -> int:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM outline_nodes")
    row = cur.fetchone()
    conn.close()
    return int(row["c"]) if row else 0


# ===================== Embedding cache =====================

def get_embedding(object_type: str, object_id: str, model_name: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT content_sha256, vector, created_at
        FROM embedding_cache
        WHERE object_type=? AND object_id=? AND model_name=?
    """, (object_type, object_id, model_name))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        vec = pickle.loads(row["vector"])
    except Exception:
        vec = None
    return {"content_sha256": row["content_sha256"], "vector": vec, "created_at": row["created_at"]}


def put_embedding(object_type: str, object_id: str, model_name: str,
                  content_sha256: str, vector: Any, created_at: Optional[int] = None):
    blob = sqlite3.Binary(pickle.dumps(vector))
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO embedding_cache (object_type, object_id, model_name, content_sha256, vector, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_type, object_id, model_name) DO UPDATE SET
            content_sha256=excluded.content_sha256,
            vector=excluded.vector,
            created_at=excluded.created_at
    """, (object_type, object_id, model_name, content_sha256, blob, created_at))
    conn.commit()
    conn.close()


# ---- Helpers for outline vectors ----

def has_any_outline_vecs(model_name: str = "default") -> bool:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT 1 FROM embedding_cache
        WHERE object_type='outline_title' AND model_name=? LIMIT 1
    """, (model_name,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def get_outline_vec(page_id: str, model_name: str = "default") -> Optional[np.ndarray]:
    """
    Берём embedding для уровня страницы (level=0) из embedding_cache.
    Если у страницы несколько узлов (в теории не должно), берём «самый новый».
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT ec.vector
        FROM embedding_cache ec
        JOIN outline_nodes on outline_nodes.id = ec.object_id
        WHERE ec.object_type='outline_title'
          AND ec.model_name=?
          AND outline_nodes.page_id=?
          AND outline_nodes.level=0
        ORDER BY ec.created_at DESC
        LIMIT 1
    """, (model_name, page_id))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        vec = pickle.loads(row["vector"])
        return np.asarray(vec, dtype=np.float32)
    except Exception:
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ===================== Search helpers =====================

def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize(text: str) -> str:
    words = normalize(text).split()
    norm_words = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(norm_words)


def find_pages_by_title_like(query: str, limit: int = 20) -> List[Dict]:
    """Простой LIKE-поиск по заголовку; возвращает [{page_id, title, parent_id, space_id}]."""
    if not query:
        return []
    q = f"%{normalize(query)}%"
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, parent_id, project_id
        FROM pages
        WHERE lower(replace(title,'—','-')) LIKE ?
        ORDER BY LENGTH(title) ASC
        LIMIT ?
    """, (q, int(limit)))
    rows = cur.fetchall()
    conn.close()
    return [{"page_id": r["id"], "title": r["title"], "parent_id": r["parent_id"], "space_id": r["project_id"]} for r in rows]


def find_candidates_by_title(query: str, limit: int = 20) -> List[Dict]:
    """
    Лемматизированный поиск + расширение дочерними.
    Возвращает [{"id": "...", "title": "..."}]
    """
    if not query:
        return []

    norm_q = lemmatize(query)
    words = norm_q.split()
    print(f"\n🔍 Поиск кандидатов: '{query}' → лемматизировано: '{norm_q}' (слов: {len(words)})")

    conn = _connect()
    cur = conn.cursor()
    like_clauses = " OR ".join([f"lower(replace(title,'—','-')) LIKE ?" for _ in words])
    params = [f"%{w}%" for w in words]

    sql = f"""
        SELECT id, title
        FROM pages
        WHERE {like_clauses}
        ORDER BY LENGTH(title) ASC
        LIMIT ?
    """
    cur.execute(sql, (*params, int(limit)))
    rows = cur.fetchall()
    conn.close()

    candidates = [{"id": str(r["id"]), "title": r["title"]} for r in rows]
    print(f"   → Найдено по заголовкам: {len(candidates)}")

    expanded = []
    seen = set()
    for c in candidates:
        if c["id"] not in seen:
            expanded.append(c); seen.add(c["id"])
            print(f"   ✅ Кандидат: {c['title']} (id={c['id']})")

        children = get_child_pages(c["id"])
        if children:
            print(f"      ↪ Добавляем дочерние для '{c['title']}': {len(children)} шт.")
        for ch in children:
            if ch["id"] not in seen:
                expanded.append(ch); seen.add(ch["id"])
                print(f"         • {ch['title']} (id={ch['id']})")

    print(f"   → Всего кандидатов (с дочерними): {len(expanded)}")
    return expanded
