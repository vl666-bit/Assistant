# app/retrieval_outline.py
from typing import List, Tuple
import numpy as np
from app.db import pages_db
from app.embedding import get_embeddings


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def search_outline(query: str, top_nodes: int = 12, top_pages: int = 6, model_name: str = "default") -> List[str]:
    """
    1) берём все outline_title эмбеддинги (страницы + h1..h6),
    2) считаем близость к запросу,
    3) выбираем top_nodes узлов, из них собираем top_pages уникальных page_id.
    Если эмбеддингов нет — fallback LIKE по заголовкам страниц.
    """
    vec_q = np.asarray(get_embeddings([query])[0], dtype=np.float32)

    nodes = pages_db.get_all_outline_nodes() or []
    scored: List[Tuple[float, str]] = []  # (score, page_id)

    for n in nodes:
        emb = pages_db.get_embedding("outline_title", n["id"], model_name)
        if not emb or emb.get("vector") is None:
            continue
        v = np.asarray(emb["vector"], dtype=np.float32)
        score = _cos_sim(vec_q, v)
        page_id = n.get("page_id")
        if page_id:
            scored.append((score, page_id))

    if not scored:
        # fallback: просто поиск по title в pages
        like_rows = pages_db.find_pages_by_title_like(query, limit=top_pages)
        return [r["page_id"] for r in like_rows]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max(1, top_nodes)]

    seen = set()
    pages: List[str] = []
    for _, pid in top:
        if pid not in seen:
            pages.append(pid)
            seen.add(pid)
        if len(pages) >= max(1, top_pages):
            break

    return pages
