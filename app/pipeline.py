from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from app.utils import save_file  # у тебя уже есть utils.py — используем его
from app.embedding import get_embeddings
from app.retrieval import (
    store_embeddings,
    query_similar,
    query_similar_in_pages,
)
from app.retrieval_outline import search_outline
from app.llm_stub import generate_answer
from app.db import pages_db

# ===== Чанкование =====
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Делит длинный текст на чанки фиксированной длины с перекрытием.
    """
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

# ===== Индексация локального файла =====
def process_file(filename: str, content: bytes) -> str:
    """
    Сохраняет файл, режет на чанки, считает эмбеддинги и кладёт в векторку.
    page_id = имя файла без расширения.
    """
    path: Path = save_file(filename, content)
    try:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        raw_text = ""

    texts = chunk_text(raw_text)
    if not texts:
        return f"Файл {filename} сохранён, но текст не найден (0 чанков)."

    embeddings = get_embeddings(texts)            # List[List[float]]
    vectors = np.asarray(embeddings, dtype=np.float32)
    page_id = Path(filename).stem

    store_embeddings(page_id, texts, vectors)
    return f"Файл {filename} обработан и проиндексирован ({len(texts)} чанков)."

# ===== Совместимость по score/similarity =====
def _best_score(d: Dict[str, Any]) -> Optional[float]:
    if d.get("score") is not None:
        return float(d["score"])
    if d.get("similarity") is not None:
        return float(d["similarity"])
    return None

# ===== Старый путь: общий ANN по всей базе =====
def answer_query(query: str, page_ids: Optional[List[str]] = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Возвращает: {"answer": str, "sources": list[dict], "chunks": list[dict]}
    """
    print(f"\n🟢 answer_query: query='{query}', page_ids={page_ids}, top_k={top_k}")

    qv = np.asarray(get_embeddings([query])[0], dtype=np.float32)
    hits = query_similar(qv, top_k=top_k, page_ids=page_ids)

    if not hits:
        return {"answer": "Информация не найдена.", "sources": [], "chunks": []}

    # контекст из чанков
    context = "\n\n---\n\n".join([h["document"] for h in hits if h.get("document")])

    # источники
    sources, seen = [], set()
    for h in hits:
        pid = h.get("page_id")
        if pid and pid not in seen:
            page = pages_db.get_page(pid)
            sources.append({"page_id": pid, "title": (page["title"] if page else None)})
            seen.add(pid)

    # генерация — промпт внутри llm_stub
    answer = generate_answer(query, context)
    return {"answer": answer, "sources": sources, "chunks": hits}

# ===== Новый путь: outline → ограниченные страницы → ANN по ним =====
def _format_short_history(history: Optional[List[Dict[str, str]]], limit_pairs: int = 4) -> str:
    """
    Опциональный короткий контекст диалога:
    history: [{"role": "user"|"assistant", "text": "..."}]
    Берём до 4 последних пар (макс 8 сообщений).
    """
    if not history:
        return ""
    last = history[-(limit_pairs * 2):]
    return "\n".join(f"{m.get('role')}: {m.get('text')}" for m in last if m.get("text"))

def answer_query_via_outline(
    query: str,
    *,
    top_nodes: int = 12,
    top_pages: int = 6,
    top_k: int = 12,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Поиск по оглавлению -> ограничение по страницам -> поиск чанков внутри них.
    Требует заполненный outline (OutlinePipeline.build_outline()).
    """
    print(f"\n🟢 answer_query_via_outline: '{query}' | top_nodes={top_nodes}, top_pages={top_pages}, top_k={top_k}")

    # 1) сузить по оглавлению
    candidate_pages: List[str] = search_outline(query, top_nodes=top_nodes, top_pages=top_pages) or []
    candidate_pages = [pid for pid in dict.fromkeys(candidate_pages) if pid]
    print(f"   → страницы-кандидаты: {candidate_pages}")

    if not candidate_pages:
        return {"answer": "Информация не найдена (по оглавлению).", "sources": [], "chunks": []}

    # 2) поиск по выбранным страницам
    qv = np.asarray(get_embeddings([query])[0], dtype=np.float32)
    hits = query_similar_in_pages(qv, page_ids=candidate_pages, top_k=top_k) or []
    if not hits:
        return {"answer": "Информация не найдена (после сужения по оглавлению).", "sources": [], "chunks": []}

    # 3) контекст (история + фрагменты)
    short_ctx = _format_short_history(chat_history, limit_pairs=4)
    chunks_ctx = "\n\n---\n\n".join([h["document"] for h in hits if h.get("document")])
    merged_context = (f"Последние сообщения диалога:\n{short_ctx}\n\n" if short_ctx else "") + \
                     f"Релевантные фрагменты документации:\n{chunks_ctx}"

    # 4) источники
    sources, seen = [], set()
    for h in hits:
        pid = h.get("page_id")
        if pid and pid not in seen:
            page = pages_db.get_page(pid)
            sources.append({"page_id": pid, "title": (page["title"] if page else None)})
            seen.add(pid)

    # 5) генерация
    answer = generate_answer(query, merged_context)
    return {"answer": answer, "sources": sources, "chunks": hits}
