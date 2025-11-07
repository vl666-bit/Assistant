# app/retrieval.py
import numpy as np
from typing import List, Dict, Any, Optional, Iterable, Union
from chromadb import PersistentClient
from config import CHROMA_DB_DIR

# === Chroma client / collection ===
_client = PersistentClient(path=CHROMA_DB_DIR)
_collection = _client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}  # cosine similarity
)

# Что допускает include в новых версиях Chroma
VALID_INCLUDE = {"embeddings", "documents", "metadatas", "uris", "data", "distances"}

def _normalize_include(include=None, with_distances=True):
    base = ["documents", "metadatas"]
    if with_distances:
        base.append("distances")
    if include:
        base.extend([x for x in include if x in VALID_INCLUDE])
    # дедуп
    seen = []
    for x in base:
        if x not in seen:
            seen.append(x)
    return seen

# ===== ndarray helpers =====
def _as_f32_array2d(x: Union[np.ndarray, List[List[float]], List[np.ndarray]]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x.astype(np.float32, copy=False)
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _as_f32_vec(x: Union[np.ndarray, List[float]]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False).reshape(-1)
    return np.asarray(x, dtype=np.float32).reshape(-1)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _as_f32_vec(a); b = _as_f32_vec(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _ensure_nonempty(iterable: Optional[Iterable[str]]) -> Optional[List[str]]:
    if iterable is None:
        return None
    lst = list(iterable)
    return lst if lst else []

# ===== upsert / delete =====
def store_embeddings(
    page_id: str,
    texts: List[str],
    embeddings: Union[np.ndarray, List[List[float]], List[np.ndarray]],
    extra_metadatas: Optional[List[Dict[str, Any]]] = None
) -> None:
    embs = _as_f32_array2d(embeddings)
    if len(texts) != embs.shape[0]:
        raise ValueError(f"len(texts) ({len(texts)}) != embeddings.shape[0] ({embs.shape[0]})")

    ids = [f"{page_id}_{i}" for i in range(len(texts))]
    metadatas: List[Dict[str, Any]] = []
    for i in range(len(texts)):
        base = {"page_id": page_id, "chunk_index": i}
        if extra_metadatas and i < len(extra_metadatas) and extra_metadatas[i]:
            base.update(extra_metadatas[i])
        metadatas.append(base)

    _collection.upsert(
        documents=texts,
        embeddings=embs.tolist(),
        ids=ids,
        metadatas=metadatas,
    )

def delete_by_page(page_id: str) -> int:
    _collection.delete(where={"page_id": page_id})
    return -1

def delete_by_pages(page_ids: Iterable[str]) -> int:
    ids = _ensure_nonempty(page_ids)
    if ids == []:
        return 0
    _collection.delete(where={"page_id": {"$in": ids}})
    return -1

# ===== query (единая реализация) =====
def query_similar(
    query_embedding: Union[np.ndarray, List[float]],
    top_k: int = 3,
    page_ids: Optional[Iterable[str]] = None,
    include: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Возвращает список: [{document, id, page_id, chunk_index, distance, similarity}]
    """
    q = _as_f32_vec(query_embedding)
    top_k = max(1, int(top_k))

    where = None
    if page_ids is not None:
        pi = list(page_ids)
        if len(pi) == 0:
            return []
        where = {"page_id": {"$in": pi}}

    include = _normalize_include(include, with_distances=True)

    results = _collection.query(
        query_embeddings=[q.tolist()],
        n_results=top_k,
        where=where,
        include=include,
    )

    # ids Chroma возвращает всегда, даже если не просили в include
    docs      = (results.get("documents") or [[]])[0]
    ids       = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    metas     = (results.get("metadatas") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for doc, _id, dist, meta in zip(docs, ids, distances, metas):
        similarity = None
        if dist is not None:
            similarity = 1.0 - float(dist)  # для cosine
        meta = meta or {}
        out.append({
            "document": doc,
            "id": _id,
            "page_id": meta.get("page_id"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
            "similarity": similarity,
        })

    out.sort(key=lambda x: (x["similarity"] if x["similarity"] is not None else -1e9), reverse=True)
    return out

def query_similar_in_pages(
    query_embedding: Union[np.ndarray, List[float]],
    page_ids: Iterable[str],
    top_k: int = 3,
    include: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    return query_similar(
        query_embedding=query_embedding,
        top_k=top_k,
        page_ids=page_ids,
        include=include,
    )

# ===== локальный реранкер =====
def rerank_vectors(
    query: Union[np.ndarray, List[float]],
    items: List[Dict[str, Any]],
    *,
    vec_key: str = "vec",
    text_key: str = "text",
    top_k: int = 10
) -> List[Dict[str, Any]]:
    qv = _as_f32_vec(query)
    ranked: List[Dict[str, Any]] = []
    for it in items:
        v = it.get(vec_key, None)
        if v is None:
            continue
        v = _as_f32_vec(v)
        score = _cosine(qv, v)
        ranked.append({**it, "score": float(score)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:max(1, int(top_k))]

# ===== service =====
def count_vectors() -> int:
    sample = _collection.peek()
    return len(sample["ids"]) if sample and "ids" in sample else -1

def health() -> Dict[str, Any]:
    try:
        test = _collection.query(query_embeddings=[[0.0, 0.0]], n_results=1)
        ok = True
    except Exception as e:
        ok = False
        test = {"error": str(e)}
    return {"dir": CHROMA_DB_DIR, "ok": ok, "info": test}
