# app/confluence_pipeline.py
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import deque

from app.connectors.confluence_connector import ConfluenceConnector
from app.embedding import get_embeddings
from app.retrieval import store_embeddings, query_similar_in_pages
from app.llm_stub import generate_answer
from app.pipeline import chunk_text
from app.db import pages_db, chunks_db
from app.retrieval_outline import search_outline
from app.utils import now_ts, uuid4_str, sha256_text
from config import CONFLUENCE


# --- helpers: Confluence URLs ---

def _get_confluence_base_url() -> str:
    """
    Достаём базовый URL из конфига, пробуя разные ключи и не падая.
    Для Atlassian Cloud автоматически добавляем /wiki, если его нет.
    """
    cfg = CONFLUENCE or {}
    base = (
        cfg.get("base_url")
        or cfg.get("BASE_URL")
        or cfg.get("url")
        or cfg.get("URL")
        or cfg.get("domain")
        or cfg.get("DOMAIN")
        or ""
    )
    base = (base or "").rstrip("/")
    if "atlassian.net" in base and not base.endswith("/wiki"):
        base = base + "/wiki"
    return base


def confluence_page_url(page_id: str) -> Optional[str]:
    base = _get_confluence_base_url()
    if not base:
        return None
    return f"{base}/pages/viewpage.action?pageId={page_id}"


# ========= ВСПОМОГАТЕЛЬНОЕ =========

def _ins_page_meta(
    *,
    page_id: str,
    project_id: str,
    title: str,
    parent_id: Optional[str],
    version: Optional[str] = None,
    last_modified: Optional[str] = None,
) -> None:
    """
    На этапе инициализации НЕ ставим parent_id (NULL), чтобы не ловить FK-ошибки.
    parent_id проставим вторым проходом после вставки всех страниц.
    """
    pages_db.insert_page(
        page_id=page_id,
        project_id=project_id,
        title=title,
        parent_id=None,
        last_modified=(last_modified or (version or "")),
        content=None,
        url=confluence_page_url(page_id),  # ← сохраняем ссылку сразу
    )


# ========= ОСНОВНОЙ ПАЙП =========

class ConfluencePipeline:
    def __init__(self, domain: str, email: str, api_token: str):
        self.connector = ConfluenceConnector(domain, email, api_token)

    # ===== INIT: подтянуть ВСЕ ветви и метаданные, без контента (2 фазы для FK) =====
    def init_structure(self, per_space_limit: int = 5000) -> Dict[str, Any]:
        """
        На старте:
        1) Получаем все spaces.
        2) В память собираем список страниц + пары (child -> parent).
           Сначала пробуем get_pages_meta; если пусто — BFS по get_pages_in_space + get_child_pages.
        3) Фаза 1 — вставляем ВСЕ страницы БЕЗ parent_id (NULL), но с готовым url.
        4) Фаза 2 — отдельным проходом выставляем parent_id тем, у кого родитель есть среди вставленных.
        """
        spaces = self.connector.get_spaces()
        print(f"🔄 Найдено пространств: {len(spaces)}")

        total_inserted = 0
        total_parents_set = 0
        spaces_stats: List[Dict[str, Any]] = []

        for sp in spaces:
            sid = sp["id"]
            sname = sp.get("name") or sp.get("key") or sid
            pages_db.insert_project(sid, sname)

            # --- собрать meta в память ---
            all_pages: List[Dict[str, Any]] = []              # [{"id","title"}]
            parent_links: List[Tuple[str, Optional[str]]] = [] # [(child_id, parent_id)]
            meta_pages: List[Dict[str, Any]] = []

            try:
                meta_pages = self.connector.get_pages_meta(sid, limit=per_space_limit) or []
            except Exception as e:
                print(f"⚠️ get_pages_meta failed for space {sid}: {e}")

            if meta_pages:
                for p in meta_pages:
                    pid = p.get("id")
                    if not pid:
                        continue
                    all_pages.append({"id": pid, "title": p.get("title") or pid})
                    parent_links.append((pid, p.get("parent_id")))
            else:
                # --- fallback: BFS по пространству ---
                print(f"↪️ Fallback BFS для пространства {sid}")
                roots = self.connector.get_pages_in_space(sid, limit=200) or []
                seen = set()
                q = deque()

                for r in roots:
                    pid = r.get("id")
                    if not pid or pid in seen:
                        continue
                    seen.add(pid)
                    all_pages.append({"id": pid, "title": r.get("title") or pid})
                    parent_links.append((pid, None))
                    q.append(pid)

                while q:
                    cur = q.popleft()
                    try:
                        children = self.connector.get_child_pages(cur) or []
                    except Exception as e:
                        print(f"⚠️ get_child_pages failed for page {cur}: {e}")
                        children = []
                    for ch in children:
                        cid = ch.get("id")
                        if not cid or cid in seen:
                            continue
                        seen.add(cid)
                        all_pages.append({"id": cid, "title": ch.get("title") or cid})
                        parent_links.append((cid, cur))
                        q.append(cid)

            # --- Фаза 1: вставка всех страниц без parent_id, но с url ---
            inserted_here = 0
            for p in all_pages:
                pid = p["id"]
                pages_db.insert_page(
                    page_id=pid,
                    project_id=sid,
                    title=p.get("title") or pid,
                    parent_id=None,
                    last_modified=None,
                    content=None,
                    url=confluence_page_url(pid),  # ← сохраняем ссылку
                )
                inserted_here += 1

            # --- Фаза 2: обновление parent_id (только если родитель есть среди вставленных) ---
            ids_in_space = {p["id"] for p in all_pages}
            parents_set = 0
            for child_id, parent_id in parent_links:
                if parent_id and parent_id in ids_in_space:
                    pages_db.update_parent_id(child_id, parent_id)
                    parents_set += 1

            total_inserted += inserted_here
            total_parents_set += parents_set
            spaces_stats.append({
                "space_id": sid,
                "inserted": inserted_here,
                "parents_set": parents_set
            })

        print(f"✅ Всего страниц (мета) загружено в БД: {total_inserted}; parent_id проставлено: {total_parents_set}")
        return {
            "spaces": len(spaces),
            "pages_meta_inserted": total_inserted,
            "parents_set": total_parents_set,
            "per_space": spaces_stats
        }

    def refresh_structure(self, per_space_limit: int = 5000) -> Dict[str, Any]:
        print("🔄 Обновление структуры Confluence (метаданные, без контента)...")
        return self.init_structure(per_space_limit=per_space_limit)

    # ===== Индексация одной страницы (on-demand), при необходимости — с детьми =====
    def index_page(self, page_id: str, include_children: bool = True):
        def _index_one(pid: str):
            page = self.connector.get_page(pid, with_content=True)
            text = page.get("content_text", "") if page else ""
            if not text:
                return
            chunks = chunk_text(text)
            embeddings = get_embeddings(chunks)

            # upsert в векторное хранилище (Chroma)
            store_embeddings(pid, chunks, embeddings)

            # апдейт меты (сохраним контент и при необходимости space/parent/title/url)
            existing = pages_db.get_page(pid)
            pages_db.insert_page(
                page_id=pid,
                project_id=(existing["project_id"] if existing else page.get("space_id") or "?"),
                title=page.get("title") or (existing["title"] if existing else pid),
                parent_id=(existing["parent_id"] if existing else page.get("parent_id")),
                content=text,
                url=(existing.get("url") if existing else None) or confluence_page_url(pid),  # ← не теряем url
            )

            # локальный индекс чанков (если используется)
            for i, emb in enumerate(embeddings):
                chunks_db.insert_chunk(pid, i, chunks[i], np.asarray(emb, dtype=np.float32).tobytes())

            print(f"📑 Страница '{page.get('title', pid)}' проиндексирована ({len(chunks)} чанков).")

        _index_one(page_id)
        if include_children:
            for ch in (pages_db.get_child_pages(page_id) or []):
                _index_one(ch["id"])

    # ===== ЕДИНСТВЕННЫЙ путь запроса (outline-RAG) =====
    def retrieve_via_outline(
        self,
        query: str,
        top_nodes: int = 12,
        top_pages: int = 6,
        top_chunks: int = 12,
        lazy_index_children: bool = False,
        restrict_to_dominant_space: bool = True,
    ) -> Dict[str, Any]:
        """
        1) поиск по локальному оглавлению/титлам (без запроса к Confluence),
        2) если нужных страниц ещё нет в чанках — индексируем их,
        3) поиск похожих чанков в найденных страницах,
        4) собираем контекст → ответ.
        """
        candidate_pages: List[str] = search_outline(query, top_nodes=top_nodes, top_pages=top_pages) or []
        candidate_pages = [pid for pid in dict.fromkeys(candidate_pages) if pid]
        print(f"🔎 Outline → страниц: {len(candidate_pages)} → {candidate_pages}")

        if not candidate_pages:
            return {"answer": "Информация не найдена (по оглавлению).", "sources": []}

        # --- Сужаем до доминирующего space, чтобы не тащить соседние проекты ---
        if restrict_to_dominant_space and len(candidate_pages) > 1:
            from collections import Counter
            page2space: Dict[str, str] = {}
            for pid in candidate_pages:
                p = pages_db.get_page(pid)
                if p and p.get("project_id"):
                    page2space[pid] = p["project_id"]

            counts = Counter(page2space.get(pid, "?") for pid in candidate_pages)
            counts.pop("?", None)
            if counts:
                dom_space, _ = counts.most_common(1)[0]
                filtered = [pid for pid in candidate_pages if page2space.get(pid) == dom_space]
                if filtered:
                    print(f"📦 Доминирующий space: {dom_space} → {len(filtered)}/{len(candidate_pages)} страниц")
                    candidate_pages = filtered

        # on-demand индексация (если по странице ещё нет чанков)
        for pid in candidate_pages:
            if not chunks_db.has_chunks(pid):
                self.index_page(pid, include_children=lazy_index_children)

        qv = np.asarray(get_embeddings([query])[0], dtype=np.float32)
        hits = query_similar_in_pages(qv, page_ids=candidate_pages, top_k=top_chunks) or []
        if not hits:
            return {"answer": "Информация не найдена (после сужения по оглавлению).", "sources": []}

        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        seen_pages = set()

        for h in hits:
            page_id = h.get("page_id")
            txt = h.get("document", "")
            if not txt:
                continue
            context_parts.append(f"[Источник: {page_id}]\n{txt}")

            if page_id and page_id not in seen_pages:
                meta = pages_db.get_page(page_id) or {}
                title = meta.get("title") or "❓ Unknown"
                url = (meta.get("url") or "") or confluence_page_url(page_id)  # ← сначала из БД, затем fallback
                sources.append({
                    "page_id": page_id,
                    "title": title,
                    "url": (url or "").strip(),
                })
                seen_pages.add(page_id)

        if not context_parts:
            return {"answer": "Информация не найдена (контекста нет).", "sources": list(sources)}

        context = "\n\n---\n\n".join(context_parts)
        prompt = (
            "Ты отвечаешь только на основе контекста.\n"
            "Если данных недостаточно — так и скажи.\n\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {query}\n"
            "Ответ:"
        )
        answer = (generate_answer(prompt) or "").strip()

        # Fallback, если LLM вернуло пусто — краткая выжимка контекста
        if not answer:
            preview = context.strip()
            if len(preview) > 1500:
                preview = preview[:1500] + "…"
            answer = preview if preview else "Не удалось сгенерировать ответ по контексту."
        
        print("SOURCES_DEBUG:", sources)
        return {"answer": answer, "sources": list(sources)}


# ========= СБОР ОГЛАВЛЕНИЯ/HEADINGS (БЕЗ ТЕКСТА) =========

class OutlinePipeline:
    def __init__(self, domain: str, email: str, api_token: str):
        self.cf = ConfluenceConnector(domain, email, api_token)

    def build_outline(self, per_space_limit: int = 5000):
        """
        Строим дерево оглавления (page title + headings) и эмбеддинги ТОЛЬКО для путей оглавления.
        Контент страниц НЕ тянем.
        """
        spaces = self.cf.get_spaces()
        print(f"🔄 Найдено пространств: {len(spaces)}")
        nodes_counter = 0

        for sp in spaces:
            sid = sp["id"]
            sname = sp.get("name") or sp.get("key") or sid
            pages_db.insert_project(sid, sname)

            # пытаемся получить все мета-страницы разом
            pages_meta = []
            try:
                pages_meta = self.cf.get_pages_meta(sid, limit=per_space_limit) or []
            except Exception as e:
                print(f"⚠️ get_pages_meta failed for space {sid}: {e}")
                pages_meta = []

            if not pages_meta:
                # если коннектор не отдаёт meta — не падаем; outline из title хотя бы для уже вставленных страниц
                pages_meta = pages_db.list_pages_in_space(sid) or []

            for p in pages_meta:
                pid = p["id"]
                title = p.get("title") or pid

                # гарантируем мета в pages_db (без контента)
                _ins_page_meta(
                    page_id=pid,
                    project_id=sid,
                    title=title,
                    parent_id=p.get("parent_id"),
                    version=str(p.get("version") or ""),
                    last_modified=p.get("last_modified"),
                )

                # узел оглавления = сам title страницы (level 0)
                node_page = uuid4_str()
                pages_db.upsert_outline_node(
                    id=node_page, project_id=sid, page_id=pid, heading_id=None,
                    title=title, level=0, parent_id=None, path=title, updated_at=now_ts()
                )
                vec = get_embeddings([title])[0]
                pages_db.put_embedding(
                    object_type="outline_title", object_id=node_page, model_name="default",
                    content_sha256=sha256_text(title),
                    vector=np.asarray(vec, dtype=np.float32), created_at=now_ts()
                )
                nodes_counter += 1

                # дочерние заголовки страницы
                heads = []
                try:
                    heads = self.cf.get_page_headings(pid) or []
                except Exception as e:
                    print(f"⚠️ get_page_headings failed for page {pid}: {e}")
                    heads = []

                parent_map = {0: node_page}
                for h in heads:
                    level = int(h.get("level", 1))
                    h_text = h.get("text") or ""
                    if not h_text:
                        continue
                    h_node = uuid4_str()
                    parent_id = parent_map.get(level - 1, node_page)
                    parent_path = pages_db.get_outline_path(parent_id) or title
                    path = f"{parent_path} > {h_text}"

                    pages_db.upsert_outline_node(
                        id=h_node, project_id=sid, page_id=pid, heading_id=h.get("id"),
                        title=h_text, level=level, parent_id=parent_id, path=path, updated_at=now_ts()
                    )
                    vec = get_embeddings([path])[0]
                    pages_db.put_embedding(
                        object_type="outline_title", object_id=h_node, model_name="default",
                        content_sha256=sha256_text(path),
                        vector=np.asarray(vec, dtype=np.float32), created_at=now_ts()
                    )
                    parent_map[level] = h_node
                    nodes_counter += 1

        print(f"🧱 outline built: nodes_inserted_or_updated={nodes_counter}")
