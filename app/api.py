# app/api.py
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.confluence_pipeline import ConfluencePipeline, OutlinePipeline
from app.llm_stub import generate_answer
from app.utils import format_exc
from app.db import pages_db
from config import CONFLUENCE

# === Pipelines (singletons for this module) ===
pipe = ConfluencePipeline(
    domain=CONFLUENCE["DOMAIN"],
    email=CONFLUENCE["EMAIL"],
    api_token=CONFLUENCE["API_TOKEN"],
)
outline_pipe = OutlinePipeline(
    domain=CONFLUENCE["DOMAIN"],
    email=CONFLUENCE["EMAIL"],
    api_token=CONFLUENCE["API_TOKEN"],
)

router = APIRouter()

# ====== Models ======

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    answer: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_nodes: int = 12
    top_pages: int = 6
    top_chunks: int = 12
    lazy_index_children: bool = False  # дозагрузка детей при необходимости

class SourceItem(BaseModel):
    page_id: str
    title: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)

class InitResponse(BaseModel):
    spaces: int
    pages_meta_inserted: int
    per_space: List[Dict[str, Any]] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    pages_count: int
    outline_nodes: int

# ====== Helpers ======

def _ensure_initialized_if_needed() -> Optional[Dict[str, Any]]:
    """
    Ленивая инициализация: если мета в БД ещё не подтянута — тянем всё оглавление/мету без контента.
    Возвращает статистику init, если он был выполнен, иначе None.
    """
    try:
        pages_cnt = pages_db.count()
        outline_cnt = pages_db.count_outline_nodes()
    except Exception:
        # если что-то не так со старой схемой — пытаемся проинициализировать
        pages_cnt, outline_cnt = 0, 0

    if pages_cnt == 0:
        # 1) загрузить метаданные страниц (все ветви) без контента
        init_stats = pipe.init_structure(per_space_limit=5000)
        # 2) построить оглавление (title + headings) и их эмбеддинги
        outline_pipe.build_outline(per_space_limit=5000)
        return init_stats
    elif outline_cnt == 0:
        # мета есть, но нет оглавления — строим
        outline_pipe.build_outline(per_space_limit=5000)
    return None

# ====== Routes ======

@router.get("/health", response_model=HealthResponse)
def health():
    try:
        pages_cnt = pages_db.count()
        outline_cnt = pages_db.count_outline_nodes()
        return HealthResponse(status="ok", pages_count=pages_cnt, outline_nodes=outline_cnt)
    except Exception as e:
        info = format_exc(e)
        # отдаём статус как текст в answer, но модель требует чисел — подстрахуемся нулями
        return HealthResponse(status=f"error: {info['error']}", pages_count=0, outline_nodes=0)

@router.post("/init_structure", response_model=InitResponse)
def init_structure():
    try:
        stats = pipe.init_structure(per_space_limit=5000)
        return InitResponse(**stats)
    except Exception as e:
        info = format_exc(e)
        # фоллбек, чтобы не ломать контракт
        return InitResponse(spaces=0, pages_meta_inserted=0, per_space=[{"error": info["error"]}])

@router.post("/build_outline", response_model=Dict[str, Any])
def build_outline():
    try:
        outline_pipe.build_outline(per_space_limit=5000)
        return {"status": "ok"}
    except Exception as e:
        info = format_exc(e)
        return {"status": "error", "error": info["error"], "trace": info["trace"]}

@router.post("/refresh_structure", response_model=InitResponse)
def refresh_structure():
    try:
        stats = pipe.refresh_structure(per_space_limit=5000)
        outline_pipe.build_outline(per_space_limit=5000)
        return InitResponse(**stats)
    except Exception as e:
        info = format_exc(e)
        return InitResponse(spaces=0, pages_meta_inserted=0, per_space=[{"error": info["error"]}])

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return ChatResponse(answer=generate_answer(req.prompt))
    except Exception as e:
        info = format_exc(e)
        return ChatResponse(answer=f"Ошибка в /chat: {info['error']}\n{info['trace']}")

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        # Ленивая инициализация, если БД пустая (подтянем оглавления и мету без контента)
        _ensure_initialized_if_needed()

        # Основной пайп: поиск по оглавлению → дозагрузка контента при необходимости → поиск по чанкам
        result: Dict[str, Any] = pipe.retrieve_via_outline(
            query=req.query,
            top_nodes=req.top_nodes,
            top_pages=req.top_pages,
            top_chunks=req.top_chunks,
            lazy_index_children=req.lazy_index_children,
        )
        sources = [
            SourceItem(page_id=s["page_id"], title=s.get("title") or "❓ Unknown")
            for s in (result.get("sources") or [])
        ]
        return QueryResponse(answer=result.get("answer", "Информация не найдена"), sources=sources)
    except Exception as e:
        info = format_exc(e)
        return QueryResponse(answer=f"Ошибка в /query: {info['error']}\n{info['trace']}", sources=[])
