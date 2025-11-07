# chat_ui.py
import time
import json
import requests
import streamlit as st
from typing import List, Dict, Any

from frontend_config import API_URL as API_URL_DEFAULT  # дефолтный базовый URL FastAPI

st.set_page_config(page_title="RAG Tester", page_icon="🧠", layout="wide")

# ===== helpers =====
def norm_base_url(u: str) -> str:
    u = (u or "").strip()
    while u.endswith("/"):
        u = u[:-1]
    return u

def join_api(base: str, path: str) -> str:
    base = norm_base_url(base)
    if not path.startswith("/"):
        path = "/" + path
    return base + path

def post_json(url: str, payload: dict, timeout: int):
    try:
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=timeout)
        dt = time.time() - t0
        return resp, dt, None
    except Exception as e:
        return None, 0.0, str(e)

def get_req(url: str, timeout: int):
    try:
        t0 = time.time()
        resp = requests.get(url, timeout=timeout)
        dt = time.time() - t0
        return resp, dt, None
    except Exception as e:
        return None, 0.0, str(e)

def build_history(messages: List[Dict[str, Any]], limit_pairs: int = 4):
    hist = [{"role": m["role"], "text": m["content"]}
            for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
    return hist[-(limit_pairs * 2):] if hist else None

def render_sources(sources: List[Dict[str, Any]], hide: bool = False, title: str = "🔎 Источники"):
    """
    Рисует список источников.
    Поддерживает поля: title/name/page_id/id/url/link/base/webui
    Если нет ссылок — покажет отладочный блок с «сырыми» sources.
    """
    if not sources:
        return

    container = st.expander(title) if hide else st.container()
    with container:
        has_any_url = False
        for s in sources:
            ttl = (s.get("title") or s.get("name") or s.get("page_title") or s.get("page_id") or "Источник")
            ttl = str(ttl).strip()
            pid = s.get("page_id") or s.get("id")
            url = (s.get("url") or s.get("link") or "").strip()

            # Fallback для Confluence: base + webui -> итоговая ссылка
            base = (s.get("base") or "").strip()
            webui = (s.get("webui") or "").strip()
            if not url and base and webui:
                url = f"{base.rstrip('/')}/{webui.lstrip('/')}"

            if url:
                has_any_url = True
                st.markdown(f"- **{ttl}** (ID: `{pid}`) — [Открыть]({url})")
            else:
                st.markdown(f"- **{ttl}** (ID: `{pid}`) — нет ссылки")

        # Если ни одной ссылки не было — покажем «сырые» данные для быстрой диагностики
        if not has_any_url:
            with st.expander("ℹ️ Debug: raw sources"):
                st.json(sources)


# ===== Session defaults =====
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "...", "content": "...", "sources": [...]}]

if "API_URL" not in st.session_state:
    st.session_state.API_URL = API_URL_DEFAULT

if "endpoint" not in st.session_state:
    st.session_state.endpoint = "query (RAG via outline)"

# ===== Sidebar =====
st.sidebar.header("⚙️ Настройки")

st.session_state.API_URL = st.sidebar.text_input(
    "API base URL",
    value=st.session_state.API_URL,
    help="Адрес FastAPI (например, http://127.0.0.1:8000)"
)
API_URL = norm_base_url(st.session_state.API_URL)

cols = st.sidebar.columns(3)
with cols[0]:
    if st.button("🧹 Очистить чат"):
        st.session_state.messages = []
with cols[1]:
    if st.button("🩺 Health"):
        url = join_api(API_URL, "/health")
        resp, dt, err = get_req(url, timeout=8)
        if err or resp is None:
            st.sidebar.error(f"Health error: {err}")
        else:
            try:
                st.sidebar.success(f"OK {resp.status_code} · {dt:.2f}s")
                st.sidebar.json(resp.json())
            except Exception:
                st.sidebar.warning(f"Ответ не JSON · {dt:.2f}s")
                st.sidebar.text(resp.text)
with cols[2]:
    if st.button("📄 Docs"):
        try:
            t0 = time.time()
            resp = requests.get(join_api(API_URL, "/docs"), timeout=8)
            dt = time.time() - t0
            if resp.status_code in (200, 404):
                st.sidebar.success(f"API откликается ({resp.status_code}) · {dt:.2f}s")
            else:
                st.sidebar.warning(f"HTTP {resp.status_code} · {dt:.2f}s")
        except Exception as e:
            st.sidebar.error(f"Ошибка: {e}")

st.sidebar.markdown("---")
svc_cols = st.sidebar.columns(3)
with svc_cols[0]:
    if st.button("⬇️ Init structure"):
        url = join_api(API_URL, "/init_structure")
        resp, dt, err = post_json(url, {}, timeout=600)
        if err or resp is None:
            st.sidebar.error(f"Init error: {err}")
        else:
            if resp.status_code == 200:
                st.sidebar.success(f"Инициализировано · {dt:.1f}s")
                st.sidebar.json(resp.json())
            else:
                st.sidebar.error(f"HTTP {resp.status_code}")
                st.sidebar.text(resp.text)
with svc_cols[1]:
    if st.button("🧱 Build outline"):
        url = join_api(API_URL, "/build_outline")
        resp, dt, err = post_json(url, {}, timeout=600)
        if err or resp is None:
            st.sidebar.error(f"Outline error: {err}")
        else:
            if resp.status_code == 200:
                st.sidebar.success(f"Оглавление построено · {dt:.1f}s")
                st.sidebar.json(resp.json())
            else:
                st.sidebar.error(f"HTTP {resp.status_code}")
                st.sidebar.text(resp.text)
with svc_cols[2]:
    if st.button("🔄 Refresh"):
        url = join_api(API_URL, "/refresh_structure")
        resp, dt, err = post_json(url, {}, timeout=600)
        if err or resp is None:
            st.sidebar.error(f"Refresh error: {err}")
        else:
            if resp.status_code == 200:
                st.sidebar.success(f"Обновлено · {dt:.1f}s")
                st.sidebar.json(resp.json())
            else:
                st.sidebar.error(f"HTTP {resp.status_code}")
                st.sidebar.text(resp.text)

# выбор эндпоинта — только 3 варианта
options = [
    "chat (LLM)",
    "query (RAG via outline)",
    "upload (ingest file)",
]
try:
    default_idx = options.index(st.session_state.endpoint)
except ValueError:
    default_idx = 1
endpoint = st.sidebar.selectbox("Эндпоинт", options, index=default_idx)
st.session_state.endpoint = endpoint

# Общие настройки
timeout_s = st.sidebar.slider("timeout, сек", min_value=5, max_value=600, value=60)
hide_sources = st.sidebar.checkbox("Скрывать источники в аккордеоне", value=False)

# Параметры для /query (outline)
with st.sidebar.expander("Параметры /query"):
    top_nodes = st.slider("top_nodes (узлов оглавления)", 5, 50, 12, key="top_nodes")
    top_pages = st.slider("top_pages (страниц после сужения)", 1, 50, 6, key="top_pages")
    top_chunks = st.slider("top_chunks (чанков в ответе)", 1, 50, 12, key="top_chunks")
    lazy_index_children = st.checkbox("Индексировать детей при нехватке", value=False)

st.title("🧠 RAG Tester")

# ===== История =====
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"], hide=hide_sources)

# ===== Ввод пользователя =====
user_input = st.chat_input("Введи вопрос…")

if user_input:
    # Показать сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Выполнить запрос
    with st.chat_message("assistant"):
        with st.spinner("Запрос выполняется…"):
            answer = "❌ Эндпоинт не выбран"
            sources: List[Dict[str, Any]] = []
            dt = 0.0

            if endpoint == "chat (LLM)":
                url = join_api(API_URL, "/chat")
                payload = {"prompt": user_input}

            elif endpoint == "query (RAG via outline)":
                url = join_api(API_URL, "/query")
                payload = {
                    "query": user_input,
                    "top_nodes": top_nodes,
                    "top_pages": top_pages,
                    "top_chunks": top_chunks,
                    "lazy_index_children": lazy_index_children,
                }

            elif endpoint == "upload (ingest file)":
                url, payload = "", {}
            else:
                url, payload = "", {}

            if url:
                resp, dt, err = post_json(url, payload, timeout_s)
                if err:
                    answer = f"❌ Ошибка подключения: {err}"
                elif resp is None:
                    answer = "❌ Нет ответа от сервера"
                else:
                    # попытка распарсить JSON
                    try:
                        data = resp.json()
                    except Exception:
                        data = None

                    if resp.status_code != 200:
                        answer = f"⚠️ HTTP {resp.status_code}: {data if data is not None else resp.text}"
                    else:
                        if isinstance(data, dict):
                            ans = data.get("answer")
                            if isinstance(ans, (dict, list)):
                                answer = "```json\n" + json.dumps(ans, ensure_ascii=False, indent=2) + "\n```"
                            else:
                                answer = (ans or "").strip()
                            sources = data.get("sources", []) or []
                        else:
                            # неожиданный тип -> показываем как строку
                            answer = str(data) if data is not None else (resp.text or "")

                        if not answer:
                            answer = "Не удалось сгенерировать ответ. Проверь контекст/оглавление и повтори запрос."

            # Рисуем ответ + источники
            st.markdown(answer)
            render_sources(sources, hide=hide_sources)
            st.caption(f"⏱ {dt:.2f} сек")

    # Сохраняем ответ ассистента — вместе с источниками
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

# ===== Upload =====
if endpoint == "upload (ingest file)":
    st.subheader("📤 Загрузка файла в базу (/upload)")
    file = st.file_uploader("Выбери файл", type=["txt", "pdf", "md", "json", "html", "csv"])
    if file is not None:
        if st.button("Загрузить"):
            with st.spinner("Грузим файл…"):
                try:
                    files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
                    t0 = time.time()
                    resp = requests.post(join_api(API_URL, "/upload"), files=files, timeout=timeout_s)
                    dt = time.time() - t0
                    if resp.status_code == 200:
                        st.success("Готово")
                        try:
                            st.json(resp.json())
                        except Exception:
                            st.text(resp.text)
                    else:
                        st.error(f"HTTP {resp.status_code}")
                        st.text(resp.text)
                    st.caption(f"⏱ {dt:.2f} сек")
                except Exception as e:
                    st.error(f"❌ Ошибка подключения: {e}")
