# app/connectors/confluence_connector.py
import time
import re
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse

import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup


class ConfluenceConnector:
    def __init__(self, domain: str, email: str, api_token: str):
        self.base = self._normalize_base(domain)
        self.base_url = f"{self.base}/rest/api"  # совместимо с твоим прежним именем
        self.auth = HTTPBasicAuth(email, api_token)

        # Сессия с ретраями (429/5xx)
        self.sess = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.sess.mount("https://", adapter)
        self.sess.mount("http://", adapter)

        self._headers = {"Accept": "application/json"}

    # ---------- base/normalization ----------

    @staticmethod
    def _normalize_base(domain: str) -> str:
        d = (domain or "").strip().rstrip("/")
        if not d:
            raise ValueError("Confluence domain is empty")
        # Если домен уже со схемой
        if d.startswith("http://") or d.startswith("https://"):
            p = urlparse(d)
            host = p.netloc or p.path  # на случай кривых строк
            # поддержим вариант, когда уже указан /wiki
            path = p.path.rstrip("/")
            if path and path != "/":
                # оставляем только /wiki, если он присутствует; остальное отбрасываем
                path = "/wiki" if "/wiki" in path.split("/") else ""
            else:
                path = ""
        else:
            host = d
            path = ""

        # У Atlassian Cloud API обычно за /wiki — если не указано, добавим.
        if not path:
            path = "/wiki"

        return f"https://{host}{path}"

    # ---------- low-level ----------

    def _request(self, method: str, url: str, *, params=None, json=None, headers=None, timeout: float = 15.0):
        h = dict(self._headers)
        if headers:
            h.update(headers)
        resp = self.sess.request(method, url, params=params, json=json, headers=h, auth=self.auth, timeout=timeout)
        if resp.status_code >= 400:
            # популярная ошибка из логов: 403 "Current user not permitted to use Confluence"
            raise Exception(f"[Confluence {resp.status_code}] {resp.text or resp.reason} | url={url} | params={params}")
        return resp

    def _get_rest(self, path: str, *, params: Optional[Dict[str, Any]] = None):
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self.base_url}{path}"
        return self._request("GET", url, params=params)

    # ================== SPACES ==================
    def get_spaces(self, limit: int = 50) -> List[Dict]:
        """
        Получить список пространств (spaces). Возвращает [{"id": <spaceKey>, "name": <name>}].
        """
        resp = self._get_rest("/space", params={"limit": limit})
        data = resp.json() or {}
        results = data.get("results", []) or []
        return [{"id": str(space.get("key")), "name": str(space.get("name"))} for space in results]

    # ================== PAGES (basic) ==================
    def get_pages_in_space(self, space_key: str, limit: int = 50) -> List[Dict]:
        """
        Получить список страниц в пространстве (id + title).
        Забирает все страницы постранично (limit на одну страницу запроса).
        """
        start = 0
        out: List[Dict] = []
        while True:
            params = {
                "spaceKey": space_key,
                "limit": limit,
                "start": start,
                "type": "page",  # важно, чтобы не прилетали blogpost/attachment
            }
            resp = self._get_rest("/content", params=params)
            data = resp.json() or {}
            results = data.get("results", []) or []
            out.extend([{"id": str(p.get("id")), "title": str(p.get("title", ""))} for p in results])

            size = int(data.get("size", len(results)))
            limit_val = int(data.get("limit", limit))
            start += limit_val
            if size < limit_val or len(results) == 0:
                break
        return out

    def get_child_pages(self, parent_id: str, limit: int = 50) -> List[Dict]:
        """
        Получить подстраницы у документа.
        """
        start = 0
        out: List[Dict] = []
        while True:
            params = {"limit": limit, "start": start}
            resp = self._get_rest(f"/content/{parent_id}/child/page", params=params)
            data = resp.json() or {}
            results = data.get("results", []) or []
            out.extend([{"id": str(p.get("id")), "title": str(p.get("title", ""))} for p in results])

            size = int(data.get("size", len(results)))
            limit_val = int(data.get("limit", limit))
            start += limit_val
            if size < limit_val or len(results) == 0:
                break
        return out

    def get_page(self, page_id: str, with_content: bool = True) -> Dict:
        """
        Получить страницу по ID (с контентом или только мета).
        """
        expand = "body.storage,version,ancestors" if with_content else "version,ancestors"
        resp = self._get_rest(f"/content/{page_id}", params={"expand": expand})
        data = resp.json() or {}

        ancestors = data.get("ancestors") or []
        # В Confluence ancestors идут от корня к непосредственному родителю — последний элемент и есть parent
        parent_id = str(ancestors[-1].get("id")) if ancestors else None

        page = {
            "id": str(data.get("id")),
            "title": str(data.get("title", "")),
            "version": ((data.get("version") or {}).get("number")),
            "parent_id": parent_id,
            "space_id": ((data.get("space") or {}).get("key") or (data.get("space") or {}).get("id")),
        }

        if with_content:
            html_content = (((data.get("body") or {}).get("storage") or {}).get("value")) or ""
            text_content = self._html_to_text(html_content)
            page.update({"content_text": text_content, "content_html": html_content})
        return page

    # ================== PAGES (meta for pipeline) ==================
    def get_pages_meta(self, space_key: str, limit: int = 200, start: int = 0) -> List[Dict]:
        """
        Метаданные страниц в space (id, title, version.number, parent_id).
        Пагинация до полного списка.
        """
        out: List[Dict] = []
        cur_start = start
        while True:
            params = {
                "spaceKey": space_key,
                "limit": limit,
                "start": cur_start,
                "expand": "version,ancestors",
                "type": "page",
            }
            resp = self._get_rest("/content", params=params)
            data = resp.json() or {}
            results = data.get("results", []) or []
            for page in results:
                version = (page.get("version") or {}).get("number")
                ancestors = page.get("ancestors") or []
                parent_id = str(ancestors[-1].get("id")) if ancestors else None
                out.append({
                    "id": str(page.get("id")),
                    "title": str(page.get("title", "")),
                    "version": version,
                    "parent_id": parent_id
                })

            size = int(data.get("size", len(results)))
            limit_val = int(data.get("limit", limit))
            cur_start += limit_val
            if size < limit_val or len(results) == 0:
                break
        return out

    def get_page_meta(self, page_id: str) -> Dict:
        """
        Метаданные конкретной страницы: id, title, version, parent_id.
        """
        resp = self._get_rest(f"/content/{page_id}", params={"expand": "version,ancestors"})
        data = resp.json() or {}
        ancestors = data.get("ancestors") or []
        parent_id = str(ancestors[-1].get("id")) if ancestors else None
        return {
            "id": str(data.get("id")),
            "title": str(data.get("title", "")),
            "version": (data.get("version") or {}).get("number"),
            "parent_id": parent_id,
        }

    def get_page_body(self, page_id: str) -> str:
        """
        Возвращает HTML тела страницы (storage).
        """
        resp = self._get_rest(f"/content/{page_id}", params={"expand": "body.storage"})
        data = resp.json() or {}
        return (((data.get("body") or {}).get("storage") or {}).get("value") or "") or ""

    def get_page_headings(self, page_id: str) -> List[Dict]:
        """
        Возвращает заголовки (h1..h6) страницы с их уровнями и id (если есть).
        Формат: [{"level": 1, "id": "h1-anchor", "text": "Intro"}, ...]
        """
        html = self.get_page_body(page_id)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        headings: List[Dict] = []
        for tag in soup.find_all(re.compile(r"^h[1-6]$", re.IGNORECASE)):
            level = int(tag.name[1])
            text = tag.get_text(" ", strip=True)
            anchor_id = tag.get("id")  # может быть None
            if text:
                headings.append({"level": level, "id": anchor_id, "text": text})
        return headings

    # ================== SEARCH ==================
    def search_pages(self, query: str, limit: int = 5, space_key: Optional[str] = None) -> List[Dict]:
        """
        Поиск страниц по CQL. Можно ограничить space.
        """
        cql = f'text~"{query}" and type=page'
        if space_key:
            cql = f'space="{space_key}" and {cql}'
        resp = self._get_rest("/search", params={"cql": cql, "limit": limit})
        data = resp.json() or {}
        results = []
        for item in data.get("results", []) or []:
            content = item.get("content") or {}
            cid = content.get("id")
            title = content.get("title")
            if cid and title:
                results.append({"id": str(cid), "title": str(title)})
        return results

    # ================== UTILS ==================
    def _html_to_text(self, html: str) -> str:
        """
        Превращает HTML в текст. По умолчанию — простой get_text с переносами.
        При необходимости можно доработать фильтрацию макросов/таблиц.
        """
        soup = BeautifulSoup(html or "", "html.parser")
        text = soup.get_text(separator="\n")
        return (text or "").strip()
