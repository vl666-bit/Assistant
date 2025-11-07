# main.py
import uvicorn
from fastapi import FastAPI

from app.db import init_db
from app.api import router as api_router
from app.confluence_pipeline import ConfluencePipeline, OutlinePipeline
from config import CONFLUENCE

app = FastAPI(title="RAG API", version="1.0.0")


@app.on_event("startup")
def startup_event():
    print(">>> init_db вызван!")
    try:
        # 1) создаём схемы БД (projects/pages/outline/chunks/embeddings)
        init_db()
    except Exception as e:
        import traceback
        print("!!! init_db failed:", e)
        print(traceback.format_exc())
        raise

    # 2) на старте — сразу тянем МЕТА (ветви) без контента + строим оглавление
    #    если что-то пойдёт не так (например, 403 по токену), не валим приложение
    try:
        print(">>> init_structure (Confluence meta, без контента)…")
        pipe = ConfluencePipeline(
            domain=CONFLUENCE["DOMAIN"],
            email=CONFLUENCE["EMAIL"],
            api_token=CONFLUENCE["API_TOKEN"],
        )
        stats = pipe.init_structure(per_space_limit=5000)
        print(f"✔ meta loaded: spaces={stats.get('spaces')}, pages={stats.get('pages_meta_inserted')}")

        print(">>> build_outline (titles + headings + vectors)…")
        outline_pipe = OutlinePipeline(
            domain=CONFLUENCE["DOMAIN"],
            email=CONFLUENCE["EMAIL"],
            api_token=CONFLUENCE["API_TOKEN"],
        )
        outline_pipe.build_outline(per_space_limit=5000)
        print("✔ outline built")
    except Exception as e:
        import traceback
        print("!!! startup meta/outline fetch failed (will continue without it):", e)
        print(traceback.format_exc())
        # не делаем raise — сервер продолжит работу; можно вручную вызвать /init_structure и /build_outline


# Подключаем роуты
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
