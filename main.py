from fastapi import FastAPI
from app.api import router

app = FastAPI(title="RAG System")

app.include_router(router)
