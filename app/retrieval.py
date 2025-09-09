import chromadb
from chromadb.config import Settings
from app.config import CHROMA_DB_DIR

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_DB_DIR
))

collection = client.get_or_create_collection(name="rag_docs")

def store_embeddings(texts: list[str], embeddings: list[list[float]]):
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        collection.add(
            documents=[text],
            embeddings=[emb],
            ids=[f"doc_{i}"]
        )
