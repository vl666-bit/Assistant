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

def query_similar(query_embedding: list[float], top_k: int = 3) -> list[dict]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return [
        {"document": doc, "id": _id}
        for doc, _id in zip(results['documents'][0], results['ids'][0])
    ]
