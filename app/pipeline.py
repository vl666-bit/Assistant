from app.utils import save_file
from app.embedding import get_embeddings
from app.retrieval import store_embeddings, query_similar
from app.llm_stub import generate_answer

def process_file(filename: str, content: bytes):
    path = save_file(filename, content)
    texts = [path.read_text(encoding="utf-8")]
    embeddings = get_embeddings(texts)
    store_embeddings(texts, embeddings)
    return f"Файл {filename} обработан и проиндексирован"

def answer_query(query: str) -> str:
    query_embedding = get_embeddings([query])[0]
    context_docs = query_similar(query_embedding)
    context = "\n".join([doc["document"] for doc in context_docs])
    return generate_answer(query, context)
