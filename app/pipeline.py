from app.utils import save_file
from app.embedding import get_embeddings
from app.retrieval import store_embeddings
from app.llm_stub import generate_answer

def process_file(filename: str, content: bytes):
    path = save_file(filename, content)
    texts = [path.read_text(encoding="utf-8")]  # пока упрощённо
    embeddings = get_embeddings(texts)
    store_embeddings(texts, embeddings)
    return f"Файл {filename} обработан и проиндексирован"

def answer_query(query: str) -> str:
    # заглушка, подключим retrieval позже
    context = "Это тестовый контекст"
    return generate_answer(query, context)
