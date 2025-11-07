# app/llm_stub.py
import os
from llama_cpp import Llama as _Llama

# Путь к модели: ../models/твоя-модель.gguf (папка models рядом с main.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "vigostral-7b-chat.Q4_K_M.gguf")

_LLM = None  # ленивый синглтон

# Единый системный промт
SYSTEM_PROMPT = (
    "Ты — ассистент.\n"
    "Всегда отвечай на русском языке.\n"
    "Никогда не переводишь вопрос пользователя.\n"
    "Отвечай на том же языке, на котором задан вопрос.\n"
    "Не повторяй вопрос в ответе.\n"
    "Отвечай кратко и по существу.\n"
    "Не придумывай фактов и биографий.\n"
)

def _get_llm():
    global _LLM
    if _LLM is None:
        _LLM = _Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=max(1, os.cpu_count() or 4),
            n_batch=512,
            verbose=False,
        )
    return _LLM


def generate_answer(query: str, context: str = "") -> str:
    """
    Генерация ответа с учётом вопроса и (опционально) контекста.
    Системный промт добавляется всегда.
    """
    if context:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Контекст:\n{context}\n\n"
            f"Пользователь: {query}\nАссистент:"
        )
    else:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Пользователь: {query}\nАссистент:"
        )

    llm = _get_llm()
    result = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        stop=["Пользователь:", "Ассистент:"],
        echo=False,
    )
    return result["choices"][0]["text"].strip()
