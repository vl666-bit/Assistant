# app/embedding.py
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

# Загружаем модель один раз
model = SentenceTransformer(EMBED_MODEL)


def get_embeddings(texts: list[str], *, normalize: bool = False) -> np.ndarray:
    """
    Возвращает эмбеддинги для списка текстов.
    Args:
        texts: список строк
        normalize: если True — нормализовать вектора (||v||=1)
    Returns:
        np.ndarray float32, shape = (n_texts, dim)
    """
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)
