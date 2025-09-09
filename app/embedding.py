from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL

model = SentenceTransformer(EMBED_MODEL)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, convert_to_numpy=True).tolist()
