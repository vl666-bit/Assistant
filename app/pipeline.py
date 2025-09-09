from app.retrieval import query_similar
from app.embedding import get_embeddings
from app.llm_stub import generate_answer

def answer_query(query: str) -> str:
    query_embedding = get_embeddings([query])[0]
    context_docs = query_similar(query_embedding)
    context = "\n".join([doc['document'] for doc in context_docs])
    return generate_answer(query, context)
