import os

def load_documents_from_folder(folder_path: str, chunk_size: int = 500):
    """
    Загружает все .txt документы из папки и разбивает на чанки.
    """
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                for i in range(0, len(text), chunk_size):
                    chunks.append(text[i:i+chunk_size])
    return chunks
