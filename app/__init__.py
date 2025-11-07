from .pages_db import init_pages_db
from .chunks_db import init_chunks_db


def init_db():
    print(">>> init_db вызван!")
    init_pages_db()
    init_chunks_db()
