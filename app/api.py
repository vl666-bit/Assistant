from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.pipeline import process_file, answer_query

router = APIRouter()

# === Upload ===
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    result = process_file(file.filename, content)
    return {"status": "ok", "details": result}

# === Ask ===
class QueryInput(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(data: QueryInput):
    response = answer_query(data.question)
    return {"answer": response}
