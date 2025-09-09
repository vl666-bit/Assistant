from fastapi import APIRouter, UploadFile, File
from app.pipeline import process_file

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    result = process_file(file.filename, content)
    return {"status": "ok", "details": result}
