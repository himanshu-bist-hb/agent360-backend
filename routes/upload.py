from fastapi import APIRouter, File, UploadFile
from services.data_service import process_upload

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and validate a CSV/Excel file."""
    return await process_upload(file)
