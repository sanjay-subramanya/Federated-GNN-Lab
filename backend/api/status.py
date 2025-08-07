from fastapi import APIRouter, Request
from pathlib import Path
from pydantic import BaseModel
from config.settings import Config

router = APIRouter()

class RunIdRequest(BaseModel):
    run_id: str

@router.get("/dissect/status")
def get_analysis_status(request: Request, run_id: str):
    run_id = run_id.strip()
    model_dir = Config.model_dir / run_id
    is_ready = model_dir.exists() and any(model_dir.iterdir())
    return {"ready": is_ready}
