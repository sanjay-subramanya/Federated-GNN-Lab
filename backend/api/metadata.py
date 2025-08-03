from fastapi import APIRouter, Request, Query, HTTPException
from pydantic import BaseModel
import json
from config.settings import Config
from utils.blob_utils import download_file_from_blob

router = APIRouter()

class TrainMetadata(BaseModel):
    num_clients: int
    num_rounds: int
    last_training_time: str

@router.get("/train-metadata", response_model=TrainMetadata)
def get_train_metadata(request: Request, run_id: str = Query(default=None)):
    if not run_id:
        path = Config.model_dir / "_train_metadata.json"
    else:
        path = Config.model_dir / run_id / "_train_metadata.json"

    if not path.exists():
        try:
            download_file_from_blob(str(path), str(path))
        except Exception as e:
            raise HTTPException(status_code=404, detail="Metadata file not found.")

    with open(str(path), "r") as f:
        raw = json.load(f)

    return TrainMetadata(**raw)