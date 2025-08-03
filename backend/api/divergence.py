import json
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, RootModel
from typing import Dict, Optional, List
from pathlib import Path
from config.settings import Config
from utils.logging_utils import configure_logging
from utils.blob_utils import load_file_from_blob_if_needed

configure_logging()
logger = logging.getLogger(__name__)

router = APIRouter()

class RoundDivergenceData(BaseModel):
    round: int
    global_loss: float
    client_divergence: Dict[str, Dict[str, float]]

class DivergenceHistoryResponse(RootModel[List[RoundDivergenceData]]):
    pass

@router.get("/dissect/divergence-history", response_model=DivergenceHistoryResponse)
def get_divergence_history(
    request: Request,
    run_id: Optional[str] = None
    ):

    if run_id:
        current_model_dir = Config.model_dir / run_id
        blob_prefix = f"saved_models/{run_id}"
        logger.info(f"Received request to get divergence history for run_id: {run_id}")
    else:
        current_model_dir = Config.model_dir  
        blob_prefix = "saved_models"

    path = current_model_dir / "_divergence_metrics.json"
    blob_key = f"{blob_prefix}/_divergence_metrics.json"

    try:
        local_path = load_file_from_blob_if_needed(blob_key, str(path))
    except Exception as e:
        logger.warning(f"Divergence metrics file not found at {path} (blob: {blob_key}): {e}")
        raise HTTPException(status_code=404, detail="Divergence metrics file not found.")

    try:
        with open(local_path, "r") as f:
            data = json.load(f)
        return DivergenceHistoryResponse.model_validate(data)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from divergence metrics file at {path}: {e}")
        raise HTTPException(status_code=500, detail="Error loading divergence metrics data.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading divergence metrics: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
