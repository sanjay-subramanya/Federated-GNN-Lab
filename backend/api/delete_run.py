import logging
import shutil
from fastapi import APIRouter, Request

from config.settings import Config
from utils.blob_utils import delete_run_from_blob
from utils.logging_utils import configure_logging
from api.status import RunIdRequest

configure_logging()
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/delete-run")
async def delete_run(request: Request, req: RunIdRequest):
    run_id = req.run_id.strip()
    try:
        if Config.upload_to_blob:
            delete_run_from_blob(run_id)
            logger.info(f"Deleted blob for runId: {run_id}")

        # Also delete local copy if exists
        run_folder = Config.model_dir / run_id

        if run_folder.exists():
            shutil.rmtree(run_folder)
        logger.info(f"Deleted local folder for runId: {run_id}")
        return {"status": "deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
