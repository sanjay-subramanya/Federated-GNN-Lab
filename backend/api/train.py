import json
import time
import logging
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Generator
from data.loader import load_and_partition_data
from utils.logging_utils import configure_logging
from trainer.manual_simulation import run_manual_simulation

configure_logging()
logger = logging.getLogger(__name__)

router = APIRouter()

class TrainConfig(BaseModel):
    num_clients: int
    num_rounds: int

@router.post("/train")
async def stream_training(request: Request, req: TrainConfig):
    num_rounds = req.num_rounds
    num_clients = req.num_clients
    ctx = request.app.state.ctx

    client_datasets, num_features, num_classes = load_and_partition_data(
        ctx.protein_df_raw, ctx.phen_df_raw, num_clients=num_clients
    )

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    async def training_generator() -> Generator[str, None, None]:
        try:
            for round_result in run_manual_simulation(
                client_datasets=client_datasets,
                num_features=num_features,
                num_classes=num_classes,
                num_rounds=num_rounds,
                yield_per_round=True,
                run_id=run_id,
            ):
                round_result['run_id'] = run_id
                yield json.dumps(round_result) + "\n"
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error in training_generator: {str(e)}")
            raise

    logger.info(f"Training stream completed for run_id: {run_id}")
    return StreamingResponse(
        training_generator(),
        headers={"X-Run-Id": run_id, "Content-Type": "text/event-stream"},
        media_type="text/event-stream"
    )
