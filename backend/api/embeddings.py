import torch
import logging
import umap
import gc
import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, List, Optional
from functools import lru_cache
from pathlib import Path
from utils.embedding import extract_patient_embeddings
from utils.blob_utils import load_file_from_blob_if_needed
from utils.logging_utils import configure_logging
from utils.metadata import load_metadata
from config.settings import Config

configure_logging()
logger = logging.getLogger(__name__)

router = APIRouter()

class EmbeddingPoint(BaseModel):
    x: float
    y: float
    patient_id: str
    label: str 

class DissectEmbeddingsResponse(BaseModel):
    embeddings: Dict[str, List[EmbeddingPoint]]

@router.get("/dissect/embeddings", response_model=DissectEmbeddingsResponse)
def get_dissection_embeddings(
    request: Request,
    run_id: Optional[str] = None
    ):
    ctx = request.app.state.ctx

    if run_id:
        current_model_dir = Config.model_dir / run_id
        blob_prefix = f"saved_models/{run_id}"
    else:
        current_model_dir = Config.model_dir  
        blob_prefix = "saved_models" 

    global_model_path = current_model_dir / "global_model_manual.pt"
    model_paths = {"global": global_model_path}

    metadata_path = current_model_dir / f"_train_metadata.json"
    metadata_blob_key = f"{blob_prefix}/_train_metadata.json"

    local_metadata_path = load_file_from_blob_if_needed(metadata_path, metadata_blob_key)
    num_clients_trained, num_rounds_trained = load_metadata(local_metadata_path)

    for i in range(1, 1 + num_clients_trained):
        model_paths[f"client_{i}"] = current_model_dir / f"client_{i}_model.pt"

    label_map = ctx.merged_df["vital_status.demographic"].loc[ctx.protein_df.index]
    embedding_model_map = []
    combined_embeddings = None

    for model_name, path in model_paths.items():
        blob_key = f"{blob_prefix}/{path.name}"
        if not path.exists():
            try:
                # Attempt to download from blob
                load_file_from_blob_if_needed(blob_key, path)
            except Exception as e:
                logger.warning(f"Model file not found for {model_name} at {path} (blob: {blob_key}): {e}")
                logger.warning(f"Directory contents: {list(Path(path.parent).iterdir())}")
                continue

        try:
            model = ctx._load_model(path, blob_key)
            logger.info(f"Loading and extracting embeddings from {model_name}...")
            embedding_df = extract_patient_embeddings(model=model, protein_df=ctx.protein_df)

            embedding_array = embedding_df.values.astype(np.float32)
            if combined_embeddings is None:
                combined_embeddings = embedding_array
            else:
                combined_embeddings = np.concatenate((combined_embeddings, embedding_array), axis=0)

            # all_raw_embeddings_list.append(embedding_df.values)
            embedding_model_map.append({
                "model_name": model_name,
                "count": len(embedding_df),
                "patient_ids": embedding_df.index.tolist()
            })

            del model, embedding_df, embedding_array
            gc.collect()

        except Exception as e:
            logger.error(f"Unexpected error for {model_name} at {path} (blob: {blob_key}): {e}")

    if combined_embeddings.shape[0] == 0 or combined_embeddings.shape[1] == 0:
        logger.warning("No embeddings extracted from any models. Returning empty.")
        return DissectEmbeddingsResponse(embeddings={})

    embedding_2d_combined = None
    try:
        reducer = umap.UMAP(n_neighbors=12, min_dist=0.1, random_state=10, low_memory=True)
        embedding_2d_combined = reducer.fit_transform(combined_embeddings)
        logger.info("UMAP fitting and transformation complete.")
    except Exception as e:
        logger.error(f"UMAP fitting failed: {e}")
        raise HTTPException(status_code=500, detail=f"UMAP fitting failed: {e}")

    del combined_embeddings, reducer
    gc.collect()

    final_results: Dict[str, List[EmbeddingPoint]] = {}
    current_idx = 0
    for entry in embedding_model_map:
        model_name = entry["model_name"]
        count = entry["count"]
        patient_ids = entry["patient_ids"]
        model_2d_embeddings = embedding_2d_combined[current_idx : current_idx + count]

        points = []
        for i, (x, y) in enumerate(model_2d_embeddings):
            pid = patient_ids[i]
            label = label_map.get(pid, "Unknown")
            points.append(EmbeddingPoint(x=float(x), y=float(y), patient_id=pid, label=label))

        final_results[model_name] = points
        current_idx += count

    logger.info("Finished /dissect/embeddings endpoint processing.")
    
    del embedding_2d_combined
    gc.collect()

    return DissectEmbeddingsResponse(embeddings=final_results)
