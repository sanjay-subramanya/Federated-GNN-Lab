import os
import requests
import logging
import json
import vercel_blob
from utils.logging_utils import configure_logging
from config.settings import Config

configure_logging()
logger = logging.getLogger(__name__)

def download_file_from_blob(blob_key: str, dest_local_path: str) -> None:
    logger.info(f"Downloading blob file with key: {blob_key}")
    response = vercel_blob.download_file(f"{Config.vercel_blob_store_url}/{blob_key}", dest_local_path)

    os.makedirs(os.path.dirname(dest_local_path), exist_ok=True)

    with open(dest_local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logger.info(f"Successfully downloaded blob file {blob_key} to {dest_local_path}")


def load_file_from_blob_if_needed(blob_key: str, dest_local_path: str) -> str:
    if os.path.exists(dest_local_path):
        logger.debug(f"File found locally: {dest_local_path}")
        return dest_local_path
    
    logger.info(f"File not found locally, downloading from blob: {blob_key}")
    try:
        download_file_from_blob(blob_key, str(dest_local_path))
        logger.info(f"Successfully cached {blob_key} to {dest_local_path}")
        return dest_local_path
    except Exception as e:
        logger.error(f"Failed to download {blob_key} to {dest_local_path}: {str(e)}")
        raise


def upload_file_to_blob(blob_key: str, dest_local_path: str) -> str:
    with open(dest_local_path, "rb") as f:
        response = vercel_blob.put(blob_key, f.read())
        return response.get("url")


def delete_run_from_blob(run_id: str) -> None:
    blob_key = f"saved_models/{run_id}"
    metadata_blob_key = f"{blob_key}/_train_metadata.json"
    local_metadata_path = Config.model_dir / run_id / "_train_metadata.json"

    try:
        if not os.path.exists(local_metadata_path):
            download_file_from_blob(metadata_blob_key, str(local_metadata_path))

        with open(str(local_metadata_path), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        num_clients = int(metadata.get("num_clients", 5))  # fallback to 5 if missing
    except Exception as e:
        logger.warning(f"Could not fetch metadata for run {run_id}: {e}")
        num_clients = 5

    files = ["global_model_manual.pt", "_train_metadata.json", "_divergence_metrics.json"]
    files += [f"client_{i+1}_model.pt" for i in range(num_clients)]
    keys_to_delete = [f"{blob_key}/{fname}" for fname in files]

    for key in keys_to_delete:
        response = vercel_blob.delete(f"{Config.vercel_blob_store_url}/{key}")
